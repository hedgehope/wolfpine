/*
 * rvv_fmt.h: the RVV side of the dwarf kernels' format pairs (the `rvv` variants
 * in kernels/naiad/dwarfs/<name>/rvv/).
 *
 * Select the pair with -DDW_IN_FMT=NAIAD_MFMT_<IN> -DDW_ACC_FMT=NAIAD_MFMT_<ACC>
 * (the dwarf specs' cflags), and the accumulator LMUL with RV_AL (1, 2, 4 or 8;
 * default 4) before including. Supported pairs: fp64/fp64, fp32/fp32, fp16/fp32,
 * bf16/fp32. Arithmetic is always in the accumulator format; a 16-bit input is
 * read at half the accumulator's LMUL (same VL) and enters through the widening
 * forms, so it is never rounded again:
 *
 *   fp16: Zvfh  vfwadd.wv / vfwadd.vv / vfwmacc.vf / vfwcvt.f.f.v / vfncvt.f.f.w
 *   bf16: Zvfbfwma vfwmaccbf16.vf (an add is "+ 1.0 * x": the product is exact,
 *         so it rounds once, exactly like an fp32 add of the widened value),
 *         Zvfbfmin vfwcvtbf16.f.f.v / vfncvtbf16.f.f.w
 *
 * Types and helpers (VL is always in accumulator elements):
 *   rv_acc_v / rv_in_v       accumulator vector / input-format vector (same VL)
 *   rv_acc_s / rv_in_s       scalar element types (rv_in_s: double, float, _Float16, __bf16)
 *   rv_in_bits_t             the storage type of one input element (raw bits)
 *   rv_setvl(n), rv_vlmax()
 *   rv_load_in / rv_store_in, rv_load_acc / rv_store_acc, rv_load_in_strided / rv_store_acc_strided
 *   rv_splat(x, vl)          accumulator vector of x
 *   rv_reduce_usum(a, vl)    vfredusum of a's first vl elements, as a scalar
 *   rv_widen(x) / rv_narrow(a)    input <-> accumulator (RNE, frm = 0)
 *   rv_acc_add_in(a, x)      a + widen(x); rv_acc_add_in_tu keeps a's tail (VL < VLMAX)
 *   rv_add_in2(x, y)         widen(x) + widen(y), one rounding; rv_sub_in2 likewise
 *   rv_fma_vf(a, s, x)       a + s * x, s and x input-format, fused
 *   rv_mul_vf(s, x)          s * x into the accumulator format (one rounding)
 *   rv_in_scalar(p, i)       element i of input-format storage, as rv_in_s
 *
 * Everything else uses the overloaded RVV intrinsics (__riscv_vfadd, ...) on rv_acc_v.
 * The scalar Zfhmin conversions are not used (Naiad does not model them): _Float16
 * and __bf16 scalars are only loaded (flh) and passed to .vf forms.
 */
#ifndef NAIAD_RVV_FMT_H
#define NAIAD_RVV_FMT_H

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "naiad.h"

#if !defined(__riscv_v)
#error "rvv_fmt.h needs a design with the V extension"
#endif
#if !defined(DW_IN_FMT) || !defined(DW_ACC_FMT)
#error "define DW_IN_FMT and DW_ACC_FMT (NAIAD_MFMT_* codes)"
#endif
#ifndef RV_AL
#define RV_AL 4
#endif

#define RV_INLINE static inline __attribute__((always_inline, unused))
/*
 * Put RV_KERNEL on every non-inlined function of an rvv kernel. The core is
 * single-issue and in order with no vector chaining, so instruction order
 * decides how much of the vector unit's latency is hidden; LLVM's default
 * (generic) scheduling model knows nothing about vector latencies, and its
 * spacemit-x60 model (LLVM 20) schedules no differently. SiFive's X280 is an
 * in-order RVV core whose scheduling model does interleave loads and
 * independent vector work, so the kernels are scheduled for it (codegen and
 * the ISA are unchanged: "tune" only picks the scheduling model).
 */
#ifndef RV_TUNE
#define RV_TUNE "sifive-x280"
#endif
#define RV_KERNEL __attribute__((target("tune=" RV_TUNE)))
/*
 * RV_MEM_BARRIER(): a compiler-only barrier (an empty volatile asm with a
 * "memory" clobber; it emits nothing) that no load or store crosses. The
 * scheduler sinks each vector load down to its first use, which on this
 * in-order core without chaining exposes the whole load latency; a barrier
 * after a batch of loads keeps them issued early, where the kernel puts them.
 * The NAIAD_IO arrays already escape; a kernel's own static buffers need
 * RV_ESCAPE once.
 */
#define RV_MEM_BARRIER() __asm__ __volatile__("" ::: "memory")
/*
 * RV_PIN(v): an empty volatile asm that "modifies" vector v in place. Every
 * later use of v depends on it and volatile asms keep their order, so the
 * scheduler can neither sink v's producer below the pin nor hoist v's
 * consumers above it. The pins are how the kernels keep a hand-scheduled
 * order (a batch of loads, then the arithmetic on them).
 */
#define RV_PIN(v) __asm__ __volatile__("" : "+vr"(v))
/* RV_USE(v): the input-only form: v's producer cannot sink below it (it creates
 * no new value, so it never costs a register copy, but consumers may still move up). */
#define RV_USE(v) __asm__ __volatile__("" ::"vr"(v))
/* RV_FUSE(s): RV_USE for a scalar FP value (an f register). */
#define RV_FUSE(s) __asm__ __volatile__("" ::"f"(s))
/* Alias analysis lets loads from memory whose address never escapes cross the
 * barrier; RV_ESCAPE(p) (also emits nothing) makes p's object escape once. */
#define RV_ESCAPE(p) __asm__ __volatile__("" ::"r"(p) : "memory")
#define RV_C_(a, b) a##b
#define RV_C(a, b) RV_C_(a, b)
#define RV_C3(a, b, c) RV_C(RV_C(a, b), c)
#define RV_C4(a, b, c, d) RV_C(RV_C3(a, b, c), d)

/* LMUL suffixes: the accumulator's (m4) and half of it (m2), for 16-bit inputs. */
#define RV_HALF_1 f2
#define RV_HALF_2 1
#define RV_HALF_4 2
#define RV_HALF_8 4
#define RV_AM RV_C(m, RV_AL)
#define RV_HM RV_C(m, RV_C(RV_HALF_, RV_AL))

/* -- the accumulator ----------------------------------------------------------- */
#if DW_ACC_FMT == NAIAD_MFMT_FP64
#define RV_AE 64
typedef double rv_acc_s;
#elif DW_ACC_FMT == NAIAD_MFMT_FP32
#define RV_AE 32
typedef float rv_acc_s;
#else
#error "rvv_fmt.h: DW_ACC_FMT must be fp64 or fp32"
#endif
typedef RV_C4(vfloat, RV_AE, RV_AM, _t) rv_acc_v;

RV_INLINE size_t rv_setvl(size_t n) { return RV_C3(__riscv_vsetvl_e, RV_AE, RV_AM)(n); }
RV_INLINE size_t rv_vlmax(void) { return RV_C3(__riscv_vsetvlmax_e, RV_AE, RV_AM)(); }
RV_INLINE rv_acc_v rv_load_acc(const rv_acc_s *p, size_t vl) {
    return RV_C4(__riscv_vle, RV_AE, _v_f, RV_C(RV_AE, RV_AM))(p, vl);
}
RV_INLINE void rv_store_acc(rv_acc_s *p, rv_acc_v v, size_t vl) {
    RV_C4(__riscv_vse, RV_AE, _v_f, RV_C(RV_AE, RV_AM))(p, v, vl);
}
/* Store v's elements stride_bytes apart (vsse; the timing model charges one element a cycle). */
RV_INLINE void rv_store_acc_strided(rv_acc_s *p, ptrdiff_t stride_bytes, rv_acc_v v, size_t vl) {
    RV_C4(__riscv_vsse, RV_AE, _v_f, RV_C(RV_AE, RV_AM))(p, stride_bytes, v, vl);
}
RV_INLINE rv_acc_v rv_splat(rv_acc_s x, size_t vl) { return RV_C(__riscv_vfmv_v_f_f, RV_C(RV_AE, RV_AM))(x, vl); }
typedef RV_C3(vfloat, RV_AE, m1_t) rv_acc_m1;
/* The unordered sum of a's first vl elements (vfredusum), as a scalar. */
RV_INLINE rv_acc_s rv_reduce_usum(rv_acc_v a, size_t vl) {
    rv_acc_m1 z = RV_C3(__riscv_vfmv_s_f_f, RV_AE, m1)(0, 1);
    return __riscv_vfmv_f(__riscv_vfredusum(a, z, vl));
}

/* -- the input format ------------------------------------------------------------ */
#if DW_IN_FMT == NAIAD_MFMT_FP64 || DW_IN_FMT == NAIAD_MFMT_FP32
#if (DW_IN_FMT == NAIAD_MFMT_FP64) != (DW_ACC_FMT == NAIAD_MFMT_FP64)
#error "rvv_fmt.h: fp64 inputs pair with an fp64 accumulator, fp32 with fp32"
#endif
#define RV_SAME 1
typedef rv_acc_s rv_in_s;
typedef rv_acc_v rv_in_v;
#if RV_AE == 64
typedef uint64_t rv_in_bits_t;
#else
typedef uint32_t rv_in_bits_t;
#endif
RV_INLINE rv_in_v rv_load_in(const void *p, size_t vl) { return rv_load_acc((const rv_acc_s *)p, vl); }
RV_INLINE rv_in_v rv_load_in_strided(const void *p, ptrdiff_t stride_bytes, size_t vl) {
    return RV_C4(__riscv_vlse, RV_AE, _v_f, RV_C(RV_AE, RV_AM))((const rv_acc_s *)p, stride_bytes, vl);
}
RV_INLINE void rv_store_in(void *p, rv_in_v v, size_t vl) { rv_store_acc((rv_acc_s *)p, v, vl); }
RV_INLINE rv_acc_v rv_widen(rv_in_v x, size_t vl) {
    (void)vl;
    return x;
}
RV_INLINE rv_in_v rv_narrow(rv_acc_v a, size_t vl) {
    (void)vl;
    return a;
}
RV_INLINE rv_acc_v rv_acc_add_in(rv_acc_v a, rv_in_v x, size_t vl) { return __riscv_vfadd(a, x, vl); }
RV_INLINE rv_acc_v rv_acc_add_in_tu(rv_acc_v a, rv_in_v x, size_t vl) { return __riscv_vfadd_tu(a, a, x, vl); }
RV_INLINE rv_acc_v rv_add_in2(rv_in_v x, rv_in_v y, size_t vl) { return __riscv_vfadd(x, y, vl); }
RV_INLINE rv_acc_v rv_sub_in2(rv_in_v x, rv_in_v y, size_t vl) { return __riscv_vfsub(x, y, vl); }
RV_INLINE rv_acc_v rv_fma_vf(rv_acc_v a, rv_in_s s, rv_in_v x, size_t vl) { return __riscv_vfmacc(a, s, x, vl); }
RV_INLINE rv_acc_v rv_mul_vf(rv_in_s s, rv_in_v x, size_t vl) { return __riscv_vfmul(x, s, vl); }

#elif DW_IN_FMT == NAIAD_MFMT_FP16 || DW_IN_FMT == NAIAD_MFMT_BF16
#if RV_AE != 32
#error "rvv_fmt.h: 16-bit inputs pair with an fp32 accumulator"
#endif
#define RV_SAME 0
typedef uint16_t rv_in_bits_t;
#if DW_IN_FMT == NAIAD_MFMT_FP16
typedef _Float16 rv_in_s;
typedef RV_C3(vfloat16, RV_HM, _t) rv_in_v;
RV_INLINE rv_in_v rv_load_in(const void *p, size_t vl) {
    return RV_C(__riscv_vle16_v_f16, RV_HM)((const _Float16 *)p, vl);
}
RV_INLINE rv_in_v rv_load_in_strided(const void *p, ptrdiff_t stride_bytes, size_t vl) {
    return RV_C(__riscv_vlse16_v_f16, RV_HM)((const _Float16 *)p, stride_bytes, vl);
}
RV_INLINE void rv_store_in(void *p, rv_in_v v, size_t vl) { RV_C(__riscv_vse16_v_f16, RV_HM)((_Float16 *)p, v, vl); }
RV_INLINE rv_acc_v rv_widen(rv_in_v x, size_t vl) { return __riscv_vfwcvt_f(x, vl); }
RV_INLINE rv_in_v rv_narrow(rv_acc_v a, size_t vl) { return __riscv_vfncvt_f(a, vl); }
RV_INLINE rv_acc_v rv_acc_add_in(rv_acc_v a, rv_in_v x, size_t vl) { return __riscv_vfwadd_wv(a, x, vl); }
RV_INLINE rv_acc_v rv_acc_add_in_tu(rv_acc_v a, rv_in_v x, size_t vl) { return __riscv_vfwadd_wv_tu(a, a, x, vl); }
RV_INLINE rv_acc_v rv_add_in2(rv_in_v x, rv_in_v y, size_t vl) { return __riscv_vfwadd_vv(x, y, vl); }
RV_INLINE rv_acc_v rv_sub_in2(rv_in_v x, rv_in_v y, size_t vl) { return __riscv_vfwsub_vv(x, y, vl); }
RV_INLINE rv_acc_v rv_fma_vf(rv_acc_v a, rv_in_s s, rv_in_v x, size_t vl) { return __riscv_vfwmacc(a, s, x, vl); }
RV_INLINE rv_acc_v rv_mul_vf(rv_in_s s, rv_in_v x, size_t vl) { return __riscv_vfwmul(x, s, vl); }
#else
typedef __bf16 rv_in_s;
typedef RV_C3(vbfloat16, RV_HM, _t) rv_in_v;
RV_INLINE rv_in_v rv_load_in(const void *p, size_t vl) {
    return RV_C(__riscv_vle16_v_bf16, RV_HM)((const __bf16 *)p, vl);
}
RV_INLINE rv_in_v rv_load_in_strided(const void *p, ptrdiff_t stride_bytes, size_t vl) {
    return RV_C(__riscv_vlse16_v_bf16, RV_HM)((const __bf16 *)p, stride_bytes, vl);
}
RV_INLINE void rv_store_in(void *p, rv_in_v v, size_t vl) { RV_C(__riscv_vse16_v_bf16, RV_HM)((__bf16 *)p, v, vl); }
RV_INLINE rv_acc_v rv_widen(rv_in_v x, size_t vl) { return RV_C(__riscv_vfwcvtbf16_f_f_v_f32, RV_AM)(x, vl); }
RV_INLINE rv_in_v rv_narrow(rv_acc_v a, size_t vl) { return RV_C(__riscv_vfncvtbf16_f_f_w_bf16, RV_HM)(a, vl); }
/* bf16 1.0 as a scalar operand. */
RV_INLINE __bf16 rv_bf16_one(void) {
    union {
        uint16_t u;
        __bf16 b;
    } c = {.u = 0x3F80u};
    return c.b;
}
RV_INLINE rv_acc_v rv_fma_vf(rv_acc_v a, rv_in_s s, rv_in_v x, size_t vl) { return __riscv_vfwmaccbf16(a, s, x, vl); }
/* Zvfbfwma has no widening multiply: an FMA into zero (exact product, one rounding). */
RV_INLINE rv_acc_v rv_mul_vf(rv_in_s s, rv_in_v x, size_t vl) { return rv_fma_vf(rv_splat(0, vl), s, x, vl); }
RV_INLINE rv_acc_v rv_acc_add_in(rv_acc_v a, rv_in_v x, size_t vl) { return rv_fma_vf(a, rv_bf16_one(), x, vl); }
RV_INLINE rv_acc_v rv_acc_add_in_tu(rv_acc_v a, rv_in_v x, size_t vl) {
    return __riscv_vfwmaccbf16_tu(a, rv_bf16_one(), x, vl);
}
RV_INLINE rv_acc_v rv_add_in2(rv_in_v x, rv_in_v y, size_t vl) { return rv_acc_add_in(rv_widen(x, vl), y, vl); }
/* bf16 -1.0 as a scalar operand. */
RV_INLINE __bf16 rv_bf16_minus_one(void) {
    union {
        uint16_t u;
        __bf16 b;
    } c = {.u = 0xBF80u};
    return c.b;
}
RV_INLINE rv_acc_v rv_sub_in2(rv_in_v x, rv_in_v y, size_t vl) {
    return rv_fma_vf(rv_widen(x, vl), rv_bf16_minus_one(), y, vl);
}
#endif

#else
#error "rvv_fmt.h: DW_IN_FMT must be fp64, fp32, fp16 or bf16"
#endif

RV_INLINE rv_in_s rv_in_scalar(const void *p, long i) { return ((const rv_in_s *)p)[i]; }

#endif /* NAIAD_RVV_FMT_H */
