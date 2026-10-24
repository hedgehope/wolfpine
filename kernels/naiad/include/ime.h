/*
 * ime.h: .insn wrappers for the Naiad integrated matrix mode (IME: the Zvvm draft,
 * riscv/integrated-matrix-extension @ 6effbcae).
 *
 * IME has no state of its own: A, B and C are RVV register groups, so every operand
 * here is an RVV value (any vuint / vfloat type of the right group size; the bits are
 * opaque to the instruction) and the compiler allocates the registers through the "vr"
 * constraint. Opcodes, funct3/funct7 and the vtype field positions come from the
 * design's naiad.h (NAIAD_IME_*), generated from framework/matrix/ime.py.
 *
 * Clang 20 knows neither Zvvm nor the lambda field, so vtype is written with vsetvl
 * from a computed value (IME_VTYPE), which is how the draft says the IME fields are
 * set; vsetvli / vsetivli (and so every RVV intrinsic) keep a supported lambda, bs and
 * altfmt_A/B, but replace vl and LMUL. The wrappers below use whatever vtype and vl are
 * current: issue ime_vsetvl first, with no RVV intrinsic between it and the wrappers
 * that depend on it, or put the whole step in one asm statement.
 *
 *   IME_VTYPE(sew, lmul, lambda, altfmt, altfmt_a, altfmt_b)   a vtype value (lambda 0 = 000)
 *   ime_vsetvl(avl, vtype) / ime_vsetvlmax(vtype)               vsetvl; returns vl
 *   ime_vtype() / ime_lambda()                                  read vtype / the selected lambda
 *   ime_vmmacc(c, a, b), ime_vwmmacc, ime_vqmmacc, ime_v8wmmacc       integer C += A . B^T (W 1/2/4/8)
 *   ime_vfmmacc(c, a, b), ime_vfwmmacc, ime_vfqmmacc, ime_vf8wmmacc   FP
 *   ime_vmtl(vd, base, ld) / ime_vmttl                          tile load (order-preserving / transposing)
 *   ime_vmts(vs3, base, ld) / ime_vmtts                         tile store
 *   ime_vmtl_l(vd, base, ld, lambda) ...                        with an immediate lambda (L1..L64)
 *   ime_vmtl_m(vd, base, ld, mask) / ime_vmts_m ...             masked by `mask` (bound to v0)
 *   ime_vzero(c, vtype)                                         vsetvl to VLMAX of vtype, then c = 0
 *
 * ld is the leading dimension in SEW-wide storage elements (0: the natural one).
 */
#ifndef NAIAD_IME_H
#define NAIAD_IME_H

#include <stdint.h>

#include "naiad.h"

#ifndef NAIAD_HAS_IME
#error "this design has no IME unit (matrix: {mode: ime} in the design file)"
#endif
#ifndef __riscv_vector
#error "ime.h needs the vector extension in -march (the design's NAIAD_MARCH has it)"
#endif

#include <riscv_vector.h>

#define IME_INLINE static inline __attribute__((always_inline, unused))

#define IME_SEW_CODE(sew) ((sew) == 8 ? 0 : (sew) == 16 ? 1 : (sew) == 32 ? 2 : 3)
#define IME_LMUL_CODE(lmul) ((lmul) == 1 ? 0 : (lmul) == 2 ? 1 : (lmul) == 4 ? 2 : 3)
#define IME_LAMBDA_ENC(lam)                                                                        \
    ((lam) == 0    ? 0                                                                             \
     : (lam) == 1  ? 1                                                                             \
     : (lam) == 2  ? 2                                                                             \
     : (lam) == 4  ? 3                                                                             \
     : (lam) == 8  ? 4                                                                             \
     : (lam) == 16 ? 5                                                                             \
     : (lam) == 32 ? 6                                                                             \
                   : 7)

/* vta = vma = 0 (undisturbed), as the IME draft's tile policies assume. */
#define IME_VTYPE(sew, lmul, lam, altfmt, altfmt_a, altfmt_b)                                      \
    (((unsigned long)IME_LAMBDA_ENC(lam) << NAIAD_IME_VTYPE_LAMBDA_SHIFT) |                         \
     ((unsigned long)(altfmt_a) << NAIAD_IME_VTYPE_ALTFMT_A_SHIFT) |                               \
     ((unsigned long)(altfmt_b) << NAIAD_IME_VTYPE_ALTFMT_B_SHIFT) |                               \
     ((unsigned long)(altfmt) << NAIAD_IME_VTYPE_ALTFMT_SHIFT) | ((unsigned long)IME_SEW_CODE(sew) << 3) | \
     (unsigned long)IME_LMUL_CODE(lmul))

IME_INLINE unsigned long ime_vsetvl(unsigned long avl, unsigned long vtype) {
    unsigned long vl;
    __asm__ volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(avl), "r"(vtype));
    return vl;
}

IME_INLINE unsigned long ime_vsetvlmax(unsigned long vtype) {
    unsigned long vl;
    __asm__ volatile("vsetvl %0, zero, %1" : "=r"(vl) : "r"(vtype));
    return vl;
}

IME_INLINE unsigned long ime_vtype(void) {
    unsigned long vt;
    __asm__ volatile("csrr %0, vtype" : "=r"(vt));
    return vt;
}

IME_INLINE int ime_lambda(void) {
    unsigned enc = (unsigned)(ime_vtype() >> NAIAD_IME_VTYPE_LAMBDA_SHIFT) & 7u;
    return enc ? 1 << (enc - 1) : 0;
}

/* -- multiply-accumulates: c += a . b^T ------------------------------------------------ */

#define IME_MMA_(insn, c, a, b)                                                                    \
    __asm__ volatile(".insn r %[ime_op], %[ime_f3], %[ime_f7], %[ime_vd], %[ime_va], %[ime_vb]"                            \
                     : [ime_vd] "+vr"(c)                                                               \
                     : [ime_op] "i"(NAIAD_IME_##insn##_OPCODE), [ime_f3] "i"(NAIAD_IME_##insn##_FUNCT3),   \
                       [ime_f7] "i"(NAIAD_IME_##insn##_FUNCT7), [ime_va] "vr"(a), [ime_vb] "vr"(b))

#define ime_vmmacc(c, a, b) IME_MMA_(VMMACC_VV, c, a, b)
#define ime_vwmmacc(c, a, b) IME_MMA_(VWMMACC_VV, c, a, b)
#define ime_vqmmacc(c, a, b) IME_MMA_(VQMMACC_VV, c, a, b)
#define ime_v8wmmacc(c, a, b) IME_MMA_(V8WMMACC_VV, c, a, b)
#define ime_vfmmacc(c, a, b) IME_MMA_(VFMMACC_VV, c, a, b)
#define ime_vfwmmacc(c, a, b) IME_MMA_(VFWMMACC_VV, c, a, b)
#define ime_vfqmmacc(c, a, b) IME_MMA_(VFQMMACC_VV, c, a, b)
#define ime_vf8wmmacc(c, a, b) IME_MMA_(VF8WMMACC_VV, c, a, b)

/* -- tile loads and stores ---------------------------------------------------------------- */

#define IME_F7_(insn, lam, vm) ((NAIAD_IME_##insn##_FUNCT7 & ~1) | (vm) | (IME_LAMBDA_ENC(lam) << NAIAD_IME_LS_LAMBDA_SHIFT))

#define IME_LOAD_(insn, lam, vd, base, ld)                                                         \
    __asm__ volatile(".insn r %[ime_op], %[ime_f3], %[ime_f7], %[ime_vd], %[ime_rs1], %[ime_rs2]"                          \
                     : [ime_vd] "=vr"(vd)                                                              \
                     : [ime_op] "i"(NAIAD_IME_##insn##_OPCODE), [ime_f3] "i"(NAIAD_IME_##insn##_FUNCT3),   \
                       [ime_f7] "i"(IME_F7_(insn, lam, 1)), [ime_rs1] "r"(base), [ime_rs2] "r"((long)(ld))     \
                     : "memory")

#define IME_LOAD_M_(insn, lam, vd, base, ld, mask)                                                 \
    __asm__ volatile(".insn r %[ime_op], %[ime_f3], %[ime_f7], %[ime_vd], %[ime_rs1], %[ime_rs2]"                          \
                     : [ime_vd] "+vr"(vd)                                                              \
                     : [ime_op] "i"(NAIAD_IME_##insn##_OPCODE), [ime_f3] "i"(NAIAD_IME_##insn##_FUNCT3),   \
                       [ime_f7] "i"(IME_F7_(insn, lam, 0)), [ime_rs1] "r"(base), [ime_rs2] "r"((long)(ld)),    \
                       [ime_m] "vm"(mask)                                                              \
                     : "memory")

#define IME_STORE_(insn, lam, vs3, base, ld)                                                       \
    __asm__ volatile(".insn r %[ime_op], %[ime_f3], %[ime_f7], %[ime_vs], %[ime_rs1], %[ime_rs2]"                          \
                     :                                                                             \
                     : [ime_op] "i"(NAIAD_IME_##insn##_OPCODE), [ime_f3] "i"(NAIAD_IME_##insn##_FUNCT3),   \
                       [ime_f7] "i"(IME_F7_(insn, lam, 1)), [ime_vs] "vr"(vs3), [ime_rs1] "r"(base),           \
                       [ime_rs2] "r"((long)(ld))                                                       \
                     : "memory")

#define IME_STORE_M_(insn, lam, vs3, base, ld, mask)                                               \
    __asm__ volatile(".insn r %[ime_op], %[ime_f3], %[ime_f7], %[ime_vs], %[ime_rs1], %[ime_rs2]"                          \
                     :                                                                             \
                     : [ime_op] "i"(NAIAD_IME_##insn##_OPCODE), [ime_f3] "i"(NAIAD_IME_##insn##_FUNCT3),   \
                       [ime_f7] "i"(IME_F7_(insn, lam, 0)), [ime_vs] "vr"(vs3), [ime_rs1] "r"(base),           \
                       [ime_rs2] "r"((long)(ld)), [ime_m] "vm"(mask)                                       \
                     : "memory")

#define ime_vmtl(vd, base, ld) IME_LOAD_(VMTL_V, 0, vd, base, ld)
#define ime_vmttl(vd, base, ld) IME_LOAD_(VMTTL_V, 0, vd, base, ld)
#define ime_vmts(vs3, base, ld) IME_STORE_(VMTS_V, 0, vs3, base, ld)
#define ime_vmtts(vs3, base, ld) IME_STORE_(VMTTS_V, 0, vs3, base, ld)
#define ime_vmtl_l(vd, base, ld, lam) IME_LOAD_(VMTL_V, lam, vd, base, ld)
#define ime_vmttl_l(vd, base, ld, lam) IME_LOAD_(VMTTL_V, lam, vd, base, ld)
#define ime_vmts_l(vs3, base, ld, lam) IME_STORE_(VMTS_V, lam, vs3, base, ld)
#define ime_vmtts_l(vs3, base, ld, lam) IME_STORE_(VMTTS_V, lam, vs3, base, ld)
#define ime_vmtl_m(vd, base, ld, mask) IME_LOAD_M_(VMTL_V, 0, vd, base, ld, mask)
#define ime_vmttl_m(vd, base, ld, mask) IME_LOAD_M_(VMTTL_V, 0, vd, base, ld, mask)
#define ime_vmts_m(vs3, base, ld, mask) IME_STORE_M_(VMTS_V, 0, vs3, base, ld, mask)
#define ime_vmtts_m(vs3, base, ld, mask) IME_STORE_M_(VMTTS_V, 0, vs3, base, ld, mask)

/* c = 0 over the whole group: vsetvl to VLMAX of `vtype` (its LMUL must be c's), then vmv.v.i.
 * Leaves that vtype and vl installed. */
#define ime_vzero(c, vtype)                                                                        \
    do {                                                                                           \
        unsigned long ime_t_;                                                                      \
        __asm__ volatile("vsetvl %[ime_t], zero, %[ime_vt]\n\tvmv.v.i %[ime_c], 0"                             \
                         : [ime_c] "=vr"(c), [ime_t] "=&r"(ime_t_)                                         \
                         : [ime_vt] "r"((unsigned long)(vtype)));                                      \
        (void)ime_t_;                                                                              \
    } while (0)

#endif /* NAIAD_IME_H */
