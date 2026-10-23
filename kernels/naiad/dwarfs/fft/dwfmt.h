/*
 * dwfmt.h: the compile-time format plumbing shared by this dwarf's variants.
 *
 * The harness picks the formats with -DDW_IN_FMT=NAIAD_MFMT_<X> and
 * -DDW_ACC_FMT=NAIAD_MFMT_<Y>. From those this header defines
 *
 *   in_t      storage type of an input-format element (bf16/fp16 as raw uint16_t)
 *   real      the scalar compute type: double for an fp64 accumulator, else float
 *   acc_t     storage type of an accumulator-format element (== real here)
 *   in_to_real(v) / real_to_in(x)   conversions, round-to-nearest-even
 *   dw_sqrt(x)                      fsqrt.s / fsqrt.d (there is no libm)
 *
 * The core has F and D but no Zfh/Zfbfmin arithmetic, so bf16 and fp16 are
 * converted in software: these conversions are the vector/scalar <-> matrix
 * handoff cost a unit without native narrow-format moves would pay.
 *
 * A private copy per dwarf (kernels/naiad/dwarfs/{fft,nbody}/dwfmt.h, kept
 * identical), until a shared kernels/naiad/include/ header exists.
 */
#ifndef DW_FMT_H
#define DW_FMT_H

#include <stdint.h>

#include "naiad.h"

#ifndef DW_IN_FMT
#define DW_IN_FMT NAIAD_MFMT_FP32
#endif
#ifndef DW_ACC_FMT
#define DW_ACC_FMT NAIAD_MFMT_FP32
#endif

#define DW_INLINE static inline __attribute__((always_inline))

/* -- the scalar compute / accumulator type --------------------------------- */
#if DW_ACC_FMT == NAIAD_MFMT_FP64
typedef double real;
#define DW_ACC_BYTES 8
DW_INLINE double dw_sqrt(double x) {
    double r;
    __asm__("fsqrt.d %0, %1" : "=f"(r) : "f"(x));
    return r;
}
#elif DW_ACC_FMT == NAIAD_MFMT_FP32
typedef float real;
#define DW_ACC_BYTES 4
DW_INLINE float dw_sqrt(float x) {
    float r;
    __asm__("fsqrt.s %0, %1" : "=f"(r) : "f"(x));
    return r;
}
#else
#error "DW_ACC_FMT must be fp32 or fp64"
#endif
typedef real acc_t;

DW_INLINE uint32_t dw_f2u(float f) {
    union {
        float f;
        uint32_t u;
    } c = {.f = f};
    return c.u;
}
DW_INLINE float dw_u2f(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } c = {.u = u};
    return c.f;
}

/* -- the input (matrix operand) type ---------------------------------------- */
#if DW_IN_FMT == NAIAD_MFMT_FP64
typedef double in_t;
#define DW_IN_BYTES 8
DW_INLINE real in_to_real(in_t v) { return (real)v; }
DW_INLINE in_t real_to_in(real x) { return (in_t)x; }
#elif DW_IN_FMT == NAIAD_MFMT_FP32
typedef float in_t;
#define DW_IN_BYTES 4
DW_INLINE real in_to_real(in_t v) { return (real)v; }
DW_INLINE in_t real_to_in(real x) { return (in_t)x; }
#elif DW_IN_FMT == NAIAD_MFMT_BF16
typedef uint16_t in_t;
#define DW_IN_BYTES 2
DW_INLINE real in_to_real(in_t v) { return (real)dw_u2f((uint32_t)v << 16); }
/* fp32 -> bf16, round to nearest even; NaN stays a (quiet) NaN. With an fp64
 * accumulator the double is first narrowed to float, which can double-round:
 * the formats this dwarf lists never pair bf16 inputs with an fp64 acc. */
DW_INLINE in_t real_to_in(real x) {
    uint32_t u = dw_f2u((float)x);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (in_t)((u >> 16) | 0x40);
    }
    return (in_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
#elif DW_IN_FMT == NAIAD_MFMT_FP16
typedef uint16_t in_t;
#define DW_IN_BYTES 2
DW_INLINE real in_to_real(in_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0) {
        float f = (float)m * 5.9604644775390625e-8f; /* m * 2^-24, exact */
        return (real)(s ? -f : f);
    }
    if (e == 31) {
        return (real)dw_u2f(s | 0x7f800000u | (m << 13));
    }
    return (real)dw_u2f(s | ((e + 112u) << 23) | (m << 13));
}
/* fp32 -> fp16, round to nearest even, overflow to inf, gradual underflow. */
DW_INLINE in_t real_to_in(real x) {
    float f = (float)x;
    uint32_t u = dw_f2u(f), s = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) {
        return (in_t)(s | (a > 0x7f800000u ? 0x7e00u : 0x7c00u));
    }
    if (a >= 0x477ff000u) { /* >= 65520 rounds to inf */
        return (in_t)(s | 0x7c00u);
    }
    if (a < 0x38800000u) { /* below 2^-14: subnormal, m = rne(|f| * 2^24) */
        float t = dw_u2f(a) * 16777216.0f;
        long m;
        __asm__("fcvt.l.s %0, %1, rne" : "=r"(m) : "f"(t));
        return (in_t)(s | (uint32_t)m);
    }
    return (in_t)(s | ((a - (112u << 23) + 0xfffu + ((a >> 13) & 1u)) >> 13));
}
#else
#error "DW_IN_FMT must be fp64, fp32, bf16 or fp16"
#endif

#endif /* DW_FMT_H */
