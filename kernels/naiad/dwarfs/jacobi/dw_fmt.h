/*
 * dw_fmt.h: storage types and conversions for the dwarf kernels' format pairs.
 *
 * Shared by the jacobi, reduce and scan dwarfs (include it as
 * "../../jacobi/dw_fmt.h" from a variant directory). A candidate for
 * kernels/naiad/include/ once that directory's owner wants it.
 *
 * The spec's cflags select the pair at compile time:
 *   -DDW_IN_FMT=NAIAD_MFMT_<IN>  -DDW_ACC_FMT=NAIAD_MFMT_<ACC>
 * Supported pairs: fp64/fp64, fp32/fp32, bf16/fp32, fp16/fp32.
 *
 *   dw_in_t    storage of one input element (raw bits: uint16/32/64)
 *   dw_acc_t   the C type of one accumulator element (float or double)
 *   DW_IN_BYTES / DW_ACC_BYTES
 *   dw_in_to_acc(bits) / dw_acc_to_in(value)  (round to nearest even; finite values)
 *   dw_in_bits(double) / dw_acc_bits(double)  bit patterns of small exact constants
 *
 * The core has F and D but no Zfh/Zfbfmin here, so bf16 and fp16 conversions
 * are bit manipulation (fp16 also uses one fmul / fadd for the subnormal range).
 */
#ifndef DW_FMT_H
#define DW_FMT_H

#include <stdint.h>

#include "naiad.h"

#if !defined(DW_IN_FMT) || !defined(DW_ACC_FMT)
#error "define DW_IN_FMT and DW_ACC_FMT (NAIAD_MFMT_* codes); the dwarf spec's cflags do"
#endif

static inline uint32_t dw_f2u(float f) {
    union {
        float f;
        uint32_t u;
    } c = {.f = f};
    return c.u;
}
static inline float dw_u2f(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } c = {.u = u};
    return c.f;
}
static inline uint64_t dw_d2u(double d) {
    union {
        double d;
        uint64_t u;
    } c = {.d = d};
    return c.u;
}
static inline double dw_u2d(uint64_t u) {
    union {
        uint64_t u;
        double d;
    } c = {.u = u};
    return c.d;
}

/* bf16 <-> fp32 (no NaN inputs). */
static inline float dw_bf16_to_f32(uint16_t h) { return dw_u2f((uint32_t)h << 16); }
static inline uint16_t dw_f32_to_bf16(float f) {
    uint32_t u = dw_f2u(f);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

/* fp16 <-> fp32, finite values, subnormals included. */
static inline float dw_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    float mag = dw_u2f((uint32_t)(h & 0x7FFFu) << 13) * dw_u2f(0x77800000u); /* x 2^112 */
    return dw_u2f(dw_f2u(mag) | sign);
}
static inline uint16_t dw_f32_to_f16(float f) {
    uint32_t x = dw_f2u(f);
    uint32_t sign = x & 0x80000000u;
    uint16_t o;
    x ^= sign;
    if (x >= 0x477FF000u) { /* rounds to >= 2^16: inf */
        o = 0x7C00u;
    } else if (x < 0x38800000u) { /* below fp16's smallest normal: let an fp32 add round it */
        o = (uint16_t)(dw_f2u(dw_u2f(x) + 0.5f) - 0x3F000000u);
    } else {
        uint32_t odd = (x >> 13) & 1u;
        x += ((uint32_t)(15 - 127) << 23) + 0xFFFu + odd;
        o = (uint16_t)(x >> 13);
    }
    return o | (uint16_t)(sign >> 16);
}

/* -- the accumulator side ----------------------------------------------------- */

#if DW_ACC_FMT == NAIAD_MFMT_FP64
typedef double dw_acc_t;
#define DW_ACC_BYTES 8
static inline uint64_t dw_acc_bits(double v) { return dw_d2u(v); }
#elif DW_ACC_FMT == NAIAD_MFMT_FP32
typedef float dw_acc_t;
#define DW_ACC_BYTES 4
static inline uint64_t dw_acc_bits(double v) { return dw_f2u((float)v); }
#else
#error "DW_ACC_FMT must be fp64 or fp32"
#endif

/* -- the input side --------------------------------------------------------------- */

#if DW_IN_FMT == NAIAD_MFMT_FP64
#if DW_ACC_FMT != NAIAD_MFMT_FP64
#error "fp64 inputs need an fp64 accumulator"
#endif
typedef uint64_t dw_in_t;
#define DW_IN_BYTES 8
static inline dw_acc_t dw_in_to_acc(dw_in_t b) { return dw_u2d(b); }
static inline dw_in_t dw_acc_to_in(dw_acc_t v) { return dw_d2u(v); }
static inline dw_in_t dw_in_bits(double v) { return dw_d2u(v); }
#else
#if DW_ACC_FMT != NAIAD_MFMT_FP32
#error "fp32 / bf16 / fp16 inputs need an fp32 accumulator"
#endif
#if DW_IN_FMT == NAIAD_MFMT_FP32
typedef uint32_t dw_in_t;
#define DW_IN_BYTES 4
static inline dw_acc_t dw_in_to_acc(dw_in_t b) { return dw_u2f(b); }
static inline dw_in_t dw_acc_to_in(dw_acc_t v) { return dw_f2u(v); }
#elif DW_IN_FMT == NAIAD_MFMT_BF16
typedef uint16_t dw_in_t;
#define DW_IN_BYTES 2
static inline dw_acc_t dw_in_to_acc(dw_in_t b) { return dw_bf16_to_f32(b); }
static inline dw_in_t dw_acc_to_in(dw_acc_t v) { return dw_f32_to_bf16(v); }
#elif DW_IN_FMT == NAIAD_MFMT_FP16
typedef uint16_t dw_in_t;
#define DW_IN_BYTES 2
static inline dw_acc_t dw_in_to_acc(dw_in_t b) { return dw_f16_to_f32(b); }
static inline dw_in_t dw_acc_to_in(dw_acc_t v) { return dw_f32_to_f16(v); }
#else
#error "DW_IN_FMT must be fp64, fp32, bf16 or fp16"
#endif
/* Small exact constants only (0, 1, 1/4): exact in every input format. */
static inline dw_in_t dw_in_bits(double v) { return dw_acc_to_in((float)v); }
#endif

/* Same format on both sides: the matrix unit's accumulator can be stored as the next input. */
#define DW_SAME_FMT (DW_IN_FMT == DW_ACC_FMT)

#endif /* DW_FMT_H */
