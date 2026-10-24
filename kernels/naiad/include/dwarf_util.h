/*
 * dwarf_util.h: small helpers shared by the dwarf kernels (kernels/naiad/dwarfs/).
 *
 *   naiad_<fmt>_t                C storage type of a matrix format's element
 *                                (fp64 double, fp32 float, fp16/bf16 raw uint16_t bits,
 *                                int8 int8_t, int32 int32_t); NAIAD_T(fmt) pastes the name.
 *   naiad_f32_bits / naiad_bits_f32, naiad_f64_bits / naiad_bits_f64   bit-pattern puns
 *   naiad_bf16_to_f32 / naiad_fp16_to_f32   widen raw bf16 / fp16 bits to float (exact)
 *   naiad_f32_to_bf16                     narrow a float to bf16 bits, round to nearest even
 *   naiad_pack_tile                       copy a rows x cols block of a strided array into a
 *                                         dense, zero-padded prow x pcol buffer (tile padding)
 *   naiad_unpack_tile                     copy the valid rows x cols of a dense buffer back out
 *   naiad_zero                            zero n bytes, word at a time when aligned
 *   naiad_mx_drain                        (designs with a matrix unit) wait until every
 *                                         accumulator's pending astore has read it, so a
 *                                         timed region ends after its last result left the unit
 *
 * Everything is static inline and needs only stdint; the scalar baselines use the
 * conversions to read the same half-precision storage the matrix unit reads (the
 * dwarfs' designs have no Zfh / Zfbfmin).
 */
#ifndef NAIAD_DWARF_UTIL_H
#define NAIAD_DWARF_UTIL_H

#include <stddef.h>
#include <stdint.h>

#include "naiad.h"

typedef double naiad_fp64_t;
typedef float naiad_fp32_t;
typedef uint16_t naiad_fp16_t;
typedef uint16_t naiad_bf16_t;
typedef int8_t naiad_int8_t;
typedef int32_t naiad_int32_t;

#define NAIAD_T_(fmt) naiad_##fmt##_t
#define NAIAD_T(fmt) NAIAD_T_(fmt)

#define NAIAD_DU_INLINE static inline __attribute__((always_inline, unused))

NAIAD_DU_INLINE uint32_t naiad_f32_bits(float f) {
    union {
        float f;
        uint32_t u;
    } v = {.f = f};
    return v.u;
}

NAIAD_DU_INLINE float naiad_bits_f32(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } v = {.u = u};
    return v.f;
}

NAIAD_DU_INLINE uint64_t naiad_f64_bits(double d) {
    union {
        double d;
        uint64_t u;
    } v = {.d = d};
    return v.u;
}

NAIAD_DU_INLINE double naiad_bits_f64(uint64_t u) {
    union {
        uint64_t u;
        double d;
    } v = {.u = u};
    return v.d;
}

NAIAD_DU_INLINE float naiad_bf16_to_f32(uint16_t h) { return naiad_bits_f32((uint32_t)h << 16); }

NAIAD_DU_INLINE float naiad_fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else if (exp) {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    } else if (!man) {
        bits = sign;
    } else { /* subnormal: renormalise */
        exp = 113u;
        while (!(man & 0x400u)) {
            man <<= 1;
            exp--;
        }
        bits = sign | (exp << 23) | ((man & 0x3ffu) << 13);
    }
    return naiad_bits_f32(bits);
}

NAIAD_DU_INLINE uint16_t naiad_f32_to_bf16(float f) {
    uint32_t u = naiad_f32_bits(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t)((u >> 16) | 0x40u); /* quiet NaN */
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

NAIAD_DU_INLINE void naiad_zero(void *dst, size_t n) {
    if (((uintptr_t)dst & 7u) == 0 && (n & 7u) == 0) {
        uint64_t *d = (uint64_t *)dst;
        for (size_t i = 0; i < n / 8; i++) {
            d[i] = 0;
        }
        return;
    }
    uint8_t *d = (uint8_t *)dst;
    for (size_t i = 0; i < n; i++) {
        d[i] = 0;
    }
}

/* Element copy of `es` bytes (1, 2, 4 or 8), so packing moves words, not bytes. */
#define NAIAD_DU_COPY_ROW(T, d, s, n)                                                              \
    do {                                                                                           \
        T *d_ = (T *)(d);                                                                          \
        const T *s_ = (const T *)(s);                                                              \
        for (long c_ = 0; c_ < (n); c_++) {                                                        \
            d_[c_] = s_[c_];                                                                       \
        }                                                                                          \
    } while (0)

NAIAD_DU_INLINE void naiad_du_copy(void *d, const void *s, long n, int es) {
    switch (es) {
    case 8:
        NAIAD_DU_COPY_ROW(uint64_t, d, s, n);
        break;
    case 4:
        NAIAD_DU_COPY_ROW(uint32_t, d, s, n);
        break;
    case 2:
        NAIAD_DU_COPY_ROW(uint16_t, d, s, n);
        break;
    default:
        NAIAD_DU_COPY_ROW(uint8_t, d, s, n);
        break;
    }
}

/*
 * dst (prow x pcol elements of es bytes, dense, row stride pcol) <- the rows x cols block
 * of src (row stride src_ld elements), zero elsewhere. rows <= prow, cols <= pcol.
 */
static inline __attribute__((unused)) void naiad_pack_tile(void *dst, long prow, long pcol, const void *src,
                                                           long src_ld, long rows, long cols, int es) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    for (long r = 0; r < rows; r++) {
        naiad_du_copy(d + r * pcol * es, s + r * src_ld * es, cols, es);
        if (cols < pcol) {
            naiad_zero(d + (r * pcol + cols) * es, (size_t)((pcol - cols) * es));
        }
    }
    if (rows < prow) {
        naiad_zero(d + rows * pcol * es, (size_t)((prow - rows) * pcol * es));
    }
}

/* dst (row stride dst_ld elements) <- the first rows x cols of the dense pcol-wide src. */
static inline __attribute__((unused)) void naiad_unpack_tile(void *dst, long dst_ld, const void *src, long pcol,
                                                             long rows, long cols, int es) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    for (long r = 0; r < rows; r++) {
        naiad_du_copy(d + r * dst_ld * es, s + r * pcol * es, cols, es);
    }
}

#if defined(NAIAD_HAS_MATRIX) && defined(__OPTIMIZE__)
#include "matrix.h"

/* azero waits for the accumulator's in-flight readers (an astore) before writing it,
 * so zeroing every accumulator is a store fence. It clobbers the accumulators. */
NAIAD_DU_INLINE void naiad_mx_drain(void) {
#define NAIAD_DU_DRAIN(d)                                                                          \
    if ((d) < NAIAD_MATRIX_ACCS) {                                                                 \
        mx_azero((d) < NAIAD_MATRIX_ACCS ? (d) : 0);                                               \
    }
    NAIAD_DU_DRAIN(0) NAIAD_DU_DRAIN(1) NAIAD_DU_DRAIN(2) NAIAD_DU_DRAIN(3)
    NAIAD_DU_DRAIN(4) NAIAD_DU_DRAIN(5) NAIAD_DU_DRAIN(6) NAIAD_DU_DRAIN(7)
    NAIAD_DU_DRAIN(8) NAIAD_DU_DRAIN(9) NAIAD_DU_DRAIN(10) NAIAD_DU_DRAIN(11)
    NAIAD_DU_DRAIN(12) NAIAD_DU_DRAIN(13) NAIAD_DU_DRAIN(14) NAIAD_DU_DRAIN(15)
#undef NAIAD_DU_DRAIN
    mx_scratch_fence(); /* without acc_mem, astore is amov + tstore through scratch tiles */
}
#endif

#endif /* NAIAD_DWARF_UTIL_H */
