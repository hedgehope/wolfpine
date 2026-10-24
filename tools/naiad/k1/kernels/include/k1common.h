/*
 * k1common.h: what the K1 / Naiad comparison kernels share.
 *
 * The same source builds for Naiad (and Spike) through kernels/naiad/Makefile,
 * and for Linux on the SpacemiT K1 with -DNAIAD_PLATFORM_LINUX (see
 * tools/naiad/k1/Makefile). Inputs come from a fixed LCG, so both sides compute
 * on identical data, and results are printed as hex bit patterns (Naiad's
 * printf has no floating point, and bits are what we compare anyway):
 *
 *   R <case> <index> <hex>          a result that must match bit for bit
 *   A <case> <index> <hex> <type>   a result compared with a tolerance (an
 *                                   implementation-defined order, vfredusum)
 *   T <case> <ticks> <unit> <reps>  the time for <reps> calls (not compared)
 *
 * tools/naiad/k1/compare.py diffs two such outputs.
 *
 * Strip-mining always asks vsetvl for min(remaining, VLMAX): for an AVL in
 * (VLMAX, 2 VLMAX) the spec lets hardware pick any vl in [ceil(AVL/2), VLMAX],
 * which would change the chunking, and so the FP result, of accumulating kernels.
 */
#ifndef K1COMMON_H
#define K1COMMON_H

#include "naiad_platform.h"

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#ifndef NAIAD_PLATFORM_LINUX
static inline const char *naiad_clock_unit(void) { return "cycles"; }
#endif

/* Timed repetitions: one call on the simulator, many on hardware. */
#ifndef K1_REPS
#ifdef NAIAD_PLATFORM_LINUX
#define K1_REPS 2000
#else
#define K1_REPS 1
#endif
#endif

static uint64_t k1_state = 0x9E3779B97F4A7C15ull;

static inline void k1_seed(uint64_t s) { k1_state = s * 0x9E3779B97F4A7C15ull + 1; }

static inline uint32_t k1_rand32(void) {
    k1_state = k1_state * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(k1_state >> 32);
}

/* Uniform in [-1, 1), exactly representable (24 random bits), so no rounding differs. */
static inline float k1_randf(void) { return (float)((int32_t)k1_rand32() >> 8) * (1.0f / 8388608.0f); }

static inline double k1_randd(void) {
    int64_t hi = (int32_t)k1_rand32();
    uint64_t lo = k1_rand32();
    return (double)((hi << 21) ^ (int64_t)(lo >> 11)) * (1.0 / 9007199254740992.0) * 2.0;
}

static inline uint32_t k1_f32_bits(float f) {
    union {
        float f;
        uint32_t u;
    } c = {.f = f};
    return c.u;
}

static inline uint64_t k1_f64_bits(double d) {
    union {
        double d;
        uint64_t u;
    } c = {.d = d};
    return c.u;
}

static inline void k1_header(const char *kernel) {
    naiad_printf("# kernel %s vlenb %lu clock %s\n", kernel, (unsigned long)__riscv_vlenb(), naiad_clock_unit());
}

static inline void k1_emit_u32(const char *name, const uint32_t *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        naiad_printf("R %s %lu %08x\n", name, (unsigned long)i, v[i]);
    }
}

static inline void k1_emit_f32(const char *name, const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        naiad_printf("R %s %lu %08x\n", name, (unsigned long)i, k1_f32_bits(v[i]));
    }
}

static inline void k1_emit_f64(const char *name, const double *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        naiad_printf("R %s %lu %016llx\n", name, (unsigned long)i, (unsigned long long)k1_f64_bits(v[i]));
    }
}

static inline void k1_emit_approx_f32(const char *name, const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        naiad_printf("A %s %lu %08x f32\n", name, (unsigned long)i, k1_f32_bits(v[i]));
    }
}

static inline void k1_time(const char *name, uint64_t t0, uint64_t t1, unsigned reps) {
    naiad_printf("T %s %llu %s %u\n", name, (unsigned long long)(t1 - t0), naiad_clock_unit(), reps);
}

/* Keep the compiler from hoisting or deleting repeated kernel calls. */
#define K1_CLOBBER() __asm__ volatile("" ::: "memory")
/* ... and from treating a local result buffer as unobserved. */
#define K1_ESCAPE(p) __asm__ volatile("" : : "r"(p) : "memory")

#endif /* K1COMMON_H */
