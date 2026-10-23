/*
 * gemm_data.h: the GEMM kernels' shared problem, so gemm_ame and gemm_scalar
 * compute the same C = A x B and can be compared.
 *
 * A and B are small integers generated from their indices, so every product
 * and sum is exact in int32 and in fp32 (|C| stays far below 2^24), and the
 * result is checked against a host-precomputed FNV-1a hash of C. The hashes
 * below were computed with numpy for the square sizes listed; at any other
 * size the kernels still self-check but report the hash as unchecked.
 *
 * No F/D is assumed: fp32 values are built and read back as bit patterns.
 */
#ifndef GEMM_DATA_H
#define GEMM_DATA_H

#include <stdint.h>

#ifndef GEMM_M
#define GEMM_M 64
#endif
#ifndef GEMM_N
#define GEMM_N GEMM_M
#endif
#ifndef GEMM_K
#define GEMM_K GEMM_M
#endif

#if GEMM_M == 64 && GEMM_N == 64 && GEMM_K == 64
#define GEMM_EXPECTED_HASH 0xe176ce7c83e716daull
#elif GEMM_M == 32 && GEMM_N == 32 && GEMM_K == 32
#define GEMM_EXPECTED_HASH 0xf74074eeb61955deull
#endif

static inline int gemm_a(int i, int k) { return ((i * 7 + k * 3 + i * k) % 9) - 4; }
static inline int gemm_b(int k, int j) { return ((k * 5 + j * 11 + 2 * k * j) % 7) - 3; }

/* FNV-1a over C's elements as 32-bit words, row-major. */
static inline uint64_t gemm_hash_step(uint64_t h, int32_t v) {
    return (h ^ (uint32_t)v) * 0x100000001b3ull;
}
#define GEMM_HASH_INIT 0xcbf29ce484222325ull

/* An integer (|v| < 2^24) as an fp32 bit pattern, with integer arithmetic only. */
static inline uint32_t gemm_i2f(int32_t v) {
    if (v == 0) {
        return 0;
    }
    uint32_t sign = v < 0 ? 0x80000000u : 0;
    uint32_t mag = v < 0 ? (uint32_t)-v : (uint32_t)v;
    int e = 0; /* the data is small, so count up from the bottom */
    while (mag >> (e + 1)) {
        e++;
    }
    return sign | ((uint32_t)(e + 127) << 23) | ((mag << (23 - e)) & 0x7FFFFFu);
}

/* An integer-valued fp32 bit pattern back to the integer (truncating any fraction). */
static inline int32_t gemm_f2i(uint32_t bits) {
    int e = (int)((bits >> 23) & 0xFF) - 127;
    if (e < 0) {
        return 0;
    }
    uint32_t mant = (bits & 0x7FFFFFu) | 0x800000u;
    int32_t mag = (int32_t)(e >= 23 ? mant << (e - 23) : mant >> (23 - e));
    return bits >> 31 ? -mag : mag;
}

#endif /* GEMM_DATA_H */
