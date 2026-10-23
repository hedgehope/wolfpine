/*
 * gemm, scalar variant: C = A . B as a 4 x 4 register-blocked F/D loop.
 *
 * Computes in the accumulator's precision (double for fp64, else float).
 * Inputs stored in a narrower format (bf16/fp16 bits) are widened once into a
 * compute-precision copy inside the timed region: the dwarfs' design has no
 * Zfh/Zfbfmin, so this O(n^2) conversion is part of the scalar cost. A bf16
 * accumulator format is produced by rounding the fp32 result once on store.
 */
#include <stdint.h>

#include "dwarf_util.h"
#include "naiad_platform.h"

#define M GEMM_M
#define N GEMM_N
#define K GEMM_K

#ifdef GEMM_ACC_FP64
typedef double ct_t;
#else
typedef float ct_t;
#endif
typedef NAIAD_T(GEMM_IN) in_t;
typedef NAIAD_T(GEMM_ACC) acc_t;

NAIAD_IO in_t A[M * K];
NAIAD_IO in_t B[K * N];
NAIAD_IO acc_t C[M * N];
NAIAD_IO uint64_t region_cycles;

#if defined(GEMM_IN_BF16) || defined(GEMM_IN_FP16) || (defined(GEMM_IN_FP32) && defined(GEMM_ACC_FP64))
#define WIDEN 1
static ct_t Aw[M * K], Bw[K * N];

static inline ct_t widen(in_t v) {
#if defined(GEMM_IN_BF16)
    return (ct_t)naiad_bf16_to_f32(v);
#elif defined(GEMM_IN_FP16)
    return (ct_t)naiad_fp16_to_f32(v);
#else
    return (ct_t)v;
#endif
}
#endif

static inline void put(long i, long j, ct_t v) {
#ifdef GEMM_ACC_BF16
    C[i * N + j] = naiad_f32_to_bf16(v);
#else
    C[i * N + j] = (acc_t)v;
#endif
}

static void gemm(const ct_t *a, const ct_t *b) {
    long i = 0;
    for (; i + 4 <= M; i += 4) {
        long j = 0;
        for (; j + 4 <= N; j += 4) {
            ct_t c[4][4] = {{0}};
            for (long k = 0; k < K; k++) {
                ct_t av[4], bv[4];
                for (int r = 0; r < 4; r++) {
                    av[r] = a[(i + r) * K + k];
                    bv[r] = b[k * N + j + r];
                }
                for (int r = 0; r < 4; r++) {
                    for (int s = 0; s < 4; s++) {
                        c[r][s] += av[r] * bv[s];
                    }
                }
            }
            for (int r = 0; r < 4; r++) {
                for (int s = 0; s < 4; s++) {
                    put(i + r, j + s, c[r][s]);
                }
            }
        }
        for (; j < N; j++) {
            for (int r = 0; r < 4; r++) {
                ct_t acc = 0;
                for (long k = 0; k < K; k++) {
                    acc += a[(i + r) * K + k] * b[k * N + j];
                }
                put(i + r, j, acc);
            }
        }
    }
    for (; i < M; i++) {
        for (long j = 0; j < N; j++) {
            ct_t acc = 0;
            for (long k = 0; k < K; k++) {
                acc += a[i * K + k] * b[k * N + j];
            }
            put(i, j, acc);
        }
    }
}

int main(void) {
    uint64_t t0 = naiad_cycles();
#ifdef WIDEN
    for (long i = 0; i < M * K; i++) {
        Aw[i] = widen(A[i]);
    }
    for (long i = 0; i < K * N; i++) {
        Bw[i] = widen(B[i]);
    }
    gemm(Aw, Bw);
#else
    gemm((const ct_t *)A, (const ct_t *)B);
#endif
    region_cycles = naiad_cycles() - t0;
    return 0;
}
