/*
 * gemm_scalar: the scalar-only baseline for gemm_ame, int32 C = A x B on the
 * integer core alone (same data as gemm_ame, see gemm_data.h), for crossover
 * comparisons. Prints the cycles spent in the GEMM and PASS iff C's hash
 * matches the host-precomputed one.
 */
#include "gemm_data.h"
#include "naiad_platform.h"

#define GM GEMM_M
#define GN GEMM_N
#define GK GEMM_K

static int32_t A[GM][GK], B[GK][GN], C[GM][GN];

int main(void) {
    for (int i = 0; i < GM; i++) {
        for (int k = 0; k < GK; k++) {
            A[i][k] = gemm_a(i, k);
        }
    }
    for (int k = 0; k < GK; k++) {
        for (int j = 0; j < GN; j++) {
            B[k][j] = gemm_b(k, j);
        }
    }

    uint64_t start = naiad_cycles();
    for (int i = 0; i < GM; i++) {
        for (int k = 0; k < GK; k++) {
            int32_t a = A[i][k];
            for (int j = 0; j < GN; j++) {
                C[i][j] += a * B[k][j];
            }
        }
    }
    uint64_t cycles = naiad_cycles() - start;

    uint64_t h = GEMM_HASH_INIT;
    for (int i = 0; i < GM; i++) {
        for (int j = 0; j < GN; j++) {
            h = gemm_hash_step(h, C[i][j]);
        }
    }
    naiad_printf("gemm_scalar %dx%dx%d int32: %llu cycles, hash %llx\n", GM, GN, GK, (unsigned long long)cycles,
                 (unsigned long long)h);
#ifdef GEMM_EXPECTED_HASH
    int ok = h == GEMM_EXPECTED_HASH;
    naiad_printf("hash %s\n", ok ? "ok" : "MISMATCH");
#else
    int ok = 1;
    naiad_puts("hash unchecked (no precomputed hash for this size)\n");
#endif
    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
