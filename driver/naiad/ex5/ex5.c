/*
 * ex5: a GEMM on the AME matrix unit. Design: naiad-ame-timed (RV64IMA, a
 * 16x16x16 AME unit with 8 tiles and 4 accumulators, the cycle model on).
 *
 * C[M x N] = A[M x K] . B[K x N] in int8 with int32 accumulation, through
 * naiad_gemm_int8_int32 from naiad_gemm.h, which kernels/naiad/Makefile emits
 * for the design (python -m framework.naiad.emit.gemm): a blocked driver over
 * a 2 x 2-accumulator micro-kernel, built on matrix.h's .insn wrappers
 * (mx_tload_a/b, mx_mma, mx_astore). M, N and K need not be tile multiples;
 * the driver zero-pads edge tiles. The result is checked element for element
 * against a plain scalar triple loop, and both are timed.
 *
 * Inputs: when the host (ex5.py, through framework.naiad.kernel_io) writes
 * m/n/k and A/B before the run, those are used; otherwise (m == 0, as in a
 * plain `wolfpine run naiad`) the program fills a default 40 x 24 x 56
 * problem itself. NAIAD_IO arrays live in .noinit, which crt0 does not zero,
 * so a host write after the load survives to main.
 */
#include "naiad_gemm.h"
#include "naiad_platform.h"

#define MAX 64

NAIAD_IO volatile int64_t m, n, k;
NAIAD_IO int8_t A[MAX * MAX];
NAIAD_IO int8_t B[MAX * MAX];
NAIAD_IO int32_t C[MAX * MAX];
static int32_t ref[MAX * MAX];

int main(void) {
    if (m == 0) { /* no host inputs: a default, ragged, problem */
        m = 40, n = 24, k = 56;
        for (long i = 0; i < m * k; i++) {
            A[i] = (int8_t)((i * 37 + 11) % 255 - 127);
        }
        for (long i = 0; i < k * n; i++) {
            B[i] = (int8_t)((i * 53 + 5) % 255 - 127);
        }
    }
    long M = m, N = n, K = k;
    if (M > MAX || N > MAX || K > MAX || M <= 0 || N <= 0 || K <= 0) {
        naiad_printf("ex5: bad shape %ld x %ld x %ld (max %d)\nFAIL\n", M, N, K, MAX);
        return 2;
    }
    naiad_printf("ex5: %ld x %ld x %ld int8 GEMM on %s (%dx%dx%d tiles)\n", M, N, K, NAIAD_DESIGN, NAIAD_MATRIX_M,
                 NAIAD_MATRIX_N, NAIAD_MATRIX_K);

    /* On the matrix unit. Row strides are in elements: A is M x K, B is K x N, C is M x N. */
    uint64_t start = naiad_cycles();
    naiad_gemm_int8_int32(M, N, K, A, K, B, N, C, N);
    naiad_mx_drain(); /* the timed region ends when the last astore has left the unit */
    uint64_t ame_cycles = naiad_cycles() - start;

    /* The scalar reference. */
    start = naiad_cycles();
    for (long i = 0; i < M; i++) {
        for (long j = 0; j < N; j++) {
            int32_t sum = 0;
            for (long p = 0; p < K; p++) {
                sum += A[i * K + p] * B[p * N + j];
            }
            ref[i * N + j] = sum;
        }
    }
    uint64_t scalar_cycles = naiad_cycles() - start;

    long bad = 0;
    for (long i = 0; i < M * N; i++) {
        if (C[i] != ref[i] && bad++ < 4) {
            naiad_printf("  C[%ld][%ld] = %d, scalar %d\n", i / N, i % N, (int)C[i], (int)ref[i]);
        }
    }
    naiad_printf("  AME %lu cycles, scalar %lu cycles (%lux); %lu mma issued\n", (unsigned long)ame_cycles,
                 (unsigned long)scalar_cycles, (unsigned long)(scalar_cycles / (ame_cycles ? ame_cycles : 1)),
                 (unsigned long)naiad_perf_read(NAIAD_PERF_MATRIX_MMA));
    naiad_printf("  mismatches: %ld of %ld\n", bad, M * N);
    /* One mma per (row tile, column tile, K tile), edges included. */
#define TILES(x, t) (((x) + (t) - 1) / (t))
    long mmas = TILES(M, NAIAD_MATRIX_M) * TILES(N, NAIAD_MATRIX_N) * TILES(K, NAIAD_MATRIX_K);
    int ok = bad == 0 && (long)naiad_perf_read(NAIAD_PERF_MATRIX_MMA) == mmas;
    naiad_printf("  mma count %s (expect %ld)\n", (long)naiad_perf_read(NAIAD_PERF_MATRIX_MMA) == mmas ? "ok" : "MISMATCH",
                 mmas);
    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
