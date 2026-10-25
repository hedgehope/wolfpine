/*
 * apsp_ame_mc: apsp_ame's blocked Floyd-Warshall with each phase's blocks split
 * across the harts of a multi-hart design (each hart has its own matrix unit).
 *
 * Per diagonal block k:
 *   1. every hart computes the closure D_kk* itself, in a private scratch copy
 *      (redundant work, 4 squarings, instead of a barrier between phases 1 and
 *      2), and keeps it in tile 0; hart 0 stores that tile back to D after the
 *      panel barrier (phases 2 and 3 never read D_kk from memory);
 *   2. the 2 (NB - 1) row- and column-panel blocks, split evenly; barrier;
 *   3. the (NB - 1)^2 remaining blocks in row order, split into contiguous runs
 *      (a hart reloads its a operand D_ik only when its run reaches a new row);
 *      barrier.
 * Two barriers per k. Min-plus over small integers is exact, so the result is
 * bit-identical to apsp_ame's. On one hart the closure works in D itself and
 * this is apsp_ame's instruction sequence plus no-op barriers.
 *
 * Timed region (hart 0's counter): a barrier every hart has reached, to the
 * last k's closing barrier.
 */
#include "matrix.h"
#include "naiad_platform.h"

#include "dwarf_util.h"

#ifndef APSP_N
#error "define APSP_N (a multiple of 16)"
#endif
#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16 || NAIAD_MATRIX_TILES < 6 ||     \
    NAIAD_MATRIX_ACCS < 2
#error "apsp_ame_mc is written for a 16 x 16 x 16 unit with at least 6 tiles and 2 accumulators"
#endif
#ifndef NAIAD_MATRIX_FEATURE_SEMIRING
#error "apsp_ame_mc needs the matrix unit's semiring feature (mma.minplus)"
#endif
#ifndef NAIAD_MATRIX_FEATURE_ACC_MEM
#error "apsp_ame_mc needs aload / astore (matrix.acc_mem)"
#endif
#define N APSP_N
#define NB (N / 16)
#define FMT APSP_MFMT
typedef NAIAD_T(APSP_FMT) ap_t;
#define RS ((long)N * (long)sizeof(ap_t)) /* row stride in bytes */
#define SRS (16L * (long)sizeof(ap_t))    /* scratch row stride */

NAIAD_IO ap_t D[N * N];
NAIAD_IO uint64_t region_cycles;

#if NAIAD_HARTS > 1
static ap_t scratch[NAIAD_HARTS][256] __attribute__((aligned(64)));
#endif

#define BLK(bi, bj) (&D[(long)(bi) * 16 * N + (long)(bj) * 16])

#define UPDATE(d, ta, tb, bi, bj)                                                                      \
    do {                                                                                               \
        mx_aload(d, FMT, BLK(bi, bj), RS);                                                             \
        mx_mma_minplus(d, ta, tb, FMT, FMT);                                                           \
        mx_astore(d, FMT, BLK(bi, bj), RS);                                                            \
    } while (0)

/* [lo, hi) of n items for hart h, the first n % NAIAD_HARTS harts one more. */
#define SPLIT(n, lo, hi)                                                                               \
    const int lo = (int)h * ((n) / NAIAD_HARTS) + ((int)h < (n) % NAIAD_HARTS ? (int)h : (n) % NAIAD_HARTS); \
    const int hi = lo + (n) / NAIAD_HARTS + ((int)h < (n) % NAIAD_HARTS)

/* The x-th index other than k, and the end (exclusive) of a run of n such indices from 0. */
#define SKIP(x, k) ((x) < (k) ? (x) : (x) + 1)
#define SKIP_END(n, k) ((n) == 0 ? 0 : SKIP((n) - 1, k) + 1)

int main(void) {
    const unsigned h = naiad_hartid();
    naiad_barrier();
    uint64_t t0 = naiad_cycles();
    for (int k = 0; k < NB; k++) {
        /* 1. the diagonal block's closure, by every hart */
        ap_t *dkk = BLK(k, k);
#if NAIAD_HARTS > 1
        ap_t *cl = scratch[h];
        const long cs = SRS;
#else
        ap_t *cl = dkk;
        const long cs = RS;
#endif
        mx_aload(0, FMT, dkk, RS);
        for (int s = 0; s < 4; s++) {
            mx_tload_a(0, FMT, s ? cl : dkk, s ? cs : RS);
            mx_mma_minplus(0, 0, 0, FMT, FMT);
            mx_astore(0, FMT, cl, cs);
        }
        mx_tload_a(0, FMT, cl, cs); /* D_kk*, resident for this k */

        /* 2. the panels: items 0 .. NB - 2 are the row panel's blocks D_kj, the rest the column panel's D_ik */
        {
            SPLIT(2 * (NB - 1), q0, q1);
            const int ra = q0 < NB - 1 ? q0 : NB - 1, rb = q1 < NB - 1 ? q1 : NB - 1;
            for (int j = SKIP(ra, k); j < SKIP_END(rb, k); j++) {
                if (j == k) {
                    continue;
                }
                if (j & 1) {
                    mx_tload_b(3, FMT, BLK(k, j), RS);
                    UPDATE(1, 0, 3, k, j);
                } else {
                    mx_tload_b(2, FMT, BLK(k, j), RS);
                    UPDATE(0, 0, 2, k, j);
                }
            }
            const int ca = (q0 > NB - 1 ? q0 : NB - 1) - (NB - 1), cb = (q1 > NB - 1 ? q1 : NB - 1) - (NB - 1);
            for (int i = SKIP(ca, k); i < SKIP_END(cb, k); i++) {
                if (i == k) {
                    continue;
                }
                if (i & 1) {
                    mx_tload_a(5, FMT, BLK(i, k), RS);
                    UPDATE(1, 5, 0, i, k);
                } else {
                    mx_tload_a(4, FMT, BLK(i, k), RS);
                    UPDATE(0, 4, 0, i, k);
                }
            }
        }
        if (NAIAD_HARTS > 1) {
            naiad_mx_drain();
            naiad_barrier();
            if (h == 0) {
                mx_tstore_a(0, FMT, dkk, RS); /* D_kk* back to D: no hart reads D_kk again before the next barrier */
            }
        }

        /* 3. the rest: item ii * (NB - 1) + jj is block (i, j), i and j the ii-th and jj-th indices != k */
        {
            SPLIT((NB - 1) * (NB - 1), q0, q1);
            const int ii0 = q0 / (NB - 1), jj0 = q0 % (NB - 1), ii1 = q1 / (NB - 1), jj1 = q1 % (NB - 1);
            for (int ii = ii0; ii <= ii1 && ii < NB - 1; ii++) {
                const int ja = ii == ii0 ? jj0 : 0, jb = ii == ii1 ? jj1 : NB - 1;
                if (ja >= jb) {
                    continue;
                }
                const int i = SKIP(ii, k);
                mx_tload_a(1, FMT, BLK(i, k), RS);
                for (int j = SKIP(ja, k); j < SKIP_END(jb, k); j++) {
                    if (j == k) {
                        continue;
                    }
                    if (j & 1) {
                        mx_tload_b(3, FMT, BLK(k, j), RS);
                        UPDATE(1, 1, 3, i, j);
                    } else {
                        mx_tload_b(2, FMT, BLK(k, j), RS);
                        UPDATE(0, 1, 2, i, j);
                    }
                }
            }
        }
        if (NAIAD_HARTS > 1) {
            naiad_mx_drain();
            naiad_barrier();
        }
    }
    naiad_mx_drain();
    if (h == 0) {
        region_cycles = naiad_cycles() - t0;
    }
    return 0;
}
