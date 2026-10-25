/*
 * apsp_ame: all-pairs shortest paths by blocked Floyd-Warshall on the matrix
 * unit's tropical mma (mma.minplus: acc = min(acc, min_k a[i][k] + b[k][j])).
 *
 * D (n x n, row-major, in place) in APSP_FMT; the tiles, the accumulators and
 * memory all hold that one format. For each diagonal block k (see spec.py):
 *   1. D_kk <- D_kk*: aload it, then four squarings acc = min(acc, T T) with
 *      T = D_kk reloaded after every astore (one tile is both operands);
 *   2. D_kj <- min(D_kj, D_kk* D_kj) and D_ik <- min(D_ik, D_ik D_kk*);
 *   3. D_ij <- min(D_ij, D_ik D_kj) for i, j != k.
 * Every update starts its accumulator from the block it replaces (aload: azero's
 * 0 is not min's identity) and stores it back. Two accumulators and two b tiles
 * alternate between consecutive blocks so a block's astore overlaps the next
 * block's loads.
 *
 * Tiles: 0 = D_kk*, 1 = the phase-3 a operand D_ik, 2 / 3 = b operands, 4 / 5 =
 * the column panel's a operands. Accumulators 0 / 1.
 *
 * Host interface: D (in and out), region_cycles.
 */
#include "matrix.h"
#include "naiad_platform.h"

#include "dwarf_util.h"

#ifndef APSP_N
#error "define APSP_N (a multiple of 16)"
#endif
#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16 || NAIAD_MATRIX_TILES < 6 ||     \
    NAIAD_MATRIX_ACCS < 2
#error "apsp_ame is written for a 16 x 16 x 16 unit with at least 6 tiles and 2 accumulators"
#endif
#ifndef NAIAD_MATRIX_FEATURE_SEMIRING
#error "apsp_ame needs the matrix unit's semiring feature (mma.minplus)"
#endif
#ifndef NAIAD_MATRIX_FEATURE_ACC_MEM
#error "apsp_ame needs aload / astore (matrix.acc_mem)"
#endif
#define N APSP_N
#define NB (N / 16)
#define FMT APSP_MFMT
typedef NAIAD_T(APSP_FMT) ap_t;
#define RS ((long)N * (long)sizeof(ap_t)) /* row stride in bytes */

NAIAD_IO ap_t D[N * N];
NAIAD_IO uint64_t region_cycles;

#define BLK(bi, bj) (&D[(long)(bi) * 16 * N + (long)(bj) * 16])

/* acc d <- block (bi, bj); acc d = min(acc d, ta tb); block <- acc d. */
#define UPDATE(d, ta, tb, bi, bj)                                                                      \
    do {                                                                                               \
        mx_aload(d, FMT, BLK(bi, bj), RS);                                                             \
        mx_mma_minplus(d, ta, tb, FMT, FMT);                                                           \
        mx_astore(d, FMT, BLK(bi, bj), RS);                                                            \
    } while (0)

int main(void) {
    uint64_t t0 = naiad_cycles();
    for (int k = 0; k < NB; k++) {
        /* 1. the diagonal block's closure */
        ap_t *dkk = BLK(k, k);
        mx_aload(0, FMT, dkk, RS);
        for (int s = 0; s < 4; s++) {
            mx_tload_a(0, FMT, dkk, RS);
            mx_mma_minplus(0, 0, 0, FMT, FMT);
            mx_astore(0, FMT, dkk, RS);
        }
        mx_tload_a(0, FMT, dkk, RS); /* D_kk*, resident for this k */

        /* 2. row panel D_kj = D_kk* D_kj, then column panel D_ik = D_ik D_kk* */
        for (int j = 0; j < NB; j++) {
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
        for (int i = 0; i < NB; i++) {
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

        /* 3. the rest */
        for (int i = 0; i < NB; i++) {
            if (i == k) {
                continue;
            }
            mx_tload_a(1, FMT, BLK(i, k), RS);
            for (int j = 0; j < NB; j++) {
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
    naiad_mx_drain();
    region_cycles = naiad_cycles() - t0;
    return 0;
}
