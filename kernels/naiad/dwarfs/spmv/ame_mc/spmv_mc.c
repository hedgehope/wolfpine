/*
 * spmv_ame_mc: spmv_ame with the block rows split across the harts of a
 * multi-hart design (each hart has its own matrix unit).
 *
 * Hart h takes a contiguous run of block rows holding about NNZB / NAIAD_HARTS
 * blocks (split on rowptr, so the work balances by blocks, not by rows; a
 * hart's run starts at the first row whose blocks begin at or after its share)
 * and walks it exactly as spmv_ame walks the whole matrix: one flat loop over
 * its blocks, loads one block ahead, two accumulators alternating by block row.
 * Every Y row has one writer and the same sum order as spmv_ame, so results
 * are bit-identical. nv < 16 staging tiles are per hart. On one hart this is
 * spmv_ame plus the split (a few loads of rowptr) and two no-op barriers.
 *
 * Timed region (hart 0's counter): a barrier every hart has reached, to a
 * barrier after every hart's last astore has left its unit. The split is
 * inside it.
 */
#include "matrix.h"
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"
#include "dwarf_util.h"

#if !defined(SPMV_N) || !defined(SPMV_NNZB) || !defined(SPMV_NV)
#error "define SPMV_N, SPMV_NNZB and SPMV_NV (the spec's cflags do)"
#endif
#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16 || NAIAD_MATRIX_TILES < 4 ||     \
    NAIAD_MATRIX_ACCS < 2
#error "spmv_ame_mc is written for a 16 x 16 x 16 unit with at least 4 tiles and 2 accumulators"
#endif
#define N SPMV_N
#define NB (N / 16)
#define NNZB SPMV_NNZB
#define NV SPMV_NV
#define IN DW_IN_FMT
#define ACC DW_ACC_FMT
#define BLK_STRIDE (16L * DW_IN_BYTES)
#define X_STRIDE ((long)NV * DW_IN_BYTES)

NAIAD_IO int32_t rowptr[NB + 1];
NAIAD_IO int32_t colidx[NNZB];
NAIAD_IO dw_in_t vals[NNZB * 256];
NAIAD_IO dw_in_t X[N * NV + 16];
NAIAD_IO dw_acc_t Y[N * NV];
NAIAD_IO uint64_t region_cycles;

#if NV < 16
static dw_acc_t stage[NAIAD_HARTS][2][256] __attribute__((aligned(64)));
#endif

#define LOAD(ta, tb, p)                                                                                \
    do {                                                                                               \
        mx_tload_a(ta, IN, &vals[(long)(p) * 256], BLK_STRIDE);                                        \
        mx_tload_b(tb, IN, &X[(long)colidx[p] * 16 * NV], X_STRIDE);                                   \
    } while (0)

#if NV == 16
#define STORE_ACC(d, bi)                                                                               \
    do {                                                                                               \
        mx_astore(d, ACC, &Y[(long)(bi) * 256], 16L * DW_ACC_BYTES);                                   \
        mx_azero(d);                                                                                   \
    } while (0)
#else
#define STORE_ACC(d, bi)                                                                               \
    do {                                                                                               \
        mx_astore(d, ACC, st[d], 16L * DW_ACC_BYTES);                                                  \
        mx_azero(d);                                                                                   \
        dw_acc_t *y_ = &Y[(long)(bi) * 16 * NV];                                                       \
        for (int r_ = 0; r_ < 16; r_++) {                                                              \
            for (int v_ = 0; v_ < NV; v_++) {                                                          \
                y_[r_ * NV + v_] = st[d][r_ * 16 + v_];                                                \
            }                                                                                          \
        }                                                                                              \
    } while (0)
#endif

#if NV == 16
typedef int stage_t;
#else
typedef dw_acc_t (*stage_t)[256];
#endif

static inline __attribute__((always_inline)) void store_row(int bi, stage_t st) {
    (void)st;
    if (bi & 1) {
        STORE_ACC(1, bi);
    } else {
        STORE_ACC(0, bi);
    }
}

#define ADVANCE(p)                                                                                     \
    while ((p) >= rend) {                                                                              \
        store_row(bi, st);                                                                             \
        bi++;                                                                                          \
        rend = rowptr[bi + 1];                                                                         \
    }

#define MMA(ta, tb)                                                                                    \
    do {                                                                                               \
        if (bi & 1) {                                                                                  \
            mx_mma(1, ta, tb, IN, ACC);                                                                \
        } else {                                                                                       \
            mx_mma(0, ta, tb, IN, ACC);                                                                \
        }                                                                                              \
    } while (0)

/* The first block row of hart g's run: the first row whose blocks start at or after g's share. */
static int first_row(unsigned g) {
    if (g == 0) {
        return 0;
    }
    if (g >= NAIAD_HARTS) {
        return NB;
    }
    const long want = ((long)NNZB * g + NAIAD_HARTS / 2) / NAIAD_HARTS;
    int b = 0;
    while (b < NB && rowptr[b] < want) {
        b++;
    }
    return b;
}

int main(void) {
    const unsigned h = naiad_hartid();
#if NV == 16
    const stage_t st = 0;
#else
    const stage_t st = stage[h];
#endif
    naiad_barrier();
    uint64_t t0 = naiad_cycles();
    const int b0 = first_row(h), b1 = NAIAD_HARTS == 1 ? NB : first_row(h + 1);
    if (b1 > b0) {
        const int p1 = rowptr[b1];
        mx_azero(0);
        mx_azero(1);
        int bi = b0;
        int rend = rowptr[b0 + 1];
        int p = rowptr[b0];
        if (p < p1) {
            LOAD(0, 1, p);
            for (;;) {
                if (p + 1 < p1) {
                    LOAD(2, 3, p + 1);
                }
                ADVANCE(p);
                MMA(0, 1);
                if (++p >= p1) {
                    break;
                }
                if (p + 1 < p1) {
                    LOAD(0, 1, p + 1);
                }
                ADVANCE(p);
                MMA(2, 3);
                if (++p >= p1) {
                    break;
                }
            }
        }
        for (; bi < b1; bi++) { /* the last row with blocks, and any empty rows after it */
            store_row(bi, st);
        }
        naiad_mx_drain();
    }
    naiad_barrier();
    if (h == 0) {
        region_cycles = naiad_cycles() - t0;
    }
    return 0;
}
