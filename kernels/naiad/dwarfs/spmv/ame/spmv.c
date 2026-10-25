/*
 * spmv_ame: BSR sparse matrix times nv vectors, Y = A X, one mma per 16 x 16 block.
 *
 * Each nonzero block p of block row bi is the "a" tile (tload.a of its 256
 * contiguous elements) and the 16 rows of X it touches the "b" tile (tload.b,
 * row stride nv elements), accumulated into the block row's accumulator. Two
 * accumulators alternate between block rows, so a row's astore overlaps the
 * next row's mma, and two tile pairs alternate between blocks, so block p + 1's
 * loads are issued before block p's mma, also across block-row boundaries (the
 * walk is one flat loop over blocks; a row ends when p reaches rowptr[bi + 1]).
 *
 * nv < 16: the b tile's columns nv..15 are the following X rows (junk), which
 * only reach accumulator columns that are never stored: the accumulator goes to
 * a staging tile and its first nv columns are copied to Y. X carries 16
 * elements of slack so the last block's tile load stays inside the array.
 *
 * Host interface: rowptr, colidx, vals, X (in), Y (out), region_cycles.
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
#error "spmv_ame is written for a 16 x 16 x 16 unit with at least 4 tiles and 2 accumulators"
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
static dw_acc_t stage[2][256] __attribute__((aligned(64)));
#endif

/* Load block p into tile pair (ta, tb). */
#define LOAD(ta, tb, p)                                                                                \
    do {                                                                                               \
        mx_tload_a(ta, IN, &vals[(long)(p) * 256], BLK_STRIDE);                                        \
        mx_tload_b(tb, IN, &X[(long)colidx[p] * 16 * NV], X_STRIDE);                                   \
    } while (0)

/* Store block row bi's accumulator d to Y and clear it for row bi + 2. */
#if NV == 16
#define STORE_ACC(d, bi)                                                                               \
    do {                                                                                               \
        mx_astore(d, ACC, &Y[(long)(bi) * 256], 16L * DW_ACC_BYTES);                                   \
        mx_azero(d);                                                                                   \
    } while (0)
#else
#define STORE_ACC(d, bi)                                                                               \
    do {                                                                                               \
        mx_astore(d, ACC, stage[d], 16L * DW_ACC_BYTES);                                               \
        mx_azero(d);                                                                                   \
        dw_acc_t *y_ = &Y[(long)(bi) * 16 * NV];                                                       \
        for (int r_ = 0; r_ < 16; r_++) {                                                              \
            for (int v_ = 0; v_ < NV; v_++) {                                                          \
                y_[r_ * NV + v_] = stage[d][r_ * 16 + v_];                                             \
            }                                                                                          \
        }                                                                                              \
    } while (0)
#endif

static inline __attribute__((always_inline)) void store_row(int bi) {
    if (bi & 1) {
        STORE_ACC(1, bi);
    } else {
        STORE_ACC(0, bi);
    }
}

/* Close every block row that ends at or before block p. */
#define ADVANCE(p)                                                                                     \
    while ((p) >= rend) {                                                                              \
        store_row(bi);                                                                                 \
        bi++;                                                                                          \
        rend = rowptr[bi + 1];                                                                         \
    }

/* The mma of block p (tiles ta, tb) into the current block row's accumulator. */
#define MMA(ta, tb)                                                                                    \
    do {                                                                                               \
        if (bi & 1) {                                                                                  \
            mx_mma(1, ta, tb, IN, ACC);                                                                \
        } else {                                                                                       \
            mx_mma(0, ta, tb, IN, ACC);                                                                \
        }                                                                                              \
    } while (0)

int main(void) {
    uint64_t t0 = naiad_cycles();
    mx_azero(0);
    mx_azero(1);
    int bi = 0;
    int rend = rowptr[1];
    int p = 0;
    LOAD(0, 1, 0);
    for (;;) {
        /* tiles 0/1 hold block p */
        if (p + 1 < NNZB) {
            LOAD(2, 3, p + 1);
        }
        ADVANCE(p);
        MMA(0, 1);
        if (++p >= NNZB) {
            break;
        }
        /* tiles 2/3 hold block p */
        if (p + 1 < NNZB) {
            LOAD(0, 1, p + 1);
        }
        ADVANCE(p);
        MMA(2, 3);
        if (++p >= NNZB) {
            break;
        }
    }
    for (; bi < NB; bi++) { /* the last row with blocks, and any empty rows after it */
        store_row(bi);
    }
    naiad_mx_drain();
    region_cycles = naiad_cycles() - t0;
    return 0;
}
