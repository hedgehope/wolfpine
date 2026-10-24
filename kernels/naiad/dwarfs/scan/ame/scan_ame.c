/*
 * scan_ame: the inclusive prefix sum y = scan(x) of a length-N vector, as
 * products with triangular ones tiles.
 *
 * x is N / 256 consecutive 16 x 16 row-major tiles X_c (row r holds elements
 * 16 r .. 16 r + 15 of the tile). In one tile:
 *   X U        row-wise inclusive scans          (U upper-triangular ones, k <= j)
 *   Q = Ls X   column sums over the earlier rows (Ls strictly lower-triangular ones)
 *   Q J        the sum of every earlier row, broadcast along the row (J all ones)
 * so X U + Q J is the in-tile scan. Q is an accumulator, and an accumulator
 * cannot feed an mma, so it makes one round trip through memory (astore, then
 * tload as an accumulator-format tile): the AME has no accumulator-to-tile move.
 *
 * Carry across tiles: the tile's scan plus y[256 c - 1], the last output of the
 * previous tile. SCAN_EPILOGUE 0 (default) keeps it on the matrix unit: a
 * stride-0 load of the previous tile's last output row R (every row of the
 * tile is that row) times Sel (ones in row 15) broadcasts R[15] to every
 * element: one more load and one more mma. SCAN_EPILOGUE 1 reads y[256 c - 1]
 * on the scalar side and adds it with aadds (AME epilogue feature).
 * So 4 mma (3 with the epilogue) per 256 outputs; the carry is a serial chain
 * through memory from one tile to the next.
 *
 * Feature switches (AME features the default design lists only in a sweep):
 *   SCAN_MMOV 1 (acc_move): Q goes accumulator -> tile by amov.t.a (Ztt
 *     mmov.m.a) instead of the astore + tload round trip.
 *   SCAN_PREFIX 1 (reduce, acc_move, shift; convert for 16-bit inputs): the
 *     in-tile scan without triangular products. With X in the accumulator
 *     format (tconv), P = tprefix.row X (row-wise scans), and the earlier-rows
 *     totals are E = tshift(tprefix.col(treduce.row X), 1 row): the row totals
 *     broadcast along each row, scanned down the columns, made exclusive. Then
 *     acc = P (amov.a.t) + E I (one mma with the identity). 1 mma per tile
 *     (2 with the matrix carry) instead of 3 (4).
 *
 * Host interface: x (in, N input-format elements), y (N accumulator-format
 * elements), region_cycles.
 */
#include "matrix.h"
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"
#include "dwarf_util.h"

#ifndef SCAN_N
#error "define SCAN_N (a multiple of 256)"
#endif
#ifndef SCAN_EPILOGUE
#define SCAN_EPILOGUE 0
#endif
#ifndef SCAN_MMOV
#define SCAN_MMOV 0
#endif
#ifndef SCAN_PREFIX
#define SCAN_PREFIX 0
#endif
#define N SCAN_N
#define IN DW_IN_FMT
#define ACC DW_ACC_FMT
#define TS (16L * DW_IN_BYTES)
#define ATS (16L * DW_ACC_BYTES)

#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16 || NAIAD_MATRIX_TILES < 7 ||           \
    NAIAD_MATRIX_ACCS < 3
#error "scan_ame is written for a 16 x 16 x 16 unit with at least 7 tiles and 3 accumulators"
#endif
#if N % 256
#error "SCAN_N must be a multiple of 256"
#endif
#if SCAN_EPILOGUE && !defined(NAIAD_MATRIX_FEATURE_EPILOGUE)
#error "SCAN_EPILOGUE needs the matrix unit's epilogue feature"
#endif
#if (SCAN_MMOV || SCAN_PREFIX) && !defined(NAIAD_MATRIX_FEATURE_ACC_MOVE)
#error "SCAN_MMOV / SCAN_PREFIX need the matrix unit's acc_move feature"
#endif
#if SCAN_PREFIX && (!defined(NAIAD_MATRIX_FEATURE_REDUCE) || !defined(NAIAD_MATRIX_FEATURE_SHIFT))
#error "SCAN_PREFIX needs the matrix unit's reduce and shift features"
#endif
#if SCAN_PREFIX && !DW_SAME_FMT && !defined(NAIAD_MATRIX_FEATURE_CONVERT)
#error "SCAN_PREFIX with a 16-bit input format needs the matrix unit's convert feature"
#endif

NAIAD_IO dw_in_t x[N];
NAIAD_IO dw_acc_t y[N];
NAIAD_IO uint64_t region_cycles;

static dw_in_t cu[256] __attribute__((aligned(64)));      /* U  */
static dw_in_t cl[256] __attribute__((aligned(64)));      /* Ls */
static dw_acc_t cj[256] __attribute__((aligned(64)));     /* J  */
static dw_acc_t csel[256] __attribute__((aligned(64)));   /* Sel */
#if !SCAN_MMOV && !SCAN_PREFIX
static dw_acc_t qbuf[2][256] __attribute__((aligned(64))); /* Q round trip, double-buffered */
#endif
#if SCAN_PREFIX
static dw_acc_t cid[256] __attribute__((aligned(64))); /* I, accumulator format */
#endif

int main(void) {
    for (int k = 0; k < 16; k++) {
        for (int j = 0; j < 16; j++) {
            cu[16 * k + j] = dw_in_bits(k <= j ? 1.0 : 0.0);
            cl[16 * k + j] = dw_in_bits(j < k ? 1.0 : 0.0);
            cj[16 * k + j] = (dw_acc_t)1;
            csel[16 * k + j] = (dw_acc_t)(k == 15 ? 1 : 0);
#if SCAN_PREFIX
            cid[16 * k + j] = (dw_acc_t)(k == j ? 1 : 0);
#endif
        }
    }

    uint64_t t0 = naiad_cycles();
#if SCAN_PREFIX
    mx_tload_b(0, ACC, cid, ATS);
#else
    mx_tload_b(0, IN, cu, TS);
    mx_tload_a(1, IN, cl, TS);
    mx_tload_b(2, ACC, cj, ATS);
#endif
#if !SCAN_EPILOGUE
    mx_tload_b(3, ACC, csel, ATS);
#endif
    for (int c = 0; c < N / 256; c++) {
        const dw_in_t *xc = x + 256 * c;
        dw_acc_t *yc = y + 256 * c;
#if !SCAN_MMOV && !SCAN_PREFIX
        dw_acc_t *q = qbuf[c & 1];
#endif
        mx_tload_a(4, IN, xc, TS);
#if SCAN_PREFIX
#if !DW_SAME_FMT
        mx_tconv(4, 4, 0, IN, ACC); /* X in the accumulator format */
#endif
        mx_tprefix(5, 4, 0, 0, ACC, MX_RED_ADD); /* P: row-wise inclusive scans */
        mx_treduce(2, 4, 0, 0, ACC, MX_RED_ADD); /* row totals, along each row */
        mx_tprefix(2, 2, 0, 1, ACC, MX_RED_ADD); /* scanned down the columns */
        mx_tshift(2, 2, 0, 0, ACC, 1);           /* exclusive: E[i] = that, row i - 1 */
        mx_amov_at(0, 5, ACC, 0);                /* acc = P */
        mx_mma(0, 2, 0, ACC, ACC);               /* + E I */
#else
        mx_azero(0);
        mx_azero(1);
        mx_mma(1, 1, 4, IN, ACC); /* Q = Ls X */
        mx_mma(0, 4, 0, IN, ACC); /* X U */
#if SCAN_MMOV
        mx_amov_ta(5, 1, ACC, 0);  /* Q -> tile, no memory */
#else
        mx_astore(1, ACC, q, ATS);
        mx_tload_a(5, ACC, q, ATS);
#endif
        mx_mma(0, 5, 2, ACC, ACC); /* + Q J */
#endif
        if (c) {
#if SCAN_EPILOGUE
            dw_acc_t carry = yc[-1];
            mx_aadds(0, ACC, dw_acc_bits(carry));
#else
            mx_tload_a(6, ACC, yc - 16, 0); /* every row = the previous tile's last row */
            mx_mma(0, 6, 3, ACC, ACC);      /* + its element 15, everywhere */
#endif
        }
        mx_astore(0, ACC, yc, ATS);
    }
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - t0;
    return 0;
}
