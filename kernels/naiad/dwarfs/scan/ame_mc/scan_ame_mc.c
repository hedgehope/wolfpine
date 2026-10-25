/*
 * scan_ame_mc: scan_ame (default switches: matrix carry, Q through memory) with
 * x split across the harts of a multi-hart design, as a reduce-then-scan.
 *
 * Hart h owns a contiguous run of 256-element tiles. The carry chain of
 * scan_ame is serial from one tile to the next, so the harts cannot simply
 * scan their runs at once: each needs the sum of everything before it.
 *   1. reduce: every hart but the last sums its run on its own matrix unit
 *      (acc = sum_c X_c J, the reduce dwarf's product; one tile load and one
 *      mma per tile, against the scan's four mma), then its 16 partials on the
 *      scalar side, into total[h];
 *   2. barrier;
 *   3. scan: hart h's carry-in is total[0] + ... + total[h-1] (scalar adds),
 *      fed to its first tile exactly like a carry from a previous tile (a
 *      stride-0 row whose element 15 is the carry, times Sel), and the run is
 *      scanned as in scan_ame.
 * The reduce pass is the parallel scan's extra work (about 1/8 of the scan's
 * time per tile); on one hart it is skipped and this is scan_ame plus two
 * no-op barriers. The carry-ins are computed in a different order from the
 * sequential chain, so with several harts the last bits can differ.
 *
 * Timed region (hart 0's counter): a barrier every hart has reached, to a
 * barrier after every hart's last astore has left its unit.
 */
#include "matrix.h"
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"
#include "dwarf_util.h"

#ifndef SCAN_N
#error "define SCAN_N (a multiple of 256)"
#endif
#if (defined(SCAN_EPILOGUE) && SCAN_EPILOGUE) || (defined(SCAN_MMOV) && SCAN_MMOV) || (defined(SCAN_PREFIX) && SCAN_PREFIX)
#error "scan_ame_mc has only scan_ame's default path"
#endif
#define N SCAN_N
#define IN DW_IN_FMT
#define ACC DW_ACC_FMT
#define TS (16L * DW_IN_BYTES)
#define ATS (16L * DW_ACC_BYTES)
#define TILES (N / 256)

#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16 || NAIAD_MATRIX_TILES < 8 ||           \
    NAIAD_MATRIX_ACCS < 3
#error "scan_ame_mc is written for a 16 x 16 x 16 unit with at least 8 tiles and 3 accumulators"
#endif
#if N % 256
#error "SCAN_N must be a multiple of 256"
#endif

NAIAD_IO dw_in_t x[N];
NAIAD_IO dw_acc_t y[N];
NAIAD_IO uint64_t region_cycles;

static dw_in_t cu[256] __attribute__((aligned(64)));    /* U  */
static dw_in_t cl[256] __attribute__((aligned(64)));    /* Ls */
static dw_in_t cjin[256] __attribute__((aligned(64)));  /* J, input format (the reduce pass) */
static dw_acc_t cj[256] __attribute__((aligned(64)));   /* J  */
static dw_acc_t csel[256] __attribute__((aligned(64))); /* Sel */
/* Per hart: the Q round trip (double-buffered), the reduce pass's partials, its total and its carry-in row. */
static struct {
    dw_acc_t q[2][256];
    dw_acc_t part[256];
    dw_acc_t carry[16];
    dw_acc_t total;
} hs[NAIAD_HARTS] __attribute__((aligned(64)));

int main(void) {
    const unsigned h = naiad_hartid();
    /* Hart h's tiles: [c0, c1), the first TILES % NAIAD_HARTS harts one more. */
    const int per = TILES / NAIAD_HARTS, extra = TILES % NAIAD_HARTS;
    const int c0 = (int)h * per + ((int)h < extra ? (int)h : extra);
    const int c1 = c0 + per + ((int)h < extra);
    if (h == 0) {
        for (int k = 0; k < 16; k++) {
            for (int j = 0; j < 16; j++) {
                cu[16 * k + j] = dw_in_bits(k <= j ? 1.0 : 0.0);
                cl[16 * k + j] = dw_in_bits(j < k ? 1.0 : 0.0);
                cjin[16 * k + j] = dw_in_bits(1.0);
                cj[16 * k + j] = (dw_acc_t)1;
                csel[16 * k + j] = (dw_acc_t)(k == 15 ? 1 : 0);
            }
        }
    }
    naiad_barrier();

    uint64_t t0 = naiad_cycles();
    if (NAIAD_HARTS > 1) {
        /* 1. reduce this hart's run (the last hart's total is never needed) */
        if (h + 1 < NAIAD_HARTS) {
            mx_tload_b(7, IN, cjin, TS);
            mx_azero(2);
            int c = c0;
            for (; c + 1 < c1; c += 2) {
                mx_tload_a(4, IN, x + 256 * c, TS);
                mx_tload_a(5, IN, x + 256 * (c + 1), TS);
                mx_mma(2, 4, 7, IN, ACC);
                mx_mma(2, 5, 7, IN, ACC);
            }
            if (c < c1) {
                mx_tload_a(4, IN, x + 256 * c, TS);
                mx_mma(2, 4, 7, IN, ACC);
            }
            mx_astore(2, ACC, hs[h].part, ATS);
            dw_acc_t s = 0;
            for (int i = 0; i < 16; i++) {
                s += hs[h].part[16 * i];
            }
            hs[h].total = s;
        }
        naiad_barrier();
        /* 3. this hart's carry-in */
        if (h > 0) {
            dw_acc_t carry = hs[0].total;
            for (unsigned g = 1; g < h; g++) {
                carry += hs[g].total;
            }
            for (int j = 0; j < 15; j++) {
                hs[h].carry[j] = 0;
            }
            hs[h].carry[15] = carry;
        }
    }

    mx_tload_b(0, IN, cu, TS);
    mx_tload_a(1, IN, cl, TS);
    mx_tload_b(2, ACC, cj, ATS);
    mx_tload_b(3, ACC, csel, ATS);
    for (int c = c0; c < c1; c++) {
        const dw_in_t *xc = x + 256 * c;
        dw_acc_t *yc = y + 256 * c;
        dw_acc_t *q = hs[h].q[c & 1];
        mx_tload_a(4, IN, xc, TS);
        mx_azero(0);
        mx_azero(1);
        mx_mma(1, 1, 4, IN, ACC); /* Q = Ls X */
        mx_mma(0, 4, 0, IN, ACC); /* X U */
        mx_astore(1, ACC, q, ATS);
        mx_tload_a(5, ACC, q, ATS);
        mx_mma(0, 5, 2, ACC, ACC); /* + Q J */
        if (c > c0) {
            mx_tload_a(6, ACC, yc - 16, 0); /* every row = the previous tile's last row */
            mx_mma(0, 6, 3, ACC, ACC);      /* + its element 15, everywhere */
        } else if (h > 0) {
            mx_tload_a(6, ACC, hs[h].carry, 0); /* every row = (0, ..., 0, carry-in) */
            mx_mma(0, 6, 3, ACC, ACC);
        }
        mx_astore(0, ACC, yc, ATS);
    }
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    naiad_barrier();
    if (h == 0) {
        region_cycles = naiad_cycles() - t0;
    }
    return 0;
}
