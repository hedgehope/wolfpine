/*
 * jacobi_ame_mc: jacobi_ame with the output tile rows split across the harts of
 * a multi-hart design (each hart has its own matrix unit).
 *
 * The sweeps are jacobi_ame's own (this file includes it, without its main,
 * and with JACOBI_ROW0 / JACOBI_ROW1 set to this hart's run of tile rows); every
 * switch works as there. Each sweep writes a disjoint set of rows of the next
 * buffer and reads the rows on both sides of its run, so the harts meet at a
 * barrier after every sweep (after their stores have left the unit) and, for
 * bf16, the re-rounding pass touches only the hart's own rows. Every output has
 * one writer and the same sums, so results are bit-identical to jacobi_ame's.
 * (N - 2) / 16 tile rows is the parallelism: at N = 66 four, so a fifth hart
 * has nothing to do.
 *
 * Timed region (hart 0's counter): a barrier every hart has reached, to the
 * last sweep's barrier. On one hart this is jacobi_ame plus no-op barriers.
 */
#include "naiad_platform.h"

#if defined(JACOBI_RADIUS) && JACOBI_RADIUS != 1
#error "jacobi_ame_mc is radius 1 only (use the single-hart variant for JACOBI_RADIUS > 1)"
#endif

#if NAIAD_HARTS > 1
static int jacobi_row0(void);
static int jacobi_row1(void);
#define JACOBI_ROW0 jacobi_row0()
#define JACOBI_ROW1 jacobi_row1()
#endif

#define JACOBI_NO_MAIN 1
#include "../ame/jacobi_ame.c"

#if NAIAD_HARTS > 1
/* Hart h's tile rows: [r0, r1) of TT, the first TT % NAIAD_HARTS harts one more. */
static int jacobi_row0(void) {
    const int h = (int)naiad_hartid();
    return h * (TT / NAIAD_HARTS) + (h < TT % NAIAD_HARTS ? h : TT % NAIAD_HARTS);
}
static int jacobi_row1(void) {
    const int h = (int)naiad_hartid();
    return jacobi_row0() + TT / NAIAD_HARTS + (h < TT % NAIAD_HARTS);
}
#endif

int main(void) {
    if (naiad_hartid() == 0) {
        make_constants();
        for (int i = 0; i < N * N; i++) {
            work.g[0][i] = work.g[1][i] = grid[i];
            out[i] = dw_in_to_acc(grid[i]);
        }
    }
    naiad_barrier();

    uint64_t t0 = naiad_cycles();
    const int mine = JACOBI_ROW1 > JACOBI_ROW0;
    for (int it = 0; it < JACOBI_ITERS; it++) {
        const dw_in_t *src = work.g[it & 1];
        dw_in_t *next = work.g[(it + 1) & 1];
        int last = it == JACOBI_ITERS - 1;
        if (mine) {
            if (DW_SAME_FMT && !last) {
                sweep(src, (char *)next, RS, DW_IN_BYTES CONV_ARG(0));
#if JACOBI_MCONV
            } else if (!last) {
                sweep(src, (char *)next, RS, DW_IN_BYTES, 1);
#endif
            } else {
                sweep(src, (char *)out, ORS, DW_ACC_BYTES CONV_ARG(0));
                if (!last) {
                    to_input_format(next);
                }
            }
            if (NAIAD_HARTS > 1) {
                naiad_mx_drain(); /* this sweep's rows are in memory before the barrier says so */
            }
        }
        naiad_barrier();
    }
    naiad_mx_drain();
    if (naiad_hartid() == 0) {
        region_cycles = naiad_cycles() - t0;
    }
    return 0;
}
