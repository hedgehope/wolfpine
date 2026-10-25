/*
 * reduce_ame_mc: reduce_ame (the default mma finish) with x split across the
 * harts of a multi-hart design; each hart has its own matrix unit.
 *
 * Hart h sums its contiguous run of 1024-element groups into its own
 * accumulator (acc = sum_c X_c J, as reduce_ame) and stores the 16 replicated
 * partials to part[h]. After a barrier hart 0 finishes all of them on its unit:
 * acc' = sum_h J' P_h, one mma per hart instead of reduce_ame's one. On one hart
 * this is reduce_ame's instruction sequence plus two no-op barriers.
 *
 * The sum's order differs from reduce_ame's once there are several harts (each
 * hart's partials are summed separately and then combined), so the result can
 * differ in the last bits; it is the same blocked accumulation, just split.
 *
 * Timed region (hart 0's counter): a barrier every hart has reached, to hart
 * 0's last astore of the combined sum.
 */
#include "matrix.h"
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"
#include "dwarf_util.h"

#ifndef REDUCE_N
#error "define REDUCE_N (a multiple of 1024)"
#endif
#if defined(REDUCE_FINISH_MMA) && !REDUCE_FINISH_MMA
#error "reduce_ame_mc only has the mma finish"
#endif
#if (defined(REDUCE_FINISH_MMOV) && REDUCE_FINISH_MMOV) || (defined(REDUCE_FINISH_NATIVE) && REDUCE_FINISH_NATIVE)
#error "reduce_ame_mc only has the mma finish"
#endif
#define N REDUCE_N
#define IN DW_IN_FMT
#define ACC DW_ACC_FMT
#define TS (16L * DW_IN_BYTES)
#define ATS (16L * DW_ACC_BYTES)
#define GROUPS (N / 1024)

#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16 || NAIAD_MATRIX_TILES < 7
#error "reduce_ame_mc is written for a 16 x 16 x 16 unit with at least 7 tiles"
#endif
#if N % 1024
#error "REDUCE_N must be a multiple of 1024"
#endif

NAIAD_IO dw_in_t x[N];
NAIAD_IO dw_acc_t result[1];
NAIAD_IO uint64_t region_cycles;

static dw_in_t ones[256] __attribute__((aligned(64)));
static dw_acc_t ones_acc[256] __attribute__((aligned(64)));
static dw_acc_t part[NAIAD_HARTS][256] __attribute__((aligned(64)));
static dw_acc_t total[256] __attribute__((aligned(64)));

int main(void) {
    const unsigned h = naiad_hartid();
    /* Hart h's groups: [g0, g1), the first GROUPS % NAIAD_HARTS harts one more. */
    const long per = GROUPS / NAIAD_HARTS, extra = GROUPS % NAIAD_HARTS;
    const long g0 = h * per + ((long)h < extra ? (long)h : extra);
    const long g1 = g0 + per + ((long)h < extra);
    if (h == 0) {
        for (int i = 0; i < 256; i++) {
            ones[i] = dw_in_bits(1.0);
            ones_acc[i] = (dw_acc_t)1;
        }
    }
    naiad_barrier();

    uint64_t t0 = naiad_cycles();
    mx_tload_b(0, IN, ones, TS);
    mx_azero(0);
    for (const dw_in_t *p = x + 1024 * g0; p < x + 1024 * g1; p += 1024) {
        mx_tload_a(1, IN, p, TS);
        mx_tload_a(2, IN, p + 256, TS);
        mx_tload_a(3, IN, p + 512, TS);
        mx_tload_a(4, IN, p + 768, TS);
        mx_mma(0, 1, 0, IN, ACC);
        mx_mma(0, 2, 0, IN, ACC);
        mx_mma(0, 3, 0, IN, ACC);
        mx_mma(0, 4, 0, IN, ACC);
    }
    mx_astore(0, ACC, part[h], ATS);
    if (NAIAD_HARTS > 1) {
        naiad_mx_drain(); /* part[h] is in memory before the barrier says so */
    }
    naiad_barrier();
    if (h == 0) {
        for (unsigned g = 0; g < NAIAD_HARTS; g++) {
            mx_tload_b(5, ACC, part[g], ATS);
            if (g == 0) {
                mx_tload_a(6, ACC, ones_acc, ATS);
                mx_azero(1);
            }
            mx_mma(1, 6, 5, ACC, ACC);
        }
        mx_astore(1, ACC, total, ATS);
        result[0] = total[0];
        naiad_mx_drain();
        region_cycles = naiad_cycles() - t0;
    }
    return 0;
}
