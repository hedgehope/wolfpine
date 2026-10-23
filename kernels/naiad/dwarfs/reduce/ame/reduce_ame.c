/*
 * reduce_ame: the sum of a length-N vector as a product with a ones tile.
 *
 * x is laid out as N / 256 consecutive 16 x 16 row-major tiles X_c. Each is an
 * "a" operand against a ones "b" tile J, all into one accumulator:
 *   acc = sum_c X_c J,   so acc[i][j] = sum_c sum_k X_c[i][k] = p_i for every j,
 * 16 partial sums, each replicated across the 16 columns (the 16x redundancy a
 * ones *vector* costs on a tile unit). The finish (REDUCE_FINISH_MMA):
 *   1 (default): store acc, reload it as an accumulator-format "b" tile P
 *     (P[k][j] = p_k) and form J' P with an accumulator-format ones tile J':
 *     every element is sum_k p_k. One more mma, on the matrix unit.
 *   0: store acc and add the 16 partials on the scalar side.
 * Four tile registers rotate so the loads of one group of four tiles run ahead
 * of their mma (the L1 port, not the mma pipe, is the bound).
 *
 * Host interface: x (in, N input-format elements), result (the sum, one
 * accumulator-format element), region_cycles.
 */
#include "matrix.h"
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"
#include "dwarf_util.h"

#ifndef REDUCE_N
#error "define REDUCE_N (a multiple of 1024)"
#endif
#ifndef REDUCE_FINISH_MMA
#define REDUCE_FINISH_MMA 1
#endif
#define N REDUCE_N
#define IN DW_IN_FMT
#define ACC DW_ACC_FMT
#define TS (16L * DW_IN_BYTES)
#define ATS (16L * DW_ACC_BYTES)

#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16 || NAIAD_MATRIX_TILES < 7
#error "reduce_ame is written for a 16 x 16 x 16 unit with at least 7 tiles"
#endif
#if N % 1024
#error "REDUCE_N must be a multiple of 1024"
#endif

NAIAD_IO dw_in_t x[N];
NAIAD_IO dw_acc_t result[1];
NAIAD_IO uint64_t region_cycles;

static dw_in_t ones[256] __attribute__((aligned(64)));
static dw_acc_t ones_acc[256] __attribute__((aligned(64)));
static dw_acc_t part[256] __attribute__((aligned(64)));

int main(void) {
    for (int i = 0; i < 256; i++) {
        ones[i] = dw_in_bits(1.0);
        ones_acc[i] = (dw_acc_t)1;
    }

    uint64_t t0 = naiad_cycles();
    mx_tload_b(0, IN, ones, TS);
    mx_azero(0);
    for (const dw_in_t *p = x; p < x + N; p += 1024) {
        mx_tload_a(1, IN, p, TS);
        mx_tload_a(2, IN, p + 256, TS);
        mx_tload_a(3, IN, p + 512, TS);
        mx_tload_a(4, IN, p + 768, TS);
        mx_mma(0, 1, 0, IN, ACC);
        mx_mma(0, 2, 0, IN, ACC);
        mx_mma(0, 3, 0, IN, ACC);
        mx_mma(0, 4, 0, IN, ACC);
    }
    mx_astore(0, ACC, part, ATS);
#if REDUCE_FINISH_MMA
    mx_tload_b(5, ACC, part, ATS);
    mx_tload_a(6, ACC, ones_acc, ATS);
    mx_azero(1);
    mx_mma(1, 6, 5, ACC, ACC);
    mx_astore(1, ACC, part, ATS);
    dw_acc_t s = part[0];
#else
    dw_acc_t s = 0;
    for (int i = 0; i < 16; i++) {
        s += part[16 * i];
    }
#endif
    result[0] = s;
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - t0;
    return 0;
}
