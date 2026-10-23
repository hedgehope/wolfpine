/*
 * jacobi_scalar: the scalar F/D baseline for jacobi_ame, with the same host
 * interface and the same rounding: the grid is stored in the input format
 * between sweeps, each point is computed in the accumulator format as
 * (up + down + left + right) * 1/4, and the last sweep is written to out in the
 * accumulator format.
 */
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"

#ifndef JACOBI_N
#error "define JACOBI_N"
#endif
#ifndef JACOBI_ITERS
#define JACOBI_ITERS 8
#endif
#define N JACOBI_N

NAIAD_IO dw_in_t grid[N * N];
NAIAD_IO dw_acc_t out[N * N];
NAIAD_IO uint64_t region_cycles;

static dw_in_t g[2][N * N] __attribute__((section(".noinit"), aligned(64)));

int main(void) {
    for (int i = 0; i < N * N; i++) {
        g[0][i] = g[1][i] = grid[i];
        out[i] = dw_in_to_acc(grid[i]);
    }
    const dw_acc_t q = (dw_acc_t)0.25;

    uint64_t t0 = naiad_cycles();
    for (int it = 0; it < JACOBI_ITERS; it++) {
        const dw_in_t *s = g[it & 1];
        dw_in_t *d = g[(it + 1) & 1];
        int last = it == JACOBI_ITERS - 1;
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                dw_acc_t v = (dw_in_to_acc(s[(i - 1) * N + j]) + dw_in_to_acc(s[(i + 1) * N + j]) +
                              dw_in_to_acc(s[i * N + j - 1]) + dw_in_to_acc(s[i * N + j + 1])) *
                             q;
                if (last) {
                    out[i * N + j] = v;
                } else {
                    d[i * N + j] = dw_acc_to_in(v);
                }
            }
        }
    }
    region_cycles = naiad_cycles() - t0;
    return 0;
}
