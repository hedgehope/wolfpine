/*
 * jacobi_scalar: the scalar F/D baseline for jacobi_ame, with the same host
 * interface and the same rounding: the grid is stored in the input format
 * between sweeps, each point is computed in the accumulator format as
 * (up + down + left + right) * 1/4, and the last sweep is written to out in the
 * accumulator format.
 *
 * JACOBI_RADIUS r > 1 (default 1): the radius-r star stencil, each point the
 * mean of the 4 r points at distance 1..r along its row and column, summed as
 * (((up1 + down1) + left1) + right1) + up2 + down2 + left2 + right2 + ...
 * (left to right), times 1 / (4 r), over a boundary ring of width r.
 */
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"

#ifndef JACOBI_N
#error "define JACOBI_N"
#endif
#ifndef JACOBI_ITERS
#define JACOBI_ITERS 8
#endif
#ifndef JACOBI_RADIUS
#define JACOBI_RADIUS 1
#endif
#define N JACOBI_N
#define R JACOBI_RADIUS

NAIAD_IO dw_in_t grid[N * N];
NAIAD_IO dw_acc_t out[N * N];
NAIAD_IO uint64_t region_cycles;

static dw_in_t g[2][N * N] __attribute__((section(".noinit"), aligned(64)));

int main(void) {
    for (int i = 0; i < N * N; i++) {
        g[0][i] = g[1][i] = grid[i];
        out[i] = dw_in_to_acc(grid[i]);
    }
#if R == 1
    const dw_acc_t q = (dw_acc_t)0.25;
#else
    const dw_acc_t q = (dw_acc_t)(1.0 / (4 * R));
#endif

    uint64_t t0 = naiad_cycles();
    for (int it = 0; it < JACOBI_ITERS; it++) {
        const dw_in_t *s = g[it & 1];
        dw_in_t *d = g[(it + 1) & 1];
        int last = it == JACOBI_ITERS - 1;
#if R == 1
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                dw_acc_t v = (dw_in_to_acc(s[(i - 1) * N + j]) + dw_in_to_acc(s[(i + 1) * N + j]) +
                              dw_in_to_acc(s[i * N + j - 1]) + dw_in_to_acc(s[i * N + j + 1])) *
                             q;
#else
        for (int i = R; i < N - R; i++) {
            for (int j = R; j < N - R; j++) {
                const dw_in_t *p = &s[i * N + j];
                dw_acc_t v = dw_in_to_acc(p[-N]) + dw_in_to_acc(p[N]);
                v += dw_in_to_acc(p[-1]);
                v += dw_in_to_acc(p[1]);
#pragma clang loop unroll(full)
                for (int k = 2; k <= R; k++) {
                    v += dw_in_to_acc(p[-k * N]);
                    v += dw_in_to_acc(p[k * N]);
                    v += dw_in_to_acc(p[-k]);
                    v += dw_in_to_acc(p[k]);
                }
                v *= q;
#endif
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
