/*
 * jacobi_rvv: the RVV baseline for jacobi_ame, with the same host interface
 * and the same rounding as jacobi_scalar: the grid is stored in the input
 * format between sweeps, each point is ((up + down) + left) + right in the
 * accumulator format, times 1/4, and the last sweep goes to out in the
 * accumulator format. The order of the adds is the scalar loop's, so fp32 and
 * fp64 results match it bit for bit.
 *
 * Vectorised along a row: the interior columns are strip-mined at LMUL 4
 * (32 fp32 / 16 fp64 lanes at VLEN 256) and every neighbour is one unit-stride
 * load: up and down from the rows above and below, left and right as
 * unaligned loads one element either side (no slides: they would move work
 * from the idle-ish load unit onto the datapath, which a 5-point stencil
 * already balances against it: 4 loads + 1 store against 4 arithmetic ops).
 * Two rows go together so their dependent add chains interleave on the
 * in-order core (one row's chain is 4 ops of latency startup + occupancy).
 * 16-bit inputs are loaded at LMUL 2 and enter through the widening adds
 * (Zvfh vfwadd; bf16 vfwmaccbf16 by 1.0, one rounding like an fp32 add), and
 * each intermediate sweep is narrowed back with vfncvt / vfncvtbf16 (RNE).
 */
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"

#define RV_AL 4
#include "rvv_fmt.h"

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

/*
 * One sweep from s; STORE(i, j, v, vl) writes row i's new values. Rows pair up
 * (N - 2 is even). The order is hand-scheduled for the in-order core (the
 * compiler keeps it): row A's four loads, then row B's interleaved with A's
 * adds, then B's adds, so each add finds its operands loaded and the load unit
 * never waits for the datapath.
 */
#define SWEEP(STORE)                                                                       \
    for (long j = 1, vl; j < N - 1; j += vl) {                                             \
        vl = (long)rv_setvl(N - 1 - j);                                                    \
        for (long i = 1; i < N - 1; i += 2) {                                              \
            const dw_in_t *pa = &s[i * N + j], *pb = pa + N;                               \
            rv_in_v ua = rv_load_in(pa - N, vl), da = rv_load_in(pa + N, vl);              \
            rv_in_v ub = rv_load_in(pa, vl), db = rv_load_in(pb + N, vl);                  \
            RV_PIN(ua); RV_PIN(da);                                                        \
            rv_in_v la = rv_load_in(pa - 1, vl), lb = rv_load_in(pb - 1, vl);              \
            RV_PIN(ub); RV_PIN(db);                                                        \
            rv_acc_v va = rv_add_in2(ua, da, vl);                                          \
            rv_in_v ra = rv_load_in(pa + 1, vl), rb = rv_load_in(pb + 1, vl);              \
            rv_acc_v vb = rv_add_in2(ub, db, vl);                                          \
            RV_PIN(la); RV_PIN(lb);                                                        \
            RV_PIN(va); RV_PIN(vb);                                                        \
            va = rv_acc_add_in(va, la, vl);                                                \
            vb = rv_acc_add_in(vb, lb, vl);                                                \
            RV_PIN(ra); RV_PIN(rb); RV_PIN(va); RV_PIN(vb);                                \
            va = rv_acc_add_in(va, ra, vl);                                                \
            vb = rv_acc_add_in(vb, rb, vl);                                                \
            RV_PIN(va); RV_PIN(vb);                                                        \
            va = __riscv_vfmul(va, (rv_acc_s)0.25, vl);                                    \
            vb = __riscv_vfmul(vb, (rv_acc_s)0.25, vl);                                    \
            STORE(i, j, va, vl);                                                           \
            STORE(i + 1, j, vb, vl);                                                       \
        }                                                                                  \
    }
#define STORE_IN(i, j, v, vl) rv_store_in(&d[(i) * N + (j)], rv_narrow((v), (vl)), (vl))
#define STORE_OUT(i, j, v, vl) rv_store_acc(&out[(i) * N + (j)], (v), (vl))

RV_KERNEL int main(void) {
    for (int i = 0; i < N * N; i++) {
        g[0][i] = g[1][i] = grid[i];
        out[i] = dw_in_to_acc(grid[i]);
    }

    uint64_t t0 = naiad_cycles();
    for (int it = 0; it < JACOBI_ITERS; it++) {
        const dw_in_t *s = g[it & 1];
        dw_in_t *d = g[(it + 1) & 1];
        if (it == JACOBI_ITERS - 1) {
            SWEEP(STORE_OUT)
        } else {
            SWEEP(STORE_IN)
        }
    }
    region_cycles = naiad_cycles() - t0;
    return 0;
}
