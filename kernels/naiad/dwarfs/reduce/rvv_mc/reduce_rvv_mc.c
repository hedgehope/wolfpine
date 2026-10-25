/*
 * reduce_rvv_mc: reduce_rvv with x split across the harts of a multi-hart
 * design (each hart has its own vector unit).
 *
 * Hart h reduces its contiguous slice (whole VLMAX chunks, the last hart takes
 * the tail) exactly as reduce_rvv does the whole vector, and stores its scalar
 * partial to its own 64-byte line; after a barrier hart 0 adds the partials in
 * hart order. On one hart this is reduce_rvv plus two no-op barriers. With
 * several harts the sum's order differs, so the last bits can.
 *
 * Timed region (hart 0's counter): a barrier every hart has reached, to the
 * combined sum.
 */
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"

#define RV_AL 4
#include "rvv_fmt.h"

#ifndef REDUCE_N
#error "define REDUCE_N"
#endif
#define N REDUCE_N

NAIAD_IO dw_in_t x[N];
NAIAD_IO dw_acc_t result[1];
NAIAD_IO uint64_t region_cycles;

/* One partial per hart, each on its own cache line. */
static struct {
    rv_acc_s s;
    char pad[64 - sizeof(rv_acc_s)];
} partial[NAIAD_HARTS] __attribute__((aligned(64)));

RV_KERNEL static rv_acc_s reduce_slice(const dw_in_t *xs, long n) {
    const long vlmax = (long)rv_vlmax();
    rv_acc_v acc0 = rv_splat(0, vlmax), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    long i = 0;
    if (n >= 4 * vlmax) {
        rv_in_v a = rv_load_in(&xs[0], vlmax), b = rv_load_in(&xs[vlmax], vlmax);
        rv_in_v c = rv_load_in(&xs[2 * vlmax], vlmax), d = rv_load_in(&xs[3 * vlmax], vlmax);
        for (i = 4 * vlmax; i + 4 * vlmax <= n; i += 4 * vlmax) {
            acc0 = rv_acc_add_in(acc0, a, vlmax);
            a = rv_load_in(&xs[i], vlmax);
            acc1 = rv_acc_add_in(acc1, b, vlmax);
            b = rv_load_in(&xs[i + vlmax], vlmax);
            acc2 = rv_acc_add_in(acc2, c, vlmax);
            c = rv_load_in(&xs[i + 2 * vlmax], vlmax);
            acc3 = rv_acc_add_in(acc3, d, vlmax);
            d = rv_load_in(&xs[i + 3 * vlmax], vlmax);
        }
        acc0 = rv_acc_add_in(acc0, a, vlmax);
        acc1 = rv_acc_add_in(acc1, b, vlmax);
        acc2 = rv_acc_add_in(acc2, c, vlmax);
        acc3 = rv_acc_add_in(acc3, d, vlmax);
    }
    for (long vl; i < n; i += vl) { /* tail: lanes past vl keep their partial sums */
        vl = (long)rv_setvl(n - i);
        acc0 = rv_acc_add_in_tu(acc0, rv_load_in(&xs[i], vl), vl);
    }
    acc0 = __riscv_vfadd(__riscv_vfadd(acc0, acc1, vlmax), __riscv_vfadd(acc2, acc3, vlmax), vlmax);
    return rv_reduce_usum(acc0, vlmax);
}

RV_KERNEL int main(void) {
    const unsigned h = naiad_hartid();
    naiad_barrier();
    uint64_t t0 = naiad_cycles();
    /* Hart h's slice: whole VLMAX chunks, as evenly as they divide; the last hart takes the tail. */
    const long vlmax = (long)rv_vlmax();
    const long chunks = N / vlmax, per = chunks / NAIAD_HARTS, extra = chunks % NAIAD_HARTS;
    const long c0 = h * per + ((long)h < extra ? (long)h : extra);
    const long c1 = c0 + per + ((long)h < extra);
    const long i0 = c0 * vlmax, i1 = h == NAIAD_HARTS - 1 ? N : c1 * vlmax;
    rv_acc_s s = reduce_slice(x + i0, i1 - i0);
    if (NAIAD_HARTS > 1) {
        partial[h].s = s;
        naiad_barrier();
        if (h == 0) {
            s = partial[0].s;
            for (unsigned g = 1; g < NAIAD_HARTS; g++) {
                s += partial[g].s;
            }
        }
    }
    if (h == 0) {
        result[0] = s;
        region_cycles = naiad_cycles() - t0;
    }
    return 0;
}
