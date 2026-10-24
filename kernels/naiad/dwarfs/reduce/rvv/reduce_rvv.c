/*
 * reduce_rvv: the RVV baseline for reduce_ame.
 *
 * Four LMUL-4 vector accumulators in the accumulator format take successive
 * VLMAX-element chunks of x (4 x 32 fp32 lanes at VLEN 256), with each
 * chunk's load issued one iteration ahead of its add (software pipelining by
 * hand: the in-order core would otherwise wait out every load's latency), so
 * the load unit streams one register group per add. 16-bit inputs are read at
 * LMUL 2 and added by the widening forms (fp16 vfwadd.wv; bf16 vfwmaccbf16 by
 * 1.0), so they are never rounded again. The lanes are then combined with
 * vfadds and a single
 * **unordered** reduction, vfredusum:
 *
 *  - Accuracy: the error is set by the lane-parallel accumulation (128
 *    interleaved partial sums of n/128 terms each), not by the final
 *    combine, so the reduction order barely matters; an ordered
 *    vfredosum would not make the result the sequential sum either.
 *  - Speed: vfredusum may be a tree (the timing model charges log2(lanes)
 *    extra cycles); vfredosum is defined as a sequential chain, which real
 *    implementations execute at about one element per cycle.
 *  - The simulator evaluates vfredusum in element order (as Spike does), so
 *    results are deterministic.
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

RV_KERNEL int main(void) {
    uint64_t t0 = naiad_cycles();
    const long vlmax = (long)rv_vlmax();
    rv_acc_v acc0 = rv_splat(0, vlmax), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    long i = 0;
    if (N >= 4 * vlmax) {
        rv_in_v a = rv_load_in(&x[0], vlmax), b = rv_load_in(&x[vlmax], vlmax);
        rv_in_v c = rv_load_in(&x[2 * vlmax], vlmax), d = rv_load_in(&x[3 * vlmax], vlmax);
        for (i = 4 * vlmax; i + 4 * vlmax <= N; i += 4 * vlmax) {
            acc0 = rv_acc_add_in(acc0, a, vlmax);
            a = rv_load_in(&x[i], vlmax);
            acc1 = rv_acc_add_in(acc1, b, vlmax);
            b = rv_load_in(&x[i + vlmax], vlmax);
            acc2 = rv_acc_add_in(acc2, c, vlmax);
            c = rv_load_in(&x[i + 2 * vlmax], vlmax);
            acc3 = rv_acc_add_in(acc3, d, vlmax);
            d = rv_load_in(&x[i + 3 * vlmax], vlmax);
        }
        acc0 = rv_acc_add_in(acc0, a, vlmax);
        acc1 = rv_acc_add_in(acc1, b, vlmax);
        acc2 = rv_acc_add_in(acc2, c, vlmax);
        acc3 = rv_acc_add_in(acc3, d, vlmax);
    }
    for (long vl; i < N; i += vl) { /* tail: lanes past vl keep their partial sums */
        vl = (long)rv_setvl(N - i);
        acc0 = rv_acc_add_in_tu(acc0, rv_load_in(&x[i], vl), vl);
    }
    acc0 = __riscv_vfadd(__riscv_vfadd(acc0, acc1, vlmax), __riscv_vfadd(acc2, acc3, vlmax), vlmax);
    rv_acc_s s = rv_reduce_usum(acc0, vlmax);
    result[0] = s;
    region_cycles = naiad_cycles() - t0;
    return 0;
}
