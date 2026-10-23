/* reduce_scalar: the scalar F/D baseline for reduce_ame: one running sum in the accumulator format. */
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"

#ifndef REDUCE_N
#error "define REDUCE_N"
#endif
#define N REDUCE_N

NAIAD_IO dw_in_t x[N];
NAIAD_IO dw_acc_t result[1];
NAIAD_IO uint64_t region_cycles;

int main(void) {
    uint64_t t0 = naiad_cycles();
    dw_acc_t s = 0;
    for (int i = 0; i < N; i++) {
        s += dw_in_to_acc(x[i]);
    }
    result[0] = s;
    region_cycles = naiad_cycles() - t0;
    return 0;
}
