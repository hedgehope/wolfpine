/* scan_scalar: the scalar F/D baseline for scan_ame: y[i] = y[i - 1] + x[i] in the accumulator format. */
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"

#ifndef SCAN_N
#error "define SCAN_N"
#endif
#define N SCAN_N

NAIAD_IO dw_in_t x[N];
NAIAD_IO dw_acc_t y[N];
NAIAD_IO uint64_t region_cycles;

int main(void) {
    uint64_t t0 = naiad_cycles();
    dw_acc_t s = 0;
    for (int i = 0; i < N; i++) {
        s += dw_in_to_acc(x[i]);
        y[i] = s;
    }
    region_cycles = naiad_cycles() - t0;
    return 0;
}
