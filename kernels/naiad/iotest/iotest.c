/* Host-I/O smoke kernel for framework/naiad/kernel_io.py: y = 2 * x + n. */
#include <stdint.h>

#include "naiad_platform.h"

#define MAX_N 64

NAIAD_IO volatile int64_t n;
NAIAD_IO int64_t x[MAX_N];
NAIAD_IO int64_t y[MAX_N];

int main(void) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = 2 * x[i] + n;
    }
    return 0;
}
