/*
 * toymm (scalar): the toy dwarf's F/D baseline, C = A x B in TOY_ACC_T on the
 * scalar core. Only fp32/fp32 and fp64/fp64 build (spec.cflags refuses the rest).
 */
#include "naiad_platform.h"

NAIAD_IO TOY_IN_T A[TOY_N][TOY_N];
NAIAD_IO TOY_IN_T B[TOY_N][TOY_N];
NAIAD_IO TOY_ACC_T C[TOY_N][TOY_N];
NAIAD_IO uint64_t region_cycles;

int main(void) {
    uint64_t start = naiad_cycles();
    for (int i = 0; i < TOY_N; i++) {
        for (int j = 0; j < TOY_N; j++) {
            TOY_ACC_T s = 0;
            for (int k = 0; k < TOY_N; k++) {
                s += A[i][k] * B[k][j];
            }
            C[i][j] = s;
        }
    }
    region_cycles = naiad_cycles() - start;
    return 0;
}
