/*
 * toymm (ame): the harness's toy dwarf, C = A x B (TOY_N square) on the matrix
 * unit, one accumulator per output tile. Inputs and outputs are NAIAD_IO
 * arrays the host writes and reads; the GEMM loop is the timed region.
 */
#include "matrix.h"
#include "naiad_platform.h"

#if TOY_N % NAIAD_MATRIX_M || TOY_N % NAIAD_MATRIX_N || TOY_N % NAIAD_MATRIX_K
#error "TOY_N must be a multiple of the tile M, N and K"
#endif

NAIAD_IO TOY_IN_T A[TOY_N][TOY_N];
NAIAD_IO TOY_IN_T B[TOY_N][TOY_N];
NAIAD_IO TOY_ACC_T C[TOY_N][TOY_N];
NAIAD_IO uint64_t region_cycles;

int main(void) {
    const long in_row = TOY_N * sizeof(TOY_IN_T), acc_row = TOY_N * sizeof(TOY_ACC_T);
    uint64_t start = naiad_cycles();
    for (int i0 = 0; i0 < TOY_N; i0 += NAIAD_MATRIX_M) {
        for (int j0 = 0; j0 < TOY_N; j0 += NAIAD_MATRIX_N) {
            mx_azero(0);
            for (int k0 = 0; k0 < TOY_N; k0 += NAIAD_MATRIX_K) {
                mx_tload_a(0, TOY_IN_FMT, &A[i0][k0], in_row);
                mx_tload_b(1, TOY_IN_FMT, &B[k0][j0], in_row);
                mx_mma(0, 0, 1, TOY_IN_FMT, TOY_ACC_FMT);
            }
            mx_astore(0, TOY_ACC_FMT, &C[i0][j0], acc_row);
        }
    }
    region_cycles = naiad_cycles() - start;
    return 0;
}
