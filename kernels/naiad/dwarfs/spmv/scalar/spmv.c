/*
 * spmv_scalar: the scalar F/D baseline for spmv_ame: the same BSR walk, one
 * running sum per (row, vector) in the accumulator format, blocks in storage
 * order, columns in order within a block.
 */
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"

#if !defined(SPMV_N) || !defined(SPMV_NNZB) || !defined(SPMV_NV)
#error "define SPMV_N, SPMV_NNZB and SPMV_NV"
#endif
#define N SPMV_N
#define NB (N / 16)
#define NNZB SPMV_NNZB
#define NV SPMV_NV

NAIAD_IO int32_t rowptr[NB + 1];
NAIAD_IO int32_t colidx[NNZB];
NAIAD_IO dw_in_t vals[NNZB * 256];
NAIAD_IO dw_in_t X[N * NV + 16];
NAIAD_IO dw_acc_t Y[N * NV];
NAIAD_IO uint64_t region_cycles;

int main(void) {
    uint64_t t0 = naiad_cycles();
    for (int bi = 0; bi < NB; bi++) {
        const int p0 = rowptr[bi], p1 = rowptr[bi + 1];
        for (int r = 0; r < 16; r++) {
            for (int v = 0; v < NV; v++) {
                dw_acc_t s = 0;
                for (int p = p0; p < p1; p++) {
                    const dw_in_t *a = &vals[(long)p * 256 + r * 16];
                    const dw_in_t *x = &X[(long)colidx[p] * 16 * NV + v];
                    for (int c = 0; c < 16; c++) {
                        s += dw_in_to_acc(a[c]) * dw_in_to_acc(x[c * NV]);
                    }
                }
                Y[(long)(bi * 16 + r) * NV + v] = s;
            }
        }
    }
    region_cycles = naiad_cycles() - t0;
    return 0;
}
