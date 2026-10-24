/*
 * gemm, IME variant: C = A . B through the emitted blocked driver (naiad_gemm.h) on an
 * integrated matrix unit (Zvvm; framework/naiad/emit/gemm_ime.py), straight on the I/O
 * arrays. Needs an IME design (naiad-ime.yaml, naiad-ime-timed.yaml).
 *
 * Built with -DGEMM_M/N/K and -DGEMM_IN/GEMM_ACC (format names) by the dwarf harness
 * (kernels/naiad/dwarfs/gemm/spec.py). The timed region is the driver call: its last
 * C tile stores are vector stores, posted like any other.
 */
#include <stdint.h>

#include "dwarf_util.h"
#include "naiad_platform.h"

#ifndef NAIAD_MATRIX_IME
#error "the gemm ime variant needs an IME design (matrix: {mode: ime}), e.g. naiad-ime-timed.yaml"
#endif

#define NAIAD_GEMM_IN GEMM_IN
#define NAIAD_GEMM_ACC GEMM_ACC
#include "naiad_gemm.h"

NAIAD_IO naiad_gemm_in_t A[GEMM_M * GEMM_K];
NAIAD_IO naiad_gemm_in_t B[GEMM_K * GEMM_N];
NAIAD_IO naiad_gemm_acc_t C[GEMM_M * GEMM_N];
NAIAD_IO uint64_t region_cycles;

int main(void) {
    uint64_t t0 = naiad_cycles();
    naiad_gemm(GEMM_M, GEMM_N, GEMM_K, A, GEMM_K, B, GEMM_N, C, GEMM_N);
    region_cycles = naiad_cycles() - t0;
    return 0;
}
