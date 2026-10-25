/*
 * gemm, VME variant: C = A . B through the emitted Zvt driver (naiad_gemm.h,
 * framework/naiad/emit/gemm_vme.py) on the vector-coupled matrix unit, straight on the I/O
 * arrays. Needs a VME design (naiad-vme.yaml, naiad-vme-timed.yaml).
 *
 * The driver transposes each block row of A into a panel through the tile state, reads B in
 * place, and keeps a BM x BN block of accumulator tiles over the K loop; the block shape is
 * the emitter's choice for the design (see the header's per-pair comment).
 * bf16 -> bf16: Zvt has no bf16 accumulator, so the fp32 result goes to a scratch and one RVV
 * pass rounds it (vfncvtbf16), as the rvv variant does.
 *
 * Built with -DGEMM_M/N/K and -DGEMM_IN/GEMM_ACC (format names) by the dwarf harness
 * (kernels/naiad/dwarfs/gemm/spec.py).
 */
#include <stddef.h>
#include <stdint.h>

#include "dwarf_util.h"
#include "naiad_platform.h"

#ifndef NAIAD_MATRIX_VME
#error "the gemm vme variant needs a VME design (matrix: {mode: vme}), e.g. naiad-vme-timed.yaml"
#endif

#if defined(GEMM_IN_BF16) && defined(GEMM_ACC_BF16)
#define OUT_BF16 1
#define NAIAD_GEMM_IN bf16
#define NAIAD_GEMM_ACC fp32
#else
#define NAIAD_GEMM_IN GEMM_IN
#define NAIAD_GEMM_ACC GEMM_ACC
#endif
#define NAIAD_GEMM_MAX_K GEMM_K /* size the driver's static panels to this problem */
#define NAIAD_GEMM_MAX_N GEMM_N
#include "naiad_gemm.h"

NAIAD_IO naiad_gemm_in_t A[GEMM_M * GEMM_K];
NAIAD_IO naiad_gemm_in_t B[GEMM_K * GEMM_N];
#ifdef OUT_BF16
NAIAD_IO uint16_t C[GEMM_M * GEMM_N];
static float Cf[GEMM_M * GEMM_N] __attribute__((aligned(64)));
#define COUT Cf
#else
NAIAD_IO naiad_gemm_acc_t C[GEMM_M * GEMM_N];
#define COUT C
#endif
NAIAD_IO uint64_t region_cycles;

#ifdef OUT_BF16
#include <riscv_vector.h>

static void round_out(void) {
    /* Two independent chains a step, so the in-order vector unit overlaps load, convert and store. */
    size_t total = (size_t)GEMM_M * GEMM_N, i = 0;
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    for (; i + 2 * vlmax <= total; i += 2 * vlmax) {
        vfloat32m8_t x0 = __riscv_vle32_v_f32m8(&Cf[i], vlmax);
        vfloat32m8_t x1 = __riscv_vle32_v_f32m8(&Cf[i + vlmax], vlmax);
        vbfloat16m4_t y0 = __riscv_vfncvtbf16_f_f_w_bf16m4(x0, vlmax);
        vbfloat16m4_t y1 = __riscv_vfncvtbf16_f_f_w_bf16m4(x1, vlmax);
        __riscv_vse16_v_bf16m4((__bf16 *)&C[i], y0, vlmax);
        __riscv_vse16_v_bf16m4((__bf16 *)&C[i + vlmax], y1, vlmax);
    }
    while (i < total) {
        size_t vl = __riscv_vsetvl_e32m8(total - i);
        vbfloat16m4_t y = __riscv_vfncvtbf16_f_f_w_bf16m4(__riscv_vle32_v_f32m8(&Cf[i], vl), vl);
        __riscv_vse16_v_bf16m4((__bf16 *)&C[i], y, vl);
        i += vl;
    }
}
#endif

int main(void) {
    uint64_t t0 = naiad_cycles();
    naiad_gemm(GEMM_M, GEMM_N, GEMM_K, A, GEMM_K, B, GEMM_N, COUT, GEMM_N);
#ifdef OUT_BF16
    round_out();
#endif
    region_cycles = naiad_cycles() - t0;
    return 0;
}
