/*
 * gemm, rvv variant: C = A . B with the register-blocked RVV outer-product
 * micro-kernel of rvv_mxm.h (a 5 x VL accumulator block at LMUL 4 for
 * fp32/fp64, 6 x VL for 16-bit inputs; one B-row vector load and one A scalar
 * per vfmacc.vf, every load one step ahead).
 *
 * Arithmetic is in the accumulator format: fp64 for fp64, fp32 otherwise.
 * 16-bit inputs are read in place (no widening pass: vfwmacc.vf /
 * vfwmaccbf16.vf multiply them exactly and round each sum once in fp32), and
 * a bf16 accumulator format is the fp32 result rounded once on store, as the
 * scalar variant does (RVV has no bf16 arithmetic to accumulate in).
 */
#include <stdint.h>

#include "dwarf_util.h"
#include "naiad_platform.h"

#if defined(GEMM_IN_FP64)
#define DW_IN_FMT NAIAD_MFMT_FP64
#elif defined(GEMM_IN_FP32)
#define DW_IN_FMT NAIAD_MFMT_FP32
#elif defined(GEMM_IN_FP16)
#define DW_IN_FMT NAIAD_MFMT_FP16
#elif defined(GEMM_IN_BF16)
#define DW_IN_FMT NAIAD_MFMT_BF16
#else
#error "gemm rvv: unsupported input format"
#endif
#if defined(GEMM_ACC_FP64)
#define DW_ACC_FMT NAIAD_MFMT_FP64
#elif defined(GEMM_ACC_FP32)
#define DW_ACC_FMT NAIAD_MFMT_FP32
#elif defined(GEMM_ACC_BF16)
#define DW_ACC_FMT NAIAD_MFMT_FP32
#define RV_OUT_BF16 1
#else
#error "gemm rvv: unsupported accumulator format"
#endif

#define RV_AL 4
#include "rvv_fmt.h"
#include "rvv_mxm.h"

#define M GEMM_M
#define N GEMM_N
#define K GEMM_K

typedef NAIAD_T(GEMM_IN) in_t;
typedef NAIAD_T(GEMM_ACC) acc_t;

_Static_assert(sizeof(in_t) == sizeof(rv_in_bits_t), "input storage width");
_Static_assert(sizeof(acc_t) == sizeof(rv_out_t), "output storage width");

NAIAD_IO in_t A[M * K];
NAIAD_IO in_t B[K * N];
NAIAD_IO acc_t C[M * N];
NAIAD_IO uint64_t region_cycles;

RV_KERNEL int main(void) {
    uint64_t t0 = naiad_cycles();
    rv_mxm(M, N, K, A, K, (const rv_in_bits_t *)B, N, (rv_out_t *)C, N);
    region_cycles = naiad_cycles() - t0;
    return 0;
}
