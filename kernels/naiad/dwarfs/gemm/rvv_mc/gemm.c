/*
 * gemm, rvv_mc variant: the rvv variant with C's rows split across the harts
 * of a multi-hart design (each hart has its own vector unit). Rows go out in
 * whole RV_MR-row micro-kernel blocks, as evenly as they divide; every C
 * element is computed exactly as in the rvv variant, so results are
 * bit-identical. On one hart this is the rvv variant plus no-op barriers.
 *
 * Timed region (hart 0's counter): a barrier every hart has reached, to a
 * barrier after every hart's last store.
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
    const long h = (long)naiad_hartid();
    const long blocks = (M + RV_MR - 1) / RV_MR, per = blocks / NAIAD_HARTS, extra = blocks % NAIAD_HARTS;
    const long b0 = h * per + (h < extra ? h : extra), b1 = b0 + per + (h < extra);
    const long r0 = b0 * RV_MR, r1 = b1 * RV_MR < M ? b1 * RV_MR : M;
    naiad_barrier();
    uint64_t t0 = naiad_cycles();
    if (r1 > r0) {
        rv_mxm(r1 - r0, N, K, A + r0 * K, K, (const rv_in_bits_t *)B, N, (rv_out_t *)C + r0 * N, N);
    }
    naiad_barrier();
    if (h == 0) {
        region_cycles = naiad_cycles() - t0;
    }
    return 0;
}
