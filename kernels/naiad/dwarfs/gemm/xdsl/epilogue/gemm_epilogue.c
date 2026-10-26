/*
 * gemm with an element-wise epilogue, xDSL variants: D = f(A . B, Y), fp32.
 *
 *   XDSL_EPI_RSQRT 1   D = (A . B) * rsqrt(Y)        (the nbody weight's shape)
 *   XDSL_EPI_RSQRT 0   D = max(0.5 (A . B) + Y, 0)   (scale, bias, ReLU)
 *
 * XDSL_FUSED 1: one kernel compiled by xDSL from a linalg.matmul followed by the
 * epilogue as a linalg.generic; convert-linalg-to-riscv-ame fuses the generic into the
 * accumulators with the custom-3 element-wise instructions (aew.ax / aew.at / aew.aa,
 * aewu rsqrt), so D is stored once and C never goes to memory unscaled. Needs a design
 * with the ewise and ewise_tx features (run_epilogue.py writes one).
 * XDSL_FUSED 0: the same xDSL matmul without the epilogue (gemm/xdsl's kernel), then
 * the epilogue as a C loop on the scalar core, the way it runs without the fusion.
 *
 * XDSL_ASM names the kernel's .S (relative to this directory; run_epilogue.py sets it).
 * The timed region is the call, plus the scalar loop (unfused), plus the drain.
 *
 * Host interface: A (M x K), B (K x N), Y (M x N, fp32), C (M x N, fp32: D), region_cycles.
 */
#include <stdint.h>

#include "dwarf_util.h"
#include "naiad_platform.h"

#ifndef XDSL_ASM
#error "define XDSL_ASM (the kernel's .S path; run_epilogue.py does)"
#endif
#define XDSL_STR_(x) #x
#define XDSL_STR(x) XDSL_STR_(x)

__asm__(".include \"" XDSL_STR(XDSL_ASM) ".S\"");

NAIAD_IO float A[GEMM_M * GEMM_K];
NAIAD_IO float B[GEMM_K * GEMM_N];
NAIAD_IO float Y[GEMM_M * GEMM_N];
NAIAD_IO float C[GEMM_M * GEMM_N];
NAIAD_IO uint64_t region_cycles;

#if XDSL_FUSED
void gemm_epilogue_xdsl(const float *a, const float *b, float *c, const float *y);
#else
void gemm_xdsl(const float *a, const float *b, float *c);

static inline float sqrt_f32(float x) {
    float r;
    __asm__("fsqrt.s %0, %1" : "=f"(r) : "f"(x));
    return r;
}
#endif

int main(void) {
    uint64_t t0 = naiad_cycles();
#if XDSL_FUSED
    gemm_epilogue_xdsl(A, B, C, Y);
#else
    gemm_xdsl(A, B, C);
    for (int i = 0; i < GEMM_M * GEMM_N; i++) {
#if XDSL_EPI_RSQRT
        C[i] = C[i] * (1.0f / sqrt_f32(Y[i]));
#else
        float v = 0.5f * C[i] + Y[i];
        C[i] = v > 0.0f ? v : 0.0f;
#endif
    }
#endif
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - t0;
    return 0;
}
