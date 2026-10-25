/*
 * gemm, Tensix variant: C = A . B offloaded to a Blackhole Tensix tile on the
 * bus (naiad-tensix.yaml; framework/naiad/ip/tensix.py), driven by MMIO and DMA.
 *
 * The tile runs a captured tt-metal compute kernel (kernels/naiad/tensix/
 * capture/src/kernels/compute/gemm.cpp: a compute-only matmul_tiles loop over
 * L1-resident CBs) from the image header the capture produced:
 * tensix_gemm_fp32.h (bf16 in, fp32 Dst and C) or tensix_gemm_bf16.h (bf16 in,
 * 16-bit Dst, bf16 C). Both HiFi2.
 *
 * Outside the timed region: the one-time bring-up (firmware + init handshake,
 * tensix_boot), recorded as boot_cycles. The timed region is the whole offload
 * round trip: the launch image (kernels, CB config, launch message) DMA'd in,
 * the runtime args (Mt, Kt, Nt), A and B tilized into L1 by 2D DMA (one
 * transfer per 16x16 face), the doorbell, the wait for DONE, and C untilized
 * back by DMA. tensix_cycles is the tile's own BUSY count for the launch.
 *
 * M, N and K must be multiples of 32 (whole tiles), and each operand at most
 * the image's 64-tile CB.
 */
#include <stdint.h>

#include "naiad_platform.h"

#if defined(GEMM_ACC_FP32)
#include "tensix_gemm_fp32.h"
#define IMG tensix_gemm_fp32
#define CB_A TENSIX_GEMM_FP32_CB0_ADDR
#define CB_B TENSIX_GEMM_FP32_CB1_ADDR
#define CB_C TENSIX_GEMM_FP32_CB2_ADDR
#define CB_PAGES TENSIX_GEMM_FP32_CB0_PAGES
typedef float c_t;
#elif defined(GEMM_ACC_BF16)
#include "tensix_gemm_bf16.h"
#define IMG tensix_gemm_bf16
#define CB_A TENSIX_GEMM_BF16_CB0_ADDR
#define CB_B TENSIX_GEMM_BF16_CB1_ADDR
#define CB_C TENSIX_GEMM_BF16_CB2_ADDR
#define CB_PAGES TENSIX_GEMM_BF16_CB0_PAGES
typedef uint16_t c_t;
#else
#error "the tensix gemm takes bf16 in and an fp32 or bf16 C (-DGEMM_ACC_FP32 / -DGEMM_ACC_BF16)"
#endif

#if !defined(GEMM_IN_BF16)
#error "the tensix gemm takes bf16 inputs"
#endif
#if GEMM_M % 32 || GEMM_N % 32 || GEMM_K % 32
#error "the tensix gemm needs M, N and K to be multiples of 32"
#endif
#define MT (GEMM_M / 32)
#define NT (GEMM_N / 32)
#define KT (GEMM_K / 32)
#if MT * KT > CB_PAGES || KT * NT > CB_PAGES || MT * NT > CB_PAGES
#error "an operand does not fit the image's CBs"
#endif

NAIAD_IO uint16_t A[GEMM_M * GEMM_K];
NAIAD_IO uint16_t B[GEMM_K * GEMM_N];
NAIAD_IO c_t C[GEMM_M * GEMM_N];
NAIAD_IO uint64_t region_cycles;
NAIAD_IO uint64_t boot_cycles;
NAIAD_IO uint64_t tensix_cycles;
/* Region split: launch image + args, operands in, doorbell to DONE seen, result out. */
NAIAD_IO uint64_t phase_cycles[4];

int main(void) {
    uint64_t t0 = naiad_cycles();
    tensix_check(tensix_boot(&IMG), "boot");
    uint64_t t1 = naiad_cycles();
    boot_cycles = t1 - t0;

    uint64_t busy0 = *tensix_reg(NAIAD_TENSIX_REG_BUSY_CYCLES);
    const uint32_t args[3] = {MT, KT, NT};
    int rc = tensix_load_launch(&IMG);
    tensix_set_args(&IMG, args, 3);
    uint64_t t2 = naiad_cycles();
    rc |= tensix_dma_tilize(A, GEMM_K, 2, CB_A, MT, KT);
    rc |= tensix_dma_tilize(B, GEMM_N, 2, CB_B, KT, NT);
    rc |= tensix_dma_wait_all();
    tensix_check(rc, "operands in");
    uint64_t t3 = naiad_cycles();
    tensix_launch();
    tensix_wait();
    uint64_t t4 = naiad_cycles();
    rc = tensix_dma_untilize(CB_C, C, GEMM_N, sizeof(c_t), MT, NT);
    rc |= tensix_dma_wait_all();
    tensix_check(rc, "result out");
    uint64_t t5 = naiad_cycles();
    phase_cycles[0] = t2 - t1;
    phase_cycles[1] = t3 - t2;
    phase_cycles[2] = t4 - t3;
    phase_cycles[3] = t5 - t4;
    region_cycles = t5 - t1;
    tensix_cycles = *tensix_reg(NAIAD_TENSIX_REG_BUSY_CYCLES) - busy0;
    return 0;
}
