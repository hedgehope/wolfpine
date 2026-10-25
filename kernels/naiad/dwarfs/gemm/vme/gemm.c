/*
 * gemm, VME variant: C = A . B on the vector-coupled matrix unit (Zvt v0.4; kernels/naiad/include/vme.h).
 * Needs a VME design (naiad-vme.yaml, naiad-vme-timed.yaml).
 *
 * Zvt multiplies are outer products of K-rows: C[tm, tn] += At[tk, tm]^T . B[tk, tn] with
 * tk <= KMAX (1 for fp32/fp64, 2 for fp16/bf16), so A is needed by columns. The timed region:
 *   1. A^T into a padded scratch At[K][MP] through the tile state (the draft's transpose: a
 *      tile is filled by rows with vtle, drained by columns with vtse, at TEW = the input width;
 *      rows past M are zeroed with vtzero), MP = M rounded up to the 2 x 2 tile block.
 *   2. B is used in place when N is a multiple of the block (else copied into a padded Bp).
 *   3. For each 2E x 2E C block (E = the accumulator tile edge, TE or TE/2 at TEW 64): zero
 *      four accumulator tiles, then per K step load two At rows-groups and two B row-groups
 *      (plain vleN.v, KMAX rows each) and issue four multiplies; the next step's loads are
 *      interleaved with the current step's multiplies (two operand buffers). The C tiles are
 *      then stored row by row with vtse (tm, tn clipped to the matrix edge).
 *   bf16 -> bf16: the fp32 tiles go to a scratch, then one RVV pass rounds them (vfncvtbf16),
 *   since Zvt has no bf16 accumulator (as the rvv variant does).
 *
 * Built with -DGEMM_M/N/K and -DGEMM_IN/GEMM_ACC (format names) by the dwarf harness.
 */
#include <stddef.h>
#include <stdint.h>

#include "dwarf_util.h"
#include "naiad_platform.h"

#ifndef NAIAD_MATRIX_VME
#error "the gemm vme variant needs a VME design (matrix: {mode: vme}), e.g. naiad-vme-timed.yaml"
#endif
#include "vme.h"

#define M GEMM_M
#define N GEMM_N
#define K GEMM_K

#if defined(GEMM_IN_FP32) && defined(GEMM_ACC_FP32)
#define SEW 32
#define TW 1
#define KMAX 1
#define MMA vme_vtfmm
typedef float in_t;
typedef float acc_t;
#elif defined(GEMM_IN_FP64) && defined(GEMM_ACC_FP64)
#define SEW 64
#define TW 1
#define KMAX 1
#define MMA vme_vtfmm
typedef double in_t;
typedef double acc_t;
#elif defined(GEMM_IN_FP16) && defined(GEMM_ACC_FP32)
#define SEW 16
#define TW 2
#define KMAX 2
#define MMA vme_vtfmm
typedef uint16_t in_t;
typedef float acc_t;
#elif defined(GEMM_IN_BF16) && (defined(GEMM_ACC_FP32) || defined(GEMM_ACC_BF16))
#define SEW 16
#define TW 2
#define KMAX 2
#define ALT 1
#define MMA vme_vtfmm_alt
typedef uint16_t in_t;
#if defined(GEMM_ACC_BF16)
#define OUT_BF16 1
typedef uint16_t acc_t;
#else
typedef float acc_t;
#endif
#else
#error "gemm vme: unsupported format pair (Zvt has fp32, fp64, fp16->fp32, bf16->fp32; bf16->bf16 via fp32)"
#endif
#ifndef ALT
#define ALT 0
#endif

#define TEW (SEW * TW) /* the accumulator tiles' element width */
#define E (TEW < 64 ? NAIAD_VME_TE : NAIAD_VME_TE / 2)
#define BLK (2 * E)
#define RUP(x) (((x) + BLK - 1) / BLK * BLK)
#define MP RUP(M)
#define NP RUP(N)
/* LMUL msetmtype derives: min(8/KMAX, 8/TWIDEN, ceil(ETE / (VLEN/SEW))); the register map
 * below assumes it is at most 4 (KMAX 1) or 2 (KMAX 2). */
_Static_assert((E * SEW + NAIAD_VLEN - 1) / NAIAD_VLEN <= 4 / KMAX, "operand rows do not fit the slot map");

/* Accumulator tiles (TEW 32: mt0/4/8/12; TEW 64: mt0/2/4/6) and the transpose tile. */
#define T00 0
#define T01 (TEW == 64 ? 2 : 4)
#define T10 (2 * T01)
#define T11 (3 * T01)

/* Operand slots: KMAX 1 -> v0, v4, ..., v28 (one LMUL-4 row); KMAX 2 -> rows at v[s], v[s+4]. */
#if KMAX == 1
#define XA0 0
#define XA1 4
#define XB0 8
#define XB1 12
#define YA0 16
#define YA1 20
#define YB0 24
#define YB1 28
#define LOADOP(r0, p, ld) vme_vle(SEW, r0, (p))
#else
#define XA0 0
#define XA1 2
#define XB0 8
#define XB1 10
#define YA0 16
#define YA1 18
#define YB0 24
#define YB1 26
#define VME_PLUS4_(r) VME_PLUS4_##r
#define VME_PLUS4_0 4
#define VME_PLUS4_2 6
#define VME_PLUS4_8 12
#define VME_PLUS4_10 14
#define VME_PLUS4_16 20
#define VME_PLUS4_18 22
#define VME_PLUS4_24 28
#define VME_PLUS4_26 30
#define LOADOP(r0, p, ld)                                                                          \
    do {                                                                                           \
        vme_vle(SEW, r0, (p));                                                                     \
        vme_vle(SEW, VME_PLUS4_(r0), (p) + (ld));                                                  \
    } while (0)
#endif

NAIAD_IO in_t A[M * K];
NAIAD_IO in_t B[K * N];
NAIAD_IO acc_t C[M * N];
NAIAD_IO uint64_t region_cycles;

static in_t At[K * MP] __attribute__((aligned(64)));
#if NP != N
static in_t Bp[K * NP] __attribute__((aligned(64)));
#endif
#ifdef OUT_BF16
static float Cf[M * N] __attribute__((aligned(64)));
#define COUT Cf
#else
#define COUT C
#endif

/* Step 1: At = A^T, zero-padded to MP columns, through tile mt0 at TEW = SEW. */
static void transpose_a(void) {
    vme_msetmtypei(SEW, 0, 1);
    for (int i0 = 0; i0 < MP; i0 += E) {
        for (int k0 = 0; k0 < K; k0 += E) {
            int kc = K - k0 < E ? K - k0 : E;
            vme_msettm(E);
            vme_msettn(E);
            vme_vtzero(0);
            vme_msettn(kc);
#pragma clang loop unroll_count(8)
            for (int r = 0; r < E && i0 + r < M; r++)
                vme_vtle(SEW, &A[(size_t)(i0 + r) * K + k0], VME_TSS(0, 0, r));
            vme_msettn(E);
#pragma clang loop unroll_count(8)
            for (int c = 0; c < kc; c++)
                vme_vtse(SEW, &At[(size_t)(k0 + c) * MP + i0], VME_TSS(0, 1, c));
        }
    }
}

/* One K step's operands: A rows from At (stride MP), B rows from Bs (stride LDB). */
#define LOAD_STEP(A0, A1, B0, B1, k)                                                               \
    do {                                                                                           \
        const in_t *pa_ = &At[(size_t)(k) * MP + i0];                                              \
        const in_t *pb_ = &Bs[(size_t)(k) * LDB + j0];                                             \
        LOADOP(A0, pa_, MP);                                                                       \
        LOADOP(B0, pb_, LDB);                                                                      \
        LOADOP(B1, pb_ + E, LDB);                                                                  \
        LOADOP(A1, pa_ + E, MP);                                                                   \
    } while (0)

/* Four multiplies on buffer (A0, A1, B0, B1), interleaved with the next step's loads into (NA0..). */
#define MMA_LOAD(A0, A1, B0, B1, NA0, NA1, NB0, NB1, kn)                                           \
    do {                                                                                           \
        const in_t *pa_ = &At[(size_t)(kn) * MP + i0];                                             \
        const in_t *pb_ = &Bs[(size_t)(kn) * LDB + j0];                                            \
        MMA(T00, A0, B0);                                                                          \
        LOADOP(NA0, pa_, MP);                                                                      \
        MMA(T01, A0, B1);                                                                          \
        LOADOP(NB0, pb_, LDB);                                                                     \
        MMA(T10, A1, B0);                                                                          \
        LOADOP(NB1, pb_ + E, LDB);                                                                 \
        MMA(T11, A1, B1);                                                                          \
        LOADOP(NA1, pa_ + E, MP);                                                                  \
    } while (0)

#define MMA_ONLY(A0, A1, B0, B1)                                                                   \
    do {                                                                                           \
        MMA(T00, A0, B0);                                                                          \
        MMA(T01, A0, B1);                                                                          \
        MMA(T10, A1, B0);                                                                          \
        MMA(T11, A1, B1);                                                                          \
    } while (0)

static void store_tile(int tile, int i, int j) {
    int rows = M - i < E ? M - i : E;
    int cols = N - j < E ? N - j : E;
    if (rows <= 0 || cols <= 0)
        return;
    vme_msettn(cols);
#pragma clang loop unroll_count(8)
    for (int r = 0; r < rows; r++)
        vme_vtse(TEW, &COUT[(size_t)(i + r) * N + j], VME_TSS(tile, 0, r));
}

static void zero_tiles(void) {
    /* vtzero is at TEW = SEW, so the accumulators are cleared under an e<TEW> configuration. */
    vme_msetmtypei(TEW, 0, 1);
    vme_msettm(E);
    vme_msettn(E);
    vme_vtzero(T00);
    vme_vtzero(T01);
    vme_vtzero(T10);
    vme_vtzero(T11);
    vme_msetmtypei(SEW, ALT, TW);
    vme_msettm(E);
    vme_msettk(KMAX);
    vme_msettn(E);
}

static void gemm(void) {
#if NP != N
#pragma clang loop vectorize(disable) interleave(disable)
    for (int k = 0; k < K; k++)
        for (int j = 0; j < NP; j++)
            Bp[(size_t)k * NP + j] = j < N ? B[(size_t)k * N + j] : 0;
    const in_t *Bs = Bp;
#define LDB NP
#else
    const in_t *Bs = B;
#define LDB N
#endif
    transpose_a();
    const int FULL = K / KMAX;
    for (int i0 = 0; i0 < MP; i0 += BLK) {
        for (int j0 = 0; j0 < NP; j0 += BLK) {
            zero_tiles();
            if (FULL > 0) {
                LOAD_STEP(XA0, XA1, XB0, XB1, 0);
                int s = 0;
                for (; s + 2 < FULL; s += 2) {
                    MMA_LOAD(XA0, XA1, XB0, XB1, YA0, YA1, YB0, YB1, (s + 1) * KMAX);
                    MMA_LOAD(YA0, YA1, YB0, YB1, XA0, XA1, XB0, XB1, (s + 2) * KMAX);
                }
                if (s + 1 < FULL) {
                    MMA_LOAD(XA0, XA1, XB0, XB1, YA0, YA1, YB0, YB1, (s + 1) * KMAX);
                    MMA_ONLY(YA0, YA1, YB0, YB1);
                } else {
                    MMA_ONLY(XA0, XA1, XB0, XB1);
                }
            }
#if K % KMAX
            /* The last K row on its own (tk = 1): only row 0 of each operand is read. */
            vme_msettk(K % KMAX);
            {
                const in_t *pa_ = &At[(size_t)(K - 1) * MP + i0];
                const in_t *pb_ = &Bs[(size_t)(K - 1) * LDB + j0];
                vme_vle(SEW, XA0, pa_);
                vme_vle(SEW, XA1, pa_ + E);
                vme_vle(SEW, XB0, pb_);
                vme_vle(SEW, XB1, pb_ + E);
            }
            MMA_ONLY(XA0, XA1, XB0, XB1);
#endif
            store_tile(T00, i0, j0);
            store_tile(T01, i0, j0 + E);
            store_tile(T10, i0 + E, j0);
            store_tile(T11, i0 + E, j0 + E);
        }
    }
    vme_vtdiscard();
}

#ifdef OUT_BF16
#include <riscv_vector.h>

static void round_out(void) {
    /* Two independent chains a step, so the in-order vector unit overlaps load, convert and store. */
    size_t total = (size_t)M * N, i = 0;
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
    gemm();
#ifdef OUT_BF16
    round_out();
#endif
    region_cycles = naiad_cycles() - t0;
    return 0;
}
