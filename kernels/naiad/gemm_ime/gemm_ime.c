/*
 * gemm_ime: C = A x B on the integrated matrix mode (IME, Zvvm), blocked over M, N and K.
 *
 * Two problems run: fp32 in / fp32 acc (vfmmacc.vv) and bf16 in / fp32 acc
 * (vfwmmacc.vv, W = 2). Both use SEW 32 at lambda 1, so a C tile is M x M with
 * M = VLEN / 32 in EMUL_C = M registers, and LMUL 4 gives K_eff = 4 (fp32) or 8
 * (bf16) per mma. Each output block is 1 x 2 C tiles (two C groups resident over
 * the K loop), so every A tile is used twice:
 *
 *   A tile: vmtl.v  from row-major A (LD = K in storage elements),
 *   B tile: vmttl.v from row-major B (fp32: B itself, LD = N), or from Bp, B with
 *           row pairs interleaved into 32-bit storage elements (bf16: built once
 *           with RVV vle16 + vsseg2e16, then the same transposing load).
 *
 * A, B and C are NAIAD_IO arrays the kernel fills itself (an LCG), so the host test
 * can read all three back and check C against numpy. In-kernel, every element is
 * also checked bit for bit against a scalar loop with the same rounding sequence
 * (the design's default accumulation disclosure, G = 1 / psm = 0 / rnd = frm: each
 * sub-dot-product rounded to fp32, then added), and the run prints PASS.
 */
#include <stdint.h>

#include "dwarf_util.h"
#include "ime.h"
#include "naiad_platform.h"

#ifndef GEMM_M
#define GEMM_M 32
#endif
#ifndef GEMM_N
#define GEMM_N 32
#endif
#ifndef GEMM_K
#define GEMM_K 32
#endif

#define TM (NAIAD_VLEN / 32) /* M = N_max at SEW 32, lambda 1 */
#define LMUL 4
#define KF (1 * LMUL)     /* fp32: K_eff = lambda * LMUL */
#define KB (1 * 2 * LMUL) /* bf16: K_eff = lambda * W * LMUL */

#if TM != 8
#error "gemm_ime is written for VLEN 256 (an 8 x 8 fp32 C tile in 8 registers)"
#endif
#if GEMM_M % TM || GEMM_N % (2 * TM) || GEMM_K % KB
#error "GEMM_M must be a multiple of 8, GEMM_N of 16 and GEMM_K of 8"
#endif

NAIAD_IO float Af[GEMM_M * GEMM_K], Bf[GEMM_K * GEMM_N], Cf[GEMM_M * GEMM_N];
NAIAD_IO uint16_t Ab[GEMM_M * GEMM_K], Bb[GEMM_K * GEMM_N];
NAIAD_IO float Cb[GEMM_M * GEMM_N];
NAIAD_IO uint64_t cycles_fp32, cycles_bf16;
static uint32_t Bp[GEMM_K / 2 * GEMM_N]; /* bf16 B with row pairs packed: row q = (B[2q][j], B[2q+1][j]) */

#define VT_K IME_VTYPE(32, LMUL, 1, 0, 0, 0)
#define VT_KB IME_VTYPE(32, LMUL, 1, 0, 1, 1) /* altfmt_A = altfmt_B = 1: bf16 inputs */
#define VT_C IME_VTYPE(32, 8, 1, 0, 0, 0)     /* LMUL = EMUL_C = 8 for C loads/stores */

static uint32_t lcg_state = 12345u;
static float lcg_float(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float)(int32_t)(lcg_state & 0xffffff00u) / 2147483648.0f; /* [-1, 1), 24 bits */
}

/* One blocked GEMM: rows i0.., two C tiles of columns j0.. and j0 + TM. */
#define IME_GEMM(ASRC, LDA, BSRC, LDB, CDST, KE, VTK, MMA)                                         \
    for (int i0 = 0; i0 < GEMM_M; i0 += TM) {                                                      \
        for (int j0 = 0; j0 < GEMM_N; j0 += 2 * TM) {                                              \
            vuint32m8_t c0, c1;                                                                    \
            ime_vzero(c0, VT_C);                                                                   \
            ime_vzero(c1, VT_C);                                                                   \
            ime_vsetvlmax(VTK);                                                                    \
            for (int k0 = 0; k0 < GEMM_K; k0 += (KE)) {                                            \
                vuint32m4_t a, b0, b1;                                                             \
                ime_vmtl(a, &(ASRC)[i0 * (LDA) + k0 * (LDA) / GEMM_K], LDA);                       \
                ime_vmttl(b0, &(BSRC)[k0 * (LDB) / ((KE) / KF) + j0], LDB);                        \
                ime_vmttl(b1, &(BSRC)[k0 * (LDB) / ((KE) / KF) + j0 + TM], LDB);                   \
                MMA(c0, a, b0);                                                                    \
                MMA(c1, a, b1);                                                                    \
            }                                                                                      \
            ime_vsetvlmax(VT_C);                                                                   \
            ime_vmts(c0, &(CDST)[i0 * GEMM_N + j0], GEMM_N);                                       \
            ime_vmts(c1, &(CDST)[i0 * GEMM_N + j0 + TM], GEMM_N);                                  \
        }                                                                                          \
    }

#pragma clang fp contract(off)

static int check(const char *what, const float *got, const float *want) {
    int bad = 0;
    for (int i = 0; i < GEMM_M * GEMM_N; i++) {
        if (naiad_f32_bits(got[i]) != naiad_f32_bits(want[i])) {
            if (bad < 4) {
                naiad_printf("%s: MISMATCH at %d: %08x vs %08x\n", what, i, (unsigned)naiad_f32_bits(got[i]),
                             (unsigned)naiad_f32_bits(want[i]));
            }
            bad++;
        }
    }
    naiad_printf("gemm_ime %s %dx%dx%d: %s\n", what, GEMM_M, GEMM_N, GEMM_K, bad ? "FAIL" : "ok");
    return bad;
}

static float ref[GEMM_M * GEMM_N];

int main(void) {
    for (int i = 0; i < GEMM_M * GEMM_K; i++) {
        Af[i] = lcg_float();
        Ab[i] = naiad_f32_to_bf16(lcg_float());
    }
    for (int i = 0; i < GEMM_K * GEMM_N; i++) {
        Bf[i] = lcg_float();
        Bb[i] = naiad_f32_to_bf16(lcg_float());
    }
    naiad_printf("gemm_ime: VLEN %d, C tile %dx%d, lambda %d after vsetvl\n", NAIAD_VLEN, TM, TM,
                 (ime_vsetvlmax(VT_K), ime_lambda()));

    /* fp32: A row-major (LD = K), B row-major through the transposing load (LD = N). */
    uint64_t t0 = naiad_cycles();
    IME_GEMM(Af, GEMM_K, Bf, GEMM_N, Cf, KF, VT_K, ime_vfmmacc);
    cycles_fp32 = naiad_cycles() - t0;

    /* bf16: pair B's rows into 32-bit storage elements with RVV (unit-stride loads, a 2-field
     * segment store), then the same transposing tile load with LD = N; A's K-adjacent pairs are
     * already packed (LD = K / 2 storage elements). */
    t0 = naiad_cycles();
    for (int q = 0; q < GEMM_K / 2; q++) {
        for (int j = 0; j < GEMM_N;) {
            size_t vl = __riscv_vsetvl_e16m2(GEMM_N - j);
            vuint16m2_t lo = __riscv_vle16_v_u16m2(&Bb[(2 * q) * GEMM_N + j], vl);
            vuint16m2_t hi = __riscv_vle16_v_u16m2(&Bb[(2 * q + 1) * GEMM_N + j], vl);
            __riscv_vsseg2e16_v_u16m2x2((uint16_t *)&Bp[q * GEMM_N + j], __riscv_vcreate_v_u16m2x2(lo, hi), vl);
            j += vl;
        }
    }
    IME_GEMM(((uint32_t *)Ab), (GEMM_K / 2), Bp, GEMM_N, Cb, KB, VT_KB, ime_vfwmmacc);
    cycles_bf16 = naiad_cycles() - t0;
    naiad_printf("cycles: fp32 %lu, bf16 %lu (%d MACs each)\n", (unsigned long)cycles_fp32,
                 (unsigned long)cycles_bf16, GEMM_M * GEMM_N * GEMM_K);

    /* Scalar references with the same rounding sequence. */
    for (int i = 0; i < GEMM_M; i++) {
        for (int j = 0; j < GEMM_N; j++) {
            float c = 0.0f;
            for (int k = 0; k < GEMM_K; k++) {
                float p = Af[i * GEMM_K + k] * Bf[k * GEMM_N + j];
                c = c + p;
            }
            ref[i * GEMM_N + j] = c;
        }
    }
    int bad = check("fp32", Cf, ref);
    for (int i = 0; i < GEMM_M; i++) {
        for (int j = 0; j < GEMM_N; j++) {
            float c = 0.0f;
            for (int k = 0; k < GEMM_K; k += 2) {
                double p0 = (double)naiad_bf16_to_f32(Ab[i * GEMM_K + k]) * naiad_bf16_to_f32(Bb[k * GEMM_N + j]);
                double p1 =
                    (double)naiad_bf16_to_f32(Ab[i * GEMM_K + k + 1]) * naiad_bf16_to_f32(Bb[(k + 1) * GEMM_N + j]);
                c = c + (float)(p0 + p1);
            }
            ref[i * GEMM_N + j] = c;
        }
    }
    bad += check("bf16", Cb, ref);
    naiad_puts(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
}
