/*
 * ztt_gemm: Ztt v0.6 (the RISC-V AME draft) on a matrix: {mode: ztt} design, e.g. naiad-ztt06.
 *
 * Probes the unit (amenlen, ameudsz, an unsupported datatype reporting amestatus.UN), then runs
 * two blocked GEMMs with the draft's own instructions and checks them against a scalar
 * reference:
 *   int8 x int8 -> int32   C[M x P] = A[M x K] x B[K x P], every square loaded with mls.st
 *   bf16 x bf16 -> fp32    the same shape, B given transposed and used through mmulbtacc.2d
 * The bf16 inputs are small integers, so every product and sum is exact in fp32 and the
 * reference needs no rounding model. Returns the number of failed checks.
 */
#include "naiad_platform.h"
#include "ztt.h"

#define ZN NAIAD_ZTT_N
#define GM (2 * ZN)
#define GK (4 * ZN)
#define GP (3 * ZN)

static int8_t a8[GM][GK], b8[GK][GP];
static int32_t c32[GM][GP];
static uint16_t a16[GM][GK], bt16[GP][GK]; /* bf16 bit patterns; B transposed */
static float cf[GM][GP];
static unsigned bad;

static void check(const char *what, long long got, long long want) {
    naiad_printf("  %s: %lld (expect %lld) %s\n", what, got, want, got == want ? "ok" : "MISMATCH");
    bad += got != want;
}

static uint16_t bf16_of_int(int v) { /* exact for |v| < 256 */
    float f = (float)v;
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

static void gemm_int8(void) {
    ztt_msettyp(0, NAIAD_ZTT_DT_INT8);
    ztt_msettyp(1, NAIAD_ZTT_DT_INT8);
    ztt_msettyp(4, NAIAD_ZTT_DT_INT32); /* m4-m7: one int32 square */
    ztt_asettyp(0, NAIAD_ZTT_DT_INT32);
    for (int i = 0; i < GM; i += ZN) {
        for (int j = 0; j < GP; j += ZN) {
            ztt_mzero_acc(0);
            for (int k = 0; k < GK; k += ZN) {
                ztt_mls_st(0, &a8[i][k], GK);
                ztt_mls_st(1, &b8[k][j], GP);
                ztt_mma(MMULACC_2D, 0, 0, 1);
            }
            ztt_r2(MMOV_M_A, 4, 0);
            ztt_mss_st(4, &c32[i][j], GP * sizeof(int32_t));
        }
    }
}

static void gemm_bf16(void) {
    ztt_msettyp(0, NAIAD_ZTT_DT_BF16); /* m0-m1 */
    ztt_msettyp(2, NAIAD_ZTT_DT_BF16); /* m2-m3 */
    ztt_msettyp(4, NAIAD_ZTT_DT_FP32); /* m4-m7 */
    ztt_asettyp(1, NAIAD_ZTT_DT_FP32);
    for (int i = 0; i < GM; i += ZN) {
        for (int j = 0; j < GP; j += ZN) {
            ztt_mzero_acc(1);
            for (int k = 0; k < GK; k += ZN) {
                ztt_mls_st(0, &a16[i][k], GK * sizeof(uint16_t));
                ztt_mls_st(2, &bt16[j][k], GK * sizeof(uint16_t)); /* rows of B^T */
                ztt_mma(MMULBTACC_2D, 1, 0, 2);                       /* C += A x (B^T)^T */
            }
            ztt_r2(MMOV_M_A, 4, 1);
            ztt_mss_st(4, &cf[i][j], GP * sizeof(float));
        }
    }
}

int main(void) {
    unsigned long got = ztt_enable();
    naiad_printf("ztt_gemm: Ztt v0.6, N = %d, %d M registers, %d accumulators\n", ZN, NAIAD_ZTT_M_REGS,
                 NAIAD_ZTT_ACC_REGS);
    check("ame.acquire (SUCCESS | STATUS << 1)", (long long)got, 1);
    check("amenlen", (long long)ztt_csrr(AMENLEN), ZN);
    check("ameudsz", (long long)ztt_csrr(AMEUDSZ), NAIAD_ZTT_UNIT_BITS);
    check("ameown", (long long)ztt_csrr(AMEOWN), 1);
    ztt_csrw(AMESTATUS, 0);
    ztt_msettyp(8, NAIAD_ZTT_DT_INT4); /* packed: not supported here */
    check("msettyp int4 -> amestatus.UN", (long long)ZTT_UN(), 1);
    ztt_csrw(AMESTATUS, 0);

    for (int i = 0; i < GM; i++)
        for (int k = 0; k < GK; k++) {
            a8[i][k] = (int8_t)((i * 7 + k * 3) % 23 - 11);
            a16[i][k] = bf16_of_int((i + 2 * k) % 9 - 4);
        }
    for (int k = 0; k < GK; k++)
        for (int j = 0; j < GP; j++) {
            b8[k][j] = (int8_t)((k * 5 + j * 11) % 29 - 14);
            bt16[j][k] = bf16_of_int((3 * k + j) % 7 - 3);
        }

    gemm_int8();
    unsigned wrong = 0;
    for (int i = 0; i < GM; i++)
        for (int j = 0; j < GP; j++) {
            int32_t want = 0;
            for (int k = 0; k < GK; k++) want += (int32_t)a8[i][k] * b8[k][j];
            wrong += c32[i][j] != want;
        }
    check("int8 GEMM: wrong elements", wrong, 0);

    gemm_bf16();
    wrong = 0;
    for (int i = 0; i < GM; i++)
        for (int j = 0; j < GP; j++) {
            int want = 0;
            for (int k = 0; k < GK; k++) want += ((i + 2 * k) % 9 - 4) * ((3 * k + j) % 7 - 3);
            wrong += cf[i][j] != (float)want;
        }
    check("bf16 GEMM: wrong elements", wrong, 0);
    check("amestatus after the GEMMs", (long long)ztt_csrr(AMESTATUS), 0);
    ztt_release();
    naiad_printf(bad ? "ZTT_GEMM FAIL\n" : "ZTT_GEMM OK\n");
    return (int)bad;
}
