/*
 * gemm_ame: C = A x B on the attached matrix unit, blocked over M, N and K.
 *
 * Each step of the outer loops owns a 2 x 2 block of output tiles in four
 * accumulators; each K step loads two "a" tiles and two "b" tiles and issues
 * four mma, so every loaded tile is used twice. The same problem runs twice:
 * fp32 in / fp32 acc (integer-valued data, so exact), and int8 in / int32 acc.
 * The checks: the unit's configuration matches the design's naiad.h, the two
 * results agree element for element, C's hash matches the host-precomputed
 * one (gemm_data.h), the diagonal matches a scalar recomputation, and the
 * perf counters saw the expected number of mma.
 */
#include "gemm_data.h"
#include "matrix.h"
#include "naiad_platform.h"

#define GM GEMM_M
#define GN GEMM_N
#define GK GEMM_K
#define TM NAIAD_MATRIX_M
#define TN NAIAD_MATRIX_N
#define TK NAIAD_MATRIX_K

#if GM % (2 * TM) || GN % (2 * TN) || GK % TK
#error "GEMM_M/N must be multiples of twice the tile M/N, and GEMM_K a multiple of the tile K"
#endif
#if NAIAD_MATRIX_TILES < 4 || NAIAD_MATRIX_ACCS < 4
#error "gemm_ame needs at least 4 tiles and 4 accumulators"
#endif

static uint32_t Af[GM][GK], Bf[GK][GN], Cf[GM][GN];
static int8_t Ai[GM][GK], Bi[GK][GN];
static int32_t Ci[GM][GN];

/* One blocked GEMM: a, b, c are the arrays, es/ec their element sizes, fi/fc the formats. */
#define BLOCKED_GEMM(a, b, c, es, ec, fi, fc)                                                      \
    for (int i0 = 0; i0 < GM; i0 += 2 * TM) {                                                      \
        for (int j0 = 0; j0 < GN; j0 += 2 * TN) {                                                  \
            mx_azero(0);                                                                           \
            mx_azero(1);                                                                           \
            mx_azero(2);                                                                           \
            mx_azero(3);                                                                           \
            for (int k0 = 0; k0 < GK; k0 += TK) {                                                  \
                mx_tload_a(0, fi, &a[i0][k0], GK * (es));                                          \
                mx_tload_a(1, fi, &a[i0 + TM][k0], GK * (es));                                     \
                mx_tload_b(2, fi, &b[k0][j0], GN * (es));                                          \
                mx_tload_b(3, fi, &b[k0][j0 + TN], GN * (es));                                     \
                mx_mma(0, 0, 2, fi, fc);                                                           \
                mx_mma(1, 0, 3, fi, fc);                                                           \
                mx_mma(2, 1, 2, fi, fc);                                                           \
                mx_mma(3, 1, 3, fi, fc);                                                           \
            }                                                                                      \
            mx_astore(0, fc, &c[i0][j0], GN * (ec));                                               \
            mx_astore(1, fc, &c[i0][j0 + TN], GN * (ec));                                          \
            mx_astore(2, fc, &c[i0 + TM][j0], GN * (ec));                                          \
            mx_astore(3, fc, &c[i0 + TM][j0 + TN], GN * (ec));                                     \
        }                                                                                          \
    }

static int check(const char *what, long long got, long long want) {
    naiad_printf("  %s: %lld (expect %lld) %s\n", what, got, want, got == want ? "ok" : "MISMATCH");
    return got == want;
}

int main(void) {
    int ok = 1;
    naiad_printf("gemm_ame %dx%dx%d on a %lux%lux%lu unit (%lu tiles, %lu accs, features %lx)\n", GM, GN, GK,
                 mx_mcfg(NAIAD_MCFG_M), mx_mcfg(NAIAD_MCFG_N), mx_mcfg(NAIAD_MCFG_K), mx_mcfg(NAIAD_MCFG_TILES),
                 mx_mcfg(NAIAD_MCFG_ACCS), mx_mcfg(NAIAD_MCFG_FEATURES));
    ok &= check("mcfg M/N/K match naiad.h",
                mx_mcfg(NAIAD_MCFG_M) == TM && mx_mcfg(NAIAD_MCFG_N) == TN && mx_mcfg(NAIAD_MCFG_K) == TK, 1);
    ok &= check("fp32 and int8 inputs implemented",
                mx_has_in_format(NAIAD_MFMT_FP32) && mx_has_in_format(NAIAD_MFMT_INT8), 1);

    for (int i = 0; i < GM; i++) {
        for (int k = 0; k < GK; k++) {
            Ai[i][k] = (int8_t)gemm_a(i, k);
            Af[i][k] = gemm_i2f(gemm_a(i, k));
        }
    }
    for (int k = 0; k < GK; k++) {
        for (int j = 0; j < GN; j++) {
            Bi[k][j] = (int8_t)gemm_b(k, j);
            Bf[k][j] = gemm_i2f(gemm_b(k, j));
        }
    }

    uint64_t start = naiad_cycles();
    BLOCKED_GEMM(Af, Bf, Cf, 4, 4, NAIAD_MFMT_FP32, NAIAD_MFMT_FP32)
    uint64_t fp32_cycles = naiad_cycles() - start;
    start = naiad_cycles();
    BLOCKED_GEMM(Ai, Bi, Ci, 1, 4, NAIAD_MFMT_INT8, NAIAD_MFMT_INT32)
    uint64_t int8_cycles = naiad_cycles() - start;
    naiad_printf("  fp32 GEMM: %llu cycles; int8 GEMM: %llu cycles\n", (unsigned long long)fp32_cycles,
                 (unsigned long long)int8_cycles);

    int agree = 1;
    uint64_t h = GEMM_HASH_INIT;
    for (int i = 0; i < GM; i++) {
        for (int j = 0; j < GN; j++) {
            agree &= gemm_f2i(Cf[i][j]) == Ci[i][j];
            h = gemm_hash_step(h, Ci[i][j]);
        }
    }
    ok &= check("fp32 result == int8/int32 result", agree, 1);
    int diag = 1;
    for (int i = 0; i < GM && i < GN; i++) {
        int32_t want = 0;
        for (int k = 0; k < GK; k++) {
            want += gemm_a(i, k) * gemm_b(k, i);
        }
        diag &= Ci[i][i] == want;
    }
    ok &= check("diagonal == scalar recomputation", diag, 1);
    naiad_printf("  hash %llx\n", (unsigned long long)h);
#ifdef GEMM_EXPECTED_HASH
    ok &= check("hash matches host", h == GEMM_EXPECTED_HASH, 1);
#endif
#ifdef NAIAD_PERF_MATRIX_MMA
    long long mmas = 2LL * (GM / TM) * (GN / TN) * (GK / TK);
    ok &= check("mma issued", (long long)naiad_perf_read(NAIAD_PERF_MATRIX_MMA), mmas);
    ok &= check("MACs", (long long)naiad_perf_read(NAIAD_PERF_MATRIX_MACS), mmas * TM * TN * TK);
#endif
    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
