/*
 * xsmt_gemm: an int8 -> int32 GEMM on SpacemiT's IME instructions (XSMTVDot,
 * `smt.vmadot`), the K1's real-hardware proxy for Naiad's IME geometry.
 * Board only, optional: needs GCC >= 16.1 or LLVM >= 22.1 (`-march=..._xsmtvdot`).
 *
 * On the K1 (VLEN 256) one `smt.vmadot vd, vs1, vs2` is C(4x4, int32) += A(4x8,
 * int8) x B(8x4, int8), with
 *   vs1 = A, row-major (32 bytes),
 *   vs2 = B^T, i.e. B column-major (32 bytes),
 *   vd, vd+1 = C (16 int32; vd even; must not overlap vs1/vs2), LMUL = 1 only.
 * SpacemiT's IME doc (v0.6) says C is row-major in one place and
 * "implementation-defined" in another, so the program first probes one tile
 * against a scalar reference and reports the layout it finds (row- or
 * column-major) before using it.
 *
 * vtype: the instructions are run under `vsetvli e8, m1, vl = 32`. The doc
 * gives no explicit sequence; build with -DXSMT_E32 to run them under
 * e32, m1 instead if the hardware rejects e8.
 *
 * Output: R lines for the IME result and the scalar reference (identical when it
 * works; compare.py can diff two boards' runs), T lines for the IME and scalar
 * GEMM, and exit status 1 if the IME result differs from the reference.
 */
#include "k1common.h"

#ifndef XSMT_VMADOT
#define XSMT_VMADOT "smt.vmadot"
#endif
/* Appended after the operands; e.g. ", i8" if the assembler wants a type operand. */
#ifndef XSMT_SUFFIX
#define XSMT_SUFFIX ""
#endif

#define TM 4
#define TN 4
#define TK 8
#define M 16
#define N 16
#define K 64

static int8_t A[M * K], B[K * N];
static int32_t C_ref[M * N], C_ime[M * N];
static int8_t Apack[(M / TM) * (K / TK) * 32], Bpack[(N / TN) * (K / TK) * 32];

static inline void set_vtype(void) {
#ifdef XSMT_E32
    (void)__riscv_vsetvl_e32m1(8);
#else
    (void)__riscv_vsetvl_e8m1(32);
#endif
}

/* c (vd, vd+1 as one m2 group, so vd is even) += a x b^T. */
static inline vint32m2_t vmadot(vint32m2_t c, vint8m1_t a, vint8m1_t b) {
#ifdef XSMT_EMULATE
    /* The documented semantics in scalar code (row-major C), to test this
     * harness without the hardware (e.g. under qemu): make xsmt XSMT_EMULATE=1. */
    int8_t av[32], bv[32];
    int32_t cv[16];
    __riscv_vse8_v_i8m1(av, a, 32);
    __riscv_vse8_v_i8m1(bv, b, 32);
    __riscv_vse32_v_i32m2(cv, c, 16);
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            for (int q = 0; q < TK; q++) cv[i * TN + j] += (int32_t)av[i * TK + q] * (int32_t)bv[j * TK + q];
    return __riscv_vle32_v_i32m2(cv, 16);
#else
    __asm__ volatile(XSMT_VMADOT " %0, %1, %2" XSMT_SUFFIX : "+vr"(c) : "vr"(a), "vr"(b));
    return c;
#endif
}

static void gemm_ref(void) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            C_ref[i * N + j] = s;
        }
}

static void pack(void) {
    for (int ti = 0; ti < M / TM; ti++)
        for (int kb = 0; kb < K / TK; kb++)
            for (int i = 0; i < TM; i++)
                for (int q = 0; q < TK; q++)
                    Apack[(ti * (K / TK) + kb) * 32 + i * TK + q] = A[(ti * TM + i) * K + kb * TK + q];
    for (int tj = 0; tj < N / TN; tj++)
        for (int kb = 0; kb < K / TK; kb++)
            for (int j = 0; j < TN; j++)
                for (int q = 0; q < TK; q++)
                    Bpack[(tj * (K / TK) + kb) * 32 + j * TK + q] = B[(kb * TK + q) * N + tj * TN + j];
}

/* One C tile over the whole K; the 16 results in register order. */
static void tile(int ti, int tj, int32_t out[16]) {
    set_vtype();
    vint32m2_t c = __riscv_vmv_v_x_i32m2(0, 16);
    for (int kb = 0; kb < K / TK; kb++) {
        set_vtype();
        vint8m1_t a = __riscv_vle8_v_i8m1(Apack + (ti * (K / TK) + kb) * 32, 32);
        vint8m1_t b = __riscv_vle8_v_i8m1(Bpack + (tj * (K / TK) + kb) * 32, 32);
        c = vmadot(c, a, b);
    }
    __riscv_vse32_v_i32m2(out, c, 16);
}

static int row_major = 1;

static void gemm_ime(void) {
    int32_t t[16];
    for (int ti = 0; ti < M / TM; ti++)
        for (int tj = 0; tj < N / TN; tj++) {
            tile(ti, tj, t);
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    C_ime[(ti * TM + i) * N + tj * TN + j] = row_major ? t[i * TN + j] : t[j * TM + i];
        }
}

int main(void) {
    k1_header("xsmt_gemm");
    k1_seed(2027);
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)k1_rand32();
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)k1_rand32();
    gemm_ref();
    pack();

    /* Probe the C layout on tile (1, 2). */
    int32_t t[16];
    tile(1, 2, t);
    int rm = 1, cm = 1;
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) {
            int32_t want = C_ref[(1 * TM + i) * N + 2 * TN + j];
            rm &= t[i * TN + j] == want;
            cm &= t[j * TM + i] == want;
        }
    naiad_printf("# xsmt C layout: %s\n", rm ? "row-major" : cm ? "column-major" : "neither (see R xsmt_probe)");
    k1_emit_u32("xsmt_probe", (const uint32_t *)t, 16);
    row_major = rm || !cm;

    gemm_ime();
    int bad = 0;
    for (int i = 0; i < M * N; i++) bad += C_ime[i] != C_ref[i];
    naiad_printf("# xsmt gemm %dx%dx%d: %d of %d elements differ from the scalar reference\n", M, N, K, bad, M * N);
    k1_emit_u32("xsmt_ime_16x16x64", (const uint32_t *)C_ime, M * N);
    k1_emit_u32("xsmt_ref_16x16x64", (const uint32_t *)C_ref, M * N);

    uint64_t t0 = naiad_cycles();
    for (unsigned r = 0; r < K1_REPS; r++) {
        gemm_ime();
        K1_CLOBBER();
    }
    uint64_t t1 = naiad_cycles();
    k1_time("xsmt_ime_16x16x64", t0, t1, K1_REPS);
    t0 = naiad_cycles();
    for (unsigned r = 0; r < K1_REPS; r++) {
        gemm_ref();
        K1_CLOBBER();
    }
    t1 = naiad_cycles();
    k1_time("xsmt_scalar_16x16x64", t0, t1, K1_REPS);
    return bad != 0;
}
