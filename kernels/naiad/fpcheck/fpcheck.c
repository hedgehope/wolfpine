/*
 * fpcheck: F and D on Naiad (build against naiad-fp-rv64.yaml).
 *
 * Arithmetic, sqrt, fused multiply-add, conversions (including the saturating
 * and rounding-mode-specific ones), fflags, frm, NaN-boxing and a 16-element
 * dot product, in float and in double. Every expected value is a literal bit
 * pattern computed on the host (numpy for the RNE arithmetic, exact rationals
 * for the fused results), so this checks the simulator against IEEE rather
 * than against itself. Inputs are volatile so nothing is folded at compile time.
 *
 * Instructions whose C spelling is undefined behaviour (out-of-range
 * conversions) or that need a particular rounding mode or flag access are
 * written as inline asm.
 */
#include "naiad_platform.h"

/* The unfused dot products must be a separate multiply and add. */
#pragma STDC FP_CONTRACT OFF

#if !defined(__riscv_d) || __riscv_flen != 64
#error "fpcheck needs D: build against naiad-fp-rv64.yaml (rv64imafd, lp64d)"
#endif

#define N 16

/* a[i] = 1 / (i + 3), b[i] = (i + 1) * 0.3 - 2, each step rounded in its format. */
static volatile uint32_t a_f_bits[N] = {
    0x3eaaaaab, 0x3e800000, 0x3e4ccccd, 0x3e2aaaab, 0x3e124925, 0x3e000000, 0x3de38e39, 0x3dcccccd,
    0x3dba2e8c, 0x3daaaaab, 0x3d9d89d9, 0x3d924925, 0x3d888889, 0x3d800000, 0x3d70f0f1, 0x3d638e39,
};
static volatile uint32_t b_f_bits[N] = {
    0xbfd9999a, 0xbfb33333, 0xbf8ccccc, 0xbf4ccccc, 0xbf000000, 0xbe4cccc8, 0x3dcccce0, 0x3eccccd0,
    0x3f333334, 0x3f800000, 0x3fa66668, 0x3fccccce, 0x3ff33334, 0x400cccce, 0x40200000, 0x40333334,
};
static volatile uint64_t a_d_bits[N] = {
    0x3fd5555555555555, 0x3fd0000000000000, 0x3fc999999999999a, 0x3fc5555555555555,
    0x3fc2492492492492, 0x3fc0000000000000, 0x3fbc71c71c71c71c, 0x3fb999999999999a,
    0x3fb745d1745d1746, 0x3fb5555555555555, 0x3fb3b13b13b13b14, 0x3fb2492492492492,
    0x3fb1111111111111, 0x3fb0000000000000, 0x3fae1e1e1e1e1e1e, 0x3fac71c71c71c71c,
};
static volatile uint64_t b_d_bits[N] = {
    0xbffb333333333333, 0xbff6666666666666, 0xbff199999999999a, 0xbfe999999999999a,
    0xbfe0000000000000, 0xbfc99999999999a0, 0x3fb99999999999a0, 0x3fd9999999999998,
    0x3fe6666666666664, 0x3ff0000000000000, 0x3ff4cccccccccccc, 0x3ff9999999999998,
    0x3ffe666666666666, 0x400199999999999a, 0x4004000000000000, 0x4006666666666666,
};

static float f_of(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, sizeof f);
    return f;
}
static double d_of(uint64_t b) {
    double d;
    __builtin_memcpy(&d, &b, sizeof d);
    return d;
}
static uint32_t bits_f(float f) {
    uint32_t b;
    __builtin_memcpy(&b, &f, sizeof b);
    return b;
}
static uint64_t bits_d(double d) {
    uint64_t b;
    __builtin_memcpy(&b, &d, sizeof b);
    return b;
}

/* -- inline asm wrappers ----------------------------------------------------- */

static inline float fsqrt_s(float a) {
    float r;
    __asm__ volatile("fsqrt.s %0, %1" : "=f"(r) : "f"(a));
    return r;
}
static inline double fsqrt_d(double a) {
    double r;
    __asm__ volatile("fsqrt.d %0, %1" : "=f"(r) : "f"(a));
    return r;
}
static inline float fmadd_s(float a, float b, float c) {
    float r;
    __asm__ volatile("fmadd.s %0, %1, %2, %3" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}
static inline double fmadd_d(double a, double b, double c) {
    double r;
    __asm__ volatile("fmadd.d %0, %1, %2, %3" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}
static inline float fmsub_s(float a, float b, float c) {
    float r;
    __asm__ volatile("fmsub.s %0, %1, %2, %3" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}
static inline double fmsub_d(double a, double b, double c) {
    double r;
    __asm__ volatile("fmsub.d %0, %1, %2, %3" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}
static inline double fdiv_d(double a, double b) {
    double r;
    __asm__ volatile("fdiv.d %0, %1, %2" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
static inline double fmul_d(double a, double b) {
    double r;
    __asm__ volatile("fmul.d %0, %1, %2" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
static inline double fadd_d_dyn(double a, double b) {
    double r;
    __asm__ volatile("fadd.d %0, %1, %2, dyn" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
static inline double fmin_d(double a, double b) {
    double r;
    __asm__ volatile("fmin.d %0, %1, %2" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
static inline long fcvt_w_d(double a) { /* rm = rne */
    long r;
    __asm__ volatile("fcvt.w.d %0, %1, rne" : "=r"(r) : "f"(a));
    return r;
}
static inline long fcvt_wu_d(double a) {
    long r;
    __asm__ volatile("fcvt.wu.d %0, %1, rtz" : "=r"(r) : "f"(a));
    return r;
}
static inline long fcvt_l_d_rdn(double a) {
    long r;
    __asm__ volatile("fcvt.l.d %0, %1, rdn" : "=r"(r) : "f"(a));
    return r;
}
static inline uint64_t fmv_x_d_of_single(float a) {
    uint64_t r;
    __asm__ volatile("fmv.x.d %0, %1" : "=r"(r) : "f"(a));
    return r;
}
static inline unsigned long fflags_swap(unsigned long v) {
    unsigned long old;
    __asm__ volatile("fsflags %0, %1" : "=r"(old) : "r"(v));
    return old;
}
static inline void set_frm(unsigned long v) { __asm__ volatile("fsrm %0" : : "r"(v)); }

/* -- checking ------------------------------------------------------------------ */

static int failures;

static void check(const char *what, uint64_t got, uint64_t want) {
    int ok = got == want;
    naiad_printf("  %s: %016llx (expect %016llx) %s\n", what, (unsigned long long)got,
                 (unsigned long long)want, ok ? "ok" : "MISMATCH");
    failures += !ok;
}

/* fflags accrued by `expr`, which is evaluated for its side effect. */
#define FLAGS_OF(expr)                                                                             \
    ({                                                                                             \
        fflags_swap(0);                                                                            \
        volatile double sink_ = (expr);                                                            \
        (void)sink_;                                                                               \
        fflags_swap(0);                                                                            \
    })

static volatile float xf = 1.1f, yf = 3.3f, pf = 0x1.001p+0f, qf = 0x1.ffep-1f; /* 1 +/- 2**-12 */
static volatile double xd = 1.1, yd = 3.3, third = 1.0 / 3.0, one = 1.0, zero = 0.0;
static volatile double pd = 0x1.00000004p+0, qd = 0x1.fffffff8p-1; /* 1 +/- 2**-30 */
static volatile double big = 0x1.fffffffffffffp+1023, minus_one = -1.0, nan_d = __builtin_nan("");
static volatile int i_big = 16777217;
static volatile long l_big = 9007199254740993L;

__attribute__((noinline)) static void run_checks(void) {
    naiad_puts("single:\n");
    check("1.1f + 3.3f", bits_f(xf + yf), 0x408ccccd);
    check("1.1f - 3.3f", bits_f(xf - yf), 0xc00ccccc);
    check("1.1f * 3.3f", bits_f(xf * yf), 0x406851ec);
    check("1.1f / 3.3f", bits_f(xf / yf), 0x3eaaaaab);
    check("sqrt(3.3f)", bits_f(fsqrt_s(yf)), 0x3fe88607);
    check("fmsub (1+2^-12)(1-2^-12)-1", bits_f(fmsub_s(pf, qf, 1.0f)), 0xb3800000);
    check("NaN-boxed in the f-register", fmv_x_d_of_single(xf), 0xffffffff3f8ccccdull);

    naiad_puts("double:\n");
    check("1.1 + 3.3", bits_d(xd + yd), 0x401199999999999a);
    check("1.1 - 3.3", bits_d(xd - yd), 0xc001999999999999);
    check("1.1 * 3.3", bits_d(xd * yd), 0x400d0a3d70a3d70a);
    check("1.1 / 3.3", bits_d(xd / yd), 0x3fd5555555555556);
    check("sqrt(3.3)", bits_d(fsqrt_d(yd)), 0x3ffd10c0e60be308);
    check("fmsub (1+2^-30)(1-2^-30)-1", bits_d(fmsub_d(pd, qd, one)), 0xbc30000000000000);
    check("fmin(-0, +0)", bits_d(fmin_d(-zero, zero)), 0x8000000000000000);

    naiad_puts("dot products:\n");
    float sf = 0.0f, sff = 0.0f;
    double sd = 0.0, sdf = 0.0;
    for (int i = 0; i < N; i++) {
        float af = f_of(a_f_bits[i]), bf = f_of(b_f_bits[i]);
        float prod = af * bf;
        sf = sf + prod;
        sff = fmadd_s(af, bf, sff);
        double ad = d_of(a_d_bits[i]), bd = d_of(b_d_bits[i]);
        double prodd = ad * bd;
        sd = sd + prodd;
        sdf = fmadd_d(ad, bd, sdf);
    }
    check("float, mul then add", bits_f(sf), 0xbec649ae);
    check("float, fmadd", bits_f(sff), 0xbec649ad);
    check("double, mul then add", bits_d(sd), 0xbfd8c93642e523a8);
    check("double, fmadd", bits_d(sdf), 0xbfd8c93642e523a4);

    naiad_puts("conversions:\n");
    check("(float)(1.0/3.0)", bits_f((float)third), 0x3eaaaaab);
    check("(double)1.1f", bits_d((double)xf), 0x3ff19999a0000000);
    check("(float)16777217", bits_f((float)i_big), 0x4b800000);
    check("(double)(2^53+1)", bits_d((double)l_big), 0x4340000000000000);
    check("(int)-2.7 (rtz)", (uint64_t)(int64_t)(int)(-2.7 * one), (uint64_t)-2);
    check("(unsigned long)3e9", (unsigned long)(3e9 * one), 3000000000ull);
    check("fcvt.w.d 2.5, rne", (uint64_t)fcvt_w_d(2.5 * one), 2);
    check("fcvt.l.d -2.5, rdn", (uint64_t)fcvt_l_d_rdn(-2.5 * one), (uint64_t)-3);
    check("fcvt.w.d 1e10 saturates", (uint64_t)fcvt_w_d(1e10 * one), 0x7fffffff);
    check("fcvt.w.d NaN", (uint64_t)fcvt_w_d(nan_d), 0x7fffffff);
    check("fcvt.wu.d -1 saturates", (uint64_t)fcvt_wu_d(minus_one), 0);

    naiad_puts("fflags and frm:\n");
    check("1/3: NX", FLAGS_OF(fdiv_d(one, 3.0 * one)), 0x01);
    check("1/0: DZ", FLAGS_OF(fdiv_d(one, zero)), 0x08);
    check("sqrt(-1): NV", FLAGS_OF(fsqrt_d(minus_one)), 0x10);
    check("max*2: OF|NX", FLAGS_OF(fmul_d(big, 2.0 * one)), 0x05);
    check("2*0.5: none", FLAGS_OF(fmul_d(2.0 * one, 0.5 * one)), 0x00);
    set_frm(3); /* RUP */
    check("1/3 + 1, frm = rup", bits_d(fadd_d_dyn(third, one)), 0x3ff5555555555556);
    set_frm(0); /* RNE */
    check("1/3 + 1, frm = rne", bits_d(fadd_d_dyn(third, one)), 0x3ff5555555555555);
}

int main(void) {
    /* mstatus.FS = Initial: Wolfpine does not need it, but on Spike (and on
     * hardware) FP instructions trap while FS is Off. Set before any FP code. */
    __asm__ volatile("csrs mstatus, %0" : : "r"(1ul << 13));
    naiad_printf("fpcheck on naiad: design %s (%s)\n", NAIAD_DESIGN, NAIAD_MARCH);
    run_checks();
    naiad_puts(failures ? "FAIL\n" : "PASS\n");
    return failures ? 1 : 0;
}
