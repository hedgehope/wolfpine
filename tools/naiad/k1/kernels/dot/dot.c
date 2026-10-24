/*
 * dot: three fp32 dot products.
 *   dot_osum  per chunk vfmul + vfredosum into a running scalar (ordered: exact)
 *   dot_f64   vfwmul to fp64 + vfredosum (ordered: exact)
 *   dot_vacc  a vector of partial sums (vfmacc, tail undisturbed) then one
 *             vfredusum: the unordered sum's order is implementation-defined,
 *             so it is compared with a tolerance
 */
#include "k1common.h"

static float dot_osum(size_t n, const float *x, const float *y) {
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m1_t acc = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m4(n - i < vlmax ? n - i : vlmax);
        vfloat32m4_t p = __riscv_vfmul_vv_f32m4(__riscv_vle32_v_f32m4(x + i, vl), __riscv_vle32_v_f32m4(y + i, vl), vl);
        acc = __riscv_vfredosum_vs_f32m4_f32m1(p, acc, vl);
        i += vl;
    }
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

static double dot_f64(size_t n, const float *x, const float *y) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat64m1_t acc = __riscv_vfmv_s_f_f64m1(0.0, 1);
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m2(n - i < vlmax ? n - i : vlmax);
        vfloat64m4_t p = __riscv_vfwmul_vv_f64m4(__riscv_vle32_v_f32m2(x + i, vl), __riscv_vle32_v_f32m2(y + i, vl), vl);
        acc = __riscv_vfredosum_vs_f64m4_f64m1(p, acc, vl);
        i += vl;
    }
    return __riscv_vfmv_f_s_f64m1_f64(acc);
}

static float dot_vacc(size_t n, const float *x, const float *y) {
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m4(n - i < vlmax ? n - i : vlmax);
        acc = __riscv_vfmacc_vv_f32m4_tu(acc, __riscv_vle32_v_f32m4(x + i, vl), __riscv_vle32_v_f32m4(y + i, vl), vl);
        i += vl;
    }
    vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc, zero, vlmax));
}

#define NMAX 1000
static float x[NMAX], y[NMAX];

int main(void) {
    static const size_t sizes[] = {1, 31, 32, 33, 200, 1000};
    k1_header("dot");
    float os[6], va[6];
    double wd[6];
    for (unsigned s = 0; s < 6; s++) {
        size_t n = sizes[s];
        k1_seed(100 + n);
        for (size_t i = 0; i < n; i++) {
            x[i] = k1_randf();
            y[i] = k1_randf();
        }
        os[s] = dot_osum(n, x, y);
        wd[s] = dot_f64(n, x, y);
        va[s] = dot_vacc(n, x, y);
    }
    k1_emit_f32("dot_osum", os, 6);
    k1_emit_f64("dot_f64", wd, 6);
    k1_emit_approx_f32("dot_vacc", va, 6);
    volatile float sink = 0;
    uint64_t t0 = naiad_cycles();
    for (unsigned r = 0; r < K1_REPS; r++) {
        sink += dot_osum(NMAX, x, y);
        K1_CLOBBER();
    }
    uint64_t t1 = naiad_cycles();
    k1_time("dot_osum_n1000", t0, t1, K1_REPS);
    t0 = naiad_cycles();
    for (unsigned r = 0; r < K1_REPS; r++) {
        sink += dot_vacc(NMAX, x, y);
        K1_CLOBBER();
    }
    t1 = naiad_cycles();
    k1_time("dot_vacc_n1000", t0, t1, K1_REPS);
    (void)sink;
    return 0;
}
