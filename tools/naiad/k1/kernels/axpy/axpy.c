/* axpy: y = a * x + y (vfmacc.vf), fp32 at LMUL 2 and fp64 at LMUL 1. */
#include "k1common.h"

static void axpy_f32(size_t n, float a, const float *x, float *y) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m2(n - i < vlmax ? n - i : vlmax);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(x + i, vl);
        vfloat32m2_t vy = __riscv_vle32_v_f32m2(y + i, vl);
        __riscv_vse32_v_f32m2(y + i, __riscv_vfmacc_vf_f32m2(vy, a, vx, vl), vl);
        i += vl;
    }
}

static void axpy_f64(size_t n, double a, const double *x, double *y) {
    size_t vlmax = __riscv_vsetvlmax_e64m1();
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e64m1(n - i < vlmax ? n - i : vlmax);
        vfloat64m1_t vx = __riscv_vle64_v_f64m1(x + i, vl);
        vfloat64m1_t vy = __riscv_vle64_v_f64m1(y + i, vl);
        __riscv_vse64_v_f64m1(y + i, __riscv_vfmacc_vf_f64m1(vy, a, vx, vl), vl);
        i += vl;
    }
}

#define NMAX 1000
static float xf[NMAX], yf[NMAX];
static double xd[NMAX], yd[NMAX];

int main(void) {
    static const size_t sizes[] = {1, 13, 64, 257};
    static const char *f32_names[] = {"axpy_f32_n1", "axpy_f32_n13", "axpy_f32_n64", "axpy_f32_n257"};
    static const char *f64_names[] = {"axpy_f64_n1", "axpy_f64_n13", "axpy_f64_n64", "axpy_f64_n257"};
    k1_header("axpy");
    for (unsigned s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
        size_t n = sizes[s];
        k1_seed(n);
        for (size_t i = 0; i < n; i++) {
            xf[i] = k1_randf();
            yf[i] = k1_randf();
            xd[i] = k1_randd();
            yd[i] = k1_randd();
        }
        float a = k1_randf();
        double ad = k1_randd();
        axpy_f32(n, a, xf, yf);
        axpy_f64(n, ad, xd, yd);
        k1_emit_f32(f32_names[s], yf, n);
        k1_emit_f64(f64_names[s], yd, n);
    }
    uint64_t t0 = naiad_cycles();
    for (unsigned r = 0; r < K1_REPS; r++) {
        axpy_f32(NMAX, 0.5f, xf, yf);
        K1_CLOBBER();
    }
    uint64_t t1 = naiad_cycles();
    k1_time("axpy_f32_n1000", t0, t1, K1_REPS);
    return 0;
}
