/*
 * reduce: integer and FP reductions, strip-mined with a running scalar.
 *   vredsum / vredmax / vredminu (int32), vwredsum (int16 -> int32)
 *   vfredosum / vfredmax / vfredmin (fp32, exact), vfredusum (tolerance)
 */
#include "k1common.h"

#define NMAX 1000
static int32_t xi[NMAX];
static int16_t xh[NMAX];
static float xf[NMAX];

static void reduce_all(size_t n, uint32_t *out_i, float *out_f, float *out_u) {
    vint32m1_t sum = __riscv_vmv_s_x_i32m1(0, 1), mx = __riscv_vmv_s_x_i32m1(INT32_MIN, 1);
    vuint32m1_t mnu = __riscv_vmv_s_x_u32m1(UINT32_MAX, 1);
    vfloat32m1_t fs = __riscv_vfmv_s_f_f32m1(0.0f, 1), fmx = __riscv_vfmv_s_f_f32m1(-1e30f, 1);
    vfloat32m1_t fmn = __riscv_vfmv_s_f_f32m1(1e30f, 1), fu = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    size_t vlmax4 = __riscv_vsetvlmax_e32m4();
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m4(n - i < vlmax4 ? n - i : vlmax4);
        vint32m4_t v = __riscv_vle32_v_i32m4(xi + i, vl);
        sum = __riscv_vredsum_vs_i32m4_i32m1(v, sum, vl);
        mx = __riscv_vredmax_vs_i32m4_i32m1(v, mx, vl);
        mnu = __riscv_vredminu_vs_u32m4_u32m1(__riscv_vreinterpret_v_i32m4_u32m4(v), mnu, vl);
        vfloat32m4_t f = __riscv_vle32_v_f32m4(xf + i, vl);
        fs = __riscv_vfredosum_vs_f32m4_f32m1(f, fs, vl);
        fmx = __riscv_vfredmax_vs_f32m4_f32m1(f, fmx, vl);
        fmn = __riscv_vfredmin_vs_f32m4_f32m1(f, fmn, vl);
        fu = __riscv_vfredusum_vs_f32m4_f32m1(f, fu, vl);
        i += vl;
    }
    vint32m1_t wsum = __riscv_vmv_s_x_i32m1(0, 1);
    size_t vlmax = __riscv_vsetvlmax_e16m2();
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e16m2(n - i < vlmax ? n - i : vlmax);
        wsum = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vle16_v_i16m2(xh + i, vl), wsum, vl);
        i += vl;
    }
    out_i[0] = (uint32_t)__riscv_vmv_x_s_i32m1_i32(sum);
    out_i[1] = (uint32_t)__riscv_vmv_x_s_i32m1_i32(mx);
    out_i[2] = __riscv_vmv_x_s_u32m1_u32(mnu);
    out_i[3] = (uint32_t)__riscv_vmv_x_s_i32m1_i32(wsum);
    out_f[0] = __riscv_vfmv_f_s_f32m1_f32(fs);
    out_f[1] = __riscv_vfmv_f_s_f32m1_f32(fmx);
    out_f[2] = __riscv_vfmv_f_s_f32m1_f32(fmn);
    out_u[0] = __riscv_vfmv_f_s_f32m1_f32(fu);
}

int main(void) {
    static const size_t sizes[] = {1, 17, 32, 33, 999};
    static const char *iname[] = {"reduce_int_n1", "reduce_int_n17", "reduce_int_n32", "reduce_int_n33", "reduce_int_n999"};
    static const char *fname[] = {"reduce_fp_n1", "reduce_fp_n17", "reduce_fp_n32", "reduce_fp_n33", "reduce_fp_n999"};
    static const char *uname[] = {"reduce_usum_n1", "reduce_usum_n17", "reduce_usum_n32", "reduce_usum_n33", "reduce_usum_n999"};
    k1_header("reduce");
    for (unsigned s = 0; s < 5; s++) {
        size_t n = sizes[s];
        k1_seed(300 + n);
        for (size_t i = 0; i < n; i++) {
            xi[i] = (int32_t)k1_rand32();
            xh[i] = (int16_t)k1_rand32();
            xf[i] = k1_randf() * 1000.0f;
        }
        uint32_t ri[4];
        float rf[3], ru[1];
        reduce_all(n, ri, rf, ru);
        k1_emit_u32(iname[s], ri, 4);
        k1_emit_f32(fname[s], rf, 3);
        k1_emit_approx_f32(uname[s], ru, 1);
    }
    uint32_t ri[4];
    float rf[3], ru[1];
    uint64_t t0 = naiad_cycles();
    for (unsigned r = 0; r < K1_REPS; r++) {
        reduce_all(NMAX - 1, ri, rf, ru);
        K1_ESCAPE(ri);
        K1_ESCAPE(rf);
        K1_ESCAPE(ru);
    }
    uint64_t t1 = naiad_cycles();
    k1_time("reduce_all_n999", t0, t1, K1_REPS);
    return 0;
}
