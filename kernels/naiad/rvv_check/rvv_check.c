/*
 * rvv_check: the RVV unit end to end (build against naiad-rvv.yaml).
 *
 * Each check computes something with RVV intrinsics and compares it with a
 * scalar reference: exact integer maths, or the scalar F/D unit (validated
 * separately against Spike) evaluated in the same order and with the same
 * roundings, so FP results must agree bit for bit. Checks whose vector order
 * legitimately differs from the scalar one (the unordered reduction) compare
 * within a tolerance instead. fp16 and bf16 are checked without scalar half
 * arithmetic (the scalar core has no Zfh): fp16 against the same sum done in
 * fp32 through the widening/narrowing converts, bf16 with integer bit maths.
 *
 * Prints one line per check, then PASS (exit 0) or FAIL (exit 1).
 */
#include <riscv_vector.h>

#include "naiad_platform.h"

#pragma STDC FP_CONTRACT OFF

#if !defined(__riscv_v)
#error "rvv_check needs V: build against naiad-rvv.yaml"
#endif

#define N 37 /* not a multiple of any VLMAX, so every strip-mined loop has a tail */

static int failures;

/* Fused multiply-adds as single instructions (-fno-builtin turns __builtin_fma* into libm calls). */
static float fmaf_(float a, float b, float c) {
    float r;
    __asm__("fmadd.s %0, %1, %2, %3" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

static double fma_(double a, double b, double c) {
    double r;
    __asm__("fmadd.d %0, %1, %2, %3" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

static void check(const char *name, int ok) {
    naiad_printf("%s: %s\n", name, ok ? "ok" : "MISMATCH");
    if (!ok)
        failures++;
}

static uint32_t f2u(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}

static uint64_t d2u(double d) {
    uint64_t u;
    __builtin_memcpy(&u, &d, 8);
    return u;
}

static volatile float xs[N], ys[N], zs[N];
static volatile double xd[N], yd[N];
static volatile int32_t xi[N];
static volatile int16_t hi16[N];

static void init(void) {
    for (int i = 0; i < N; i++) {
        xs[i] = 1.0f / (float)(i + 3);
        ys[i] = (float)(i + 1) * 0.3f - 2.0f;
        zs[i] = (float)((i * 7) % 11) - 5.0f;
        xd[i] = 1.0 / (double)(i + 3);
        yd[i] = (double)(i + 1) * 0.3 - 2.0;
        xi[i] = (i * 2654435761u) ^ 0x5a5a5a5a;
        hi16[i] = (int16_t)((i * 977) - 9000);
    }
}

/* y = a*x + y, fp32, strip-mined, fused per element. */
static void check_axpy_f32(void) {
    float a = 0.7f, y[N], ref[N];
    for (int i = 0; i < N; i++) {
        y[i] = ys[i];
        ref[i] = fmaf_(a, xs[i], ys[i]);
    }
    for (size_t i = 0, vl; i < N; i += vl) {
        vl = __riscv_vsetvl_e32m1(N - i);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1((const float *)&xs[i], vl);
        vfloat32m1_t vy = __riscv_vle32_v_f32m1(&y[i], vl);
        __riscv_vse32_v_f32m1(&y[i], __riscv_vfmacc_vf_f32m1(vy, a, vx, vl), vl);
    }
    int ok = 1;
    for (int i = 0; i < N; i++)
        ok &= f2u(y[i]) == f2u(ref[i]);
    check("axpy f32 (vfmacc.vf, LMUL 1)", ok);
}

/* fp64 axpy at LMUL 4. */
static void check_axpy_f64(void) {
    double a = -1.3, y[N], ref[N];
    for (int i = 0; i < N; i++) {
        y[i] = yd[i];
        ref[i] = fma_(a, xd[i], yd[i]);
    }
    for (size_t i = 0, vl; i < N; i += vl) {
        vl = __riscv_vsetvl_e64m4(N - i);
        vfloat64m4_t vx = __riscv_vle64_v_f64m4((const double *)&xd[i], vl);
        vfloat64m4_t vy = __riscv_vle64_v_f64m4(&y[i], vl);
        __riscv_vse64_v_f64m4(&y[i], __riscv_vfmacc_vf_f64m4(vy, a, vx, vl), vl);
    }
    int ok = 1;
    for (int i = 0; i < N; i++)
        ok &= d2u(y[i]) == d2u(ref[i]);
    check("axpy f64 (vfmacc.vf, LMUL 4)", ok);
}

/* Dot products: ordered (bit-exact against a sequential scalar loop) and unordered (tolerance). */
static void check_dot(void) {
    float seq = 0.0f;
    for (int i = 0; i < N; i++)
        seq = seq + xs[i] * ys[i];
    vfloat32m1_t acc = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    for (size_t i = 0, vl; i < N; i += vl) {
        vl = __riscv_vsetvl_e32m2(N - i);
        vfloat32m2_t p = __riscv_vfmul_vv_f32m2(__riscv_vle32_v_f32m2((const float *)&xs[i], vl),
                                                __riscv_vle32_v_f32m2((const float *)&ys[i], vl), vl);
        acc = __riscv_vfredosum_vs_f32m2_f32m1(p, acc, vl);
    }
    check("dot f32 ordered (vfredosum)", f2u(__riscv_vfmv_f_s_f32m1_f32(acc)) == f2u(seq));

    double exact = 0.0;
    for (int i = 0; i < N; i++)
        exact += (double)xs[i] * (double)ys[i];
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (size_t i = 0, vl; i < N; i += vl) {
        vl = __riscv_vsetvl_e32m1(N - i);
        vacc = __riscv_vfmacc_vv_f32m1_tu(vacc, __riscv_vle32_v_f32m1((const float *)&xs[i], vl),
                                          __riscv_vle32_v_f32m1((const float *)&ys[i], vl), vl);
    }
    vfloat32m1_t sum = __riscv_vfredusum_vs_f32m1_f32m1(vacc, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlmax);
    double got = __riscv_vfmv_f_s_f32m1_f32(sum), err = got - exact;
    check("dot f32 unordered (vfmacc + vfredusum), |err| < 1e-5", err < 1e-5 && err > -1e-5);
}

/* A 4x(2*VLEN/32) fp32 GEMM micro-kernel: C[4][n] += A[4][K] * B[K][n], B rows at LMUL 2. */
#define GM 4
#define GK 9
static void check_gemm(void) {
    size_t n = __riscv_vsetvlmax_e32m2();
    if (n > 32)
        n = 32;
    float A[GM][GK], B[GK][32], C[GM][32], R[GM][32];
    for (int i = 0; i < GM; i++)
        for (int k = 0; k < GK; k++)
            A[i][k] = xs[(i * GK + k) % N] * 3.0f;
    for (int k = 0; k < GK; k++)
        for (size_t j = 0; j < n; j++)
            B[k][j] = ys[(k * 5 + j) % N];
    for (int i = 0; i < GM; i++)
        for (size_t j = 0; j < n; j++) {
            float c = zs[(i + j) % N];
            C[i][j] = c;
            for (int k = 0; k < GK; k++)
                c = fmaf_(A[i][k], B[k][j], c);
            R[i][j] = c;
        }
    size_t vl = __riscv_vsetvl_e32m2(n);
    vfloat32m2_t c0 = __riscv_vle32_v_f32m2(C[0], vl), c1 = __riscv_vle32_v_f32m2(C[1], vl);
    vfloat32m2_t c2 = __riscv_vle32_v_f32m2(C[2], vl), c3 = __riscv_vle32_v_f32m2(C[3], vl);
    for (int k = 0; k < GK; k++) {
        vfloat32m2_t b = __riscv_vle32_v_f32m2(B[k], vl);
        c0 = __riscv_vfmacc_vf_f32m2(c0, A[0][k], b, vl);
        c1 = __riscv_vfmacc_vf_f32m2(c1, A[1][k], b, vl);
        c2 = __riscv_vfmacc_vf_f32m2(c2, A[2][k], b, vl);
        c3 = __riscv_vfmacc_vf_f32m2(c3, A[3][k], b, vl);
    }
    __riscv_vse32_v_f32m2(C[0], c0, vl);
    __riscv_vse32_v_f32m2(C[1], c1, vl);
    __riscv_vse32_v_f32m2(C[2], c2, vl);
    __riscv_vse32_v_f32m2(C[3], c3, vl);
    int ok = 1;
    for (int i = 0; i < GM; i++)
        for (size_t j = 0; j < n; j++)
            ok &= f2u(C[i][j]) == f2u(R[i][j]);
    check("gemm f32 4xn micro-kernel (vfmacc.vf, LMUL 2)", ok);
}

/* Integer reductions, including the widening one. */
static void check_reductions(void) {
    int32_t sum = 0, mx = INT32_MIN;
    uint32_t xr = 0;
    int32_t wsum = 0;
    for (int i = 0; i < N; i++) {
        sum += xi[i];
        mx = xi[i] > mx ? xi[i] : mx;
        xr ^= (uint32_t)xi[i];
        wsum += hi16[i];
    }
    vint32m1_t vsum = __riscv_vmv_s_x_i32m1(0, 1), vmax = __riscv_vmv_s_x_i32m1(INT32_MIN, 1);
    vuint32m1_t vxor = __riscv_vmv_s_x_u32m1(0, 1);
    vint32m1_t vw = __riscv_vmv_s_x_i32m1(0, 1);
    for (size_t i = 0, vl; i < N; i += vl) {
        vl = __riscv_vsetvl_e32m4(N - i);
        vint32m4_t v = __riscv_vle32_v_i32m4((const int32_t *)&xi[i], vl);
        vsum = __riscv_vredsum_vs_i32m4_i32m1(v, vsum, vl);
        vmax = __riscv_vredmax_vs_i32m4_i32m1(v, vmax, vl);
        vxor = __riscv_vredxor_vs_u32m4_u32m1(__riscv_vreinterpret_v_i32m4_u32m4(v), vxor, vl);
    }
    for (size_t i = 0, vl; i < N; i += vl) {
        vl = __riscv_vsetvl_e16m2(N - i);
        vw = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vle16_v_i16m2((const int16_t *)&hi16[i], vl), vw, vl);
    }
    check("vredsum.vs i32 (LMUL 4)", __riscv_vmv_x_s_i32m1_i32(vsum) == sum);
    check("vredmax.vs i32", __riscv_vmv_x_s_i32m1_i32(vmax) == mx);
    check("vredxor.vs u32", __riscv_vmv_x_s_u32m1_u32(vxor) == xr);
    check("vwredsum.vs i16 -> i32", __riscv_vmv_x_s_i32m1_i32(vw) == wsum);
}

/* Strided and indexed loads, an indexed store, and a segment load. */
static void check_memory(void) {
    float M[8][N];
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < N; c++)
            M[r][c] = (float)(r * 100 + c);
    /* Column 5 of M by a stride of one row. */
    size_t vl = __riscv_vsetvl_e32m1(8);
    float col[8];
    __riscv_vse32_v_f32m1(col, __riscv_vlse32_v_f32m1(&M[0][5], sizeof(M[0]), vl), vl);
    int ok = 1;
    for (size_t r = 0; r < vl; r++)
        ok &= col[r] == (float)(r * 100 + 5);
    check("strided load vlse32 (a matrix column)", ok);

    /* Gather M[0][perm[i]] by byte offsets, then scatter it back reversed. */
    uint32_t off[N];
    float g[N], s[N];
    for (int i = 0; i < N; i++)
        off[i] = (uint32_t)(((i * 13) % N) * sizeof(float));
    for (size_t i = 0, l; i < N; i += l) {
        l = __riscv_vsetvl_e32m2(N - i);
        vuint32m2_t vo = __riscv_vle32_v_u32m2(&off[i], l);
        vfloat32m2_t v = __riscv_vluxei32_v_f32m2(M[0], vo, l);
        __riscv_vse32_v_f32m2(&g[i], v, l);
        vuint32m2_t rev = __riscv_vrsub_vx_u32m2(__riscv_vid_v_u32m2(l), (uint32_t)(N - 1 - i), l);
        __riscv_vsoxei32_v_f32m2(s, __riscv_vsll_vx_u32m2(rev, 2, l), v, l);
    }
    ok = 1;
    for (int i = 0; i < N; i++)
        ok &= g[i] == (float)((i * 13) % N) && s[N - 1 - i] == g[i];
    check("indexed load vluxei32 + indexed store vsoxei32", ok);

    /* Deinterleave (re, im) pairs with vlseg2e32. */
    float cplx[2 * N], re[N], im[N];
    for (int i = 0; i < N; i++) {
        cplx[2 * i] = (float)i;
        cplx[2 * i + 1] = (float)-i;
    }
    for (size_t i = 0, l; i < N; i += l) {
        l = __riscv_vsetvl_e32m1(N - i);
        vfloat32m1x2_t t = __riscv_vlseg2e32_v_f32m1x2(&cplx[2 * i], l);
        __riscv_vse32_v_f32m1(&re[i], __riscv_vget_v_f32m1x2_f32m1(t, 0), l);
        __riscv_vse32_v_f32m1(&im[i], __riscv_vget_v_f32m1x2_f32m1(t, 1), l);
    }
    ok = 1;
    for (int i = 0; i < N; i++)
        ok &= re[i] == (float)i && im[i] == (float)-i;
    check("segment load vlseg2e32 (deinterleave)", ok);
}

/* Masks: compare, masked add with undisturbed inactive elements, vcpop, vfirst, vcompress. */
static void check_masks(void) {
    size_t vl = __riscv_vsetvl_e32m2(N < 16 ? N : 16);
    vfloat32m2_t y = __riscv_vle32_v_f32m2((const float *)ys, vl);
    vfloat32m2_t x = __riscv_vle32_v_f32m2((const float *)xs, vl);
    vbool16_t neg = __riscv_vmflt_vf_f32m2_b16(y, 0.0f, vl);
    vfloat32m2_t r = __riscv_vfadd_vv_f32m2_mu(neg, y, y, x, vl); /* y + x where y < 0, else y */
    float out[16], packed[16];
    __riscv_vse32_v_f32m2(out, r, vl);
    int ok = 1, count = 0, first = -1;
    for (size_t i = 0; i < vl; i++) {
        int m = ys[i] < 0.0f;
        ok &= f2u(out[i]) == f2u(m ? ys[i] + xs[i] : ys[i]);
        if (m) {
            if (first < 0)
                first = (int)i;
            count++;
        }
    }
    check("masked vfadd (mask undisturbed)", ok);
    check("vcpop.m", (int)__riscv_vcpop_m_b16(neg, vl) == count);
    check("vfirst.m", (int)__riscv_vfirst_m_b16(neg, vl) == first);
    vfloat32m2_t c = __riscv_vcompress_vm_f32m2(y, neg, vl);
    __riscv_vse32_v_f32m2(packed, c, (size_t)count);
    ok = 1;
    for (size_t i = 0, k = 0; i < vl; i++)
        if (ys[i] < 0.0f)
            ok &= f2u(packed[k++]) == f2u(ys[i]);
    check("vcompress.vm", ok);
    /* Merge the scalar 9 into the non-negative lanes. */
    vfloat32m2_t mg = __riscv_vfmerge_vfm_f32m2(y, 9.0f, __riscv_vmnot_m_b16(neg, vl), vl);
    __riscv_vse32_v_f32m2(out, mg, vl);
    ok = 1;
    for (size_t i = 0; i < vl; i++)
        ok &= out[i] == (ys[i] < 0.0f ? ys[i] : 9.0f);
    check("vfmerge.vfm + vmnot", ok);
}

/* Permutes and integer widening / narrowing. */
static void check_permutes(void) {
    uint32_t a[16], out[16];
    size_t vl = __riscv_vsetvl_e32m2(16);
    for (int i = 0; i < 16; i++)
        a[i] = (uint32_t)(i * 3 + 1);
    vuint32m2_t v = __riscv_vle32_v_u32m2(a, vl);
    __riscv_vse32_v_u32m2(out, __riscv_vslidedown_vx_u32m2(v, 3, vl), vl);
    int ok = 1;
    for (size_t i = 0; i < vl; i++)
        ok &= out[i] == (i + 3 < vl ? a[i + 3] : 0);
    check("vslidedown.vx", ok);
    __riscv_vse32_v_u32m2(out, __riscv_vslide1up_vx_u32m2(v, 99, vl), vl);
    ok = out[0] == 99;
    for (size_t i = 1; i < vl; i++)
        ok &= out[i] == a[i - 1];
    check("vslide1up.vx", ok);
    vuint32m2_t idx = __riscv_vrsub_vx_u32m2(__riscv_vid_v_u32m2(vl), (uint32_t)(vl - 1), vl);
    __riscv_vse32_v_u32m2(out, __riscv_vrgather_vv_u32m2(v, idx, vl), vl);
    ok = 1;
    for (size_t i = 0; i < vl; i++)
        ok &= out[i] == a[vl - 1 - i];
    check("vrgather.vv (reverse)", ok);

    int16_t p[16], q[16];
    int32_t w[16];
    int8_t nb[16];
    for (int i = 0; i < 16; i++) {
        p[i] = (int16_t)(hi16[i] * 3);
        q[i] = (int16_t)(-hi16[15 - i]);
    }
    vl = __riscv_vsetvl_e16m1(16);
    vint32m2_t prod = __riscv_vwmul_vv_i32m2(__riscv_vle16_v_i16m1(p, vl), __riscv_vle16_v_i16m1(q, vl), vl);
    __riscv_vse32_v_i32m2(w, prod, vl);
    vint8mf2_t clipped = __riscv_vnclip_wx_i8mf2(__riscv_vle16_v_i16m1(p, vl), 6, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_i8mf2(nb, clipped, vl);
    ok = 1;
    int ok2 = 1;
    for (size_t i = 0; i < vl; i++) {
        ok &= w[i] == (int32_t)p[i] * (int32_t)q[i];
        int32_t sh = ((int32_t)p[i] + 32) >> 6; /* round-to-nearest-up */
        int32_t sat = sh > 127 ? 127 : sh < -128 ? -128 : sh;
        ok2 &= nb[i] == sat;
    }
    check("vwmul.vv i16 -> i32", ok);
    check("vnclip.wx i16 -> i8 (rnu, LMUL 1/2)", ok2);
}

/* fp64 -> fp32 narrowing, fp16 (Zvfh) arithmetic and bf16 (Zvfbfmin / Zvfbfwma). */
static uint16_t bf16_rne(float f) {
    uint32_t u = f2u(f);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

static float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

static void check_formats(void) {
    float nf[N];
    for (size_t i = 0, l; i < N; i += l) {
        l = __riscv_vsetvl_e32m1(N - i);
        __riscv_vse32_v_f32m1(&nf[i], __riscv_vfncvt_f_f_w_f32m1(__riscv_vle64_v_f64m2((const double *)&xd[i], l), l), l);
    }
    int ok = 1;
    for (int i = 0; i < N; i++)
        ok &= f2u(nf[i]) == f2u((float)xd[i]);
    check("vfncvt.f.f.w f64 -> f32", ok);

    /* fp16: a + b directly, and via fp32 (exact widening, one fp32 add, one narrowing). */
    uint16_t ha[16], hb[16], direct[16], via[16];
    size_t vl = __riscv_vsetvl_e32m2(16);
    vfloat32m2_t fa = __riscv_vle32_v_f32m2((const float *)ys, vl);
    vfloat32m2_t fb = __riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2((const float *)xs, vl), 7.0f, vl);
    __riscv_vse16_v_u16m1(ha, __riscv_vreinterpret_v_f16m1_u16m1(__riscv_vfncvt_f_f_w_f16m1(fa, vl)), vl);
    __riscv_vse16_v_u16m1(hb, __riscv_vreinterpret_v_f16m1_u16m1(__riscv_vfncvt_f_f_w_f16m1(fb, vl)), vl);
    vfloat16m1_t va = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vle16_v_u16m1(ha, vl));
    vfloat16m1_t vb = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vle16_v_u16m1(hb, vl));
    __riscv_vse16_v_u16m1(direct, __riscv_vreinterpret_v_f16m1_u16m1(__riscv_vfadd_vv_f16m1(va, vb, vl)), vl);
    vfloat32m2_t sum32 = __riscv_vfadd_vv_f32m2(__riscv_vfwcvt_f_f_v_f32m2(va, vl), __riscv_vfwcvt_f_f_v_f32m2(vb, vl), vl);
    __riscv_vse16_v_u16m1(via, __riscv_vreinterpret_v_f16m1_u16m1(__riscv_vfncvt_f_f_w_f16m1(sum32, vl)), vl);
    ok = 1;
    for (size_t i = 0; i < vl; i++)
        ok &= direct[i] == via[i];
    check("vfadd.vv f16 (Zvfh) = f32 add rounded once to f16", ok);

    /* bf16: fp32 -> bf16 (RNE) and acc + a*b in fp32 (vfwmaccbf16, products exact in fp32). */
    uint16_t ba[16], bb[16];
    float acc[16], ref[16];
    vbfloat16m1_t vba = __riscv_vfncvtbf16_f_f_w_bf16m1(fa, vl);
    vbfloat16m1_t vbb = __riscv_vfncvtbf16_f_f_w_bf16m1(fb, vl);
    __riscv_vse16_v_bf16m1((__bf16 *)ba, vba, vl);
    __riscv_vse16_v_bf16m1((__bf16 *)bb, vbb, vl);
    ok = 1;
    for (size_t i = 0; i < vl; i++) {
        ok &= ba[i] == bf16_rne(ys[i]) && bb[i] == bf16_rne(xs[i] * 7.0f);
        ref[i] = fmaf_(bf16_to_f32(ba[i]), bf16_to_f32(bb[i]), zs[i]);
    }
    check("vfncvtbf16.f.f.w f32 -> bf16 (Zvfbfmin)", ok);
    vfloat32m2_t vacc = __riscv_vle32_v_f32m2((const float *)zs, vl);
    __riscv_vse32_v_f32m2(acc, __riscv_vfwmaccbf16_vv_f32m2(vacc, vba, vbb, vl), vl);
    ok = 1;
    for (size_t i = 0; i < vl; i++)
        ok &= f2u(acc[i]) == f2u(ref[i]);
    check("vfwmaccbf16.vv (Zvfbfwma)", ok);
}

int main(void) {
    init();
    naiad_printf("rvv_check: VLEN = %d bits\n", (int)(__riscv_vsetvlmax_e8m1() * 8));
    check_axpy_f32();
    check_axpy_f64();
    check_dot();
    check_gemm();
    check_reductions();
    check_memory();
    check_masks();
    check_permutes();
    check_formats();
    naiad_puts(failures ? "FAIL\n" : "PASS\n");
    return failures ? 1 : 0;
}
