/*
 * nbody_rvv: the RVV baseline, the direct all-pairs method of nbody_scalar
 * (same pair law, unit masses, every ordered pair, r^2 from coordinate
 * differences, so no expansion and no cancellation beyond the subtraction).
 *
 * Vectorised over i: a block of VL particles' coordinates and force
 * accumulators sits in vector registers (LMUL 2: 16 fp32 / 8 fp64 lanes at
 * VLEN 256) and every j is a scalar (.vf forms), so forces accumulate in
 * registers with no reductions. Positions are first gathered into
 * structure-of-arrays planes in acc_fmt (three strided loads per block,
 * widening 16-bit inputs), inside the timed region as the scalar variant's
 * widening copy is.
 *
 * r^2 = (dx dx + dy dy) + dz dz, fused as in the scalar loop, with
 * dx = x_j - x_i. The d2 output (n <= 256) is stored by rows of j: r^2 of the
 * pair (i, j) computed as (x_j - x_i)^2 + ... is bit-identical to the (j, i)
 * pair's (negation is exact), so d2[j][i0 .. i0 + VL) is one unit-stride store.
 * The diagonal needs no mask for softened gravity: dx = dy = dz = 0 there and
 * the weight is finite (eps^2 > 0), so it adds exactly 0. Lennard-Jones
 * (-DNB_LJ=1) masks the diagonal and the cut-off with a merge.
 *
 * The softened weight w = (r^2 + eps^2)^(-3/2). Default (NB_RVV_EXACT=0):
 * y = vfrsqrt7(s), refined by Newton steps y *= 1.5 - (s / 2) y^2 (two for fp32,
 * four for fp64: each about doubles the 7 correct bits), then w = y^3. That is
 * about 9-13 pipelined ops against vfsqrt + vfdiv, which occupy the datapath
 * div_ii = 8 times longer per element group in the timing model (and are slow,
 * usually unpipelined, in hardware). Accuracy: the Newton result is within a few
 * ulp of the correctly rounded rsqrt, so w carries a few ulp of acc_fmt more
 * error than the scalar loop's sqrt and divide; NB_RVV_EXACT=1 uses
 * vfsqrt + vfmul + vfrdiv, the scalar loop's arithmetic.
 */
#include "../nbody_io.h"

#define RV_AL 2
#include "rvv_fmt.h"

#ifndef NB_RVV_EXACT
#define NB_RVV_EXACT 0
#endif
#if RV_AE == 64
#define NB_NEWTON 4
#else
#define NB_NEWTON 2
#endif

#define N NB_N
/* Mask type bits for LMUL 2: SEW / LMUL. */
#if RV_AE == 64
#define RV_MASKBITS 32
#else
#define RV_MASKBITS 16
#endif

static acc_t PX[N], PY[N], PZ[N];

/* w(r^2) for every lane (the soft law; the LJ law is in the loop). */
RV_INLINE rv_acc_v soft_weight(rv_acc_v r2, rv_acc_v onehalf, size_t vl) {
    rv_acc_v s = __riscv_vfadd(r2, params[0], vl);
#if NB_RVV_EXACT
    return __riscv_vfrdiv(__riscv_vfmul(s, __riscv_vfsqrt(s, vl), vl), (rv_acc_s)1, vl);
#else
    rv_acc_v h = __riscv_vfmul(s, (rv_acc_s)0.5, vl), y = __riscv_vfrsqrt7(s, vl);
    for (int it = 0; it < NB_NEWTON; it++) {
        rv_acc_v t = __riscv_vfmul(y, y, vl);
        t = __riscv_vfnmsub(t, h, onehalf, vl); /* 1.5 - h y^2 */
        y = __riscv_vfmul(y, t, vl);
    }
    return __riscv_vfmul(__riscv_vfmul(y, y, vl), y, vl);
#endif
}

#if RV_AE == 64
typedef vuint64m2_t rv_idx_v;
#define rv_vid __riscv_vid_v_u64m2
#else
typedef vuint32m2_t rv_idx_v;
#define rv_vid __riscv_vid_v_u32m2
#endif

/*
 * The Lennard-Jones weight, -24 eps inv s6 (2 s6 - 1) with inv = 1 / r^2,
 * s6 = (sigma^2 inv)^3, in the scalar loop's order; 0 at or past the cut-off
 * and in lane jl (the diagonal, where r^2 = 0), merged after the arithmetic.
 */
RV_INLINE rv_acc_v lj_weight(rv_acc_v r2, rv_idx_v vid, long jl, size_t vl) {
    rv_acc_v inv = __riscv_vfrdiv(r2, (rv_acc_s)1, vl);
    rv_acc_v s2 = __riscv_vfmul(inv, params[1], vl);
    rv_acc_v s6 = __riscv_vfmul(__riscv_vfmul(s2, s2, vl), s2, vl);
    rv_acc_v w = __riscv_vfmul(__riscv_vfmul(inv, -params[2], vl), s6, vl);
    w = __riscv_vfmul(w, __riscv_vfsub(__riscv_vfmul(s6, (rv_acc_s)2, vl), (rv_acc_s)1, vl), vl);
    RV_C(vbool, RV_C(RV_MASKBITS, _t)) off = __riscv_vmor(__riscv_vmfge(r2, params[3], vl),
                                                          __riscv_vmseq(vid, (size_t)jl, vl), vl);
    return __riscv_vfmerge(w, (rv_acc_s)0, off, vl);
}

RV_KERNEL int main(void) {
    uint64_t start = naiad_cycles();
    for (long i = 0, vl; i < N; i += vl) { /* positions into SoA planes, in acc_fmt */
        vl = (long)rv_setvl(N - i);
        const ptrdiff_t st = 3 * sizeof(in_t);
        rv_store_acc(PX + i, rv_widen(rv_load_in_strided(pos + 3 * i, st, vl), vl), vl);
        rv_store_acc(PY + i, rv_widen(rv_load_in_strided(pos + 3 * i + 1, st, vl), vl), vl);
        rv_store_acc(PZ + i, rv_widen(rv_load_in_strided(pos + 3 * i + 2, st, vl), vl), vl);
    }
    for (long i0 = 0, vl; i0 < N; i0 += vl) {
        vl = (long)rv_setvl(N - i0);
        const rv_acc_v xi = rv_load_acc(PX + i0, vl), yi = rv_load_acc(PY + i0, vl), zi = rv_load_acc(PZ + i0, vl);
        const rv_acc_v onehalf = rv_splat((rv_acc_s)1.5, vl);
        rv_acc_v fx = rv_splat(0, vl), fy = fx, fz = fx;
#if NB_LJ
        const rv_idx_v vid = rv_vid(vl);
#endif
        long j = 0;
#if !NB_LJ && !NB_RVV_EXACT
        /*
         * Two j at a time, their chains interleaved step by step (the Newton
         * steps pinned: the compiler would otherwise serialise them), so each op
         * hides the other chain's latency. xi / yi / zi are reloaded per pair (the load unit is
         * idle here) to free their registers for the second chain.
         */
        for (; j + 2 <= N; j += 2) {
            const rv_acc_s xa = PX[j], ya = PY[j], za = PZ[j], xb = PX[j + 1], yb = PY[j + 1], zb = PZ[j + 1];
            rv_acc_v pxi = rv_load_acc(PX + i0, vl);
            RV_USE(pxi);
            rv_acc_v dxa = __riscv_vfrsub(pxi, xa, vl), dxb = __riscv_vfrsub(pxi, xb, vl);
            rv_acc_v pyi = rv_load_acc(PY + i0, vl);
            RV_USE(pyi);
            rv_acc_v dya = __riscv_vfrsub(pyi, ya, vl), dyb = __riscv_vfrsub(pyi, yb, vl);
            rv_acc_v pzi = rv_load_acc(PZ + i0, vl);
            RV_USE(pzi);
            rv_acc_v dza = __riscv_vfrsub(pzi, za, vl), dzb = __riscv_vfrsub(pzi, zb, vl);
            rv_acc_v ra = __riscv_vfmul(dxa, dxa, vl), rb = __riscv_vfmul(dxb, dxb, vl);
            ra = __riscv_vfmacc(ra, dya, dya, vl);
            rb = __riscv_vfmacc(rb, dyb, dyb, vl);
            ra = __riscv_vfmacc(ra, dza, dza, vl);
            rb = __riscv_vfmacc(rb, dzb, dzb, vl);
#if NB_SAVE_D2
            rv_store_acc(d2 + j * N + i0, ra, vl);
            rv_store_acc(d2 + (j + 1) * N + i0, rb, vl);
#endif
            ra = __riscv_vfadd(ra, params[0], vl); /* s */
            rb = __riscv_vfadd(rb, params[0], vl);
            rv_acc_v ha = __riscv_vfmul(ra, (rv_acc_s)0.5, vl), hb = __riscv_vfmul(rb, (rv_acc_s)0.5, vl);
            rv_acc_v ya_ = __riscv_vfrsqrt7(ra, vl), yb_ = __riscv_vfrsqrt7(rb, vl);
            for (int it = 0; it < NB_NEWTON; it++) {
                RV_PIN(ya_);
                RV_PIN(yb_);
                rv_acc_v ta = __riscv_vfmul(ya_, ya_, vl), tb = __riscv_vfmul(yb_, yb_, vl);
                ta = __riscv_vfnmsub(ta, ha, onehalf, vl);
                tb = __riscv_vfnmsub(tb, hb, onehalf, vl);
                ya_ = __riscv_vfmul(ya_, ta, vl);
                yb_ = __riscv_vfmul(yb_, tb, vl);
            }
            rv_acc_v wa = __riscv_vfmul(ya_, ya_, vl), wb = __riscv_vfmul(yb_, yb_, vl);
            wa = __riscv_vfmul(wa, ya_, vl);
            wb = __riscv_vfmul(wb, yb_, vl);
            fx = __riscv_vfmacc(fx, wa, dxa, vl);
            fy = __riscv_vfmacc(fy, wa, dya, vl);
            fz = __riscv_vfmacc(fz, wa, dza, vl);
            fx = __riscv_vfmacc(fx, wb, dxb, vl);
            fy = __riscv_vfmacc(fy, wb, dyb, vl);
            fz = __riscv_vfmacc(fz, wb, dzb, vl);
        }
#endif
        for (; j < N; j++) {
            const rv_acc_s xj = PX[j], yj = PY[j], zj = PZ[j];
            rv_acc_v dx = __riscv_vfrsub(xi, xj, vl), dy = __riscv_vfrsub(yi, yj, vl), dz = __riscv_vfrsub(zi, zj, vl);
            rv_acc_v r2 = __riscv_vfmacc(__riscv_vfmacc(__riscv_vfmul(dx, dx, vl), dy, dy, vl), dz, dz, vl);
#if NB_SAVE_D2
            rv_store_acc(d2 + j * N + i0, r2, vl);
#endif
#if NB_LJ
            rv_acc_v w = lj_weight(r2, vid, j - i0, vl);
#else
            rv_acc_v w = soft_weight(r2, onehalf, vl);
#endif
            fx = __riscv_vfmacc(fx, w, dx, vl);
            fy = __riscv_vfmacc(fy, w, dy, vl);
            fz = __riscv_vfmacc(fz, w, dz, vl);
        }
        const ptrdiff_t st = 3 * sizeof(acc_t);
        rv_store_acc_strided(force + 3 * i0, st, fx, vl);
        rv_store_acc_strided(force + 3 * i0 + 1, st, fy, vl);
        rv_store_acc_strided(force + 3 * i0 + 2, st, fz, vl);
    }
    region_cycles = naiad_cycles() - start;
    return 0;
}
