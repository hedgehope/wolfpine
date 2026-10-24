/*
 * fft_rvv: the RVV baseline, a vectorised four-step FFT built from radix-4
 * (and one radix-2) Stockham passes in which every butterfly operand is a
 * whole row.
 *
 * With n = n1 n2 (n1 = FFT_R0, the AME plan's first radix, so the stage
 * twiddle table twr/twi starts with exactly the n1 x n2 matrix needed below)
 * and x viewed as n1 rows of n2 (x[a][b] = x[a n2 + b]):
 *
 *   1. Y[k1][b] = sum_a x[a][b] W_n1^(a k1): n2 independent length-n1 FFTs,
 *      vectorised across the n2 columns.
 *   2. Y[k1][b] *= W_n^(b k1) (an elementwise complex multiply by twr/twi),
 *      stored transposed, Z[b][k1]: one strided store per row, the only
 *      non-unit-stride access (2n elements per transform).
 *   3. X[k1 + n1 k2] = sum_b Z[b][k1] W_n2^(b k2): n1 FFTs of length n2,
 *      vectorised across the n1 columns; the last pass writes y, whose
 *      row-major order k2 n1 + k1 is the natural order.
 *
 * Each R-point FFT over rows is Stockham autosort (out of place, ping-pong,
 * natural order, no bit reversal): a radix-r pass with current span ns turns
 * rows j + q R/r (q < r) into rows (j / ns) r ns + k + m ns (m < r, k = j mod
 * ns) after twiddling input q by W_(r ns)^(q k). For a fixed j / ns the k
 * rows of every input and output are contiguous, so the loops just walk
 * blocks, and the twiddles are scalars per row (.vf forms). Radix 4 (log2 R
 * even) or one radix-2 pass then radix 4: the radix-4 butterfly does three
 * complex multiplies and a two-level add tree on 8 loaded vectors, plenty of
 * independent work for the in-order core, with 16 loads/stores per 28 ops
 * where two radix-2 passes would need 32 per 32.
 *
 * Arithmetic is in the accumulator format (fp32, or fp64 for fp64); 16-bit
 * inputs are widened by the first pass's adds (vfwadd / vfwsub for fp16;
 * bf16 widens and adds -1.0 x by vfwmaccbf16), and nothing is rounded back
 * to in_fmt between passes (the scalar baseline does the same). Twiddles are
 * the host's fp64 values rounded once to acc_fmt: W_n^e for e < n is taken
 * from w2 (W_n^k, k < n/2) by a sign flip, once, before the timed region (a
 * plan step, as an FFT library's twiddle table is). LMUL 2: 8 input vectors
 * plus temporaries fit the register file (16 fp32 / 8 fp64 lanes at VLEN 256).
 */
#include "../fft_io.h"

#define RV_AL 2
#include "rvv_fmt.h"

#if FFT_N & (FFT_N - 1)
#error "FFT_N must be a power of two"
#endif
#define N1 FFT_R0
#define N2 (FFT_N / FFT_R0)
#if N1 < 4 || N2 < 4
#error "fft_rvv needs n1, n2 >= 4"
#endif

static acc_t bufr[3][FFT_N], bufi[3][FFT_N]; /* ping-pong planes */
static acc_t wfr[FFT_N], wfi[FFT_N];          /* W_n^e, e < n */

/* A complex plane pair. */
typedef struct {
    acc_t *r, *i;
} cplane;

/* (xr, xi) *= (wr, wi), in place: xr wr - xi wi, xi wr + xr wi. */
#define CMUL(xr, xi, wr, wi, vl)                                                  \
    do {                                                                          \
        rv_acc_v t_r = __riscv_vfnmsac(__riscv_vfmul(xr, wr, vl), wi, xi, vl);    \
        xi = __riscv_vfmacc(__riscv_vfmul(xi, wr, vl), wi, xr, vl);               \
        xr = t_r;                                                                 \
    } while (0)

/*
 * The 4-point DFT of (x0..x3) stored to d + m * dm (y1 = b - i d, y3 = b + i d
 * with a = x0 + x2, b = x0 - x2, c = x1 + x3, d = x1 - x3). ADD2 / SUB2 form a
 * and b, c and d from the loaded inputs (widening for the first pass).
 */
#define DFT4_STORE(ADD2, SUB2, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, dr, di, dm, vl) \
    do {                                                                             \
        rv_acc_v a_r = ADD2(x0r, x2r, vl), a_i = ADD2(x0i, x2i, vl);                 \
        rv_acc_v b_r = SUB2(x0r, x2r, vl), b_i = SUB2(x0i, x2i, vl);                 \
        rv_acc_v c_r = ADD2(x1r, x3r, vl), c_i = ADD2(x1i, x3i, vl);                 \
        rv_acc_v d_r = SUB2(x1r, x3r, vl), d_i = SUB2(x1i, x3i, vl);                 \
        rv_store_acc((dr), __riscv_vfadd(a_r, c_r, vl), vl);                         \
        rv_store_acc((di), __riscv_vfadd(a_i, c_i, vl), vl);                         \
        rv_store_acc((dr) + 2 * (dm), __riscv_vfsub(a_r, c_r, vl), vl);              \
        rv_store_acc((di) + 2 * (dm), __riscv_vfsub(a_i, c_i, vl), vl);              \
        rv_store_acc((dr) + (dm), __riscv_vfadd(b_r, d_i, vl), vl);                  \
        rv_store_acc((di) + (dm), __riscv_vfsub(b_i, d_r, vl), vl);                  \
        rv_store_acc((dr) + 3 * (dm), __riscv_vfsub(b_r, d_i, vl), vl);              \
        rv_store_acc((di) + 3 * (dm), __riscv_vfadd(b_i, d_r, vl), vl);              \
    } while (0)
#define ACC_ADD(x, y, vl) __riscv_vfadd(x, y, vl)
#define ACC_SUB(x, y, vl) __riscv_vfsub(x, y, vl)

/* Load the four inputs of a radix-4 butterfly (rows s + q * sq) with LOAD, pinned. */
#define LOAD4(T, LOAD, sr, si, sq, vl)                                               \
    T x0r = LOAD((sr), vl), x0i = LOAD((si), vl);                                    \
    T x1r = LOAD((sr) + (sq), vl), x1i = LOAD((si) + (sq), vl);                      \
    T x2r = LOAD((sr) + 2 * (sq), vl), x2i = LOAD((si) + 2 * (sq), vl);              \
    T x3r = LOAD((sr) + 3 * (sq), vl), x3i = LOAD((si) + 3 * (sq), vl);              \
    RV_USE(x0r);                                                                     \
    RV_USE(x0i);                                                                     \
    RV_USE(x1r);                                                                     \
    RV_USE(x1i);                                                                     \
    RV_USE(x2r);                                                                     \
    RV_USE(x2i);                                                                     \
    RV_USE(x3r);                                                                     \
    RV_USE(x3i)

/* One radix-4 pass over R rows of W, span ns, from s to d. */
RV_INLINE void pass4(long R, long W, long ns, cplane s, cplane d) {
    const long sq = (R / 4) * W, dm = ns * W, tstep = FFT_N / (4 * ns);
    for (long g = 0; g < R / (4 * ns); g++) {
        for (long k = 0; k < ns; k++) {
            const acc_t w1r = wfr[k * tstep], w1i = wfi[k * tstep];
            const acc_t w2r_ = wfr[2 * k * tstep], w2i_ = wfi[2 * k * tstep];
            const acc_t w3r = wfr[3 * k * tstep], w3i = wfi[3 * k * tstep];
            const long so = (g * ns + k) * W, d0 = (g * 4 * ns + k) * W;
            for (long c = 0, vl; c < W; c += vl) {
                vl = (long)rv_setvl(W - c);
                LOAD4(rv_acc_v, rv_load_acc, s.r + so + c, s.i + so + c, sq, vl);
                if (k) {
                    CMUL(x1r, x1i, w1r, w1i, vl);
                    CMUL(x2r, x2i, w2r_, w2i_, vl);
                    CMUL(x3r, x3i, w3r, w3i, vl);
                }
                DFT4_STORE(ACC_ADD, ACC_SUB, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, d.r + d0 + c, d.i + d0 + c, dm,
                           vl);
            }
        }
    }
}

/* A radix-2 pass with ns = 1 (no twiddles) over R rows of W, from s to d; LOAD / T / ADD2 / SUB2 as above. */
#define PASS2_FIRST(T, LOAD, ADD2, SUB2, R, W, sr, si, d)                                                   \
    for (long j = 0; j < (R) / 2; j++) {                                                                    \
        const long a_ = j * (W), b_ = (j + (R) / 2) * (W), o_ = 2 * j * (W);                                \
        for (long c = 0, vl; c < (W); c += vl) {                                                            \
            vl = (long)rv_setvl((W) - c);                                                                   \
            T xr = LOAD((sr) + a_ + c, vl), xi = LOAD((si) + a_ + c, vl);                                   \
            T yr = LOAD((sr) + b_ + c, vl), yi = LOAD((si) + b_ + c, vl);                                   \
            RV_USE(xr);                                                                                     \
            RV_USE(xi);                                                                                     \
            RV_USE(yr);                                                                                     \
            RV_USE(yi);                                                                                     \
            rv_store_acc((d).r + o_ + c, ADD2(xr, yr, vl), vl);                                             \
            rv_store_acc((d).i + o_ + c, ADD2(xi, yi, vl), vl);                                             \
            rv_store_acc((d).r + o_ + (W) + c, SUB2(xr, yr, vl), vl);                                       \
            rv_store_acc((d).i + o_ + (W) + c, SUB2(xi, yi, vl), vl);                                       \
        }                                                                                                   \
    }

/* A radix-4 pass with ns = 1 (no twiddles) over R rows of W, from s to d. */
#define PASS4_FIRST(T, LOAD, ADD2, SUB2, R, W, sr, si, d)                                                   \
    for (long j = 0; j < (R) / 4; j++) {                                                                    \
        const long a_ = j * (W), o_ = 4 * j * (W), sq_ = ((R) / 4) * (W);                                   \
        for (long c = 0, vl; c < (W); c += vl) {                                                            \
            vl = (long)rv_setvl((W) - c);                                                                   \
            LOAD4(T, LOAD, (sr) + a_ + c, (si) + a_ + c, sq_, vl);                                          \
            DFT4_STORE(ADD2, SUB2, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, (d).r + o_ + c, (d).i + o_ + c, (W), \
                       vl);                                                                                 \
        }                                                                                                   \
    }

/* Step 2: Z[b][k1] = Y[k1][b] W_n^(b k1) (twr/twi start with W_n^(k1 b) at [k1 n2 + b]). */
RV_INLINE void twiddle_transpose(cplane s, cplane d) {
    for (long k1 = 0; k1 < N1; k1++) {
        for (long c = 0, vl; c < N2; c += vl) {
            vl = (long)rv_setvl(N2 - c);
            rv_acc_v yr_ = rv_load_acc(s.r + k1 * N2 + c, vl), yi_ = rv_load_acc(s.i + k1 * N2 + c, vl);
            rv_acc_v wr = rv_load_acc(twr + k1 * N2 + c, vl), wi = rv_load_acc(twi + k1 * N2 + c, vl);
            RV_USE(yr_);
            RV_USE(yi_);
            RV_USE(wr);
            RV_USE(wi);
            rv_acc_v tr = __riscv_vfnmsac(__riscv_vfmul(yr_, wr, vl), wi, yi_, vl);
            rv_acc_v ti = __riscv_vfmacc(__riscv_vfmul(yi_, wr, vl), wi, yr_, vl);
            rv_store_acc_strided(d.r + c * N1 + k1, (ptrdiff_t)(N1 * sizeof(acc_t)), tr, vl);
            rv_store_acc_strided(d.i + c * N1 + k1, (ptrdiff_t)(N1 * sizeof(acc_t)), ti, vl);
        }
    }
}

/*
 * The radix-4 passes ns = ns0, 4 ns0, ... < R over rows of W, from cur, the
 * last one writing fin, with spare as scratch (all distinct): each pass goes to
 * fin when an odd number of passes remain. Returns the plane holding the result.
 */
RV_INLINE cplane rest4(long R, long W, long ns0, cplane cur, cplane spare, cplane fin) {
    long m = 0;
    for (long ns = ns0; ns < R; ns *= 4) {
        m++;
    }
    cplane src = cur;
    for (long ns = ns0; ns < R; ns *= 4, m--) {
        cplane dst = (m & 1) ? fin : (src.r == spare.r ? cur : spare);
        pass4(R, W, ns, src, dst);
        src = dst;
    }
    return src;
}

#define LG2_ODD(x) (((x) & 0xAAAAAAAAL) != 0) /* log2 of a power of two is odd */
#define FIRST_RADIX(R) (LG2_ODD(R) ? 2 : 4)
#define ACC_LOAD(p, vl) rv_load_acc((p), (vl))
#define IN_LOAD(p, vl) rv_load_in((p), (vl))

RV_KERNEL int main(void) {
    for (long e = 0; e < FFT_N / 2; e++) { /* the plan: W_n^e for e < n, from W_n^k (k < n/2) */
        wfr[e] = w2r[e];
        wfi[e] = w2i[e];
        wfr[e + FFT_N / 2] = -w2r[e];
        wfi[e + FFT_N / 2] = -w2i[e];
    }
    const cplane A = {bufr[0], bufi[0]}, B = {bufr[1], bufi[1]}, C = {bufr[2], bufi[2]};
    uint64_t start = naiad_cycles();
    for (int t = 0; t < FFT_BATCH; t++) {
        const in_t *xr_t = xr + (long)t * FFT_N, *xi_t = xi + (long)t * FFT_N;
        const cplane Y = {yr + (long)t * FFT_N, yi + (long)t * FFT_N};
        /* step 1: the first pass from the input into A, the rest ending in C */
        if (FIRST_RADIX(N1) == 2) {
            PASS2_FIRST(rv_in_v, IN_LOAD, rv_add_in2, rv_sub_in2, N1, N2, xr_t, xi_t, A)
        } else {
            PASS4_FIRST(rv_in_v, IN_LOAD, rv_add_in2, rv_sub_in2, N1, N2, xr_t, xi_t, A)
        }
        const cplane y1 = rest4(N1, N2, FIRST_RADIX(N1), A, B, C);
        /* step 2: Z into whichever of A / C step 1 did not end in */
        const cplane Z = y1.r == A.r ? C : A;
        twiddle_transpose(y1, Z);
        /* step 3: n2-point FFTs over rows of n1, the first pass Z -> B (or y), the rest ending in y */
        const cplane Z1 = FIRST_RADIX(N2) < N2 ? B : Y;
        if (FIRST_RADIX(N2) == 2) {
            PASS2_FIRST(rv_acc_v, ACC_LOAD, ACC_ADD, ACC_SUB, N2, N1, Z.r, Z.i, Z1)
        } else {
            PASS4_FIRST(rv_acc_v, ACC_LOAD, ACC_ADD, ACC_SUB, N2, N1, Z.r, Z.i, Z1)
        }
        rest4(N2, N1, FIRST_RADIX(N2), Z1, Z, Y);
    }
    region_cycles = naiad_cycles() - start;
    return 0;
}
