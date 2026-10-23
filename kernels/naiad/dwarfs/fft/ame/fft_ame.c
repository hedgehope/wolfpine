/*
 * fft_ame: batched complex FFT with the DFT-16 / DFT-32 blocks on the matrix unit.
 *
 * Formulation (Stockham autosort, so the result lands in natural order with
 * no separate permutation). Before stage s the data is B = r_0 ... r_{s-1}
 * independent sub-problems of length m = N / B, each contiguous; view one as
 * an r x m' matrix (m' = m / r), x[b][n1][j] at b*m + n1*m' + j. The stage is
 *
 *   y[k][b][j] = sum_n1 F_r[k][n1] x[b][n1][j]            (matrix unit)
 *   y[k][b][j] *= W_m^(k j)        (all but the last stage; scalar side)
 *
 * stored at k*B*m' + b*m' + j, which is the next stage's layout with block
 * b' = k*B + b. F_r is the "a" operand (M = k, K = n1); the data is "b"
 * (K = n1, N = j). When m' >= 16 a b tile is 16 n1-rows of 16 j-columns
 * (row stride m'); at the last stage m' = 1, so the columns are blocks and a
 * tile is loaded transposed (tload.b, t=1: memory holds 16 rows of 16 n1).
 *
 * Complex arithmetic on a real unit. Default: four real mma per tile pair,
 * as the real-block form [Re -Im; Im Re]:
 *   acc_re = Fr*Xr + (-Fi)*Xi,  acc_im = Fi*Xr + Fr*Xi
 * Both sums stay inside the accumulator (one rounding at the end, no
 * accumulator-to-accumulator op needed), and -Fi is exact, so the error is
 * that of an ordinary fp GEMM. -DFFT_GAUSS=1 switches to the 3-mma Gauss
 * form, T1 = Fr*Xr, T2 = Fi*Xi, T3 = (Fr+Fi)*(Xr+Xi), re = T1 - T2,
 * im = T3 - T1 - T2: 25% fewer MACs, but it needs a scalar pass to form
 * Xr + Xi (one more rounding to in_fmt), a third accumulator plane, and
 * the combine cancels, which hurts most at reduced precision.
 *
 * Handoff: the matrix unit writes acc_fmt planes (yr, yi); the scalar pass
 * multiplies by the twiddles in acc_fmt and rounds back to in_fmt into the
 * working planes (xr, xi) for the next stage's tile loads. Twiddles and the
 * DFT matrices come from the host (fp64, rounded once to their format).
 */
#include "../fft_io.h"
#include "matrix.h"
#include "dwarf_util.h"

#define ES DW_IN_BYTES
#define EA DW_ACC_BYTES
#define FI DW_IN_FMT
#define FA DW_ACC_FMT

#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16
#error "fft_ame is written for a 16x16x16 unit (DFT-16 blocks)"
#endif
#if NAIAD_MATRIX_TILES < 6 || NAIAD_MATRIX_ACCS < 3
#error "fft_ame needs 6 tiles and 3 accumulators"
#endif
#ifndef FFT_GAUSS
#define FFT_GAUSS 0
#endif
#if FFT_GAUSS
/* The Gauss form's extra planes: Xr + Xi (in_fmt) and T3 (acc_fmt). */
static in_t xs[FFT_TOT];
static acc_t y3[FFT_TOT];
#endif

#ifndef FFT_R1
#define FFT_R1 0
#endif
#ifndef FFT_R2
#define FFT_R2 0
#endif
static const int radix[3] = {FFT_R0, FFT_R1, FFT_R2};

/* Tiles: 0 Xr, 1 Xi, 2 Xs | 3 Fr, 4 Fi, 5 -Fi (or Fr+Fi). Accs: 0 re (T1), 1 im (T2), 2 T3. */
#define T_XR 0
#define T_XI 1
#define T_XS 2
#define T_FR 3
#define T_FI 4
#define T_FN 5

DW_INLINE void load_x(int t, const in_t *p, long stride_b, int transposed) {
    if (transposed) {
        switch (t) {
        case T_XR: mx_tload_bt(T_XR, FI, p, stride_b); break;
        case T_XI: mx_tload_bt(T_XI, FI, p, stride_b); break;
        default: mx_tload_bt(T_XS, FI, p, stride_b); break;
        }
    } else {
        switch (t) {
        case T_XR: mx_tload_b(T_XR, FI, p, stride_b); break;
        case T_XI: mx_tload_b(T_XI, FI, p, stride_b); break;
        default: mx_tload_b(T_XS, FI, p, stride_b); break;
        }
    }
}

DW_INLINE void load_f(int R, int kb, int kk) {
    const in_t *fr = R == 16 ? f16r : f32r, *fi = R == 16 ? f16i : f32i;
    const in_t *fn = R == 16 ? (FFT_GAUSS ? f16s : f16n) : (FFT_GAUSS ? f32s : f32n);
    long off = 16L * kb * R + 16L * kk;
    mx_tload_a(T_FR, FI, fr + off, R * ES);
    mx_tload_a(T_FI, FI, fi + off, R * ES);
    mx_tload_a(T_FN, FI, fn + off, R * ES);
}

/*
 * One 16-column slab of a stage: x0 points at (n1 = 0, column 0) of each
 * plane-relative offset `xo`, consecutive 16-row K blocks are kstep elements
 * apart, rows xstride elements apart; the output's first row is at `yo`,
 * rows ystride elements apart, the 16-row k blocks krow elements apart.
 */
DW_INLINE void slab(int R, int t, long xo, long kstep, long xstride, int transposed, long yo, long ystride,
                    long krow) {
    const long base = (long)t * FFT_N;
    for (int kb = 0; kb < R / 16; kb++) {
        mx_azero(0);
        mx_azero(1);
#if FFT_GAUSS
        mx_azero(2);
#endif
        for (int kk = 0; kk < R / 16; kk++) {
            long o = base + xo + kk * kstep;
            load_x(T_XR, xr + o, xstride * ES, transposed);
            load_x(T_XI, xi + o, xstride * ES, transposed);
#if FFT_GAUSS
            load_x(T_XS, xs + o, xstride * ES, transposed);
#endif
            if (R != 16) {
                load_f(R, kb, kk);
            }
#if FFT_GAUSS
            mx_mma(0, T_FR, T_XR, FI, FA);
            mx_mma(1, T_FI, T_XI, FI, FA);
            mx_mma(2, T_FN, T_XS, FI, FA);
#else
            mx_mma(0, T_FR, T_XR, FI, FA);
            mx_mma(0, T_FN, T_XI, FI, FA);
            mx_mma(1, T_FI, T_XR, FI, FA);
            mx_mma(1, T_FR, T_XI, FI, FA);
#endif
        }
        long o = base + yo + kb * krow;
        mx_astore(0, FA, yr + o, ystride * EA);
        mx_astore(1, FA, yi + o, ystride * EA);
#if FFT_GAUSS
        mx_astore(2, FA, y3 + o, ystride * EA);
#endif
    }
}

DW_INLINE void stage(int R, int B, int mp) {
    const long m = (long)R * mp;
    if (R == 16) {
        load_f(16, 0, 0); /* resident for the whole stage */
    }
    for (int t = 0; t < FFT_BATCH; t++) {
        if (mp >= 16) {
            for (int b = 0; b < B; b++) {
                for (int j0 = 0; j0 < mp; j0 += 16) {
                    slab(R, t, b * m + j0, 16L * mp, mp, 0, (long)b * mp + j0, (long)B * mp, 16L * B * mp);
                }
            }
        } else { /* mp == 1: columns are blocks, loaded transposed */
            for (int b0 = 0; b0 < B; b0 += 16) {
                slab(R, t, (long)b0 * R, 16, R, 1, b0, B, 16L * B);
            }
        }
    }
}

static void stage16(int B, int mp) { stage(16, B, mp); }
static void stage32(int B, int mp) { stage(32, B, mp); }

/* y (acc_fmt) * twiddle -> x (in_fmt), for a non-final stage's output. */
static void twiddle(int R, int B, int mp, const acc_t *wr, const acc_t *wi) {
    for (int t = 0; t < FFT_BATCH; t++) {
        long o = (long)t * FFT_N;
        for (int k = 0; k < R; k++) {
            for (int b = 0; b < B; b++) {
                for (int j = 0; j < mp; j++, o++) {
#if FFT_GAUSS
                    real t1 = y3[o] - yr[o] - yi[o];
                    real ar = yr[o] - yi[o], ai = t1;
#else
                    real ar = yr[o], ai = yi[o];
#endif
                    real c = wr[k * mp + j], s = wi[k * mp + j];
                    real pr = ar * c - ai * s, pi = ar * s + ai * c;
                    xr[o] = real_to_in(pr);
                    xi[o] = real_to_in(pi);
#if FFT_GAUSS
                    xs[o] = real_to_in(pr + pi);
#endif
                }
            }
        }
    }
}

int main(void) {
    uint64_t start = naiad_cycles();
#if FFT_GAUSS
    for (long o = 0; o < FFT_TOT; o++) {
        xs[o] = real_to_in(in_to_real(xr[o]) + in_to_real(xi[o]));
    }
#endif
    int B = 1;
    long off = 0;
    for (int s = 0; s < FFT_S; s++) {
        int R = radix[s], mp = FFT_N / (B * R);
        if (R == 16) {
            stage16(B, mp);
        } else {
            stage32(B, mp);
        }
        if (s < FFT_S - 1) {
            twiddle(R, B, mp, twr + off, twi + off);
            off += (long)R * mp;
        }
        B *= R;
    }
#if FFT_GAUSS
    for (long o = 0; o < FFT_TOT; o++) {
        real t1 = yr[o], t2 = yi[o];
        yr[o] = t1 - t2;
        yi[o] = y3[o] - t1 - t2;
    }
#endif
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - start;
    return 0;
}
