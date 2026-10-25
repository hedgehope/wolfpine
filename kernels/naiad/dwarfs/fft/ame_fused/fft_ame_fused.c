/*
 * fft_ame_fused: fft_ame with the inter-stage twiddles applied inside the matrix unit.
 *
 * The plan, the DFT-16 / DFT-32 blocks and the real-block complex mma are
 * fft_ame's (../ame/fft_ame.c, whose header explains the Stockham layout).
 * What changes is the handoff between stages. fft_ame stores each stage's
 * output to acc_fmt planes and a scalar pass multiplies by the twiddles and
 * rounds back to in_fmt; here each 16 x 16 output block is twiddled while it
 * is still in the accumulators (re in acc 0, im in acc 1), with the twiddle
 * block W = C + iS loaded as two acc_fmt tiles (feature ewise):
 *
 *   acc2  = acc0 * S          aew.at mul
 *   acc2 += acc1 * C          amacc.at        (im' = re S + im C, fused)
 *   acc0  = acc0 * C          aew.at mul
 *   acc0 -= acc1 * S          amacc.at neg    (re' = re C - im S, fused)
 *
 * -DFFT_EW_FMA=0 uses separate multiplies and an add (six ops, one more
 * rounding per output, and accumulator 3 as a temporary): the fused-multiply-
 * add ablation. The result goes straight to the next stage's in_fmt planes:
 * an astore when in_fmt = acc_fmt; otherwise amov.t.a + tconv + tstore.a
 * (features acc_move, convert), or, without them, an astore to acc_fmt
 * scratch and a scalar rounding pass (the conversion ablation). No scalar
 * arithmetic touches the data. The final stage astores acc_fmt to (yr, yi).
 *
 * Stage s reads one pair of in_fmt planes and writes another: x -> p -> x ...,
 * the last into y. Twiddle tiles depend only on (k block, column block), so
 * they are loaded once per (kb, j0) and reused across the batch and the
 * sub-problems; the twiddle values are fft_ame's (twr, twi in acc_fmt).
 */
#include "../fft_io.h"
#include "matrix.h"
#include "dwarf_util.h"

#define ES DW_IN_BYTES
#define EA DW_ACC_BYTES
#define FI DW_IN_FMT
#define FA DW_ACC_FMT
#define NARROW (DW_IN_FMT != DW_ACC_FMT)

#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16
#error "fft_ame_fused is written for a 16x16x16 unit (DFT-16 blocks)"
#endif
#if NAIAD_MATRIX_TILES < 8 || NAIAD_MATRIX_ACCS < 3
#error "fft_ame_fused needs 8 tiles and 3 accumulators"
#endif
#ifndef NAIAD_MATRIX_FEATURE_EWISE
#error "fft_ame_fused needs the matrix unit's ewise feature (matrix.features: [..., ewise])"
#endif
#ifndef NAIAD_MATRIX_FEATURE_ACC_MEM
#error "fft_ame_fused needs aload/astore (matrix.acc_mem)"
#endif
#ifndef FFT_EW_FMA
#define FFT_EW_FMA 1
#endif
#if !FFT_EW_FMA && NAIAD_MATRIX_ACCS < 4
#error "-DFFT_EW_FMA=0 needs a fourth accumulator"
#endif
/* The narrowing store: on the unit when it can move and convert, else in software. */
#if NARROW && defined(NAIAD_MATRIX_FEATURE_ACC_MOVE) && defined(NAIAD_MATRIX_FEATURE_CONVERT)
#define FFT_UNIT_CONVERT 1
#else
#define FFT_UNIT_CONVERT 0
#endif
#define FFT_SW_CONVERT (NARROW && !FFT_UNIT_CONVERT)

#ifndef FFT_R1
#define FFT_R1 0
#endif
#ifndef FFT_R2
#define FFT_R2 0
#endif
static const int radix[3] = {FFT_R0, FFT_R1, FFT_R2};

/* The second pair of in_fmt working planes (the first is xr, xi). */
static in_t pr[FFT_TOT];
static in_t pi[FFT_TOT];

/* Tiles: 0 Xr, 1 Xi, 2 the narrowing store's staging tile | 3 Fr, 4 Fi, 5 -Fi | 6 C, 7 S. */
#define T_XR 0
#define T_XI 1
#define T_ST 2
#define T_FR 3
#define T_FI 4
#define T_FN 5
#define T_C 6
#define T_S 7

DW_INLINE void load_x(int t, const in_t *p, long stride_b, int transposed) {
    if (transposed) {
        if (t == T_XR) {
            mx_tload_bt(T_XR, FI, p, stride_b);
        } else {
            mx_tload_bt(T_XI, FI, p, stride_b);
        }
    } else {
        if (t == T_XR) {
            mx_tload_b(T_XR, FI, p, stride_b);
        } else {
            mx_tload_b(T_XI, FI, p, stride_b);
        }
    }
}

DW_INLINE void load_f(int R, int kb, int kk) {
    const in_t *fr = R == 16 ? f16r : f32r, *fi = R == 16 ? f16i : f32i, *fn = R == 16 ? f16n : f32n;
    long off = 16L * kb * R + 16L * kk;
    mx_tload_a(T_FR, FI, fr + off, R * ES);
    mx_tload_a(T_FI, FI, fi + off, R * ES);
    mx_tload_a(T_FN, FI, fn + off, R * ES);
}

/* acc0 + i acc1 = the 16 x 16 output block kb of F_R x (the slab at offset o of planes sr, si). */
DW_INLINE void dft_block(int R, int kb, const in_t *sr, const in_t *si, long o, long kstep, long xstride,
                         int transposed) {
    mx_azero(0);
    mx_azero(1);
    for (int kk = 0; kk < R / 16; kk++) {
        long oo = o + kk * kstep;
        load_x(T_XR, sr + oo, xstride * ES, transposed);
        load_x(T_XI, si + oo, xstride * ES, transposed);
        if (R != 16) {
            load_f(R, kb, kk);
        }
        mx_mma(0, T_FR, T_XR, FI, FA);
        mx_mma(0, T_FN, T_XI, FI, FA);
        mx_mma(1, T_FI, T_XR, FI, FA);
        mx_mma(1, T_FR, T_XI, FI, FA);
    }
}

/* (acc0 + i acc1) * (C + i S): re' in acc0, im' in acc2. */
DW_INLINE void twiddle_block(void) {
#if FFT_EW_FMA
    mx_aew_at(2, 0, T_S, FA, MX_EW_MUL);
    mx_amacc_at(2, 1, T_C, FA, 0);
    mx_aew_at(0, 0, T_C, FA, MX_EW_MUL);
    mx_amacc_at(0, 1, T_S, FA, 1);
#else
    mx_aew_at(2, 0, T_S, FA, MX_EW_MUL);
    mx_aew_at(3, 1, T_C, FA, MX_EW_MUL);
    mx_aew_aa(2, 2, 3, FA, MX_EW_ADD);
    mx_aew_at(0, 0, T_C, FA, MX_EW_MUL);
    mx_aew_at(3, 1, T_S, FA, MX_EW_MUL);
    mx_aew_aa(0, 0, 3, FA, MX_EW_SUB);
#endif
}

/* The twiddled block (acc0, acc2) to the next stage's planes at element offset o. */
DW_INLINE void store_next(in_t *dr, in_t *di, long o, long stride) {
#if FFT_UNIT_CONVERT
    mx_amov_ta(T_ST, 0, FA, 0);
    mx_tconv(T_ST, T_ST, 0, FA, FI);
    mx_tstore_a(T_ST, FI, dr + o, stride * ES);
    mx_amov_ta(T_ST, 2, FA, 0);
    mx_tconv(T_ST, T_ST, 0, FA, FI);
    mx_tstore_a(T_ST, FI, di + o, stride * ES);
#elif FFT_SW_CONVERT
    (void)dr;
    (void)di;
    mx_astore(0, FA, yr + o, stride * EA); /* acc_fmt scratch: rounded after the stage */
    mx_astore(2, FA, yi + o, stride * EA);
#else
    mx_astore(0, FA, dr + o, stride * ES);
    mx_astore(2, FA, di + o, stride * ES);
#endif
}

static void stage(int R, int B, int mp, int last, const in_t *sr, const in_t *si, in_t *dr, in_t *di,
                  const acc_t *wr, const acc_t *wi) {
    const long m = (long)R * mp;
    if (R == 16) {
        load_f(16, 0, 0); /* resident for the whole stage */
    }
    if (mp >= 16) {
        const long ys = (long)B * mp; /* output row stride; k blocks are 16 rows apart */
        for (int kb = 0; kb < R / 16; kb++) {
            for (int j0 = 0; j0 < mp; j0 += 16) {
                if (!last) {
                    mx_tload_a(T_C, FA, wr + 16L * kb * mp + j0, mp * EA);
                    mx_tload_a(T_S, FA, wi + 16L * kb * mp + j0, mp * EA);
                }
                for (int t = 0; t < FFT_BATCH; t++) {
                    const long base = (long)t * FFT_N;
                    for (int b = 0; b < B; b++) {
                        dft_block(R, kb, sr, si, base + b * m + j0, 16L * mp, mp, 0);
                        const long o = base + (long)b * mp + j0 + 16L * kb * ys;
                        if (last) {
                            mx_astore(0, FA, yr + o, ys * EA);
                            mx_astore(1, FA, yi + o, ys * EA);
                        } else {
                            twiddle_block();
                            store_next(dr, di, o, ys);
                        }
                    }
                }
            }
        }
    } else { /* mp == 1: the last stage; columns are blocks, loaded transposed */
        for (int t = 0; t < FFT_BATCH; t++) {
            const long base = (long)t * FFT_N;
            for (int b0 = 0; b0 < B; b0 += 16) {
                for (int kb = 0; kb < R / 16; kb++) {
                    dft_block(R, kb, sr, si, base + (long)b0 * R, 16, R, 1);
                    const long o = base + b0 + 16L * kb * B;
                    mx_astore(0, FA, yr + o, (long)B * EA);
                    mx_astore(1, FA, yi + o, (long)B * EA);
                }
            }
        }
    }
#if FFT_SW_CONVERT
    if (!last) { /* the conversion ablation: round the acc_fmt scratch to the next planes */
        naiad_mx_drain();
        for (long o = 0; o < FFT_TOT; o++) {
            dr[o] = real_to_in(yr[o]);
            di[o] = real_to_in(yi[o]);
        }
    }
#endif
}

int main(void) {
    uint64_t start = naiad_cycles();
    int B = 1;
    long off = 0;
    const in_t *sr = xr, *si = xi;
    for (int s = 0; s < FFT_S; s++) {
        const int R = radix[s], mp = FFT_N / (B * R), last = s == FFT_S - 1;
        in_t *dr = s % 2 == 0 ? pr : xr, *di = s % 2 == 0 ? pi : xi;
        stage(R, B, mp, last, sr, si, dr, di, twr + off, twi + off);
        if (!last) {
            off += (long)R * mp;
            sr = dr;
            si = di;
        }
        B *= R;
    }
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - start;
    return 0;
}
