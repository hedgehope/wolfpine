/*
 * fft_ame_rvv: fft_ame with the inter-stage twiddle pass on the vector unit (the unfused hybrid).
 *
 * The epilogue-fusion study's control: everything is fft_ame (../ame/fft_ame.c,
 * the DFT blocks on the matrix unit, acc_fmt planes out), except that the
 * twiddle multiply and the rounding back to in_fmt run as RVV over each
 * contiguous run of j (LMUL 4, acc_fmt arithmetic, 16-bit results narrowed
 * with vfncvt / vfncvtbf16, RNE) instead of on the scalar core. The data still
 * makes the round trip accumulator -> memory -> vector registers -> memory ->
 * tile; ame_fused keeps it in the unit. re' = ar c - ai s with one fused
 * multiply-subtract, im' = ar s + ai c likewise, as the fused variant's amacc.
 */
#include "../fft_io.h"

#define RV_AL 4
#include "rvv_fmt.h"

#if FFT_GAUSS
#error "fft_ame_rvv uses the four-mma complex product"
#endif

RV_KERNEL static void twiddle(int R, int B, int mp, const acc_t *wr, const acc_t *wi) {
    for (int t = 0; t < FFT_BATCH; t++) {
        long o = (long)t * FFT_N;
        for (int k = 0; k < R; k++) {
            const acc_t *cw = wr + (long)k * mp, *sw = wi + (long)k * mp;
            for (int b = 0; b < B; b++, o += mp) {
                for (long j = 0, vl; j < mp; j += vl) {
                    vl = (long)rv_setvl(mp - j);
                    rv_acc_v ar = rv_load_acc(yr + o + j, vl), ai = rv_load_acc(yi + o + j, vl);
                    rv_acc_v c = rv_load_acc(cw + j, vl), s = rv_load_acc(sw + j, vl);
                    rv_acc_v pr = __riscv_vfnmsac(__riscv_vfmul(ar, c, vl), ai, s, vl);
                    rv_acc_v pi = __riscv_vfmacc(__riscv_vfmul(ar, s, vl), ai, c, vl);
                    rv_store_in(xr + o + j, rv_narrow(pr, vl), vl);
                    rv_store_in(xi + o + j, rv_narrow(pi, vl), vl);
                }
            }
        }
    }
}
#define FFT_TWIDDLE_DEFINED 1

#include "../ame/fft_ame.c"
