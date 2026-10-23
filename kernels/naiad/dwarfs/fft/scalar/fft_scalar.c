/*
 * fft_scalar: the scalar baseline, a textbook iterative radix-2 decimation-in-
 * time Cooley-Tukey FFT in the accumulator format (float for fp32, double for
 * fp64) on the F/D unit, one transform after another.
 *
 * The timed region covers everything after the host's write: widening the
 * in_fmt input into the output planes in bit-reversed order (which is the
 * algorithm's own permutation, plus a software bf16/fp16 decode for narrow
 * inputs), then log2(N) butterfly passes in place. The twiddles W_N^k are
 * host-computed in fp64 and rounded once to the accumulator format (w2r/w2i).
 */
#include "../fft_io.h"

#if FFT_N & (FFT_N - 1)
#error "FFT_N must be a power of two"
#endif

int main(void) {
    uint64_t start = naiad_cycles();
    for (int t = 0; t < FFT_BATCH; t++) {
        const in_t *ir = xr + (long)t * FFT_N, *ii = xi + (long)t * FFT_N;
        acc_t *ar = yr + (long)t * FFT_N, *ai = yi + (long)t * FFT_N;
        for (unsigned i = 0, r = 0; i < FFT_N; i++) {
            ar[r] = in_to_real(ir[i]);
            ai[r] = in_to_real(ii[i]);
            /* r = bit-reverse(i + 1): a reversed-carry increment */
            unsigned bit = FFT_N >> 1;
            while (r & bit) {
                r ^= bit;
                bit >>= 1;
            }
            r |= bit;
        }
        for (int len = 2, step = FFT_N / 2; len <= FFT_N; len <<= 1, step >>= 1) {
            int half = len >> 1;
            for (int k = 0; k < half; k++) {
                real c = w2r[k * step], s = w2i[k * step];
                for (int i = k; i < FFT_N; i += len) {
                    real ur = ar[i], ui = ai[i];
                    real vr = ar[i + half] * c - ai[i + half] * s;
                    real vi = ar[i + half] * s + ai[i + half] * c;
                    ar[i] = ur + vr;
                    ai[i] = ui + vi;
                    ar[i + half] = ur - vr;
                    ai[i + half] = ui - vi;
                }
            }
        }
    }
    region_cycles = naiad_cycles() - start;
    return 0;
}
