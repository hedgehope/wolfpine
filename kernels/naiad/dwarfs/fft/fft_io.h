/*
 * fft_io.h: the host-visible arrays of the fft dwarf (both variants declare
 * all of them, because the host writes every input symbol). See spec.py.
 *
 * Problem: FFT_BATCH independent complex transforms of length FFT_N, stored
 * as split planes (re, im), transform t at [t * FFT_N]. The AME variant's
 * plan is FFT_S stages of radices FFT_R0, FFT_R1, FFT_R2 (16 or 32).
 *
 *   xr, xi       input, in_fmt; the AME variant reuses them as its in_fmt
 *                working planes, so they are clobbered
 *   yr, yi       output, acc_fmt, natural order
 *   twr, twi     AME stage twiddles, acc_fmt: for each stage s but the last,
 *                W_{m_s}^{k j} at [off_s + k * m'_s + j] (k < r_s, j < m'_s)
 *   w2r, w2i     radix-2 twiddles W_N^k, k < N/2, acc_fmt (scalar variant)
 *   f16*, f32*   the DFT-16 / DFT-32 matrices, in_fmt, row-major:
 *                re, im, -im and re + im (Gauss), each rounded once from fp64
 */
#ifndef FFT_IO_H
#define FFT_IO_H

#include "naiad_platform.h"
#include "dwfmt.h"

#ifndef FFT_N
#error "FFT_N (transform length) must be set"
#endif
#ifndef FFT_BATCH
#define FFT_BATCH 1
#endif
#ifndef FFT_TW_LEN
#define FFT_TW_LEN 1
#endif
#define FFT_TOT (FFT_N * FFT_BATCH)

NAIAD_IO in_t xr[FFT_TOT];
NAIAD_IO in_t xi[FFT_TOT];
NAIAD_IO acc_t yr[FFT_TOT];
NAIAD_IO acc_t yi[FFT_TOT];
NAIAD_IO acc_t twr[FFT_TW_LEN];
NAIAD_IO acc_t twi[FFT_TW_LEN];
NAIAD_IO acc_t w2r[FFT_N / 2];
NAIAD_IO acc_t w2i[FFT_N / 2];
NAIAD_IO in_t f16r[16 * 16];
NAIAD_IO in_t f16i[16 * 16];
NAIAD_IO in_t f16n[16 * 16];
NAIAD_IO in_t f16s[16 * 16];
NAIAD_IO in_t f32r[32 * 32];
NAIAD_IO in_t f32i[32 * 32];
NAIAD_IO in_t f32n[32 * 32];
NAIAD_IO in_t f32s[32 * 32];
NAIAD_IO uint64_t region_cycles;

#endif /* FFT_IO_H */
