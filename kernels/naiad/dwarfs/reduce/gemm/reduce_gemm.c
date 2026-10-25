/*
 * reduce, IME / VME variants: the sum of a length-N vector as one call of the emitted GEMM
 * driver (naiad_gemm.h) against a ones operand whose rows overlap, so it is never a K x N
 * matrix in memory. Built for an IME design (Zvvm; framework/naiad/emit/gemm_ime.py) or a VME
 * design (Zvt; framework/naiad/emit/gemm_vme.py); the orientation is each driver's cheap one:
 *
 *   IME  x is A (MB x K row-major, read by plain tile loads), ones is B (K x NB) at ldb = 1
 *        over a run of K + NB ones (Zvvm's stride 0 means the packed default, not a broadcast):
 *        C[i][j] = sum_k x[i K + k] for every j. The total is the sum of C's MB row sums.
 *        Redundancy: NB MACs per element (8 at VLEN 256), against AME's 16.
 *   VME  x is B (K x NC row-major, read in place by whole-register loads), ones is A
 *        (MB x K, lda = 0; the driver's transposed A panel is built from the one row):
 *        C[i][j] = sum_k x[k NC + j] for every i. The total is one RVV reduction over C's
 *        first row. Redundancy: MB MACs per element (the driver's block rows, 64 at TE 32).
 *        NC = min(256, N / 64), so K >= 64 keeps B in place (K a multiple of the body).
 *
 * The driver is the GEMM library a compiler would call, unchanged: it re-reads the ones row
 * for every K step (IME: a transposing tile load at W = 1; the driver's B panel at W > 1) where a hand-written kernel would keep it in registers.
 *
 * Built with -DREDUCE_N, -DREDUCE_IN/_ACC (format names) and -DREDUCE_IN_UP/_ACC_UP (upper
 * case) by the dwarf harness (kernels/naiad/dwarfs/reduce/spec.py). Host interface as
 * reduce_ame: x (in), result (one accumulator-format element), region_cycles.
 */
#include <stddef.h>
#include <stdint.h>

#include "dwarf_util.h"
#include "naiad_platform.h"

#if !defined(NAIAD_MATRIX_IME) && !defined(NAIAD_MATRIX_VME)
#error "reduce's ime / vme variants need an IME or VME design (naiad-ime-timed.yaml, naiad-vme-timed.yaml)"
#endif
#ifndef REDUCE_N
#error "define REDUCE_N"
#endif
#define N REDUCE_N

#define NAIAD_GEMM_IN REDUCE_IN
#define NAIAD_GEMM_ACC REDUCE_ACC
#ifdef NAIAD_MATRIX_VME
#define NC (N / 64 < 256 ? N / 64 : 256)
#define KR (N / NC)
#define NAIAD_GEMM_MAX_K KR
#define NAIAD_GEMM_MAX_N NC
#endif
#include "naiad_gemm.h"

#define RG_CAT3_(a, b, c) NAIAD_GEMM_##a##_##b##_##c
#define RG_CAT3(a, b, c) RG_CAT3_(a, b, c)
#define MB RG_CAT3(REDUCE_IN_UP, REDUCE_ACC_UP, MB)
#define NB RG_CAT3(REDUCE_IN_UP, REDUCE_ACC_UP, NB)

NAIAD_IO naiad_gemm_in_t x[N];
NAIAD_IO naiad_gemm_acc_t result[1];
NAIAD_IO uint64_t region_cycles;

/* 1.0 in the input format (bits for the 16-bit formats). */
#if defined(REDUCE_IN_FP64) || defined(REDUCE_IN_FP32)
#define ONE ((naiad_gemm_in_t)1)
#elif defined(REDUCE_IN_BF16)
#define ONE ((naiad_gemm_in_t)0x3F80)
#elif defined(REDUCE_IN_FP16)
#define ONE ((naiad_gemm_in_t)0x3C00)
#else
#error "unsupported REDUCE_IN"
#endif

#ifdef NAIAD_MATRIX_IME
#if N % MB
#error "REDUCE_N must be a multiple of the driver's MB"
#endif
#define KR (N / MB)
static naiad_gemm_in_t ones[KR + NB] __attribute__((aligned(64)));
static naiad_gemm_acc_t C[MB * NB] __attribute__((aligned(64)));
#else
static naiad_gemm_in_t ones[KR] __attribute__((aligned(64)));
static naiad_gemm_acc_t C[MB * NC] __attribute__((aligned(64)));
#include <riscv_vector.h>

static naiad_gemm_acc_t row_sum(const naiad_gemm_acc_t *row, long n) {
#ifdef REDUCE_ACC_FP64
    size_t vl = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t s = __riscv_vfmv_v_f_f64m1(0.0, vl);
    for (long i = 0; i < n; i += (long)vl) {
        vl = __riscv_vsetvl_e64m8((size_t)(n - i));
        s = __riscv_vfredusum_vs_f64m8_f64m1(__riscv_vle64_v_f64m8(&row[i], vl), s, vl);
    }
    return __riscv_vfmv_f_s_f64m1_f64(s);
#else
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t s = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (long i = 0; i < n; i += (long)vl) {
        vl = __riscv_vsetvl_e32m8((size_t)(n - i));
        s = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vle32_v_f32m8(&row[i], vl), s, vl);
    }
    return __riscv_vfmv_f_s_f32m1_f32(s);
#endif
}
#endif

int main(void) {
    for (unsigned i = 0; i < sizeof ones / sizeof ones[0]; i++) {
        ones[i] = ONE;
    }
    uint64_t t0 = naiad_cycles();
#ifdef NAIAD_MATRIX_IME
    naiad_gemm(MB, NB, KR, x, KR, ones, 1, C, NB);
    naiad_gemm_acc_t s = 0;
    for (int i = 0; i < MB; i++) {
        s += C[i * NB];
    }
#else
    naiad_gemm(MB, NC, KR, ones, 0, x, NC, C, NC);
    naiad_gemm_acc_t s = row_sum(C, NC);
#endif
    result[0] = s;
    region_cycles = naiad_cycles() - t0;
    return 0;
}
