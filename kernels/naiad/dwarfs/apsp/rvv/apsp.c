/*
 * apsp_rvv: the RVV baseline for apsp_ame: Floyd-Warshall with the j loop
 * vectorised at LMUL 4 (32 fp32 / 16 fp64 lanes at VLEN 256):
 * D_i[j..] = vfmin(D_i[j..], vfadd.vf(D_k[j..], D[i][k])). The j chunks are the
 * outer loop and the rows the inner one (legal, since step k changes neither row
 * k nor column k), so each chunk of row k is loaded once per step. Rows go two
 * at a time, alternating between two register sets so the next pair loads while
 * the current pair computes; within a pair both adds issue before both mins
 * (no chaining). About half the datapath / LSU bound (8 + 8 cycles per fp32
 * row chunk): the rest is issue (address arithmetic, vsetvli after the pins).
 */
#include "naiad_platform.h"

#include <riscv_vector.h>

#include "dwarf_util.h"

#ifndef APSP_N
#error "define APSP_N"
#endif
#define N APSP_N
#if N % 8
#error "apsp_rvv takes rows eight at a time: APSP_N must be a multiple of 8"
#endif
#ifdef APSP_FMT_FP32
typedef float ap_t;
typedef vfloat32m4_t ap_v;
#define AP_SETVL __riscv_vsetvl_e32m4
#define AP_LOAD __riscv_vle32_v_f32m4
#define AP_STORE __riscv_vse32_v_f32m4
#elif defined(APSP_FMT_FP64)
typedef double ap_t;
typedef vfloat64m4_t ap_v;
#define AP_SETVL __riscv_vsetvl_e64m4
#define AP_LOAD __riscv_vle64_v_f64m4
#define AP_STORE __riscv_vse64_v_f64m4
#else
#error "apsp_rvv supports fp32 and fp64"
#endif
#define AP_PIN(v) __asm__ __volatile__("" : "+vr"(v))
#define AP_FUSE(s) __asm__ __volatile__("" ::"f"(s))

NAIAD_IO ap_t D[N * N];
NAIAD_IO uint64_t region_cycles;

/* Two rows r0, r0 + 1 of chunk j into set X (vectors X0, X1, scalars sX0, sX1 = D[r][k]). */
#define LOAD2(X, r0)                                                                                   \
    do {                                                                                               \
        const ap_t *p_ = &D[(long)(r0) * N + j];                                                       \
        X##0 = AP_LOAD(p_, vl);                                                                        \
        X##1 = AP_LOAD(p_ + N, vl);                                                                    \
        s##X##0 = p_[k - j];                                                                           \
        s##X##1 = p_[N + k - j];                                                                       \
        AP_PIN(X##0);                                                                                  \
        AP_PIN(X##1);                                                                                  \
        AP_FUSE(s##X##0);                                                                              \
        AP_FUSE(s##X##1);                                                                              \
    } while (0)

/* rows r0, r0 + 1 = min(rows, row k + D[r][k]): the two adds, then the two mins, then the stores
 * (no chaining: each consumer waits for its producer's whole result). */
#define UPDATE2(X, r0)                                                                                 \
    do {                                                                                               \
        ap_t *p_ = &D[(long)(r0) * N + j];                                                             \
        ap_v t0 = __riscv_vfadd(kv, s##X##0, vl), t1 = __riscv_vfadd(kv, s##X##1, vl);                \
        AP_PIN(t0);                                                                                    \
        AP_PIN(t1);                                                                                    \
        X##0 = __riscv_vfmin(X##0, t0, vl);                                                            \
        X##1 = __riscv_vfmin(X##1, t1, vl);                                                            \
        AP_PIN(X##0);                                                                                  \
        AP_PIN(X##1);                                                                                  \
        AP_STORE(p_, X##0, vl);                                                                        \
        AP_STORE(p_ + N, X##1, vl);                                                                    \
    } while (0)

__attribute__((target("tune=sifive-x280"))) int main(void) {
    uint64_t t0 = naiad_cycles();
    for (int k = 0; k < N; k++) {
        const ap_t *dk = &D[(long)k * N];
        /* j chunks outside, rows inside (legal: step k changes neither row k nor
         * column k), so one load of row k serves every row. Rows go two at a
         * time, ping-ponging between register sets A and B so the next two
         * rows load while the current two compute (and no vmv copies). */
        for (long j = 0, vl; j < N; j += vl) {
            vl = (long)AP_SETVL(N - j);
            ap_v kv = AP_LOAD(dk + j, vl);
            AP_PIN(kv);
            ap_v A0, A1, B0, B1;
            ap_t sA0, sA1, sB0, sB1;
            LOAD2(A, 0);
            for (int i = 0; i < N; i += 4) {
                LOAD2(B, i + 2);
                UPDATE2(A, i);
                if (i + 4 < N) {
                    LOAD2(A, i + 4);
                }
                UPDATE2(B, i + 2);
            }
        }
    }
    region_cycles = naiad_cycles() - t0;
    return 0;
}
