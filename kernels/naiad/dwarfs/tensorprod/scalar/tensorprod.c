/*
 * tensorprod, scalar variant: ur / us / ut as Nekbone's mxm calls, each a 4 x 4
 * register-blocked F/D matmul (the same blocking as the scalar GEMM, so loads
 * overlap the FMAs), written in the AME variant's padded layouts (tp_layout.h;
 * the scalar code only indexes them, never touches the padding). Computes in
 * the accumulator's precision; bf16/fp16 inputs are widened once inside the
 * timed region (the dwarfs' design has no Zfh/Zfbfmin).
 */
#include "../tp_layout.h"

#ifdef TP_ACC_FP64
typedef double ct_t;
#else
typedef float ct_t;
#endif

#if defined(TP_IN_BF16) || defined(TP_IN_FP16)
static ct_t Dw[P * P], uw[E * P3];

static inline ct_t widen(in_t v) {
#if defined(TP_IN_BF16)
    return (ct_t)naiad_bf16_to_f32(v);
#else
    return (ct_t)naiad_fp16_to_f32(v);
#endif
}
#define DS Dw
#define US uw
#else
#define DS ((const ct_t *)D)
#define US ((const ct_t *)u)
#endif

/*
 * out(i, j) = sum_l a(i, l) b(l, j) for i < m, j < n, l < kk, with every operand
 * addressed by (row, column) strides. Always inlined, so the strides and sizes
 * fold to constants at each call.
 */
static inline __attribute__((always_inline)) void mxm(long m, long n, long kk, const ct_t *a, long ars, long acs,
                                                      const ct_t *b, long brs, long bcs, acc_t *out, long ors,
                                                      long ocs) {
    long i = 0;
    for (; i + 4 <= m; i += 4) {
        long j = 0;
        for (; j + 4 <= n; j += 4) {
            ct_t c[4][4] = {{0}};
            for (long l = 0; l < kk; l++) {
                ct_t av[4], bv[4];
                for (int r = 0; r < 4; r++) {
                    av[r] = a[(i + r) * ars + l * acs];
                    bv[r] = b[l * brs + (j + r) * bcs];
                }
                for (int r = 0; r < 4; r++) {
                    for (int s = 0; s < 4; s++) {
                        c[r][s] += av[r] * bv[s];
                    }
                }
            }
            for (int r = 0; r < 4; r++) {
                for (int s = 0; s < 4; s++) {
                    out[(i + r) * ors + (j + s) * ocs] = (acc_t)c[r][s];
                }
            }
        }
        for (; j < n; j++) {
            ct_t c[4] = {0};
            for (long l = 0; l < kk; l++) {
                const ct_t bv = b[l * brs + j * bcs];
                for (int r = 0; r < 4; r++) {
                    c[r] += a[(i + r) * ars + l * acs] * bv;
                }
            }
            for (int r = 0; r < 4; r++) {
                out[(i + r) * ors + j * ocs] = (acc_t)c[r];
            }
        }
    }
    for (; i + 2 <= m; i += 2) {
        long j = 0;
        for (; j + 4 <= n; j += 4) {
            ct_t c[2][4] = {{0}};
            for (long l = 0; l < kk; l++) {
                ct_t av[2], bv[4];
                for (int r = 0; r < 2; r++) {
                    av[r] = a[(i + r) * ars + l * acs];
                }
                for (int s = 0; s < 4; s++) {
                    bv[s] = b[l * brs + (j + s) * bcs];
                }
                for (int r = 0; r < 2; r++) {
                    for (int s = 0; s < 4; s++) {
                        c[r][s] += av[r] * bv[s];
                    }
                }
            }
            for (int r = 0; r < 2; r++) {
                for (int s = 0; s < 4; s++) {
                    out[(i + r) * ors + (j + s) * ocs] = (acc_t)c[r][s];
                }
            }
        }
        for (; j < n; j++) {
            for (int r = 0; r < 2; r++) {
                ct_t c = 0;
                for (long l = 0; l < kk; l++) {
                    c += a[(i + r) * ars + l * acs] * b[l * brs + j * bcs];
                }
                out[(i + r) * ors + j * ocs] = (acc_t)c;
            }
        }
    }
    for (; i < m; i++) {
        long j = 0;
        for (; j < n; j++) {
            ct_t c = 0;
            for (long l = 0; l < kk; l++) {
                c += a[i * ars + l * acs] * b[l * brs + j * bcs];
            }
            out[i * ors + j * ocs] = (acc_t)c;
        }
    }
}

int main(void) {
    tp_dims[0] = P;
    tp_dims[1] = E;
    uint64_t t0 = naiad_cycles();
#if defined(TP_IN_BF16) || defined(TP_IN_FP16)
    for (long i = 0; i < P * P; i++) {
        Dw[i] = widen(D[i]);
    }
    for (long i = 0; i < E * P3; i++) {
        uw[i] = widen(u[i]);
    }
#endif
    const ct_t *d = DS;
    /* ur (E p^2 i-rows x p) = U . D^T: b(l, i) = D[i][l]. Output i-row R lands at
       out_r[(R / Q) * T + (R % Q) * P]: one strided matrix when Q * P == T or Q == 1. */
#if Q * P == T || Q == 1
    mxm(E * P2, P, P, US, P, 1, d, 1, P, out_r, Q == 1 ? T : P, 1);
#else
    for (long v = 0; v < CDIV(E * P2, Q); v++) {
        const long rows = E * P2 - v * Q < Q ? E * P2 - v * Q : Q;
        mxm(rows, P, P, US + v * Q * P, P, 1, d, 1, P, out_r + v * T, P, 1);
    }
#endif
    for (long e = 0; e < E; e++) {
        const ct_t *ue = US + e * P3;
        /* ut_e (p x p^2) = D . U_e */
        mxm(P, P2, P, d, P, 1, ue, P2, 1, out_t + e * P * LDT, LDT, 1);
        /* us_(e,k) (p x p) = D . u_(e,k), per slab */
        for (long k = 0; k < P; k++) {
            mxm(P, P, P, d, P, 1, ue + k * P2, P, 1, out_s + (e * P + k) * P * T, T, 1);
        }
    }
    region_cycles = naiad_cycles() - t0;
    return 0;
}
