/*
 * tensorprod, rvv variant: ur / us / ut as Nekbone's three mxm calls, each the
 * register-blocked RVV outer-product micro-kernel of rvv_mxm.h (vector along
 * the contiguous output index, scalar operand from the other matrix), written
 * in the shared padded layouts (tp_layout.h; the padding is never touched):
 *
 *   ut_e = D . U_e        (p x p^2, k = p): vectors of p^2
 *   us_e = D . U'_e       (p x p^2) after an axis swap, then scattered (below)
 *   ur = U . D^T          (E p^2 x p): vectors of p only; D^T is built once
 *
 * us contracts the middle index, so per slab it is a p x p x p matmul with
 * p-element vectors. Swapping u's k and l axes first (u'[e][l][k][i], unit-
 * stride copies of p-element rows) makes us_e[j][(k, i)] = sum_l D[j][l]
 * u'_e[l][(k, i)] one p x p^2 GEMM with long vectors, like ut; its rows are
 * then copied (again p-element unit-stride rows) into out_s's layout. The two
 * copies move 4 E p^3 elements and cost far less than the short-vector
 * products they replace. ur contracts the contiguous index, and no unit-stride
 * reordering gives it long vectors (that needs a transpose, i.e. strided
 * accesses at one element a cycle), so it keeps p-element vectors: one
 * scalar A load per vfmacc, which at p = 8 makes the single-issue front end
 * the bound.
 *
 * 16-bit inputs are read in place and multiplied exactly by vfwmacc.vf /
 * vfwmaccbf16.vf, accumulating in fp32.
 */
#include "../tp_layout.h"

#define DW_IN_FMT TP_MFMT_IN
#define DW_ACC_FMT TP_MFMT_ACC
#define RV_AL 4
#include "rvv_fmt.h"
#include "rvv_mxm.h"

_Static_assert(sizeof(in_t) == sizeof(rv_in_bits_t), "input storage width");
_Static_assert(sizeof(acc_t) == sizeof(rv_out_t), "output storage width");

static rv_in_bits_t DT[P * P];
static rv_in_bits_t uswap[E * P3]; /* u with the k and l axes swapped: [e][l][k][i] */
static rv_out_t cs[E * P3];        /* us per element, [e][j][k][i] */

/*
 * dst row (e, a, b) = src row (e, b, a), for p-element rows (dst row at
 * e * des + a * dxs + b * dys, src row at e * ses + b * sxs + a * sys), four b at a time: the four
 * loads issue (and are pinned) before the four stores, so a store never
 * waits out its own load's latency.
 */
#define COPY_SWAP(LOAD, STORE, VT, dst, des, dxs, dys, src, ses, sxs, sys)                         \
    for (long e = 0; e < E; e++) {                                                                 \
        for (long a = 0; a < P; a++) {                                                             \
            const __typeof__(*(src)) *s_ = (src) + e * (ses) + a * (sys);                          \
            __typeof__(*(dst)) *d_ = (dst) + e * (des) + a * (dxs);                                \
            long b = 0;                                                                            \
            for (; b + 4 <= P; b += 4) {                                                           \
                VT x0 = LOAD(s_ + b * (sxs)), x1 = LOAD(s_ + (b + 1) * (sxs));                     \
                VT x2 = LOAD(s_ + (b + 2) * (sxs)), x3 = LOAD(s_ + (b + 3) * (sxs));               \
                RV_USE(x0);                                                                        \
                RV_USE(x1);                                                                        \
                RV_USE(x2);                                                                        \
                RV_USE(x3);                                                                        \
                STORE(d_ + b * (dys), x0);                                                         \
                STORE(d_ + (b + 1) * (dys), x1);                                                   \
                STORE(d_ + (b + 2) * (dys), x2);                                                   \
                STORE(d_ + (b + 3) * (dys), x3);                                                   \
            }                                                                                      \
            for (; b < P; b++) {                                                                   \
                STORE(d_ + b * (dys), LOAD(s_ + b * (sxs)));                                       \
            }                                                                                      \
        }                                                                                          \
    }
#define LD_IN(p) rv_load_in((p), vlp)
#define ST_IN(p, x) rv_store_in((p), (x), vlp)
#define LD_ACC(p) rv_load_acc((p), vlp)
#define ST_ACC(p, x) rv_store_acc((p), (x), vlp)

RV_KERNEL int main(void) {
    tp_dims[0] = P;
    tp_dims[1] = E;
    const rv_in_bits_t *d = (const rv_in_bits_t *)D, *us = (const rv_in_bits_t *)u;
    uint64_t t0 = naiad_cycles();
    for (long i = 0; i < P; i++) { /* D^T, for ur's B operand (raw bits: no rounding) */
        for (long l = 0; l < P; l++) {
            DT[l * P + i] = d[i * P + l];
        }
    }
    /* ur (E p^2 i-rows x p) = U . D^T. Output i-row R lands at out_r[(R / Q) * T + (R % Q) * P]:
       one strided matrix when Q * P == T or Q == 1. */
#if Q * P == T || Q == 1
    rv_mxm(E * P2, P, P, us, P, DT, P, (rv_out_t *)out_r, Q == 1 ? T : P);
#else
    for (long v = 0; v < CDIV(E * P2, Q); v++) {
        const long rows = E * P2 - v * Q < Q ? E * P2 - v * Q : Q;
        rv_mxm(rows, P, P, us + v * Q * P, P, DT, P, (rv_out_t *)out_r + v * T, P);
    }
#endif
    for (long e = 0; e < E; e++) { /* ut_e (p x p^2) = D . U_e */
        rv_mxm(P, P2, P, d, P, us + e * P3, P2, (rv_out_t *)out_t + e * P * LDT, LDT);
    }
    const size_t vlp = rv_setvl(P); /* p <= 16 <= VLMAX at LMUL 4 */
    /* u'[e][l][k] = u[e][k][l]: dst (a = l, b = k) at l p^2 + k p; src (k, l) at k p^2 + l p */
    COPY_SWAP(LD_IN, ST_IN, rv_in_v, uswap, P3, P2, P, us, P3, P2, P);
    for (long e = 0; e < E; e++) { /* us_e[j][(k, i)] = D . U'_e */
        rv_mxm(P, P2, P, d, P, uswap + e * P3, P2, cs + e * P3, P2);
    }
    /* out_s row (e p^2 + k p + j) = cs[e][j][k]: dst (a = k, b = j) at (k p + j) T; src (j, k) at j p^2 + k p */
    COPY_SWAP(LD_ACC, ST_ACC, rv_acc_v, (rv_out_t *)out_s, P2 * T, P * T, T, cs, P3, P2, P);
    region_cycles = naiad_cycles() - t0;
    return 0;
}
