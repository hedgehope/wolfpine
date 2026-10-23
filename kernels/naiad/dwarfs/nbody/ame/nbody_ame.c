/*
 * nbody_ame: all-pairs forces with the squared distances from the matrix unit.
 *
 * The expansion |x_i - x_j|^2 = |x_i|^2 + |x_j|^2 - 2 x_i.x_j is one GEMM over
 * augmented coordinates, padded to the tile K:
 *
 *   A[i] = [-2x_i, -2y_i, -2z_i, |x_i|^2, 1, 0 ...]      (N x K, "a")
 *   B[:, j] = [x_j, y_j, z_j, 1, |x_j|^2, 0 ...]          (K x N, "b")
 *   D = A B,  D[i][j] = r_ij^2                            (acc_fmt)
 *
 * so the norms go through the matrix unit as in_fmt operands too. That is
 * the formulation whose cancellation the study measures: when |x|^2 >> r^2
 * (close pairs away from the origin), D's error is ~u_in * |x|^2, not
 * ~u * r^2. -DNB_NORMS_SCALAR=1 keeps only -2 x_i.x_j in the GEMM and adds
 * the acc_fmt norms on the scalar side (error ~u_acc * |x|^2 instead).
 *
 * Forces on the scalar side, one 16 x 16 D tile at a time: r^2 = max(D, 0)
 * (cancellation can make D negative), then F_i += w(r^2) (x_j - x_i) with the
 * coordinate differences in acc_fmt from the in_fmt positions. The GEMM thus
 * replaces the 3 multiplies and 2 adds of r^2 per pair, not the differences,
 * which the force direction needs anyway. ../ame_fgemm moves the force sum to
 * the matrix unit as well.
 */
#include "../nbody_io.h"
#include "matrix.h"
#include "dwarf_util.h"

#define ES DW_IN_BYTES
#define EA DW_ACC_BYTES
#define FI DW_IN_FMT
#define FA DW_ACC_FMT
#define TM NAIAD_MATRIX_M
#define TN NAIAD_MATRIX_N
#define TK NAIAD_MATRIX_K
#define N NB_N

#if N % TM || N % TN
#error "NB_N must be a multiple of the tile M and N"
#endif
#if TK < 5
#error "the augmented distance GEMM needs K >= 5"
#endif
#ifndef NB_NORMS_SCALAR
#define NB_NORMS_SCALAR 0
#endif
#ifndef NB_FORCE_GEMM
#define NB_FORCE_GEMM 0
#endif

static in_t Aop[N * TK];
static in_t Bop[TK * N];
static real P[N][3];
#if NB_NORMS_SCALAR
static real nrm[N];
#endif
#if !NB_SAVE_D2
static acc_t dbuf[TM * TN];
#endif

#if NB_FORCE_GEMM
#if TN != TK || TN < 4
#error "the force GEMM needs N == K >= 4"
#endif
/* W tile (rows i, cols j) and Xop[j] = [x, y, z, 1, 0 ...]: F += W Xop. */
static in_t Wt[TM * TK];
static in_t Xop[N * TN];
static acc_t Fo[TM * TN];
#endif

static void build_operands(void) {
    for (int i = 0; i < N; i++) {
        real x = in_to_real(pos[3 * i]), y = in_to_real(pos[3 * i + 1]), z = in_to_real(pos[3 * i + 2]);
        real n2 = x * x + y * y + z * z;
        P[i][0] = x;
        P[i][1] = y;
        P[i][2] = z;
        in_t *a = &Aop[i * TK];
        a[0] = real_to_in(-2 * x);
        a[1] = real_to_in(-2 * y);
        a[2] = real_to_in(-2 * z);
        Bop[0 * N + i] = pos[3 * i];
        Bop[1 * N + i] = pos[3 * i + 1];
        Bop[2 * N + i] = pos[3 * i + 2];
#if NB_NORMS_SCALAR
        nrm[i] = n2;
#else
        in_t n2i = real_to_in(n2), one = real_to_in(1);
        a[3] = n2i;
        a[4] = one;
        Bop[3 * N + i] = one;
        Bop[4 * N + i] = n2i;
#endif
#if NB_FORCE_GEMM
        in_t *xo = &Xop[i * TN];
        xo[0] = pos[3 * i];
        xo[1] = pos[3 * i + 1];
        xo[2] = pos[3 * i + 2];
        xo[3] = real_to_in(1);
#endif
    }
}

int main(void) {
    uint64_t start = naiad_cycles();
    build_operands();
    for (int i0 = 0; i0 < N; i0 += TM) {
        real f[TM][3] = {{0}};
        mx_tload_a(0, FI, &Aop[i0 * TK], TK * ES);
#if NB_FORCE_GEMM
        mx_azero(2);
#endif
        for (int j0 = 0; j0 < N; j0 += TN) {
            mx_azero(0);
            mx_tload_b(1, FI, &Bop[j0], N * ES);
            mx_mma(0, 0, 1, FI, FA);
#if NB_SAVE_D2
            acc_t *D = &d2[(long)i0 * N + j0];
            const long ds = N;
#else
            acc_t *D = dbuf;
            const long ds = TN;
#endif
            mx_astore(0, FA, D, ds * EA);
            for (int ii = 0; ii < TM; ii++) {
                const int i = i0 + ii;
#if NB_FORCE_GEMM
                for (int jj = 0; jj < TN; jj++) {
                    real r2 = D[ii * ds + jj];
                    r2 = r2 > 0 ? r2 : 0;
                    real w = i == j0 + jj ? 0 : pair_weight(r2) * params[4];
                    Wt[ii * TK + jj] = real_to_in(w);
                }
#else
                const real xi = P[i][0], yi = P[i][1], zi = P[i][2];
                real fx = 0, fy = 0, fz = 0;
                for (int jj = 0; jj < TN; jj++) {
                    const int j = j0 + jj;
                    real r2 = D[ii * ds + jj];
#if NB_NORMS_SCALAR
                    r2 += nrm[i] + nrm[j];
#endif
                    r2 = r2 > 0 ? r2 : 0;
                    if (j == i) {
                        continue;
                    }
                    real w = pair_weight(r2);
                    fx += w * (P[j][0] - xi);
                    fy += w * (P[j][1] - yi);
                    fz += w * (P[j][2] - zi);
                }
                f[ii][0] += fx;
                f[ii][1] += fy;
                f[ii][2] += fz;
#endif
            }
#if NB_FORCE_GEMM
            mx_tload_a(2, FI, Wt, TK * ES);
            mx_tload_b(3, FI, &Xop[j0 * TN], TN * ES);
            mx_mma(2, 2, 3, FI, FA);
#endif
        }
#if NB_FORCE_GEMM
        /* F_i = (sum_j w_ij x_j - x_i sum_j w_ij) / scale: a second cancellation. */
        mx_astore(2, FA, Fo, TN * EA);
        for (int ii = 0; ii < TM; ii++) {
            const real s = Fo[ii * TN + 3], inv = 1 / params[4];
            for (int c = 0; c < 3; c++) {
                f[ii][c] = (Fo[ii * TN + c] - P[i0 + ii][c] * s) * inv;
            }
        }
#endif
        for (int ii = 0; ii < TM; ii++) {
            for (int c = 0; c < 3; c++) {
                force[3 * (i0 + ii) + c] = f[ii][c];
            }
        }
    }
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - start;
    return 0;
}
