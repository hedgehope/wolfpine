/*
 * nbody_ame_fused: all-pairs softened gravity with the pair weights computed inside the matrix unit.
 *
 * The distances are nbody_ame's expansion GEMM, D = A B (../ame/nbody_ame.c),
 * and the force sum is nbody_ame_fgemm's second GEMM, F = W X - x (W 1). In
 * between, where those variants hand every D tile to the scalar core (a clamp,
 * an fsqrt and an fdiv per pair), this one stays on the unit (feature ewise,
 * ewise_tx for the rsqrt):
 *
 *   D = max(D, 0)                  aew.ax max, x0       (cancellation can make D < 0)
 *   D = D + eps^2                  aew.ax add
 *   D = rsqrt(D)                   aewu rsqrt           r = (r^2 + eps^2)^(-1/2)
 *   T = D                          amov.t.a             (feature acc_move)
 *   D = D * T                      aew.at mul
 *   D = D * T                      aew.at mul           w = r^3
 *   D = D * M                      aew.at mul           diagonal tiles only: M = 1 - I
 *   T = D                          amov.t.a
 *   acc2 += T x Xf                 mma, acc_fmt x acc_fmt
 *
 * Every step depends on the one before, and the core issues in order with no
 * queue in front of the unit, so one chain exposes every latency. Two column
 * tiles therefore run interleaved, instruction by instruction: chain 0 in
 * accumulator 0 with tiles 1 (B), 4 (T), 5 (Xf), chain 1 in accumulator 3 with
 * tiles 3, 6, 7. Tile 0 holds the row block's A, tile 2 the mask, acc 2 is F.

 * Unlike ame_fgemm, w is not rounded to in_fmt: the force GEMM multiplies the
 * acc_fmt weights by acc_fmt positions (Xf[j] = [x_j, y_j, z_j, 1, 0 ...], exact
 * copies of the in_fmt positions), so it needs acc_fmt to be an input format too
 * (fp32 / fp64 are on the dwarfs design) and costs that format's passes. The
 * scalar side only finishes F_i = (W X)_i - x_i (W 1)_i per 16-row block.
 *
 * Ablations, by the design's feature list: without ewise_tx the weights go
 * through memory to a scalar w(r^2 + eps^2) loop and back as a tile; without
 * acc_move W goes to memory and back as a tile (no scalar work).
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
#if TK < 5 || TN != TK || TM != TN
#error "nbody_ame_fused needs a square unit with K >= 5 (the force GEMM's W is an M x N accumulator)"
#endif
#if NAIAD_MATRIX_ACCS < 4 || NAIAD_MATRIX_TILES < 6
#error "nbody_ame_fused needs 4 accumulators and 6 tiles"
#endif
#if (N / TN) % 2
#error "nbody_ame_fused pairs the column tiles: NB_N / N must be even"
#endif
#ifndef NAIAD_MATRIX_FEATURE_EWISE
#error "nbody_ame_fused needs the matrix unit's ewise feature (matrix.features: [..., ewise])"
#endif
#ifndef NAIAD_MATRIX_FEATURE_ACC_MEM
#error "nbody_ame_fused needs aload/astore (matrix.acc_mem)"
#endif
#if NB_LJ
#error "nbody_ame_fused implements the softened-gravity law only"
#endif
#ifdef NAIAD_MATRIX_FEATURE_EWISE_TX
#define NB_UNIT_RSQRT 1
#else
#define NB_UNIT_RSQRT 0
#endif
#ifdef NAIAD_MATRIX_FEATURE_ACC_MOVE
#define NB_UNIT_MOVE 1
#else
#define NB_UNIT_MOVE 0
#endif

static in_t Aop[N * TK];
static in_t Bop[TK * N];
static acc_t Xf[N * TN];
static acc_t mask[TM * TN];
static acc_t Fo[TM * TN];
static real P[N][3];
#if !NB_UNIT_RSQRT || !NB_UNIT_MOVE
static acc_t wbuf[2][TM * TN]; /* the ablations' memory round trip, one per chain */
#endif

DW_INLINE unsigned long acc_bits(real x) {
#if DW_ACC_FMT == NAIAD_MFMT_FP64
    return naiad_f64_bits(x);
#else
    return naiad_f32_bits(x);
#endif
}

static void build_operands(void) {
    for (int i = 0; i < N; i++) {
        real x = in_to_real(pos[3 * i]), y = in_to_real(pos[3 * i + 1]), z = in_to_real(pos[3 * i + 2]);
        real n2 = x * x + y * y + z * z;
        P[i][0] = x;
        P[i][1] = y;
        P[i][2] = z;
        in_t *a = &Aop[i * TK];
        in_t n2i = real_to_in(n2), one = real_to_in(1);
        a[0] = real_to_in(-2 * x);
        a[1] = real_to_in(-2 * y);
        a[2] = real_to_in(-2 * z);
        a[3] = n2i;
        a[4] = one;
        Bop[0 * N + i] = pos[3 * i];
        Bop[1 * N + i] = pos[3 * i + 1];
        Bop[2 * N + i] = pos[3 * i + 2];
        Bop[3 * N + i] = one;
        Bop[4 * N + i] = n2i;
        acc_t *xf = &Xf[i * TN];
        xf[0] = x;
        xf[1] = y;
        xf[2] = z;
        xf[3] = 1;
    }
    for (int ii = 0; ii < TM; ii++) {
        for (int jj = 0; jj < TN; jj++) {
            mask[ii * TN + jj] = ii == jj ? 0 : 1;
        }
    }
}

/* Chain c (0 or 1): accumulator, B tile, T tile, Xf tile. */
#define NB_D(c) ((c) ? 3 : 0)
#define NB_B(c) ((c) ? 3 : 1)
#define NB_T(c) ((c) ? 6 : 4)
#define NB_X(c) ((c) ? 7 : 5)

/* tile T(c) = acc D(c): on the unit, or through memory without acc_move. */
#if NB_UNIT_MOVE
#define NB_TO_TILE(c) mx_amov_ta(NB_T(c), NB_D(c), FA, 0)
#else
#define NB_TO_TILE(c)                                                                              \
    do {                                                                                           \
        mx_astore(NB_D(c), FA, wbuf[c], TN * EA);                                                  \
        mx_tload_a(NB_T(c), FA, wbuf[c], TN * EA);                                                 \
    } while (0)
#endif

#if NB_SAVE_D2 /* the raw GEMM output, as nbody_ame */
#define NB_SAVE(c, j) mx_astore(NB_D(c), FA, &d2[(long)i0 * N + (j)], N * EA)
#else
#define NB_SAVE(c, j) ((void)0)
#endif

int main(void) {
    uint64_t start = naiad_cycles();
    build_operands();
    const unsigned long eps2 = acc_bits(params[0]);
    mx_tload_a(2, FA, mask, TN * EA); /* 1 - I, for the diagonal tiles */
    /* B tiles do not depend on the row block: each pair prefetches the next pair's
     * (wrapping to column 0 for the next row block), and the loads sit between
     * element-wise ops, so the in-order core never waits on a busy load path. */
    mx_tload_b(NB_B(0), FI, &Bop[0], N * ES);
    mx_tload_b(NB_B(1), FI, &Bop[TN], N * ES);
    for (int i0 = 0; i0 < N; i0 += TM) {
        mx_tload_a(0, FI, &Aop[i0 * TK], TK * ES);
        mx_azero(2);
        for (int j0 = 0; j0 < N; j0 += 2 * TN) {
            const int j1 = j0 + TN, jn = (j0 + 2 * TN) % N;
            mx_azero(NB_D(0));
            mx_azero(NB_D(1));
            mx_mma(NB_D(0), 0, NB_B(0), FI, FA);
            mx_mma(NB_D(1), 0, NB_B(1), FI, FA);
            NB_SAVE(0, j0);
            NB_SAVE(1, j1);
            mx_aew_ax(NB_D(0), NB_D(0), FA, MX_EW_MAX, 0);
            mx_tload_b(NB_X(0), FA, &Xf[j0 * TN], TN * EA);
            mx_aew_ax(NB_D(1), NB_D(1), FA, MX_EW_MAX, 0);
            mx_tload_b(NB_X(1), FA, &Xf[j1 * TN], TN * EA);
            mx_aew_ax(NB_D(0), NB_D(0), FA, MX_EW_ADD, eps2);
            mx_tload_b(NB_B(0), FI, &Bop[jn], N * ES);
            mx_aew_ax(NB_D(1), NB_D(1), FA, MX_EW_ADD, eps2);
            mx_tload_b(NB_B(1), FI, &Bop[jn + TN], N * ES);
#if NB_UNIT_RSQRT
            mx_aewu(NB_D(0), NB_D(0), FA, MX_EWU_RSQRT);
            mx_aewu(NB_D(1), NB_D(1), FA, MX_EWU_RSQRT);
            NB_TO_TILE(0);
            NB_TO_TILE(1);
            mx_aew_at(NB_D(0), NB_D(0), NB_T(0), FA, MX_EW_MUL);
            mx_aew_at(NB_D(1), NB_D(1), NB_T(1), FA, MX_EW_MUL);
            mx_aew_at(NB_D(0), NB_D(0), NB_T(0), FA, MX_EW_MUL);
            mx_aew_at(NB_D(1), NB_D(1), NB_T(1), FA, MX_EW_MUL);
            if (i0 == j0) {
                mx_aew_at(NB_D(0), NB_D(0), 2, FA, MX_EW_MUL);
            } else if (i0 == j1) {
                mx_aew_at(NB_D(1), NB_D(1), 2, FA, MX_EW_MUL);
            }
            NB_TO_TILE(0);
            NB_TO_TILE(1);
#else /* no rsqrt on the unit: s = r^2 + eps^2 to memory, w(s) on the scalar side */
            for (int c = 0; c < 2; c++) {
                if (c == 0) {
                    mx_astore(NB_D(0), FA, wbuf[0], TN * EA);
                } else {
                    mx_astore(NB_D(1), FA, wbuf[1], TN * EA);
                }
                acc_t *w = wbuf[c];
                for (int k = 0; k < TM * TN; k++) {
                    real sk = w[k];
                    w[k] = (real)1 / (sk * dw_sqrt(sk));
                }
                if (i0 == (c ? j1 : j0)) {
                    for (int k = 0; k < TM; k++) {
                        w[k * TN + k] = 0;
                    }
                }
            }
            mx_tload_a(NB_T(0), FA, wbuf[0], TN * EA);
            mx_tload_a(NB_T(1), FA, wbuf[1], TN * EA);
#endif
            mx_mma(2, NB_T(0), NB_X(0), FA, FA);
            mx_mma(2, NB_T(1), NB_X(1), FA, FA);
        }
        /* F_i = (W X)_i - x_i (W 1)_i */
        mx_astore(2, FA, Fo, TN * EA);
        for (int ii = 0; ii < TM; ii++) {
            const real s = Fo[ii * TN + 3];
            for (int c = 0; c < 3; c++) {
                force[3 * (i0 + ii) + c] = Fo[ii * TN + c] - P[i0 + ii][c] * s;
            }
        }
    }
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - start;
    return 0;
}
