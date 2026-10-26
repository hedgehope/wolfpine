/*
 * jacobi_ame: JACOBI_ITERS sweeps of the 5-point Jacobi stencil on an N x N grid
 * with a fixed boundary ring, on the attached matrix unit. The formulation is in
 * the spec's docstring (kernels/naiad/dwarfs/jacobi/spec.py); in short, the
 * (N-2) x (N-2) interior is cut into 16 x 16 output tiles, and each tile's
 * neighbour sum is a short chain of mma into one accumulator:
 *
 *   JACOBI_FORM 0 (window): acc = Q.Wup + Q.Wdn + Wlt.Q + Wrt.Q, where the W are
 *     the tile's window moved one row / column (the shift is in the load
 *     address) and Q = I. 4 tile loads, 4 mma.
 *   JACOBI_FORM 1 (banded): the horizontal sum is U.T for the block-tridiagonal
 *     T: C.T0 + Lt.Elt + Rt.Ert, with Lt / C / Rt rotating through three tile
 *     registers along a tile row (one new load per tile). The vertical sum is
 *       JACOBI_TSHIFT 0: Q.Wup + Q.Wdn (two window loads), or
 *       JACOBI_TSHIFT 1: T0.C + Ah.H, where H is ONE tile load with a 17-row
 *         stride (its rows 0 and 1 are the grid rows just above and below the
 *         tile) and a tshift moves those two rows to 14 and 15, zeroing the 14
 *         junk rows, so both halo rows cost one load and one mma.
 *   JACOBI_EPILOGUE 1: the 1/4 is one ascal per tile; 0: it is folded into the
 *     constant tiles (exact: 1/4 is a power of two).
 *
 * The grid is stored in the input format between sweeps, so with a reduced
 * input format each sweep's fp32 results are rounded back to it: through the
 * matrix unit (an identity mma into a bf16 accumulator) for bf16, on the
 * scalar side for fp16 (the design has no fp16 accumulator format).
 *
 * Feature switches (AME features the default design lists only in a sweep):
 *   JACOBI_MCONV 1 (acc_move + convert; Ztt mmov.m.a + mconv.ew): with a 16-bit
 *     input format, an intermediate sweep moves each output tile accumulator ->
 *     tile, converts it to the input format on the unit and stores the tile, so
 *     neither the bf16 identity mma pass nor the fp16 scalar pass is needed.
 *   JACOBI_TSHIFT2 1 (JACOBI_FORM 0 only; shift2, NOT in Ztt): the four windows
 *     are funnel shifts (tshift2) of the left / centre / right tiles, rotating
 *     along the tile row as in the banded form, and of one fused-halo load H
 *     (base: the row below the tile, stride -17 rows, so H[0] is the row below
 *     and H[1] the row above): 2 loads per tile instead of 4, plus 1 tshift and
 *     4 tshift2.
 *
 * JACOBI_RADIUS r > 1 (2 or 4; default 1): the radius-r star stencil, N = 16 t
 * + 2 r (spec docstring, "Radius r > 1"). T_r's tiles T0_r / Elt_r / Ert_r
 * replace T0 / Elt / Ert, and the fused-halo tshift does not generalise, so:
 *   JACOBI_FORM 1 (banded): C.T0 + Lt.Elt + Rt.Ert + Pu.Wup + Pd.Wdn, the
 *     windows moved r rows up / down (Pu takes the rows up to the tile's row
 *     15 - r, Pd the rest). 5 mma, 3 loads (Rt, Wup, Wdn) per tile. The eight
 *     tiles are T0 / Pu / Elt / Ert / Pd and the three rotating ones, so both
 *     windows go through the left tile's register once its mma has read it,
 *     and the software pipeline runs the next tile's first two mma under the
 *     loads (JACOBI_TSHIFT has no effect).
 *   JACOBI_FORM 0 (window): Pu.Wup + Pd.Wdn + Wlt.Pl + Wrt.Pr. 4 mma, 4 loads.
 * JACOBI_MCONV and JACOBI_TSHIFT2 are radius 1 only (every tile is in use).
 *
 * Host interface: grid (in, N x N input format), out (N x N accumulator
 * format: the last sweep, before any rounding to the input format),
 * region_cycles.
 */
#include "matrix.h"
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"
#include "dwarf_util.h"

#ifndef JACOBI_N
#error "define JACOBI_N (the grid edge, 16 t + 2)"
#endif
#ifndef JACOBI_ITERS
#define JACOBI_ITERS 8
#endif
#ifndef JACOBI_FORM
#define JACOBI_FORM 1
#endif
#ifndef JACOBI_EPILOGUE
#define JACOBI_EPILOGUE 0
#endif
#ifndef JACOBI_TSHIFT
#define JACOBI_TSHIFT 1
#endif
#ifndef JACOBI_MCONV
#define JACOBI_MCONV 0
#endif
#ifndef JACOBI_TSHIFT2
#define JACOBI_TSHIFT2 0
#endif
#ifndef JACOBI_RADIUS
#define JACOBI_RADIUS 1
#endif

#define N JACOBI_N
#define R JACOBI_RADIUS
#define TT ((N - 2 * R) / 16) /* output tiles per dimension */
/* The tile rows [JACOBI_ROW0, JACOBI_ROW1) the sweeps cover: all of them, unless an
 * includer (jacobi/ame_mc, one run of rows per hart) defines them. */
#ifndef JACOBI_ROW0
#define JACOBI_ROW0 0
#define JACOBI_ROW1 TT
#endif
#define IN DW_IN_FMT
#define ACC DW_ACC_FMT
#define RS ((long)N * DW_IN_BYTES)  /* input row stride, bytes */
#define ORS ((long)N * DW_ACC_BYTES) /* output row stride, bytes */

#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16 || NAIAD_MATRIX_TILES < 8
#error "jacobi_ame is written for a 16 x 16 x 16 unit with 8 tiles"
#endif
#if (N - 2 * R) % 16 || N < 16 + 2 * R
#error "JACOBI_N must be 16 t + 2 JACOBI_RADIUS"
#endif
#if R != 1 && R != 2 && R != 4
#error "JACOBI_RADIUS must be 1, 2 or 4"
#endif
#if R > 1 && (JACOBI_MCONV || JACOBI_TSHIFT2)
#error "JACOBI_MCONV and JACOBI_TSHIFT2 are radius 1 only"
#endif
#if R == 1 && JACOBI_FORM == 1 && JACOBI_TSHIFT && !defined(NAIAD_MATRIX_FEATURE_SHIFT)
#error "JACOBI_TSHIFT needs the matrix unit's shift feature"
#endif
#if JACOBI_EPILOGUE && !defined(NAIAD_MATRIX_FEATURE_EPILOGUE)
#error "JACOBI_EPILOGUE needs the matrix unit's epilogue feature"
#endif
#if JACOBI_MCONV && (!defined(NAIAD_MATRIX_FEATURE_ACC_MOVE) || !defined(NAIAD_MATRIX_FEATURE_CONVERT))
#error "JACOBI_MCONV needs the matrix unit's acc_move and convert features"
#endif
#if JACOBI_TSHIFT2 && (JACOBI_FORM != 0 || !defined(NAIAD_MATRIX_FEATURE_SHIFT2) || !defined(NAIAD_MATRIX_FEATURE_SHIFT))
#error "JACOBI_TSHIFT2 needs JACOBI_FORM 0 and the matrix unit's shift2 and shift features"
#endif
/* The tile JACOBI_MCONV converts through: free in every form except tshift2's. */
#define CONV_TILE (JACOBI_TSHIFT2 ? 8 : 7)
#if JACOBI_MCONV && NAIAD_MATRIX_TILES <= CONV_TILE
#error "JACOBI_MCONV with JACOBI_TSHIFT2 needs a ninth tile"
#endif

NAIAD_IO dw_in_t grid[N * N];
NAIAD_IO dw_acc_t out[N * N];
NAIAD_IO uint64_t region_cycles;

/* The two sweep buffers. The fused-halo load reads 15 * 17 rows past a tile, so
 * with JACOBI_TSHIFT the second buffer is followed by that much mapped slack;
 * what it holds never matters, since the tshift drops those rows. */
#define USES_STRIDED_HALO (R == 1 && JACOBI_FORM == 1 && JACOBI_TSHIFT)
static struct {
#if JACOBI_TSHIFT2
    dw_in_t pre[255 * N]; /* the -17-row-stride halo load reads up to 239 rows before a tile */
#endif
    dw_in_t g[2][N * N];
#if USES_STRIDED_HALO
    dw_in_t slack[255 * N];
#endif
} work __attribute__((section(".noinit"), aligned(64)));

#if R == 1
/* Constant tiles, 16 x 16 input-format elements (or accumulator format, for cid). */
enum { C_Q, C_T0, C_ELT, C_ERT, C_AH, NCONST };
static dw_in_t cst[NCONST][256] __attribute__((aligned(64)));
static dw_acc_t cid[256] __attribute__((aligned(64)));

#define S_SCALE (JACOBI_EPILOGUE ? 1.0 : 0.25)

static void make_constants(void) {
    dw_in_t s = dw_in_bits(S_SCALE), z = dw_in_bits(0.0);
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            cst[C_Q][16 * i + j] = i == j ? s : z;
            cst[C_T0][16 * i + j] = (i - j == 1 || j - i == 1) ? s : z;
            cst[C_ELT][16 * i + j] = (i == 15 && j == 0) ? s : z; /* (Lt.E)[:,0]  = Lt[:,15] */
            cst[C_ERT][16 * i + j] = (i == 0 && j == 15) ? s : z; /* (Rt.E)[:,15] = Rt[:,0] */
            cst[C_AH][16 * i + j] = ((i == 0 && j == 14) || (i == 15 && j == 15)) ? s : z;
            cid[16 * i + j] = i == j ? (dw_acc_t)1 : (dw_acc_t)0;
        }
    }
}

#else /* R > 1 */
/* Constant tiles for T_r and the r-row windows (cid: the accumulator-format identity). */
enum { C_T0, C_ELT, C_ERT, C_PU, C_PD, C_PL, C_PR, NCONST };
static dw_in_t cst[NCONST][256] __attribute__((aligned(64)));
static dw_acc_t cid[256] __attribute__((aligned(64)));

#define S_SCALE (JACOBI_EPILOGUE ? 1.0 : 1.0 / (4 * R))

static int in_band(int k) { return k != 0 && k >= -R && k <= R; }

static void make_constants(void) {
    dw_in_t s = dw_in_bits(S_SCALE), z = dw_in_bits(0.0);
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            /* b views (X.T): column l of X feeds output column j */
            cst[C_T0][16 * i + j] = in_band(i - j) ? s : z;
            cst[C_ELT][16 * i + j] = (j + 16 - i >= 1 && j + 16 - i <= R) ? s : z; /* Lt col i = grid col -16 + i */
            cst[C_ERT][16 * i + j] = (i + 16 - j >= 1 && i + 16 - j <= R) ? s : z; /* Rt col i = grid col 16 + i */
            /* a views (P.W): Wup row j = grid row j - r, Wdn row j = grid row j + r (all of Wup's
             * rows, Wdn's from 16 - r on, so no grid row is counted twice) */
            cst[C_PU][16 * i + j] = in_band(i - j + R) ? s : z;
            cst[C_PD][16 * i + j] = (j >= 16 - 2 * R && in_band(i - j - R)) ? s : z;
            /* b views of the same for the column windows Wlt / Wrt: Pl = Pu^T, Pr = Pd^T */
            cst[C_PL][16 * j + i] = cst[C_PU][16 * i + j];
            cst[C_PR][16 * j + i] = cst[C_PD][16 * i + j];
            cid[16 * i + j] = i == j ? (dw_acc_t)1 : (dw_acc_t)0;
        }
    }
}
#endif

#if JACOBI_EPILOGUE
#define EPI(A) mx_ascal(A, ACC, dw_acc_bits(1.0 / (4 * R)))
#else
#define EPI(A) ((void)0)
#endif

/* Storing a finished output tile: an astore, or with JACOBI_MCONV into an
 * input-format grid (conv), accumulator -> tile -> converted tile -> memory. */
#define ASTORE(A) mx_astore(A, ACC, d - 16 * des, ds)
#if JACOBI_MCONV
#define CONV_STORE(A)                                                                              \
    do {                                                                                           \
        mx_amov_ta(CONV_TILE, A, ACC, 1);                                                          \
        mx_tconv(CONV_TILE, CONV_TILE, 1, ACC, IN);                                                \
        mx_tstore_b(CONV_TILE, IN, d - 16 * des, ds);                                              \
    } while (0)
#define STORE(A)                                                                                   \
    do {                                                                                           \
        if (conv) {                                                                                \
            CONV_STORE(A);                                                                         \
        } else {                                                                                   \
            ASTORE(A);                                                                             \
        }                                                                                          \
    } while (0)
#define SWEEP_CONV , int conv
#define CONV_ARG(x) , x
#else
#define STORE(A) ASTORE(A)
#define SWEEP_CONV
#define CONV_ARG(x)
#endif

/* Both forms are software-pipelined along a tile row: a tile's loads issue
 * before the previous tile's astore, and consecutive tiles use different
 * accumulators, so the in-order core does not stall the L1 port (shared by
 * tile loads and stores) behind an accumulator that is still in flight. */

#if R == 1
/* -- form 0: four shifted windows ------------------------------------------------ */
#if JACOBI_FORM == 0
/* Tile registers: 0 Q, 1..4 the windows. */
#define WSTEP(A, PA)                                                                               \
    do {                                                                                           \
        mx_tload_b(1, IN, c - N, RS);                                                              \
        mx_tload_b(2, IN, c + N, RS);                                                              \
        mx_tload_a(3, IN, c - 1, RS);                                                              \
        mx_tload_a(4, IN, c + 1, RS);                                                              \
        if (J) {                                                                                   \
            STORE(PA);                                                                             \
        }                                                                                          \
        mx_azero(A);                                                                               \
        mx_mma(A, 0, 1, IN, ACC);                                                                  \
        mx_mma(A, 0, 2, IN, ACC);                                                                  \
        mx_mma(A, 3, 0, IN, ACC);                                                                  \
        mx_mma(A, 4, 0, IN, ACC);                                                                  \
        EPI(A);                                                                                    \
        c += 16;                                                                                   \
        d += 16 * des;                                                                             \
    } while (0)

#if JACOBI_TSHIFT2
/* Tile registers: 0 Q, 1..3 rotating left / centre / right, 4 the halo H (then
 * Wup), 5 X (then Wrt), 6 Wdn, 7 Wlt. Accumulators 0..2 rotate with the tiles.
 * Software-pipelined: the next tile's two loads (its right tile into this
 * tile's left register, its halo into 4) issue between this tile's mma, once
 * the shifts have consumed what those registers held; the mma run in the
 * window form's order, so the sums are the same. */
#define ESZ MX_ESZ(DW_IN_BYTES)
#define HLOAD(at) mx_tload_b(4, IN, (at) + 16 * N, -17 * RS) /* H[0] row below, H[1] row above */
#define W2STEP(L, C, R, A, PA)                                                                     \
    do {                                                                                           \
        mx_tshift(5, 4, 0, 0, IN, 14);   /* X[15] = H[1], the row above */                         \
        mx_tshift2(6, C, 4, 0, ESZ, 1);  /* Wdn: rows 1..16 of C ++ H */                           \
        mx_tshift2(4, 5, C, 0, ESZ, 15); /* Wup: rows 15..30 of X ++ C */                          \
        mx_tshift2(5, C, R, 1, ESZ, 1);  /* Wrt: columns 1..16 of C ++ R */                        \
        mx_tshift2(7, L, C, 1, ESZ, 15); /* Wlt: columns 15..30 of L ++ C */                       \
        mx_azero(A);                                                                               \
        mx_mma(A, 0, 4, IN, ACC);                                                                  \
        mx_mma(A, 0, 6, IN, ACC);                                                                  \
        if (J + 1 < TT) {                                                                          \
            mx_tload_a(L, IN, c + 32, RS);                                                         \
            HLOAD(c + 16);                                                                         \
        }                                                                                          \
        if (J) {                                                                                   \
            STORE(PA);                                                                             \
        }                                                                                          \
        mx_mma(A, 7, 0, IN, ACC);                                                                  \
        mx_mma(A, 5, 0, IN, ACC);                                                                  \
        EPI(A);                                                                                    \
        c += 16;                                                                                   \
        d += 16 * des;                                                                             \
    } while (0)

static void sweep(const dw_in_t *src, char *dst, long ds, long des SWEEP_CONV) {
    mx_tload_a(0, IN, cst[C_Q], 16 * DW_IN_BYTES);
    for (int I = JACOBI_ROW0; I < JACOBI_ROW1; I++) {
        const dw_in_t *c = src + (long)(1 + 16 * I) * N + 1;
        char *d = dst + (1 + 16 * I) * ds + des;
        mx_tload_a(1, IN, c - 16, RS); /* its column 15 is the tile row's left boundary column */
        mx_tload_a(2, IN, c, RS);
        mx_tload_a(3, IN, c + 16, RS);
        HLOAD(c);
        int J = 0;
        for (;;) {
            W2STEP(1, 2, 3, 0, 2);
            if (++J == TT) {
                STORE(0);
                break;
            }
            W2STEP(2, 3, 1, 1, 0);
            if (++J == TT) {
                STORE(1);
                break;
            }
            W2STEP(3, 1, 2, 2, 1);
            if (++J == TT) {
                STORE(2);
                break;
            }
        }
    }
}
#else
static void sweep(const dw_in_t *src, char *dst, long ds, long des SWEEP_CONV) {
    mx_tload_a(0, IN, cst[C_Q], 16 * DW_IN_BYTES);
    for (int I = JACOBI_ROW0; I < JACOBI_ROW1; I++) {
        const dw_in_t *c = src + (long)(1 + 16 * I) * N + 1;
        char *d = dst + (1 + 16 * I) * ds + des;
        int J = 0;
        for (;;) {
            WSTEP(0, 1);
            if (++J == TT) {
                STORE(0);
                break;
            }
            WSTEP(1, 0);
            if (++J == TT) {
                STORE(1);
                break;
            }
        }
    }
}
#endif
#else
/* -- form 1: banded horizontal (rotating Lt / C / Rt), window or fused-halo vertical -- */
/* Tile registers: 0 T0, 1 Q or Ah, 2 Elt, 3 Ert, 4..6 rotating, 7 vertical operand.
 * Accumulators 0..2 rotate with the tiles. */
#if JACOBI_TSHIFT
#define VLOAD() mx_tload_b(7, IN, c - N, 17 * RS)
#define VMMA(A, C)                                                                                 \
    mx_mma(A, 0, C, IN, ACC);                                                                      \
    mx_tshift(7, 7, 1, 0, IN, 14);                                                                 \
    mx_mma(A, 1, 7, IN, ACC)
#else
#define VLOAD() mx_tload_b(7, IN, c - N, RS)
#define VMMA(A, C)                                                                                 \
    mx_mma(A, 1, 7, IN, ACC);                                                                      \
    mx_tload_b(7, IN, c + N, RS);                                                                  \
    mx_mma(A, 1, 7, IN, ACC)
#endif
/* With JACOBI_MCONV a converting store goes through tile 7, the vertical operand,
 * so it must come before the step's VLOAD. */
#if JACOBI_MCONV
#define EARLY_STORE(PA)                                                                            \
    do {                                                                                           \
        if (J && conv) {                                                                           \
            CONV_STORE(PA);                                                                        \
        }                                                                                          \
    } while (0)
#define LATE_STORE_COND (J && !conv)
#else
#define EARLY_STORE(PA) ((void)0)
#define LATE_STORE_COND J
#endif
#define BSTEP(L, C, R, A, PA)                                                                      \
    do {                                                                                           \
        mx_tload_a(R, IN, c + 16, RS);                                                             \
        EARLY_STORE(PA);                                                                           \
        VLOAD();                                                                                   \
        if (LATE_STORE_COND) {                                                                     \
            ASTORE(PA);                                                                            \
        }                                                                                          \
        mx_azero(A);                                                                               \
        mx_mma(A, C, 0, IN, ACC);                                                                  \
        mx_mma(A, L, 2, IN, ACC);                                                                  \
        VMMA(A, C);                                                                                \
        mx_mma(A, R, 3, IN, ACC);                                                                  \
        EPI(A);                                                                                    \
        c += 16;                                                                                   \
        d += 16 * des;                                                                             \
    } while (0)

static void sweep(const dw_in_t *src, char *dst, long ds, long des SWEEP_CONV) {
    mx_tload_b(0, IN, cst[C_T0], 16 * DW_IN_BYTES);
    mx_tload_a(1, IN, cst[JACOBI_TSHIFT ? C_AH : C_Q], 16 * DW_IN_BYTES);
    mx_tload_b(2, IN, cst[C_ELT], 16 * DW_IN_BYTES);
    mx_tload_b(3, IN, cst[C_ERT], 16 * DW_IN_BYTES);
    for (int I = JACOBI_ROW0; I < JACOBI_ROW1; I++) {
        const dw_in_t *c = src + (long)(1 + 16 * I) * N + 1;
        char *d = dst + (1 + 16 * I) * ds + des;
        mx_tload_a(4, IN, c - 16, RS); /* its column 15 is the tile row's left boundary column */
        mx_tload_a(5, IN, c, RS);
        int J = 0;
        for (;;) {
            BSTEP(4, 5, 6, 0, 2);
            if (++J == TT) {
                STORE(0);
                break;
            }
            BSTEP(5, 6, 4, 1, 0);
            if (++J == TT) {
                STORE(1);
                break;
            }
            BSTEP(6, 4, 5, 2, 1);
            if (++J == TT) {
                STORE(2);
                break;
            }
        }
    }
}
#endif

#else /* R > 1 */
#define RN ((long)R * N) /* r grid rows, elements */
#if JACOBI_FORM == 0
/* -- radius r, form 0: four r-shifted windows ---------------------------------------- */
/* Tile registers: 0 Pu, 1 Pd, 2 Pl, 3 Pr, 4 Wup, 5 Wdn, 6 Wlt, 7 Wrt. Accumulators 0, 1 alternate;
 * a tile's loads issue before the previous tile's store, as in the radius-1 window form. */
#define WRSTEP(A, PA)                                                                              \
    do {                                                                                           \
        mx_tload_b(4, IN, c - RN, RS);                                                             \
        mx_tload_b(5, IN, c + RN, RS);                                                             \
        mx_tload_a(6, IN, c - R, RS);                                                              \
        mx_tload_a(7, IN, c + R, RS);                                                              \
        if (J) {                                                                                   \
            STORE(PA);                                                                             \
        }                                                                                          \
        mx_azero(A);                                                                               \
        mx_mma(A, 0, 4, IN, ACC);                                                                  \
        mx_mma(A, 1, 5, IN, ACC);                                                                  \
        mx_mma(A, 6, 2, IN, ACC);                                                                  \
        mx_mma(A, 7, 3, IN, ACC);                                                                  \
        EPI(A);                                                                                    \
        c += 16;                                                                                   \
        d += 16 * des;                                                                             \
    } while (0)

static void sweep(const dw_in_t *src, char *dst, long ds, long des SWEEP_CONV) {
    mx_tload_a(0, IN, cst[C_PU], 16 * DW_IN_BYTES);
    mx_tload_a(1, IN, cst[C_PD], 16 * DW_IN_BYTES);
    mx_tload_b(2, IN, cst[C_PL], 16 * DW_IN_BYTES);
    mx_tload_b(3, IN, cst[C_PR], 16 * DW_IN_BYTES);
    for (int I = JACOBI_ROW0; I < JACOBI_ROW1; I++) {
        const dw_in_t *c = src + (long)(R + 16 * I) * N + R;
        char *d = dst + (R + 16 * I) * ds + R * des;
        int J = 0;
        for (;;) {
            WRSTEP(0, 1);
            if (++J == TT) {
                STORE(0);
                break;
            }
            WRSTEP(1, 0);
            if (++J == TT) {
                STORE(1);
                break;
            }
        }
    }
}
#else
/* -- radius r, form 1: banded horizontal, two r-shifted windows vertical ------------------ */
/*
 * Tile registers: 0 T0, 1 Pu, 2 Elt, 3 Ert, 7 Pd, 4..6 rotating Lt / C / Rt. Accumulators
 * 0..2 rotate with the tiles (A this tile, NA the next, PA the previous). Every tile is
 * taken, so the windows Wup / Wdn go through Lt's register once Lt.Elt has read it, and
 * then the next tile's Rt (two tiles right) does. A step enters with this tile's
 * Lt.Elt and C.T0 issued, Wup loading into Lt's register and Rt loaded; it issues the
 * next tile's Lt.Elt (its Lt is this C) and C.T0 (its C is this Rt) between the window
 * loads and their mma, so each load has two mma of cover and the core waits on a WAR
 * only for an mma already in the pipe.
 */
#define BRSTEP(TL, TC, TR, A, NA, PA)                                                              \
    do {                                                                                           \
        mx_mma(A, 1, TL, IN, ACC);         /* Pu.Wup */                                            \
        mx_tload_b(TL, IN, c + RN, RS);    /* Wdn */                                               \
        if (J + 1 < TT) {                                                                          \
            mx_azero(NA);                                                                          \
            mx_mma(NA, TC, 2, IN, ACC);    /* next Lt.Elt */                                       \
            mx_tload_b(TC, IN, c + 16 - RN, RS); /* next Wup */                                    \
        }                                                                                          \
        mx_mma(A, TR, 3, IN, ACC);         /* Rt.Ert */                                            \
        if (J) {                                                                                   \
            STORE(PA);                                                                             \
        }                                                                                          \
        mx_mma(A, 7, TL, IN, ACC);         /* Pd.Wdn */                                            \
        if (J + 1 < TT) {                                                                          \
            mx_mma(NA, TR, 0, IN, ACC);    /* next C.T0 */                                         \
            mx_tload_a(TL, IN, c + 32, RS); /* next Rt */                                          \
        }                                                                                          \
        EPI(A);                                                                                    \
        c += 16;                                                                                   \
        d += 16 * des;                                                                             \
    } while (0)

static void sweep(const dw_in_t *src, char *dst, long ds, long des SWEEP_CONV) {
    mx_tload_b(0, IN, cst[C_T0], 16 * DW_IN_BYTES);
    mx_tload_a(1, IN, cst[C_PU], 16 * DW_IN_BYTES);
    mx_tload_b(2, IN, cst[C_ELT], 16 * DW_IN_BYTES);
    mx_tload_b(3, IN, cst[C_ERT], 16 * DW_IN_BYTES);
    mx_tload_a(7, IN, cst[C_PD], 16 * DW_IN_BYTES);
    for (int I = JACOBI_ROW0; I < JACOBI_ROW1; I++) {
        const dw_in_t *c = src + (long)(R + 16 * I) * N + R;
        char *d = dst + (R + 16 * I) * ds + R * des;
        mx_tload_a(4, IN, c - 16, RS); /* its last r columns are the tile row's left boundary */
        mx_tload_a(5, IN, c, RS);
        mx_tload_a(6, IN, c + 16, RS);
        mx_azero(0);
        mx_mma(0, 4, 2, IN, ACC);
        mx_mma(0, 5, 0, IN, ACC);
        mx_tload_b(4, IN, c - RN, RS); /* Wup */
        int J = 0;
        for (;;) {
            BRSTEP(4, 5, 6, 0, 1, 2);
            if (++J == TT) {
                STORE(0);
                break;
            }
            BRSTEP(5, 6, 4, 1, 2, 0);
            if (++J == TT) {
                STORE(1);
                break;
            }
            BRSTEP(6, 4, 5, 2, 0, 1);
            if (++J == TT) {
                STORE(2);
                break;
            }
        }
    }
}
#endif
#endif

/* Round the interior of out (accumulator format) into dst (input format). */
static void to_input_format(dw_in_t *dst) {
#if DW_SAME_FMT
    (void)dst;
#elif DW_IN_FMT == NAIAD_MFMT_BF16
    mx_tload_a(0, ACC, cid, 16 * DW_ACC_BYTES);
    for (int I = JACOBI_ROW0; I < JACOBI_ROW1; I++) {
        for (int J = 0; J < TT; J++) {
            long o = (long)(R + 16 * I) * N + R + 16 * J;
            mx_tload_b(1, ACC, out + o, ORS);
            mx_azero(1);
            mx_mma(1, 0, 1, ACC, NAIAD_MFMT_BF16);
            mx_astore(1, NAIAD_MFMT_BF16, dst + o, RS);
        }
    }
#else
    for (int i = R + 16 * JACOBI_ROW0; i < R + 16 * JACOBI_ROW1; i++) {
        for (int j = R; j < N - R; j++) {
            dst[i * N + j] = dw_acc_to_in(out[i * N + j]);
        }
    }
#endif
}

#ifndef JACOBI_NO_MAIN /* jacobi/ame_mc has its own */
int main(void) {
    make_constants();
    for (int i = 0; i < N * N; i++) {
        work.g[0][i] = work.g[1][i] = grid[i];
        out[i] = dw_in_to_acc(grid[i]);
    }

    uint64_t t0 = naiad_cycles();
    for (int it = 0; it < JACOBI_ITERS; it++) {
        const dw_in_t *src = work.g[it & 1];
        dw_in_t *next = work.g[(it + 1) & 1];
        int last = it == JACOBI_ITERS - 1;
        if (DW_SAME_FMT && !last) {
            sweep(src, (char *)next, RS, DW_IN_BYTES CONV_ARG(0));
#if JACOBI_MCONV
        } else if (!last) {
            sweep(src, (char *)next, RS, DW_IN_BYTES, 1); /* converted on the unit */
#endif
        } else {
            sweep(src, (char *)out, ORS, DW_ACC_BYTES CONV_ARG(0));
            if (!last) {
                to_input_format(next);
            }
        }
    }
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - t0;
    return 0;
}
#endif
