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

#define N JACOBI_N
#define TT ((N - 2) / 16) /* output tiles per dimension */
#define IN DW_IN_FMT
#define ACC DW_ACC_FMT
#define RS ((long)N * DW_IN_BYTES)  /* input row stride, bytes */
#define ORS ((long)N * DW_ACC_BYTES) /* output row stride, bytes */

#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16 || NAIAD_MATRIX_TILES < 8
#error "jacobi_ame is written for a 16 x 16 x 16 unit with 8 tiles"
#endif
#if (N - 2) % 16 || N < 18
#error "JACOBI_N must be 16 t + 2"
#endif
#if JACOBI_FORM == 1 && JACOBI_TSHIFT && !defined(NAIAD_MATRIX_FEATURE_SHIFT)
#error "JACOBI_TSHIFT needs the matrix unit's shift feature"
#endif
#if JACOBI_EPILOGUE && !defined(NAIAD_MATRIX_FEATURE_EPILOGUE)
#error "JACOBI_EPILOGUE needs the matrix unit's epilogue feature"
#endif

NAIAD_IO dw_in_t grid[N * N];
NAIAD_IO dw_acc_t out[N * N];
NAIAD_IO uint64_t region_cycles;

/* The two sweep buffers. The fused-halo load reads 15 * 17 rows past a tile, so
 * with JACOBI_TSHIFT the second buffer is followed by that much mapped slack;
 * what it holds never matters, since the tshift drops those rows. */
#define USES_STRIDED_HALO (JACOBI_FORM == 1 && JACOBI_TSHIFT)
static struct {
    dw_in_t g[2][N * N];
#if USES_STRIDED_HALO
    dw_in_t slack[255 * N];
#endif
} work __attribute__((section(".noinit"), aligned(64)));

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

#if JACOBI_EPILOGUE
#define EPI(A) mx_ascal(A, ACC, dw_acc_bits(0.25))
#else
#define EPI(A) ((void)0)
#endif

/* Both forms are software-pipelined along a tile row: a tile's loads issue
 * before the previous tile's astore, and consecutive tiles use different
 * accumulators, so the in-order core does not stall the L1 port (shared by
 * tile loads and stores) behind an accumulator that is still in flight. */

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
            mx_astore(PA, ACC, d - 16 * des, ds);                                                  \
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

static void sweep(const dw_in_t *src, char *dst, long ds, long des) {
    mx_tload_a(0, IN, cst[C_Q], 16 * DW_IN_BYTES);
    for (int I = 0; I < TT; I++) {
        const dw_in_t *c = src + (long)(1 + 16 * I) * N + 1;
        char *d = dst + (1 + 16 * I) * ds + des;
        int J = 0;
        for (;;) {
            WSTEP(0, 1);
            if (++J == TT) {
                mx_astore(0, ACC, d - 16 * des, ds);
                break;
            }
            WSTEP(1, 0);
            if (++J == TT) {
                mx_astore(1, ACC, d - 16 * des, ds);
                break;
            }
        }
    }
}
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
#define BSTEP(L, C, R, A, PA)                                                                      \
    do {                                                                                           \
        mx_tload_a(R, IN, c + 16, RS);                                                             \
        VLOAD();                                                                                   \
        if (J) {                                                                                   \
            mx_astore(PA, ACC, d - 16 * des, ds);                                                  \
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

static void sweep(const dw_in_t *src, char *dst, long ds, long des) {
    mx_tload_b(0, IN, cst[C_T0], 16 * DW_IN_BYTES);
    mx_tload_a(1, IN, cst[JACOBI_TSHIFT ? C_AH : C_Q], 16 * DW_IN_BYTES);
    mx_tload_b(2, IN, cst[C_ELT], 16 * DW_IN_BYTES);
    mx_tload_b(3, IN, cst[C_ERT], 16 * DW_IN_BYTES);
    for (int I = 0; I < TT; I++) {
        const dw_in_t *c = src + (long)(1 + 16 * I) * N + 1;
        char *d = dst + (1 + 16 * I) * ds + des;
        mx_tload_a(4, IN, c - 16, RS); /* its column 15 is the tile row's left boundary column */
        mx_tload_a(5, IN, c, RS);
        int J = 0;
        for (;;) {
            BSTEP(4, 5, 6, 0, 2);
            if (++J == TT) {
                mx_astore(0, ACC, d - 16 * des, ds);
                break;
            }
            BSTEP(5, 6, 4, 1, 0);
            if (++J == TT) {
                mx_astore(1, ACC, d - 16 * des, ds);
                break;
            }
            BSTEP(6, 4, 5, 2, 1);
            if (++J == TT) {
                mx_astore(2, ACC, d - 16 * des, ds);
                break;
            }
        }
    }
}
#endif

/* Round the interior of out (accumulator format) into dst (input format). */
static void to_input_format(dw_in_t *dst) {
#if DW_SAME_FMT
    (void)dst;
#elif DW_IN_FMT == NAIAD_MFMT_BF16
    mx_tload_a(0, ACC, cid, 16 * DW_ACC_BYTES);
    for (int I = 0; I < TT; I++) {
        for (int J = 0; J < TT; J++) {
            long o = (long)(1 + 16 * I) * N + 1 + 16 * J;
            mx_tload_b(1, ACC, out + o, ORS);
            mx_azero(1);
            mx_mma(1, 0, 1, ACC, NAIAD_MFMT_BF16);
            mx_astore(1, NAIAD_MFMT_BF16, dst + o, RS);
        }
    }
#else
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            dst[i * N + j] = dw_acc_to_in(out[i * N + j]);
        }
    }
#endif
}

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
            sweep(src, (char *)next, RS, DW_IN_BYTES);
        } else {
            sweep(src, (char *)out, ORS, DW_ACC_BYTES);
            if (!last) {
                to_input_format(next);
            }
        }
    }
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - t0;
    return 0;
}
