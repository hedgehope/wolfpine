/*
 * nbody_ame_rvv: ame_fused's two GEMMs with the pair weights on the vector unit (the unfused hybrid).
 *
 * The epilogue-fusion study's control for nbody: the distance GEMM D = A B
 * (nbody_ame) and the force GEMM F = W X - x (W 1) in acc_fmt (ame_fused),
 * with the weights between them computed by RVV instead of on the
 * accumulators: each D tile is astored to a buffer, the vector unit clamps it,
 * adds eps^2 and forms w = y^3 with y = vfrsqrt7 + Newton steps (nbody_rvv's
 * soft-law arithmetic: two steps for fp32, four for fp64), zeroes the
 * diagonal on diagonal tiles, and W goes back to the unit as a tile. The tile
 * makes the round trip accumulator -> memory -> vector registers -> memory ->
 * tile that ame_fused avoids. Softened gravity only.
 */
#include "../nbody_io.h"
#include "matrix.h"
#include "dwarf_util.h"

#define RV_AL 8
#include "rvv_fmt.h"

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
#error "nbody_ame_rvv needs a square unit with K >= 5"
#endif
#if NB_LJ
#error "nbody_ame_rvv implements the softened-gravity law only"
#endif
#if RV_AE == 64
#define NB_NEWTON 4
#else
#define NB_NEWTON 2
#endif

static in_t Aop[N * TK];
static in_t Bop[TK * N];
static acc_t Xf[N * TN];
static acc_t Fo[TM * TN];
static acc_t wbuf[TM * TN];
static real P[N][3];

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
}

/* wbuf = w(max(wbuf, 0)) in place, the soft law by rsqrt7 + Newton. */
RV_KERNEL static void weights(void) {
    const rv_acc_s eps2 = params[0];
    for (long k = 0, vl; k < TM * TN; k += vl) {
        vl = (long)rv_setvl(TM * TN - k);
        rv_acc_v s = __riscv_vfadd(__riscv_vfmax(rv_load_acc(wbuf + k, vl), (rv_acc_s)0, vl), eps2, vl);
        rv_acc_v h = __riscv_vfmul(s, (rv_acc_s)0.5, vl), y = __riscv_vfrsqrt7(s, vl);
        for (int it = 0; it < NB_NEWTON; it++) {
            rv_acc_v t = __riscv_vfmul(y, y, vl);
            t = __riscv_vfnmsub(t, h, rv_splat((rv_acc_s)1.5, vl), vl); /* 1.5 - h y^2 */
            y = __riscv_vfmul(y, t, vl);
        }
        rv_store_acc(wbuf + k, __riscv_vfmul(__riscv_vfmul(y, y, vl), y, vl), vl);
    }
}

RV_KERNEL int main(void) {
    uint64_t start = naiad_cycles();
    build_operands();
    for (int i0 = 0; i0 < N; i0 += TM) {
        mx_tload_a(0, FI, &Aop[i0 * TK], TK * ES);
        mx_azero(2);
        for (int j0 = 0; j0 < N; j0 += TN) {
            mx_azero(0);
            mx_tload_b(1, FI, &Bop[j0], N * ES);
            mx_mma(0, 0, 1, FI, FA);
#if NB_SAVE_D2
            mx_astore(0, FA, &d2[(long)i0 * N + j0], N * EA);
#endif
            mx_astore(0, FA, wbuf, TN * EA);
            weights();
            if (i0 == j0) {
                for (int k = 0; k < TM; k++) {
                    wbuf[k * TN + k] = 0;
                }
            }
            mx_tload_a(4, FA, wbuf, TN * EA);
            mx_tload_b(5, FA, &Xf[j0 * TN], TN * EA);
            mx_mma(2, 4, 5, FA, FA);
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
