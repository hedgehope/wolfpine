/*
 * spmv_rvv: the RVV baseline for spmv_ame (same BSR storage, VLEN-agnostic up to
 * VL 16: LMUL 2 for fp32, 4 for fp64, half that for 16-bit inputs).
 *
 * NV >= 2: vectors over the NV right-hand sides (VL = NV). For four rows r of a
 * block row at a time, every block p and column c adds A_p[r][c] * X_c, a
 * vfmacc.vf of one scalar element with the X row (16-bit inputs: the widening
 * vfwmacc / vfwmaccbf16), so each X row is loaded once per four FMAs; the next
 * X row's load is issued before the current row's FMAs (the core is in order,
 * without chaining).
 *
 * NV == 1 (a true SpMV): vectors over a block's 16 columns. For four rows at a
 * time, acc_r += A_p[r][:] * x_blk (vfmacc.vv) over the block row's blocks, then
 * one vfredusum per row.
 *
 * Host interface as spmv_ame.
 */
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"

#if DW_ACC_FMT == NAIAD_MFMT_FP64
#define RV_AL 4
#else
#define RV_AL 2
#endif
#include "rvv_fmt.h"

#if !defined(SPMV_N) || !defined(SPMV_NNZB) || !defined(SPMV_NV)
#error "define SPMV_N, SPMV_NNZB and SPMV_NV"
#endif
#define N SPMV_N
#define NB (N / 16)
#define NNZB SPMV_NNZB
#define NV SPMV_NV

NAIAD_IO int32_t rowptr[NB + 1];
NAIAD_IO int32_t colidx[NNZB];
NAIAD_IO dw_in_t vals[NNZB * 256];
NAIAD_IO dw_in_t X[N * NV + 16];
NAIAD_IO dw_acc_t Y[N * NV];
NAIAD_IO uint64_t region_cycles;

#if NV >= 2

/* Keep a scalar A element's load ahead of its FMA (no "f" register class for __bf16 in clang 20). */
#if DW_IN_FMT == NAIAD_MFMT_BF16
#define SP_FUSE(s) ((void)(s))
#else
#define SP_FUSE(s) RV_FUSE(s)
#endif

/* Rows per group: 8 accumulators (16 registers) at LMUL 2, 4 at fp64's LMUL 4. */
#if RV_AL == 4
#define RG 4
#define FOR_ROWS(M) M(0) M(1) M(2) M(3)
#else
#define RG 8
#define FOR_ROWS(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#endif
#define DECL_Y(k) rv_acc_v y##k = rv_splat(0, vl);
#define DECL_S(k) rv_in_s s##k = rv_in_scalar(a, 16 * (k));
#define NEXT_S(k) n##k = rv_in_scalar(a, 16 * (k) + c + 1);
#define FUSE_S(k) SP_FUSE(n##k);
#define DECL_N(k) rv_in_s n##k = s##k;
#define FMA_Y(k) y##k = rv_fma_vf(y##k, s##k, xv, vl);
#define SHIFT_S(k) s##k = n##k;
#define STORE_Y(k) rv_store_acc(y + (k) * NV, y##k, vl);

RV_KERNEL int main(void) {
    uint64_t t0 = naiad_cycles();
    const size_t vl = rv_setvl(NV);
    for (int bi = 0; bi < NB; bi++) {
        const int p0 = rowptr[bi], p1 = rowptr[bi + 1];
        for (int r0 = 0; r0 < 16; r0 += RG) {
            FOR_ROWS(DECL_Y)
            for (int p = p0; p < p1; p++) {
                const dw_in_t *a = &vals[(long)p * 256 + r0 * 16];
                const dw_in_t *xb = &X[(long)colidx[p] * 16 * NV];
                rv_in_v xv = rv_load_in(xb, vl);
                FOR_ROWS(DECL_S)
                RV_PIN(xv);
#pragma clang loop unroll(full)
                for (int c = 0; c < 16; c++) {
                    /* column c + 1's X row and RG A elements, one column ahead of their FMAs */
                    rv_in_v xn = xv;
                    FOR_ROWS(DECL_N)
                    if (c < 15) {
                        xn = rv_load_in(xb + (c + 1) * NV, vl);
                        FOR_ROWS(NEXT_S)
                        RV_PIN(xn);
                        FOR_ROWS(FUSE_S)
                    }
                    FOR_ROWS(FMA_Y)
                    xv = xn;
                    FOR_ROWS(SHIFT_S)
                }
            }
            dw_acc_t *y = &Y[(long)(bi * 16 + r0) * NV];
            FOR_ROWS(STORE_Y)
        }
    }
    region_cycles = naiad_cycles() - t0;
    return 0;
}

#else /* NV == 1 */

/* acc + a * x, elementwise, a and x in the input format (widening for 16-bit inputs). */
RV_INLINE rv_acc_v sp_fma_vv(rv_acc_v acc, rv_in_v a, rv_in_v x, size_t vl) {
#if RV_SAME
    return __riscv_vfmacc(acc, a, x, vl);
#elif DW_IN_FMT == NAIAD_MFMT_FP16
    return __riscv_vfwmacc(acc, a, x, vl);
#else
    return __riscv_vfwmaccbf16(acc, a, x, vl);
#endif
}

RV_KERNEL int main(void) {
    uint64_t t0 = naiad_cycles();
    const size_t vl = rv_setvl(16);
    for (int bi = 0; bi < NB; bi++) {
        const int p0 = rowptr[bi], p1 = rowptr[bi + 1];
        for (int r0 = 0; r0 < 16; r0 += 4) {
            rv_acc_v y0 = rv_splat(0, vl), y1 = y0, y2 = y0, y3 = y0;
            for (int p = p0; p < p1; p++) {
                const dw_in_t *a = &vals[(long)p * 256 + r0 * 16];
                rv_in_v xv = rv_load_in(&X[(long)colidx[p] * 16], vl);
                rv_in_v a0 = rv_load_in(a, vl), a1 = rv_load_in(a + 16, vl);
                rv_in_v a2 = rv_load_in(a + 32, vl), a3 = rv_load_in(a + 48, vl);
                RV_PIN(xv);
                RV_PIN(a0);
                RV_PIN(a1);
                RV_PIN(a2);
                RV_PIN(a3);
                y0 = sp_fma_vv(y0, a0, xv, vl);
                y1 = sp_fma_vv(y1, a1, xv, vl);
                y2 = sp_fma_vv(y2, a2, xv, vl);
                y3 = sp_fma_vv(y3, a3, xv, vl);
            }
            dw_acc_t *y = &Y[bi * 16 + r0];
            y[0] = rv_reduce_usum(y0, vl);
            y[1] = rv_reduce_usum(y1, vl);
            y[2] = rv_reduce_usum(y2, vl);
            y[3] = rv_reduce_usum(y3, vl);
        }
    }
    region_cycles = naiad_cycles() - t0;
    return 0;
}

#endif
