/*
 * rvv_mxm.h: the RVV register-blocked outer-product matmul shared by the gemm
 * and tensorprod `rvv` variants. Include after rvv_fmt.h (RV_AL 4).
 *
 *   rv_mxm(m, n, k, a, lda, b, ldb, c, ldc)
 *       C[i][j] = sum_l A[i][l] B[l][j]   (i < m, j < n, l < k)
 *       A: element (i, l) at a[i * lda + l], input format (raw storage)
 *       B: row l at b + l * ldb, unit stride, input format
 *       C: row i at c + i * ldc, unit stride, rv_out_t (RV_OUT_* below)
 *
 * The micro-kernel holds an RV_MR x VL block of C in RV_MR accumulator
 * groups (LMUL 4: 5 x 32 fp32 / 5 x 16 fp64, or 6 x 32 fp32 for 16-bit
 * inputs, at VLEN 256), and for each l loads one row of B as a vector and
 * RV_MR scalars of A, and issues RV_MR vfmacc.vf (vfwmacc.vf /
 * vfwmaccbf16.vf for 16-bit inputs, so products are exact and each sum
 * rounds once, in the accumulator format). That is one vector load and
 * RV_MR scalar loads per RV_MR FMAs: the datapath, not the load unit, is the
 * bound once VL is a full register group. The next B row and the next A
 * scalars are loaded one step ahead (two B buffers and two scalar sets,
 * unrolled by two), so the in-order core never waits for a load. B is walked in VL-wide column panels
 * and the panel is reused by every row block (all of it sits in L1).
 *
 * Output: RV_OUT_ACC (default) stores the accumulator format; RV_OUT_BF16
 * rounds an fp32 accumulator to bf16 once on store (vfncvtbf16, RNE).
 */
#ifndef NAIAD_RVV_MXM_H
#define NAIAD_RVV_MXM_H

#ifndef NAIAD_RVV_FMT_H
#error "include rvv_fmt.h first"
#endif

/* Rows per block: 6 accumulators + two LMUL-2 B buffers for 16-bit inputs (28
 * registers); 5 + two LMUL-4 buffers for fp32/fp64 (28: six would need all 32
 * and the allocator answers with whole-register copies). */
#ifndef RV_MR
#if RV_SAME
#define RV_MR 5
#else
#define RV_MR 6
#endif
#endif
#if RV_MR < 1 || RV_MR > 6
#error "RV_MR must be 1..6"
#endif

#ifdef RV_OUT_BF16
#if RV_AE != 32
#error "RV_OUT_BF16 needs an fp32 accumulator"
#endif
typedef uint16_t rv_out_t;
RV_INLINE void rv_store_out(rv_out_t *p, rv_acc_v v, size_t vl) {
    RV_C(__riscv_vse16_v_bf16, RV_HM)((__bf16 *)p, RV_C(__riscv_vfncvtbf16_f_f_w_bf16, RV_HM)(v, vl), vl);
}
#else
typedef rv_acc_s rv_out_t;
RV_INLINE void rv_store_out(rv_out_t *p, rv_acc_v v, size_t vl) { rv_store_acc(p, v, vl); }
#endif

#ifndef RV_MXM_PINS
#define RV_MXM_PINS 1
#endif
#if RV_MXM_PINS
#define RV_MXM_PIN(v) RV_USE(v)
#if RV_SAME
#define RV_MXM_FPIN(s) RV_FUSE(s)
#define RV_MXM_STEP_PIN(v) ((void)0)
#else
/* An "f" asm operand cannot hold a _Float16 / __bf16, so 16-bit inputs pin the
 * step's B vector in place instead (measured: it also keeps the scalar loads
 * ahead; with fp32/fp64 B the same pin makes the allocator copy registers). */
#define RV_MXM_FPIN(s) ((void)0)
#define RV_MXM_STEP_PIN(v) RV_PIN(v)
#endif
#else
#define RV_MXM_PIN(v) ((void)0)
#define RV_MXM_FPIN(s) ((void)0)
#define RV_MXM_STEP_PIN(v) ((void)0)
#endif
/*
 * The accumulators pass through an in-out pin at the start of every step:
 * the step's FMAs consume the pin's outputs, and volatile asms keep their
 * order, so the FMAs cannot be hoisted above the previous step's loads (which
 * the load pins hold in place).
 */
#ifndef RV_MXM_ACC_PINS
#define RV_MXM_ACC_PINS 1
#endif
#if RV_MXM_PINS && RV_MXM_ACC_PINS
#define RV_MXM_ACC_PIN(v) RV_PIN(v)
#else
#define RV_MXM_ACC_PIN(v) ((void)0)
#endif

/* Scalar A element (i, l) of raw input-format storage. */
#define RV_A(r, l) rv_in_scalar(a, (long)(r) * lda + (l))

/*
 * One MR x vl block. MR is a constant at every call (the switch below), so
 * the unused accumulators fold away. Steps alternate between two B buffers
 * and two sets of A scalars (k unrolled by two); each step's loads are issued
 * before the previous step's FMAs and pinned there (RV_USE / RV_FUSE: the
 * scheduler would otherwise sink every load onto its first use).
 */
#define RV_UK_STEP(bv, S)                                         \
    do {                                                          \
        RV_MXM_STEP_PIN(bv);                                      \
        RV_MXM_ACC_PIN(c0);                                       \
        if (MR > 1) RV_MXM_ACC_PIN(c1);                           \
        if (MR > 2) RV_MXM_ACC_PIN(c2);                           \
        if (MR > 3) RV_MXM_ACC_PIN(c3);                           \
        if (MR > 4) RV_MXM_ACC_PIN(c4);                           \
        if (MR > 5) RV_MXM_ACC_PIN(c5);                           \
        c0 = rv_fma_vf(c0, S##0, bv, vl);                         \
        if (MR > 1) c1 = rv_fma_vf(c1, S##1, bv, vl);             \
        if (MR > 2) c2 = rv_fma_vf(c2, S##2, bv, vl);             \
        if (MR > 3) c3 = rv_fma_vf(c3, S##3, bv, vl);             \
        if (MR > 4) c4 = rv_fma_vf(c4, S##4, bv, vl);             \
        if (MR > 5) c5 = rv_fma_vf(c5, S##5, bv, vl);             \
    } while (0)
/* The first step multiplies instead (no accumulator initialisation). */
#define RV_UK_FIRST(bv, S)                                        \
    do {                                                          \
        c0 = rv_mul_vf(S##0, bv, vl);                             \
        if (MR > 1) c1 = rv_mul_vf(S##1, bv, vl);                 \
        if (MR > 2) c2 = rv_mul_vf(S##2, bv, vl);                 \
        if (MR > 3) c3 = rv_mul_vf(S##3, bv, vl);                 \
        if (MR > 4) c4 = rv_mul_vf(S##4, bv, vl);                 \
        if (MR > 5) c5 = rv_mul_vf(S##5, bv, vl);                 \
    } while (0)
/* Load step l: B row l into bv, A column l into the scalars S0..S5, pinned. */
#define RV_UK_LOAD(bv, S, l)                                      \
    do {                                                          \
        bv = rv_load_in(b + (l) * ldb, vl);                       \
        S##0 = RV_A(0, l);                                        \
        if (MR > 1) S##1 = RV_A(1, l);                            \
        if (MR > 2) S##2 = RV_A(2, l);                            \
        if (MR > 3) S##3 = RV_A(3, l);                            \
        if (MR > 4) S##4 = RV_A(4, l);                            \
        if (MR > 5) S##5 = RV_A(5, l);                            \
        RV_MXM_PIN(bv);                                           \
        RV_MXM_FPIN(S##0);                                        \
        if (MR > 1) RV_MXM_FPIN(S##1);                            \
        if (MR > 2) RV_MXM_FPIN(S##2);                            \
        if (MR > 3) RV_MXM_FPIN(S##3);                            \
        if (MR > 4) RV_MXM_FPIN(S##4);                            \
        if (MR > 5) RV_MXM_FPIN(S##5);                            \
    } while (0)

RV_INLINE void rv_ukernel(const int MR, size_t vl, long k, const void *a, long lda, const rv_in_bits_t *b, long ldb,
                          rv_out_t *c, long ldc) {
    rv_acc_v c0, c1, c2, c3, c4, c5;
    rv_in_s p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0; /* even steps */
    rv_in_s q0 = 0, q1 = 0, q2 = 0, q3 = 0, q4 = 0, q5 = 0; /* odd steps */
    rv_in_v b0, b1;
    RV_UK_LOAD(b0, p, 0);
    if (k > 1) {
        RV_UK_LOAD(b1, q, 1);
    }
    RV_UK_FIRST(b0, p);
    long l = 1;
    for (; l + 2 < k; l += 2) { /* steps l (odd, b1/q) and l + 1 (even, b0/p) */
        RV_UK_LOAD(b0, p, l + 1);
        RV_UK_STEP(b1, q);
        RV_UK_LOAD(b1, q, l + 2);
        RV_UK_STEP(b0, p);
    }
    if (l + 1 < k) { /* steps l and l + 1 left */
        RV_UK_LOAD(b0, p, l + 1);
        RV_UK_STEP(b1, q);
        RV_UK_STEP(b0, p);
    } else if (l < k) { /* step l left */
        RV_UK_STEP(b1, q);
    }
    rv_store_out(c, c0, vl);
    if (MR > 1) rv_store_out(c + ldc, c1, vl);
    if (MR > 2) rv_store_out(c + 2 * ldc, c2, vl);
    if (MR > 3) rv_store_out(c + 3 * ldc, c3, vl);
    if (MR > 4) rv_store_out(c + 4 * ldc, c4, vl);
    if (MR > 5) rv_store_out(c + 5 * ldc, c5, vl);
}

/* C = A . B over VL-wide column panels and balanced blocks of at most RV_MR rows; k >= 1. */
RV_INLINE void rv_mxm(long m, long n, long k, const void *a, long lda, const rv_in_bits_t *b, long ldb, rv_out_t *c,
                      long ldc) {
    for (long j = 0, vl; j < n; j += vl) {
        vl = (long)rv_setvl(n - j);
        const rv_in_bits_t *bj = b + j;
        const long blocks = (m + RV_MR - 1) / RV_MR;
        for (long i = 0, bi = 0, mr; i < m; i += mr, bi++) {
            mr = (m - i + (blocks - bi) - 1) / (blocks - bi); /* balanced: 12 rows -> 4 + 4 + 4 */
            const void *ai = (const rv_in_bits_t *)a + i * lda;
            rv_out_t *cij = c + i * ldc + j;
            switch (mr) {
            case 6:
                rv_ukernel(6, vl, k, ai, lda, bj, ldb, cij, ldc);
                break;
            case 5:
                rv_ukernel(5, vl, k, ai, lda, bj, ldb, cij, ldc);
                break;
            case 4:
                rv_ukernel(4, vl, k, ai, lda, bj, ldb, cij, ldc);
                break;
            case 3:
                rv_ukernel(3, vl, k, ai, lda, bj, ldb, cij, ldc);
                break;
            case 2:
                rv_ukernel(2, vl, k, ai, lda, bj, ldb, cij, ldc);
                break;
            default:
                rv_ukernel(1, vl, k, ai, lda, bj, ldb, cij, ldc);
                break;
            }
        }
    }
}

#endif /* NAIAD_RVV_MXM_H */
