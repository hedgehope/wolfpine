/*
 * scan_rvv: the RVV baseline for scan_ame: y = inclusive prefix sum of x.
 *
 * x is taken in chunks of VLMAX elements at LMUL 2 (16 fp32 / 8 fp64 at VLEN
 * 256), four chunks at a time so their independent work interleaves on the
 * in-order core. Each chunk is scanned in registers by the log-step
 * (Hillis-Steele) scan: log2(VL) rounds, s = 1, 2, 4, ..., of
 * v[i] += v[i - s] for i >= s: a vslideup by s (into an undefined register, so
 * no zero fill is copied) and a vfadd masked by vid >= s (mask-undisturbed, so
 * the low s lanes keep their values). The round masks are built once and
 * reloaded with vlm.v (a load-unit op, off the datapath), one per round for
 * all four chunks, and the four slides issue before the four adds so no
 * chunk waits on its own slide. The chunk's total (its last element, vslidedown +
 * vfmv.f.s) feeds a scalar carry chain, which is the only serial dependence
 * (one fadd per chunk), and the chunk is stored as carry + v (vfadd.vf).
 * 16-bit inputs are widened once on load (vfwcvt / vfwcvtbf16).
 *
 * Why LMUL 2: per element the scan costs (2 log2 VL + 3) vector ops of VL / 8
 * datapath cycles each. LMUL 1 needs fewer datapath cycles (1.1 / element)
 * but issues about 2 instructions per datapath cycle, so the single-issue
 * front end bounds it; LMUL 2 (1.4) is the fastest that stays datapath bound;
 * LMUL 4 is 1.6.
 *
 * Rounding differs from the sequential loop: within a chunk the sums form a
 * tree, and each output is carry + (in-chunk prefix), the carry being the
 * sequential sum of the chunk totals.
 */
#include "naiad_platform.h"

#include "../../jacobi/dw_fmt.h"

#define RV_AL 2
#include "rvv_fmt.h"

#ifndef SCAN_N
#error "define SCAN_N"
#endif
#define N SCAN_N

NAIAD_IO dw_in_t x[N];
NAIAD_IO dw_acc_t y[N];
NAIAD_IO uint64_t region_cycles;

/* The last of v's first vl elements. */
RV_INLINE rv_acc_s last(rv_acc_v v, size_t vl) { return __riscv_vfmv_f(__riscv_vslidedown(v, vl - 1, vl)); }

#if RV_AL != 2
#error "scan_rvv's mask and index types are written for LMUL 2"
#endif
#if RV_AE == 64
typedef vbool32_t rv_mask_t;
typedef vuint64m2_t rv_idx_v;
#define rv_vid __riscv_vid_v_u64m2
#define rv_vundef __riscv_vundefined_f64m2
#else
typedef vbool16_t rv_mask_t;
typedef vuint32m2_t rv_idx_v;
#define rv_vid __riscv_vid_v_u32m2
#define rv_vundef __riscv_vundefined_f32m2
#endif
#if RV_AE == 64
#define rv_vlm __riscv_vlm_v_b32
#define rv_vsm __riscv_vsm_v_b32
#else
#define rv_vlm __riscv_vlm_v_b16
#define rv_vsm __riscv_vsm_v_b16
#endif

/* masks[k]: vid >= 2^k, for each scan round k (VL <= 2048 elements). */
#define MAX_ROUNDS 11
static uint8_t masks[MAX_ROUNDS][256] __attribute__((aligned(64)));

/* One scan round: v[i] += v[i - s] for i >= s (m = vid >= s). */
RV_INLINE rv_acc_v round1(rv_acc_v v, rv_mask_t m, size_t s, size_t vl) {
    rv_acc_v t = __riscv_vslideup(rv_vundef(), v, s, vl);
    return __riscv_vfadd_mu(m, v, v, t, vl);
}

/* In-register inclusive scan of v's first vl elements. */
RV_INLINE rv_acc_v scan1(rv_acc_v v, rv_idx_v id, size_t vl) {
    for (size_t s = 1; s < vl; s <<= 1) {
        v = round1(v, __riscv_vmsgeu(id, s, vl), s, vl);
    }
    return v;
}

/* One scan round of the four chunks a..d: shift sh, mask masks[k]; the slides issue before the adds. */
#define ROUND(k, sh)                                                         \
    do {                                                                     \
        rv_mask_t m_ = rv_vlm(masks[k], vlmax);                              \
        rv_acc_v sa_ = __riscv_vslideup(rv_vundef(), a, (sh), vlmax);        \
        rv_acc_v sb_ = __riscv_vslideup(rv_vundef(), b, (sh), vlmax);        \
        rv_acc_v sc_ = __riscv_vslideup(rv_vundef(), c, (sh), vlmax);        \
        rv_acc_v sd_ = __riscv_vslideup(rv_vundef(), d, (sh), vlmax);        \
        a = __riscv_vfadd_mu(m_, a, a, sa_, vlmax);                          \
        b = __riscv_vfadd_mu(m_, b, b, sb_, vlmax);                          \
        c = __riscv_vfadd_mu(m_, c, c, sc_, vlmax);                          \
        d = __riscv_vfadd_mu(m_, d, d, sd_, vlmax);                          \
    } while (0)

/*
 * Scan groups of four chunks from x[i] on while a whole group fits, updating i
 * and carry; ROUNDS is the statement that runs every round on a..d. The next
 * group's loads issue after the first round (software pipelining: the
 * in-order core would otherwise wait out the load latency every group); the
 * last group reloads itself instead of reading past x.
 */
#define LOAD4(j)                                                                                                   \
    do {                                                                                                           \
        na = rv_widen(rv_load_in(&x[(j)], vlmax), vlmax);                                                          \
        nb = rv_widen(rv_load_in(&x[(j) + vlmax], vlmax), vlmax);                                                  \
        nc = rv_widen(rv_load_in(&x[(j) + 2 * vlmax], vlmax), vlmax);                                              \
        nd = rv_widen(rv_load_in(&x[(j) + 3 * vlmax], vlmax), vlmax);                                              \
    } while (0)
#define SCAN_GROUPS(ROUND0, ROUNDS)                                                                                \
    if (i + 4 * vlmax <= N) {                                                                                      \
        rv_acc_v na, nb, nc, nd;                                                                                   \
        LOAD4(i);                                                                                                  \
        for (; i + 4 * vlmax <= N; i += 4 * vlmax) {                                                               \
            rv_acc_v a = na, b = nb, c = nc, d = nd;                                                               \
            ROUND0;                                                                                                \
            const long next = i + 8 * vlmax <= N ? i + 4 * vlmax : i;                                              \
            LOAD4(next);                                                                                           \
            ROUNDS;                                                                                                \
            /* The totals first: they are off the carry chain. */                                                  \
            rv_acc_s ta = last(a, vlmax), tb = last(b, vlmax), tc = last(c, vlmax), td = last(d, vlmax);           \
            rv_store_acc(&y[i], __riscv_vfadd(a, carry, vlmax), vlmax);                                            \
            carry += ta;                                                                                           \
            rv_store_acc(&y[i + vlmax], __riscv_vfadd(b, carry, vlmax), vlmax);                                    \
            carry += tb;                                                                                           \
            rv_store_acc(&y[i + 2 * vlmax], __riscv_vfadd(c, carry, vlmax), vlmax);                                \
            carry += tc;                                                                                           \
            rv_store_acc(&y[i + 3 * vlmax], __riscv_vfadd(d, carry, vlmax), vlmax);                                \
            carry += td;                                                                                           \
        }                                                                                                          \
    }

/* VLMAX on the smallest VLEN the build allows (zvl<N>b): the rounds are unrolled for it. */
#define MIN_VLMAX (__riscv_v_min_vlen * RV_AL / RV_AE)

RV_KERNEL int main(void) {
    uint64_t t0 = naiad_cycles();
    const long vlmax = (long)rv_vlmax();
    const rv_idx_v id = rv_vid(vlmax);
    long rounds = 0;
    while ((1L << rounds) < vlmax) {
        rv_vsm(masks[rounds], __riscv_vmsgeu(id, (size_t)1 << rounds, vlmax), vlmax);
        rounds++;
    }
    rv_acc_s carry = 0;
    long i = 0;
    if (vlmax == MIN_VLMAX) { /* VLEN is the minimum: unrolled rounds, immediate shifts */
        SCAN_GROUPS(ROUND(0, 1), {
#if MIN_VLMAX > 2
            ROUND(1, 2);
#endif
#if MIN_VLMAX > 4
            ROUND(2, 4);
#endif
#if MIN_VLMAX > 8
            ROUND(3, 8);
#endif
#if MIN_VLMAX > 16
            ROUND(4, 16);
#endif
#if MIN_VLMAX > 32
#error "unroll more rounds for this VLEN"
#endif
        })
    } else { /* any other VLEN */
        SCAN_GROUPS(ROUND(0, 1), for (long k = 1; k < rounds; k++) ROUND(k, (size_t)1 << k))
    }
    for (size_t vl; i < N; i += (long)vl) { /* tail, one chunk at a time */
        vl = rv_setvl(N - i);
        rv_acc_v a = scan1(rv_widen(rv_load_in(&x[i], vl), vl), id, vl);
        rv_store_acc(&y[i], __riscv_vfadd(a, carry, vl), vl);
        carry += last(a, vl);
    }
    region_cycles = naiad_cycles() - t0;
    return 0;
}
