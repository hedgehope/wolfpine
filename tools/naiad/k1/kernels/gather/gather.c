/*
 * gather: the irregular-access instructions the dwarfs lean on.
 *   strided    vlse32: a column of a row-major matrix
 *   gather     vluxei32 / vloxei32 with a permutation (and repeats)
 *   scatter    vsoxei32 to a permutation
 *   segment    vlseg2e32: deinterleave (re, im) pairs; vsseg2e32 re-interleave swapped
 *   rgather    vrgather.vv within a register group (a reversal)
 *   compress   vcompress.vm of the positive elements
 */
#include "k1common.h"

#define ROWS 70
#define COLS 13
#define NP 100

static float mat[ROWS * COLS], col[ROWS];
static float src[NP], gat[NP], gat_o[NP], sct[NP], cpx[2 * NP], re[NP], im[NP], swapped[2 * NP];
static uint32_t perm[NP], rep[NP];
static float rev[64], comp[NP];
static uint32_t ncomp;

static void kernels(void) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    for (size_t i = 0; i < ROWS;) {
        size_t vl = __riscv_vsetvl_e32m2(ROWS - i < vlmax ? ROWS - i : vlmax);
        __riscv_vse32_v_f32m2(col + i, __riscv_vlse32_v_f32m2(mat + i * COLS + 5, COLS * sizeof(float), vl), vl);
        i += vl;
    }
    for (size_t i = 0; i < NP;) {
        size_t vl = __riscv_vsetvl_e32m2(NP - i < vlmax ? NP - i : vlmax);
        vuint32m2_t idx = __riscv_vsll_vx_u32m2(__riscv_vle32_v_u32m2(perm + i, vl), 2, vl);
        vuint32m2_t ridx = __riscv_vsll_vx_u32m2(__riscv_vle32_v_u32m2(rep + i, vl), 2, vl);
        __riscv_vse32_v_f32m2(gat + i, __riscv_vluxei32_v_f32m2(src, idx, vl), vl);
        __riscv_vse32_v_f32m2(gat_o + i, __riscv_vloxei32_v_f32m2(src, ridx, vl), vl);
        __riscv_vsoxei32_v_f32m2(sct, idx, __riscv_vle32_v_f32m2(src + i, vl), vl);
        vfloat32m2x2_t seg = __riscv_vlseg2e32_v_f32m2x2(cpx + 2 * i, vl);
        vfloat32m2_t r = __riscv_vget_v_f32m2x2_f32m2(seg, 0), m = __riscv_vget_v_f32m2x2_f32m2(seg, 1);
        __riscv_vse32_v_f32m2(re + i, r, vl);
        __riscv_vse32_v_f32m2(im + i, m, vl);
        __riscv_vsseg2e32_v_f32m2x2(swapped + 2 * i, __riscv_vcreate_v_f32m2x2(m, r), vl);
        i += vl;
    }
    size_t vl = __riscv_vsetvl_e32m8(64);
    vuint32m8_t ridx = __riscv_vrsub_vx_u32m8(__riscv_vid_v_u32m8(vl), (uint32_t)(vl - 1), vl);
    __riscv_vse32_v_f32m8(rev, __riscv_vrgather_vv_f32m8(__riscv_vle32_v_f32m8(src, vl), ridx, vl), vl);
    vfloat32m8_t v = __riscv_vle32_v_f32m8(src, vl);
    vbool4_t pos = __riscv_vmfgt_vf_f32m8_b4(v, 0.0f, vl);
    ncomp = (uint32_t)__riscv_vcpop_m_b4(pos, vl);
    __riscv_vse32_v_f32m8(comp, __riscv_vcompress_vm_f32m8(v, pos, vl), ncomp);
}

int main(void) {
    k1_header("gather");
    k1_seed(42);
    for (size_t i = 0; i < ROWS * COLS; i++) mat[i] = k1_randf();
    for (size_t i = 0; i < NP; i++) {
        src[i] = k1_randf();
        perm[i] = (uint32_t)i;
        rep[i] = k1_rand32() % NP;
    }
    for (size_t i = NP - 1; i > 0; i--) { /* Fisher-Yates */
        size_t j = k1_rand32() % (i + 1);
        uint32_t t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
    for (size_t i = 0; i < 2 * NP; i++) cpx[i] = k1_randf();
    kernels();
    k1_emit_f32("strided_col5", col, ROWS);
    k1_emit_f32("gather_perm", gat, NP);
    k1_emit_f32("gather_ordered_repeats", gat_o, NP);
    k1_emit_f32("scatter_perm", sct, NP);
    k1_emit_f32("segment_re", re, NP);
    k1_emit_f32("segment_im", im, NP);
    k1_emit_f32("segment_swapped", swapped, 2 * NP);
    k1_emit_f32("rgather_reverse", rev, 64);
    k1_emit_u32("compress_count", &ncomp, 1);
    k1_emit_f32("compress_values", comp, ncomp);
    uint64_t t0 = naiad_cycles();
    for (unsigned r = 0; r < K1_REPS; r++) {
        kernels();
        K1_CLOBBER();
    }
    uint64_t t1 = naiad_cycles();
    k1_time("gather_all", t0, t1, K1_REPS);
    return 0;
}
