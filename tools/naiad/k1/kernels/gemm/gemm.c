/*
 * gemm: the classic RVV outer-product micro-kernel (C[MR x NR] += A[:, k] * B[k, :]
 * with vfmacc.vf, NR = one LMUL-2 register group = 16 fp32 at VLEN 256), driven
 * over a small blocked GEMM with M/N edges, in fp32 and int32 (vmacc.vx).
 */
#include "k1common.h"

#define MR 4
#define M 37
#define N 45
#define K 33

static float Af[M * K], Bf[K * N], Cf[M * N];
static int32_t Ai[M * K], Bi[K * N], Ci[M * N];

/* C[mr x nr] (row stride ldc) += A[mr x K] (row stride K) * B[K x nr] (row stride N). */
static void ukernel_f32(size_t mr, size_t nr, const float *A, const float *B, float *C) {
    size_t vl = __riscv_vsetvl_e32m2(nr);
    vfloat32m2_t c0 = __riscv_vle32_v_f32m2(C + 0 * N, vl);
    vfloat32m2_t c1 = mr > 1 ? __riscv_vle32_v_f32m2(C + 1 * N, vl) : c0;
    vfloat32m2_t c2 = mr > 2 ? __riscv_vle32_v_f32m2(C + 2 * N, vl) : c0;
    vfloat32m2_t c3 = mr > 3 ? __riscv_vle32_v_f32m2(C + 3 * N, vl) : c0;
    for (size_t k = 0; k < K; k++) {
        vfloat32m2_t b = __riscv_vle32_v_f32m2(B + k * N, vl);
        c0 = __riscv_vfmacc_vf_f32m2(c0, A[0 * K + k], b, vl);
        if (mr > 1) c1 = __riscv_vfmacc_vf_f32m2(c1, A[1 * K + k], b, vl);
        if (mr > 2) c2 = __riscv_vfmacc_vf_f32m2(c2, A[2 * K + k], b, vl);
        if (mr > 3) c3 = __riscv_vfmacc_vf_f32m2(c3, A[3 * K + k], b, vl);
    }
    __riscv_vse32_v_f32m2(C + 0 * N, c0, vl);
    if (mr > 1) __riscv_vse32_v_f32m2(C + 1 * N, c1, vl);
    if (mr > 2) __riscv_vse32_v_f32m2(C + 2 * N, c2, vl);
    if (mr > 3) __riscv_vse32_v_f32m2(C + 3 * N, c3, vl);
}

static void ukernel_i32(size_t mr, size_t nr, const int32_t *A, const int32_t *B, int32_t *C) {
    size_t vl = __riscv_vsetvl_e32m2(nr);
    vint32m2_t c0 = __riscv_vle32_v_i32m2(C + 0 * N, vl);
    vint32m2_t c1 = mr > 1 ? __riscv_vle32_v_i32m2(C + 1 * N, vl) : c0;
    vint32m2_t c2 = mr > 2 ? __riscv_vle32_v_i32m2(C + 2 * N, vl) : c0;
    vint32m2_t c3 = mr > 3 ? __riscv_vle32_v_i32m2(C + 3 * N, vl) : c0;
    for (size_t k = 0; k < K; k++) {
        vint32m2_t b = __riscv_vle32_v_i32m2(B + k * N, vl);
        c0 = __riscv_vmacc_vx_i32m2(c0, A[0 * K + k], b, vl);
        if (mr > 1) c1 = __riscv_vmacc_vx_i32m2(c1, A[1 * K + k], b, vl);
        if (mr > 2) c2 = __riscv_vmacc_vx_i32m2(c2, A[2 * K + k], b, vl);
        if (mr > 3) c3 = __riscv_vmacc_vx_i32m2(c3, A[3 * K + k], b, vl);
    }
    __riscv_vse32_v_i32m2(C + 0 * N, c0, vl);
    if (mr > 1) __riscv_vse32_v_i32m2(C + 1 * N, c1, vl);
    if (mr > 2) __riscv_vse32_v_i32m2(C + 2 * N, c2, vl);
    if (mr > 3) __riscv_vse32_v_i32m2(C + 3 * N, c3, vl);
}

static void gemm_f32(void) {
    size_t nr_max = __riscv_vsetvlmax_e32m2();
    for (size_t i = 0; i < M; i += MR) {
        for (size_t j = 0; j < N; j += nr_max) {
            size_t mr = M - i < MR ? M - i : MR, nr = N - j < nr_max ? N - j : nr_max;
            ukernel_f32(mr, nr, Af + i * K, Bf + j, Cf + i * N + j);
        }
    }
}

static void gemm_i32(void) {
    size_t nr_max = __riscv_vsetvlmax_e32m2();
    for (size_t i = 0; i < M; i += MR) {
        for (size_t j = 0; j < N; j += nr_max) {
            size_t mr = M - i < MR ? M - i : MR, nr = N - j < nr_max ? N - j : nr_max;
            ukernel_i32(mr, nr, Ai + i * K, Bi + j, Ci + i * N + j);
        }
    }
}

int main(void) {
    k1_header("gemm");
    k1_seed(7);
    for (size_t i = 0; i < M * K; i++) {
        Af[i] = k1_randf();
        Ai[i] = (int32_t)k1_rand32() >> 20;
    }
    for (size_t i = 0; i < K * N; i++) {
        Bf[i] = k1_randf();
        Bi[i] = (int32_t)k1_rand32() >> 20;
    }
    for (size_t i = 0; i < M * N; i++) {
        Cf[i] = k1_randf();
        Ci[i] = (int32_t)k1_rand32();  /* wraps: int32 overflow is modular in vmacc */
    }
    gemm_f32();
    gemm_i32();
    k1_emit_f32("gemm_f32_37x45x33", Cf, M * N);
    k1_emit_u32("gemm_i32_37x45x33", (const uint32_t *)Ci, M * N);
    uint64_t t0 = naiad_cycles();
    unsigned reps = K1_REPS / 10 ? K1_REPS / 10 : 1;
    for (unsigned r = 0; r < reps; r++) {
        gemm_f32();
        K1_CLOBBER();
    }
    uint64_t t1 = naiad_cycles();
    k1_time("gemm_f32_37x45x33", t0, t1, reps);
    return 0;
}
