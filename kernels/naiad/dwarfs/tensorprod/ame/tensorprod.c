/*
 * tensorprod, AME variant: ur / us / ut of E p^3 elements on the matrix unit.
 *
 * Setup (untimed: the operator is fixed across a whole Nekbone solve) builds the
 * zero-padded block-diagonal operators I_q (x) D (for t and s, used as the "a"
 * operand) and I_q (x) D^T (for r, the "b" operand), and zeroes u's tail. The
 * timed region loads them into tiles 0 and 5, then runs three streams of
 * independent mma: each job loads one tile of u, zeroes an accumulator, does one
 * mma against the resident operator (K = one tile, so no K loop) and stores the
 * result. Tile loads run two jobs ahead and stores one job behind, rotating
 * over stream tiles 1-4 and accumulators 0-3, so the load, mma and store paths
 * overlap. Layouts: tp_layout.h / spec.py.
 */
#include "../tp_layout.h"
#include "matrix.h"

#if NAIAD_MATRIX_M != T || NAIAD_MATRIX_N != T || NAIAD_MATRIX_K != T
#error "tensorprod's AME variant assumes a square 16 x 16 x 16 unit"
#endif
#if NAIAD_MATRIX_TILES < 6 || NAIAD_MATRIX_ACCS < 4
#error "tensorprod's AME variant needs 6 tiles and 4 accumulators"
#endif

#define FI TP_MFMT_IN
#define FA TP_MFMT_ACC
#define ES ((long)sizeof(in_t))
#define EC ((long)sizeof(acc_t))

static in_t Dblk[T * T] __attribute__((aligned(64)));  /* I_q (x) D, padded */
static in_t DTblk[T * T] __attribute__((aligned(64))); /* I_q (x) D^T, padded */

/*
 * The software pipeline over jobs 0..NJOBS-1, slot S = job & 3 (a constant, so tile
 * and accumulator indices fold to immediates): load job+2, zero + mma job, store
 * job-1. LOAD(tile, job), MMA(acc, tile) and STORE(acc, job) are per direction.
 */
#define TP_STEP(S)                                                                                 \
    do {                                                                                           \
        const long j_ = jb + (S);                                                                  \
        if (j_ <= NJOBS) {                                                                         \
            if (j_ + 2 < NJOBS) {                                                                  \
                LOAD(1 + (((S) + 2) & 3), j_ + 2);                                                 \
            }                                                                                      \
            if (j_ < NJOBS) {                                                                      \
                mx_azero(S);                                                                       \
                MMA(S, 1 + (S));                                                                   \
            }                                                                                      \
            if (j_ >= 1) {                                                                         \
                STORE(((S) + 3) & 3, j_ - 1);                                                      \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#define TP_STREAM()                                                                                \
    do {                                                                                           \
        LOAD(1, 0);                                                                                \
        if (NJOBS > 1) {                                                                           \
            LOAD(2, 1);                                                                            \
        }                                                                                          \
        for (long jb = 0; jb <= NJOBS; jb += 4) {                                                  \
            TP_STEP(0);                                                                            \
            TP_STEP(1);                                                                            \
            TP_STEP(2);                                                                            \
            TP_STEP(3);                                                                            \
        }                                                                                          \
    } while (0)

/* t: job = (group g of Q elements, column tile c); b operand = Q*P rows of P^2. */
static void stream_t(void) {
#define NJOBS JOBS_T
#define LOAD(t, j) mx_tload_b(t, FI, u + ((j) / (LDT / T)) * Q * P3 + ((j) % (LDT / T)) * T, P2 * ES)
#define MMA(d, t) mx_mma(d, 0, t, FI, FA)
#define STORE(d, j) mx_astore(d, FA, out_t + ((j) / (LDT / T)) * Q * P * LDT + ((j) % (LDT / T)) * T, LDT * EC)
    TP_STREAM();
#undef NJOBS
#undef LOAD
#undef MMA
#undef STORE
}

/* r: job = T rows of u viewed with row length Q*P; a operand. */
static void stream_r(void) {
#define NJOBS JOBS_R
#define LOAD(t, j) mx_tload_a(t, FI, u + (j) * T * Q * P, Q * P * ES)
#define MMA(d, t) mx_mma(d, t, 5, FI, FA)
#define STORE(d, j) mx_astore(d, FA, out_r + (j) * T * T, T * EC)
    TP_STREAM();
#undef NJOBS
#undef LOAD
#undef MMA
#undef STORE
}

/* s: job = group of Q slabs (e, k), each P rows of P; b operand. */
static void stream_s(void) {
#define NJOBS JOBS_S
#define LOAD(t, j) mx_tload_b(t, FI, u + (j) * Q * P2, P * ES)
#define MMA(d, t) mx_mma(d, 0, t, FI, FA)
#define STORE(d, j) mx_astore(d, FA, out_s + (j) * Q * P * T, T * EC)
    TP_STREAM();
#undef NJOBS
#undef LOAD
#undef MMA
#undef STORE
}

int main(void) {
    tp_dims[0] = P;
    tp_dims[1] = E;
    naiad_zero(Dblk, sizeof(Dblk));
    naiad_zero(DTblk, sizeof(DTblk));
    for (int c = 0; c < Q; c++) {
        for (int i = 0; i < P; i++) {
            for (int l = 0; l < P; l++) {
                Dblk[(c * P + i) * T + c * P + l] = D[i * P + l];
                DTblk[(c * P + l) * T + c * P + i] = D[i * P + l];
            }
        }
    }
    naiad_zero(u + E * P3, sizeof(u) - sizeof(in_t) * E * P3);

    uint64_t t0 = naiad_cycles();
    mx_tload_a(0, FI, Dblk, T * ES);
    mx_tload_b(5, FI, DTblk, T * ES);
    stream_t();
    stream_r();
    stream_s();
    naiad_mx_drain();
    region_cycles = naiad_cycles() - t0;
    return 0;
}
