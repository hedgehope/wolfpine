/*
 * counters: every counter-hub event a kernel can read changes by exactly what
 * the code between two reads did.
 *
 * Each measured region is a noinline function around an inline-asm loop of n
 * iterations, run with n = N and n = 0: the difference of the two deltas is the
 * loop's own count, whatever the compiler put around it (the reads, the call).
 * Matrix and scratchpad events need no calibration, since only the region's own
 * instructions touch them. Events the design binds to a Zihpm counter are read
 * with csrr (naiad_counter folds to one instruction); the rest through the
 * perfcounters block. Needs a design like naiad-counters.yaml.
 */
#include "matrix.h"
#include "naiad_platform.h"

#ifndef NAIAD_HAS_HPM
#error "counters needs a design with counters.hpm bindings (e.g. naiad-counters.yaml)"
#endif

#define N 100
#define TILE_BYTES (NAIAD_MATRIX_M * NAIAD_MATRIX_K * 4)

static uint32_t A[NAIAD_MATRIX_M][NAIAD_MATRIX_K], B[NAIAD_MATRIX_K][NAIAD_MATRIX_N];
static uint32_t C[NAIAD_MATRIX_M][NAIAD_MATRIX_N], T[NAIAD_MATRIX_M][NAIAD_MATRIX_K];
static uint64_t dram_buf[N];

static int check(const char *what, long long got, long long want) {
    naiad_printf("  %s: %lld (expect %lld) %s\n", what, got, want, got == want ? "ok" : "MISMATCH");
    return got == want;
}

/* 3n scalar instructions (beqz, addi, j) plus a constant. */
static __attribute__((noinline)) uint64_t scalar_loop(long n) {
    uint64_t before = naiad_counter(INSTRET_SCALAR);
    __asm__ volatile("1: beqz %0, 2f\n addi %0, %0, -1\n j 1b\n2:" : "+r"(n));
    return naiad_counter(INSTRET_SCALAR) - before;
}

/* n fadd.d plus 3n scalar instructions; returns the fp, scalar and total deltas. */
static __attribute__((noinline)) void fp_loop(long n, uint64_t out[3]) {
    double x = 1.0;
    uint64_t fp = naiad_counter(INSTRET_FP);
    uint64_t scalar = naiad_counter(INSTRET_SCALAR);
    uint64_t total = naiad_instret();
    __asm__ volatile("1: beqz %0, 2f\n fadd.d %1, %1, %1\n addi %0, %0, -1\n j 1b\n2:" : "+r"(n), "+f"(x));
    out[2] = naiad_instret() - total;
    out[1] = naiad_counter(INSTRET_SCALAR) - scalar;
    out[0] = naiad_counter(INSTRET_FP) - fp;
}

/* n 8-byte stores to (or loads from) p: the bus.dram write (read) byte delta. */
static __attribute__((noinline)) uint64_t dram_store_loop(volatile uint64_t *p, long n) {
    uint64_t before = naiad_counter(BUS_DRAM_WRITE_BYTES);
    __asm__ volatile("1: beqz %0, 2f\n sd zero, 0(%1)\n addi %1, %1, 8\n addi %0, %0, -1\n j 1b\n2:"
                     : "+r"(n), "+r"(p)
                     :
                     : "memory");
    return naiad_counter(BUS_DRAM_WRITE_BYTES) - before;
}

static __attribute__((noinline)) uint64_t dram_load_loop(volatile uint64_t *p, long n) {
    uint64_t before = naiad_counter(BUS_DRAM_READ_BYTES);
    __asm__ volatile("1: beqz %0, 2f\n ld t0, 0(%1)\n addi %1, %1, 8\n addi %0, %0, -1\n j 1b\n2:"
                     : "+r"(n), "+r"(p)
                     :
                     : "t0", "memory");
    return naiad_counter(BUS_DRAM_READ_BYTES) - before;
}

int main(void) {
    int ok = 1;
    naiad_printf("counters on %s\n", NAIAD_DESIGN);

    /* Instruction classes. */
    long long scalar = (long long)(scalar_loop(N) - scalar_loop(0));
    ok &= check("instret.scalar, 3-instruction loop x N", scalar, 3 * N);
    uint64_t with[3], without[3];
    fp_loop(N, with);
    fp_loop(0, without);
    ok &= check("instret.fp, N fadd.d", (long long)(with[0] - without[0]), N);
    ok &= check("instret.scalar around them", (long long)(with[1] - without[1]), 3 * N);
    ok &= check("minstret = scalar + fp", (long long)(with[2] - without[2]), 4 * N);

    /* Matrix: five matrix instructions, one mma, and the unit's own traffic. */
    for (int i = 0; i < NAIAD_MATRIX_M; i++) {
        for (int k = 0; k < NAIAD_MATRIX_K; k++) {
            A[i][k] = 0x3F800000u; /* 1.0f */
            B[k][i] = 0x40000000u; /* 2.0f */
        }
    }
    uint64_t m_instr = naiad_counter(INSTRET_MATRIX), m_mma = naiad_counter(MATRIX_MMA);
    uint64_t m_macs = naiad_counter(MATRIX_MACS), m_in = naiad_counter(MATRIX_MEM_TO_TILE_BYTES);
    uint64_t m_out = naiad_counter(MATRIX_ACC_TO_MEM_BYTES), m_tout = naiad_counter(MATRIX_TILE_TO_MEM_BYTES);
    uint64_t m_mma_mmio = naiad_perf_read(NAIAD_PERFOFF_MATRIX_MMA);
    uint64_t m_bus = naiad_counter(BUS_DRAM_READ_BYTES);
    mx_azero(0);
    mx_tload_a(0, NAIAD_MFMT_FP32, A, sizeof A[0]);
    mx_tload_b(1, NAIAD_MFMT_FP32, B, sizeof B[0]);
    mx_mma(0, 0, 1, NAIAD_MFMT_FP32, NAIAD_MFMT_FP32);
    mx_astore(0, NAIAD_MFMT_FP32, C, sizeof C[0]);
    mx_tstore_a(0, NAIAD_MFMT_FP32, T, sizeof T[0]);
    uint64_t bus = naiad_counter(BUS_DRAM_READ_BYTES) - m_bus;
    ok &= check("instret.matrix, 6 matrix instructions", (long long)(naiad_counter(INSTRET_MATRIX) - m_instr), 6);
    ok &= check("matrix.mma (hpm)", (long long)(naiad_counter(MATRIX_MMA) - m_mma), 1);
    ok &= check("matrix.mma (mmio)", (long long)(naiad_perf_read(NAIAD_PERFOFF_MATRIX_MMA) - m_mma_mmio), 1);
    ok &= check("matrix.macs", (long long)(naiad_counter(MATRIX_MACS) - m_macs),
                NAIAD_MATRIX_M * NAIAD_MATRIX_N * NAIAD_MATRIX_K);
    ok &= check("matrix.mem_to_tile_bytes", (long long)(naiad_counter(MATRIX_MEM_TO_TILE_BYTES) - m_in),
                2 * TILE_BYTES);
    ok &= check("matrix.acc_to_mem_bytes", (long long)(naiad_counter(MATRIX_ACC_TO_MEM_BYTES) - m_out),
                NAIAD_MATRIX_M * NAIAD_MATRIX_N * 4);
    ok &= check("matrix.tile_to_mem_bytes (mmio)", (long long)(naiad_counter(MATRIX_TILE_TO_MEM_BYTES) - m_tout),
                TILE_BYTES);
    ok &= check("tile loads not on the scalar bus", bus < 2 * TILE_BYTES, 1);
    ok &= check("C[0][0] = K * 2.0f", C[0][0] == 0x42000000u, 1);

    /* Bytes per block: DRAM (calibrated), and the scratchpad (only the loops touch it). */
    ok &= check("bus.dram.write_bytes, N sd",
                (long long)(dram_store_loop(dram_buf, N) - dram_store_loop(dram_buf, 0)), 8 * N);
    ok &= check("bus.dram.read_bytes, N ld",
                (long long)(dram_load_loop(dram_buf, N) - dram_load_loop(dram_buf, 0)), 8 * N);
    volatile uint64_t *scratch = (volatile uint64_t *)(uintptr_t)NAIAD_SCRATCH_BASE;
    uint64_t sw = naiad_counter(BUS_SCRATCH_WRITE_BYTES), sr = naiad_counter(BUS_SCRATCH_READ_BYTES);
    for (int i = 0; i < N; i++) {
        scratch[i] = i;
    }
    uint64_t sum = 0;
    for (int i = 0; i < N; i++) {
        sum += scratch[i];
    }
    ok &= check("bus.scratch.write_bytes", (long long)(naiad_counter(BUS_SCRATCH_WRITE_BYTES) - sw), 8 * N);
    ok &= check("bus.scratch.read_bytes (mmio)", (long long)(naiad_counter(BUS_SCRATCH_READ_BYTES) - sr), 8 * N);
    ok &= check("scratch sum", (long long)sum, N * (N - 1) / 2);

    /* An IP block's own counter (MMIO), and Zihpm write/event semantics. */
    uint64_t chars = naiad_counter(HTIF_CHARS);
    naiad_puts("  abc\n");
    ok &= check("htif.chars after 6 characters", (long long)(naiad_counter(HTIF_CHARS) - chars), 6);
    uint64_t written, event3, event20, unbound;
    /* csrw sets the offset, then retires: the next csrr sees 1000 + 1. */
    __asm__ volatile("csrw mhpmcounter3, %1\n csrr %0, hpmcounter3" : "=r"(written) : "r"(1000L));
    __asm__ volatile("csrr %0, mhpmevent3" : "=r"(event3));
    __asm__ volatile("csrr %0, mhpmevent20" : "=r"(event20));
    __asm__ volatile("csrr %0, hpmcounter20" : "=r"(unbound));
    ok &= check("mhpmcounter3 written to 1000, read after one more", (long long)written, 1001);
    ok &= check("mhpmevent3 (bound)", (long long)event3, 3);
    ok &= check("mhpmevent20 / hpmcounter20 (unbound)", (long long)(event20 | unbound), 0);
    ok &= check("instret.vector (no RVV yet)", (long long)naiad_counter(INSTRET_VECTOR), 0);

    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
