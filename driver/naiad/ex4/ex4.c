/*
 * ex4: reading the simulator's counters from a program. Design: naiad-counters
 * (RV64IMAFD + Zihpm, an AME unit, a perfcounters block).
 *
 * Three ways in, all reading the same counter hub the end-of-run report prints:
 *   naiad_cycles() / naiad_instret()   the Zicntr cycle / instret CSRs;
 *   naiad_counter(EVENT)               any hub event: one csrr when the design binds
 *                                      it to a Zihpm hpmcounter (counters.hpm), else
 *                                      a load from the perfcounters block;
 *   naiad_perf_read(NAIAD_PERF_X)      a perfcounters register by its offset.
 *
 * The measured region is a dot product. Its own instruction count is the
 * difference between running it on N elements and on 0, which cancels the
 * calls and counter reads around it. That count is checked against the loop's
 * hand-counted shape (an asm loop, so the compiler cannot reshape it), and
 * the DRAM read bytes against the 2 * N * 4 bytes it loads. naiad-counters
 * has no timing: section, so every instruction is one cycle (IPC 1.00). Run
 * with --set timing.scalar.load=4: any timing field turns the cycle model on
 * (the rest at their defaults, including DRAM latency), and cycles and IPC
 * move while instret and the byte counts stay put.
 */
#include "naiad_platform.h"

#define N 256

static int32_t x[N], y[N];

struct delta {
    uint64_t cycles, instret, scalar, dram_read, perf_cycles;
};

/* 8 instructions per element (two address adds, two lw, mul, add, addi, bne); the
 * leading beqz runs for n = 0 too, so it cancels in the difference. */
static __attribute__((noinline)) int64_t dot(const int32_t *a, const int32_t *b, long n, struct delta *d) {
    int64_t sum = 0;
    long off = 0, end = n * 4;
    uint64_t c = naiad_cycles(), i = naiad_instret(), s = naiad_counter(INSTRET_SCALAR);
    uint64_t r = naiad_counter(BUS_DRAM_READ_BYTES), p = naiad_perf_read(NAIAD_PERF_CYCLE);
    __asm__ volatile("   beqz %[e], 2f\n"
                     "1: add t0, %[a], %[o]\n"
                     "   add t1, %[b], %[o]\n"
                     "   lw t0, 0(t0)\n"
                     "   lw t1, 0(t1)\n"
                     "   mul t0, t0, t1\n"
                     "   add %[s], %[s], t0\n"
                     "   addi %[o], %[o], 4\n"
                     "   bne %[o], %[e], 1b\n"
                     "2:"
                     : [s] "+r"(sum), [o] "+r"(off)
                     : [a] "r"(a), [b] "r"(b), [e] "r"(end)
                     : "t0", "t1", "memory");
    d->perf_cycles = naiad_perf_read(NAIAD_PERF_CYCLE) - p;
    d->dram_read = naiad_counter(BUS_DRAM_READ_BYTES) - r;
    d->scalar = naiad_counter(INSTRET_SCALAR) - s;
    d->instret = naiad_instret() - i;
    d->cycles = naiad_cycles() - c;
    return sum;
}

static int check(const char *what, long got, long want) {
    naiad_printf("  %s: %ld (expect %ld) %s\n", what, got, want, got == want ? "ok" : "MISMATCH");
    return got == want;
}

int main(void) {
    int ok = 1;
    int64_t want = 0;
    for (int i = 0; i < N; i++) {
        x[i] = i - 100;
        y[i] = 3 * i + 7;
        want += (int64_t)x[i] * y[i];
    }
    naiad_printf("ex4: counters on %s\n", NAIAD_DESIGN);

    struct delta full, empty;
    int64_t got = dot(x, y, N, &full);
    dot(x, y, 0, &empty);
    long loop_instret = (long)(full.instret - empty.instret);
    long loop_cycles = (long)(full.cycles - empty.cycles);
    naiad_printf("  dot product of %d: %ld cycles, %ld instructions, IPC %ld.%02ld\n", N, loop_cycles, loop_instret,
                 loop_instret / loop_cycles, loop_instret * 100 / loop_cycles % 100);

    ok &= check("dot product", (long)got, (long)want);
    ok &= check("minstret: 8 instructions x N", loop_instret, 8 * N);
    ok &= check("instret.scalar (hpm) agrees", (long)(full.scalar - empty.scalar), 8 * N);
    ok &= check("bus.dram.read_bytes: 2 x N x 4", (long)(full.dram_read - empty.dram_read), 8 * N);
    ok &= check("cycles >= instret (one issue per cycle)", loop_cycles >= loop_instret, 1);
    /* The perfcounters block's cycle counts the same clock as mcycle (its reads sit
     * inside the CSR reads, so compare the loops' own counts, not the windows). */
    ok &= check("perf block cycles agree with mcycle", (long)(full.perf_cycles - empty.perf_cycles), loop_cycles);

    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
