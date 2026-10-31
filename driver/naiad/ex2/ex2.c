/*
 * ex2: the CLINT timer. Design: naiad-base-rv64 (one RV64IM hart).
 *
 * The CLINT (SiFive layout) has one shared 64-bit mtime, which counts SoC
 * cycles / NAIAD_CLINT_CYCLES_PER_TICK, and a 64-bit mtimecmp per hart. The
 * timer is "due" for hart h when mtime >= mtimecmp[h]. This program sets
 * mtimecmp DELAY ticks ahead, enables the timer in mie.MTIE and waits with
 *
 *     while (mtime < target) wfi;
 *
 * which is the portable idiom: wfi may return early, so the condition is
 * re-checked. mstatus.MIE stays clear throughout.
 *
 * With mstatus.MIE clear a pending, enabled interrupt only ends a wfi; no trap is
 * taken (ex7 sets up mtvec and takes them). With the timer enabled the hart really
 * sleeps in wfi until mtime >= mtimecmp, so the loop below runs about once. The
 * same program behaves the same on Spike.
 */
#include "naiad_platform.h"

#define DELAY 2000
#define MIE_MTIE (1UL << 7)

static volatile uint64_t *clint64(unsigned long offset) {
    return (volatile uint64_t *)(NAIAD_CLINT_ADDR + offset);
}

static int check(const char *what, long got, long want) {
    naiad_printf("  %s: %ld (expect %ld) %s\n", what, got, want, got == want ? "ok" : "MISMATCH");
    return got == want;
}

int main(void) {
    int ok = 1;
    volatile uint64_t *mtime = clint64(NAIAD_CLINT_MTIME);
    volatile uint64_t *mtimecmp = clint64(NAIAD_CLINT_MTIMECMP + 8 * naiad_hartid());
    naiad_printf("ex2: CLINT timer on %s, %lu cycle(s) per mtime tick\n", NAIAD_DESIGN,
                 (unsigned long)NAIAD_CLINT_CYCLES_PER_TICK);

    /* mtime is the SoC clock: over a stretch of code it moves with mcycle. */
    uint64_t t0 = *mtime, c0 = naiad_cycles();
    for (volatile int i = 0; i < 100; i++) {
    }
    uint64_t dt = *mtime - t0, dc = naiad_cycles() - c0;
    naiad_printf("  100-iteration loop: %lu mtime ticks, %lu cycles\n", (unsigned long)dt, (unsigned long)dc);
    ok &= check("mtime and mcycle agree to within 2", dt * NAIAD_CLINT_CYCLES_PER_TICK + 2 >= dc &&
                                                          dc + 2 >= dt * NAIAD_CLINT_CYCLES_PER_TICK, 1);

    /* Arm the timer and wait for it. */
    uint64_t target = *mtime + DELAY;
    *mtimecmp = target;
    ok &= check("mtimecmp reads back", *mtimecmp == target, 1);
    __asm__ volatile("csrs mie, %0" ::"r"(MIE_MTIE));
    unsigned long mie;
    __asm__ volatile("csrr %0, mie" : "=r"(mie));
    ok &= check("mie.MTIE set", (mie & MIE_MTIE) != 0, 1);

    unsigned long wfis = 0;
    while (*mtime < target) {
        __asm__ volatile("wfi");
        wfis++;
    }
    uint64_t woke = *mtime;
    __asm__ volatile("csrc mie, %0" ::"r"(MIE_MTIE));

    naiad_printf("  armed for mtime %lu, woke at %lu after %lu wfi\n", (unsigned long)target, (unsigned long)woke,
                 wfis);
    ok &= check("woke at or after the deadline", woke >= target, 1);
    ok &= check("and within 32 ticks of it", woke - target < 32, 1);

    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
