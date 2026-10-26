/*
 * ex6: several harts. Design: ex6/naiad-quad.yaml (four RV64IMA harts).
 *
 * On a multi-hart design every hart runs main. The generated crt0.S lets hart
 * 0 zero .bss, then releases harts 1..3 by setting their CLINT msip (each
 * waits in wfi with mie.MSIE set); hart 0's return from main is the exit, and
 * the other harts park when they return.
 *
 * Each hart first sleeps until its own timer, (hart + 1) * DELAY ticks ahead:
 * on a multi-hart design wfi really sleeps (the core stops ticking until
 * mtime >= mtimecmp), so each hart's wfi count stays at about 1 where ex2's
 * single hart polled ~150 times. Then each hart sums its quarter of 1..TOTAL
 * into partial[hart] and adds it to a shared total with an AMO;
 * naiad_barrier() orders the phases, and hart 0 checks both results.
 */
#include "naiad_platform.h"

#ifndef NAIAD_MULTICORE
#error "ex6 needs a design with cores.count > 1 (ex6/naiad-quad.yaml)"
#endif

#define TOTAL 4000
#define DELAY 1000
#define MIE_MTIE (1UL << 7)

static volatile uint64_t partial[NAIAD_HARTS];
static volatile uint64_t wfis[NAIAD_HARTS];
static volatile uint64_t total;

static volatile uint64_t *clint64(unsigned long offset) {
    return (volatile uint64_t *)(NAIAD_CLINT_ADDR + offset);
}

int main(void) {
    unsigned h = naiad_hartid();
    if (h == 0) {
        naiad_printf("ex6: %d harts on %s\n", NAIAD_HARTS, NAIAD_DESIGN);
    }
    naiad_barrier(); /* every hart is up */

    /* Phase 1: sleep until this hart's timer. */
    volatile uint64_t *mtime = clint64(NAIAD_CLINT_MTIME);
    uint64_t target = *mtime + (uint64_t)DELAY * (h + 1);
    *clint64(NAIAD_CLINT_MTIMECMP + 8 * h) = target;
    __asm__ volatile("csrs mie, %0" ::"r"(MIE_MTIE));
    uint64_t n = 0;
    while (*mtime < target) {
        __asm__ volatile("wfi");
        n++;
    }
    __asm__ volatile("csrc mie, %0" ::"r"(MIE_MTIE));
    wfis[h] = n;

    /* Phase 2: this hart's slice of 1..TOTAL. */
    uint64_t lo = 1 + (uint64_t)h * TOTAL / NAIAD_HARTS, hi = (uint64_t)(h + 1) * TOTAL / NAIAD_HARTS;
    uint64_t sum = 0;
    for (uint64_t i = lo; i <= hi; i++) {
        sum += i;
    }
    partial[h] = sum;
    __atomic_fetch_add(&total, sum, __ATOMIC_RELAXED);
    naiad_barrier(); /* every partial and the AMO total are in */

    if (h != 0) {
        return 0; /* parks: only hart 0's return ends the run */
    }
    uint64_t combined = 0;
    int ok = 1;
    for (unsigned i = 0; i < NAIAD_HARTS; i++) {
        naiad_printf("  hart %u: partial %lu, %lu wfi\n", i, (unsigned long)partial[i], (unsigned long)wfis[i]);
        combined += partial[i];
        ok &= wfis[i] <= 2; /* slept, rather than polled */
    }
    uint64_t want = (uint64_t)TOTAL * (TOTAL + 1) / 2;
    naiad_printf("  sum of partials %lu, AMO total %lu (expect %lu)\n", (unsigned long)combined, (unsigned long)total, (unsigned long)want);
    ok &= combined == want && total == want;
    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
