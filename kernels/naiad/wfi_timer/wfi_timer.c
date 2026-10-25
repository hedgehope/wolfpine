/*
 * wfi_timer: wfi wakes on the CLINT timer (mtime >= mtimecmp[hart], mie.MTIE).
 *
 * Every hart programs its own mtimecmp DELAY * (hart + 1) ticks ahead, sets
 * mie.MTIE (mstatus.MIE stays clear, so no trap would be taken even on
 * hardware) and waits with `while (mtime < target) wfi;`. On Naiad the hart
 * sleeps once and wakes when its timer is due, so it retires a handful of
 * instructions while waiting; a nop wfi would spin the loop thousands of
 * times. Each hart records the mtime it woke at and its wfi count; hart 0
 * prints one line per hart, then PASS/FAIL (woke at or after its target).
 */
#include <stdint.h>

#include "naiad_platform.h"

#ifndef NAIAD_MULTICORE
#error "wfi_timer needs a design with cores.count > 1"
#endif

#define DELAY 5000
#define MIE_MTIE (1UL << 7)

static volatile uint64_t target_of[NAIAD_HARTS];
static volatile uint64_t woke_at[NAIAD_HARTS];
static volatile uint64_t wfis[NAIAD_HARTS];

static volatile uint64_t *clint64(unsigned long offset) {
    return (volatile uint64_t *)(NAIAD_CLINT_ADDR + offset);
}

int main(void) {
    unsigned h = naiad_hartid();
    naiad_barrier();
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
    woke_at[h] = *mtime;
    target_of[h] = target;
    wfis[h] = n;
    naiad_barrier();
    if (h != 0) {
        return 0;
    }
    int ok = 1;
    for (unsigned i = 0; i < NAIAD_HARTS; i++) {
        naiad_printf("hart %u: target %lu woke %lu after %lu wfi\n", i, (unsigned long)target_of[i],
                     (unsigned long)woke_at[i], (unsigned long)wfis[i]);
        ok &= woke_at[i] >= target_of[i];
    }
    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
