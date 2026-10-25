/*
 * coherence: a cache-coherence microbenchmark for multi-hart designs.
 *
 * Every hart runs three phases of ITERS read-modify-writes, each between two
 * barriers, and hart 0 times each phase (mcycle) and counts the L1
 * invalidations in it (when the design binds l1d.invalidations to a Zihpm
 * counter):
 *   false   each hart increments its own counter, all counters in one line;
 *   padded  the same, with each counter in its own line;
 *   true    every hart amoadds one shared counter.
 * With private MSI caches (cache.private: true) false sharing ping-pongs the
 * line between the harts (an invalidation and a snoop writeback per write),
 * padded does not, and true sharing pays the same per AMO. With a shared cache
 * (or none) false and padded cost the same. The counters are checked, so the
 * program also PASSes or FAILs on its results. Hart 0 prints
 *   <phase> cycles=<n> invalidations=<n>
 * per phase, then PASS/FAIL.
 */
#include <stdint.h>

#include "naiad_platform.h"

#ifndef NAIAD_MULTICORE
#error "coherence needs a design with cores.count > 1"
#endif

#define ITERS 200
#define LINE 64

static volatile uint64_t packed[NAIAD_HARTS] __attribute__((aligned(LINE)));
static struct {
    volatile uint64_t v;
    char pad[LINE - sizeof(uint64_t)];
} padded[NAIAD_HARTS] __attribute__((aligned(LINE)));
static volatile uint64_t shared_counter __attribute__((aligned(LINE)));

static uint64_t invalidations(void) {
#ifdef NAIAD_COUNTER_L1D_INVALIDATIONS
    return naiad_counter(L1D_INVALIDATIONS);
#else
    return 0;
#endif
}

static void run_phase(int which, const char *name) {
    unsigned h = naiad_hartid();
    naiad_barrier();
    uint64_t c0 = naiad_cycles(), i0 = invalidations();
    for (int i = 0; i < ITERS; i++) {
        if (which == 0) {
            packed[h] = packed[h] + 1;
        } else if (which == 1) {
            padded[h].v = padded[h].v + 1;
        } else {
            __atomic_fetch_add(&shared_counter, 1, __ATOMIC_RELAXED);
        }
    }
    naiad_barrier();
    if (h == 0) {
        uint64_t c1 = naiad_cycles(), i1 = invalidations();
        naiad_printf("%s cycles=%lu invalidations=%lu\n", name, (unsigned long)(c1 - c0), (unsigned long)(i1 - i0));
    }
}

int main(void) {
    run_phase(0, "false");
    run_phase(1, "padded");
    run_phase(2, "true");
    naiad_barrier();
    if (naiad_hartid() != 0) {
        return 0;
    }
    int ok = shared_counter == (uint64_t)NAIAD_HARTS * ITERS;
    for (unsigned h = 0; h < NAIAD_HARTS; h++) {
        ok &= packed[h] == ITERS && padded[h].v == ITERS;
    }
    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
