/*
 * multicore: the multi-hart platform contract, checked from C on every hart.
 *
 * Built against a design with cores.count > 1 (every hart runs main; hart 0's
 * return is the exit). Checks, with ITERS rounds per hart:
 *   - each hart sees its own mhartid and its own stack (NAIAD_STACK_SIZE apart);
 *   - an amoadd counter, an lr/sc (compare-and-swap loop) counter and a
 *     lock-protected plain counter all reach NAIAD_HARTS * ITERS: the lr/sc one
 *     only if another hart's store breaks a reservation;
 *   - naiad_barrier() separates phases: after it every hart sees every other
 *     hart's phase-1 writes, over several rounds (the barrier is reusable).
 * Prints one line per hart (in hart order), then PASS/FAIL from hart 0.
 */
#include <stdint.h>

#include "naiad_platform.h"

#ifndef NAIAD_MULTICORE
#error "multicore needs a design with cores.count > 1"
#endif

#define ITERS 50
#define ROUNDS 4

static volatile uint64_t amo_counter;
static volatile uint64_t lrsc_counter;
static volatile uint64_t locked_counter;
static naiad_lock_t counter_lock;
static volatile uintptr_t stack_of[NAIAD_HARTS];
static volatile unsigned seen[NAIAD_HARTS];
static volatile unsigned round_value[NAIAD_HARTS];
static volatile unsigned barrier_errors;

static void lrsc_increment(volatile uint64_t *p) {
    uint64_t old, tmp;
    /* Explicit lr.d / sc.d so the test does not depend on how the compiler lowers a CAS. */
    __asm__ volatile("1:\n"
                     "  lr.d   %0, (%2)\n"
                     "  addi   %1, %0, 1\n"
                     "  sc.d   %1, %1, (%2)\n"
                     "  bnez   %1, 1b\n"
                     : "=&r"(old), "=&r"(tmp)
                     : "r"(p)
                     : "memory");
}

int main(void) {
    unsigned h = naiad_hartid();
    int marker = 0;
    stack_of[h] = (uintptr_t)&marker;
    seen[h] = h + 1;

    for (int i = 0; i < ITERS; i++) {
        __atomic_fetch_add(&amo_counter, 1, __ATOMIC_RELAXED);
        lrsc_increment(&lrsc_counter);
        naiad_lock(&counter_lock);
        locked_counter = locked_counter + 1;
        naiad_unlock(&counter_lock);
    }
    naiad_barrier();

    for (unsigned r = 1; r <= ROUNDS; r++) {
        round_value[h] = r * 100 + h;
        naiad_barrier();
        for (unsigned o = 0; o < NAIAD_HARTS; o++) {
            if (round_value[o] != r * 100 + o) {
                __atomic_fetch_add(&barrier_errors, 1, __ATOMIC_RELAXED);
            }
        }
        naiad_barrier();
    }

    /* One line per hart, in hart order (so the console is deterministic on Spike too). */
    for (unsigned o = 0; o < NAIAD_HARTS; o++) {
        if (o == h) {
            naiad_printf("hart %u of %u: up\n", h, (unsigned)NAIAD_HARTS);
        }
        naiad_barrier();
    }
    if (h != 0) {
        return 0; /* parks: only hart 0's return is the exit */
    }

    int bad = 0;
    uint64_t want = (uint64_t)NAIAD_HARTS * ITERS;
    if (amo_counter != want || lrsc_counter != want || locked_counter != want) {
        naiad_printf("counters: amo %lu lr/sc %lu locked %lu, want %lu\n", (unsigned long)amo_counter,
                     (unsigned long)lrsc_counter, (unsigned long)locked_counter, (unsigned long)want);
        bad = 1;
    }
    for (unsigned o = 0; o < NAIAD_HARTS; o++) {
        if (seen[o] != o + 1) {
            naiad_printf("hart %u never ran main\n", o);
            bad = 1;
        }
        if (o && stack_of[o - 1] - stack_of[o] != NAIAD_STACK_SIZE) {
            naiad_printf("harts %u/%u stacks not %lu apart\n", o - 1, o, (unsigned long)NAIAD_STACK_SIZE);
            bad = 1;
        }
    }
    if (barrier_errors) {
        naiad_printf("barrier: %u stale reads\n", barrier_errors);
        bad = 1;
    }
    naiad_puts(bad ? "FAIL\n" : "PASS\n");
    return bad;
}
