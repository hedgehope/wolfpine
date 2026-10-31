/*
 * ex7: traps and interrupts. Design: naiad-base-rv64 (one RV64IM hart + CLINT).
 *
 * One machine-mode handler, installed in mtvec (direct mode), serves everything:
 *
 *   - synchronous exceptions: an illegal instruction (mcause 2, mtval = the
 *     instruction word), ecall (11), ebreak (3) and a misaligned load (4, mtval =
 *     the address). The handler records mcause/mtval/mepc and steps mepc over the
 *     faulting 4-byte instruction before it returns with mret.
 *   - the machine timer interrupt (mcause = 1<<63 | 7), raised while
 *     mtime >= mtimecmp and silenced by moving mtimecmp. The main loop sleeps in
 *     `wfi` and counts the ticks the handler delivers.
 *   - the machine software interrupt (3), raised by a store to msip and silenced
 *     by clearing it.
 *
 * What Naiad models (framework/pe/rv/trap.py): delivery arms when mtvec is
 * non-zero. MIE/MPIE/MPP are handled as the privileged spec says (an M-mode-only
 * hart: MPP reads 3); an interrupt is taken at an instruction boundary when
 * mstatus.MIE and the mie bit are set and the CLINT says it is pending; wfi wakes
 * on a pending enabled interrupt and the trap is then taken with mepc = wfi + 4.
 * A trap costs the one cycle it takes to redirect, and nothing in the cost model.
 * The unaligned load traps only because mtvec is set; a program with no handler
 * still runs one as before, and an illegal instruction with no handler ends the
 * run with an error.
 */
#include "naiad_platform.h"

#define MIE_MSIE (1UL << 3)
#define MIE_MTIE (1UL << 7)
#define MSTATUS_MIE (1UL << 3)
#define CAUSE_INTERRUPT (1UL << 63)
#define PERIOD 500
#define TICKS 3

static volatile unsigned long last_cause, last_tval, last_epc;
static volatile unsigned long exceptions, timer_irqs, soft_irqs;

static volatile uint64_t *clint64(unsigned long offset) {
    return (volatile uint64_t *)(NAIAD_CLINT_ADDR + offset);
}

__attribute__((interrupt("machine"), aligned(4))) void trap_handler(void) {
    unsigned long cause, tval, epc;
    __asm__ volatile("csrr %0, mcause" : "=r"(cause));
    __asm__ volatile("csrr %0, mtval" : "=r"(tval));
    __asm__ volatile("csrr %0, mepc" : "=r"(epc));
    last_cause = cause;
    last_tval = tval;
    last_epc = epc;
    if (cause & CAUSE_INTERRUPT) {
        if ((cause & 0xFF) == 7) { /* timer: push the deadline out */
            timer_irqs++;
            *clint64(NAIAD_CLINT_MTIMECMP) = *clint64(NAIAD_CLINT_MTIME) + PERIOD;
        } else { /* machine software interrupt: clear msip */
            soft_irqs++;
            *(volatile uint32_t *)(NAIAD_CLINT_ADDR + NAIAD_CLINT_MSIP) = 0;
        }
    } else { /* synchronous: skip the 4-byte instruction that trapped */
        exceptions++;
        __asm__ volatile("csrw mepc, %0" ::"r"(epc + 4));
    }
}

static int check(const char *what, unsigned long got, unsigned long want) {
    naiad_printf("  %-34s 0x%lx (expect 0x%lx) %s\n", what, got, want, got == want ? "ok" : "MISMATCH");
    return got == want;
}

int main(void) {
    int ok = 1;
    naiad_printf("ex7: traps and interrupts on %s\n", NAIAD_DESIGN);

    __asm__ volatile("csrw mtvec, %0" ::"r"((unsigned long)trap_handler));
    unsigned long mtvec;
    __asm__ volatile("csrr %0, mtvec" : "=r"(mtvec));
    ok &= check("mtvec reads back (direct mode)", mtvec, (unsigned long)trap_handler);

    /* Synchronous exceptions. */
    __asm__ volatile(".word 0xFFFFFFFF"); /* not an instruction */
    ok &= check("illegal instruction: mcause", last_cause, 2);
    ok &= check("illegal instruction: mtval", last_tval, 0xFFFFFFFFul);
    ok &= check("handler ran once", exceptions, 1);

    __asm__ volatile("ecall");
    ok &= check("ecall from M-mode: mcause", last_cause, 11);
    __asm__ volatile("ebreak");
    ok &= check("ebreak: mcause", last_cause, 3);
    ok &= check("ebreak: mtval is its pc", last_tval, last_epc);

    static char buf[16] __attribute__((aligned(8)));
    char *odd = buf + 1;
    unsigned long sink;
    __asm__ volatile("ld %0, 0(%1)" : "=r"(sink) : "r"(odd)); /* misaligned: traps, loads nothing */
    ok &= check("misaligned load: mcause", last_cause, 4);
    ok &= check("misaligned load: mtval", last_tval, (unsigned long)odd);
    ok &= check("four exceptions in all", exceptions, 4);

    /* Software interrupt: pending the moment msip is set, taken once MIE is on. */
    __asm__ volatile("csrs mie, %0" ::"r"(MIE_MSIE));
    *(volatile uint32_t *)(NAIAD_CLINT_ADDR + NAIAD_CLINT_MSIP) = 1;
    ok &= check("msip set, MIE clear: not taken", soft_irqs, 0);
    __asm__ volatile("csrs mstatus, %0" ::"r"(MSTATUS_MIE));
    __asm__ volatile("nop");
    __asm__ volatile("csrc mstatus, %0" ::"r"(MSTATUS_MIE));
    ok &= check("MIE set: software interrupt taken", soft_irqs, 1);
    ok &= check("mcause = interrupt | 3", last_cause, CAUSE_INTERRUPT | 3);
    __asm__ volatile("csrc mie, %0" ::"r"(MIE_MSIE));

    /* Timer interrupts: wfi sleeps, the handler runs when the deadline passes. */
    uint64_t start = *clint64(NAIAD_CLINT_MTIME);
    *clint64(NAIAD_CLINT_MTIMECMP) = start + PERIOD;
    __asm__ volatile("csrs mie, %0" ::"r"(MIE_MTIE));
    __asm__ volatile("csrs mstatus, %0" ::"r"(MSTATUS_MIE));
    unsigned long wfis = 0;
    while (timer_irqs < TICKS) {
        __asm__ volatile("wfi");
        wfis++;
    }
    __asm__ volatile("csrc mstatus, %0" ::"r"(MSTATUS_MIE));
    uint64_t elapsed = *clint64(NAIAD_CLINT_MTIME) - start;
    naiad_printf("  %lu timer interrupts, %lu wfi\n", timer_irqs, wfis);
    ok &= check("timer interrupts delivered", timer_irqs, TICKS);
    ok &= check("mcause = interrupt | 7", last_cause, CAUSE_INTERRUPT | 7);
    ok &= check("at least TICKS * PERIOD elapsed", elapsed >= (uint64_t)TICKS * PERIOD, 1);
    ok &= check("slept: about one wfi per interrupt", wfis <= 2 * TICKS, 1);

    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
