/*
 * ex1: console output and exit codes over HTIF. Design: naiad-base (RV32IM).
 *
 * A Naiad program talks to the host through two 64-bit words, tohost and
 * fromhost, that the generated crt0.S places in a .tohost section (the same
 * protocol Spike speaks). naiad_putchar sends HTIF device 1 / command 1 with
 * one character; naiad_exit(code) sends (code << 1) | 1, which halts the SoC
 * with that exit code. Returning from main does the same through crt0.
 *
 * The checks are what crt0 and the loader must get right before anything else
 * works: .bss zeroed, .data initialised, .rodata readable, a working stack,
 * and the M extension. The run's exit code is 0 iff every check passes.
 */
#include "naiad_platform.h"

static uint32_t zeroed[64];                 /* .bss: crt0 zeroes it */
static volatile uint32_t initialised = 42;  /* .data: the loader copies it in */
static const char greeting[] = "naiad";     /* .rodata */

static int check(const char *what, long got, long want) {
    naiad_printf("  %s: %ld (expect %ld) %s\n", what, got, want, got == want ? "ok" : "MISMATCH");
    return got == want;
}

/* Recursion exercises the stack crt0 set up. */
static uint32_t gcd(uint32_t a, uint32_t b) { return b ? gcd(b, a % b) : a; }

/* Not returning from main: naiad_exit ends the run from anywhere. */
__attribute__((noreturn)) static void finish(int ok) {
    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    naiad_exit(ok ? 0 : 1);
}

int main(void) {
    int ok = 1;
    naiad_printf("ex1: hello from %s on design %s, rv%d (%s)\n", greeting, NAIAD_DESIGN, NAIAD_XLEN, NAIAD_MARCH);

    uint32_t any = 0;
    for (unsigned i = 0; i < sizeof zeroed / sizeof zeroed[0]; i++) {
        any |= zeroed[i];
    }
    ok &= check(".bss zeroed", any, 0);
    ok &= check(".data initialised", initialised, 42);
    ok &= check(".rodata length", (long)strlen(greeting), 5);
    ok &= check("gcd(1071, 462)", gcd(1071, 462), 21);
    /* volatile, so the multiply and divide happen at run time, not in the compiler. */
    volatile int a = -1234, b = 56;
    ok &= check("mul", a * b, -69104);
    ok &= check("div", a / b, -22);
    ok &= check("rem", a % b, -2);

    naiad_printf("  cycles so far: %lu\n", (unsigned long)naiad_cycles());
    finish(ok);
}
