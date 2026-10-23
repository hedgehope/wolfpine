/*
 * hello: the Naiad bring-up kernel.
 *
 * Exercises what the loader and crt0 must get right (.data initialised, .bss
 * zeroed, .rodata readable, a working stack) and the M extension, prints the
 * results over HTIF, and returns 0 iff every one matches its known answer.
 */
#include "naiad_platform.h"

#define SIEVE_N 1000

static unsigned char composite[SIEVE_N]; /* .bss: must start zeroed */
static volatile int data_word = 0x1234;  /* .data: must arrive initialised */
static const uint32_t fib_table[] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89};

static unsigned count_primes(unsigned n) {
    unsigned count = 0;
    for (unsigned i = 2; i < n; i++) {
        if (composite[i]) {
            continue;
        }
        count++;
        for (unsigned j = i * i; j < n; j += i) {
            composite[j] = 1;
        }
    }
    return count;
}

static uint32_t fib(unsigned n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

static uint64_t sum_of_squares(uint32_t n) {
    uint64_t sum = 0;
    for (uint32_t i = 1; i <= n; i++) {
        sum += (uint64_t)i * i;
    }
    return sum;
}

static int check(const char *what, long long got, long long want) {
    naiad_printf("  %s: %lld (expect %lld) %s\n", what, got, want, got == want ? "ok" : "MISMATCH");
    return got == want;
}

int main(void) {
    uint64_t start = naiad_cycles();
    int ok = 1;

    naiad_printf("hello from naiad: design %s, rv%d (%s)\n", NAIAD_DESIGN, NAIAD_XLEN, NAIAD_MARCH);

    unsigned bss_clean = 1;
    for (unsigned i = 0; i < SIEVE_N; i++) {
        bss_clean &= composite[i] == 0;
    }
    ok &= check("bss zeroed", bss_clean, 1);
    ok &= check("data word", data_word, 0x1234);
    ok &= check("primes below 1000", count_primes(SIEVE_N), 168);
    ok &= check("fib(11), recursive", fib(11), fib_table[11]);
    ok &= check("sum of squares 1..1000", (long long)sum_of_squares(1000), 333833500LL);
    /* volatile so the divisions happen at run time, not in the compiler. */
    volatile int num = -1000, pos = 1000, den = 7;
    volatile uint64_t big = 0x123456789ABCull;
    ok &= check("signed div", num / den, -142);
    ok &= check("rem", pos % den, 6);
    ok &= check("signed rem", num % den, -6);
    ok &= check("u64 div", (long long)(big / 1000u), 20015998343LL);

    uint64_t elapsed = naiad_cycles() - start;
    naiad_printf("cycles in main: %llu\n", (unsigned long long)elapsed);
    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
