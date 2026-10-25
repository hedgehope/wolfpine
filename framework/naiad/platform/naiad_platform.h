/*
 * naiad_platform.h: the few services a freestanding Naiad program needs.
 *
 *   naiad_putchar / naiad_puts / naiad_printf   console, over HTIF device 1 cmd 1
 *   naiad_exit(code)                            HTIF exit, tohost = (code << 1) | 1
 *   naiad_cycles() / naiad_instret()            Zicsr counters when the design has
 *                                               Zicsr, else the CLINT's mtime
 *   naiad_perf_read(offset)                     the perfcounters block, if present
 *   naiad_counter(EVENT)                        a counter-hub event, through its Zihpm
 *                                               counter if the design binds one, else
 *                                               the perfcounters block
 *   naiad_hartid() / NAIAD_HARTS                this hart's mhartid and the hart count
 *   naiad_barrier()                             all harts meet (a no-op on one hart)
 *   naiad_lock(&l) / naiad_unlock(&l)           a spinlock on a naiad_lock_t
 *
 * Multi-hart designs (NAIAD_MULTICORE, from naiad.h): every hart runs main, and
 * hart 0's return from main is the exit (see the generated crt0.S). The console
 * is shared: naiad_printf / naiad_puts hold a lock for the whole string, so
 * lines from different harts do not interleave mid-string, and each HTIF
 * command is sent under a second lock (Spike polls tohost, so two harts must
 * not both see it idle and overwrite each other's command).
 *
 * Include after nothing else: it pulls in the design's generated naiad.h (put the
 * generated directory on the include path). The same tohost/fromhost protocol is
 * what Spike speaks, so programs using only this header also run there.
 *
 * Built with -DNAIAD_PLATFORM_LINUX, the same names come from
 * naiad_platform_linux.h instead: a hosted Linux user-space back end (stdio,
 * clock_gettime or a perf-event cycle counter), for running Naiad kernels
 * natively on RVV hardware such as the SpacemiT K1.
 */
#ifndef NAIAD_PLATFORM_H
#define NAIAD_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

#ifdef NAIAD_PLATFORM_LINUX
#include "naiad_platform_linux.h"
#else /* the freestanding Naiad / Spike back end */

#include "naiad.h"

#if !defined(__riscv_xlen) || __riscv_xlen != NAIAD_XLEN
#error "compiler XLEN does not match the design's NAIAD_XLEN (check -march / --target)"
#endif

/* Defined in the generated crt0.S, in the .tohost section. */
extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

#define NAIAD_CLINT_MTIME_OFFSET 0xBFF8u

/* Place a kernel input/output in .noinit, which crt0 does not zero, so the host
 * can write it after loading the ELF (framework.naiad.kernel_io.run_kernel) and
 * read it back by symbol. 64-byte aligned, so tiles and DMA rows line up. */
#define NAIAD_IO __attribute__((section(".noinit"), aligned(64)))

#ifndef NAIAD_HARTS
#define NAIAD_HARTS 1
#endif

/* A spinlock word (amoswap.w.aq / a release store); zero is unlocked. */
typedef volatile int naiad_lock_t;

#ifdef NAIAD_MULTICORE
static inline unsigned naiad_hartid(void) {
    unsigned long h;
    __asm__ volatile("csrr %0, mhartid" : "=r"(h));
    return (unsigned)h;
}

static inline void naiad_lock(naiad_lock_t *l) {
    while (__atomic_exchange_n(l, 1, __ATOMIC_ACQUIRE)) {
        while (*l) {
        }
    }
}

static inline void naiad_unlock(naiad_lock_t *l) { __atomic_store_n(l, 0, __ATOMIC_RELEASE); }

/* naiad_libc.c: the HTIF command lock and the console (whole-string) lock. */
extern naiad_lock_t naiad_htif_lock;
extern naiad_lock_t naiad_console_lock;

/* Sense-reversing barrier over all NAIAD_HARTS harts (naiad_libc.c). */
void naiad_barrier(void);
#else
static inline unsigned naiad_hartid(void) { return 0; }
static inline void naiad_lock(naiad_lock_t *l) { (void)l; }
static inline void naiad_unlock(naiad_lock_t *l) { (void)l; }
static inline void naiad_barrier(void) {}
#endif

static inline void naiad_htif_send(uint64_t cmd) {
#ifdef NAIAD_MULTICORE
    naiad_lock(&naiad_htif_lock);
#endif
    while (tohost != 0) {
    }
#if __riscv_xlen == 32
    /* High word first: the store to the low word is the doorbell. */
    volatile uint32_t *word = (volatile uint32_t *)&tohost;
    word[1] = (uint32_t)(cmd >> 32);
    word[0] = (uint32_t)cmd;
#else
    tohost = cmd;
#endif
#ifdef NAIAD_MULTICORE
    naiad_unlock(&naiad_htif_lock);
#endif
}

/* Spike's console device does not ack a write through fromhost, so the only
 * handshake is naiad_htif_send waiting for the previous tohost to be consumed. */
static inline void naiad_putchar(int ch) {
    naiad_htif_send(((uint64_t)1 << 56) | ((uint64_t)1 << 48) | (uint8_t)ch);
}

/* naiad_puts without the console lock (for code already holding it, as naiad_printf does). */
static inline void naiad_puts_unlocked(const char *s) {
    while (*s) {
        naiad_putchar(*s++);
    }
}

static inline void naiad_puts(const char *s) {
#ifdef NAIAD_MULTICORE
    naiad_lock(&naiad_console_lock);
#endif
    naiad_puts_unlocked(s);
#ifdef NAIAD_MULTICORE
    naiad_unlock(&naiad_console_lock);
#endif
}

__attribute__((noreturn)) static inline void naiad_exit(int code) {
    naiad_htif_send(((uint64_t)(uint32_t)code << 1) | 1);
    for (;;) {
    }
}

static inline uint64_t naiad_read64_hilo(volatile uint32_t *lo, volatile uint32_t *hi) {
    uint32_t h, l;
    do {
        h = *hi;
        l = *lo;
    } while (h != *hi);
    return ((uint64_t)h << 32) | l;
}

#ifdef NAIAD_CLINT_ADDR
static inline uint64_t naiad_mtime(void) {
    volatile uint32_t *mtime = (volatile uint32_t *)(uintptr_t)(NAIAD_CLINT_ADDR + NAIAD_CLINT_MTIME_OFFSET);
#if __riscv_xlen == 32
    return naiad_read64_hilo(mtime, mtime + 1);
#else
    return *(volatile uint64_t *)mtime;
#endif
}
#endif

#ifdef NAIAD_PERF_ADDR
/* Read the perfcounters block register at byte `offset` (NAIAD_<BLOCK>_<COUNTER>). */
static inline uint64_t naiad_perf_read(uintptr_t offset) {
    volatile uint32_t *reg = (volatile uint32_t *)(uintptr_t)(NAIAD_PERF_ADDR + offset);
#if __riscv_xlen == 32
    return naiad_read64_hilo(reg, reg + 1);
#else
    return *(volatile uint64_t *)reg;
#endif
}
#endif

/* CSRs by number (cycle 0xC00, instret 0xC02, +0x80 for the RV32 high halves). */
#if defined(__riscv_zicsr)
#if __riscv_xlen == 32
#define NAIAD_READ_CSR64(lo, hi)                                                                   \
    ({                                                                                             \
        uint32_t h_, l_, h2_;                                                                      \
        do {                                                                                       \
            __asm__ volatile("csrr %0, " #hi : "=r"(h_));                                          \
            __asm__ volatile("csrr %0, " #lo : "=r"(l_));                                          \
            __asm__ volatile("csrr %0, " #hi : "=r"(h2_));                                         \
        } while (h_ != h2_);                                                                       \
        ((uint64_t)h_ << 32) | l_;                                                                 \
    })
#else
#define NAIAD_READ_CSR64(lo, hi)                                                                   \
    ({                                                                                             \
        uint64_t v_;                                                                               \
        __asm__ volatile("csrr %0, " #lo : "=r"(v_));                                              \
        v_;                                                                                        \
    })
#endif
#endif

static inline uint64_t naiad_cycles(void) {
#if defined(__riscv_zicsr)
    return NAIAD_READ_CSR64(0xC00, 0xC80);
#elif defined(NAIAD_CLINT_ADDR)
    return naiad_mtime();
#else
    return 0;
#endif
}

static inline uint64_t naiad_instret(void) {
#if defined(__riscv_zicsr)
    return NAIAD_READ_CSR64(0xC02, 0xC82);
#else
    return 0;
#endif
}

/* Zihpm: hpmcounter<n> (0xC00 + n). csrr takes the CSR number as an immediate,
 * so a run-time n goes through a switch of constant reads; with a constant n
 * (the NAIAD_HPM_* defines) it folds to one csrr at -O1 and above. */
#if defined(NAIAD_HAS_HPM) && defined(__riscv_zicsr) && __riscv_xlen == 64
#define NAIAD_HPM_CASE(n)                                                                          \
    case n:                                                                                        \
        __asm__ volatile("csrr %0, %1" : "=r"(v) : "i"(0xC00 + (n)));                              \
        return v
static inline __attribute__((always_inline)) uint64_t naiad_hpm_read(int n) {
    uint64_t v;
    switch (n) {
        NAIAD_HPM_CASE(3); NAIAD_HPM_CASE(4); NAIAD_HPM_CASE(5); NAIAD_HPM_CASE(6);
        NAIAD_HPM_CASE(7); NAIAD_HPM_CASE(8); NAIAD_HPM_CASE(9); NAIAD_HPM_CASE(10);
        NAIAD_HPM_CASE(11); NAIAD_HPM_CASE(12); NAIAD_HPM_CASE(13); NAIAD_HPM_CASE(14);
        NAIAD_HPM_CASE(15); NAIAD_HPM_CASE(16); NAIAD_HPM_CASE(17); NAIAD_HPM_CASE(18);
        NAIAD_HPM_CASE(19); NAIAD_HPM_CASE(20); NAIAD_HPM_CASE(21); NAIAD_HPM_CASE(22);
        NAIAD_HPM_CASE(23); NAIAD_HPM_CASE(24); NAIAD_HPM_CASE(25); NAIAD_HPM_CASE(26);
        NAIAD_HPM_CASE(27); NAIAD_HPM_CASE(28); NAIAD_HPM_CASE(29); NAIAD_HPM_CASE(30);
        NAIAD_HPM_CASE(31);
    default:
        return 0;
    }
}
#undef NAIAD_HPM_CASE
#endif

/* naiad_counter(EVENT): the value of a counter-hub event, where EVENT is the
 * event's name as an identifier (matrix.macs -> MATRIX_MACS, hart0.instret.fp
 * -> HART0_INSTRET_FP). naiad.h defines NAIAD_COUNTER_<EVENT> as "hpm, offset"
 * for every event the design binds to a Zihpm counter or lists in its first
 * perfcounters block; the Zihpm counter wins, since it is one instruction and
 * moves no bus bytes. An event the design does neither for does not compile. */
#define naiad_counter(event) naiad_counter_read(NAIAD_COUNTER_##event)

static inline __attribute__((always_inline)) uint64_t naiad_counter_read(int hpm, long offset) {
#if defined(NAIAD_HAS_HPM) && defined(__riscv_zicsr) && __riscv_xlen == 64
    if (hpm) {
        return naiad_hpm_read(hpm);
    }
#else
    (void)hpm;
#endif
#ifdef NAIAD_PERF_ADDR
    if (offset >= 0) {
        return naiad_perf_read((uintptr_t)offset);
    }
#else
    (void)offset;
#endif
    return 0;
}

/* naiad_libc.c: the freestanding shim (compilers emit calls to the mem* functions). */
void *memcpy(void *dst, const void *src, size_t n);
void *memmove(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);

/* printf-lite: %d %i %u %x %X %p %s %c %%, with l / ll length and a field width
 * with optional 0 flag. No floating point. Returns the number of characters. */
int naiad_printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

#endif /* NAIAD_PLATFORM_LINUX */
#endif /* NAIAD_PLATFORM_H */
