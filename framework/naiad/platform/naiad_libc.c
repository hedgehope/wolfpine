/*
 * naiad_libc.c: the smallest libc a freestanding Naiad kernel needs.
 *
 * There is no newlib, so this supplies what the compiler may call implicitly
 * (mem*, and on RV32 the 64-bit division helpers compiler-rt would provide) plus
 * a printf-lite on top of naiad_putchar.
 */
#include <stdarg.h>

#include "naiad_platform.h"

void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = dst;
    const unsigned char *s = src;
    while (n--) {
        *d++ = *s++;
    }
    return dst;
}

void *memmove(void *dst, const void *src, size_t n) {
    unsigned char *d = dst;
    const unsigned char *s = src;
    if (d < s) {
        while (n--) {
            *d++ = *s++;
        }
    } else {
        while (n--) {
            d[n] = s[n];
        }
    }
    return dst;
}

void *memset(void *dst, int c, size_t n) {
    unsigned char *d = dst;
    while (n--) {
        *d++ = (unsigned char)c;
    }
    return dst;
}

int memcmp(const void *a, const void *b, size_t n) {
    const unsigned char *x = a, *y = b;
    for (; n; n--, x++, y++) {
        if (*x != *y) {
            return *x - *y;
        }
    }
    return 0;
}

size_t strlen(const char *s) {
    size_t n = 0;
    while (s[n]) {
        n++;
    }
    return n;
}

#if __riscv_xlen == 32
/* 64-bit division for RV32 (normally from compiler-rt, which this toolchain lacks). */
static uint64_t udivmod64(uint64_t n, uint64_t d, uint64_t *rem) {
    uint64_t q = 0, r = 0;
    for (int i = 63; i >= 0; i--) {
        r = (r << 1) | ((n >> i) & 1);
        if (r >= d) {
            r -= d;
            q |= (uint64_t)1 << i;
        }
    }
    if (rem) {
        *rem = r;
    }
    return q;
}

uint64_t __udivdi3(uint64_t n, uint64_t d) { return udivmod64(n, d, 0); }

uint64_t __umoddi3(uint64_t n, uint64_t d) {
    uint64_t r;
    udivmod64(n, d, &r);
    return r;
}

int64_t __divdi3(int64_t n, int64_t d) {
    int neg = (n < 0) != (d < 0);
    uint64_t q = udivmod64(n < 0 ? -(uint64_t)n : (uint64_t)n, d < 0 ? -(uint64_t)d : (uint64_t)d, 0);
    return neg ? -(int64_t)q : (int64_t)q;
}

int64_t __moddi3(int64_t n, int64_t d) {
    uint64_t r;
    udivmod64(n < 0 ? -(uint64_t)n : (uint64_t)n, d < 0 ? -(uint64_t)d : (uint64_t)d, &r);
    return n < 0 ? -(int64_t)r : (int64_t)r;
}
#endif

static int put_unsigned(unsigned long long v, unsigned base, int upper, int width, char pad, int left) {
    char buf[24];
    const char *digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    int n = 0, out = 0;
    do {
        buf[n++] = digits[v % base];
        v /= base;
    } while (v);
    int total = n;
    if (!left) {
        for (; width > n; width--, out++) {
            naiad_putchar(pad);
        }
    }
    while (n) {
        naiad_putchar(buf[--n]);
        out++;
    }
    for (; left && width > total; width--, out++) { /* '-': pad on the right, with spaces */
        naiad_putchar(' ');
    }
    return out;
}

#ifdef NAIAD_MULTICORE
naiad_lock_t naiad_htif_lock;
naiad_lock_t naiad_console_lock;

/* Sense-reversing barrier: the last hart to arrive resets the count and flips
 * the shared sense; the others spin until it matches their own. */
static volatile unsigned naiad_barrier_count;
static volatile unsigned naiad_barrier_sense;
static unsigned naiad_barrier_local[NAIAD_HARTS];

void naiad_barrier(void) {
    unsigned h = naiad_hartid();
    unsigned sense = naiad_barrier_local[h] ^ 1u;
    naiad_barrier_local[h] = sense;
    if (__atomic_add_fetch(&naiad_barrier_count, 1u, __ATOMIC_ACQ_REL) == NAIAD_HARTS) {
        __atomic_store_n(&naiad_barrier_count, 0u, __ATOMIC_RELAXED);
        __atomic_store_n(&naiad_barrier_sense, sense, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&naiad_barrier_sense, __ATOMIC_ACQUIRE) != sense) {
        }
    }
}
#endif

int naiad_printf(const char *fmt, ...) {
    va_list ap;
    int out = 0;
#ifdef NAIAD_MULTICORE
    naiad_lock(&naiad_console_lock);
#endif
    va_start(ap, fmt);
    for (; *fmt; fmt++) {
        if (*fmt != '%') {
            naiad_putchar(*fmt);
            out++;
            continue;
        }
        fmt++;
        char pad = ' ';
        int width = 0, longs = 0, left = 0;
        for (;; fmt++) { /* flags: '-' (left-justify) and '0' (zero-pad; ignored with '-') */
            if (*fmt == '-') {
                left = 1;
            } else if (*fmt == '0') {
                pad = '0';
            } else {
                break;
            }
        }
        if (left) {
            pad = ' ';
        }
        while (*fmt >= '0' && *fmt <= '9') {
            width = width * 10 + (*fmt++ - '0');
        }
        while (*fmt == 'l') {
            longs++;
            fmt++;
        }
        switch (*fmt) {
        case 'd':
        case 'i': {
            long long v = longs >= 2 ? va_arg(ap, long long) : longs ? va_arg(ap, long) : va_arg(ap, int);
            if (v < 0) {
                naiad_putchar('-');
                out++;
                if (width) {
                    width--;
                }
            }
            out += put_unsigned(v < 0 ? -(unsigned long long)v : (unsigned long long)v, 10, 0, width, pad, left);
            break;
        }
        case 'u':
        case 'x':
        case 'X': {
            unsigned long long v = longs >= 2 ? va_arg(ap, unsigned long long)
                                   : longs    ? va_arg(ap, unsigned long)
                                              : va_arg(ap, unsigned int);
            out += put_unsigned(v, *fmt == 'u' ? 10 : 16, *fmt == 'X', width, pad, left);
            break;
        }
        case 'p':
            naiad_puts_unlocked("0x");
            out += 2 + put_unsigned((uintptr_t)va_arg(ap, void *), 16, 0, width, pad, left);
            break;
        case 's': {
            const char *s = va_arg(ap, const char *);
            int len = (int)strlen(s);
            for (; !left && width > len; width--, out++) {
                naiad_putchar(' ');
            }
            naiad_puts_unlocked(s);
            out += len;
            for (; left && width > len; width--, out++) {
                naiad_putchar(' ');
            }
            break;
        }
        case 'c':
            for (; !left && width > 1; width--, out++) {
                naiad_putchar(' ');
            }
            naiad_putchar(va_arg(ap, int));
            out++;
            for (; left && width > 1; width--, out++) {
                naiad_putchar(' ');
            }
            break;
        case '%':
            naiad_putchar('%');
            out++;
            break;
        default: /* unknown conversion, or a '%' ending the string: echo it */
            naiad_putchar('%');
            out++;
            if (!*fmt) {
                fmt--;
                break;
            }
            naiad_putchar(*fmt);
            out++;
            break;
        }
    }
    va_end(ap);
#ifdef NAIAD_MULTICORE
    naiad_unlock(&naiad_console_lock);
#endif
    return out;
}
