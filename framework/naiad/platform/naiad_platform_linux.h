/*
 * naiad_platform_linux.h: naiad_platform.h's Linux user-space back end.
 *
 * Included by naiad_platform.h when NAIAD_PLATFORM_LINUX is defined; do not
 * include it directly. It gives a kernel written against the Naiad platform
 * header the same names on a hosted RV64 Linux system (the SpacemiT K1, say):
 *
 *   naiad_putchar / naiad_puts / naiad_printf   stdio (naiad_printf is printf)
 *   naiad_exit(code)                            flush stdout, exit(code)
 *   naiad_cycles()                              see below
 *   naiad_instret()                             0 (or perf instructions, below)
 *   naiad_counter(EVENT)                        0: Naiad's counter hub has no
 *                                               hardware equivalent
 *   NAIAD_IO                                    nothing (inputs are not host-written)
 *
 * naiad_cycles(): Linux (6.6+) disables user rdcycle by default, so the
 * default is CLOCK_MONOTONIC_RAW in nanoseconds. Two alternatives:
 *
 *   -DNAIAD_LINUX_PERF=1     perf_event_open(PERF_COUNT_HW_CPU_CYCLES) (and
 *                            instructions for naiad_instret); needs
 *                            kernel.perf_event_paranoid <= 2 (usual default)
 *   -DNAIAD_LINUX_RDCYCLE=1  the rdcycle / rdinstret CSRs directly; needs
 *                            `sysctl kernel.perf_user_access=2`, else SIGILL
 *
 * naiad_clock_unit() names the unit ("ns" or "cycles") for reports.
 */
#ifndef NAIAD_PLATFORM_LINUX_H
#define NAIAD_PLATFORM_LINUX_H

#ifndef NAIAD_PLATFORM_H
#error "include naiad_platform.h (with -DNAIAD_PLATFORM_LINUX), not naiad_platform_linux.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(NAIAD_LINUX_PERF) && NAIAD_LINUX_PERF
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#define NAIAD_IO

static inline void naiad_putchar(int ch) { putchar(ch); }

static inline void naiad_puts(const char *s) { fputs(s, stdout); }

#define naiad_printf printf

__attribute__((noreturn)) static inline void naiad_exit(int code) {
    fflush(stdout);
    exit(code);
}

#if defined(NAIAD_LINUX_PERF) && NAIAD_LINUX_PERF
static inline int naiad_perf_fd(int which) {
    static int fds[2] = {-1, -1};
    int i = which == PERF_COUNT_HW_CPU_CYCLES ? 0 : 1;
    if (fds[i] < 0) {
        struct perf_event_attr attr;
        memset(&attr, 0, sizeof attr);
        attr.type = PERF_TYPE_HARDWARE;
        attr.size = sizeof attr;
        attr.config = (uint64_t)which;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        fds[i] = (int)syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
        if (fds[i] < 0) {
            perror("naiad_platform_linux: perf_event_open");
            exit(2);
        }
    }
    return fds[i];
}

static inline uint64_t naiad_perf_value(int which) {
    uint64_t v = 0;
    if (read(naiad_perf_fd(which), &v, sizeof v) != (ssize_t)sizeof v) {
        perror("naiad_platform_linux: read");
        exit(2);
    }
    return v;
}

static inline uint64_t naiad_cycles(void) { return naiad_perf_value(PERF_COUNT_HW_CPU_CYCLES); }
static inline uint64_t naiad_instret(void) { return naiad_perf_value(PERF_COUNT_HW_INSTRUCTIONS); }
static inline const char *naiad_clock_unit(void) { return "cycles"; }
#elif defined(NAIAD_LINUX_RDCYCLE) && NAIAD_LINUX_RDCYCLE
static inline uint64_t naiad_cycles(void) {
    uint64_t v;
    __asm__ volatile("rdcycle %0" : "=r"(v));
    return v;
}
static inline uint64_t naiad_instret(void) {
    uint64_t v;
    __asm__ volatile("rdinstret %0" : "=r"(v));
    return v;
}
static inline const char *naiad_clock_unit(void) { return "cycles"; }
#else
static inline uint64_t naiad_cycles(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}
static inline uint64_t naiad_instret(void) { return 0; }
static inline const char *naiad_clock_unit(void) { return "ns"; }
#endif

#define naiad_counter(event) ((uint64_t)0)

#endif /* NAIAD_PLATFORM_LINUX_H */
