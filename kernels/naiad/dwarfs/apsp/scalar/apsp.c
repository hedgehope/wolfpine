/*
 * apsp_scalar: the scalar F/D baseline for apsp_ame: plain Floyd-Warshall in
 * place, for k, i, j: D[i][j] = fmin(D[i][j], D[i][k] + D[k][j]). Row k does not
 * change during step k (D[k][k] = 0), so the in-place update is exact.
 */
#include "naiad_platform.h"

#include "dwarf_util.h"

#ifndef APSP_N
#error "define APSP_N"
#endif
#if !defined(APSP_FMT_FP32) && !defined(APSP_FMT_FP64)
#error "apsp_scalar supports fp32 and fp64"
#endif
#define N APSP_N
typedef NAIAD_T(APSP_FMT) ap_t;
#ifdef APSP_FMT_FP32
#define AP_MIN __builtin_fminf
#else
#define AP_MIN __builtin_fmin
#endif

NAIAD_IO ap_t D[N * N];
NAIAD_IO uint64_t region_cycles;

int main(void) {
    uint64_t t0 = naiad_cycles();
    for (int k = 0; k < N; k++) {
        const ap_t *dk = &D[(long)k * N];
        for (int i = 0; i < N; i++) {
            ap_t *di = &D[(long)i * N];
            const ap_t dik = di[k];
            /* four elements at a time, all loads first (di may alias dk, which the
             * compiler cannot reorder around; N is a multiple of 16) */
            for (int j = 0; j < N; j += 4) {
                const ap_t a0 = di[j], a1 = di[j + 1], a2 = di[j + 2], a3 = di[j + 3];
                const ap_t b0 = dk[j], b1 = dk[j + 1], b2 = dk[j + 2], b3 = dk[j + 3];
                di[j] = AP_MIN(a0, dik + b0);
                di[j + 1] = AP_MIN(a1, dik + b1);
                di[j + 2] = AP_MIN(a2, dik + b2);
                di[j + 3] = AP_MIN(a3, dik + b3);
            }
        }
    }
    region_cycles = naiad_cycles() - t0;
    return 0;
}
