/*
 * emittest: drives the emitted naiad_gemm.h at run-time shapes, for
 * framework/naiad/emit/gemm_test.py (not a dwarf variant).
 *
 * cfg = {m, n, k, lda, ldb, ldc, accumulate, op}; op 0 = fp32/fp32 driver,
 * 1 = int8/int32 driver, 2 = the fp32/fp32 micro-kernel on one MB x NB block
 * (k a multiple of the tile K). The host fills A/B/C (strided by lda/ldb/ldc).
 */
#include <stdint.h>

#include "dwarf_util.h"
#include "naiad_gemm.h"
#include "naiad_platform.h"

#define MAXE (96 * 96)

NAIAD_IO int64_t cfg[8];
NAIAD_IO float Af[MAXE];
NAIAD_IO float Bf[MAXE];
NAIAD_IO float Cf[MAXE];
NAIAD_IO int8_t Ai[MAXE];
NAIAD_IO int8_t Bi[MAXE];
NAIAD_IO int32_t Ci[MAXE];
NAIAD_IO uint64_t region_cycles;

int main(void) {
    const long m = cfg[0], n = cfg[1], k = cfg[2], lda = cfg[3], ldb = cfg[4], ldc = cfg[5];
    const int acc = (int)cfg[6];
    uint64_t t0 = naiad_cycles();
    switch (cfg[7]) {
    case 0:
        if (acc) {
            naiad_gemm_fp32_fp32_acc(m, n, k, Af, lda, Bf, ldb, Cf, ldc);
        } else {
            naiad_gemm_fp32_fp32(m, n, k, Af, lda, Bf, ldb, Cf, ldc);
        }
        break;
    case 1:
        if (acc) {
            naiad_gemm_int8_int32_acc(m, n, k, Ai, lda, Bi, ldb, Ci, ldc);
        } else {
            naiad_gemm_int8_int32(m, n, k, Ai, lda, Bi, ldb, Ci, ldc);
        }
        break;
    case 2:
        naiad_gemm_fp32_fp32_ukernel(k, Af, lda, Bf, ldb, Cf, ldc, acc);
        break;
    default:
        return 2;
    }
    naiad_mx_drain();
    region_cycles = naiad_cycles() - t0;
    return 0;
}
