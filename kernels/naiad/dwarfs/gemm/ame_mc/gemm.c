/*
 * gemm, multi-hart AME variant: the ame variant's blocked driver (naiad_gemm.h),
 * with C's rows split across the design's harts (each hart has its own matrix
 * unit). Needs a multi-hart design (naiad-dwarfs-4c.yaml); on one hart it is the
 * ame variant plus two no-op barriers.
 *
 * Rows go out in whole NAIAD_GEMM_MB-row blocks, as evenly as they divide, so
 * the only padded row block (when m is not a multiple of MB) belongs to one
 * hart. The emitted driver's edge-staging buffers are single, file-scope
 * arrays, so when n or k has a padded edge (every hart would pack into them)
 * the harts take turns under a lock instead: correct, but serial.
 *
 * The timed region (hart 0's cycle counter) runs from a barrier all harts have
 * reached to a barrier after every hart has drained its last accumulator stores.
 */
#include <stdint.h>

#include "dwarf_util.h"
#include "naiad_platform.h"

#define NAIAD_GEMM_IN GEMM_IN
#define NAIAD_GEMM_ACC GEMM_ACC
#include "naiad_gemm.h"

NAIAD_IO naiad_gemm_in_t A[GEMM_M * GEMM_K];
NAIAD_IO naiad_gemm_in_t B[GEMM_K * GEMM_N];
NAIAD_IO naiad_gemm_acc_t C[GEMM_M * GEMM_N];
NAIAD_IO uint64_t region_cycles;

#define ROW_BLOCKS ((GEMM_M + NAIAD_GEMM_MB - 1) / NAIAD_GEMM_MB)
#define SHARED_STAGING (GEMM_N % NAIAD_GEMM_NB != 0 || GEMM_K % NAIAD_GEMM_KB != 0)

static naiad_lock_t staging_lock;

int main(void) {
    const unsigned h = naiad_hartid();
    /* Hart h's row blocks: [b0, b1), the first (ROW_BLOCKS % NAIAD_HARTS) harts one more. */
    const long per = ROW_BLOCKS / NAIAD_HARTS, extra = ROW_BLOCKS % NAIAD_HARTS;
    const long b0 = h * per + (h < extra ? h : extra);
    const long b1 = b0 + per + (h < extra);
    const long r0 = b0 * NAIAD_GEMM_MB;
    const long r1 = b1 * NAIAD_GEMM_MB < GEMM_M ? b1 * NAIAD_GEMM_MB : GEMM_M;

    naiad_barrier();
    uint64_t t0 = naiad_cycles();
    if (r1 > r0) {
        if (SHARED_STAGING) {
            naiad_lock(&staging_lock);
        }
        naiad_gemm(r1 - r0, GEMM_N, GEMM_K, A + r0 * GEMM_K, GEMM_K, B, GEMM_N, C + r0 * GEMM_N, GEMM_N);
        naiad_mx_drain();
        if (SHARED_STAGING) {
            naiad_unlock(&staging_lock);
        }
    }
    naiad_barrier();
    if (h == 0) {
        region_cycles = naiad_cycles() - t0;
    }
    return 0;
}
