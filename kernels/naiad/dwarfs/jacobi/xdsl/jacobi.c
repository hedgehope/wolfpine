/*
 * jacobi, xDSL variant: JACOBI_ITERS sweeps of the 5-point Jacobi stencil, each one a
 * call of a kernel compiled by xDSL (convert-linalg-to-riscv-ame), not from C.
 *
 * The source of one sweep is the stencil as a linalg.generic over four shifted
 * memref.subviews of the grid (up, down, left, right) into the interior subview of
 * the next grid, with the body 0.25 * ((up + down) + (left + right)). The lowering
 * recognises a linear combination of 2-d views and emits the ame variant's window
 * form (JACOBI_FORM 0): per 16 x 16 output tile, each view's tile is an "a" operand
 * against the scaled identity 0.25 I, four loads and four mma into one accumulator;
 * the views of one grid share a running pointer, and a tile's loads issue before the
 * previous tile's astore, with the two accumulators alternating.
 *
 * The kernel is prebuilt assembly, one file per (size, formats) point, named by the
 * runner (run_xdsl.py) through -DXDSL_ASM=<basename>; its MLIR is
 * func.func @jacobi_xdsl(src, dst). As in the ame variant the grid is copied into two
 * sweep buffers before the timed region, the sweeps ping-pong between them, and the
 * last one writes out. Same-format pairs only (fp32/fp32, fp64/fp64): the 16-bit
 * pairs' per-sweep re-rounding is not lowered.
 *
 * Host interface as jacobi_ame: grid (in, N x N), out (N x N, the last sweep),
 * region_cycles.
 */
#include <stdint.h>

#include "../../jacobi/dw_fmt.h"
#include "dwarf_util.h"
#include "naiad_platform.h"

#ifndef XDSL_ASM
#error "define XDSL_ASM (the kernel's .S basename; run_xdsl.py does)"
#endif
#if !DW_SAME_FMT
#error "the xdsl jacobi variant takes fp32/fp32 or fp64/fp64"
#endif
#define XDSL_STR_(x) #x
#define XDSL_STR(x) XDSL_STR_(x)

__asm__(".include \"" XDSL_STR(XDSL_ASM) ".S\"");

#define N JACOBI_N

void jacobi_xdsl(const void *src, void *dst);

NAIAD_IO dw_in_t grid[N * N];
NAIAD_IO dw_acc_t out[N * N];
NAIAD_IO uint64_t region_cycles;

static dw_in_t work[2][N * N] __attribute__((aligned(64)));

int main(void) {
    for (int i = 0; i < N * N; i++) {
        work[0][i] = work[1][i] = grid[i];
        out[i] = dw_in_to_acc(grid[i]);
    }

    uint64_t t0 = naiad_cycles();
    for (int it = 0; it < JACOBI_ITERS; it++) {
        void *dst = it == JACOBI_ITERS - 1 ? (void *)out : (void *)work[(it + 1) & 1];
        jacobi_xdsl(work[it & 1], dst);
    }
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - t0;
    return 0;
}
