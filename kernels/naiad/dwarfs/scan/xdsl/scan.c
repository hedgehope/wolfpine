/*
 * scan, xDSL variant: the inclusive prefix sum of a length-N vector through a kernel
 * compiled by xDSL (convert-linalg-to-riscv-ame), not from C.
 *
 * The source is the scan's definition as a linalg.generic, the masked triangular sum
 *   y[i] = sum_k (k <= i ? x[k] : 0)
 * over (i, k), i parallel and k a reduction, with the mask from linalg.index. Written
 * out by convert-linalg-to-loops it is O(N^2); the lowering recognises it and emits the
 * ame variant's tile form instead: per 16 x 16 tile X U + (Ls X) J, Ls X through
 * memory (two alternating buffers), and the carry as a stride-0 load of the previous
 * tile's last output row times Sel. A tail of fewer than 256 elements runs through
 * zero-padded staging tiles, so N need not be a multiple of 256.
 *
 * The kernel is prebuilt assembly, one file per (size, formats) point, named by the
 * runner (run_xdsl.py) through -DXDSL_ASM=<basename>; its MLIR is
 * func.func @scan_xdsl(x, y) with a zero linalg.fill of y. The timed region matches
 * the ame variant's: the call plus the drain of its last astore.
 *
 * Host interface as scan_ame: x (in, N input-format elements), y (N
 * accumulator-format elements), region_cycles.
 */
#include <stdint.h>

#include "../../jacobi/dw_fmt.h"
#include "dwarf_util.h"
#include "naiad_platform.h"

#ifndef XDSL_ASM
#error "define XDSL_ASM (the kernel's .S basename; run_xdsl.py does)"
#endif
#define XDSL_STR_(x) #x
#define XDSL_STR(x) XDSL_STR_(x)

__asm__(".include \"" XDSL_STR(XDSL_ASM) ".S\"");

void scan_xdsl(const void *x, void *y);

NAIAD_IO dw_in_t x[SCAN_N];
NAIAD_IO dw_acc_t y[SCAN_N];
NAIAD_IO uint64_t region_cycles;

int main(void) {
    uint64_t t0 = naiad_cycles();
    scan_xdsl(x, y);
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - t0;
    return 0;
}
