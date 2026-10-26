/*
 * reduce, xDSL variant: the sum of a length-N vector through a kernel compiled by xDSL
 * (convert-linalg-to-riscv-ame) from a linalg.reduce (fp32, fp64) or a linalg.generic
 * sum with a widening extf (16-bit inputs), not from C.
 *
 * The lowering is the ame variant's ones-tile form: acc = sum_c X_c J over the
 * 16 x 16 tiles of x, four loads ahead of their mmas, then the partials finished by
 * a second mma J' P. A tail of fewer than 256 elements goes through a zero-padded tile,
 * so N need not be a multiple of 256 (the spec's own sizes are multiples of 1024).
 *
 * The kernel is prebuilt assembly, one file per (size, formats) point, named by the
 * runner (run_xdsl.py) through -DXDSL_ASM=<basename>; its MLIR is
 * func.func @reduce_xdsl(x, result) with a zero linalg.fill of the result. The timed
 * region matches the ame variant's: the call plus the drain of its last astore.
 *
 * Host interface as reduce_ame: x (in), result (one accumulator-format element),
 * region_cycles.
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

void reduce_xdsl(const void *x, void *result);

NAIAD_IO dw_in_t x[REDUCE_N];
NAIAD_IO dw_acc_t result[1];
NAIAD_IO uint64_t region_cycles;

int main(void) {
    uint64_t t0 = naiad_cycles();
    reduce_xdsl(x, result);
    naiad_mx_drain(); /* the region ends once the last astore has left the unit */
    region_cycles = naiad_cycles() - t0;
    return 0;
}
