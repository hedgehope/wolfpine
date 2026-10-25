// Compute-only tiled GEMM over L1-resident operands, for the Naiad Tensix
// accelerator block: C (Mt x Nt tiles) = A (Mt x Kt) @ B (Kt x Nt).
//
// There is no reader or writer kernel. A, B and C are circular buffers bound
// to fixed L1 addresses (set_globally_allocated_address); the Naiad host DMAs
// the operand tiles in before the launch and the result tiles out after DONE,
// as vecadd_sharding does. So the kernel never calls cb_wait_front (nothing
// produces): operand tiles are addressed by index from the CBs' start, and C
// is packed in row-major tile order (cb_push_back advances the write pointer).
//
// Runtime args: Mt, Kt, Nt. A is Mt x Kt tiles row-major, B is Kt x Nt.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"

constexpr auto cb_a = tt::CBIndex::c_0;
constexpr auto cb_b = tt::CBIndex::c_1;
constexpr auto cb_c = tt::CBIndex::c_2;

void kernel_main() {
    const uint32_t mt = get_arg_val<uint32_t>(0);
    const uint32_t kt = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);

    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_c);
    matmul_init(cb_a, cb_b);

    for (uint32_t m = 0; m < mt; m++) {
        for (uint32_t n = 0; n < nt; n++) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < kt; k++) {
                matmul_tiles(cb_a, cb_b, m * kt + k, k * nt + n, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_c);
            tile_regs_release();
            cb_push_back(cb_c, 1);
        }
    }
}
