// One output tile accumulating two K tiles into the same DST, the shape a
// blocked GEMM's K loop has: C = A[:, 0:32] @ B[0:32, :] + A[:, 32:64] @ B[32:64, :].

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_in0 = tt::CBIndex::c_0;
constexpr auto cb_in1 = tt::CBIndex::c_1;
constexpr auto cb_out = tt::CBIndex::c_16;

constexpr uint32_t NUM_IN_TILES = 2;

void kernel_main() {
    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);

    cb_wait_front(cb_in0, NUM_IN_TILES);
    cb_wait_front(cb_in1, NUM_IN_TILES);

    acquire_dst();
    for (uint32_t kt = 0; kt < NUM_IN_TILES; kt++) {
        matmul_tiles(cb_in0, cb_in1, kt, kt, 0);
    }
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    release_dst();

    cb_pop_front(cb_in0, NUM_IN_TILES);
    cb_pop_front(cb_in1, NUM_IN_TILES);
}
