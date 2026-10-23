// Op test for `matmul_tiles` on random bf16 operands, as the end-to-end tie for
// Naiad's bit-exact Tensix numerics (framework/matrix/tensix_numerics.py, B6).
//
// One 32x64 @ 64x32 product, K split over two tiles that accumulate into one
// DST (matmul_tiles(0, 0) then matmul_tiles(1, 1)), so the result depends on
// the datapath's truncated products, the group alignment, the per-MVMUL round
// into Dst *and* the order tt-llk issues the MVMULs and fidelity phases in.
// The operands are deterministic xorshift bf16 patterns with mixed exponents.
//
// MATMULRAND_FIDELITY (LoFi | HiFi2 | HiFi3 | HiFi4, default HiFi4) and
// MATMULRAND_FP32_DEST (0 | 1, default 0) pick the compute config. Dumps the
// operand tiles as `OPDIFF_INPUT:<hex>` and the packed output tile as
// `OPDIFF_RESULT:<hex>` (tile layout, bf16 words) so optests/diff.sh can diff it
// against ttsim and optests/matmulrand/check.py against tile_mac.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t TILE_ELEMS = 32 * 32;
constexpr uint32_t NUM_IN_TILES = 2;  // K = 64

static uint32_t rng_state = 0x2545F491u;
static uint32_t next_rand() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// A bf16 pattern with a random sign, exponent in [118, 136] and mantissa.
static uint16_t rand_bf16() {
    uint32_t r = next_rand();
    uint32_t sign = r & 1;
    uint32_t exp = 118 + (r >> 1) % 19;
    uint32_t man = (r >> 8) & 0x7F;
    return static_cast<uint16_t>((sign << 15) | (exp << 7) | man);
}

int main(int argc, char** argv) {
    const char* fid_env = std::getenv("MATMULRAND_FIDELITY");
    const std::string fid = fid_env ? fid_env : "HiFi4";
    MathFidelity fidelity = MathFidelity::HiFi4;
    if (fid == "LoFi") {
        fidelity = MathFidelity::LoFi;
    } else if (fid == "HiFi2") {
        fidelity = MathFidelity::HiFi2;
    } else if (fid == "HiFi3") {
        fidelity = MathFidelity::HiFi3;
    } else if (fid != "HiFi4") {
        fprintf(stderr, "unknown MATMULRAND_FIDELITY=%s\n", fid.c_str());
        return 2;
    }
    const char* fp32_env = std::getenv("MATMULRAND_FP32_DEST");
    const bool fp32_dest = fp32_env && std::strcmp(fp32_env, "1") == 0;

    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t tile_bytes = sizeof(uint16_t) * TILE_ELEMS;

    InterleavedBufferConfig in_config{
        .device = device,
        .size = NUM_IN_TILES * tile_bytes,
        .page_size = NUM_IN_TILES * tile_bytes,
        .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig out_config{
        .device = device, .size = tile_bytes, .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    auto src0_dram = CreateBuffer(in_config);
    auto src1_dram = CreateBuffer(in_config);
    auto dst_dram = CreateBuffer(out_config);

    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(NUM_IN_TILES * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in0_config);
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(NUM_IN_TILES * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_1, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in1_config);
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_16, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Raw tile-layout words: tile kt of in0 is A[:, 32kt:32kt+32], of in1 B[32kt:32kt+32, :].
    std::vector<uint16_t> src0_vec(NUM_IN_TILES * TILE_ELEMS);
    std::vector<uint16_t> src1_vec(NUM_IN_TILES * TILE_ELEMS);
    for (auto& v : src0_vec) {
        v = rand_bf16();
    }
    for (auto& v : src1_vec) {
        v = rand_bf16();
    }
    tt::tt_metal::detail::WriteToBuffer(src0_dram, src0_vec);
    tt::tt_metal::detail::WriteToBuffer(src1_dram, src1_vec);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {src0_dram->address(), src1_dram->address(), NUM_IN_TILES});
    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), 1});
    CreateKernel(
        program,
        "kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{.math_fidelity = fidelity, .fp32_dest_acc_en = fp32_dest});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint16_t> out_vec(TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_vec);
    CloseDevice(device);

    printf("MATMULRAND_CONFIG: fidelity=%s fp32_dest=%d\n", fid.c_str(), fp32_dest ? 1 : 0);
    printf("OPDIFF_INPUT:");
    for (auto v : src0_vec) {
        printf("%04x", v);
    }
    for (auto v : src1_vec) {
        printf("%04x", v);
    }
    printf("\n");
    printf("OPDIFF_RESULT:");
    for (auto v : out_vec) {
        printf("%04x", v);
    }
    printf("\n");
    printf("Completed successfully on the device, with 1 output tile\n");
    return 0;
}
