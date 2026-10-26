// Capture program for the Naiad Tensix accelerator's GEMM image.
//
// A tt-metal host program, run once against the Wolfpine Blackhole simulator
// with TT_SIM_RECORD set (see ../capture.sh). The wire trace it leaves holds
// everything a Tensix worker needs to run kernels/compute/gemm.cpp: the five
// baby cores' firmware, the init handshake, the compiled TRISC kernels, the
// launch message (CB config, runtime args). framework.ip.tensix.image turns
// that trace into an image the Naiad host program loads into the tile's L1,
// with no tt-metal host involved at run time.
//
// The three operand buffers are single-core height-sharded L1 buffers, so they
// sit at fixed L1 addresses on logical core (0, 0), each CAP_TILES tiles deep:
// the image's CB config is fixed, and the Naiad driver only rewrites the
// runtime args (Mt, Kt, Nt) per launch.
//
// TENSIXGEMM_OUT=fp32 (default): bf16 in, fp32 Dst (fp32_dest_acc_en), fp32 C.
// TENSIXGEMM_OUT=bf16:           bf16 in, 16-bit Dst, bf16 C.
// TENSIXGEMM_FIDELITY=LoFi|HiFi2 (default)|HiFi3|HiFi4.
// The capture runs a 64x64x64 problem and checks it against a host reference.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>

using namespace tt::tt_metal;
using tt::CBIndex;
using tt::DataFormat;

constexpr uint32_t TILE = 32;
constexpr uint32_t TILE_ELEMS = TILE * TILE;
constexpr uint32_t CAP_TILES = 64;
constexpr uint32_t MT = 2, KT = 2, NT = 2;

static uint32_t rng_state = 0x2545F491u;
static uint32_t next_rand() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float bf16_to_float(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// A small random bf16 in [-2, 2).
static uint16_t rand_bf16() {
    float f = (static_cast<int32_t>(next_rand() % 4096) - 2048) / 1024.0f;
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}

// Offset of element (r, c) inside a 32x32 tile: four 16x16 faces, row-major.
static uint32_t tile_offset(uint32_t r, uint32_t c) {
    return ((r / 16) * 2 + (c / 16)) * 256 + (r % 16) * 16 + (c % 16);
}

static std::shared_ptr<Buffer> make_l1(IDevice* device, uint32_t tile_bytes) {
    ShardSpecBuffer spec(
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))),
        {CAP_TILES * TILE, TILE},
        ShardOrientation::ROW_MAJOR,
        {TILE, TILE},
        {CAP_TILES, 1});
    ShardedBufferConfig config{
        .device = device,
        .size = CAP_TILES * tile_bytes,
        .page_size = tile_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = spec};
    return CreateBuffer(config);
}

static void make_cb(Program& program, CoreCoord core, CBIndex cb, DataFormat fmt, uint32_t tile_bytes, Buffer& buf) {
    CircularBufferConfig config = CircularBufferConfig(CAP_TILES * tile_bytes, {{cb, fmt}})
                                      .set_page_size(cb, tile_bytes)
                                      .set_globally_allocated_address(buf);
    CreateCircularBuffer(program, core, config);
}

int main() {
    const char* out_env = std::getenv("TENSIXGEMM_OUT");
    const std::string out = out_env ? out_env : "fp32";
    if (out != "fp32" && out != "bf16") {
        fprintf(stderr, "unknown TENSIXGEMM_OUT=%s\n", out.c_str());
        return 2;
    }
    const bool fp32_out = out == "fp32";
    const char* fid_env = std::getenv("TENSIXGEMM_FIDELITY");
    const std::string fid = fid_env ? fid_env : "HiFi2";
    MathFidelity fidelity = MathFidelity::HiFi2;
    if (fid == "LoFi") {
        fidelity = MathFidelity::LoFi;
    } else if (fid == "HiFi3") {
        fidelity = MathFidelity::HiFi3;
    } else if (fid == "HiFi4") {
        fidelity = MathFidelity::HiFi4;
    } else if (fid != "HiFi2") {
        fprintf(stderr, "unknown TENSIXGEMM_FIDELITY=%s\n", fid.c_str());
        return 2;
    }

    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t in_tile_bytes = 2 * TILE_ELEMS;
    const uint32_t out_tile_bytes = (fp32_out ? 4 : 2) * TILE_ELEMS;

    auto a_buf = make_l1(device, in_tile_bytes);
    auto b_buf = make_l1(device, in_tile_bytes);
    auto c_buf = make_l1(device, out_tile_bytes);
    make_cb(program, core, CBIndex::c_0, DataFormat::Float16_b, in_tile_bytes, *a_buf);
    make_cb(program, core, CBIndex::c_1, DataFormat::Float16_b, in_tile_bytes, *b_buf);
    make_cb(program, core, CBIndex::c_2, fp32_out ? DataFormat::Float32 : DataFormat::Float16_b, out_tile_bytes, *c_buf);

    // Row-major A (MT*32 x KT*32) and B (KT*32 x NT*32), then tiled.
    const uint32_t M = MT * TILE, K = KT * TILE, N = NT * TILE;
    std::vector<uint16_t> a(M * K), b(K * N);
    for (auto& v : a) {
        v = rand_bf16();
    }
    for (auto& v : b) {
        v = rand_bf16();
    }
    std::vector<uint16_t> a_tiles(CAP_TILES * TILE_ELEMS, 0), b_tiles(CAP_TILES * TILE_ELEMS, 0);
    for (uint32_t r = 0; r < M; r++) {
        for (uint32_t c = 0; c < K; c++) {
            a_tiles[((r / TILE) * KT + c / TILE) * TILE_ELEMS + tile_offset(r % TILE, c % TILE)] = a[r * K + c];
        }
    }
    for (uint32_t r = 0; r < K; r++) {
        for (uint32_t c = 0; c < N; c++) {
            b_tiles[((r / TILE) * NT + c / TILE) * TILE_ELEMS + tile_offset(r % TILE, c % TILE)] = b[r * N + c];
        }
    }
    detail::WriteToBuffer(a_buf, a_tiles);
    detail::WriteToBuffer(b_buf, b_tiles);

    KernelHandle compute = CreateKernel(
        program,
        "kernels/compute/gemm.cpp",
        core,
        ComputeConfig{.math_fidelity = fidelity, .fp32_dest_acc_en = fp32_out});
    SetRuntimeArgs(program, compute, core, {MT, KT, NT});

    detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> c_raw;
    detail::ReadFromBuffer(c_buf, c_raw);
    CloseDevice(device);

    double max_err = 0.0, max_ref = 0.0;
    for (uint32_t r = 0; r < M; r++) {
        for (uint32_t c = 0; c < N; c++) {
            double ref = 0.0;
            for (uint32_t k = 0; k < K; k++) {
                ref += static_cast<double>(bf16_to_float(a[r * K + k])) * bf16_to_float(b[k * N + c]);
            }
            const uint32_t idx = ((r / TILE) * NT + c / TILE) * TILE_ELEMS + tile_offset(r % TILE, c % TILE);
            float got;
            if (fp32_out) {
                std::memcpy(&got, &c_raw[idx], 4);
            } else {
                const uint16_t h = static_cast<uint16_t>(c_raw[idx / 2] >> (16 * (idx % 2)));
                got = bf16_to_float(h);
            }
            max_err = std::fmax(max_err, std::fabs(got - ref));
            max_ref = std::fmax(max_ref, std::fabs(ref));
        }
    }
    printf(
        "TENSIXGEMM_CAPTURE: out=%s fidelity=%s a=0x%x b=0x%x c=0x%x cap_tiles=%u\n",
        out.c_str(),
        fid.c_str(),
        static_cast<uint32_t>(a_buf->address()),
        static_cast<uint32_t>(b_buf->address()),
        static_cast<uint32_t>(c_buf->address()),
        CAP_TILES);
    printf("TENSIXGEMM_ERROR: max_abs=%g max_ref=%g\n", max_err, max_ref);
    if (!(max_err <= 0.02 * max_ref)) {
        printf("Failure on the device: max abs error %g\n", max_err);
        return 1;
    }
    printf("Completed successfully on the device\n");
    return 0;
}
