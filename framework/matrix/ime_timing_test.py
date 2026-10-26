"""IME's cycle model (``timing.matrix`` in ``mode: ime``): schema and microbenchmarks with analytic counts.

The method of ``framework/pe/rv/timing/vector_test.py``: a hand-encoded body
between two ``rdcycle`` reads. An instruction issuing at ``c1 + 1 + x``
followed directly by the second read gives ``x + 2``. The configuration (a
``vsetvl`` carrying lambda) runs before the first read.

Geometry used throughout: VLEN 256, SEW 32, lambda 2, LMUL 1, so M = N = 4,
K_eff = 2, EMUL_C = 2: an mma is 32 MACs and reads A 32 B + B 32 B + C 64 B =
128 B and writes C 64 B.
"""

import copy

import pytest

from framework.matrix import ime
from framework.matrix.ime import encode
from framework.naiad.builder import build
from framework.naiad.design import DesignError, parse_design
from framework.pe.rv.isa.isa64_testing import csr
from framework.pe.rv.isa.v.vtesting import OPIVI, OPIVV, opv, vsetvl, vtype

PROG = 0x80000000
SP = 0x10000000
T0, T1, A0, A1, A2, A3 = 5, 6, 10, 11, 12, 13
SPIN = 0x0000006F
DRAM_LAT, MEM_STARTUP, LSU = 10, 2, 32


def _raw(matrix=None, vector=None):
    timing = {
        "scalar": {"alu": 1, "load": 1, "store": 1, "branch_taken": 0, "csr": 1},
        "memory": {
            "sp": {"latency": 0, "bandwidth": None},
            "dram": {"latency": DRAM_LAT, "bandwidth": None},
        },
        "vector": {
            "datapath_bits": 256,
            "startup": 4,
            "mem_startup": MEM_STARTUP,
            "mem_bytes_per_cycle": LSU,
        },
        "matrix": {
            "macs_per_cycle": 512,
            "pipeline_depth": 3,
            "vrf_read_bytes_per_cycle": 1024,
            "vrf_write_bytes_per_cycle": 1024,
        },
    }
    timing["matrix"].update(matrix or {})
    timing["vector"].update(vector or {})
    return {
        "name": "imebench",
        "cores": {
            "xlen": 64,
            "isa": ["i", "m", "f", "d", "v", "zicsr", "zicntr"],
            "vlen": 256,
            "elen": 64,
            "reset_pc": PROG,
        },
        "blocks": [
            {"name": "sp", "type": "scratchpad", "base": SP, "window": 0x10000},
            {"name": "dram", "type": "dram", "base": PROG, "window": "1MiB"},
        ],
        "matrix": {
            "mode": "ime",
            "lambdas": {32: [1, 2]},
            "extensions": ["Zvvfp32mm", "Zvvmtls", "Zvvmttls"],
        },
        "timing": copy.deepcopy(timing),
    }


def _run(raw, body, lmul=1, lam=2, regs=None):
    soc = build(parse_design(raw))
    soc.reset()
    core = soc.cores[0]
    regs = dict(regs or {})
    regs.setdefault(A0, SP)
    regs[A3] = vtype(32, lmul) | (
        ime.encode_lambda(lam) << ime.LAMBDA_SHIFT if lam else 0
    )
    words = [vsetvl(A2, 0, A3), csr(2, 0xC00, rd=T0, rs1=0), *body]
    words += [csr(2, 0xC00, rd=T1, rs1=0), SPIN]
    for i, w in enumerate(words):
        soc.memory.write(PROG + 4 * i, w.to_bytes(4, "little"))
    for idx, value in regs.items():
        core.register_file[idx].write(value.to_bytes(8, "little"))
    spin = PROG + 4 * (len(words) - 1)
    for cycle in range(20000):
        soc.clock.clock_tick(cycle)
        if core.pc_register.read_uint() == spin:
            break
    else:
        raise AssertionError("the program did not reach its spin loop")
    rf = core.register_file
    return rf[T1].read_uint() - rf[T0].read_uint(), soc


def _mma(vd, vs1=8, vs2=12):
    return encode("vfmmacc.vv", vd, vs1, vs2)


def _vadd(vd, vs2, vs1):
    return opv(OPIVV, 0x00, vd, vs2, vs1)


def _vmv_vi(vd):
    return opv(OPIVI, 0x17, vd, 0, 0)


# -- the mma's three bounds --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matrix,occ",
    [
        ({"macs_per_cycle": 4}, 8),  # compute: 32 MACs / 4
        ({"vrf_read_bytes_per_cycle": 16}, 8),  # reads: 128 B / 16
        ({"vrf_write_bytes_per_cycle": 8}, 8),  # write-back: 64 B / 8
        ({"mma_ii": 5}, 5),
        ({"format_passes": {"fp32": 3}, "macs_per_cycle": 8}, 12),  # 32 * 3 / 8
    ],
)
def test_independent_mmas_issue_every_occupancy(matrix, occ):
    t, soc = _run(_raw(matrix), [_mma(16), _mma(24)])
    assert t == 2 + occ
    c = soc.counters()
    assert c["hart0.timing.matrix_mma"] == 2
    assert c["hart0.timing.matrix_macs"] == 64
    assert c["hart0.timing.matrix_busy_cycles"] == 2 * occ


def test_defaults_are_the_rvv_datapaths_ports():
    from framework.matrix.ime_timing import IMETimingSpec

    assert IMETimingSpec().ports(256) == (96, 32)
    # Read-bound at 96 B/cycle needs more than 96 B: lambda 1, LMUL 2 (M 8, K 2): 64 + 64 + 256 B.
    t, _ = _run(
        _raw({"vrf_read_bytes_per_cycle": None, "vrf_write_bytes_per_cycle": None}),
        [_mma(16, 8, 12), _mma(24, 8, 12)],
        lmul=2,
        lam=1,
    )
    # reads ceil(384 / 96) = 4, write ceil(256 / 32) = 8, compute 1: occupancy 8.
    assert t == 2 + 8


def test_chained_mma_forwards_and_c_bypass_saves_the_c_read():
    read_bound = {"vrf_read_bytes_per_cycle": 16}
    # Chained on one C: each waits only for the unit (forwarding), 0, 8, 16.
    t, _ = _run(_raw(read_bound), [_mma(16)] * 3)
    assert t == 2 + 16
    # Without forwarding each waits for the previous C: occupancy + depth - 1 = 10 apart.
    t, _ = _run(_raw({**read_bound, "acc_forwarding": False}), [_mma(16)] * 3)
    assert t == 2 + 20
    # c_bypass: the chained ones read only A and B (64 B, 4 cycles): 0, 8, 12.
    t, soc = _run(_raw({**read_bound, "c_bypass": True}), [_mma(16)] * 3)
    assert t == 2 + 12
    assert soc.counters()["hart0.timing.ime_vrf_read_busy_cycles"] == 8 + 4 + 4


# -- hazards and port contention with RVV ----------------------------------------------------------------


def test_rvv_reader_of_c_waits_for_the_whole_mma():
    # vfadd.vv v16, v16, v16 at LMUL 1 reads C's first register: ready at occ + depth - 1 = 8 + 2.
    t, _ = _run(_raw({"macs_per_cycle": 4}), [_mma(16), opv(1, 0x00, 16, 16, 16)])
    assert t == 2 + 10


def test_rvv_op_waits_for_the_read_port_an_mma_holds():
    # A read-bound mma holds the read port 8 cycles; an unrelated vadd.vv waits for it.
    t, _ = _run(_raw({"vrf_read_bytes_per_cycle": 16}), [_mma(16), _vadd(0, 1, 2)])
    assert t == 2 + 8
    # A compute-bound mma holds it 1 cycle (128 B at 1024 B/cycle): the vadd issues next cycle.
    t, _ = _run(_raw({"macs_per_cycle": 4}), [_mma(16), _vadd(0, 1, 2)])
    assert t == 2 + 1


def test_mma_waits_for_ports_rvv_holds():
    # vadd.vv at LMUL 1 reads 64 B: at 16 B/cycle it holds the read port 4 cycles; the mma waits.
    t, _ = _run(_raw({"vrf_read_bytes_per_cycle": 16}), [_vadd(0, 1, 2), _mma(16)])
    assert t == 2 + 4


def test_rvv_writer_of_an_mma_operand_waits_for_its_read():
    # vmv.v.i v8 (the mma's A) reads no register: only the WAR on A holds it, for the mma's 8 read cycles.
    t, _ = _run(_raw({"vrf_read_bytes_per_cycle": 16}), [_mma(16), _vmv_vi(8)])
    assert t == 2 + 8


def test_rvv_only_programs_time_as_the_plain_vector_model():
    from framework.pe.rv.timing.vector import VectorTiming

    body = [_vadd(0, 1, 2), _vadd(3, 4, 5), _vadd(6, 0, 3)]
    t_ime, soc = _run(
        _raw({"vrf_read_bytes_per_cycle": 1, "vrf_write_bytes_per_cycle": 1}), body
    )
    raw = _raw()
    del raw["matrix"], raw["timing"]["matrix"]
    t_rvv, soc_rvv = _run(raw, body, lam=None)
    assert type(soc_rvv.cores[0].rv_cost.vector) is VectorTiming
    assert t_ime == t_rvv


# -- tile loads and stores -------------------------------------------------------------------------------


def test_tile_load_latency_feeds_the_mma():
    # Dense vmtl (LD = 0) of 32 B from dram: t = 1; ready at mem_startup + latency + t = 13.
    t, _ = _run(
        _raw(), [encode("vmtl.v", 8, A0, 0), _mma(16)], regs={A0: PROG + 0x8000}
    )
    assert t == 2 + MEM_STARTUP + DRAM_LAT + 1


@pytest.mark.parametrize(
    "name,ld,segments",
    [
        ("vmtl.v", 0, 1),  # LD = lambda * LMUL: one run
        ("vmtl.v", 100, 4),  # VL / (lambda * LMUL) = 4 runs of 2 elements
        ("vmttl.v", 0, 1),  # LD = M = VL / (lambda * LMUL): one run
        ("vmttl.v", 100, 2),  # lambda * LMUL = 2 runs of 4 elements
        ("vmts.v", 100, 4),
    ],
)
def test_tile_transfers_cost_their_contiguous_runs(name, ld, segments):
    t, soc = _run(
        _raw(), [encode(name, 8, A0, A1), encode(name, 12, A0, A1)], regs={A1: ld}
    )
    assert t == 2 + segments  # the second waits for the LSU
    assert soc.counters()["hart0.timing.vector_mem_busy_cycles"] == 2 * segments


def test_tile_segment_rate_is_a_parameter():
    t, _ = _run(
        _raw({"tile_segments_per_cycle": 2}),
        [encode("vmtl.v", 8, A0, A1), encode("vmtl.v", 12, A0, A1)],
        regs={A1: 100},
    )
    assert t == 2 + 2


# -- schema --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matrix,message",
    [
        ({"tile_load_bandwidth": 64}, "unknown key"),
        ({"format_passes": {"bf16": 2}}, "not an input format"),
        ({"vrf_read_bytes_per_cycle": 0}, ">= 1"),
        ({"c_bypass": 1}, "true or false"),
    ],
)
def test_schema_errors(matrix, message):
    with pytest.raises(DesignError, match=message):
        parse_design(_raw(matrix))


def test_timing_off_without_a_timing_section():
    raw = _raw()
    del raw["timing"]
    soc = build(parse_design(raw))
    assert soc.cores[0].rv_cost is None


# -- the equal-resource normaliser (mode="ime") ------------------------------------------------------------


def test_normalise_ime_equal_storage_peak_and_bandwidth():
    from framework.matrix.timing import normalise

    # naiad-dwarfs' AME at fp32: 8 tiles + 4 accumulators of 16 x 16 fp32 = 12 KiB, 2048 MACs/cycle, 64 B/cycle.
    ame = normalise(4096, 12 * 1024, 64)
    assert ame.derived["storage_bytes"] == 12 * 1024
    cfg = normalise(4096, 12 * 1024, 64, mode="ime")
    d = cfg.derived
    # 12 KiB over 32 registers is VLEN 3072, so 2048 (8 KiB of VRF).
    assert (d["vlen"], d["vrf_bytes"]) == (2048, 8192)
    assert cfg.timing == {
        "macs_per_cycle": 2048,
        "vrf_read_bytes_per_cycle": 768,
        "vrf_write_bytes_per_cycle": 256,
    }
    # SEW 32 at VLEN 2048 permits lambda 2 (EMUL_C 16) and 4 (EMUL_C 4): lambda 4 has a one-instruction C store.
    assert (d["lambda"], d["m"], d["emul_c"], d["lmul"], d["k_eff"]) == (
        4,
        16,
        4,
        8,
        32,
    )
    # Reads 16*32*4*2 + 16*16*4 = 5120 B at 768 B/cycle: 7 cycles; compute 8192 / 2048 = 4; write 1024 / 256 = 4.
    assert (
        d["mma_compute_cycles"],
        d["mma_read_cycles"],
        d["mma_write_cycles"],
        d["mma_ii"],
    ) == (4, 7, 4, 7)
    assert d["vector_mem_bytes_per_cycle"] == 64
    assert cfg.matrix == {"mode": "ime"}


def test_normalise_ime_pinned_vlen_reports_the_storage_gap():
    from framework.matrix.timing import normalise

    d = normalise(4096, 12 * 1024, 64, mode="ime", vlen=256).derived
    assert d["vlen"] == 256
    assert d["storage_ratio"] == pytest.approx(1 / 12)
    # lambda 1 (M 8, EMUL_C 8), LMUL 8: 512 MACs, port-bound at 8 cycles both ways.
    assert (d["lambda"], d["m"], d["k_eff"], d["mma_ii"]) == (1, 8, 8, 8)
    assert d["achieved_flops_per_cycle"] == 128
    bf = normalise(4096, 12 * 1024, 64, mode="ime", vlen=256, in_format="bf16").derived
    assert bf["k_eff"] == 16
    assert bf["mma_macs"] == 1024


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"vlen": 384}, "power of two"),
        ({"in_format": "fp32", "acc_format": "bf16"}, "widening"),
    ],
)
def test_normalise_ime_errors(kwargs, message):
    from framework.matrix.timing import normalise

    with pytest.raises(ValueError, match=message):
        normalise(4096, 12 * 1024, 64, mode="ime", **kwargs)
    with pytest.raises(ValueError, match="minimum of 128"):
        normalise(4096, 256, 64, mode="ime")


def test_a_normalised_ime_config_builds():
    from framework.matrix.timing import normalise

    cfg = normalise(4096, 12 * 1024, 64, mode="ime", vlen=256)
    raw = _raw(cfg.timing)
    soc = build(parse_design(raw))
    t = soc.cores[0].rv_cost.vector
    assert t.ime_spec.macs_per_cycle == 2048
    assert t.ime_spec.ports(256) == (96, 32)
