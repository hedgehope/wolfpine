"""VME's cycle model (``timing.matrix`` in ``mode: vme``): schema, normaliser and microbenchmarks with analytic counts.

The method of ``ime_timing_test.py``: a hand-encoded body between two ``rdcycle`` reads; an
instruction issuing at ``c1 + 1 + x`` followed directly by the second read gives ``x + 2``. The
configuration (``msetmtypei`` + ``msett*``) runs before the first read.

Geometry: VLEN 256, TE 8, fp32 (SEW 32, TWIDEN 1, LMUL 1): a multiply is 8 x 8 x 1 = 64 MACs
and reads A 32 B + B 32 B from the VRF; a tile row is 32 B.
"""

import copy

import pytest

from framework.matrix import vme
from framework.matrix.timing import normalise
from framework.matrix.vme import encode
from framework.naiad.builder import build
from framework.naiad.design import DesignError, parse_design
from framework.pe.rv.isa.isa64_testing import csr
from framework.pe.rv.isa.v.vtesting import OPIVV, opv, vmem

PROG = 0x80000000
SP = 0x10000000
T0, T1, A0, A1, A2, A3 = 5, 6, 10, 11, 12, 13
SPIN = 0x0000006F
DRAM_LAT, MEM_STARTUP, LSU = 10, 2, 32


def _raw(matrix=None, vector=None, mode="vme"):
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
            "tile_bytes_per_cycle": 1024,
        },
    }
    timing["matrix"].update(matrix or {})
    timing["vector"].update(vector or {})
    raw = {
        "name": "vmebench",
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
        "matrix": {"mode": "vme", "te": 8, "extensions": ["Zvtfmm"]},
        "timing": copy.deepcopy(timing),
    }
    if mode is None:
        del raw["matrix"], raw["timing"]["matrix"]
    return raw


def _config(edge=8):
    return [
        encode("msetmtypei", altfmt=0, vsew=2, mtwiden=1),
        encode("msettm", rd=0, rs1=A1),
        encode("msettk", rd=0, rs1=A2),
        encode("msettn", rd=0, rs1=A1),
    ]


def _run(raw, body, prologue=None, regs=None):
    soc = build(parse_design(raw))
    soc.reset()
    core = soc.cores[0]
    regs = {A0: SP, A1: 8, A2: 1, **(regs or {})}
    words = [
        *(_config() if prologue is None else prologue),
        csr(2, 0xC00, rd=T0, rs1=0),
        *body,
    ]
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


def _mma(tile, vs2=8, vs1=16):
    return encode("vtfmm.tvv", mtd=tile >> 1, vs2=vs2, vs1=vs1)


@pytest.mark.parametrize(
    "matrix,occ",
    [
        ({"macs_per_cycle": 4}, 16),  # compute: 64 MACs / 4
        ({"vrf_read_bytes_per_cycle": 8}, 8),  # operand reads: 64 B / 8 (no C traffic)
        (
            {"vrf_write_bytes_per_cycle": 1},
            1,
        ),  # the VRF write port is not used by a multiply
        ({"mma_ii": 5}, 5),
        ({"format_passes": {"fp32": 3}, "macs_per_cycle": 8}, 24),  # 64 * 3 / 8
    ],
)
def test_independent_multiplies_issue_every_occupancy(matrix, occ):
    t, soc = _run(_raw(matrix), [_mma(0), _mma(4)])
    assert t == 2 + occ
    c = soc.counters()
    assert c["hart0.timing.matrix_mma"] == 2
    assert c["hart0.timing.matrix_macs"] == 128
    assert c["hart0.timing.matrix_busy_cycles"] == 2 * occ
    assert (
        c["hart0.matrix.vrf_a_read_bytes"] == c["hart0.matrix.vrf_b_read_bytes"] == 64
    )


@pytest.mark.parametrize("forwarding,gap", [(True, 4), (False, 4 + 3 - 1)])
def test_chained_multiplies_accumulate_in_place(forwarding, gap):
    raw = _raw({"macs_per_cycle": 16, "acc_forwarding": forwarding})  # occupancy 4
    t, _ = _run(raw, [_mma(0), _mma(0)])
    assert t == 2 + gap


def test_vtmv_waits_for_the_multiply_and_holds_the_write_port():
    raw = _raw({"macs_per_cycle": 16, "vrf_write_bytes_per_cycle": 8})
    # vtmv.v.t of a row of tile 0: waits occ 4 + depth 3 - 1 = 6, then 32 B / 8 = 4 cycles on the port.
    body = [_mma(0), encode("vtmv.v.t", vd=24, rs1=A3)]
    t, soc = _run(raw, body, regs={A3: vme.tss(0, 0)})
    assert t == 2 + 6
    t2, _ = _run(raw, [*body, opv(OPIVV, 0x00, 1, 24, 24)], regs={A3: vme.tss(0, 0)})
    # The vadd reads v24: ready 4 (port) + 1 after the move issued.
    assert t2 == 2 + 6 + 5
    assert soc.counters()["hart0.timing.vme_tile_port_busy_cycles"] == 4


def test_tile_load_uses_the_vector_lsu_and_a_reader_waits_for_it():
    raw = _raw(vector={"mem_bytes_per_cycle": 8})
    load = encode("vtle", eee=2, rs1=A0, rs2=A3)
    t, _ = _run(raw, [load, load], regs={A3: vme.tss(0, 1)})
    assert (
        t == 2 + 4
    )  # 32 B / 8: the second waits for the LSU (rows do not wait for each other)
    t, _ = _run(raw, [load, _mma(0)], regs={A3: vme.tss(0, 1)})
    assert (
        t == 2 + MEM_STARTUP + 4
    )  # the multiply reads the tile: load latency (sp: 0) + transfer


def test_rvv_only_programs_time_as_on_a_plain_rvv_design():
    body = [vmem(True, 32, 8, A0), opv(OPIVV, 0x00, 16, 8, 8), vmem(False, 32, 16, A0)]
    from framework.pe.rv.isa.v.vtesting import vsetvli, vtype

    pro = [vsetvli(A2, 0, vtype(32, 4))]
    t_vme, _ = _run(_raw(), body, prologue=pro)
    t_rvv, _ = _run(_raw(mode=None), body, prologue=pro)
    assert t_vme == t_rvv


def test_builder_installs_the_vme_model_with_counters():
    from framework.matrix.vme_isa import VMEUnit
    from framework.matrix.vme_timing import VMETiming

    soc = build(parse_design(_raw()), count_all=True)
    core = soc.cores[0]
    assert isinstance(core.matrix, VMEUnit)
    assert isinstance(core.rv_cost.vector, VMETiming)
    assert core.rv_cost.matrix is None
    c = soc.counters()
    for key in (
        "mma",
        "macs",
        "vec_to_tile_bytes",
        "tile_to_vec_bytes",
        "mem_to_tile_bytes",
        "tile_to_mem_bytes",
    ):
        assert f"hart0.matrix.{key}" in c
    for key in (
        "matrix_busy_cycles",
        "vme_vrf_read_busy_cycles",
        "vme_tile_port_busy_cycles",
    ):
        assert f"hart0.timing.{key}" in c


@pytest.mark.parametrize(
    "matrix,message",
    [
        ({"nonsense": 1}, "nonsense"),
        ({"format_passes": {"bf16": 2}}, "not an input format"),
        ({"tile_bytes_per_cycle": 0}, "tile_bytes_per_cycle"),
    ],
)
def test_schema_errors(matrix, message):
    with pytest.raises(DesignError, match=message):
        parse_design(_raw(matrix))


def test_normaliser_equal_peak_and_ports():
    cfg = normalise(4096, 12288, 64, mode="vme", vlen=256)
    assert cfg.mode == "vme"
    assert cfg.matrix == {"mode": "vme", "te": 16}
    assert cfg.timing == {
        "macs_per_cycle": 2048,
        "vrf_read_bytes_per_cycle": 96,
        "vrf_write_bytes_per_cycle": 32,
        "tile_bytes_per_cycle": 64,
    }
    d = cfg.derived
    assert (d["tm"], d["tn"], d["tk"], d["lmul"], d["mma_ii"]) == (
        16,
        16,
        1,
        2,
        2,
    )  # 128 B / 96
    assert d["storage_bytes"] == 16 * 16 * 16 + 1024
    pinned = normalise(4096, 12288, 64, mode="vme", vlen=256, te=32).derived
    assert (
        pinned["tm"],
        pinned["lmul"],
        pinned["mma_ii"],
        pinned["storage_bytes"],
    ) == (32, 4, 3, 17408)
    bf16 = normalise(
        4096, 12288, 64, mode="vme", in_format="bf16", acc_format="fp32", te=32
    ).derived
    assert (bf16["tk"], bf16["lmul"], bf16["mma_macs"]) == (2, 2, 2048)
    with pytest.raises(ValueError, match="te must be"):
        normalise(4096, 12288, 64, mode="vme", te=128)
    with pytest.raises(ValueError, match="no Zvt multiply"):
        normalise(4096, 12288, 64, mode="vme", in_format="fp32", acc_format="fp64")
