"""Multi-hart Naiad: design rules, crt0/linker generation, CLINT, wfi, LR/SC across harts, timing, counters.

The end-to-end checks build ``kernels/naiad/multicore`` (an amoadd counter, an
lr/sc counter, a spinlock and a reusable barrier on every hart) for 2 and 4
harts, run it on Naiad and, when Spike is installed, difftest it on
``spike -p<n>``; and run the gemm ``ame_mc`` dwarf on the 4-hart preset.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import MulticoreMemory, NaiadMemory, build
from framework.naiad.design import DesignError, load_design, parse_design
from framework.naiad.difftest import difftest, find_spike
from framework.naiad.dwarfs import NAIAD_CC, load_spec, run_point
from framework.naiad.gen import gen_crt0, gen_header, gen_linker

REPO = Path(__file__).resolve().parents[2]
KERNELS = REPO / "kernels" / "naiad"
RV64GC = DESIGNS_DIR / "naiad-rv64gc.yaml"
DWARFS_4C = DESIGNS_DIR / "naiad-dwarfs-4c.yaml"
CLINT = 0x02000000

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
needs_spike = pytest.mark.skipif(
    find_spike() is None, reason="needs spike (tools/naiad/build_spike.sh)"
)


def _raw(path=RV64GC, **cores):
    raw = yaml.safe_load(Path(path).read_text())
    raw["cores"].update(cores)
    return raw


def _design(count=2, path=RV64GC, **cores):
    return parse_design(_raw(path, count=count, **cores))


def _write_design(tmp_path, count, path=RV64GC):
    raw = _raw(path, count=count)
    raw["name"] = f"{raw['name']}-{count}c"
    out = tmp_path / f"{raw['name']}.yaml"
    out.write_text(yaml.safe_dump(raw))
    return out


# -- design rules -------------------------------------------------------------


def test_multicore_design_rules():
    assert _design(4).cores.count == 4
    with pytest.raises(DesignError, match="zicsr"):
        _design(2, isa=["i", "m", "a"])
    with pytest.raises(DesignError, match="zaamo"):
        _design(2, isa=["i", "m", "zicsr"])
    no_clint = _raw(count=2)
    no_clint["blocks"] = [b for b in no_clint["blocks"] if b["type"] != "clint"]
    no_clint["blocks"] = [b for b in no_clint["blocks"] if b["type"] != "perfcounters"]
    with pytest.raises(DesignError, match="clint"):
        parse_design(no_clint)
    small = _raw(count=4)
    small["program"]["stack_size"] = "512MiB"
    with pytest.raises(DesignError, match="stacks"):
        parse_design(small)
    rv32 = yaml.safe_load((DESIGNS_DIR / "naiad-base.yaml").read_text())
    rv32["cores"]["count"] = 2
    with pytest.raises(DesignError, match="xlen 64"):
        parse_design(rv32)


def test_single_hart_generated_files_have_no_multicore_parts():
    one = load_design(RV64GC)
    assert "mhartid" not in gen_crt0(one)
    assert "NAIAD_HARTS" not in gen_linker(one)
    assert "NAIAD_MULTICORE" not in gen_header(one)


def test_multicore_crt0_linker_and_header():
    d = _design(4)
    crt0, linker, header = gen_crt0(d), gen_linker(d), gen_header(d)
    assert "csrr  t0, mhartid" in crt0
    assert "wfi" in crt0
    assert hex(CLINT) in crt0  # the msip array
    assert "NAIAD_HARTS = 4;" in linker
    assert "_stack_top - NAIAD_HARTS * STACK_SIZE" in linker
    assert "#define NAIAD_MULTICORE 1" in header
    assert "#define NAIAD_HARTS 4" in header


# -- SoC pieces ---------------------------------------------------------------


def test_multicore_soc_has_per_hart_cores_and_units():
    soc = build(load_design(DWARFS_4C))
    assert isinstance(soc.memory, MulticoreMemory)
    assert [c.csrs.read(0xF14) for c in soc.cores] == [0, 1, 2, 3]
    assert len({id(u) for u in soc.matrix_units}) == 4
    assert len({id(c.vector) for c in soc.cores}) == 4
    assert len({id(c.rv_cost) for c in soc.cores}) == 4
    assert type(build(load_design(RV64GC)).memory) is NaiadMemory


def test_clint_per_hart_msip_and_mtimecmp():
    soc = build(_design(4))
    mem = soc.memory
    for h in range(4):
        mem.write(CLINT + 4 * h, (0xFFFF_FFFE | (h & 1)).to_bytes(4, "little"))
        mem.write(CLINT + 0x4000 + 8 * h, (1000 + h).to_bytes(8, "little"))
    clint = soc.blocks["clint"]
    assert clint.msip == [0, 1, 0, 1]  # WARL: bit 0 only
    assert int.from_bytes(mem.read(CLINT + 4, 4), "little") == 1
    assert int.from_bytes(mem.read(CLINT, 8), "little") == 1 << 32
    assert [
        int.from_bytes(mem.read(CLINT + 0x4000 + 8 * h, 8), "little") for h in range(4)
    ] == [1000, 1001, 1002, 1003]
    soc.reset()
    assert clint.msip == [0, 0, 0, 0]


def test_wfi_sleeps_until_msip_with_msie():
    soc = build(_design(2))
    soc.reset()
    hart1 = soc.cores[1]
    clint = soc.blocks["clint"]
    hart1.register_file.wfi()  # mie = 0: sleeps for good
    assert not hart1.active
    soc.memory.write(CLINT + 4, (1).to_bytes(4, "little"))
    assert not hart1.active  # MSIE clear: msip does not wake it
    soc.reset()
    hart1.csrs.write(0x304, 1 << 3)  # mie.MSIE
    hart1.register_file.wfi()
    assert not hart1.active
    assert 1 in clint.sleeping
    soc.memory.write(CLINT + 4, (1).to_bytes(4, "little"))
    assert hart1.active
    assert not clint.sleeping
    hart1.register_file.wfi()  # msip still pending: wfi falls through
    assert hart1.active
    soc.reset()
    hart1.csrs.write(0x304, 1 << 7)  # mie.MTIE, mtimecmp 0 <= mtime: already due, a nop
    hart1.register_file.wfi()
    assert hart1.active  # (sleeping until a later mtimecmp: coherence_test.py)


def test_any_store_breaks_other_harts_reservations():
    soc = build(_design(4))
    soc.reset()
    rfs = [c.register_file for c in soc.cores]
    base = 0x80100000
    for i, rf in enumerate(rfs):
        rf.reservation = (base + 8 * i, base + 8 * i + 8)
    soc.memory.write(base + 12, b"\x01")  # inside hart 1's reserved dword only
    assert [rf.reservation is None for rf in rfs] == [False, True, False, False]
    soc.memory.write(base, bytes(32))
    assert all(rf.reservation is None for rf in rfs)


def test_shared_ports_couple_the_harts_regions():
    soc = build(load_design(DWARFS_4C))
    dram = [c.rv_cost.memory.by_name["dram"] for c in soc.cores]
    dram[0].port_free = 77
    assert [r.port_free for r in dram] == [77] * 4
    dram[2].loads += 1
    assert [r.loads for r in dram] == [0, 0, 1, 0]  # counts stay per hart
    raw = _raw(DWARFS_4C)
    raw["timing"]["shared_ports"] = False
    private = build(parse_design(raw))
    dram = [c.rv_cost.memory.by_name["dram"] for c in private.cores]
    dram[0].port_free = 77
    assert [r.port_free for r in dram] == [77, 0, 0, 0]


def test_counters_are_per_hart_with_aggregates():
    soc = build(load_design(DWARFS_4C))
    names = soc.counter_hub.names()
    for h in range(4):
        assert f"hart{h}.instret" in names
        assert f"hart{h}.matrix.macs" in names
    assert "instret" in names
    assert "matrix.macs" in names


# -- end to end ---------------------------------------------------------------


def _build_multicore(tmp_path, design):
    build_dir = tmp_path / f"build-{design.stem}"
    subprocess.run(
        [
            "make",
            "-s",
            "-C",
            str(KERNELS),
            f"DESIGN={design}",
            "PROG=multicore",
            f"BUILD={build_dir}",
            f"NAIAD_CC={NAIAD_CC}",
            f"PYTHON={sys.executable}",
        ],
        check=True,
        capture_output=True,
    )
    return build_dir / "multicore.elf"


@needs_clang
@pytest.mark.parametrize("harts", [2, 4])
def test_atomics_lrsc_lock_and_barrier_across_harts(tmp_path, harts):
    design = _write_design(tmp_path, harts)
    elf = _build_multicore(tmp_path, design)
    soc = build(load_design(design))
    soc.htif.console_stream = None
    soc.load(elf)
    r = soc.run(2_000_000)
    console = r.console.decode()
    assert r.exit_code == 0, console
    assert console.splitlines() == [
        *(f"hart {h} of {harts}: up" for h in range(harts)),
        "PASS",
    ]
    assert r.counters["instret"] == sum(
        r.counters[f"hart{h}.instret"] for h in range(harts)
    )


@needs_clang
def test_lrsc_fails_without_cross_hart_invalidation(tmp_path, monkeypatch):
    """The C test is sensitive to the bus snoop: without it an sc wrongly succeeds."""
    design = _write_design(tmp_path, 4)
    elf = _build_multicore(tmp_path, design)
    monkeypatch.setattr(MulticoreMemory, "write", NaiadMemory.write)
    soc = build(load_design(design))
    soc.htif.console_stream = None
    soc.load(elf)
    r = soc.run(2_000_000)
    assert r.exit_code == 1
    assert b"lr/sc" in r.console


@needs_clang
@needs_spike
@pytest.mark.parametrize("harts", [2, 4])
def test_multicore_agrees_with_spike(tmp_path, harts):
    design = _write_design(tmp_path, harts)
    elf = _build_multicore(tmp_path, design)
    result = difftest(design, elf)
    assert result.ok, result.report()
    assert b"PASS" in result.naiad.console


@needs_clang
def test_gemm_ame_mc_matches_golden_and_speeds_up(tmp_path):
    spec = load_spec("gemm")
    one = run_point(spec, "ame", 128, "fp32", "fp32", build_root=tmp_path)
    four = run_point(
        spec, "ame_mc", 128, "fp32", "fp32", design=DWARFS_4C, build_root=tmp_path
    )
    assert four.exit_code == 0, four.console
    assert four.passed, four.error
    assert (
        four.error["C"]["normwise"] == one.error["C"]["normwise"]
    )  # same blocks, same sums
    # 4 row blocks of 32 over 4 harts: well over 2x (measured 3.2x).
    assert one.region_cycles / four.region_cycles > 2.5
    assert four.counters["matrix.macs"] == 128**3
    assert all(four.counters[f"hart{h}.matrix.macs"] == 128**3 // 4 for h in range(4))


@needs_clang
def test_gemm_ame_mc_with_padded_edges_serialises_correctly(tmp_path):
    spec = load_spec("gemm")
    r = run_point(
        spec,
        "ame_mc",
        (72, 40, 24),
        "fp32",
        "fp32",
        design=DWARFS_4C,
        build_root=tmp_path,
    )
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
