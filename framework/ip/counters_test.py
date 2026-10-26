import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from framework.ip.base import IPBlock
from framework.ip.counters import CounterError, CounterHub
from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import NaiadBuildError, build
from framework.naiad.design import DesignError, load_design, parse_design
from framework.naiad.gen import gen_header
from framework.naiad.registry import IP_REGISTRY
from framework.naiad.run import main
from framework.pe.rv.isa.zicsr64_isa import (
    RV64_ZICSR_ISA,
    CSRFile64,
    UnknownCSRError,
)
from framework.util.elf_testing import write_elf

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS = REPO_ROOT / "kernels" / "naiad"
COUNTERS_DESIGN = DESIGNS_DIR / "naiad-counters.yaml"
NAIAD_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")


def _raw(**counters):
    raw = yaml.safe_load(COUNTERS_DESIGN.read_text())
    if counters:
        raw["counters"] = counters
    return raw


def _soc(raw=None, **kwargs):
    return build(parse_design(_raw() if raw is None else raw), **kwargs)


class Stub:
    def __init__(self, **values):
        self.values = values

    def counters(self):
        return dict(self.values)


# -- the hub ---------------------------------------------------------------------


def test_hub_discovers_core_soc_and_block_sources():
    soc = _soc(_raw())
    core = soc.cores[0]
    core.timing = Stub(stall_mem=3, matrix_busy=7)
    prefixed = Stub(x=1)
    prefixed.counter_prefix = "noc"
    soc.shared = Stub(link_bytes=11)
    soc.other = prefixed
    core._private = Stub(hidden=1)
    hub = CounterHub.discover(soc)
    names = hub.names()
    assert names[:2] == ["cycle", "mtime"]
    assert "hart0.timing.stall_mem" in names
    assert "hart0.timing.matrix_busy" in names
    assert "timing.stall_mem" in names  # the aggregate over harts
    assert "shared.link_bytes" in names
    assert "noc.x" in names
    assert not any("hidden" in n for n in names)
    assert "htif.chars" in names
    assert "hart0.matrix.macs" in names
    # Readers are live: a source's later value is what the hub returns.
    core.timing.values["stall_mem"] = 42
    assert hub.read("timing.stall_mem") == 42
    assert hub.snapshot()["hart0.timing.stall_mem"] == 42


def test_hub_aggregates_and_legacy_names():
    soc = _soc(_raw())
    soc.cores[0].matrix.stats["macs"] = 9
    hub = soc.counter_hub
    assert hub.read("matrix.macs") == hub.read("hart0.matrix.macs") == 9
    assert hub.read("matrix_macs") == 9  # the pre-hub flat name
    assert soc.counter("htif_chars") == 0
    assert hub.resolve("htif_chars") == "htif.chars"
    with pytest.raises(CounterError, match="unknown counter event 'nope'"):
        hub.read("nope")


def test_hub_registers_a_new_ip_block_without_edits(monkeypatch):
    class Widget(IPBlock):
        mapped = False

        def counters(self):
            return {"widgets": 5}

        def read(self, addr, size):
            raise IndexError

        def write(self, addr, value, size=None):
            raise IndexError

    monkeypatch.setitem(IP_REGISTRY, "widget", Widget)
    raw = _raw()
    raw["blocks"].append({"name": "w0", "type": "widget"})
    raw["blocks"][2]["params"]["counters"].append("w0.widgets")
    soc = _soc(raw)
    assert soc.counter("w0.widgets") == 5
    perf = soc.blocks["perf"]
    assert perf.read(8 * perf.names.index("w0.widgets"), 8) == (5).to_bytes(8, "little")


def test_on_demand_families_are_off_unless_asked_for():
    raw = yaml.safe_load((DESIGNS_DIR / "naiad-ame.yaml").read_text())
    soc = _soc(raw)
    core = soc.cores[0]
    # Nothing counts: the ISA chain is exactly the bound ``run`` methods.
    assert core._isa_runs == [isa.run for isa in core.isas]
    assert "instret.scalar" not in soc.counter_hub
    with pytest.raises(CounterError, match="'classes' family"):
        soc.counter_hub.read("instret.scalar")
    soc = _soc(raw, count_all=True)
    assert soc.counter_hub.probes
    assert "bus.dram.read_bytes" in soc.counter_hub
    raw["counters"] = {"enable": ["classes"]}
    soc = _soc(raw)
    assert "instret.fp" in soc.counter_hub
    assert "bus.dram.read_bytes" not in soc.counter_hub


# -- Zihpm CSRs --------------------------------------------------------------------


def test_csr_hpm_read_write_offset_and_event_semantics():
    csrs = CSRFile64(register_file=None)
    value = [100]
    csrs.bind_hpm(3, lambda: value[0])
    assert csrs.read(0xB03) == csrs.read(0xC03) == 100
    csrs.write(0xB03, 1000)
    value[0] = 105
    assert csrs.read(0xC03) == 1005  # counts on from the written value
    csrs.write(0xB03, 0)
    assert csrs.read(0xB03) == 0
    value[0] = 104  # an event may only grow, but the offset arithmetic wraps cleanly
    assert csrs.read(0xB03) == (1 << 64) - 1
    assert csrs.read(0x323) == 3
    assert csrs.read(0x324) == 0
    csrs.write(0x323, 7)  # WARL: the binding is the design's
    assert csrs.read(0x323) == 3
    assert csrs.read(0xB1F) == csrs.read(0xC1F) == 0  # unbound: hardwired zero
    csrs.write(0xB1F, 5)
    assert csrs.read(0xB1F) == 0
    with pytest.raises(UnknownCSRError, match="read-only CSR hpmcounter3"):
        csrs.write(0xC03, 1)
    with pytest.raises(UnknownCSRError):
        csrs.read(0xC20)
    csrs.reset()
    assert csrs.read(0xB03) == 104
    with pytest.raises(ValueError, match="numbered 3-31"):
        csrs.bind_hpm(2, lambda: 0)


def _csrr(rd, csr):
    return (csr << 20) | (2 << 12) | (rd << 7) | 0x73


def test_design_bindings_reach_the_csrs():
    soc = _soc()
    core = soc.cores[0]
    soc.cores[0].matrix.stats["macs"] = 4096
    rf = core.register_file
    assert RV64_ZICSR_ISA.run(rf, soc.memory, False, _csrr(10, 0xC08))
    assert rf[10].read_uint() == 4096  # hpmcounter8 = matrix.macs
    assert RV64_ZICSR_ISA.run(rf, soc.memory, False, _csrr(10, 0x328))
    assert rf[10].read_uint() == 8


# -- design validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "counters, message",
    [
        ({"hpm": {2: "cycle"}}, "Zihpm counters are 3-31"),
        ({"hpm": {32: "cycle"}}, "Zihpm counters are 3-31"),
        ({"hpm": {3: "cycle", "3": "instret"}}, "counter 3 is bound twice"),
        ({"hpm": {3: "cycle", 4: "cycle"}}, "bound to more than one counter"),
        ({"hpm": {3: ""}}, "expected an event name"),
        ({"hpm": [3]}, "expected a mapping"),
        ({"enable": ["bogus"]}, "unknown \\['bogus'\\]"),
        ({"bogus": 1}, "unknown key"),
    ],
)
def test_bad_counters_sections_are_rejected(counters, message):
    with pytest.raises(DesignError, match=message):
        parse_design(_raw(**counters))


def test_a_repeated_yaml_key_is_rejected(tmp_path):
    text = COUNTERS_DESIGN.read_text().replace(
        "    4: instret.fp\n", "    3: instret.fp\n"
    )
    path = tmp_path / "dup.yaml"
    path.write_text(text)
    with pytest.raises(DesignError, match="key 3 appears twice"):
        load_design(path)


def test_hpm_needs_rv64_zicsr_and_zihpm():
    raw = _raw()
    raw["cores"]["isa"] = [e for e in raw["cores"]["isa"] if e != "zihpm"]
    with pytest.raises(DesignError, match="zihpm"):
        parse_design(raw)


def test_unknown_events_fail_at_build():
    with pytest.raises(NaiadBuildError, match="unknown counter event 'matrix.nope'"):
        _soc(_raw(hpm={3: "matrix.nope"}))
    raw = _raw()
    raw["blocks"][2]["params"]["counters"] = ["cycle", "dma9.bytes"]
    with pytest.raises(NaiadBuildError, match="'dma9.bytes'"):
        _soc(raw)


# -- MMIO, header, report ------------------------------------------------------------


def test_perfcounters_read_hub_events():
    soc = _soc()
    soc.cores[0].matrix.stats["tile_to_mem_bytes"] = 1024
    perf = soc.blocks["perf"]
    index = perf.names.index("matrix.tile_to_mem_bytes")
    assert int.from_bytes(perf.read(8 * index, 8), "little") == 1024
    assert int.from_bytes(perf.read(8 * index + 4, 4), "little") == 0


def test_header_names_bindings_and_offsets():
    header = gen_header(load_design(COUNTERS_DESIGN))
    assert "#define NAIAD_HAS_HPM 1" in header
    assert "#define NAIAD_HPM_MATRIX_MACS 8" in header
    assert "#define NAIAD_COUNTER_MATRIX_MACS 8, -1" in header
    assert "#define NAIAD_PERFOFF_MATRIX_MMA 0x20" in header
    assert "#define NAIAD_COUNTER_MATRIX_MMA 7, 0x20" in header
    assert "#define NAIAD_COUNTER_HTIF_CHARS 0, 0x18" in header
    assert "#define NAIAD_PERF_HTIF_CHARS 0x00000018UL" in header


def test_counters_json(tmp_path, capsys):
    tohost = 0x80001000
    program = [
        (tohost >> 12 << 12) | (5 << 7) | 0x37,  # lui t0, tohost
        (15 << 20) | (10 << 7) | 0x13,  # addi a0, x0, 15 (exit 7)
        (10 << 20) | (5 << 15) | (2 << 12) | 0x23,  # sw a0, 0(t0)
        0x6F,
    ]
    code = b"".join(w.to_bytes(4, "little") for w in program)
    elf = write_elf(
        tmp_path / "prog.elf",
        [(0x80000000, code, len(code)), (tohost, bytes(0x48), 0x48)],
        0x80000000,
        {"tohost": tohost, "fromhost": tohost + 0x40},
    )
    out = tmp_path / "counters.json"
    design = DESIGNS_DIR / "naiad-base.yaml"
    assert (
        main([str(design), str(elf), "--counters-json", str(out), "--count-all"]) == 7
    )
    data = json.loads(out.read_text())
    assert data["exit_code"] == 7
    assert data["cycles"] == 6
    assert not data["timed_out"]
    counters = data["counters"]
    assert counters["instret"] == 6
    assert counters["instret.scalar"] == 6
    assert counters["bus.dram.write_bytes"] == 4
    assert counters["bus.bootrom.read_bytes"] == 4  # the boot ROM's lw of the entry
    assert "naiad:   instret.scalar = 6" in capsys.readouterr().err


# -- end to end ---------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
def test_counters_kernel_sees_exact_deltas(tmp_path, capsys):
    subprocess.run(
        [
            "make",
            "-C",
            str(KERNELS),
            f"BUILD={tmp_path}",
            f"PYTHON={sys.executable}",
            f"NAIAD_CC={NAIAD_CC}",
            "PROG=counters",
            f"DESIGN={COUNTERS_DESIGN}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out_json = tmp_path / "c.json"
    elf = tmp_path / "counters.elf"
    assert main([str(COUNTERS_DESIGN), str(elf), "--counters-json", str(out_json)]) == 0
    out, _ = capsys.readouterr()
    assert "MISMATCH" not in out
    assert out.endswith("PASS\n")
    counters = json.loads(out_json.read_text())["counters"]
    classes = sum(
        counters[f"instret.{c}"] for c in ("scalar", "fp", "vector", "matrix")
    )
    assert classes == counters["instret"]
    assert counters["instret.matrix"] == 6
