"""RVV on Naiad: the design schema (vlen / elen), the generators, the builder wiring, the
counters, and kernels/naiad/rvv_check end to end (untimed, timed, and against Spike)."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import build
from framework.naiad.design import DesignError, load_design, parse_design
from framework.naiad.difftest import difftest
from framework.naiad.gen import gen_crt0, gen_header
from framework.naiad.run import main
from framework.pe.rv.difftest import find_spike
from framework.pe.rv.isa.v import RV64_V_ISA

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS = REPO_ROOT / "kernels" / "naiad"
RVV = DESIGNS_DIR / "naiad-rvv.yaml"
RVV_TIMED = DESIGNS_DIR / "naiad-rvv-timed.yaml"
NAIAD_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")


def _raw(isa, **cores):
    blocks = [{"name": "dram", "type": "dram", "base": 0x80000000, "window": 0x100000}]
    return {
        "name": "t",
        "cores": {"xlen": 64, "isa": isa, "reset_pc": 0x80000000, **cores},
        "blocks": blocks,
    }


def test_rvv_design():
    design = load_design(RVV)
    assert (
        design.cores.march
        == "rv64imafdv_zicsr_zicntr_zfbfmin_zvfh_zvfbfmin_zvfbfwma_zvl256b"
    )
    assert (design.cores.vlen, design.cores.elen, design.cores.mabi) == (
        256,
        64,
        "lp64d",
    )
    soc = build(design)
    core = soc.cores[0]
    assert RV64_V_ISA in core.isas
    assert core.vector is core.register_file.vector is core.csrs.vector
    assert core.vector.vlen == 256
    assert core.vector.zvfh
    assert core.vector.zvfbfwma
    assert core.csrs.misa >> 21 & 1


@pytest.mark.parametrize(
    "isa,cores,message",
    [
        (["i", "f", "d", "zvfh"], {}, "need a vector unit"),
        (["i", "f", "d"], {"vlen": 256}, "no vector extension"),
        (["i", "v"], {}, "require 'f' and 'd'"),
        (["i", "f", "d", "v"], {"vlen": 64}, "power of two in 128-65536"),
        (["i", "f", "d", "v"], {"vlen": 384}, "power of two"),
        (["i", "f", "d", "v"], {"elen": 32}, "elen must be 64"),
        (["i", "f", "d", "v", "zvfbfwma"], {}, "requires zvfbfmin and zfbfmin"),
    ],
)
def test_vector_schema_errors(isa, cores, message):
    with pytest.raises(DesignError, match=message):
        parse_design(_raw(isa, **cores))


def test_zve64d_allows_vlen_64_and_defaults():
    d = parse_design(_raw(["i", "f", "d", "zve64d"], vlen=64))
    assert d.cores.vlen == 64
    assert d.cores.march.endswith("_zve64d_zvl64b")
    assert parse_design(_raw(["i", "f", "d", "v"])).cores.vlen == 128


def test_generators_know_the_vector_unit():
    design = load_design(RVV)
    header = gen_header(design)
    assert "#define NAIAD_VLEN 256" in header
    assert "#define NAIAD_ELEN 64" in header
    assert "csrs  mstatus" in gen_crt0(design)
    assert "1 << 9" in gen_crt0(design)
    plain = load_design(DESIGNS_DIR / "naiad-fp-rv64.yaml")
    assert "NAIAD_VLEN" not in gen_header(plain)
    assert "1 << 9" not in gen_crt0(plain)


def test_counter_hub_finds_the_vector_unit():
    soc = build(load_design(RVV), count_all=True)
    counters = soc.counters()
    for key in ("instructions", "elements", "mem_to_vec_bytes", "vec_to_mem_bytes"):
        assert f"hart0.vector.{key}" in counters
        assert f"vector.{key}" in counters
    assert "hart0.instret.vector" in counters


needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)


def _build(tmp_path, design):
    subprocess.run(
        [
            "make",
            "-C",
            str(KERNELS),
            f"BUILD={tmp_path}",
            f"PYTHON={sys.executable}",
            f"NAIAD_CC={NAIAD_CC}",
            f"DESIGN={design}",
            "PROG=rvv_check",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tmp_path / "rvv_check.elf"


@needs_clang
@pytest.mark.parametrize("design", [RVV, RVV_TIMED], ids=["untimed", "timed"])
def test_rvv_check_passes(tmp_path, capsys, design):
    elf = _build(tmp_path, design)
    assert main([str(design), str(elf), "--count-all"]) == 0
    out, err = capsys.readouterr()
    assert "MISMATCH" not in out
    assert out.endswith("PASS\n")
    assert "rvv_check: VLEN = 256 bits" in out
    assert "vfwmaccbf16.vv (Zvfbfwma): ok" in out
    assert "hart0.instret.vector = " in err


@needs_clang
@pytest.mark.skipif(
    find_spike() is None, reason="needs Spike (tools/naiad/build_spike.sh)"
)
def test_rvv_check_matches_spike_commit_for_commit(tmp_path):
    elf = _build(tmp_path, RVV)
    result = difftest(RVV, elf, commits=True)
    assert result.ok, result.report()
    assert result.commits.matched > 20000
