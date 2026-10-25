"""The Naiad design's ``matrix:`` section: validation, generated defines, the built SoC, and gemm end to end."""

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from framework.matrix import formats
from framework.matrix.ame_isa import RV_AME_ISA
from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import NaiadBuildError, build
from framework.naiad.design import DesignError, load_design, parse_design
from framework.naiad.gen import gen_header
from framework.naiad.run import main

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS = REPO_ROOT / "kernels" / "naiad"
AME_DESIGN = DESIGNS_DIR / "naiad-ame.yaml"
NAIAD_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")

BASE = {
    "name": "t",
    "cores": {"xlen": 64, "isa": ["i", "m"], "reset_pc": 0x1000},
    "matrix": {"m": 8, "n": 4, "k": 2, "formats": {"in": ["fp32"], "acc": ["fp32"]}},
    "blocks": [
        {"name": "bootrom", "type": "bootrom", "base": 0x1000, "window": 0x1000},
        {"name": "dram", "type": "dram", "base": 0x80000000, "window": "1MiB"},
        {"name": "htif", "type": "htif"},
    ],
}


def _design(**matrix):
    raw = copy.deepcopy(BASE)
    raw["matrix"].update(matrix)
    return raw


def test_the_shipped_ame_design():
    design = load_design(AME_DESIGN)
    spec = design.matrix
    assert (spec.mode, spec.m, spec.n, spec.k, spec.tiles, spec.accs) == (
        "ame",
        16,
        16,
        16,
        8,
        4,
    )
    assert spec.in_formats == ("fp32", "int8")
    assert spec.acc_formats == ("fp32", "int32")
    assert "f" not in design.cores.isa
    assert "d" not in design.cores.isa


def test_a_design_without_a_matrix_section_has_none():
    assert load_design(DESIGNS_DIR / "naiad-base-rv64.yaml").matrix is None
    raw = copy.deepcopy(BASE)
    del raw["matrix"]
    soc = build(parse_design(raw))
    assert soc.matrix_units == []
    assert RV_AME_ISA not in soc.cores[0].isas


def test_defaults_and_aliases():
    spec = parse_design(
        _design(formats={"in": ["float32", "e4m3"], "acc": ["FP32"]})
    ).matrix
    assert spec.in_formats == ("fp32", "fp8_e4m3")
    assert spec.acc_formats == ("fp32",)
    assert (spec.tiles, spec.accs, spec.features, spec.k_chunk) == (8, 4, (), None)


@pytest.mark.parametrize(
    "change,message",
    [
        (
            {"mode": "ime"},
            r"matrix \(mode ime\): unknown key",
        ),  # AME keys; see ime_test.py
        (
            {"mode": "vme"},
            r"matrix \(mode vme\): unknown key",
        ),  # AME keys; see vme_test.py
        ({"mode": "xyz"}, "matrix.mode must be one of"),
        ({"tiles": 33}, "matrix.tiles must be in 1-32"),
        ({"accs": 0}, "matrix.accs must be in 1-32"),
        ({"m": 0}, "matrix.m must be in"),
        ({"formats": {"in": ["fp33"], "acc": ["fp32"]}}, "unknown format"),
        (
            {"formats": {"in": ["int8"], "acc": ["int8"]}},
            "int8 \\(code 8\\) does not fit the acc_fmt field",
        ),
        (
            {"formats": {"in": ["int8", "fp32"], "acc": ["fp32"]}},
            "int8 has no accumulator of its kind",
        ),
        (
            {"formats": {"in": ["fp32"], "acc": ["fp8_e5m2"]}},
            "not an accumulator format",
        ),
        ({"formats": {"in": ["fp32", "float32"], "acc": ["fp32"]}}, "twice"),
        ({"formats": {"in": [], "acc": ["fp32"]}}, "non-empty list"),
        ({"formats": {"inputs": ["fp32"]}}, "unknown key"),
        ({"features": ["teleport"]}, "unknown \\['teleport'\\]"),
        ({"k_chunk": 0}, "k_chunk must be >= 1"),
        ({"size": 3}, "unknown key"),
    ],
)
def test_bad_matrix_sections_are_rejected(change, message):
    with pytest.raises(DesignError, match=message):
        parse_design(_design(**change))


def test_header_defines_shape_formats_features_and_encodings():
    header = gen_header(load_design(AME_DESIGN))
    for line in (
        "#define NAIAD_HAS_MATRIX 1",
        "#define NAIAD_MATRIX_AME 1",
        "#define NAIAD_MATRIX_M 16",
        "#define NAIAD_MATRIX_TILES 8",
        "#define NAIAD_MATRIX_IN_INT8 1",
        "#define NAIAD_MATRIX_ACC_INT32 1",
        "#define NAIAD_MATRIX_FEATURE_SEMIRING 1",
        "#define NAIAD_MFMT_FP32 0x1",
        "#define NAIAD_MI_MMA_OPCODE 0x2b",
        "#define NAIAD_MI_TLOAD_A_T_SHIFT 0x4",
        "#define NAIAD_PERF_MATRIX_MMA",
    ):
        assert line in header, line
    assert "NAIAD_HAS_MATRIX" not in gen_header(
        load_design(DESIGNS_DIR / "naiad-base-rv64.yaml")
    )


def test_the_builder_attaches_one_unit_per_hart_last_in_the_chain():
    soc = build(load_design(AME_DESIGN))
    core = soc.cores[0]
    unit = soc.matrix_units[0]
    assert core.isas[-1] is RV_AME_ISA
    assert core._isa_runs[-1] == RV_AME_ISA.run
    assert core.register_file.matrix is unit
    assert core.matrix is unit
    assert unit.tiles.shape == (8, 16, 16 * 4)
    assert unit.accs.shape == (4, 16, 16 * 4)
    unit.stats["macs"] = 7
    assert soc.counters()["hart0.matrix.macs"] == 7
    assert soc.counter("matrix_macs") == 7
    soc.reset()
    assert soc.counter("matrix_macs") == 0


def test_a_bf16_design_without_ml_dtypes_fails_at_build(monkeypatch):
    design = parse_design(_design(formats={"in": ["bf16"], "acc": ["fp32"]}))
    monkeypatch.setitem(sys.modules, "ml_dtypes", None)
    monkeypatch.setattr(formats, "_ML_DTYPES", None)
    with pytest.raises(NaiadBuildError, match="ml_dtypes"):
        build(design)
    build(
        parse_design(
            _design(formats={"in": ["fp64", "fp16", "int8"], "acc": ["fp64", "int32"]})
        )
    )


# -- end to end: clang -> ELF -> Naiad --------------------------------------

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)


def _make(build_dir, **variables):
    args = [
        "make",
        "-C",
        str(KERNELS),
        f"BUILD={build_dir}",
        f"PYTHON={sys.executable}",
        f"NAIAD_CC={NAIAD_CC}",
    ]
    args += [f"{k}={v}" for k, v in variables.items()]
    subprocess.run(args, check=True, capture_output=True, text=True)


@needs_clang
def test_gemm_ame_builds_runs_and_passes(tmp_path, capsys):
    _make(tmp_path, PROG="gemm_ame", DESIGN=AME_DESIGN)
    assert main([str(AME_DESIGN), str(tmp_path / "gemm_ame.elf")]) == 0
    out, err = capsys.readouterr()
    assert "gemm_ame 64x64x64 on a 16x16x16 unit" in out
    assert "MISMATCH" not in out
    assert out.endswith("PASS\n")
    assert "hart0.matrix.mma = 128" in err
    assert "hart0.matrix.macs = 524288" in err
    # 2 GEMMs x 16 output tiles x 4 K steps x 2 tiles each: fp32 (1 KiB a tile) + int8 (256 B)
    assert "hart0.matrix.mem_to_tile_bytes = 81920" in err
    assert "hart0.matrix.acc_to_mem_bytes = 32768" in err


@needs_clang
def test_gemm_scalar_builds_runs_and_passes(tmp_path, capsys):
    _make(tmp_path, PROG="gemm_scalar", DESIGN=AME_DESIGN, CFLAGS="-O2 -DGEMM_M=32")
    assert main([str(AME_DESIGN), str(tmp_path / "gemm_scalar.elf")]) == 0
    out, err = capsys.readouterr()
    assert "gemm_scalar 32x32x32 int32" in out
    assert "hash ok" in out
    assert out.endswith("PASS\n")
    assert "hart0.matrix.mma = 0" in err
