"""IME on Naiad: the ``matrix: {mode: ime}`` schema, the generator, the builder wiring and
counters, the GEMM emitter's IME mode, and the clang end-to-end kernels (gemm_ime and the
gemm dwarf's ``ime`` variant). Spike has no Zvvm, so nothing here is difftested."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import build
from framework.naiad.design import (
    CoreSpec,
    DesignError,
    _parse_ime_matrix,
    load_design,
    parse_design,
)
from framework.naiad.gen import gen_header

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS = REPO_ROOT / "kernels" / "naiad"
IME = DESIGNS_DIR / "naiad-ime.yaml"
IME_TIMED = DESIGNS_DIR / "naiad-ime-timed.yaml"
NAIAD_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")
needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)


def _raw(matrix, isa=("i", "m", "f", "d", "v", "zicsr"), vlen=256):
    cores = {"xlen": 64, "isa": list(isa), "reset_pc": 0x80000000}
    if "v" in isa:
        cores["vlen"] = vlen
    return {
        "name": "t",
        "cores": cores,
        "blocks": [
            {"name": "dram", "type": "dram", "base": 0x80000000, "window": 0x100000}
        ],
        "matrix": {"mode": "ime", **matrix},
    }


# -- schema ------------------------------------------------------------------------------------------


def test_defaults_are_the_largest_lambda_per_sew():
    d = parse_design(_raw({}))
    spec = d.matrix
    assert spec.mode == "ime"
    assert spec.vlen == 256
    assert spec.lambdas == {8: (4,), 16: (4,), 32: (2,), 64: (2,)}
    assert spec.extensions == ("Zvvfp32mm", "Zvvmtls", "Zvvmttls")
    red = spec.reduction(32, 1, 2)
    assert (red.g, red.psm, red.rnd) == (1, 0, "frm")


def test_shipped_designs():
    for path in (IME, IME_TIMED):
        d = load_design(path)
        assert d.matrix.lambdas[32] == (1, 2)
        assert ("bf16", "fp32") in d.matrix.format_pairs
    assert load_design(IME_TIMED).timing.matrix.vrf_write_bytes_per_cycle == 32


@pytest.mark.parametrize(
    "matrix,isa,message",
    [
        ({}, ("i", "m", "f", "d", "zicsr"), "vector unit"),
        ({"extensions": ["Zvvfoo"]}, None, "not a Zvvm extension"),
        ({"extensions": ["Zvvi4i8mm"]}, None, "not modelled"),
        ({"extensions": ["Zvvfp32mm", "zvvfp32mm"]}, None, "twice"),
        ({"extensions": ["Zvvmtls"]}, None, "no multiply-accumulate"),
        ({"lambdas": {32: [4]}}, None, "not permissible"),
        ({"lambdas": {24: [1]}}, None, "SEW 24"),
        ({"fp_reduction": {"32,1,2": {"G": 4}}}, None, "power of two in 1..lambda"),
        ({"fp_reduction": {"32,1,2": {"psm": 1}}}, None, "not modelled"),
        ({"fp_reduction": {"32,1,2": {"rnd": "rtz"}}}, None, "rnd must be"),
        ({"fp_reduction": {"32,1,4": {}}}, None, "not supported"),
        ({"fp_reduction": {"32-1-2": {}}}, None, "not 'sew,w,lambda'"),
        ({"nonsense": 1}, None, "unknown key"),
    ],
)
def test_schema_errors(matrix, isa, message):
    raw = _raw(matrix) if isa is None else _raw(matrix, isa)
    with pytest.raises(DesignError, match=message):
        parse_design(raw)


def test_ime_and_vme_cannot_share_a_hart():
    cores = CoreSpec(xlen=64, isa=("i", "f", "d", "v", "zvtbase"), vlen=256, elen=64)
    with pytest.raises(DesignError, match="same LOAD-FP/STORE-FP encodings"):
        _parse_ime_matrix({"mode": "ime"}, cores)


def test_extension_names_are_case_insensitive_and_canonical():
    d = parse_design(_raw({"extensions": ["zvvbf16fp32mm", "ZVVMTLS"]}))
    assert d.matrix.extensions == ("Zvvbf16fp32mm", "Zvvmtls")


# -- generator, builder, counters -------------------------------------------------------------------------


def test_header_has_the_encodings_and_lambdas():
    h = gen_header(load_design(IME))
    assert "#define NAIAD_HAS_IME 1" in h
    assert "#define NAIAD_MATRIX_IME 1" in h
    assert "NAIAD_HAS_MATRIX" not in h  # matrix.h (AME) must not be pulled in
    assert "#define NAIAD_IME_VFMMACC_VV_FUNCT7 0x29" in h
    assert "#define NAIAD_IME_VFWMMACC_VV_FUNCT7 0x2b" in h
    assert "#define NAIAD_IME_VMTTL_V_FUNCT7 0xb" in h
    assert "#define NAIAD_IME_VTYPE_LAMBDA_SHIFT 60" in h
    assert "#define NAIAD_IME_LAMBDAS_E32 0x6" in h  # lambda 1 (enc 1) and 2 (enc 2)
    assert "#define NAIAD_IME_EXT_ZVVBF16FP32MM 1" in h


def test_builder_attaches_ime_after_rvv_with_counters():
    from framework.matrix.ime_isa import RV64_IME_ISA, IMEUnit
    from framework.matrix.ime_timing import IMETiming

    soc = build(load_design(IME_TIMED), count_all=True)
    core = soc.cores[0]
    assert isinstance(core.matrix, IMEUnit)
    assert core.matrix.vs is core.vector
    assert core.isas.index(RV64_IME_ISA) > [i.__name__ for i in core.isas].index(
        "RV64_V_ISA"
    )
    assert isinstance(core.rv_cost.vector, IMETiming)
    assert core.rv_cost.matrix is None
    c = soc.counters()
    for key in (
        "mma",
        "macs",
        "vrf_a_read_bytes",
        "vrf_b_read_bytes",
        "vrf_c_read_bytes",
        "vrf_c_write_bytes",
        "mem_to_tile_bytes",
        "tile_to_mem_bytes",
    ):
        assert f"hart0.matrix.{key}" in c
        assert f"matrix.{key}" in c
    assert "hart0.instret.matrix" in c
    assert "hart0.timing.ime_vrf_write_busy_cycles" in c


# -- the GEMM emitter's IME mode ----------------------------------------------------------------------------


def test_emitter_plans_and_header():
    from framework.naiad.emit.gemm import emit_gemm_header
    from framework.naiad.emit.gemm_ime import PairPlan

    d = load_design(IME_TIMED)
    plan = PairPlan(d, "fp32", "fp32")
    assert (plan.lam, plan.m, plan.emul_c, plan.sew, plan.w) == (1, 8, 8, 32, 1)
    assert (
        plan.bm * plan.bn * plan.emul_c
        + (plan.bm + plan.bn) * plan.lmul * (2 if plan.db else 1)
        <= 32
    )
    bf = PairPlan(d, "bf16", "fp32")
    assert bf.w == 2
    assert bf.ke == 2 * bf.lam * bf.lmul
    assert (bf.afa, bf.afb) == (1, 1)
    bb = PairPlan(d, "bf16", "bf16")
    assert (bb.sew, bb.lam, bb.altfmt) == (16, 2, 1)  # lambda 1 would need EMUL_C 16
    h = emit_gemm_header(d)
    for pair in (
        "fp64_fp64",
        "fp32_fp32",
        "bf16_fp32",
        "fp16_fp32",
        "bf16_bf16",
        "int8_int32",
    ):
        assert f"naiad_gemm_{pair}_ukernel" in h
    assert "#ifndef NAIAD_MATRIX_IME" in h
    assert "NAIAD_VLEN != 256" in h
    forced = PairPlan(d, "fp32", "fp32", block=(1, 2))
    assert (forced.bm, forced.bn) == (1, 2)
    with pytest.raises(ValueError, match="does not fit"):
        PairPlan(d, "fp32", "fp32", block=(4, 4))


def test_emitter_cli_writes_an_error_header_without_tile_loads(tmp_path):
    import yaml

    from framework.naiad.emit.gemm import main

    raw = _raw({"extensions": ["Zvvfp32mm"]})
    path = tmp_path / "d.yaml"
    path.write_text(yaml.safe_dump(raw))
    out = tmp_path / "naiad_gemm.h"
    assert main([str(path), "-o", str(out)]) == 0
    assert "#error" in out.read_text()
    assert "Zvvmtls" in out.read_text()


# -- clang end to end -----------------------------------------------------------------------------------------


def _make(tmp_path, design, prog):
    subprocess.run(
        [
            "make",
            "-C",
            str(KERNELS),
            f"BUILD={tmp_path}",
            f"PYTHON={sys.executable}",
            f"NAIAD_CC={NAIAD_CC}",
            f"DESIGN={design}",
            f"PROG={prog}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tmp_path / f"{prog}.elf"


def _numpy_gemm_fp32(A, B):
    """Each product rounded to fp32, then added in K order (G = 1, psm = 0, rnd = frm)."""
    C = np.zeros((A.shape[0], B.shape[1]), np.float32)
    for k in range(A.shape[1]):
        C = C + np.outer(A[:, k], B[k, :]).astype(np.float32)
    return C


def _numpy_gemm_bf16(A, B):
    """Each sub-dot-product of two bf16 products rounded once to fp32, then added."""
    a, b = A.astype(np.float64), B.astype(np.float64)
    C = np.zeros((A.shape[0], B.shape[1]), np.float32)
    for k in range(0, A.shape[1], 2):
        s = np.outer(a[:, k], b[k, :]) + np.outer(
            a[:, k + 1], b[k + 1, :]
        )  # exact for these data
        C = C + s.astype(np.float32)
    return C


@needs_clang
@pytest.mark.parametrize("design", [IME, IME_TIMED], ids=["untimed", "timed"])
def test_gemm_ime_passes_and_matches_numpy(tmp_path, design):
    import ml_dtypes

    from framework.naiad.kernel_io import run_kernel

    elf = _make(tmp_path, design, "gemm_ime")
    n = 32
    outs = {
        "Af": (np.float32, (n, n)),
        "Bf": (np.float32, (n, n)),
        "Cf": (np.float32, (n, n)),
        "Ab": (np.uint16, (n, n)),
        "Bb": (np.uint16, (n, n)),
        "Cb": (np.float32, (n, n)),
        "cycles_fp32": (np.uint64, (1,)),
    }
    run = run_kernel(design, elf, outputs=outs, count_all=True)
    assert run.ok, run.console
    assert run.console.endswith("PASS\n")
    assert "MISMATCH" not in run.console
    o = run.outputs
    assert np.array_equal(
        o["Cf"].view(np.uint32), _numpy_gemm_fp32(o["Af"], o["Bf"]).view(np.uint32)
    )
    bf = ml_dtypes.bfloat16
    A, B = o["Ab"].view(bf), o["Bb"].view(bf)
    assert np.array_equal(
        o["Cb"].view(np.uint32), _numpy_gemm_bf16(A, B).view(np.uint32)
    )
    assert np.abs(o["Cf"] - o["Af"].astype(np.float64) @ o["Bf"]).max() < 1e-4
    c = run.counters
    assert c["matrix.macs"] == 2 * n**3
    assert c["matrix.mma"] == n**3 // (8 * 8 * 4) + n**3 // (
        8 * 8 * 8
    )  # fp32 8 x 8 x 4, bf16 8 x 8 x 8
    if design == IME_TIMED:
        assert int(o["cycles_fp32"][0]) > 0


@needs_clang
@pytest.mark.parametrize(
    "pair", ["fp32:fp32", "fp64:fp64", "bf16:fp32", "fp16:fp32", "bf16:bf16"]
)
def test_gemm_dwarf_ime_variant(tmp_path, pair):
    from framework.naiad.dwarfs import load_spec, run_point

    spec = load_spec("gemm")
    assert spec.VARIANTS["ime"] == "ime"
    in_fmt, acc_fmt = pair.split(":")
    r = run_point(
        spec, "ime", 32, in_fmt, acc_fmt, design=IME_TIMED, build_root=tmp_path
    )
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
    assert r.counters["matrix.macs"] == 32**3
    assert 0 < r.region_cycles < r.cycles


@needs_clang
def test_gemm_dwarf_ime_ragged_edges(tmp_path):
    from framework.naiad.dwarfs import load_spec, run_point

    spec = load_spec("gemm")
    for size in ((40, 24, 20), (17, 9, 7)):
        for in_fmt, acc_fmt in (("fp32", "fp32"), ("bf16", "fp32")):
            r = run_point(
                spec, "ime", size, in_fmt, acc_fmt, design=IME, build_root=tmp_path
            )
            assert r.passed, (size, in_fmt, r.error, r.console)
            m, n, k = size
            assert r.counters["matrix.macs"] >= m * n * k
