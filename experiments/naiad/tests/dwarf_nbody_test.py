"""The nbody dwarf (kernels/naiad/dwarfs/nbody): host-side checks, then each variant on Naiad."""

import importlib.util
import shutil

import numpy as np
import pytest

from experiments.naiad.dwarfs import (
    DEFAULT_DESIGN,
    NAIAD_CC,
    build_point,
    load_spec,
    run_point,
)
from framework.matrix.accuracy import compare
from framework.naiad.kernel_io import run_kernel

SPEC = load_spec("nbody")
SMALLEST = (min(s[0] for s in SPEC.SIZES), "uniform")

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
needs_ml_dtypes = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)


def test_expansion_matches_direct_distances_in_fp64():
    inputs = SPEC.make_inputs((64, "uniform"), np.random.default_rng(0))
    pos = inputs["pos"]
    n2 = np.sum(pos**2, axis=1)
    expansion = n2[:, None] + n2[None, :] - 2 * pos @ pos.T
    np.testing.assert_allclose(expansion, SPEC.golden(inputs)["d2"], atol=1e-14)
    np.testing.assert_allclose(
        SPEC.forces(pos, inputs, expansion), SPEC.golden(inputs)["force"], rtol=1e-8
    )


def test_clustered_inputs_are_close_pairs_far_from_the_origin():
    inputs = SPEC.make_inputs((64, "clustered"), np.random.default_rng(0))
    d2 = SPEC.golden(inputs)["d2"]
    near = d2[d2 > 0].min()
    assert near < 1e-3 * np.max(np.sum(inputs["pos"] ** 2, axis=1))


def test_lj_law_and_flop_count():
    inputs = SPEC.make_inputs((64, "uniform", "lj"), np.random.default_rng(0))
    assert np.isfinite(SPEC.golden(inputs)["force"]).all()
    assert SPEC.useful_flops((64, "uniform", "lj")) == 64 * 63 * 23
    assert SPEC.useful_flops(64) == 64 * 63 * 18


@needs_clang
@needs_ml_dtypes
# Variants needing optional matrix features run in dwarf_fused_test.py.
@pytest.mark.parametrize(
    "variant", sorted(set(SPEC.VARIANTS) - set(SPEC.VARIANT_FEATURES))
)
def test_variant_matches_golden_fp32(variant, tmp_path):
    r = run_point(SPEC, variant, SMALLEST, "fp32", "fp32", build_root=tmp_path)
    assert r.exit_code == 0, r.console
    assert r.region_cycles
    assert r.region_cycles > 0
    assert r.passed, r.error
    assert set(r.error) == {"force", "d2"}
    if variant.startswith("ame"):
        assert r.counters["matrix.macs"] == 16 * SMALLEST[0] ** 2 * (
            2 if variant in ("ame_fgemm", "ame_rvv") else 1
        )


@needs_clang
@needs_ml_dtypes
def test_bf16_scalar_passes_and_the_expansion_cancels(tmp_path):
    """At bf16 the direct loop is acceptable and the GEMM expansion is not: the study's point."""
    scalar = run_point(SPEC, "scalar", SMALLEST, "bf16", "fp32", build_root=tmp_path)
    assert scalar.passed, scalar.error
    ame = run_point(SPEC, "ame", SMALLEST, "bf16", "fp32", build_root=tmp_path)
    assert ame.exit_code == 0, ame.console
    assert ame.error["force"]["normwise"] > 5 * scalar.error["force"]["normwise"]


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize("variant", ["ame", "ame_fgemm"])
def test_bf16_ame_matches_the_expansion_model(variant, tmp_path):
    """The AME kernels compute exactly the bf16 expansion, up to fp32 accumulation."""
    elf = build_point(SPEC, variant, SMALLEST, "bf16", "fp32", build_root=tmp_path)
    inputs = SPEC.make_inputs(SMALLEST, np.random.default_rng(0))
    run = run_kernel(
        DEFAULT_DESIGN,
        elf,
        SPEC.encode(inputs, "bf16", "fp32"),
        SPEC.outputs(SMALLEST, "fp32"),
    )
    assert run.ok, run.console
    got = SPEC.decode(run.outputs, "fp32")
    d2, force = SPEC.expansion_model(inputs, "bf16")
    assert compare(got["d2"], d2)["normwise"] < 1e-6
    assert compare(got["force"], force)["normwise"] < 2e-2
