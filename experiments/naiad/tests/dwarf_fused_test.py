"""The epilogue-fused variants (fft and nbody ``ame_fused``) and their feature ablations.

Each runs on naiad-dwarfs with the element-wise features listed (the
``epilogue`` sweep's design) and must match the fp64 golden as well as the
unfused ``ame`` variant does: the twiddles / pair weights are the same
arithmetic, moved from the scalar core into the accumulators.
"""

import importlib.util
import shutil

import pytest
import yaml

from experiments.naiad.dwarfs import (
    DEFAULT_DESIGN,
    NAIAD_CC,
    DwarfError,
    load_spec,
    run_point,
)

pytestmark = [
    pytest.mark.skipif(
        shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
        reason=f"needs make and {NAIAD_CC}",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
    ),
]

BASE = ["transpose", "shift", "epilogue", "semiring"]
FUSED = [*BASE, "acc_move", "convert", "ewise", "ewise_tx"]
FFT, NBODY = load_spec("fft"), load_spec("nbody")
FFT_SIZE, NB_SIZE = (256, 2), (64, "uniform")


def _design(tmp_path_factory, name, features):
    raw = yaml.safe_load(DEFAULT_DESIGN.read_text())
    raw["matrix"]["features"] = list(features)
    path = tmp_path_factory.mktemp("design") / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture(scope="module")
def fused(tmp_path_factory):
    return _design(tmp_path_factory, "fused", FUSED)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    return tmp_path_factory.mktemp("build")


def _run(spec, variant, size, formats, design, build, flags=()):
    r = run_point(
        spec, variant, size, *formats, design, build_root=build, extra_cflags=flags
    )
    assert r.exit_code == 0, r.console
    return r


@pytest.mark.parametrize(
    "formats", [("fp32", "fp32"), ("fp64", "fp64"), ("bf16", "fp32"), ("fp16", "fp32")]
)
def test_fft_fused_matches_golden_like_ame(formats, fused, build):
    r = _run(FFT, "ame_fused", FFT_SIZE, formats, fused, build)
    base = _run(FFT, "ame", FFT_SIZE, formats, fused, build)
    assert r.passed, r.error
    assert r.error["X"]["normwise"] <= 1.25 * base.error["X"]["normwise"]
    assert r.region_cycles * 5 < base.region_cycles  # the scalar twiddle pass is gone
    assert r.counters["hart0.matrix.ewise_ops"] == 4 * FFT_SIZE[0] * FFT_SIZE[
        1
    ] // 256 * (len(FFT.plan(256)) - 1)
    if formats[0] in (
        "bf16",
        "fp16",
    ):  # same twiddle arithmetic, same single RNE to in_fmt
        assert r.error["X"] == base.error["X"]


def test_fft_without_fma_and_without_unit_conversion(fused, build, tmp_path_factory):
    full = _run(FFT, "ame_fused", FFT_SIZE, ("bf16", "fp32"), fused, build)
    nofma = _run(
        FFT, "ame_fused", FFT_SIZE, ("bf16", "fp32"), fused, build, ["-DFFT_EW_FMA=0"]
    )
    assert nofma.passed, nofma.error
    assert (
        nofma.counters["hart0.matrix.ewise_ops"]
        == full.counters["hart0.matrix.ewise_ops"] * 6 // 4
    )
    # without acc_move / convert the narrowing is a scalar pass: identical bits, more cycles
    noconv = _design(tmp_path_factory, "noconv", [*BASE, "ewise"])
    sw = _run(FFT, "ame_fused", FFT_SIZE, ("bf16", "fp32"), noconv, build)
    assert sw.error["X"] == full.error["X"]
    assert sw.region_cycles > full.region_cycles


@pytest.mark.parametrize(
    "formats", [("fp32", "fp32"), ("fp64", "fp64"), ("bf16", "fp32")]
)
def test_nbody_fused_matches_ame(formats, fused, build):
    r = _run(NBODY, "ame_fused", NB_SIZE, formats, fused, build)
    base = _run(NBODY, "ame", NB_SIZE, formats, fused, build)
    assert r.error["d2"] == base.error["d2"]  # the same distance GEMM, stored raw
    assert r.error["force"]["normwise"] <= 1.25 * base.error["force"]["normwise"]
    assert r.passed == base.passed  # bf16 fails for both: the expansion cancels
    assert r.region_cycles * 10 < base.region_cycles


def test_nbody_ablations_keep_the_numbers(fused, build, tmp_path_factory):
    full = _run(NBODY, "ame_fused", NB_SIZE, ("fp32", "fp32"), fused, build)
    nomove = _design(
        tmp_path_factory, "nomove", [*BASE, "convert", "ewise", "ewise_tx"]
    )
    r = _run(NBODY, "ame_fused", NB_SIZE, ("fp32", "fp32"), nomove, build)
    assert (
        r.error == full.error
    )  # W through memory instead of amov: a bit copy either way
    assert r.region_cycles > full.region_cycles
    notx = _design(tmp_path_factory, "notx", [*BASE, "acc_move", "convert", "ewise"])
    r = _run(NBODY, "ame_fused", NB_SIZE, ("fp32", "fp32"), notx, build)
    assert r.passed, r.error  # w = 1 / (s sqrt s) on the scalar side instead of r^3
    assert r.region_cycles > 5 * full.region_cycles


def test_fused_variants_refuse_a_design_without_ewise(build):
    for spec, size in ((FFT, FFT_SIZE), (NBODY, NB_SIZE)):
        with pytest.raises(DwarfError, match="ewise"):
            run_point(
                spec,
                "ame_fused",
                size,
                "fp32",
                "fp32",
                DEFAULT_DESIGN,
                build_root=build,
            )
