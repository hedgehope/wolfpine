"""The Ztt-tracking feature switches of the AME dwarfs, and the dwarfs on the naiad-ztt preset.

Each switch runs at the dwarf's smallest size on naiad-dwarfs with every
optional matrix feature listed (the feature_ranking sweep's design) and must
match the fp64 golden. Switches that only move data (``mconv``, ``mmov``)
must also reproduce the default kernel's errors exactly.
"""

import importlib.util
import shutil

import numpy as np
import pytest
import yaml

from experiments.naiad.dwarfs import DEFAULT_DESIGN, NAIAD_CC, load_spec, run_point
from framework.matrix.encodings import TABLE
from framework.naiad import DESIGNS_DIR

pytestmark = [
    pytest.mark.skipif(
        shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
        reason=f"needs make and {NAIAD_CC}",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
    ),
]

ALL_FEATURES = [f for f in TABLE.features if f != "acc_mem"]


@pytest.fixture(scope="module")
def design(tmp_path_factory):
    raw = yaml.safe_load(DEFAULT_DESIGN.read_text())
    raw["matrix"]["features"] = ALL_FEATURES
    path = tmp_path_factory.mktemp("design") / "naiad-dwarfs-features.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


CASES = [
    ("jacobi", {"mconv": True}, ("bf16", "fp32"), True),
    ("jacobi", {"mconv": True}, ("fp16", "fp32"), True),
    ("jacobi", {"form": "window", "tshift2": True}, ("fp32", "fp32"), False),
    ("jacobi", {"form": "window", "mconv": True}, ("fp16", "fp32"), False),
    ("scan", {"mmov": True}, ("fp32", "fp32"), True),
    ("scan", {"prefix": True}, ("fp32", "fp32"), False),
    ("scan", {"prefix": True}, ("bf16", "fp32"), False),
    ("reduce", {"finish": "mmov"}, ("fp32", "fp32"), True),
    ("reduce", {"finish": "native"}, ("bf16", "fp32"), False),
]


@pytest.mark.parametrize(
    "dwarf, switches, formats, same_bits",
    CASES,
    ids=[
        f"{d}-{'-'.join(f'{k}={v}' for k, v in s.items())}-{f[0]}"
        for d, s, f, _ in CASES
    ],
)
def test_switch_matches_golden(dwarf, switches, formats, same_bits, design, tmp_path):
    spec = load_spec(dwarf)
    size = min(spec.SIZES)
    flags = spec.switch_cflags(switches)
    r = run_point(
        spec, "ame", size, *formats, design, build_root=tmp_path, extra_cflags=flags
    )
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
    if same_bits:
        base = run_point(spec, "ame", size, *formats, design, build_root=tmp_path)
        for name in base.error:
            assert r.error[name] == base.error[name]
        assert r.region_cycles != base.region_cycles  # it did change the kernel


def test_default_switches_emit_no_flags():
    for dwarf in ("jacobi", "scan", "reduce"):
        assert load_spec(dwarf).switch_cflags() == []


def test_tshift2_is_window_form_only():
    with pytest.raises(ValueError, match="form='window' only"):
        load_spec("jacobi").switch_cflags({"tshift2": True})


@pytest.mark.parametrize("dwarf", ["gemm", "jacobi", "scan", "reduce"])
def test_dwarfs_run_on_the_ztt_preset(dwarf, tmp_path):
    """No aload/astore: matrix.h lowers them to amov + a tile store through scratch tiles."""
    spec = load_spec(dwarf)
    size = min(spec.SIZES)
    r = run_point(
        spec,
        "ame",
        size,
        "fp32",
        "fp32",
        DESIGNS_DIR / "naiad-ztt.yaml",
        build_root=tmp_path,
    )
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
    assert r.counters.get("matrix.mem_to_acc_bytes", 0) == 0
    assert r.counters.get("matrix.acc_to_mem_bytes", 0) == 0
    assert r.counters["matrix.acc_to_tile_bytes"] > 0
    assert np.isfinite(r.region_cycles)
