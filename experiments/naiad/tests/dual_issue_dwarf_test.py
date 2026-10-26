"""Dual issue on a two-hart dwarf (moved from ``framework/naiad/dual_issue_test.py``)."""

import shutil

import pytest
import yaml

from experiments.naiad.dwarfs import NAIAD_CC, load_spec, multicore_design, run_point

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)


@needs_clang
def test_dual_issue_on_two_harts(tmp_path):
    """rvv_mc reduce on two harts: both pass with the same error; dual issue is faster and pairs on each hart."""
    two_harts = multicore_design(2, tmp_path / "designs")
    raw = yaml.safe_load(two_harts.read_text())
    raw["timing"]["scalar"]["issue_width"] = 2
    dual_design = tmp_path / "designs" / "dual.yaml"
    dual_design.write_text(yaml.safe_dump(raw, sort_keys=False))
    spec = load_spec("reduce")
    args = (spec, "rvv_mc", 4096, "fp32", "fp32")
    one = run_point(*args, design=two_harts, build_root=tmp_path, count_all=True)
    two = run_point(*args, design=dual_design, build_root=tmp_path, count_all=True)
    assert one.passed
    assert two.passed
    assert one.error == two.error
    assert two.region_cycles < one.region_cycles
    assert two.counters["hart0.timing.paired"] > 0
    assert two.counters["hart1.timing.paired"] > 0
