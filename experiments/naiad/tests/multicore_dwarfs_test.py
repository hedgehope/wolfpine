"""The dwarfs' multi-hart variants (``*_mc``) at their smallest size, on 1 and 2 harts.

Each is checked against its single-hart variant: on one hart (naiad-dwarfs) it
must pass with the same error and (except where the split itself costs a few
instructions) the same region cycles; on two harts (``multicore_design(2)``)
it must pass with the same error where every output keeps its sum order
(row/block splits: gemm, jacobi, spmv, apsp), and within the tolerance where
the split reorders a reduction (reduce, scan). Also: a shared region port is
held for its own bytes-per-cycle time, not the requester's transfer time; and
the gemm ``ame_mc`` variant on the 4-hart preset.
"""

import shutil

import pytest

from experiments.naiad.dwarfs import (
    DEFAULT_DESIGN,
    NAIAD_CC,
    load_spec,
    multicore_design,
    run_point,
)

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)

#: (dwarf, variant, size, in, acc, same_order, region cycles on one hart may exceed the single variant's by)
CASES = [
    ("reduce", "ame_mc", 4096, "fp32", "fp32", False, 0),
    ("reduce", "rvv_mc", 4096, "fp32", "fp32", False, 0),
    ("scan", "ame_mc", 4096, "fp32", "fp32", False, 0),
    ("spmv", "ame_mc", (128, 0.5), "fp32", "fp32", True, 16),
    ("spmv", "ame_mc", (128, 0.5, 4), "fp32", "fp32", True, 16),
    ("apsp", "ame_mc", 32, "fp32", "fp32", True, 40),
    ("jacobi", "ame_mc", 34, "fp32", "fp32", True, 0),
    ("jacobi", "ame_mc", 34, "bf16", "fp32", True, 0),
    ("jacobi", "rvv_mc", 34, "fp32", "fp32", True, 0),
    ("gemm", "rvv_mc", 32, "fp32", "fp32", True, 0),
]


def _err(r):
    return max(e["normwise"] for e in r.error.values())


@needs_clang
@pytest.mark.parametrize(
    ("dwarf", "variant", "size", "fi", "fa", "same_order", "slack"), CASES
)
def test_mc_variant_on_one_and_two_harts(
    tmp_path_factory, dwarf, variant, size, fi, fa, same_order, slack
):
    root = tmp_path_factory.getbasetemp() / "mc-build"
    spec = load_spec(dwarf)
    single = run_point(spec, variant[: -len("_mc")], size, fi, fa, build_root=root)
    one = run_point(spec, variant, size, fi, fa, design=DEFAULT_DESIGN, build_root=root)
    assert one.exit_code == 0, one.console
    assert one.passed, one.error
    assert _err(one) == _err(single)
    assert single.region_cycles <= one.region_cycles + 16
    assert one.region_cycles <= single.region_cycles + slack
    two = run_point(
        spec,
        variant,
        size,
        fi,
        fa,
        design=multicore_design(2, root / "designs"),
        build_root=root,
        count_all=True,
    )
    assert two.exit_code == 0, two.console
    assert two.passed, two.error
    if same_order:
        assert _err(two) == _err(single)
    assert two.counters["hart1.instret"] > 0  # the second hart did work


def test_shared_port_is_held_for_its_own_bandwidth():
    from framework.pe.rv.timing.core import Region, RegionTimingSpec

    region = Region("l1", 0, 1 << 20, RegionTimingSpec(latency=6, bandwidth=128))
    assert (
        region.access(0, 1024, False, 100, 16) == 6
    )  # a 64 B/cycle tile load: 16 cycles
    assert region.port_free == 108  # ... but the 128 B/cycle port is free after 8
    region.access(0, 1024, False, 200, 4)
    assert region.port_free == 204  # never longer than the transfer
    region.access(0, 8, False, 300, 1)
    assert region.port_free == 301
    unlimited = Region("l1", 0, 1 << 20, RegionTimingSpec(latency=6, bandwidth=None))
    unlimited.access(0, 1024, False, 100, 16)
    assert unlimited.port_free == 0


# -- gemm ame_mc on 4 harts (moved from framework/naiad/multicore_test.py) --------

DWARFS_4C = multicore_design(
    4
)  # naiad-dwarfs with 4 harts (generated, not a hand copy)


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
