"""The AME and IME GEMM emitters at ``timing.scalar.issue_width: 2``, end to end (clang 20).

At width 2 their K loops walk persistent tile-load pointers that the load itself advances
(``mx_tload_*_adv`` / the IME asm's ``add`` after each tile load) instead of rebuilding one per tile per
step. The arithmetic and its order must be untouched (the same error to the last bit), the dual-issue core
must not be slower than it was with the single-issue text, and width 1 must emit exactly what it did.
(The VME emitter's own test is ``gemm_vme_dwarf_test.py``.)
"""

import importlib.util
import shutil

import pytest
import yaml

from experiments.naiad.dwarfs import NAIAD_CC, load_spec, run_point
from framework.naiad import DESIGNS_DIR

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
needs_ml_dtypes = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)


def _designs(tmp_path, base, **timing):
    raw = yaml.safe_load((DESIGNS_DIR / base).read_text())
    raw["timing"]["vector"]["mem_bytes_per_cycle"] = 64
    for section, values in timing.items():
        raw["timing"][section].update(values)
    (tmp_path / "d").mkdir(exist_ok=True)
    out = {}
    for width in (1, 2):
        raw["timing"]["scalar"]["issue_width"] = width
        out[width] = tmp_path / "d" / f"{base}.{width}.yaml"
        out[width].write_text(yaml.safe_dump(raw, sort_keys=False))
    return out


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize(
    "variant, base, timing",
    [
        # a fast unit, so issue slots (not the unit) bound the K loop
        ("ame", "naiad-dwarfs.yaml", {"matrix": {"macs_per_cycle": 2048}}),
        (
            "ime",
            "naiad-ime-timed.yaml",
            {
                "matrix": {
                    "macs_per_cycle": 16384,
                    "vrf_read_bytes_per_cycle": 1024,
                    "vrf_write_bytes_per_cycle": 512,
                }
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "size, pair",
    [((64, 64, 64), ("fp32", "fp32")), ((48, 64, 40), ("bf16", "fp32"))],
    ids=lambda v: "x".join(map(str, v)) if isinstance(v[0], int) else "/".join(v),
)
def test_dual_issue_gemm_is_exact_and_not_slower(
    tmp_path, variant, base, timing, size, pair
):
    designs = _designs(tmp_path, base, **timing)
    spec = load_spec("gemm")
    one = run_point(spec, variant, size, *pair, design=designs[1], build_root=tmp_path)
    two = run_point(
        spec,
        variant,
        size,
        *pair,
        design=designs[2],
        build_root=tmp_path,
        count_all=True,
    )
    assert one.passed, one.error
    assert two.passed, two.error
    assert one.error == two.error  # the same arithmetic in the same order
    assert two.counters["matrix.macs"] == one.counters["matrix.macs"]
    assert two.region_cycles < one.region_cycles
    assert two.counters["hart0.timing.paired"] > 0
