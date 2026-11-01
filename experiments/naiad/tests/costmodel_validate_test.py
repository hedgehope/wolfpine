"""``costmodel_validate``: a sweep over the front end (``timing.scalar.issue_width``) must not pair a baseline
timed at one width with a matrix variant timed at the other."""

import yaml

from experiments.naiad import costmodel_validate as cv
from framework.naiad import DESIGNS_DIR

DESIGN = DESIGNS_DIR / "naiad-dwarfs.yaml"


def _row(variant, width, sim, pred, mv_ratio=None):
    return {
        "source": "s",
        "study": "st",
        "dwarf": "gemm",
        "variant": variant,
        "size": 64,
        "in_fmt": "fp32",
        "acc_fmt": "fp32",
        "cflags": (),
        "design": str(DESIGN),
        "design_hash": f"d{width}" if variant == "ame" else "",
        "mv_ratio": mv_ratio,
        "sim": sim,
        "pred": pred,
        "rel_err": pred / sim - 1,
        "issue_width": width,
        "point_id": f"{variant}{width}",
    }


def test_decisions_compare_variants_on_the_same_front_end():
    # at width 1 the matrix variant wins, at width 2 the (faster) baseline does; a baseline joined across
    # widths would call the width-1 decision wrong
    rows = [
        _row("ame", 1, 900, 900),
        _row("rvv", 1, 1000, 1000),
        _row("ame", 2, 900, 900),
        _row("rvv", 2, 800, 800),
    ]
    dec = cv.decisions(rows)
    assert len(dec) == 2
    assert all(d["correct"] for d in dec)
    assert {d["issue_width"]: d["best_sim"] for d in dec} == {1: "ame", 2: "rvv"}


def test_issue_width_is_read_from_the_stored_design(tmp_path):
    cv._issue_width.cache_clear()
    raw = yaml.safe_load(DESIGN.read_text())
    raw["timing"]["scalar"]["issue_width"] = 2
    path = tmp_path / "dual.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert cv._issue_width(str(path)) == 2
    assert cv._issue_width(str(DESIGN)) == 1
