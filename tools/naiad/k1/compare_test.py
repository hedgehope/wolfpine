"""Tests for tools/naiad/k1/compare.py (pure Python; the pack's builds are checked by hand).

~/projects/riscv/venv/bin/python -m pytest tools/naiad/k1/compare_test.py
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "k1_compare", Path(__file__).with_name("compare.py")
)
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

REF = """\
# kernel axpy vlenb 32 clock cycles
R axpy 0 3f800000
R axpy 1 40000000
A dot 0 3f800000 f32
T axpy 100 cycles 1
"""


def _parse(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return compare.parse(p)


def test_identical_outputs_agree(tmp_path):
    a = _parse(tmp_path, "a", REF)
    lines, bad = compare.compare(a, a)
    assert bad == 0
    assert any(line.startswith("OK      axpy (2 exact)") for line in lines)


def test_one_bit_in_an_exact_case_is_a_diff(tmp_path):
    a = _parse(tmp_path, "a", REF)
    b = _parse(tmp_path, "b", REF.replace("R axpy 1 40000000", "R axpy 1 40000001"))
    lines, bad = compare.compare(a, b)
    assert bad == 1
    assert "DIFF    axpy: 1/2 differ; first [1] ref 40000000 other 40000001" in lines


def test_tolerance_cases_allow_rounding_but_not_more(tmp_path):
    a = _parse(tmp_path, "a", REF)
    near = _parse(tmp_path, "b", REF.replace("A dot 0 3f800000", "A dot 0 3f800001"))
    far = _parse(tmp_path, "c", REF.replace("A dot 0 3f800000", "A dot 0 3f900000"))
    assert compare.compare(a, near)[1] == 0
    assert compare.compare(a, far)[1] == 1


def test_missing_cases_failures_and_timings(tmp_path):
    a = _parse(tmp_path, "a", REF)
    b = _parse(
        tmp_path,
        "b",
        REF.replace("R axpy 0 3f800000\nR axpy 1 40000000\n", "")
        + "# FAILED gemm exit 1\n",
    )
    lines, bad = compare.compare(a, b)
    assert bad == 2
    assert any(line.startswith("MISSING axpy") for line in lines)
    assert any(line.startswith("FAILED") for line in lines)
    assert a[2]["axpy"] == (100, "cycles", 1)


def test_nan_handling():
    assert compare.close(float("nan"), float("nan"), 1e-5)
    assert not compare.close(float("nan"), 1.0, 1e-5)
    assert compare.close(float("inf"), float("inf"), 1e-5)
    assert not compare.close(float("inf"), -float("inf"), 1e-5)
