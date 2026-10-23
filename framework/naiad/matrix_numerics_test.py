"""``matrix.numerics`` / ``matrix.fidelity``: the design key, its validation and the mma plumbing (B6)."""

import copy

import numpy as np
import pytest

from framework.matrix.mac import tile_mac
from framework.matrix.unit import IllegalMatrixInstruction, MatrixUnit
from framework.naiad.builder import build
from framework.naiad.design import DesignError, parse_design

BASE = {
    "name": "t",
    "cores": {"xlen": 64, "isa": ["i", "m"], "reset_pc": 0x1000},
    "matrix": {
        "m": 8,
        "n": 16,
        "k": 32,
        "formats": {"in": ["bf16", "tf32"], "acc": ["fp32", "bf16"]},
        "features": ["semiring"],
    },
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


def test_default_is_ieee():
    spec = parse_design(_design()).matrix
    assert (spec.numerics, spec.fidelity) == ("ieee", None)
    assert build(parse_design(_design())).matrix_units[0].numerics == "ieee"


def test_tensix_numerics_reaches_every_mma():
    pytest.importorskip("ml_dtypes")
    soc = build(parse_design(_design(numerics="Tensix_WH", fidelity="HiFi2")))
    unit = soc.matrix_units[0]
    assert (unit.numerics, unit.mac_options) == ("tensix_wh", {"fidelity": "HiFi2"})

    rng = np.random.default_rng(0)
    bf16 = unit.in_formats[unit.decoder.table.formats["bf16"]]
    fp32 = unit.acc_formats[unit.decoder.table.formats["fp32"]]
    A = bf16.round(rng.standard_normal((8, 32)))
    B = bf16.round(rng.standard_normal((32, 16)))
    unit.tile_view(0, "a", bf16)[...] = A
    unit.tile_view(1, "b", bf16)[...] = B
    unit.acc_view(0, fp32)[...] = 1.0
    unit.mma(0, 0, 1, bf16, fp32)
    expected = tile_mac(
        np.ones((8, 16), np.float32),
        A,
        B,
        "bf16",
        "fp32",
        numerics="tensix_wh",
        fidelity="HiFi2",
    )
    assert np.array_equal(unit.acc_view(0, fp32), expected)
    assert not np.array_equal(
        expected, tile_mac(np.ones((8, 16)), A, B, "bf16", "fp32")
    )
    with pytest.raises(IllegalMatrixInstruction, match="plus_times only"):
        unit.mma(0, 0, 1, bf16, fp32, semiring="min_plus")


@pytest.mark.parametrize(
    "change,message",
    [
        ({"numerics": "tensix"}, "matrix.numerics must be one of"),
        ({"fidelity": "HiFi4"}, "fidelity applies only to tensix"),
        ({"numerics": "tensix_bh", "fidelity": "HiFi8"}, "unknown Tensix fidelity"),
        ({"numerics": "tensix_bh", "k_chunk": 8}, "k_chunk is fixed at 16"),
        (
            {
                "numerics": "tensix_bh",
                "formats": {"in": ["bf16", "fp32"], "acc": ["fp32"]},
            },
            "bf16/tf32 inputs bit-exactly, not fp32",
        ),
        (
            {"numerics": "tensix_wh", "formats": {"in": ["bf16"], "acc": ["fp16"]}},
            "32- or 16-bit Dst",
        ),
        (
            {"numerics": "tensix_wh", "formats": {"in": ["int8"], "acc": ["int32"]}},
            "not int8",
        ),
    ],
)
def test_bad_numerics_are_rejected(change, message):
    with pytest.raises(DesignError, match=message):
        parse_design(_design(**change))


def test_unit_rejects_unmodelled_pairings_at_construction():
    kw = {"m": 4, "n": 4, "k": 16, "tiles": 2, "accs": 1}
    with pytest.raises(ValueError, match="not fp16"):
        MatrixUnit(
            **kw, in_formats=["fp16"], acc_formats=["fp32"], numerics="tensix_bh"
        )
    with pytest.raises(ValueError, match="only to tensix"):
        MatrixUnit(**kw, in_formats=["fp32"], acc_formats=["fp32"], fidelity="LoFi")
    with pytest.raises(ValueError, match="unknown matrix numerics"):
        MatrixUnit(**kw, in_formats=["fp32"], acc_formats=["fp32"], numerics="x")
