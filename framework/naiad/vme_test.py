"""VME on Naiad: the ``matrix: {mode: vme}`` schema, the generator, the builder wiring and counters.
The gemm dwarf's ``vme`` variant end to end (clang 20) is ``experiments/naiad/tests/vme_dwarf_test.py``.
Spike has no Zvt, so nothing here is difftested."""

import importlib.util

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

VME = DESIGNS_DIR / "naiad-vme.yaml"
VME_TIMED = DESIGNS_DIR / "naiad-vme-timed.yaml"


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
        "matrix": {"mode": "vme", **matrix},
    }


def test_defaults():
    spec = parse_design(_raw({})).matrix
    assert (spec.mode, spec.vlen, spec.te) == ("vme", 256, 32)
    assert spec.extensions == ("Zvtbase", "Zvtfmm")
    assert spec.tews == (32,)
    assert spec.format_pairs == (("fp32", "fp32"),)
    assert parse_design(_raw({}, vlen=128)).matrix.te == 32


def test_shipped_designs():
    for path in (VME, VME_TIMED):
        d = load_design(path)
        assert d.matrix.te == 32
        assert d.matrix.tews == (8, 16, 32, 64)
        assert ("bf16", "fp32") in d.matrix.format_pairs
        assert (
            "int8",
            "int32",
        ) in d.matrix.format_pairs
    t = load_design(VME_TIMED).timing.matrix
    assert (
        t.macs_per_cycle,
        t.vrf_read_bytes_per_cycle,
        t.vrf_write_bytes_per_cycle,
        t.tile_bytes_per_cycle,
    ) == (
        2048,
        96,
        32,
        64,
    )


@pytest.mark.parametrize(
    "matrix,isa,message",
    [
        ({}, ("i", "m", "f", "d", "zicsr"), "vector unit"),
        ({"te": 24}, None, "power of two"),
        ({"te": 128}, None, "VLEN/4 = 64"),
        ({"te": 2}, None, "power of two"),
        ({"extensions": ["Zvtfoo"]}, None, "not a Zvt v0.4 extension"),
        ({"extensions": ["Zvtbf16fmm"]}, None, r"requires \['Zvt16e'\]"),
        ({"extensions": ["Zvtdmm"]}, None, r"requires \['Zvt64e'\]"),
        ({"extensions": ["Zvt16e"]}, None, "no computational extension"),
        ({"extensions": ["Zvtfmm", "zvtfmm"]}, None, "twice"),
        ({"nonsense": 1}, None, "unknown key"),
    ],
)
def test_schema_errors(matrix, isa, message):
    raw = _raw(matrix) if isa is None else _raw(matrix, isa)
    with pytest.raises(DesignError, match=message):
        parse_design(raw)


def test_ime_and_vme_are_refused_together_in_both_directions():
    """One design has one matrix section; a second mode could only come in through cores.isa."""
    from framework.matrix.vme import parse_vme

    cores = CoreSpec(xlen=64, isa=("i", "f", "d", "v", "zvvfp32mm"), vlen=256, elen=64)
    with pytest.raises(DesignError, match="cannot share a hart with IME.*vtle/vmtl.v"):
        parse_vme({"mode": "vme"}, cores, DesignError)
    cores = CoreSpec(xlen=64, isa=("i", "f", "d", "v", "zvtbase"), vlen=256, elen=64)
    with pytest.raises(DesignError, match="same LOAD-FP/STORE-FP encodings"):
        _parse_ime_matrix({"mode": "ime"}, cores)


def test_extension_names_are_case_insensitive():
    d = parse_design(_raw({"extensions": ["zvt16E", "ZVTBF16FMM"]}))
    assert d.matrix.extensions == ("Zvtbase", "Zvt16e", "Zvtbf16fmm")


def test_header_has_the_encodings_and_te():
    h = gen_header(load_design(VME))
    assert "#define NAIAD_HAS_VME 1" in h
    assert "#define NAIAD_MATRIX_VME 1" in h
    assert "NAIAD_HAS_MATRIX" not in h
    assert "NAIAD_HAS_IME" not in h
    assert "#define NAIAD_VME_TE 32" in h
    assert "#define NAIAD_VME_VTFMM_TVV_MATCH 0xf2001077" in h
    assert "#define NAIAD_VME_EXT_ZVTBF16FMM 1" in h


def test_builder_attaches_vme_after_rvv_with_counters_and_mtype():
    from framework.matrix.vme_isa import RV64_VME_ISA, VMEUnit

    soc = build(load_design(VME_TIMED), count_all=True)
    core = soc.cores[0]
    assert isinstance(core.matrix, VMEUnit)
    assert core.matrix.vs is core.vector
    assert core.isas.index(RV64_VME_ISA) > [i.__name__ for i in core.isas].index(
        "RV64_V_ISA"
    )
    assert core.csrs.read(0xC23) == 0
    c = soc.counters()
    for key in (
        "mma",
        "macs",
        "vrf_a_read_bytes",
        "vec_to_tile_bytes",
        "tile_to_vec_bytes",
    ):
        assert f"hart0.matrix.{key}" in c
        assert f"matrix.{key}" in c
    assert "hart0.instret.matrix" in c
    soc.reset()
    assert core.csrs.read(0xC23) == 0
    assert not core.matrix.buf.any()
