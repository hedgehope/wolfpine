"""``matrix.profile: ztt``, ``matrix.acc_mem`` and the naiad-ztt preset design."""

import copy

import pytest

from framework.naiad import DESIGNS_DIR
from framework.naiad.design import DesignError, load_design, parse_design
from framework.naiad.gen import gen_header

ZTT = {
    "mode": "ame",
    "profile": "ztt",
    "m": 16,
    "n": 16,
    "k": 16,
    "tiles": 16,
    "accs": 4,
    "formats": {"in": ["fp64", "fp32"], "acc": ["fp64", "fp32"]},
    "features": [
        "transpose",
        "shift",
        "acc_move",
        "convert",
        "mma_transpose",
        "reduce",
    ],
    "acc_mem": False,
    "k_chunk": 1,
}


def _raw(**matrix):
    m = copy.deepcopy(ZTT)
    m.update(matrix)
    return {
        "name": "ztt",
        "cores": {"xlen": 64, "isa": ["i", "m"], "reset_pc": 0x80000000},
        "blocks": [
            {"name": "dram", "type": "dram", "base": 0x80000000, "window": "1MiB"}
        ],
        "matrix": {k: v for k, v in m.items() if v is not None},
    }


def test_a_ztt_unit_parses():
    spec = parse_design(_raw()).matrix
    assert (spec.profile, spec.acc_mem, spec.k_chunk, spec.tiles) == (
        "ztt",
        False,
        1,
        16,
    )


@pytest.mark.parametrize("tiles", [16, 32])
@pytest.mark.parametrize("accs", [1, 2, 4, 8, 16])
def test_every_ztt_register_count_is_accepted(tiles, accs):
    parse_design(_raw(tiles=tiles, accs=accs))


@pytest.mark.parametrize(
    "change, message",
    [
        ({"tiles": 8}, "tiles must be one of \\[16, 32\\]"),
        ({"tiles": 24}, "AME_NUM_M_REGS"),
        ({"accs": 3}, "accs must be one of"),
        ({"m": 8}, "N x N x N"),
        ({"m": 12, "n": 12, "k": 12}, "power of two"),
        ({"acc_mem": None}, "acc_mem must be false"),
        ({"acc_mem": True}, "acc_mem must be false"),
        ({"k_chunk": None}, "k_chunk must be 1"),
        ({"k_chunk": 16}, "k_chunk must be 1"),
        ({"features": ["epilogue"]}, "not in Ztt"),
        ({"features": ["shift2"]}, "not in Ztt"),
        ({"features": ["halo"]}, "not in Ztt"),
        ({"features": ["semiring"]}, "not in Ztt"),
        (
            {"formats": {"in": ["fp32", "int8"], "acc": ["fp32", "int32"]}},
            "would be packed",
        ),
    ],
)
def test_out_of_range_ztt_units_are_rejected(change, message):
    with pytest.raises(DesignError, match=message):
        parse_design(_raw(**change))


def test_without_a_profile_the_ranges_are_not_enforced():
    spec = parse_design(_raw(profile=None, tiles=8, acc_mem=None, k_chunk=None)).matrix
    assert spec.acc_mem is True
    assert spec.profile is None


@pytest.mark.parametrize(
    "change, message",
    [
        ({"profile": "zvt"}, "profile must be one of"),
        ({"acc_mem": "no"}, "true or false"),
        ({"profile": None, "m": 8, "features": ["shift2"]}, "shift2 needs a square"),
        (
            {"profile": None, "m": 8, "n": 4, "k": 16, "features": ["acc_move"]},
            "acc_move needs",
        ),
        (
            {"profile": None, "m": 32, "n": 4, "k": 4, "features": ["mma_transpose"]},
            "mma_transpose needs",
        ),
    ],
)
def test_bad_options_are_rejected(change, message):
    with pytest.raises(DesignError, match=message):
        parse_design(_raw(**change))


def test_the_header_says_whether_acc_mem_is_there():
    on = gen_header(parse_design(_raw(profile=None, acc_mem=True, k_chunk=None)))
    off = gen_header(parse_design(_raw()))
    assert "#define NAIAD_MATRIX_FEATURE_ACC_MEM 1" in on
    assert "NAIAD_MATRIX_FEATURE_ACC_MEM" not in off
    assert "#define NAIAD_MATRIX_PROFILE_ZTT 1" in off
    assert "#define NAIAD_MATRIX_FEATURE_ACC_MOVE 1" in off


def test_the_ztt_preset_is_naiad_dwarfs_with_a_ztt_unit():
    ztt = load_design(DESIGNS_DIR / "naiad-ztt.yaml")
    base = load_design(DESIGNS_DIR / "naiad-dwarfs.yaml")
    assert ztt.matrix.profile == "ztt"
    assert [(b.name, b.base, b.window, b.params) for b in ztt.blocks] == [
        (b.name, b.base, b.window, b.params) for b in base.blocks
    ]
    assert ztt.timing.scalar == base.timing.scalar
