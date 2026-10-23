"""The format registry: representability, RNE ties, and out-of-range behaviour."""

import numpy as np
import pytest

from framework.matrix.formats import FORMATS, format_of_dtype, get_format

ml_dtypes = pytest.importorskip("ml_dtypes")

FLOATS = [name for name, f in FORMATS.items() if f.is_float]
NARROW = ["fp16", "bf16", "fp8_e4m3", "fp8_e5m2"]


def _all_values(name):
    """Every non-NaN value of a <= 16-bit format, enumerated from its bit patterns."""
    f = get_format(name)
    codes = np.arange(1 << f.bits, dtype=np.uint16 if f.bits == 16 else np.uint8)
    with np.errstate(invalid="ignore"):  # the NaN codes
        vals = codes.view(f.dtype).astype(np.float64)
    return vals[~np.isnan(vals)]


@pytest.mark.parametrize("name", NARROW)
def test_every_value_round_trips(name):
    vals = _all_values(name)
    out = get_format(name).round(vals)
    assert out.dtype == get_format(name).dtype
    np.testing.assert_array_equal(out.astype(np.float64), vals)
    np.testing.assert_array_equal(
        get_format(name).round(vals.astype(np.float32)).astype(np.float64), vals
    )


@pytest.mark.parametrize("name", FLOATS)
def test_round_is_idempotent_and_representable(name):
    f = get_format(name)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4096) * np.exp2(rng.integers(-8, 8, 4096))
    once = f.round(x)
    np.testing.assert_array_equal(f.round(once), once)
    # Representable: nothing beyond the format's fraction bits survives
    mant, _ = np.frexp(once.astype(np.float64))
    np.testing.assert_array_equal(np.ldexp(mant, f.mantissa_bits + 1) % 1, 0)


def test_tf32_keeps_ten_fraction_bits_in_fp32():
    x = np.random.default_rng(2).standard_normal(4096).astype(np.float32)
    out = get_format("tf32").round(x)
    assert out.dtype == np.float32
    assert not (out.view(np.uint32) & 0x1FFF).any()
    # Never further than half a tf32 ulp from the input
    assert np.all(np.abs(out - x) <= get_format("tf32").ulp(x) / 2)


@pytest.mark.parametrize(
    "name, x, expected",
    [
        # bf16: 7 fraction bits, ulp at 1 is 2**-7
        ("bf16", 1 + 2**-8, 1.0),  # tie, 1.0 is even
        ("bf16", 1 + 3 * 2**-8, 1 + 2**-6),  # tie, rounds up to even
        ("bf16", 1 + 2**-8 + 2**-30, 1 + 2**-7),  # just past the tie, survives fp32
        (
            "bf16",
            1 + 2**-8 + 2**-40,
            1 + 2**-7,
        ),  # past the tie only in fp64: the sticky bit
        ("bf16", -(1 + 3 * 2**-8), -(1 + 2**-6)),
        # fp16: 10 fraction bits
        ("fp16", 1 + 2**-11, 1.0),
        ("fp16", 1 + 3 * 2**-11, 1 + 2**-9),
        ("fp16", 1 + 2**-11 + 2**-40, 1 + 2**-10),
        ("fp16", 2.0**-25, 0.0),  # half the smallest subnormal, tie to zero
        ("fp16", 3 * 2.0**-25, 2.0**-23),  # 1.5 subnormal ulps -> 2
        # tf32: same fraction width as fp16, fp32's exponent
        ("tf32", 1 + 2**-11, 1.0),
        ("tf32", 1 + 3 * 2**-11, 1 + 2**-9),
        ("tf32", 1 + 2**-11 + 2**-40, 1 + 2**-10),
        (
            "tf32",
            2.0**100 * (1 + 3 * 2**-11),
            2.0**100 * (1 + 2**-9),
        ),  # fp32's exponent range
    ],
)
def test_rne_ties(name, x, expected):
    got = get_format(name).round(np.array([x], dtype=np.float64)).astype(np.float64)[0]
    assert got == expected


@pytest.mark.parametrize("name", ["fp8_e4m3", "fp8_e5m2"])
def test_fp8_matches_brute_force_nearest_even(name):
    vals = np.unique(_all_values(name))
    vals = vals[np.isfinite(vals)]
    codes = get_format(name).round(vals).view(np.uint8)
    rng = np.random.default_rng(3)
    x = rng.uniform(-vals.max(), vals.max(), 2000)
    x = np.concatenate([x, (vals[1:] + vals[:-1]) / 2])  # every midpoint
    dist = np.abs(x[:, None] - vals[None, :])
    best = dist == dist.min(axis=1, keepdims=True)
    # Of the (at most two) nearest, take the even code
    even = (codes & 1) == 0
    pick = np.where(
        best.sum(axis=1) > 1,
        np.argmax(best & even[None, :], axis=1),
        np.argmax(best, axis=1),
    )
    got = get_format(name).round(x).astype(np.float64)
    np.testing.assert_array_equal(got, vals[pick])


def test_fp8_e4m3_overflow_is_nan_or_saturates():
    f = get_format("fp8_e4m3")
    assert f.max == 448
    assert not f.has_inf
    out = f.round(np.array([448.0, 460.0, 464.0, 470.0, np.inf, -1e6])).astype(
        np.float64
    )
    np.testing.assert_array_equal(
        out[:3], [448, 448, 448]
    )  # 464 ties to 448 (the even code)
    assert np.isnan(out[3:]).all()
    sat = f.round(np.array([470.0, np.inf, -1e6, np.nan]), saturate=True).astype(
        np.float64
    )
    np.testing.assert_array_equal(sat[:3], [448, 448, -448])
    assert np.isnan(sat[3])


def test_fp8_e5m2_overflow_is_inf_or_saturates():
    f = get_format("fp8_e5m2")
    assert f.max == 57344
    np.testing.assert_array_equal(
        f.round(np.array([1e5, -1e5])).astype(np.float64), [np.inf, -np.inf]
    )
    np.testing.assert_array_equal(
        f.round(np.array([1e5, -np.inf]), saturate=True).astype(np.float64),
        [57344, -57344],
    )


def test_ints_wrap_or_saturate():
    i8 = get_format("int8")
    np.testing.assert_array_equal(i8.round(np.arange(-128, 128)), np.arange(-128, 128))
    np.testing.assert_array_equal(i8.round([200, -129, 256]), [-56, 127, 0])
    np.testing.assert_array_equal(
        i8.round([2.5, 3.5, -2.5, 300.0, np.nan]), [2, 4, -2, 44, 0]
    )
    np.testing.assert_array_equal(i8.round([200, -129.0], saturate=True), [127, -128])
    assert get_format("int32").round([2**31]).tolist() == [-(2**31)]


def test_registry_lookup():
    assert get_format("bfloat16") is FORMATS["bf16"]
    assert get_format("e4m3") is FORMATS["fp8_e4m3"]
    assert format_of_dtype(np.float32) is FORMATS["fp32"]
    assert format_of_dtype(ml_dtypes.bfloat16) is FORMATS["bf16"]
    with pytest.raises(ValueError, match="unknown format"):
        get_format("fp4")
