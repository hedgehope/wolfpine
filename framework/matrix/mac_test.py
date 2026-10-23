"""tile_mac and fma_round_once against fp64 references and hand-computed cases."""

from fractions import Fraction

import numpy as np
import pytest

from framework.matrix.formats import FORMATS, get_format
from framework.matrix.mac import fma_round_once, tile_mac

pytest.importorskip("ml_dtypes")

FLOATS = [name for name, f in FORMATS.items() if f.is_float]


def _operands(M=8, K=32, N=8, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(-1, 1, (M, N)),
        rng.uniform(-1, 1, (M, K)),
        rng.uniform(-1, 1, (K, N)),
    )


@pytest.mark.parametrize("acc_fmt", FLOATS)
@pytest.mark.parametrize("in_fmt", FLOATS)
def test_tile_mac_matches_fp64_reference(in_fmt, acc_fmt):
    C, A, B = _operands()
    in_f, acc_f = get_format(in_fmt), get_format(acc_fmt)
    a, b = in_f.round(A).astype(np.float64), in_f.round(B).astype(np.float64)
    c = acc_f.round(C).astype(np.float64)
    ref = c + a @ b

    out = tile_mac(C, A, B, in_fmt, acc_fmt)
    assert out.dtype == acc_f.dtype
    assert out.shape == ref.shape
    np.testing.assert_array_equal(acc_f.round(out), out)

    # Floor-level bound: K + 3 compute roundings on the magnitudes (the +3 covers
    # narrowing fp64 inputs to fp32 and adding C), then one rounding into the
    # accumulator, with its subnormal step as an absolute floor. Doubled.
    K = A.shape[1]
    u_c = 2.0**-53 if acc_fmt == "fp64" else 2.0**-24
    u_acc = acc_f.eps / 2
    mag = np.abs(a) @ np.abs(b) + np.abs(c)
    tol = 2 * (
        (K + 3) * u_c * mag * (1 + u_acc) + u_acc * np.abs(ref) + acc_f.ulp(0.0) / 2
    )
    err = np.abs(out.astype(np.float64) - ref)
    assert np.all(err <= tol), f"max excess {np.max(err - tol)}"


def test_fp32_accumulation_beats_bf16_on_long_k():
    rng = np.random.default_rng(4)
    A, B = rng.uniform(0, 1, (4, 4096)), rng.uniform(0, 1, (4096, 4))
    bf = get_format("bf16")
    golden = bf.round(A).astype(np.float64) @ bf.round(B).astype(np.float64)

    def err(**kw):
        return np.max(
            np.abs(tile_mac(None, A, B, "bf16", **kw).astype(np.float64) - golden)
            / golden
        )

    fp32_acc, bf16_acc = err(acc_fmt="fp32"), err(acc_fmt="bf16")
    fp32_chunked, bf16_chunked = (
        err(acc_fmt="fp32", k_chunk=16),
        err(acc_fmt="bf16", k_chunk=16),
    )
    # One final bf16 rounding is ~half a bf16 ulp; fp32 accumulation is far below it
    assert fp32_acc < 1e-5 < bf16_acc <= 2.0**-8
    assert fp32_chunked < 1e-5
    # Rounding into bf16 after every 16-deep chunk stagnates: much worse again
    assert bf16_chunked > 5 * bf16_acc


def test_k_chunk_rounds_the_accumulator_per_chunk():
    # bf16 at 1.0 has an ulp of 2**-7, so 1 + 2**-9 rounds straight back to 1
    A = np.array([[1.0, 2.0**-9, 2.0**-9, 2.0**-9, 2.0**-9]])
    B = np.ones((5, 1))
    assert tile_mac(None, A, B, "bf16", "bf16").astype(np.float64)[0, 0] == 1 + 2.0**-7
    assert (
        tile_mac(None, A, B, "bf16", "bf16", k_chunk=1).astype(np.float64)[0, 0] == 1.0
    )
    # A chunk of two carries 2**-8: a tie, so 1.0 (even) again; chunks of 4 + 1 keep it
    assert (
        tile_mac(None, A, B, "bf16", "bf16", k_chunk=2).astype(np.float64)[0, 0] == 1.0
    )
    assert (
        tile_mac(None, A, B, "bf16", "bf16", k_chunk=4).astype(np.float64)[0, 0]
        == 1 + 2.0**-7
    )


def test_k_chunk_is_deterministic_and_differs_at_low_precision():
    C, A, B = _operands(16, 256, 16, seed=5)
    first = tile_mac(C, A, B, "bf16", "bf16", k_chunk=8)
    again = tile_mac(C.copy(), A.copy(), B.copy(), "bf16", "bf16", k_chunk=8)
    np.testing.assert_array_equal(first.view(np.uint16), again.view(np.uint16))
    assert (first != tile_mac(C, A, B, "bf16", "bf16")).any()
    # At fp64 accumulation the chunking is invisible to fp64-level tolerance
    np.testing.assert_allclose(
        tile_mac(C, A, B, "fp64", "fp64", k_chunk=8),
        tile_mac(C, A, B, "fp64", "fp64"),
        rtol=1e-13,
    )


def test_k_chunk_does_not_mutate_inputs():
    C, A, B = _operands()
    saved = C.copy(), A.copy(), B.copy()
    tile_mac(C, A, B, "fp16", "fp16", k_chunk=4)
    for got, want in zip((C, A, B), saved):
        np.testing.assert_array_equal(got, want)


def test_int8_matches_reference_and_wraps():
    rng = np.random.default_rng(6)
    A = rng.integers(-128, 128, (8, 64))
    B = rng.integers(-128, 128, (64, 8))
    C = rng.integers(-1000, 1000, (8, 8))
    out = tile_mac(C, A, B, "int8", "int32")
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, C + A @ B)
    np.testing.assert_array_equal(tile_mac(C, A, B, "int8", "int32", k_chunk=5), out)

    # Accumulator wraparound
    assert tile_mac([[2**31 - 1]], [[1]], [[1]], "int8", "int32")[0, 0] == -(2**31)
    # A long dot product of 127 * 127 terms past int32's range
    K = 140_000
    got = tile_mac(None, np.full((1, K), 127), np.full((K, 1), 127), "int8", "int32")[
        0, 0
    ]
    assert got == 127 * 127 * K - 2**32
    # Inputs outside int8 wrap on the way in
    assert tile_mac(None, [[200]], [[1]], "int8", "int32")[0, 0] == -56


def test_validation():
    C, A, B = _operands()
    with pytest.raises(ValueError, match="inner dimensions"):
        tile_mac(None, A, B[:-1], "fp32", "fp32")
    with pytest.raises(ValueError, match="C must be"):
        tile_mac(C[:, :-1], A, B, "fp32", "fp32")
    with pytest.raises(ValueError, match="2-D"):
        tile_mac(None, A[0], B, "fp32", "fp32")
    with pytest.raises(ValueError, match="mix float and int"):
        tile_mac(None, A, B, "int8", "fp32")
    with pytest.raises(ValueError, match="semiring"):
        tile_mac(None, A, B, "fp32", "fp32", semiring="or_and")
    with pytest.raises(ValueError, match="k_chunk"):
        tile_mac(None, A, B, "fp32", "fp32", k_chunk=0)


def test_fma_rounds_once_where_fp64_and_unfused_do_not():
    # (1 + 2**-12)**2 = 1 + 2**-11 + 2**-24: exactly halfway between two fp32s.
    # The 2**-60 addend breaks the tie upward, but fp64 alone loses it.
    a = b = 1 + 2.0**-12
    c = 2.0**-60
    want = np.float32(1 + 2.0**-11 + 2.0**-23)
    assert fma_round_once(a, b, c, "fp32") == want
    f32 = get_format("fp32")
    unfused = f32.round(f32.round(np.float64(a) * b).astype(np.float64) + c)
    naive_fp64 = f32.round(np.float64(a) * b + c)
    assert unfused == naive_fp64 == np.float32(1 + 2.0**-11)


@pytest.mark.parametrize("name", ["bf16", "fp16"])
def test_fma_is_correctly_rounded(name):
    f = get_format(name)
    codes = np.arange(1 << 16, dtype=np.uint16)
    with np.errstate(invalid="ignore"):
        allvals = codes.view(f.dtype).astype(np.float64)
    finite = np.isfinite(allvals)
    order = np.argsort(allvals[finite], kind="stable")
    vals, vcodes = allvals[finite][order], codes[finite][order]

    rng = np.random.default_rng(7)
    n = 300
    a = f.round(rng.uniform(-2, 2, n)).astype(np.float64)
    b = f.round(rng.uniform(-2, 2, n)).astype(np.float64)
    c = f.round(rng.uniform(-2, 2, n) * np.exp2(rng.integers(-20, 4, n))).astype(
        np.float64
    )
    got = fma_round_once(a, b, c, name).astype(np.float64)
    for ai, bi, ci, gi in zip(a, b, c, got):
        exact = Fraction(ai) * Fraction(bi) + Fraction(ci)
        hi = np.searchsorted(vals, float(exact))
        cands = [i for i in (hi - 1, hi, hi + 1) if 0 <= i < len(vals)]
        dists = [abs(Fraction(vals[i]) - exact) for i in cands]
        best = [i for i, d in zip(cands, dists) if d == min(dists)]
        if len(best) > 1:
            best = [i for i in best if vcodes[i] % 2 == 0] or best
        assert gi == vals[best[0]] or (gi == 0 and vals[best[0]] == 0)


def test_fma_rejects_wide_and_int_formats():
    with pytest.raises(ValueError, match="fp32 and narrower"):
        fma_round_once(1.0, 1.0, 1.0, "fp64")
    with pytest.raises(ValueError, match="fp32 and narrower"):
        fma_round_once(1, 1, 1, "int8")
