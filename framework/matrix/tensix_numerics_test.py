"""``tile_mac(numerics="tensix_*")`` against the Tensix matrix unit itself (Naiad B6).

The reference here is not the datapath functions ``tensix_numerics`` calls but
the whole Tensix backend: a ``TensixCoProcessor`` per architecture, SrcA/SrcB
written in their 19-bit storage layout, Dst seeded in its own layout, the
format/Dst-width/fidelity registers set, and real ``MVMUL`` instruction words
issued and clocked through the matrix unit -- then Dst read back. The tiling
into MVMULs (8 Dst rows x 16 columns x 16 lanes, the fidelity sweep per 32-wide
K tile) is written out here independently of :func:`mvmul_schedule`, so a slip
in the decomposition, the operand roles or a conversion would show up as a
mismatch, not cancel out.

Runs standalone (``python3 -m framework.matrix.tensix_numerics_test``) or under
pytest.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from framework.matrix.mac import NUMERICS, register_numerics, tile_mac
from framework.matrix.tensix_numerics import (
    check_formats,
    fidelity_phases,
    from_fp32_bits,
    mvmul_schedule,
    tensix_mac_bits,
    to_fp32_bits,
)

MVMUL = 0x26
FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")


# -- the hardware-model harness ----------------------------------------------


@contextmanager
def _backend(blackhole):
    """A Tensix backend; restores the process-global Wormhole config layout after."""
    from framework.arch.blackhole import BLACKHOLE_PROFILE
    from framework.pe.tensix.tensix import TensixCoProcessor
    from framework.pe.tensix.util import TensixConfigurationConstants

    try:
        yield TensixCoProcessor(
            None,
            BLACKHOLE_PROFILE.tensix_cfg_state_size if blackhole else None,
            BLACKHOLE_PROFILE.tensix_thd_state_size if blackhole else None,
            blackhole=blackhole,
        ).getBackend()
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _set_config(backend, key, value, state_id=0):
    from framework.pe.tensix.util import TensixConfigurationConstants as TCC

    addr32, shamt, mask = TCC.get_addr32(key), TCC.get_shamt(key), TCC.get_mask(key)
    old = backend.config_unit.get_config_entry(state_id, addr32)
    backend.config_unit.setConfig(
        state_id, addr32, (old & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)
    )


def _src_word(bits, tf32):
    """An fp32 pattern (already bf16/tf32-representable) in Src's Sign,Man(10),Exp(8) layout."""
    from framework.pe.tensix.util import DataFormatConversions as DFC

    if tf32:
        return DFC.TF32ToSrcTF32(int(bits) >> 13)
    return DFC.BF16ToSrcBF16(int(bits) >> 16)


def hardware_mac(C, A, B, *, tf32, dst32, fidelity, blackhole):
    """``C + A @ B`` (fp32 patterns in, fp32 patterns out) run as MVMULs on a real backend."""
    from framework.pe.tensix.backends.backend_base import DataFormat
    from framework.pe.tensix.util import DataFormatConversions as DFC

    M, K = A.shape
    N = B.shape[1]
    kpad = -(-K // 16) * 16
    Ap = np.zeros((M, kpad), np.int64)
    Ap[:, :K] = A
    Bp = np.zeros((kpad, N), np.int64)
    Bp[:K] = B
    phases = fidelity_phases(fidelity)
    out = np.zeros((M, N), np.int64)

    with _backend(blackhole) as backend:
        mu = backend.matrix_unit
        fmt = DataFormat.TF32 if tf32 else DataFormat.BF16
        _set_config(backend, "ALU_FORMAT_SPEC_REG0_SrcA", int(fmt))
        _set_config(backend, "ALU_ACC_CTRL_Fp32_enabled", int(dst32))
        # Blackhole implies the SrcA style from the format the bank was unpacked in
        backend.getSrcA(mu.srcABank).setDataFormat(fmt)
        dst = backend.getDst()
        rwc = backend.getRWC(0)

        for i0 in range(0, M, 8):
            for j0 in range(0, N, 16):
                rows, cols = min(8, M - i0), min(16, N - j0)
                for r in range(8):
                    for c in range(16):
                        v = int(C[i0 + r, j0 + c]) if r < rows and c < cols else 0
                        if dst32:
                            dst.setDst32b(r, c, DFC.FP32ToDstFormatFP32(v))
                        else:
                            dst.setDst16b(r, c, DFC.BF16ToDstFormatBF16(v >> 16))
                # tt-llk's order: per 32-wide K tile, every fidelity phase over
                # both of its 16-lane MVMULs.
                for tile in range(0, kpad, 32):
                    for phase in range(phases):
                        for k0 in range(tile, min(tile + 32, kpad), 16):
                            srcA = backend.getSrcA(mu.srcABank)
                            srcB = backend.getSrcB(mu.srcBBank)
                            for k in range(16):
                                for c in range(16):
                                    a = Bp[k0 + k, j0 + c] if c < cols else 0
                                    srcA[k, c] = _src_word(a, tf32)  # SrcA[lane][col]
                                for r in range(8):
                                    b = Ap[i0 + r, k0 + k] if r < rows else 0
                                    srcB[r, k] = _src_word(b, tf32)  # SrcB[row][lane]
                            rwc.FidelityPhase = phase
                            assert backend.issueInstruction(MVMUL << 24, 0)
                            mu.clock_tick(0)
                for r in range(rows):
                    for c in range(cols):
                        if dst32:
                            v = DFC.FP32InDstToFP32(dst.getDst32b(r, c))
                        else:
                            v = DFC.BF16InDstToBF16(dst.getDst16b(r, c)) << 16
                        out[i0 + r, j0 + c] = v
    return out


# -- operand distributions -----------------------------------------------------


def _pattern(rng, shape, exp_lo, exp_hi, zero_frac=0.0):
    """Random fp32 patterns: random sign and mantissa, exponents in [exp_lo, exp_hi]."""
    sign = rng.integers(0, 2, shape, dtype=np.int64) << 31
    exp = rng.integers(exp_lo, exp_hi + 1, shape, dtype=np.int64)
    if zero_frac:
        exp = np.where(rng.random(shape) < zero_frac, 0, exp)
    return sign | (exp << 23) | rng.integers(0, 1 << 23, shape, dtype=np.int64)


def _operands(kind, rng, M, K, N, in_fmt, acc_fmt):
    """``(C, A, B)`` as patterns representable in their formats, for one distribution."""
    if kind == "mixed":
        A = _pattern(rng, (M, K), 107, 147)
        B = _pattern(rng, (K, N), 107, 147)
        C = _pattern(rng, (M, N), 117, 137)
    elif kind == "cancel":
        # Second half of K negates the first, perturbed in the last mantissa bits:
        # every sum lands near zero and the alignment rounding decides it.
        h = K // 2
        A = _pattern(rng, (M, K), 120, 134)
        A[:, h:] = (A[:, :h] ^ (1 << 31)) ^ (rng.integers(0, 4, (M, h)) << 16)
        B = _pattern(rng, (K, N), 120, 134)
        B[h:] = B[:h]
        C = _pattern(rng, (M, N), 100, 110)
    elif kind == "denormal":
        # Denormal (exponent-0) operands and C, and exponents small enough that
        # products and whole groups underflow.
        A = _pattern(rng, (M, K), 1, 70, zero_frac=0.3)
        B = _pattern(rng, (K, N), 1, 70, zero_frac=0.3)
        C = _pattern(rng, (M, N), 0, 3)
    elif kind == "large":
        # Products past the top of the range, which saturate (no infinities)
        A = _pattern(rng, (M, K), 180, 254)
        B = _pattern(rng, (K, N), 150, 254)
        C = _pattern(rng, (M, N), 250, 254)
    else:
        raise ValueError(kind)
    as_float = lambda x: x.astype(np.uint32).view(np.float32)  # noqa: E731
    return (
        to_fp32_bits(as_float(C), acc_fmt),
        to_fp32_bits(as_float(A), in_fmt),
        to_fp32_bits(as_float(B), in_fmt),
    )


def _check(kind, arch, in_fmt, acc_fmt, fidelity, K, M=8, N=16, seed=0):
    rng = np.random.default_rng(
        [seed, K, FIDELITIES.index(fidelity), len(kind), arch == "bh"]
    )
    C, A, B = _operands(kind, rng, M, K, N, in_fmt, acc_fmt)
    expected = hardware_mac(
        C,
        A,
        B,
        tf32=in_fmt == "tf32",
        dst32=acc_fmt == "fp32",
        fidelity=fidelity,
        blackhole=arch == "bh",
    )
    got = tensix_mac_bits(
        C, A, B, dst32=acc_fmt == "fp32", fidelity=fidelity, wormhole=arch == "wh"
    )
    bad = np.argwhere(got != expected)
    assert not bad.size, (
        f"{len(bad)}/{got.size} differ, first at {tuple(bad[0])}: "
        f"{int(got[tuple(bad[0])]):#010x} vs hardware {int(expected[tuple(bad[0])]):#010x}"
    )
    return expected


# -- bit-exactness against the backend -----------------------------------------


@pytest.mark.parametrize("arch", ["bh", "wh"])
@pytest.mark.parametrize("acc_fmt", ["fp32", "bf16"])
@pytest.mark.parametrize("fidelity", FIDELITIES)
@pytest.mark.parametrize("K", [16, 32, 64])
def test_bf16_matches_tensix_backend(arch, acc_fmt, fidelity, K):
    for kind in ("mixed", "cancel", "denormal", "large"):
        _check(kind, arch, "bf16", acc_fmt, fidelity, K)


@pytest.mark.parametrize("arch", ["bh", "wh"])
@pytest.mark.parametrize("acc_fmt", ["fp32", "bf16"])
@pytest.mark.parametrize("fidelity", ["LoFi", "HiFi4"])
def test_tf32_matches_tensix_backend(arch, acc_fmt, fidelity):
    for kind in ("mixed", "cancel"):
        _check(kind, arch, "tf32", acc_fmt, fidelity, 32)


def test_multi_mvmul_footprint_and_ragged_shapes_match():
    """M past 8 rows and N past 16 columns (several Dst blocks), K not a multiple of 16."""
    for arch in ("bh", "wh"):
        _check("mixed", arch, "bf16", "fp32", "HiFi2", 40, M=19, N=33, seed=3)
        _check("cancel", arch, "bf16", "bf16", "HiFi4", 48, M=10, N=20, seed=4)


def test_tile_mac_entry_point_matches_backend():
    """The public path (float arrays in, acc dtype out) lands on the same bits."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((8, 64)).astype(np.float32)
    B = rng.standard_normal((64, 16)).astype(np.float32)
    C = rng.standard_normal((8, 16)).astype(np.float32)
    a, b, c = to_fp32_bits(A, "bf16"), to_fp32_bits(B, "bf16"), to_fp32_bits(C, "fp32")
    for numerics, blackhole in (("tensix_bh", True), ("tensix_wh", False)):
        for fidelity in ("LoFi", "HiFi4"):
            got = tile_mac(
                C, A, B, "bf16", "fp32", numerics=numerics, fidelity=fidelity
            )
            assert got.dtype == np.float32
            expected = hardware_mac(
                c, a, b, tf32=False, dst32=True, fidelity=fidelity, blackhole=blackhole
            )
            assert np.array_equal(got.view(np.uint32), expected.astype(np.uint32))


def test_wormhole_neg_one_renormalisation_differs_from_blackhole():
    """The one arch difference, reached on purpose: C = -(1 + 2**-23), A @ B = +1.

    The accumulate leaves -1 ulp at exponent 127, which Wormhole renormalises
    27 binades too low. Both arches' backends agree with their tile_mac mode.
    """
    C = np.array([[0xBF800001]], np.int64)
    A = np.zeros((1, 16), np.int64)
    A[0, 0] = 0x3F800000
    B = np.zeros((16, 1), np.int64)
    B[0, 0] = 0x3F800000
    results = {}
    for arch in ("bh", "wh"):
        expected = hardware_mac(
            C, A, B, tf32=False, dst32=True, fidelity="LoFi", blackhole=arch == "bh"
        )
        got = tensix_mac_bits(C, A, B, fidelity="LoFi", wormhole=arch == "wh")
        assert np.array_equal(got, expected)
        results[arch] = int(got[0, 0])
    assert results["bh"] == 0xB4000000  # -2**-23, the right answer
    assert ((results["wh"] >> 23) & 0xFF) == ((results["bh"] >> 23) & 0xFF) + 27


# -- against ttsim, through a real tt-metal matmul ------------------------------
#
# ``optests/matmulrand`` runs one 32x64 @ 64x32 ``matmul_tiles`` K loop on random
# bf16 operands (xorshift, ported below) at every fidelity and Dst width; these
# are ttsim's packed output tiles, captured with ``optests/diff.sh matmulrand``
# (where Wolfpine matched them too). This pins the whole decomposition --
# operand roles, tt-llk's phase order, the packer's narrowing of a 32-bit Dst
# -- to the vendor reference, not just to our own backend.

TTSIM_GOLDEN = Path(__file__).resolve().parent / "testdata" / "matmulrand_ttsim.npz"


def _matmulrand_operands():
    s, words = 0x2545F491, []
    for _ in range(4 * 1024):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        words.append(((s & 1) << 15) | ((118 + (s >> 1) % 19) << 7) | ((s >> 8) & 0x7F))
    tiles = np.array(words, np.int64).reshape(4, 1024)
    A = np.hstack([_detilize(tiles[0]), _detilize(tiles[1])]) << 16
    B = np.vstack([_detilize(tiles[2]), _detilize(tiles[3])]) << 16
    return A, B


def _detilize(words):
    return np.asarray(words).reshape(2, 2, 16, 16).transpose(0, 2, 1, 3).reshape(32, 32)


@pytest.mark.parametrize("arch", ["bh", "wh"])
@pytest.mark.parametrize("fidelity", FIDELITIES)
@pytest.mark.parametrize("acc_fmt", ["bf16", "fp32"])
def test_matches_ttsim_matmul_tiles(arch, fidelity, acc_fmt):
    from framework.pe.tensix.util import DataFormatConversions as DFC

    golden = _detilize(np.load(TTSIM_GOLDEN)[f"{arch}_{fidelity.lower()}_{acc_fmt}"])
    A, B = _matmulrand_operands()
    dst = tensix_mac_bits(
        np.zeros((32, 32), np.int64),
        A,
        B,
        dst32=acc_fmt == "fp32",
        fidelity=fidelity,
        wormhole=arch == "wh",
    )
    if acc_fmt == "fp32":
        # The packer narrows the 32-bit Dst to the bf16 output CB
        packed = [
            DFC.FP32InDstToBF16(DFC.FP32ToDstFormatFP32(int(v))) for v in dst.flat
        ]
        dst = np.array(packed, np.int64).reshape(32, 32)
    else:
        dst = dst >> 16
    assert np.array_equal(dst, golden.astype(np.int64))


# -- the decomposition and the conversions ---------------------------------------


def test_schedule_is_tt_llk_order():
    """Phases sweep each 32-wide K tile (tt-llk's MOP inner loop), tiles in K order."""
    assert list(mvmul_schedule(64, "HiFi2")) == [
        (0, 0), (16, 0), (0, 1), (16, 1),
        (32, 0), (48, 0), (32, 1), (48, 1),
    ]  # fmt: skip
    assert list(mvmul_schedule(40, "LoFi")) == [(0, 0), (16, 0), (32, 0)]
    assert list(mvmul_schedule(32, "HiFi3", phase_k=16)) == [
        (0, 0), (0, 1), (0, 2), (16, 0), (16, 1), (16, 2),
    ]  # fmt: skip
    with pytest.raises(ValueError, match="phase_k"):
        list(mvmul_schedule(32, "LoFi", phase_k=24))
    with pytest.raises(ValueError, match="fidelity"):
        fidelity_phases("HiFi5")


def test_phase_k_changes_results_only_past_one_mvmul():
    rng = np.random.default_rng(11)
    A = rng.standard_normal((8, 64)).astype(np.float32)
    B = rng.standard_normal((64, 16)).astype(np.float32)
    run = lambda **kw: tile_mac(  # noqa: E731
        None, A, B, "bf16", "bf16", numerics="tensix_bh", fidelity="HiFi4", **kw
    )
    assert not np.array_equal(
        run(phase_k=16).view(np.uint16), run(phase_k=32).view(np.uint16)
    )
    A16, B16 = A[:, :16], B[:16]
    one = [
        tile_mac(None, A16, B16, "bf16", "fp32", numerics="tensix_bh", phase_k=p)
        for p in (16, 32, 64)
    ]
    assert all(np.array_equal(one[0], o) for o in one[1:])


def test_rounding_to_src_formats_matches_ieee_path():
    """Operand rounding is Format.round's (RNE, round-to-odd from fp64), on bits."""
    pytest.importorskip("ml_dtypes")
    from framework.matrix.formats import get_format

    rng = np.random.default_rng(5)
    x64 = rng.standard_normal(20000) * 2.0 ** rng.integers(-60, 60, 20000)
    x64[:4] = [1 + 2**-8 + 2**-40, -(1 + 2**-8), 3.0e38, np.inf]
    for fmt in ("bf16", "tf32", "fp32"):
        f = get_format(fmt)
        for x in (x64, x64.astype(np.float32)):
            ref = np.asarray(f.round(x)).astype(np.float32).view(np.uint32)
            assert np.array_equal(to_fp32_bits(x, fmt), ref.astype(np.int64)), fmt
    bf = get_format("bf16").round(x64)
    assert np.array_equal(
        from_fp32_bits(to_fp32_bits(bf, "bf16"), "bf16").view(np.uint16),
        bf.view(np.uint16),
    )


def test_fidelity_orders_the_error_and_none_is_ieee():
    """More phases, less error -- but even HiFi4 is not IEEE.

    HiFi4 forms every bf16 product exactly, yet each phase's group sum is
    aligned to the group's largest product at that product's ~12-bit
    resolution, so the error is ~2**-12 of ``sum |a b|`` -- four thousand times
    the IEEE path's. HiFi3 and HiFi4 sit together (phase 3 is the two low
    slices' product, which is tiny), so only the coarse order is asserted.
    """
    rng = np.random.default_rng(1)
    A = rng.standard_normal((16, 64))
    B = rng.standard_normal((64, 16))
    fa = to_fp32_bits(A, "bf16").astype(np.uint32).view(np.float32).astype(np.float64)
    fb = to_fp32_bits(B, "bf16").astype(np.uint32).view(np.float32).astype(np.float64)
    exact, scale = fa @ fb, np.abs(fa) @ np.abs(fb)

    def err(**kw):
        got = tile_mac(None, A, B, "bf16", "fp32", **kw).astype(np.float64)
        return np.max(np.abs(got - exact) / scale)

    tensix = {f: err(numerics="tensix_bh", fidelity=f) for f in FIDELITIES}
    assert tensix["LoFi"] > 4 * tensix["HiFi2"] > 4 * tensix["HiFi4"]
    assert 2**-16 < tensix["HiFi4"] < 2**-10
    assert err() < 2**-20


# -- API -------------------------------------------------------------------------


def test_ieee_default_is_untouched_and_rejects_tensix_options():
    rng = np.random.default_rng(2)
    A, B = rng.standard_normal((4, 32)), rng.standard_normal((32, 4))
    assert np.array_equal(
        tile_mac(None, A, B, "fp32", "fp32"),
        tile_mac(None, A, B, "fp32", "fp32", numerics="ieee"),
    )
    with pytest.raises(ValueError, match="no options"):
        tile_mac(None, A, B, "fp32", "fp32", fidelity="LoFi")
    with pytest.raises(ValueError, match="unknown numerics"):
        tile_mac(None, A, B, "fp32", "fp32", numerics="tensix")


@pytest.mark.parametrize(
    "in_fmt,acc_fmt,match",
    [
        ("fp16", "fp32", "bf16/tf32 inputs"),
        ("fp32", "fp32", "bf16/tf32 inputs"),
        ("fp8_e4m3", "fp32", "bf16/tf32 inputs"),
        ("bf16", "fp16", "32- or 16-bit Dst"),
        ("bf16", "fp64", "32- or 16-bit Dst"),
        ("int8", "int32", "bf16/tf32 inputs"),
    ],
)
def test_tensix_rejects_unmodelled_formats(in_fmt, acc_fmt, match):
    A, B = np.ones((2, 16)), np.ones((16, 2))
    with pytest.raises(ValueError, match=match):
        tile_mac(None, A, B, in_fmt, acc_fmt, numerics="tensix_bh")
    with pytest.raises(ValueError, match=match):
        check_formats("tensix_wh", in_fmt, acc_fmt)


def test_tensix_rejects_other_k_chunks_and_semirings():
    A, B = np.ones((2, 32), np.float32), np.ones((32, 2), np.float32)
    ok = tile_mac(None, A, B, "bf16", "fp32", numerics="tensix_bh", k_chunk=16)
    assert np.array_equal(ok, np.full((2, 2), 32.0, np.float32))
    with pytest.raises(ValueError, match="MVMULs of 16"):
        tile_mac(None, A, B, "bf16", "fp32", numerics="tensix_bh", k_chunk=8)
    with pytest.raises(ValueError, match="plus_times only"):
        tile_mac(None, A, B, "bf16", "fp32", numerics="tensix_wh", semiring="min_plus")


def test_register_numerics_plugs_in_a_strategy():
    calls = []

    def doubled(C, A, B, in_fmt, acc_fmt, *, k_chunk=None, scale=2.0):
        calls.append((in_fmt.name, acc_fmt.name, k_chunk))
        return tile_mac(C, A, B, in_fmt, acc_fmt) * scale

    register_numerics("test_doubled", doubled)
    try:
        A, B = np.eye(2), np.eye(2)
        out = tile_mac(None, A, B, "fp32", "fp32", numerics="test_doubled", scale=3.0)
        assert np.array_equal(out, 3 * np.eye(2))
        assert calls == [("fp32", "fp32", None)]
        with pytest.raises(ValueError, match="built-in"):
            register_numerics("ieee", doubled)
    finally:
        NUMERICS.pop("test_doubled")


def test_accuracy_table_smoke():
    from framework.matrix.tensix_accuracy import accuracy_table

    rows = accuracy_table(ks=(64,), size=8)
    norm = {(mode, golden): m["normwise"] for _, mode, golden, m in rows}
    assert len(rows) == 8
    assert norm[("ieee", "datapath")] < norm[("tensix HiFi4", "datapath")]
    assert norm[("tensix HiFi4", "datapath")] < norm[("tensix LoFi", "datapath")]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
