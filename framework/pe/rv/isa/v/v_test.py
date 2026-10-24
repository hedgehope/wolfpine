"""RVV unit tests: configuration, the accessor API, integer, fixed point, masks, permutes, memory.

Each instruction family is checked against a numpy or Python-integer reference,
with masking (undisturbed inactive elements), the undisturbed tail, LMUL > 1 and
fractional LMUL, and the vl corner cases. Floating point is ``fp_test.py``.
Spike agreement is the separate end-to-end evidence (``rvv_check`` and the C2
random differential fuzzer).
"""

from fractions import Fraction

import numpy as np
import pytest

from framework.pe.rv.isa.v import isa as visa
from framework.pe.rv.isa.v.state import VILL, VectorState, VectorStateError
from framework.pe.rv.isa.v.vtesting import (
    DATA,
    M64,
    OPIVI,
    OPIVV,
    OPIVX,
    OPMVV,
    OPMVX,
    UINT,
    VCore,
    opv,
    vmem,
    vsetivli,
    vsetvl,
    vsetvli,
    vtype,
)

RNG = np.random.default_rng(7)


def rand(sew, n):
    return RNG.integers(0, 1 << sew, n, dtype=np.uint64).astype(UINT[sew])


def signed(x, sew):
    x = int(x) & ((1 << sew) - 1)
    return x - (1 << sew) if x >> (sew - 1) else x


# -- vset and CSRs -------------------------------------------------------------------


@pytest.mark.parametrize(
    "sew,lmul,vlmax",
    [(8, 1, 32), (32, 1, 8), (64, 8, 32), (32, 0.5, 4), (8, 0.125, 4), (16, 0.25, 4)],
)
def test_vsetvli_vlmax(sew, lmul, vlmax):
    c = VCore()
    assert c.vset(sew, lmul) == vlmax
    assert c.vs.vlmax == vlmax
    assert not c.vs.vill


def test_vl_rules_follow_spike():
    c = VCore()
    assert c.vset(32, 1, vl=5) == 5
    assert c.vset(32, 1, vl=100) == 8  # min(AVL, VLMAX)
    c.x(6, 3)
    c.exec(vsetvli(0, 6, vtype(32, 2)))  # rd = x0, rs1 != x0: vl = min(3, 16)
    assert c.vs.vl == 3
    c.exec(vsetvli(0, 0, vtype(16, 1)))  # x0, x0, same VLMAX (16): keep vl
    assert c.vs.vl == 3
    assert not c.vs.vill
    c.exec(vsetvli(0, 0, vtype(32, 1)))  # x0, x0, VLMAX changes: vill
    assert c.vs.vill
    assert c.vs.vl == 0
    assert c.vs.vtype == VILL
    c.exec(vsetivli(7, 31, vtype(8, 1)))
    assert c.x(7) == 31
    assert c.vs.vl == 31
    c.x(11, vtype(64, 4))
    c.x(12, 1000)
    c.exec(vsetvl(13, 12, 11))
    assert c.x(13) == 16


@pytest.mark.parametrize(
    "vt",
    [
        vtype(64, 0.5),  # SEW > ELEN * LMUL (Spike)
        vtype(16, 0.125),
        0b100,  # reserved vlmul
        (4 << 3),  # SEW 128
        1 << 8,  # a reserved bit
    ],
)
def test_unsupported_vtype_sets_vill(vt):
    c = VCore()
    c.x(11, vt)
    c.x(12, 4)
    c.exec(vsetvl(13, 12, 11))
    assert c.vs.vill
    assert c.vs.vtype == VILL
    assert c.x(13) == 0
    assert not c.legal(opv(OPIVV, 0x00, 1, 2, 3))  # every vector op is then illegal


def test_vector_csrs():
    c = VCore()
    st = c.vs
    c.vset(16, 2, vl=9)
    csrs = c.core.csrs
    assert csrs.read(0xC22) == 32
    assert csrs.read(0xC20) == 9
    assert csrs.read(0xC21) == vtype(16, 2)
    csrs.write(0x00F, 0b101)  # vcsr = {vxrm = 2, vxsat = 1}
    assert (st.vxrm, st.vxsat) == (2, 1)
    assert csrs.read(0x00A) == 2
    assert csrs.read(0x009) == 1
    csrs.write(0x008, 3)
    assert st.vstart == 3
    with pytest.raises(Exception, match="read-only"):
        csrs.write(0xC20, 1)
    assert c.core.csrs.misa >> 21 & 1  # misa.V


def test_vtype_extension_hook():
    """IME's lambda field: legal only once installed; the hook sees every vset."""
    st = VectorState(vlen=256)
    lam = 0b011 << 60
    st.set_vl(1, 1, 4, vtype(32, 1) | lam)
    assert st.vill
    seen = []

    def on_vset(requested, old, form):
        seen.append(form)
        return requested if form == "vsetvl" else requested | (old & (0x7 << 60))

    st.install_vtype_extension(0x7 << 60, on_vset)
    st.set_vl(1, 1, 4, vtype(32, 1) | lam, "vsetvl")
    assert not st.vill
    assert st.vtype == vtype(32, 1) | lam
    st.set_vl(1, 1, 4, vtype(16, 1), "vsetvli")  # vsetvli keeps lambda via the hook
    assert st.vtype == vtype(16, 1) | lam
    assert st.sew == 16
    assert seen == ["vsetvl", "vsetvli"]
    assert st.vtype_fields(st.vtype)[4] == 0
    with pytest.raises(VectorStateError):
        st.install_vtype_extension(VILL)


def test_ime_and_reserved_words_are_declined():
    """Zvvm's OP-V multiply-accumulates and mew=1 tile loads are left for a later ISA."""
    for word in (
        opv(OPIVV, 0x38, 8, 16, 24),  # vmmacc.vv-style OPIVV 0x38..0x3b
        opv(OPIVV, 0x3B, 8, 16, 24),
        opv(1, 0x14, 8, 16, 24),  # OPFVV 0x14..0x17
        opv(1, 0x17, 8, 16, 24),
        vmem(True, 64, 8, 10, mew=1),
        vmem(False, 32, 8, 10, mew=1),
    ):
        assert visa.decode(word) is None
        assert visa.RV64_V_ISA.run(None, None, False, word) is False


def test_vreg_group_views_alias_the_register_file():
    st = VectorState(vlen=256)
    g = st.vreg_group(8, 4, np.float32)
    assert g.shape == (32,)
    assert g.base is not None
    g[:] = np.arange(32)
    assert st.vrf[9].view(np.float32)[0] == 8.0  # element 8 is v9's first
    assert st.vreg_group(8, Fraction(1, 2), 16).shape == (16,)
    assert st.vreg_group(2, 1, "bf16").dtype.name == "bfloat16"
    with pytest.raises(VectorStateError):
        st.vreg_group(9, 2, 32)
    b = st.vreg_bytes(8, 2)
    b[0] = 0xAB
    assert st.vrf[8, 0] == 0xAB


def test_zvvm_tile_layout_is_buildable_from_byte_views():
    """The IME register<->tile index function as a numpy view over vreg_bytes (no copy on read-back)."""
    st = VectorState(vlen=256)
    R, lam, W = 2, 2, 1  # registers, lambda, widening
    rows = st.vlenb // 4 // (lam * W)
    span = st.vreg_bytes(8, R).view(np.float32).reshape(R, rows, lam * W)
    tile = np.arange(rows * R * lam * W, dtype=np.float32).reshape(rows, R * lam * W)
    span[...] = tile.reshape(rows, R, lam * W).transpose(1, 0, 2)
    back = span.transpose(1, 0, 2).reshape(rows, R * lam * W)
    assert np.array_equal(back, tile)
    assert np.shares_memory(span, st.vrf)


# -- integer arithmetic --------------------------------------------------------------------

_BIN_REF = {
    0x00: lambda a, b, s: a + b,
    0x02: lambda a, b, s: a - b,
    0x09: lambda a, b, s: a & b,
    0x0A: lambda a, b, s: a | b,
    0x0B: lambda a, b, s: a ^ b,
    0x04: lambda a, b, s: min(a, b),
    0x05: lambda a, b, s: min(signed(a, s), signed(b, s)),
    0x06: lambda a, b, s: max(a, b),
    0x07: lambda a, b, s: max(signed(a, s), signed(b, s)),
    0x25: lambda a, b, s: a << (b & (s - 1)),
    0x28: lambda a, b, s: a >> (b & (s - 1)),
    0x29: lambda a, b, s: signed(a, s) >> (b & (s - 1)),
}


def _opv_case(c, funct3, funct6, sew, lmul, vl, vm, ref, scalar=None):
    n = c.vs.vlen * max(1, lmul) // sew
    c.vset(sew, lmul, vl=vl)
    a, b, old = rand(sew, n), rand(sew, n), rand(sew, n)
    nreg = max(1, int(lmul))
    c.vset_data(8, a, UINT[sew], nreg)
    c.vset_data(16, b, UINT[sew], nreg)
    c.vset_data(24, old, UINT[sew], nreg)
    mask = RNG.integers(0, 2, c.vs.vlen).astype(bool)
    c.set_mask(mask)
    src = 16
    if scalar is not None:
        c.x(7, scalar)
        src = 7
    c.exec(opv(funct3, funct6, 24, 8, src, vm))
    got = c.v(24, UINT[sew], nreg)
    vlen_now = c.vs.vl
    for i in range(n):
        if i >= vlen_now or (not vm and not mask[i]):
            assert got[i] == old[i], (i, "undisturbed")
        else:
            bb = int(b[i]) if scalar is None else scalar & ((1 << sew) - 1)
            assert int(got[i]) == ref(int(a[i]), bb, sew) & ((1 << sew) - 1), (
                hex(funct6),
                sew,
                i,
            )


@pytest.mark.parametrize("funct6", sorted(_BIN_REF))
@pytest.mark.parametrize("sew,lmul", [(8, 1), (16, 2), (32, 0.5), (64, 4)])
@pytest.mark.parametrize("vm", [0, 1])
def test_integer_binary_vv(funct6, sew, lmul, vm):
    c = VCore()
    vlmax = int(c.vs.vlen * lmul // sew)
    vl = vlmax - 3 if vlmax > 3 else None
    _opv_case(c, OPIVV, funct6, sew, lmul, vl, vm, _BIN_REF[funct6])


@pytest.mark.parametrize("funct6", [0x00, 0x03, 0x0B, 0x25, 0x29])
def test_integer_binary_vx_and_vi(funct6):
    rsub = lambda a, b, s: b - a  # noqa: E731
    ref = _BIN_REF.get(funct6, rsub)
    c = VCore()
    _opv_case(c, OPIVX, funct6, 32, 2, 11, 1, ref, scalar=-5 & M64)
    # .vi: simm5 = -3 (sign-extended), or uimm5 = 29 for the shifts.
    c.vset(16, 1)
    a = rand(16, 16)
    c.vset_data(8, a, UINT[16])
    c.exec(opv(OPIVI, funct6, 24, 8, 0b11101))
    imm = 29 if funct6 in (0x25, 0x29) else -3
    got = c.v(24, UINT[16])
    assert all(
        int(got[i]) == ref(int(a[i]), imm & 0xFFFF, 16) & 0xFFFF for i in range(16)
    )


def _mul_ref(name, a, b, s):
    sa, sb = signed(a, s), signed(b, s)
    return {
        "mul": a * b,
        "mulh": (sa * sb) >> s,
        "mulhu": (a * b) >> s,
        "mulhsu": (sa * b) >> s,
    }[name]


@pytest.mark.parametrize(
    "name,f6", [("mul", 0x25), ("mulh", 0x27), ("mulhu", 0x24), ("mulhsu", 0x26)]
)
@pytest.mark.parametrize("sew", [8, 16, 32, 64])
def test_multiply_high(name, f6, sew):
    c = VCore()
    _opv_case(c, OPMVV, f6, sew, 1, None, 1, lambda a, b, s: _mul_ref(name, a, b, s))


def _div_ref(name, a, b, s):
    sa, sb = signed(a, s), signed(b, s)
    if name == "divu":
        return (1 << s) - 1 if b == 0 else a // b
    if name == "remu":
        return a if b == 0 else a % b
    if b == 0:
        return -1 if name == "div" else sa
    if sa == -(1 << (s - 1)) and sb == -1:
        return sa if name == "div" else 0
    q = abs(sa) // abs(sb) * (1 if (sa < 0) == (sb < 0) else -1)
    return q if name == "div" else sa - q * sb


@pytest.mark.parametrize(
    "name,f6", [("divu", 0x20), ("div", 0x21), ("remu", 0x22), ("rem", 0x23)]
)
@pytest.mark.parametrize("sew", [8, 64])
def test_divide_with_zero_and_overflow(name, f6, sew):
    c = VCore()
    c.vset(sew, 1)
    n = c.vs.vl
    a, b = rand(sew, n), rand(sew, n)
    a[0], b[0] = 7, 0
    a[1], b[1] = 1 << (sew - 1), (1 << sew) - 1  # INT_MIN / -1
    b[2] = 0
    c.vset_data(8, a, UINT[sew])
    c.vset_data(16, b, UINT[sew])
    c.exec(opv(OPMVV, f6, 24, 8, 16))
    got = c.v(24, UINT[sew])
    for i in range(n):
        assert int(got[i]) == _div_ref(name, int(a[i]), int(b[i]), sew) & (
            (1 << sew) - 1
        )


def test_multiply_add_family():
    c = VCore()
    c.vset(32, 1)
    a, b, d = rand(32, 8), rand(32, 8), rand(32, 8)
    for f6, ref in (
        (0x2D, lambda a, b, d: b * a + d),  # vmacc
        (0x2F, lambda a, b, d: d - b * a),  # vnmsac
        (0x29, lambda a, b, d: b * d + a),  # vmadd
        (0x2B, lambda a, b, d: a - b * d),  # vnmsub
    ):
        c.vset_data(8, a, UINT[32])
        c.vset_data(16, b, UINT[32])
        c.vset_data(24, d, UINT[32])
        c.exec(opv(OPMVV, f6, 24, 8, 16))
        got = c.v(24, UINT[32])
        assert all(
            int(got[i]) == ref(int(a[i]), int(b[i]), int(d[i])) & 0xFFFFFFFF
            for i in range(8)
        )


def test_compare_writes_mask_bits_and_leaves_the_tail():
    c = VCore()
    c.vset(16, 2, vl=20)
    a, b = rand(16, 32), rand(16, 32)
    b[:5] = a[:5]
    c.vset_data(8, a, UINT[16], 2)
    c.vset_data(16, b, UINT[16], 2)
    c.vs.vrf[4] = 0xFF
    mask = RNG.integers(0, 2, 256).astype(bool)
    c.set_mask(mask)
    c.exec(opv(OPIVV, 0x1B, 4, 8, 16, vm=0))  # vmslt.vv v4, v8, v16, v0.t
    got = c.vs.mask(4)
    for i in range(256):
        if i < 20 and mask[i]:
            assert got[i] == (signed(a[i], 16) < signed(b[i], 16))
        else:
            assert got[i]  # undisturbed (was all ones)


def test_carry_chain_matches_a_wide_add():
    """vadc / vmadc over two 32-bit limbs of a 64-bit add."""
    c = VCore()
    c.vset(32, 1)
    x, y = rand(64, 8), rand(64, 8)
    lo_x, hi_x = (
        (x & 0xFFFFFFFF).astype(np.uint32),
        (x >> np.uint64(32)).astype(np.uint32),
    )
    lo_y, hi_y = (
        (y & 0xFFFFFFFF).astype(np.uint32),
        (y >> np.uint64(32)).astype(np.uint32),
    )
    c.vset_data(8, lo_x, UINT[32])
    c.vset_data(9, lo_y, UINT[32])
    c.vset_data(10, hi_x, UINT[32])
    c.vset_data(11, hi_y, UINT[32])
    c.exec(opv(OPIVV, 0x11, 0, 8, 9, vm=1))  # vmadc.vv v0, v8, v9 (carry out)
    c.exec(opv(OPIVV, 0x00, 12, 8, 9))  # low sum
    c.exec(opv(OPIVV, 0x10, 13, 10, 11, vm=0))  # vadc.vvm v13, v10, v11, v0
    total = (c.v(12, UINT[32]).astype(np.uint64)) | (
        c.v(13, UINT[32]).astype(np.uint64) << np.uint64(32)
    )
    assert np.array_equal(total[:8], x + y)
    assert not c.legal(opv(OPIVV, 0x10, 13, 10, 11, vm=1))  # vadc needs vm = 0


def test_merge_and_move():
    c = VCore()
    c.vset(64, 1)
    a, b = rand(64, 4), rand(64, 4)
    c.vset_data(8, a, UINT[64])
    c.vset_data(16, b, UINT[64])
    c.set_mask([1, 0, 1, 0])
    c.exec(opv(OPIVV, 0x17, 24, 8, 16, vm=0))
    assert list(c.v(24, UINT[64])[:4]) == [b[0], a[1], b[2], a[3]]
    c.exec(opv(OPIVI, 0x17, 24, 0, 0b10000, vm=1))  # vmv.v.i v24, -16
    assert all(c.v(24, UINT[64])[:4] == (-16 & M64))


@pytest.mark.parametrize(
    "f6,signed_a,signed_b,op",
    [
        (0x30, 0, 0, "add"),
        (0x31, 1, 1, "add"),
        (0x33, 1, 1, "sub"),
        (0x38, 0, 0, "mul"),
        (0x3B, 1, 1, "mul"),
        (0x3A, 1, 0, "mul"),
    ],
)
@pytest.mark.parametrize("sew", [8, 32])
def test_widening_integer(f6, signed_a, signed_b, op, sew):
    c = VCore()
    c.vset(sew, 1, vl=c.vs.vlen // sew - 1)
    n = c.vs.vl
    a, b = rand(sew, n), rand(sew, n)
    c.vset_data(8, a, UINT[sew])
    c.vset_data(9, b, UINT[sew])
    c.exec(opv(OPMVV, f6, 16, 8, 9))
    got = c.v(16, UINT[2 * sew], 2)
    for i in range(n):
        x = signed(a[i], sew) if signed_a else int(a[i])
        y = signed(b[i], sew) if signed_b else int(b[i])
        r = {"add": x + y, "sub": x - y, "mul": x * y}[op]
        assert int(got[i]) == r & ((1 << 2 * sew) - 1)
    assert not c.legal(opv(OPMVV, f6, 17, 8, 9))  # misaligned wide destination
    assert c.legal(
        opv(OPMVV, f6, 8, 9, 9)
    )  # a source in the destination group's top half is legal
    assert not c.legal(opv(OPMVV, f6, 8, 8, 9))  # ... but not in its bottom half


def test_widening_rejected_at_elen():
    c = VCore()
    c.vset(64, 1)
    assert not c.legal(opv(OPMVV, 0x30, 16, 8, 9))


@pytest.mark.parametrize("vxrm", [0, 1, 2, 3])
def test_narrowing_clip_and_shift(vxrm):
    c = VCore()
    c.vs.vxrm = vxrm
    c.vset(16, 1)
    n = c.vs.vl
    a = rand(32, n)
    c.vset_data(8, a, UINT[32], 2)
    shift = 7
    c.x(7, shift)
    c.exec(opv(OPIVX, 0x2F, 16, 8, 7))  # vnclip.wx
    c.exec(opv(OPIVX, 0x2C, 17, 8, 7))  # vnsrl.wx
    clip, srl = c.v(16, UINT[16]), c.v(17, UINT[16])
    sat = False
    for i in range(n):
        v = signed(a[i], 32)
        r = _roundoff(v, shift, vxrm)
        expect = max(-(1 << 15), min((1 << 15) - 1, r))
        sat |= expect != r
        assert signed(clip[i], 16) == expect
        assert int(srl[i]) == (int(a[i]) >> shift) & 0xFFFF
    assert c.vs.vxsat == int(sat)


def _roundoff(v, d, vxrm):
    if d == 0:
        return v
    bit = (v >> (d - 1)) & 1
    rest = v & ((1 << (d - 1)) - 1)
    vd = (v >> d) & 1
    r = [bit, bit & (int(rest != 0) | vd), 0, (1 - vd) & int(bit or rest != 0)][vxrm]
    return (v >> d) + r


@pytest.mark.parametrize("vxrm", [0, 1, 2, 3])
@pytest.mark.parametrize("sew", [8, 64])
def test_fixed_point(vxrm, sew):
    c = VCore()
    c.vs.vxrm = vxrm
    c.vset(sew, 1)
    n = c.vs.vl
    a, b = rand(sew, n), rand(sew, n)
    a[0] = b[0] = 1 << (sew - 1)  # vsmul's one overflow
    c.vset_data(8, a, UINT[sew])
    c.vset_data(9, b, UINT[sew])
    lo, hi = -(1 << (sew - 1)), (1 << (sew - 1)) - 1
    m = (1 << sew) - 1
    for f6, ref in (
        (0x21, lambda x, y: max(lo, min(hi, signed(x, sew) + signed(y, sew)))),  # vsadd
        (0x20, lambda x, y: min(m, x + y)),  # vsaddu
        (0x23, lambda x, y: max(lo, min(hi, signed(x, sew) - signed(y, sew)))),  # vssub
        (0x22, lambda x, y: max(0, x - y)),  # vssubu
        (
            0x27,
            lambda x, y: max(
                lo, min(hi, _roundoff(signed(x, sew) * signed(y, sew), sew - 1, vxrm))
            ),
        ),  # vsmul
        (0x2A, lambda x, y: _roundoff(x, y & (sew - 1), vxrm)),  # vssrl
        (0x2B, lambda x, y: _roundoff(signed(x, sew), y & (sew - 1), vxrm)),  # vssra
    ):
        c.exec(opv(OPIVV, f6, 16, 8, 9))
        got = c.v(16, UINT[sew])
        for i in range(n):
            assert int(got[i]) == ref(int(a[i]), int(b[i])) & m, (hex(f6), i)
    for f6, ref in (
        (0x08, lambda x, y: _roundoff(x + y, 1, vxrm)),  # vaaddu
        (
            0x09,
            lambda x, y: _roundoff(signed(x, sew) + signed(y, sew), 1, vxrm),
        ),  # vaadd
        (0x0A, lambda x, y: _roundoff(x - y, 1, vxrm)),  # vasubu
        (
            0x0B,
            lambda x, y: _roundoff(signed(x, sew) - signed(y, sew), 1, vxrm),
        ),  # vasub
    ):
        c.exec(opv(OPMVV, f6, 16, 8, 9))
        got = c.v(16, UINT[sew])
        for i in range(n):
            assert int(got[i]) == ref(int(a[i]), int(b[i])) & m, (hex(f6), i)


def test_extensions():
    c = VCore()
    c.vset(8, 0.25)
    src = rand(8, 8)
    c.vset_data(9, src, UINT[8])
    c.vset(32, 1)
    c.exec(opv(OPMVV, 0x12, 16, 9, 5))  # vsext.vf4
    assert [signed(x, 32) for x in c.v(16, UINT[32])[:8]] == [signed(x, 8) for x in src]
    c.exec(opv(OPMVV, 0x12, 16, 9, 4))  # vzext.vf4
    assert list(c.v(16, UINT[32])[:8]) == list(src.astype(np.uint32))
    c.vset(64, 1)
    assert (
        c.legal(opv(OPMVV, 0x12, 16, 9, 2)) is not False
    )  # vzext.vf8 at e64 from e8 is legal
    c.vset(16, 1)
    assert not c.legal(opv(OPMVV, 0x12, 16, 9, 2))  # vf8 at e16: source would be e2


def test_integer_reductions_masked():
    c = VCore()
    c.vset(16, 4, vl=50)
    a = rand(16, 64)
    c.vset_data(8, a, UINT[16], 4)
    c.vset_data(4, [1234], UINT[16])
    mask = RNG.integers(0, 2, 256).astype(bool)
    c.set_mask(mask)
    act = [int(a[i]) for i in range(50) if mask[i]]
    for f6, ref in (
        (0x00, lambda xs: (1234 + sum(xs)) & 0xFFFF),
        (0x06, lambda xs: max([1234, *xs])),
        (0x05, lambda xs: min([1234, *[signed(x, 16) for x in xs]]) & 0xFFFF),
        (0x01, lambda xs: np.bitwise_and.reduce([1234, *xs])),
    ):
        c.vset_data(2, [0xDEAD, 0xBEEF], UINT[16])
        c.exec(opv(OPMVV, f6, 2, 8, 4, vm=0))
        got = c.v(2, UINT[16])
        assert int(got[0]) == int(ref(act))
        assert got[1] == 0xBEEF
    c.exec(opv(OPIVV, 0x31, 2, 8, 4))  # vwredsum.vs: the start is e32
    start32 = int(c.v(4, UINT[32])[0])
    assert (
        int(c.v(2, UINT[32])[0])
        == (signed(start32, 32) + sum(signed(x, 16) for x in a[:50])) & 0xFFFFFFFF
    )
    c.vset(16, 4, vl=0)
    c.vset_data(2, [7], UINT[16])
    c.exec(opv(OPMVV, 0x00, 2, 8, 4))
    assert c.v(2, UINT[16])[0] == 7  # vl = 0: vd untouched


def test_arithmetic_with_nonzero_vstart_is_illegal():
    c = VCore()
    c.vset(32, 1)
    c.vs.vstart = 2
    assert not c.legal(opv(OPIVV, 0x00, 1, 2, 3))


# -- masks ------------------------------------------------------------------------------------


def test_mask_logical_cpop_first():
    c = VCore()
    c.vset(8, 1, vl=29)
    a = RNG.integers(0, 2, 256).astype(bool)
    b = RNG.integers(0, 2, 256).astype(bool)
    c.set_mask(a, 8)
    c.set_mask(b, 9)
    c.vs.vrf[10] = 0xFF
    c.exec(opv(OPMVV, 0x18, 10, 8, 9))  # vmandn.mm
    got = c.vs.mask(10)
    assert list(got[:29]) == list(a[:29] & ~b[:29])
    assert got[29:].all()
    c.exec(opv(OPMVV, 0x10, 7, 8, 0x10))  # vcpop.m
    assert c.x(7) == int(a[:29].sum())
    c.set_mask(np.zeros(256, bool), 11)
    c.exec(opv(OPMVV, 0x10, 7, 11, 0x11))  # vfirst.m with none set
    assert c.x(7) == M64


def test_set_first_and_iota_and_vid():
    c = VCore()
    c.vset(32, 2, vl=12)
    src = np.zeros(256, bool)
    src[[3, 7, 9]] = True
    c.set_mask(src, 8)
    for code, expect in (
        (1, [i < 3 for i in range(12)]),
        (3, [i <= 3 for i in range(12)]),
        (2, [i == 3 for i in range(12)]),
    ):
        c.vs.vrf[9] = 0
        c.exec(opv(OPMVV, 0x14, 9, 8, code))
        assert list(c.vs.mask(9)[:12]) == expect
    c.exec(opv(OPMVV, 0x14, 16, 8, 0x10))  # viota.m v16, v8
    assert list(c.v(16, UINT[32], 2)[:12]) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3]
    c.exec(opv(OPMVV, 0x14, 16, 0, 0x11))  # vid.v
    assert list(c.v(16, UINT[32], 2)[:12]) == list(range(12))


# -- permutes ------------------------------------------------------------------------------------


def test_scalar_moves():
    c = VCore()
    c.vset(16, 1, vl=0)
    c.vset_data(8, [0x8001], UINT[16])
    c.exec(opv(OPMVV, 0x10, 7, 8, 0))  # vmv.x.s ignores vl
    assert c.x(7) == (-0x7FFF) & M64
    c.x(6, 0x1234)
    c.exec(opv(OPMVX, 0x10, 9, 0, 6))  # vmv.s.x with vl = 0: no write
    assert c.v(9, UINT[16])[0] == 0
    c.vset(16, 1)
    c.exec(opv(OPMVX, 0x10, 9, 0, 6))
    assert c.v(9, UINT[16])[0] == 0x1234


def test_slides():
    c = VCore()
    c.vset(32, 1, vl=7)
    a = np.arange(1, 9, dtype=np.uint32)
    c.vset_data(8, a, UINT[32])
    c.vset_data(16, np.full(8, 99), UINT[32])
    c.x(6, 3)
    c.exec(opv(OPIVX, 0x0E, 16, 8, 6))  # vslideup.vx by 3
    assert list(c.v(16, UINT[32])) == [99, 99, 99, 1, 2, 3, 4, 99]
    c.exec(
        opv(OPIVI, 0x0F, 17, 8, 2)
    )  # vslidedown.vi by 2: vd[i] = a[i+2] if i+2 < VLMAX
    assert list(c.v(17, UINT[32])[:7]) == [3, 4, 5, 6, 7, 8, 0]
    c.x(6, 1 << 63)
    c.exec(opv(OPIVX, 0x0F, 17, 8, 6))  # a huge offset reads zeros, no wrap
    assert not c.v(17, UINT[32])[:7].any()
    c.x(6, 42)
    c.exec(opv(OPMVX, 0x0F, 18, 8, 6))  # vslide1down.vx
    assert list(c.v(18, UINT[32])[:7]) == [2, 3, 4, 5, 6, 7, 42]
    assert not c.legal(opv(OPIVX, 0x0E, 8, 8, 6))  # slide-up onto its source


def test_gathers_and_compress():
    c = VCore()
    c.vset(16, 2)
    table = np.arange(100, 132, dtype=np.uint16)
    idx = np.array([5, 0, 31, 32, 1000, *range(27)], dtype=np.uint16)
    c.vset_data(8, table, UINT[16], 2)
    c.vset_data(12, idx, UINT[16], 2)
    c.exec(opv(OPIVV, 0x0C, 16, 8, 12))
    got = c.v(16, UINT[16], 2)
    assert list(got[:5]) == [105, 100, 131, 0, 0]
    c.vset(64, 1)  # vrgatherei16 at e64: index EMUL = 1/4
    c.vset_data(8, [10, 20, 30, 40], UINT[64])
    c.vset_data(12, [3, 3, 0, 9], UINT[16])
    c.exec(opv(OPIVV, 0x0E, 16, 8, 12))
    assert list(c.v(16, UINT[64])[:4]) == [40, 40, 10, 0]
    c.vset_data(8, [10, 20, 30, 40], UINT[64])
    c.set_mask([0, 1, 0, 1], 4)
    c.vset_data(20, [7, 7, 7, 7], UINT[64])
    c.exec(opv(OPMVV, 0x17, 20, 8, 4))
    assert list(c.v(20, UINT[64])[:4]) == [20, 40, 7, 7]


def test_whole_register_move():
    c = VCore()
    c.vs.vrf[8:12] = RNG.integers(0, 256, (4, 32), dtype=np.uint8)
    c.exec(opv(OPIVI, 0x27, 16, 8, 3))  # vmv4r.v v16, v8
    assert np.array_equal(c.vs.vrf[16:20], c.vs.vrf[8:12])
    assert not c.legal(opv(OPIVI, 0x27, 17, 8, 1))  # vmv2r.v to an odd register


# -- loads and stores ------------------------------------------------------------------------------


def test_unit_stride_masked_load_and_store():
    c = VCore()
    c.vset(32, 2, vl=13)
    data = np.arange(1000, 1016, dtype=np.uint32)
    c.mem_write(DATA, data)
    c.x(10, DATA)
    mask = RNG.integers(0, 2, 256).astype(bool)
    c.set_mask(mask)
    c.vset_data(8, np.full(16, 5), UINT[32], 2)
    c.exec(vmem(True, 32, 8, 10, vm=0))
    got = c.v(8, UINT[32], 2)
    for i in range(16):
        assert got[i] == (data[i] if i < 13 and mask[i] else 5)
    c.x(11, DATA + 0x100)
    c.mem_write(DATA + 0x100, np.full(16, 0xEE, np.uint32))
    c.exec(vmem(False, 32, 8, 11, vm=0))
    out = c.mem_read(DATA + 0x100, 16, np.uint32)
    for i in range(16):
        assert out[i] == (data[i] if i < 13 and mask[i] else 0xEE)
    assert c.vs.mem_to_vec_bytes == 4 * int(mask[:13].sum())


def test_vstart_is_honoured_by_loads_and_cleared():
    c = VCore()
    c.vset(8, 1)
    c.mem_write(DATA, np.arange(32, dtype=np.uint8))
    c.x(10, DATA)
    c.vs.vstart = 30
    c.exec(vmem(True, 8, 8, 10))
    assert list(c.v(8, UINT[8])[28:]) == [0, 0, 30, 31]
    assert c.vs.vstart == 0


def test_strided_negative_and_indexed():
    c = VCore()
    c.vset(64, 2)
    words = np.arange(64, dtype=np.uint64) * 3
    c.mem_write(DATA, words)
    c.x(10, DATA + 8 * 40)
    c.x(11, (-16) & M64)
    c.exec(vmem(True, 64, 8, 10, mop=2, rs2=11))
    assert list(c.v(8, UINT[64], 2)[:8]) == [words[40 - 2 * i] for i in range(8)]
    offsets = np.array([8 * k for k in (5, 1, 63, 0, 2, 2, 7, 9)], dtype=np.uint32)
    c.vset(32, 1)
    c.vset_data(4, offsets, UINT[32])
    c.vset(64, 2)
    c.x(10, DATA)
    c.exec(vmem(True, 32, 12, 10, mop=1, rs2=4))  # vluxei32.v v12, (a0), v4 (data e64)
    assert list(c.v(12, UINT[64], 2)[:8]) == [
        words[k] for k in (5, 1, 63, 0, 2, 2, 7, 9)
    ]
    c.vset_data(16, np.arange(8, dtype=np.uint64) + 500, UINT[64], 2)
    c.exec(
        vmem(False, 32, 16, 10, mop=3, rs2=4)
    )  # vsoxei32 (ordered: the repeated 2 keeps the last)
    mem = c.mem_read(DATA, 64, np.uint64)
    assert mem[5] == 500
    assert mem[63] == 502
    assert mem[2] == 505


def test_segment_load_store_round_trip():
    c = VCore()
    c.vset(16, 1, vl=10)
    rgb = np.arange(64, dtype=np.uint16)
    c.mem_write(DATA, rgb)
    c.x(10, DATA)
    c.exec(vmem(True, 16, 8, 10, nf=3))  # vlseg3e16.v v8..v10
    for f in range(3):
        assert list(c.v(8 + f, UINT[16])[:10]) == list(rgb[f:30:3])
    c.x(11, DATA + 0x200)
    c.exec(vmem(False, 16, 8, 11, nf=3))
    assert list(c.mem_read(DATA + 0x200, 30, np.uint16)) == list(rgb[:30])
    c.x(12, 12)
    c.exec(
        vmem(True, 16, 16, 10, mop=2, rs2=12, nf=2)
    )  # vlsseg2e16 with a 12-byte stride
    assert list(c.v(16, UINT[16])[:10]) == list(rgb[0:60:6][:10])
    assert not c.legal(vmem(True, 16, 28, 10, nf=5))  # v28 + 5 > 32


def test_whole_register_and_mask_load_store():
    c = VCore()
    blob = RNG.integers(0, 256, 128, dtype=np.uint8)
    c.mem_write(DATA, blob)
    c.x(10, DATA)
    c.vs._set_vill()  # whole-register and... vill does not matter for whole-register ops
    c.exec(vmem(True, 32, 8, 10, rs2=0x08, nf=4))  # vl4re32.v
    assert np.array_equal(c.vs.vreg_bytes(8, 4), blob)
    c.x(11, DATA + 0x400)
    c.exec(vmem(False, 8, 8, 11, rs2=0x08, nf=2))  # vs2r.v
    assert np.array_equal(c.mem_read(DATA + 0x400, 64), blob[:64])
    c.vset(8, 1, vl=19)
    c.vs.vrf[20] = 0
    c.exec(vmem(True, 8, 20, 10, rs2=0x0B))  # vlm.v: ceil(19 / 8) = 3 bytes
    assert list(c.vs.vrf[20, :4]) == [*blob[:3], 0]


def test_fault_only_first_trims_vl():
    c = VCore()
    c.vset(32, 1)
    end = 0x40000  # the end of VCore's DRAM
    c.mem_write(end - 12, np.array([1, 2, 3], np.uint32))
    c.x(10, end - 12)
    c.exec(vmem(True, 32, 8, 10, rs2=0x10))  # vle32ff.v: elements 3.. fault
    assert c.vs.vl == 3
    assert list(c.v(8, UINT[32])[:3]) == [1, 2, 3]


def test_counters():
    c = VCore()
    c.vset(32, 1)
    c.x(10, DATA)
    c.exec(vmem(True, 32, 8, 10))
    c.exec(opv(OPIVV, 0x00, 9, 8, 8))
    c.exec(vmem(False, 32, 9, 10))
    assert c.vs.counters() == {
        "instructions": 4,
        "elements": 24,
        "mem_to_vec_bytes": 32,
        "vec_to_mem_bytes": 32,
    }
