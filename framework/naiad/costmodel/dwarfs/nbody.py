"""nbody: softened-gravity forces on n bodies (``kernels/naiad/dwarfs/nbody``); size ``(n, distribution)``.

Every variant is bound by the per-pair weight ``w(r^2) = 1 / (r^2 sqrt(r^2))``:
an ``fsqrt`` and an ``fdiv`` on the scalar side, unpipelined (``fp_sqrt +
fp_div`` cycles a pair, 40 on naiad-dwarfs), so the matrix reformulations can
only remove the distance arithmetic around them.

* **scalar**: per pair 3 loads, the distance (3 sub, 3 fma), the weight and 3
  force FMAs: ``n (n - 1)`` pairs of ``~24`` instructions behind ``sqrt + div``.
* **rvv**: VL bodies at once, two j a step; the weight by ``vfrsqrt7`` and two
  Newton steps (no divide), so ~2 x 4 cycles of datapath per j and lane group.
* **ame**: the distance GEMM (one mma per 16 x 16 tile, astored to memory, which
  the scalar loads then wait for) leaves the weight and the force sum on the
  scalar side: per pair a load, clamp, add, ``fsqrt``, ``fmul``, ``fdiv`` and
  3 FMAs. **ame_fgemm** also moves the force sum to the unit (a second mma
  per tile) at the price of rounding each weight to the input format.

**ame_fused** computes the weights on the accumulators (custom-3 element-wise
ops, an rsqrt) and runs the force GEMM in acc_fmt, so no pair touches the
scalar side; ``ame_rvv`` (the RVV-weights hybrid) is not modelled.

Formulas are the compiled loops of the n <= 256 build (``listings/nbody_*``);
larger n drops the ``d2`` store and differs by one store a pair.
"""

from framework.naiad.costmodel.dwarfs.common import (
    Unsupported,
    amov_to_tile,
    ewise,
    mx_drain,
    nbytes,
    need_acc_store,
    need_ame,
    need_features,
    need_vector,
)
from framework.naiad.costmodel.engine import new_engine
from framework.naiad.costmodel.listing import Program, VState

_CACHE = {}
A = 0x10000000


def shape(size):
    if isinstance(size, (tuple, list)):
        return int(size[0]), str(size[1]) if len(size) > 1 else "uniform"
    return int(size), "uniform"


def _listing(name):
    if name not in _CACHE:
        try:
            _CACHE[name] = Program.load(name)
        except FileNotFoundError:
            raise Unsupported(f"no compiled listing {name}") from None
    return _CACHE[name]


def _split(e, n, i, body, diag):
    """``body(j)`` for j != i and ``diag()`` at j == i, as three loops (no extrapolation across the diagonal)."""
    e.loop(i, body)
    diag()
    e.loop(n - i - 1, lambda j: body(i + 1 + j))


def scalar(m, size, in_fmt, acc_fmt):
    if (in_fmt, acc_fmt) not in (("fp32", "fp32"), ("fp64", "fp64")):
        raise Unsupported("the scalar nbody runs fp32/fp32 and fp64/fp64")
    n, _ = shape(size)
    P = _listing("nbody_scalar_fp32_fp32")
    e, vs = new_engine(m), VState()
    P.follow(e, P.after(P.find("rdcycle")), A + 0x90, vs)
    # positions -> P[] (the norms loop)
    e.loop(n - 1, lambda i: P.run(e, A + 0x90, A + 0xB0, vs, last_taken=True))
    P.follow(e, A + 0x90, A + 0x108, vs)

    def row(i):
        P.run(e, A + 0x108, A + 0x13C, vs)

        def pair(j):
            P.run(e, A + 0x150, A + 0x17C, vs, last_taken=False)
            P.run(e, A + 0x180, A + 0x19C, vs)
            P.run(e, A + 0x140, A + 0x14C, vs, last_taken=j == n - 1)

        def diag():
            P.run(e, A + 0x150, A + 0x17C, vs, last_taken=True)
            P.run(e, A + 0x140, A + 0x14C, vs, last_taken=i == n - 1)

        _split(e, n, i, pair, diag)
        P.run(e, A + 0xEC, A + 0x104, vs, last_taken=i == n - 1)

    e.loop(n, row)
    return e


def rvv(m, size, in_fmt, acc_fmt):
    v = need_vector(m)
    n, _ = shape(size)
    P = _listing(f"nbody_rvv_{in_fmt}_{acc_fmt}")
    e, vs = new_engine(m), VState()
    r0 = P.find("rdcycle")
    (bh, bb), (jh, jb), (ih, ib) = P.loops()[-3:]
    sew = 64 if acc_fmt == "fp64" else 32
    vlmax = v.vlen * 2 // sew
    strips = [min(vlmax, n - s) for s in range(0, n, vlmax)]
    P.follow(e, P.after(r0), bh, vs)
    # the operand set-up loop is strip-mined too
    e.loop(len(strips), lambda i: P.run(e, bh, bb, vs, last_taken=i < len(strips) - 1))
    P.follow(e, P.after(bb), P.insns[P.index[ih]].addr, vs)

    def strip(k):
        vs.avl = strips[k]
        P.run(e, ih, P.insns[P.index[jh] - 1].addr, vs)
        e.loop(n // 2, lambda j: P.run(e, jh, jb, vs, last_taken=j < n // 2 - 1))
        P.run(e, P.after(jb), ib, vs, last_taken=k < len(strips) - 1)

    e.loop(len(strips), strip)
    vs.avl = None
    return e


def _tiles(n):
    return n // 16


def ame(m, size, in_fmt, acc_fmt):
    x = need_ame(m)
    if (x.m, x.n, x.k) != (16, 16, 16):
        raise Unsupported("nbody_ame is written for a 16 x 16 x 16 unit")
    need_acc_store(x)
    n, _ = shape(size)
    if (in_fmt, acc_fmt) not in (("fp32", "fp32"), ("fp64", "fp64")):
        return _ame_16(m, size, in_fmt, acc_fmt, fgemm=False)
    P = _listing("nbody_ame_fp32_fp32")
    P.note(A + 0x23C, "!D").note(A + 0x2E8, "@D")
    e, vs = new_engine(m), VState()
    P.follow(e, P.after(P.find("rdcycle")), A + 0x108, vs)
    e.loop(n, lambda i: P.run(e, A + 0x108, A + 0x174, vs, last_taken=i < n - 1))
    P.follow(e, A + 0x178, A + 0x1C0, vs)
    t = _tiles(n)

    def iblock(bi):
        P.run(e, A + 0x1C0, A + 0x1FC, vs)

        def tile(bj):
            P.run(e, A + 0x214, A + 0x24C, vs)

            def row(r):
                P.run(e, A + 0x294, A + 0x2CC, vs)

                def pair(j):
                    P.run(e, A + 0x2E0, A + 0x324, vs)
                    P.run(e, A + 0x2D0, A + 0x2DC, vs, last_taken=j == 15)

                if bi == bj:

                    def diag(r=r):
                        P.run(e, A + 0x2E0, A + 0x2E4, vs, last_taken=True)
                        P.run(e, A + 0x2D0, A + 0x2DC, vs, last_taken=r == 15)

                    _split(e, 16, r, pair, diag)
                else:
                    e.loop(16, pair)
                P.run(e, A + 0x250, A + 0x290, vs, last_taken=r == 15)

            e.loop(16, row)
            P.run(e, A + 0x200, A + 0x210, vs, last_taken=bj == t - 1)

        _split(e, t, bi, tile, lambda: tile(bi))
        P.follow(e, A + 0x328, A + 0x364, vs)
        P.run(e, A + 0x364, A + 0x364, vs, last_taken=bi < t - 1)

    e.loop(t, iblock)
    for a in range(x.accs):
        e.azero(a)
    return e


def ame_fgemm(m, size, in_fmt, acc_fmt):
    x = need_ame(m)
    if (x.m, x.n, x.k) != (16, 16, 16):
        raise Unsupported("nbody_ame_fgemm is written for a 16 x 16 x 16 unit")
    need_acc_store(x)
    n, _ = shape(size)
    if (in_fmt, acc_fmt) not in (("fp32", "fp32"), ("fp64", "fp64")):
        return _ame_16(m, size, in_fmt, acc_fmt, fgemm=True)
    P = _listing("nbody_ame_fgemm_fp32_fp32")
    # the D tile's astore -> the weight loop's loads; the W tile's stores -> its tile load
    P.note(A + 0x2B4, "!D").note(A + 0x31C, "@D")
    e, vs = new_engine(m), VState()
    P.follow(e, P.after(P.find("rdcycle")), A + 0x114, vs)
    e.loop(n, lambda i: P.run(e, A + 0x114, A + 0x194, vs, last_taken=i < n - 1))
    P.follow(e, A + 0x198, A + 0x1DC, vs)
    t = _tiles(n)

    def iblock(bi):
        P.follow(e, A + 0x1DC, A + 0x214, vs)
        e.alu(16 * 16 * 4 // 8 // 4)  # memset of the W tile (word stores, ~unrolled)
        P.run(e, A + 0x218, A + 0x244, vs)

        def tile(bj):
            P.run(e, A + 0x278, A + 0x2D4, vs)

            def row(r):
                P.run(e, A + 0x2EC, A + 0x2FC, vs)

                def w(j):
                    P.run(e, A + 0x314, A + 0x318, vs, last_taken=False)
                    P.run(e, A + 0x31C, A + 0x338, vs)
                    P.run(e, A + 0x300, A + 0x310, vs, last_taken=j == 15)

                if bi == bj:

                    def diag(r=r):
                        P.run(e, A + 0x314, A + 0x318, vs, last_taken=True)
                        P.run(e, A + 0x300, A + 0x310, vs, last_taken=r == 15)

                    _split(e, 16, r, w, diag)
                else:
                    e.loop(16, w)
                P.run(e, A + 0x2D8, A + 0x2E8, vs, last_taken=r == 15)

            e.loop(16, row)
            P.run(e, A + 0x248, A + 0x274, vs, last_taken=bj == t - 1)

        _split(e, t, bi, tile, lambda: tile(bi))
        P.follow(e, A + 0x33C, A + 0x3FC, vs, taken=())
        P.run(e, A + 0x3FC, A + 0x3FC, vs, last_taken=bi < t - 1)

    e.loop(t, iblock)
    for a in range(x.accs):
        e.azero(a)
    return e


#: 16-bit inputs, ``ame_fgemm``: each weight is rounded to the input format on the hot path.
#: Its per-pair path in that format's listing (the normal-number path), replayed in steady
#: state against the fp32 pair path; ``ame`` keeps its pair loop in fp32 (the same code).
_FGEMM_PAIR = {
    "fp32": (
        "nbody_ame_fgemm_fp32_fp32",
        (
            (0x10000314, 0x10000318, False),
            (0x1000031C, 0x10000338, True),
            (0x10000300, 0x10000310, False),
        ),
    ),
    "bf16": (
        "nbody_ame_fgemm_bf16_fp32",
        ((0x100003E4, 0x10000428, True), (0x10000430, 0x10000444, False)),
    ),
    "fp16": (
        "nbody_ame_fgemm_fp16_fp32",
        (
            (0x10000650, 0x1000069C, True),
            (0x100006C8, 0x100006C8, True),
            (0x100006D4, 0x100006D8, True),
            (0x100006B8, 0x100006C4, False),
        ),
    ),
}


def _steady(m, name, path, iters=24):
    """Cycles per iteration of a listing path once it is in steady state."""
    P = _listing(name)
    e, vs = new_engine(m), VState()

    def once():
        for a, b, tk in path:
            P.run(e, a, b, vs, last_taken=tk)

    for _ in range(4):
        once()
    t0 = e.t
    for _ in range(iters):
        once()
    return (e.t - t0) / iters


def _ame_16(m, size, in_fmt, acc_fmt, fgemm):
    if in_fmt not in ("bf16", "fp16") or acc_fmt != "fp32":
        raise Unsupported(f"nbody ame formats {in_fmt}/{acc_fmt}")
    n, _ = shape(size)
    base = (ame_fgemm if fgemm else ame)(m, size, "fp32", "fp32")
    if fgemm:
        delta = _steady(m, *_FGEMM_PAIR[in_fmt]) - _steady(m, *_FGEMM_PAIR["fp32"])
        base.add_cycles(round(n * (n - 1) * delta))
    return base


def ame_fused(m, size, in_fmt, acc_fmt):
    """``ame_fused``: both GEMMs, the weights on the accumulators, two column tiles interleaved.

    The operand build is ``ame``'s compiled per-particle loop plus the four
    acc_fmt position stores per particle (and the 16-bit build's conversions
    are not modelled: 16-bit inputs reuse the fp32 build, as ``ame``'s model
    does). Per pair of column tiles, in the kernel's order: two distance mmas,
    the ``d2`` astores (n <= 256), the element-wise chains (max, add, rsqrt,
    amov, two muls, the mask on a diagonal tile, amov) with the next tiles'
    loads between them, and the two force mmas in acc_fmt; the row block's F
    is combined on the scalar side (3 loads, an FMA and a store per output).
    """
    x = need_ame(m)
    if (x.m, x.n, x.k) != (16, 16, 16) or x.tiles < 8 or x.accs < 4:
        raise Unsupported("nbody_ame_fused needs a 16 x 16 x 16 unit, 8 tiles, 4 accs")
    if not x.acc_mem:
        raise Unsupported("nbody_ame_fused needs aload/astore")
    need_features(x, "ewise", "ewise_tx", "acc_move")
    n, _ = shape(size)
    if (in_fmt, acc_fmt) not in (
        ("fp32", "fp32"),
        ("fp64", "fp64"),
        ("bf16", "fp32"),
        ("fp16", "fp32"),
    ):
        raise Unsupported(f"nbody ame_fused formats {in_fmt}/{acc_fmt}")
    if (n // 16) % 2:
        raise Unsupported("nbody_ame_fused pairs the column tiles: n / 16 must be even")
    tb, ab = 256 * nbytes(in_fmt), 256 * nbytes(acc_fmt)
    ii, iif = x.ii(in_fmt), x.ii(acc_fmt)
    P = _listing("nbody_ame_fp32_fp32")
    e, vs = new_engine(m), VState()
    P.follow(e, P.after(P.find("rdcycle")), A + 0x108, vs)

    def build(i):
        P.run(e, A + 0x108, A + 0x174, vs, last_taken=i < n - 1)
        for _ in range(4):
            e.store()

    e.loop(n, build)
    e.loop(256 // 4, lambda _: (e.alu(2), e.store()))  # the 1 - I mask
    e.alu(4)
    e.tload(2, ab)
    e.tload(1, tb)
    e.tload(3, tb)
    t = _tiles(n)
    save = n <= 256

    def iblock(bi):
        e.alu(3)
        e.tload(0, tb)
        e.azero(2)

        def pair(bp):
            D, B, T, X = (0, 3), (1, 3), (4, 6), (5, 7)
            e.alu(3)
            for c in (0, 1):
                e.azero(D[c])
            for c in (0, 1):
                e.mma(D[c], 0, B[c], ii)
            if save:
                for c in (0, 1):
                    e.alu(2)
                    e.tstore(D[c], ab, acc=True)
            for c in (0, 1):
                ewise(e, acc_fmt, (D[c],), D[c])  # max
                e.alu(1)
                e.tload(X[c], ab)
            for c in (0, 1):
                ewise(e, acc_fmt, (D[c],), D[c])  # + eps^2
                e.alu(1)
                e.tload(B[c], tb)  # the next pair's distance operands
            for c in (0, 1):
                ewise(e, acc_fmt, (D[c],), D[c], tx=True)
            for c in (0, 1):
                amov_to_tile(e, D[c], T[c])
            for _ in range(2):
                for c in (0, 1):
                    ewise(e, acc_fmt, (D[c],), D[c], tiles_read=(T[c],))
            if bi // 2 == bp:  # the diagonal tile is in this pair
                e.alu(2)
                ewise(e, acc_fmt, (D[bi % 2],), D[bi % 2], tiles_read=(2,))
            for c in (0, 1):
                amov_to_tile(e, D[c], T[c])
            for c in (0, 1):
                e.mma(2, T[c], X[c], iif)
            e.alu(2)

        _split(e, t // 2, bi // 2, pair, lambda: pair(bi // 2))
        e.tstore(2, ab, acc=True)

        def out(_):
            e.load(dst="s", fp=True)
            for c in range(3):
                e.load(dst=("x", c), fp=True)
                e.load(dst=("f", c), fp=True)
                e.op("fma", dst=("o", c), srcs=("s", ("x", c), ("f", c)))
                e.store(srcs=(("o", c),), fp=True)
            e.alu(2)

        e.loop(16, out)

    e.loop(t, iblock)
    mx_drain(e, range(x.accs))
    return e


VARIANTS = {
    "scalar": scalar,
    "rvv": rvv,
    "ame": ame,
    "ame_fgemm": ame_fgemm,
    "ame_fused": ame_fused,
}
