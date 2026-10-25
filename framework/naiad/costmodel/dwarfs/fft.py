"""fft: a batch of complex n-point FFTs (``kernels/naiad/dwarfs/fft``); size ``n`` or ``(n, batch)``.

* **scalar**: radix-2, bit-reversed copy then ``log2 n`` stages of ``n / 2``
  butterflies (6 loads/stores pairs, 4 mul/fma, 4 add/sub each), from the
  compiled loops; the bit-reversal's carry loop runs ``trailing_ones(i)`` times
  for element ``i``, so its paths are enumerated exactly.
* **rvv**: the four-step FFT (``n = n1 * n2``): radix-4 (and one radix-2)
  Stockham passes over whole rows of VL elements, a twiddle-and-transpose with
  strided stores (one element a cycle on the LSU), then the column passes.
  Each pass is ``rows x strips`` of loads / complex multiplies / DFT-4 adds
  and stores in the kernel's order; datapath, LSU (strided stores) and
  result latency (no chaining) bound it in turn.
* **ame**: per stage the DFT-16/32 blocks as mmas (4 real mmas per complex
  tile pair, or 3 for ``ame_gauss``) over 16-column slabs, then -- for every
  stage but the last -- a scalar twiddle pass over all ``n * batch`` points in
  the accumulator format, rounding back to the input format (software
  rounding for bf16 / fp16: 2x / 3x the instructions). The scalar pass, not
  the matrix unit, bounds it: ~25 cycles a point.
* **ame_fused**: ``ame``'s mmas with the twiddle as four element-wise
  accumulator ops per output block (the ``ewise`` feature) and the narrowing
  store on the unit; no scalar pass, so the tile loads bound it.
* ``ame_rvv`` (the RVV-twiddle hybrid) is not modelled: it would need its
  compiled listing.
"""

import math

from framework.naiad.costmodel.dwarfs.common import (
    Unsupported,
    amov_to_tile,
    compute_fmt,
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

PLANS = {
    256: (16, 16),
    512: (32, 16),
    1024: (32, 32),
    4096: (16, 16, 16),
    8192: (32, 16, 16),
}
DEFAULT_BATCH = 4
_CACHE = {}


def shape(size):
    if isinstance(size, (tuple, list)):
        n, batch = size
    else:
        n, batch = size, DEFAULT_BATCH
    return int(n), int(batch)


def _listing(name):
    if name not in _CACHE:
        try:
            _CACHE[name] = Program.load(name)
        except FileNotFoundError:
            raise Unsupported(f"no compiled listing {name}") from None
    return _CACHE[name]


def _trailing_ones(i, bits):
    c = 0
    while c < bits and (i >> c) & 1:
        c += 1
    return c


# -- scalar ----------------------------------------------------------------------


def scalar(m, size, in_fmt, acc_fmt):
    if (in_fmt, acc_fmt) not in (("fp32", "fp32"), ("fp64", "fp64")):
        raise Unsupported("the scalar fft runs fp32/fp32 and fp64/fp64")
    n, batch = shape(size)
    lg = int(math.log2(n))
    P = _listing(
        "fft_scalar_fp32_fp32"
    )  # fp64 compiles to the same instruction classes
    e, vs = new_engine(m), VState()
    a = 0x10000000
    P.follow(e, P.after(P.find("rdcycle")), a + 0x110, vs)

    # bit-reversal copy: element body, then either the bit-clear tail or the carry loop
    def bitrev(i):
        c = _trailing_ones(i, lg)
        last = i == n - 1
        P.run(e, a + 0x140, a + 0x16C, vs, last_taken=c == 0)
        if c == 0:
            P.run(e, a + 0x130, a + 0x13C, vs, last_taken=last)
        else:
            P.run(e, a + 0x170, a + 0x180, vs, last_taken=c > 1)
            for k in range(1, c):
                P.run(e, a + 0x174, a + 0x180, vs, last_taken=k < c - 1)
            P.run(e, a + 0x184, a + 0x184, vs)
            P.run(e, a + 0x134, a + 0x13C, vs, last_taken=last)

    def transform(t):
        P.run(e, a + 0x110, a + 0x12C, vs)
        # the path depends on i's trailing ones: replay one period of the counter's pattern
        # (8 elements) and scale, which is exact for its instruction mix
        period = min(n, 8)

        def chunk(j):
            for i in range(j * period, (j + 1) * period):
                bitrev(i % n if i < n - 1 else n - 1)

        e.loop(n // period, chunk)
        P.run(e, a + 0x188, a + 0x190, vs)
        length = 2
        while length <= n:
            half = length // 2
            P.run(e, a + 0x1A0, a + 0x1C4, vs)

            def kloop(k, half=half, length=length):
                P.run(e, a + 0x1D8, a + 0x200, vs)
                inner = n // length
                e.loop(
                    inner,
                    lambda i: P.run(
                        e, a + 0x204, a + 0x258, vs, last_taken=i < inner - 1
                    ),
                )
                P.run(e, a + 0x25C, a + 0x25C, vs)
                P.run(e, a + 0x1C8, a + 0x1D4, vs, last_taken=k == half - 1)

            e.loop(half, kloop)
            P.run(e, a + 0x194, a + 0x19C, vs, last_taken=length == n)
            length *= 2
        P.run(e, a + 0x100, a + 0x10C, vs, last_taken=t == batch - 1)

    e.loop(batch, transform)
    return e


# -- rvv ---------------------------------------------------------------------------


#: Scalar (address) instructions the compiled passes issue per strip (per (g, k) for
#: ``pass4_gk``), counted in ``listings/fft_rvv_fp32_fp32``; first4 by the same ratio.
SCALAR = {"first2": 30, "first4": 44, "pass4": 16, "pass4_gk": 28, "twiddle": 30}


class _Rvv:
    """The RVV kernel's passes as abstract strip bodies (fft_rvv.c)."""

    def __init__(self, m, n, in_fmt, acc_fmt):
        self.e = new_engine(m)
        self.n = n
        cf = compute_fmt(in_fmt, acc_fmt)
        self.sew = 8 * nbytes(cf)
        self.vlmax = m.vector.vlen * 2 // self.sew  # RV_AL 2
        self.narrow = nbytes(in_fmt) < nbytes(cf)
        self.ib, self.ab = nbytes(in_fmt), nbytes(cf)

    def _strips(self, w):
        return [min(self.vlmax, w - c) for c in range(0, w, self.vlmax)]

    def _loads(self, vl, k, first):
        e = self.e
        nb = vl * (self.ib if first else self.ab)
        for i in range(k):
            e.vload(("x", i), nb)
            e.alu(1)

    def _op(self, dst, srcs, vl, first=False):
        if first and self.narrow:
            self.e.vop(dst, srcs, vl, self.sew // 2, wide=True)
        else:
            self.e.vop(dst, srcs, vl, self.sew)

    def _dft4_store(self, vl, first, extra=0):
        e = self.e
        x = [("x", i) for i in range(8)]  # x0r x0i x1r x1i x2r x2i x3r x3i
        a = [("a", i) for i in range(8)]
        # a, b (x0 +- x2), c, d (x1 +- x3), re and im
        pairs = [(0, 4), (1, 5), (0, 4), (1, 5), (2, 6), (3, 7), (2, 6), (3, 7)]
        for i, (p, q) in enumerate(pairs):
            self._op(a[i], (x[p], x[q]), vl, first)
        outs = [(0, 4), (1, 5), (0, 4), (1, 5), (2, 7), (3, 6), (2, 7), (3, 6)]
        for i, (p, q) in enumerate(outs):
            e.vop(("o", i), (a[p], a[q]), vl, self.sew)
            e.alu(1 + extra // 8 + (1 if i < extra % 8 else 0))
            e.vstore(("o", i), vl * self.ab)

    def first4(self, rows, w, first=True):
        e = self.e

        def row(_):
            e.alu(3)
            for vl in self._strips(w):
                e.vset()
                self._loads(vl, 8, first)
                e.vset()
                self._dft4_store(vl, first, SCALAR["first4"] - 16)
                e.branch()

        e.loop(rows // 4, row)

    def first2(self, rows, w, first=True):
        e = self.e

        def row(_):
            e.alu(3)
            for vl in self._strips(w):
                e.vset()
                self._loads(vl, 4, first)
                e.vset()
                extra = SCALAR["first2"] - 8
                for i, (p, q) in enumerate([(0, 2), (1, 3), (0, 2), (1, 3)]):
                    self._op(("o", i), (("x", p), ("x", q)), vl, first)
                    e.alu(1 + extra // 4 + (1 if i < extra % 4 else 0))
                    e.vstore(("o", i), vl * self.ab)
                e.vset()
                e.branch()

        e.loop(rows // 2, row)

    def pass4(self, rows, w, ns):
        e = self.e

        def gk(i):
            k = i % ns
            e.alu(SCALAR["pass4_gk"] // 2)
            for _ in range(6):
                e.load(("w", _), fp=True)
            e.alu(SCALAR["pass4_gk"] - SCALAR["pass4_gk"] // 2)
            e.vset()
            for vl in self._strips(w):
                e.vset()
                self._loads(vl, 8, False)
                if k:
                    for j in (1, 2, 3):
                        xr, xi = ("x", 2 * j), ("x", 2 * j + 1)
                        e.vop(("t", 0), (xr,), vl, self.sew, xsrcs=(("w", 2 * j - 2),))
                        e.vop(
                            ("t", 0),
                            (("t", 0), xi),
                            vl,
                            self.sew,
                            xsrcs=(("w", 2 * j - 1),),
                        )
                        e.vop(("t", 1), (xi,), vl, self.sew, xsrcs=(("w", 2 * j - 2),))
                        e.vop(
                            xi, (("t", 1), xr), vl, self.sew, xsrcs=(("w", 2 * j - 1),)
                        )
                        e.vop(
                            xr,
                            (("t", 0),),
                            regs=max(1, vl * self.sew // e.m.vector.vlen),
                        )
                self._dft4_store(vl, False, SCALAR["pass4"] - 16)
                e.branch()

        e.loop((rows // (4 * ns)) * ns, gk, period=ns if ns <= 4 else 1)

    def rest4(self, rows, w, ns0):
        ns = ns0
        while ns < rows:
            self.pass4(rows, w, ns)
            ns *= 4

    def twiddle_transpose(self, n1, n2):
        e = self.e

        def row(_):
            e.alu(3)
            for vl in self._strips(n2):
                e.vset()
                self._loads(vl, 4, False)
                e.vset()
                e.vop("tr", (("x", 0), ("x", 2)), vl, self.sew)
                e.vop("tr", ("tr", ("x", 3), ("x", 1)), vl, self.sew)
                e.vop("ti", (("x", 1), ("x", 2)), vl, self.sew)
                e.vop("ti", ("ti", ("x", 3), ("x", 0)), vl, self.sew)
                e.alu((SCALAR["twiddle"] - 4) // 2)
                e.vstore("tr", vl * self.ab, elems=vl)
                e.alu((SCALAR["twiddle"] - 4) - (SCALAR["twiddle"] - 4) // 2)
                e.vstore("ti", vl * self.ab, elems=vl)
                e.branch()

        e.loop(n1, row)


def _first_radix(r):
    return 2 if int(math.log2(r)) % 2 else 4


def rvv(m, size, in_fmt, acc_fmt):
    need_vector(m)
    n, batch = shape(size)
    if n not in PLANS:
        raise Unsupported(f"no fft plan for n={n}")
    n1 = PLANS[n][0]
    n2 = n // n1
    k = _Rvv(m, n, in_fmt, acc_fmt)
    e = k.e
    e.alu(10)

    def transform(_):
        e.alu(6)
        if _first_radix(n1) == 2:
            k.first2(n1, n2, first=True)
        else:
            k.first4(n1, n2, first=True)
        k.rest4(n1, n2, _first_radix(n1))
        k.twiddle_transpose(n1, n2)
        if _first_radix(n2) == 2:
            k.first2(n2, n1, first=False)
        else:
            k.first4(n2, n1, first=False)
        k.rest4(n2, n1, _first_radix(n2))

    e.loop(batch, transform)
    return e


# -- ame ---------------------------------------------------------------------------

#: The twiddle pass's per-point path in each format's listing (the normal-number path of
#: the software rounding for bf16 / fp16), as (start, end, taken) segments.
_TWIDDLE_PATH = {
    ("fp32", "fp32"): (("fft_ame_fp32_fp32", ((0x1000041C, 0x10000464, True),)), 12),
    ("fp64", "fp64"): (("fft_ame_fp64_fp64", ((0x1000041C, 0x10000464, True),)), 12),
    ("bf16", "fp32"): (
        (
            "fft_ame_bf16_fp32",
            ((0x100004C0, 0x10000504, True), (0x10000464, 0x100004B0, True)),
        ),
        12,
    ),
    ("fp16", "fp32"): (
        (
            "fft_ame_fp16_fp32",
            (
                (0x100004E4, 0x1000053C, False),
                (0x1000054C, 0x1000054C, True),
                (0x10000568, 0x10000568, True),
                (0x1000057C, 0x100005B8, True),
                (0x100005D0, 0x100005D0, True),
                (0x100005DC, 0x100005DC, True),
                (0x100005F0, 0x10000608, True),
            ),
        ),
        12,
    ),
}


def _ame(m, size, in_fmt, acc_fmt, gauss):
    x = need_ame(m)
    if (x.m, x.n, x.k) != (16, 16, 16) or x.tiles < 6 or x.accs < 3:
        raise Unsupported(
            "fft_ame needs a 16x16x16 unit with 6 tiles and 3 accumulators"
        )
    need_acc_store(x)
    n, batch = shape(size)
    if n not in PLANS:
        raise Unsupported(f"no fft plan for n={n}")
    if (in_fmt, acc_fmt) not in _TWIDDLE_PATH:
        raise Unsupported(f"fft_ame formats {in_fmt}/{acc_fmt}")
    (lname, path), row_overhead = _TWIDDLE_PATH[(in_fmt, acc_fmt)]
    P = _listing(lname)
    vs = VState()
    tb = 256 * nbytes(in_fmt)
    ab = 256 * nbytes(acc_fmt)
    ii = x.ii(in_fmt)
    e = new_engine(m)
    e.alu(8)
    if gauss:
        G = _listing(f"fft_ame_gauss_{in_fmt}_{acc_fmt}")
        gl = [lp for lp in G.loops() if lp[0] > G.find("rdcycle")]
        pre, gtw, fin = gl[0], gl[-2], gl[-1]
        # xs = round(xr + xi): one pass over every point
        e.loop(n * batch, lambda i: G.run(e, *pre, vs, last_taken=True))
        if in_fmt in ("fp32", "fp64"):
            gpath = ((gtw[0], gtw[1], True),)
            gdelta = 0
        else:  # the format's ame point path, plus Gauss's extra work and its third rounding
            gpath = path
            ref32 = _TWIDDLE_PATH[("fp32", "fp32")][0]
            g32 = _listing("fft_ame_gauss_fp32_fp32")
            g32l = [lp for lp in g32.loops() if lp[0] > g32.find("rdcycle")][-2]
            rounding = P.steady(m, path) - _listing(ref32[0]).steady(m, ref32[1])
            gdelta = round(
                g32.steady(m, ((g32l[0], g32l[1], True),))
                - _listing(ref32[0]).steady(m, ref32[1])
                + rounding / 2
            )
    radices = PLANS[n]
    b = 1
    for s, r in enumerate(radices):
        mp = n // (b * r)
        e.alu(6)
        if r == 16:
            for t in (3, 4, 5):
                e.tload(t, tb)

        def slab(_, r=r):
            for _kb in range(r // 16):
                e.azero(0)
                e.azero(1)
                if gauss:
                    e.azero(2)
                e.alu(2)
                for _kk in range(r // 16):
                    e.alu(2)
                    e.tload(0, tb)
                    e.tload(1, tb)
                    if gauss:
                        e.tload(2, tb)
                    if r != 16:
                        e.alu(2)
                        for t in (3, 4, 5):
                            e.tload(t, tb)
                    if gauss:
                        e.mma(0, 3, 0, ii)
                        e.mma(1, 4, 1, ii)
                        e.mma(2, 5, 2, ii)
                    else:
                        e.mma(0, 3, 0, ii)
                        e.mma(0, 5, 1, ii)
                        e.mma(1, 4, 0, ii)
                        e.mma(1, 3, 1, ii)
                e.alu(3)
                e.tstore(0, ab, acc=True)
                e.tstore(1, ab, acc=True)
                if gauss:
                    e.tstore(2, ab, acc=True)
                e.alu(2)

        slabs = b * max(1, mp // 16) if mp >= 16 else max(1, b // 16)
        e.loop(batch * slabs, slab)
        if s < len(radices) - 1:
            # twiddle: batch x r x b rows of mp points
            rows = batch * r * b

            def row(_):
                e.alu(row_overhead)

                def point(i):
                    if gauss and not gdelta:
                        G.run(e, gpath[0][0], gpath[0][1], vs, last_taken=i < mp - 1)
                        return
                    for st, en, tk in path:
                        P.run(
                            e,
                            st,
                            en,
                            vs,
                            last_taken=tk
                            if i < mp - 1 or (st, en) != path[-1][:2]
                            else False,
                        )
                    if gauss:
                        e.alu(gdelta)

                e.loop(mp, point)

            e.loop(rows, row)
        b *= r
    if gauss:  # yr = t1 - t2, yi = t3 - t1 - t2
        e.loop(n * batch, lambda i: G.run(e, *fin, vs, last_taken=True))
    mx_drain(e, range(x.accs))
    return e


def ame(m, size, in_fmt, acc_fmt):
    return _ame(m, size, in_fmt, acc_fmt, gauss=False)


def ame_gauss(m, size, in_fmt, acc_fmt):
    return _ame(m, size, in_fmt, acc_fmt, gauss=True)


def ame_fused(m, size, in_fmt, acc_fmt):
    """``ame_fused``: ``ame``'s mmas, the twiddle on the accumulators, no scalar pass.

    Per 16 x 16 output block: the DFT mmas as ``ame``; then, but for the last
    stage, four element-wise ops against the twiddle tiles (loaded once per
    (k block, column block)) and the store to the next stage's planes (an
    astore pair, or amov.t.a + tconv + tstore each for 16-bit inputs). The
    scalar side is loop control only (a few ALU ops per block, from the C).
    """
    x = need_ame(m)
    if (x.m, x.n, x.k) != (16, 16, 16) or x.tiles < 8 or x.accs < 3:
        raise Unsupported(
            "fft_ame_fused needs a 16x16x16 unit with 8 tiles and 3 accumulators"
        )
    if not x.acc_mem:
        raise Unsupported("fft_ame_fused needs aload/astore")
    need_features(x, "ewise")
    n, batch = shape(size)
    if n not in PLANS:
        raise Unsupported(f"no fft plan for n={n}")
    narrow = in_fmt != acc_fmt
    if narrow and not {"acc_move", "convert"} <= set(x.features):
        raise Unsupported("the software-conversion ablation is not modelled")
    tb, ab = 256 * nbytes(in_fmt), 256 * nbytes(acc_fmt)
    ii = x.ii(in_fmt)
    conv_ii = x.epilogue if x.tconv is None else x.tconv
    e = new_engine(m)
    e.alu(8)
    radices = PLANS[n]
    b = 1
    for s, r in enumerate(radices):
        mp = n // (b * r)
        last = s == len(radices) - 1
        e.alu(8)
        if r == 16:
            for t in (3, 4, 5):
                e.tload(t, tb)

        # Scalar counts from the compiled block loop (fp32, n = 1024): moves and
        # address arithmetic around the matrix words, one stack reload per block.
        def block(_, r=r, last=last):
            e.azero(0)
            e.azero(1)
            e.alu(6)
            for _kk in range(r // 16):
                e.tload(0, tb)
                e.tload(1, tb)
                e.alu(1)
                if r != 16:
                    for t in (3, 4, 5):
                        e.tload(t, tb)
                    e.alu(1)
                e.mma(0, 3, 0, ii)
                e.mma(0, 5, 1, ii)
                e.mma(1, 4, 0, ii)
                e.mma(1, 3, 1, ii)
                e.alu(4)
            e.alu(4)
            if last:
                e.tstore(0, ab, acc=True)
                e.tstore(1, ab, acc=True)
                e.alu(4)
                return
            ewise(e, acc_fmt, (0,), 2, tiles_read=(7,))
            ewise(e, acc_fmt, (1, 2), 2, tiles_read=(6,))
            ewise(e, acc_fmt, (0,), 0, tiles_read=(6,))
            ewise(e, acc_fmt, (0, 1), 0, tiles_read=(7,))
            e.load()
            e.alu(5)
            for a in (0, 2):
                if narrow:
                    amov_to_tile(e, a, 2)
                    e.mop(conv_ii, tiles_read=(2,), tiles_written=(2,))
                    e.tstore(2, tb)
                else:
                    e.tstore(a, ab, acc=True)

        if mp >= 16:

            def column(_, last=last):
                e.alu(4)
                if not last:
                    e.tload(6, ab)
                    e.tload(7, ab)
                e.loop(batch * b, block)

            e.loop((r // 16) * (mp // 16), column)
        else:
            e.loop(batch * max(1, b // 16) * (r // 16), block)
        b *= r
    mx_drain(e, range(x.accs))
    return e


VARIANTS = {
    "scalar": scalar,
    "rvv": rvv,
    "ame": ame,
    "ame_gauss": ame_gauss,
    "ame_fused": ame_fused,
}
