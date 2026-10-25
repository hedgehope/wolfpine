"""apsp: all-pairs shortest paths by Floyd-Warshall (``kernels/naiad/dwarfs/apsp``); size n.

* **scalar**: ``n^3`` relaxations ``d[i][j] = min(d[i][j], d[i][k] + d[k][j])``,
  4-way unrolled: per 4 j two x 4 loads, 4 adds, 4 mins, 4 stores (the compiled
  3-level nest).
* **rvv**: per k, VL-wide j chunks outside, rows inside two at a time
  (``vfadd.vf`` + ``vfmin.vv`` + a store per row): datapath and issue bound.
* **ame**: blocked Floyd-Warshall on ``mma.minplus`` with 16 x 16 blocks: per
  block step k the diagonal block's closure (4 squarings, each an astore the
  next tile load must wait for: a serial store->load chain), the row and column
  panels, then every other block (each an aload, one min-plus mma and an
  astore); ``(n/16)^2 + 3`` mmas per k, load-path bound.
"""

from framework.naiad.costmodel.dwarfs.common import (
    Unsupported,
    mx_drain,
    nbytes,
    need_ame,
    need_vector,
)
from framework.naiad.costmodel.engine import new_engine
from framework.naiad.costmodel.listing import Program, VState

_CACHE = {}


def _listing(name):
    if name not in _CACHE:
        try:
            _CACHE[name] = Program.load(name)
        except FileNotFoundError:
            raise Unsupported(f"no compiled listing {name}") from None
    return _CACHE[name]


def scalar(m, n, in_fmt, acc_fmt):
    if (in_fmt, acc_fmt) not in (("fp32", "fp32"), ("fp64", "fp64")):
        raise Unsupported("the scalar apsp runs fp32/fp32 and fp64/fp64")
    n = int(n)
    P = _listing("apsp_scalar_fp32_fp32")
    r0 = P.find("rdcycle")
    (jh, jb), (ih, ib), (kh, kb) = [lp for lp in P.loops() if lp[0] > r0][:3]
    e, vs = new_engine(m), VState()
    P.follow(e, P.after(r0), kh, vs)
    pre = lambda a, b: P.insns[P.index[b] - 1].addr  # noqa: E731

    def kstep(k):
        P.run(e, kh, pre(kh, ih), vs)

        def istep(i):
            P.run(e, ih, pre(ih, jh), vs)
            e.loop(n // 4, lambda j: P.run(e, jh, jb, vs, last_taken=j < n // 4 - 1))
            P.run(e, P.after(jb), ib, vs, last_taken=i < n - 1)

        e.loop(n, istep)
        P.run(e, P.after(ib), kb, vs, last_taken=k < n - 1)

    e.loop(n, kstep)
    return e


def rvv(m, n, in_fmt, acc_fmt):
    v = need_vector(m)
    n = int(n)
    P = _listing(f"apsp_rvv_{in_fmt}_{acc_fmt}")
    e, vs = new_engine(m), VState()
    r0 = P.find("rdcycle")
    j0 = P.find(r"^j ", after=r0)
    khead = P.target(j0)
    ktail = P.after(j0)
    kbeq = P.find(r"^beq ", after=ktail)
    jk = P.find(r"^j ", after=khead)
    chead = P.target(jk)
    ctail = P.after(jk)
    cbr = P.find(r"^bgeu ", after=ctail)
    jin = P.find(r"^j ", after=chead)
    a0 = P.after(jin)
    abr = P.find(r"^bgeu ", after=a0)
    b0 = P.target(jin)
    bbr = P.find(r"^bltu ", after=b0)
    c0 = P.after(bbr)
    cj = P.find(r"^j ", after=c0)
    sew = 64 if acc_fmt == "fp64" else 32
    vlmax = v.vlen * 4 // sew
    chunks = [min(vlmax, n - j) for j in range(0, n, vlmax)]
    iters = n // 4
    P.follow(e, P.after(r0), j0, vs)
    e.jump()

    def kstep(k):
        P.run(e, khead, jk, vs)
        for ci, vl in enumerate(chunks):
            vs.avl = vl
            if ci:
                P.run(e, chead, jin, vs)
            else:
                P.run(e, chead, jin, vs)

            def it(i):
                P.run(e, b0, bbr, vs, last_taken=False)
                P.run(e, c0, cj, vs)
                P.run(e, a0, abr, vs, last_taken=False)

            e.loop(iters - 1, it)
            P.run(e, b0, bbr, vs, last_taken=True)
            P.run(e, a0, abr, vs, last_taken=True)
            P.run(e, ctail, cbr, vs, last_taken=ci == len(chunks) - 1)
        P.run(e, ktail, kbeq, vs, last_taken=k == n - 1)

    e.loop(n, kstep)
    vs.avl = None
    return e


def ame(m, n, in_fmt, acc_fmt):
    x = need_ame(m)
    if (x.m, x.n, x.k) != (16, 16, 16) or x.tiles < 6:
        raise Unsupported("apsp_ame is written for a 16 x 16 x 16 unit with 6 tiles")
    if "semiring" not in x.features or not x.acc_mem:
        raise Unsupported("apsp_ame needs the semiring feature and aload/astore")
    if in_fmt != acc_fmt:
        raise Unsupported("the tropical mma keeps the distances in one format")
    n = int(n)
    nb = n // 16
    tb = 256 * nbytes(in_fmt)
    ii = x.ii(in_fmt)
    e = new_engine(m)
    e.alu(12)

    def update(d, ta, tb_, bi, bj):
        e.tload(d, tb, after=((bi, bj),), acc=True)
        e.mma(d, ta, tb_, ii)
        e.tstore(d, tb, acc=True, token=(bi, bj))

    def kstep(k):
        e.tload(0, tb, after=((k, k),), acc=True)
        for _ in range(4):
            e.tload(0, tb, after=((k, k),))
            e.mma(0, 0, 0, ii)
            e.tstore(0, tb, acc=True, token=(k, k))
        e.tload(0, tb, after=((k, k),))
        e.alu(10)
        for j in range(nb):
            if j == k:
                continue
            t = 3 if j & 1 else 2
            e.tload(t, tb, after=((k, j),))
            update(j & 1, 0, t, k, j)
            e.alu(7)
        e.alu(6)
        for i in range(nb):
            if i == k:
                continue
            t = 5 if i & 1 else 4
            e.tload(t, tb, after=((i, k),))
            update(i & 1, t, 0, i, k)
            e.alu(7)
        e.alu(12)
        for i in range(nb):
            if i == k:
                continue
            e.tload(1, tb, after=((i, k),))
            e.alu(11)

            def jstep(jj, i=i):
                j = jj if jj < k else jj + 1
                t = 3 if j & 1 else 2
                e.tload(t, tb, after=((k, j),))
                update(j & 1, 1, t, i, j)
                e.alu(8)

            # blocks are independent here, so the j loop's steady state extrapolates
            e.loop(nb - 1, jstep, period=2)
            e.alu(4)

    for k in range(nb):
        kstep(k)
    mx_drain(e, range(x.accs))
    return e


VARIANTS = {"scalar": scalar, "rvv": rvv, "ame": ame}
