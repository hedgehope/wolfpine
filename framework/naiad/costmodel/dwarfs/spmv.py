"""spmv: Y = A X, A in BSR with 16 x 16 blocks (``kernels/naiad/dwarfs/spmv``); size ``(n, density[, nv])``.

The block structure is data: the formula takes its expectation -- ``nnzb =
round(density (n/16)^2)`` blocks spread evenly over the ``n / 16`` block rows,
every block row stored once (empty ones as zeros).

* **scalar**: per block row, row and vector: the row's blocks as 16 multiply-adds
  each (the compiled, fully unrolled block body).
* **rvv**: nv >= 2 vectorises over the right-hand sides, RG rows a group (8, or
  4 for fp64), one X-row load and RG scalar A loads per column of a block; nv = 1
  vectorises over a block row's 16 columns with one reduction per output row.
* **ame**: per block a tile load of the block and one of the 16 X rows it touches
  (the load path, 2 KiB a block at fp32, bounds it) and one mma; per block row
  an astore and an azero (two accumulators alternate), plus, for nv < 16, a
  staging copy of the nv valid columns on the scalar side.
"""

from framework.naiad.costmodel.dwarfs.common import (
    Unsupported,
    mx_drain,
    nbytes,
    need_acc_store,
    need_ame,
    need_vector,
)
from framework.naiad.costmodel.engine import new_engine
from framework.naiad.costmodel.listing import Program, VState

_CACHE = {}


def shape(size):
    n, density = int(size[0]), float(size[1])
    nv = int(size[2]) if len(size) > 2 else 16
    nb = n // 16
    nnzb = max(1, min(nb * nb, round(density * nb * nb)))
    return n, nb, nnzb, nv


def row_blocks(nb, nnzb):
    """Blocks per block row, spread evenly (the expectation of the kernel's random structure)."""
    return [((r + 1) * nnzb) // nb - (r * nnzb) // nb for r in range(nb)]


def runs(rows):
    """``(start, length)`` of each run of consecutive block rows with the same block count."""
    out, i = [], 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j] == rows[i]:
            j += 1
        out.append((i, j - i))
        i = j
    return out


def loop_rows(e, rows, blockrow):
    """``blockrow(i)`` for every block row; only runs of equal rows are extrapolated."""
    for start, length in runs(rows):
        e.loop(length, lambda j, start=start: blockrow(start + j), period=2)


def _block_loop(P):
    """The per-block loop: the shortest loop of the timed region whose body multiply-accumulates."""
    r0 = P.find("rdcycle")
    cands = [
        lp
        for lp in P.loops()
        if lp[0] > r0 and any("macc" in i.mnem for i in P.slice(*lp))
    ]
    return min(cands, key=lambda lp: P.count(*lp))


def _listing(name):
    if name not in _CACHE:
        try:
            _CACHE[name] = Program.load(name)
        except FileNotFoundError:
            raise Unsupported(f"no compiled listing {name}") from None
    return _CACHE[name]


def scalar(m, size, in_fmt, acc_fmt):
    if (in_fmt, acc_fmt) not in (("fp32", "fp32"), ("fp64", "fp64")):
        raise Unsupported("the scalar spmv runs fp32/fp32 and fp64/fp64")
    n, nb, nnzb, nv = shape(size)
    P = _listing("spmv_scalar_fp32_fp32")
    (_, _), (_, _), (_, _), (bh, bbr) = [
        lp for lp in P.loops() if lp[0] > P.find("rdcycle")
    ][:4]
    e, vs = new_engine(m), VState()
    e.alu(12)
    rows = row_blocks(nb, nnzb)

    def blockrow(i):
        k = rows[i]
        e.alu(8)

        def rv(_):
            e.alu(4)
            e.loop(k, lambda p: P.run(e, bh, bbr, vs, last_taken=p < k - 1))
            e.alu(5)
            e.store(("s",), fp=True)

        e.loop(16 * nv, rv)

    loop_rows(e, rows, blockrow)
    return e


def rvv(m, size, in_fmt, acc_fmt):
    v = need_vector(m)
    n, nb, nnzb, nv = shape(size)
    rows = row_blocks(nb, nnzb)
    e, vs = new_engine(m), VState()
    if nv == 1:
        P, groups = _listing(f"spmv_rvv_{in_fmt}_{acc_fmt}_nv1"), 4
    else:
        P = _listing(f"spmv_rvv_{in_fmt}_{acc_fmt}")
        groups = 16 // (4 if acc_fmt == "fp64" else 8)
        vs.avl = nv
    _rvv_rows(e, vs, rows, P, groups)
    vs.avl = None
    del v
    return e


def _rvv_rows(e, vs, rows, P, groups):
    """The compiled block-row loop: a row header, then per group of output rows the accumulators'
    zeroing, the row's blocks (``listings/spmv_rvv_*``) and the reductions / stores.

    nv >= 2: groups of RG rows vectorised over the right-hand sides; nv = 1: groups of 4 rows
    vectorised over a block row's 16 columns, one ``vfredusum`` per output row.
    """
    r0 = P.find("rdcycle")
    bh, bbr = _block_loop(P)
    j0 = P.find(r"^j ", after=r0)  # the rotated row loop is entered past its exit test
    rhead = P.after(j0)
    rj = P.find(r"^j ", after=rhead)
    ghead = P.target(rj)
    gskip = P.find(r"^bge ", after=ghead)  # an empty row skips its blocks
    pre_blocks = P.insns[
        P.index[bh] - 1
    ].addr  # accumulator set-up between the test and the blocks
    gtail = P.target(gskip)
    gend = next(
        i.addr
        for i in P.insns[P.index[gtail] :]
        if i.mnem in ("bgeu", "beqz", "bne", "bltu")
    )
    P.run(e, P.after(r0), j0, vs)

    def blockrow(i):
        k = rows[i]
        P.run(e, rhead if i else P.after(rhead), rj, vs)
        for g in range(groups):
            P.run(e, ghead, gskip, vs, last_taken=k == 0)
            if k:
                if pre_blocks != gskip:
                    P.run(e, P.after(gskip), pre_blocks, vs)
                e.loop(k, lambda p: P.run(e, bh, bbr, vs, last_taken=p < k - 1))
                e.jump()
            P.run(e, gtail, gend, vs, last_taken=g == groups - 1)

    loop_rows(e, rows, blockrow)


def ame(m, size, in_fmt, acc_fmt):
    x = need_ame(m)
    if (x.m, x.n, x.k) != (16, 16, 16):
        raise Unsupported("spmv_ame is written for a 16 x 16 x 16 unit")
    need_acc_store(x)
    n, nb, nnzb, nv = shape(size)
    rows = row_blocks(nb, nnzb)
    tb = 256 * nbytes(in_fmt)
    ab = 256 * nbytes(acc_fmt)
    ii = x.ii(in_fmt)
    e = new_engine(m)
    e.azero(0)
    e.azero(1)
    e.alu(6)
    state = {"buf": 0, "acc": 0}

    def load_block(buf):
        a, b = (0, 1) if buf == 0 else (2, 3)
        e.load("col")
        e.alu(1)
        e.tload(a, tb)
        e.alu(1, dst="xb", srcs=("col",))
        e.alu(2)
        e.tload(b, tb, xsrcs=("xb",))

    def store_row(acc):
        e.alu(10)
        if nv == 16:
            e.tstore(acc, ab, acc=True)
            e.azero(acc)
        else:
            e.tstore(acc, ab, acc=True, token=("stage", acc))
            e.azero(acc)

            # the copy of the nv valid columns, unrolled: loads in batches of 12, then their stores
            left = 16 * nv
            while left > 0:
                k = min(12, left)
                for i in range(k):
                    e.load(("y", i), after=(("stage", acc),), fp=True)
                for i in range(k):
                    e.store((("y", i),), fp=True)
                e.alu(2)
                left -= k
        e.alu(5)

    load_block(0)

    def blockrow(i):
        acc = i % 2
        k = rows[i]

        def block(p):
            cur = state["buf"]
            e.alu(3)
            load_block(1 - cur)  # the next block's tiles (one ahead, across rows)
            e.alu(3)
            a, b = (0, 1) if cur == 0 else (2, 3)
            e.mma(acc, a, b, ii)
            e.alu(4)
            e.branch()
            state["buf"] = 1 - cur

        for p in range(min(k, 2)):
            block(p)
        e.loop(max(0, k - 2), lambda p: block(p), period=2)
        store_row(acc)

    loop_rows(e, rows, blockrow)
    mx_drain(e, range(x.accs))
    return e


VARIANTS = {"scalar": scalar, "rvv": rvv, "ame": ame}
