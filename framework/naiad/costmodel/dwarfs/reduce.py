"""reduce: the sum of n elements (``kernels/naiad/dwarfs/reduce``).

* **scalar**: one running sum; per element ``load, addi, fadd, bne``, the fadd
  waiting for its load: ``max(4, L + 2)`` cycles an element (L = load-to-use).
* **rvv**: four LMUL-4 accumulators, each chunk's load one iteration ahead;
  per 4 * VLMAX elements four adds and four loads, so
  ``max(4 * occ(add), 4 * T(load), ~15 issue)`` cycles, then a short combine.
  16-bit inputs load at LMUL 2 and add by the widening forms.
* **ame**: ``acc = sum_c X_c J`` (one ones tile): per 1024 elements four tile
  loads and four mmas chained in one accumulator (forwarded), so
  ``max(4 * ii, 4 * T(tile load))`` per group; the finish stores the
  accumulator, reloads it as a tile (a memory RAW) and runs one more mma
  (``finish="mma"``), or adds the 16 partials on the scalar side
  (``finish="scalar"``), or moves it with ``amov`` (``"mmov"``), or reduces it
  with ``treduce`` (``"native"``).
"""

from framework.naiad.costmodel.dwarfs.common import (
    Unsupported,
    compute_fmt,
    mx_drain,
    nbytes,
    need_ame,
    need_vector,
)
from framework.naiad.costmodel.engine import new_engine

FINISH_CFLAGS = {
    "-DREDUCE_FINISH_MMA=0": "scalar",
    "-DREDUCE_FINISH_MMOV=1": "mmov",
    "-DREDUCE_FINISH_NATIVE=1": "native",
}


def switches_from_cflags(cflags):
    out = {}
    for c in cflags:
        if c not in FINISH_CFLAGS:
            return None
        out["finish"] = FINISH_CFLAGS[c]
    return out


def scalar(m, n, in_fmt, acc_fmt):
    if (in_fmt, acc_fmt) not in (("fp32", "fp32"), ("fp64", "fp64")):
        raise Unsupported("the scalar reduce runs fp32/fp32 and fp64/fp64")
    e = new_engine(m)
    e.alu(2)

    def body(_):
        e.load("x", fp=True)
        e.alu(1)
        e.op("fadd", "s", ("s", "x"))
        e.branch()

    e.loop(n, body)
    e.store(("s",), fp=True)
    return e


def rvv(m, n, in_fmt, acc_fmt):
    v = need_vector(m)
    cf = compute_fmt(in_fmt, acc_fmt)
    sew = 8 * nbytes(cf)
    vl = v.vlen * 4 // sew
    ib = nbytes(in_fmt)
    narrow = ib < nbytes(cf)
    # 16-bit inputs: vtype e16 (LMUL 2), widening adds occupy the datapath at 2 * 16 bits.
    asew, wide = (8 * ib, True) if narrow else (sew, False)
    e = new_engine(m)
    e.vset()
    e.vop("z", (), vl, sew)
    e.alu(14)
    for k in range(4):
        e.vload(("x", k), vl * ib)
        e.alu(2)
    e.alu(12)
    e.op("mul", "t")
    if narrow and in_fmt == "bf16":
        e.load("one", fp=True)
    for k in range(3):
        e.vop(("acc", k + 1), ("z",), regs=4)
    groups = n // (4 * vl)

    def add(k):
        e.vop(
            ("acc", k),
            (("acc", k), ("x", k)),
            vl,
            asew,
            wide=wide,
            xsrcs=("one",) if narrow else (),
        )

    def body(_):
        # As compiled: vfadd, add, vle, add, add | (vfadd, vle, add) x 3 | branch.
        add(0)
        e.alu(1)
        e.vload(("x", 0), vl * ib)
        e.alu(2)
        for k in (1, 2, 3):
            add(k)
            e.vload(("x", k), vl * ib)
            e.alu(1)
        e.branch()

    e.loop(max(0, groups - 1), body)
    for k in range(4):
        add(k)
    e.alu(4)
    e.vset()
    e.vop(("acc", 0), (("acc", 0), ("acc", 1)), vl, sew)
    e.alu(3)
    e.vop(("acc", 2), (("acc", 2), ("acc", 3)), vl, sew)
    e.vop(("acc", 0), (("acc", 0), ("acc", 2)), vl, sew)
    e.vop("zero", (), 1, sew)
    e.vop("r", (("acc", 0), "zero"), vl, sew, kind="red")
    e.vset()
    e.vstore("r", nbytes(acc_fmt))
    return e


def ame(m, n, in_fmt, acc_fmt, finish="mma"):
    x = need_ame(m)
    if (x.m, x.n, x.k) != (16, 16, 16) or x.tiles < 7:
        raise Unsupported(
            "reduce_ame is written for a 16 x 16 x 16 unit with at least 7 tiles"
        )
    if finish in ("mmov", "native") and "acc_move" not in x.features:
        raise Unsupported("finish mmov/native needs acc_move")
    if finish == "native" and "reduce" not in x.features:
        raise Unsupported("finish native needs reduce")
    ii = x.ii(in_fmt)
    tb = 256 * nbytes(in_fmt)
    ab = 256 * nbytes(acc_fmt)
    e = new_engine(m)
    e.tload(0, tb)
    e.azero(0)
    e.alu(3)

    def body(_):
        # As compiled: addi, tload, addi, tload, addi, tload, tload, 4 mma, 3 ops, branch.
        for t in (1, 2, 3, 4):
            if t < 4:
                e.alu(1)
            e.tload(t, tb)
        for t in (1, 2, 3, 4):
            e.mma(0, t, 0, ii)
        e.alu(3)
        e.branch()

    e.loop(n // 1024, body)
    e.alu(2)
    if finish == "native":
        e.mop(
            16 // x.acc_to_tile_rows_per_cycle,
            x.acc_to_tile_latency,
            accs_read=(0,),
            tiles_written=(5,),
        )
        e.mop(
            16 // x.treduce_rows_per_cycle,
            x.treduce_latency,
            tiles_read=(5,),
            tiles_written=(5,),
        )
        e.tstore(5, ab, token="part")
        e.load("s", after=("part",), fp=True)
    elif finish in ("mma", "mmov"):
        if finish == "mmov":
            e.mop(
                16 // x.acc_to_tile_rows_per_cycle,
                x.acc_to_tile_latency,
                accs_read=(0,),
                tiles_written=(5,),
            )
        else:
            e.tstore(0, ab, acc=True, token="part")
            e.tload(5, ab, after=("part",))
        e.tload(6, ab)
        e.azero(1)
        e.mma(1, 6, 5, x.ii(acc_fmt))
        e.tstore(1, ab, acc=True, token="part2")
        e.load("s", after=("part2",), fp=True)
    else:
        e.tstore(0, ab, acc=True, token="part")
        e.alu(2)
        for i in range(16):
            e.load(("p", i), after=("part",), fp=True)
            e.op("fadd", "s", ("s", ("p", i)))
            e.alu(1)
    e.store(("s",), fp=True)
    mx_drain(e, range(x.accs))
    return e


VARIANTS = {"scalar": scalar, "rvv": rvv, "ame": ame}
