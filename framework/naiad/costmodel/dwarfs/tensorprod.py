"""tensorprod: Nekbone's local gradient ur / us / ut of E = 8 elements of p^3 points (``kernels/naiad/dwarfs/tensorprod``).

The three contractions are small GEMMs with p-sized dimensions, so the work is
``3 E p^4`` MACs in shapes far from any unit's sweet spot.

* **scalar**: gemm's 4 x 4 register-blocked ``mxm`` (with its 4 x 1 / 2 x 4 /
  dot-product edges when p is not a multiple of 4): ur as one
  ``E p^2 x p x p`` product, ut as E ``p x p^2 x p`` products, us as ``E p``
  ``p x p x p`` products (the shortest, so the most edge overhead).
* **rvv**: ``rv_mxm`` (gemm's RVV micro-kernel) for ur, ut and us, the latter
  after an axis-swapping copy of u (and one of its result), p-element vectors.
* **ame**: the resident operator ``I_q (x) D`` (q = 16 // p slabs per mma) and
  three streams of independent jobs, each one tile load, an azero, one mma and
  one astore, loads two jobs ahead and stores one behind; the stream bodies are
  the compiled loops (``listings/tensorprod_ame_*``), whose address arithmetic
  -- not the unit -- bounds the stream at the default peak.
"""

import math

from framework.naiad.costmodel.dwarfs.common import (
    Unsupported,
    nbytes,
    need_ame,
    need_vector,
)
from framework.naiad.costmodel.dwarfs.gemm import rvv_mxm, scalar_mxm
from framework.naiad.costmodel.engine import new_engine
from framework.naiad.costmodel.listing import Program

DEFAULT_E = 8
T = 16
_CACHE = {}


def dims(size):
    if isinstance(size, (tuple, list)):
        return int(size[0]), int(size[1])
    return int(size), DEFAULT_E


def layout(p, e):
    q = T // p
    p2 = p * p
    ldt = math.ceil(p2 / T) * T
    groups_t = math.ceil(e / q)
    rows_r = math.ceil(math.ceil(e * p2 / q) / T) * T
    return {
        "q": q,
        "ldt": ldt,
        "jobs_t": groups_t * (ldt // T),
        "jobs_r": rows_r // T,
        "jobs_s": math.ceil(e * p / q),
    }


def _listing(name):
    if name not in _CACHE:
        try:
            _CACHE[name] = Program.load(name)
        except FileNotFoundError:
            raise Unsupported(f"no compiled listing {name}") from None
    return _CACHE[name]


def scalar(m, size, in_fmt, acc_fmt):
    if (in_fmt, acc_fmt) not in (("fp32", "fp32"), ("fp64", "fp64")):
        raise Unsupported("the scalar tensorprod runs fp32/fp32 and fp64/fp64")
    p, E = dims(size)
    q = T // p
    e = new_engine(m)
    e.alu(10)
    if q * p == T or q == 1:
        scalar_mxm(e, E * p * p, p, p)
    else:
        rows = math.ceil(E * p * p / q)
        e.loop(rows, lambda v: (e.alu(6), scalar_mxm(e, q, p, p)))

    def element(_):
        e.alu(6)
        scalar_mxm(e, p, p * p, p)
        e.loop(p, lambda k: (e.alu(4), scalar_mxm(e, p, p, p)))

    e.loop(E, element)
    return e


def rvv(m, size, in_fmt, acc_fmt):
    v = need_vector(m)
    p, E = dims(size)
    q = T // p
    e = new_engine(m)
    e.alu(8)
    # D^T, scalar
    e.loop(p * p, lambda i: (e.load("d"), e.store(("d",)), e.alu(2), e.branch()))
    if q * p == T or q == 1:
        rvv_mxm(e, m, E * p * p, p, p, in_fmt, acc_fmt)
    else:
        e.loop(
            math.ceil(E * p * p / q),
            lambda i: (e.alu(6), rvv_mxm(e, m, q, p, p, in_fmt, acc_fmt)),
        )
    e.loop(E, lambda i: (e.alu(4), rvv_mxm(e, m, p, p * p, p, in_fmt, acc_fmt)))
    ib = nbytes(in_fmt)
    ab = nbytes("fp64" if "fp64" in (in_fmt, acc_fmt) else "fp32")
    vl = p  # p <= 16 <= VLMAX at LMUL 4

    def copy_swap(nbytes_el):
        def row(_):
            e.alu(4)
            for _b in range(p // 4):
                for k in range(4):
                    e.vload(("cp", k), vl * nbytes_el)
                for k in range(4):
                    e.vstore(("cp", k), vl * nbytes_el)
                e.alu(3)
                e.branch()
            for _b in range(p % 4):
                e.vload(("cp", 0), vl * nbytes_el)
                e.vstore(("cp", 0), vl * nbytes_el)
                e.alu(2)
                e.branch()

        e.loop(E * p, row)

    e.vset()
    copy_swap(ib)
    e.loop(E, lambda i: (e.alu(4), rvv_mxm(e, m, p, p * p, p, in_fmt, acc_fmt)))
    copy_swap(ab)
    del v
    return e


def stream_overheads(in_fmt, acc_fmt, p):
    """Scalar instructions per job of the t, r and s streams, from the compiled stream loops.

    Each stream's loop body (entered mid-way, closed by a backward ``j``) holds four
    jobs: 16 matrix instructions plus the address arithmetic, which grows with the
    divisions by ``LDT / 16`` that a non-power-of-two p leaves in stream t.
    """
    P = _listing(f"tensorprod_ame_{in_fmt}_{acc_fmt}_p{p}")
    r0 = P.find("rdcycle")
    out = []
    for a, b in P.loops(jumps=True):
        seg = P.slice(a, b)
        nm = sum(1 for i in seg if i.mnem.startswith("mx."))
        if a > r0 and P.insns[P.index[b]].mnem == "j" and nm == 16:
            out.append((len(seg) - nm) / 4)
    if len(out) != 3:
        raise Unsupported(
            f"tensorprod listing p={p}: expected 3 stream loops, found {len(out)}"
        )
    return out


def ame(m, size, in_fmt, acc_fmt):
    """Three job streams from the C, with each stream's compiled per-job scalar count."""
    x = need_ame(m)
    if (x.m, x.n, x.k) != (T, T, T) or x.tiles < 6 or x.accs < 4:
        raise Unsupported(
            "tensorprod's AME variant needs a 16 x 16 x 16 unit, 6 tiles, 4 accumulators"
        )
    p, E = dims(size)
    try:
        over = stream_overheads(in_fmt, acc_fmt, p)
    except Unsupported:
        over = stream_overheads(
            "fp32", "fp32", min((8, 10, 12, 16), key=lambda q: abs(q - p))
        )
    lay = layout(p, E)
    tb, ab = T * T * nbytes(in_fmt), T * T * nbytes(acc_fmt)
    ii = x.ii(in_fmt)
    e = new_engine(m)
    e.alu(10)
    e.tload(0, tb)
    e.tload(5, tb)

    def stream(jobs, a_operand, k):
        e.alu(4)
        e.tload(1, tb)
        e.tload(2, tb)
        e.alu(4)
        k0 = int(k // 2)
        k1 = int(round(k - k0))

        def job(j):
            s_ = j % 4
            if j + 2 < jobs:
                e.tload(1 + (s_ + 2) % 4, tb)
            e.alu(k0)
            if j < jobs:
                e.azero(s_)
                if a_operand:
                    e.mma(s_, 1 + s_, 5, ii)
                else:
                    e.mma(s_, 0, 1 + s_, ii)
            if j >= 1:
                e.tstore((s_ + 3) % 4, ab, acc=True)
            e.alu(k1)

        e.loop(jobs + 1, job, period=4)

    stream(lay["jobs_t"], False, over[0])
    stream(lay["jobs_r"], True, over[1])
    stream(lay["jobs_s"], False, over[2])
    for a in range(x.accs):
        e.azero(a)
    return e


VARIANTS = {"scalar": scalar, "rvv": rvv, "ame": ame}
