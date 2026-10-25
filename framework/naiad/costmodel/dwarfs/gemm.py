"""gemm: C = A . B, A m x k, B k x n (``kernels/naiad/dwarfs/gemm``).

* **scalar**: a 4 x 4 register block; per k: 8 loads, 16 FMAs, 3 loop ops. The
  first FMA reads the 5th load, so it waits ``load latency - 5`` cycles: the
  loop is load-latency bound at ``27 + max(0, L - 5)`` cycles per 16 MACs.
* **rvv**: ``rvv_mxm.h``, an ``MR x VL`` accumulator block (LMUL 4, MR 5, or 6
  for 16-bit inputs) over VL-wide column panels; per k one B-row vector load,
  MR scalar A loads and MR ``vfmacc.vf`` (each re-preceded by ``vsetvli``, the
  pins' cost). Datapath bound at ``MR * ceil(VL * EEW / datapath)`` cycles per
  step when the issue slots fit under it.
* **ame**: the emitted driver (``framework.naiad.emit.gemm``): the emitter's
  blocking (``bm x bn`` accumulators, double-buffered when the tiles allow),
  per K step ``bm + bn`` tile loads interleaved with ``bm * bn`` mmas, the
  next block's first loads under the current block's stores. Bound by
  ``max(bm*bn*ii, (bm+bn)*T_load)`` per step plus the block's fill and drain.
"""

import math

from framework.naiad.costmodel.dwarfs.common import (
    Unsupported,
    compute_fmt,
    mx_drain,
    nbytes,
    need_ame,
    need_vector,
)
from framework.naiad.costmodel.engine import new_engine


def dims(size):
    if isinstance(size, (tuple, list)):
        return tuple(int(v) for v in size)
    return int(size), int(size), int(size)


def scalar_mxm(e, M, N, K):
    """The scalar 4 x 4 register-blocked C = A . B (gemm's, and tensorprod's ``mxm``), with its edge loops."""

    def block(rows, cols):
        def kstep(_):
            for r in range(rows):
                e.load(("a", r), fp=True)
            for s_ in range(cols):
                e.load(("b", s_), fp=True)
            e.alu(1)
            for r in range(rows):
                for s_ in range(cols):
                    e.op("fma", ("c", r, s_), (("a", r), ("b", s_), ("c", r, s_)))
            e.alu(1)
            e.branch()

        e.alu(6)
        for r in range(rows):
            for s_ in range(cols):
                e.op("fmisc", ("c", r, s_))
        e.loop(K, kstep)
        for r in range(rows):
            e.alu(3 if cols > 1 else 1)
            for s_ in range(cols):
                e.store((("c", r, s_),), fp=True)
        e.alu(4)
        e.branch()

    def dot():  # one output by a dependent FMA chain (the 2 x 1 and 1 x n remainders)
        e.alu(3)

        def kstep(_):
            e.load("a0", fp=True)
            e.load("b0", fp=True)
            e.op("fma", "c0", ("a0", "b0", "c0"))
            e.alu(2)
            e.branch()

        e.loop(K, kstep)
        e.store(("c0",), fp=True)
        e.alu(2)

    def iblock4(_):
        e.alu(8)
        e.loop(N // 4, lambda j: block(4, 4))
        e.loop(N % 4, lambda j: block(4, 1))

    e.loop(M // 4, iblock4)
    if M % 4 >= 2:
        e.loop(N // 4, lambda j: block(2, 4))
        e.loop(2 * (N % 4), lambda j: dot())
    if M % 2:
        e.loop(N, lambda j: dot())


def scalar(m, size, in_fmt, acc_fmt):
    if (in_fmt, acc_fmt) not in (("fp32", "fp32"), ("fp64", "fp64")):
        raise Unsupported("the scalar gemm runs fp32/fp32 and fp64/fp64")
    M, N, K = dims(size)
    e = new_engine(m)
    e.alu(20)
    scalar_mxm(e, M, N, K)
    return e


def _rvv_geometry(m, in_fmt, acc_fmt):
    cf = compute_fmt(in_fmt, acc_fmt)
    sew = 8 * nbytes(cf)
    same = nbytes(in_fmt) == nbytes(cf)
    vl = m.vector.vlen * 4 // sew  # LMUL 4 accumulators
    mr = 5 if same else 6
    return sew, vl, mr, same


def rvv(m, size, in_fmt, acc_fmt):
    need_vector(m)
    M, N, K = dims(size)
    e = new_engine(m)
    e.alu(12)
    rvv_mxm(e, m, M, N, K, in_fmt, acc_fmt)
    return e


def rvv_mxm(e, m, M, N, K, in_fmt, acc_fmt):
    """``rv_mxm`` (rvv_mxm.h): MR x VL accumulator blocks over VL-wide column panels."""
    sew, vlmax, mr_max, same = _rvv_geometry(m, in_fmt, acc_fmt)
    ib = nbytes(in_fmt)
    ob = 2 if acc_fmt == "bf16" else sew // 8

    # 16-bit inputs: vfwmacc / vfwmaccbf16 under vtype e16 (occupancy at EEW = 2 * 16).
    vsew = sew if same else 8 * ib

    def fma(r, buf, s):
        e.vop(
            ("c", r),
            (("c", r), ("B", buf)),
            vl_[0],
            vsew,
            wide=not same,
            xsrcs=((s, r),),
        )

    vl_ = [vlmax]

    def ukernel(mr, vl):
        vl_[0] = vl
        # Prologue: step 0's B row and A scalars (set p), step 1's (set q); step 0 multiplies.
        e.vload(("B", 0), vl * ib)
        for r in range(mr):
            e.load(("p", r), fp=True)
        e.vload(("B", 1), vl * ib)
        for r in range(mr):
            e.vop(("c", r), (("B", 0),), vl, vsew, wide=not same, xsrcs=(("p", r),))
            e.load(("q", r), fp=True)
        e.alu(4)

        # The compiled pair of steps (odd step from B1/q, even step from B0/p), as
        # scheduled: the next loads interleave with this pair's vfmacc.vf, each
        # vfmacc behind a vsetvli (the asm pins make LLVM re-emit it).
        def pair(_):
            e.alu(1)
            e.vset()
            for r in range(mr - 1):
                e.load(("p", r), fp=True)
            e.vload(("B", 0), vl * ib)
            e.load(("p", mr - 1), fp=True)
            e.vset()
            fma(0, 1, "q")
            for r in range(1, mr):
                e.vset()
                e.load(("q", r - 1), fp=True)
                fma(r, 1, "q")
            e.load(("q", mr - 1), fp=True)
            e.vload(("B", 1), vl * ib)
            e.vset()
            e.alu(2)
            fma(0, 0, "p")
            for r in range(1, mr):
                e.vset()
                fma(r, 0, "p")
            e.branch()

        e.loop((K - 2) // 2 if K > 2 else 0, pair)
        left = K - 1 - 2 * ((K - 2) // 2 if K > 2 else 0)
        for s in range(left):
            buf, st = (1, "q") if s % 2 == 0 else (0, "p")
            for r in range(mr):
                e.vset()
                fma(r, buf, st)
        for r in range(mr):
            if acc_fmt == "bf16":
                e.vop(("o", r), (("c", r),), vl, 16, wide=True)
                e.vstore(("o", r), vl * ob)
            else:
                e.vstore(("c", r), vl * ob)
            e.alu(1)
        e.alu(4)
        e.branch()

    for j in range(0, N, vlmax):
        vl = min(vlmax, N - j)
        e.vset()
        e.alu(8)
        blocks = math.ceil(M / mr_max)
        rows = [M // blocks + (1 if b < M % blocks else 0) for b in range(blocks)]
        # balanced rows: blocks of equal size repeat, so loop over the distinct runs
        i = 0
        while i < blocks:
            r = rows[i]
            run = 1
            while i + run < blocks and rows[i + run] == r:
                run += 1

            def block(_, r=r):
                e.alu(4)
                e.op("div", "q", ())
                e.alu(6, srcs=("q",))
                e.load("jt")
                e.alu(1, srcs=("jt",))
                e.jump()
                ukernel(r, vl)

            e.loop(run, block)
            i += run


def _ame_blocking(x):
    from framework.naiad.emit.gemm import choose_blocking

    b = choose_blocking(x.tiles, x.accs)
    return b.bm, b.bn, b.double_buffer


def ame(m, size, in_fmt, acc_fmt):
    x = need_ame(m)
    M, N, K = dims(size)
    bm, bn, db = _ame_blocking(x)
    ii = x.ii(in_fmt)
    ta = x.m * x.k * nbytes(in_fmt)
    tb = x.k * x.n * nbytes(in_fmt)
    ab = x.m * x.n * nbytes(acc_fmt)
    MB, NB = bm * x.m, bn * x.n
    steps = K // x.k
    nblocks = math.ceil(M / MB) * math.ceil(N / NB)
    if M % MB or N % NB or K % x.k:
        # Edge blocks go through zero-padded staging (scalar packing): charged as full blocks
        # plus one scalar copy per padded element, a lower bound on the packing loop.
        pass
    accs = [i * bn + j for i in range(bm) for j in range(bn)]

    def a_tile(buf, i):
        return buf * (bm + bn) + i

    def b_tile(buf, j):
        return buf * (bm + bn) + bm + j

    def loads(buf):
        la = [("A", a_tile(buf, i), ta) for i in range(bm)]
        lb = [("B", b_tile(buf, j), tb) for j in range(bn)]
        out = []
        for i in range(max(bm, bn)):
            if i < bm:
                out.append(la[i])
            if i < bn:
                out.append(lb[i])
        return out

    def mmas(buf):
        return [
            (i * bn + j, a_tile(buf, i), b_tile(buf, j))
            for i in range(bm)
            for j in range(bn)
        ]

    e = new_engine(m)
    # Scalar instruction counts below are the compiled driver's (naiad_gemm_<in>_<acc>_run),
    # read off its issue trace: a one-block problem folds to straight-line code.
    multi = nblocks > 1
    e.alu(37 if multi else 8)

    def do_load(ld):
        e.tload(ld[1], ld[2])

    def do_mma(mm):
        e.mma(mm[0], mm[1], mm[2], ii)

    def interleaved(lds, mms):
        for i in range(max(len(lds), len(mms))):
            if i < len(lds):
                do_load(lds[i])
                e.alu(1)
            if i < len(mms):
                do_mma(mms[i])

    def kbody():
        if db:

            def pair(p):
                last = 2 * p + 2 >= steps
                interleaved(loads(1), mmas(0))
                if not last:
                    e.alu(4)
                    interleaved(loads(0), mmas(1))
                    e.alu(7)
                    e.branch()
                else:
                    e.alu(2)
                    for mm in mmas(1):
                        do_mma(mm)

            e.loop(steps // 2, pair)
            if steps % 2:
                for mm in mmas(0):
                    do_mma(mm)
        else:

            def single(s):
                if s:
                    for ld in loads(0):
                        do_load(ld)
                        e.alu(1)
                for mm in mmas(0):
                    do_mma(mm)
                e.alu(3)
                e.branch()

            e.loop(steps, single)

    def block(bi):
        if bi == 0:
            for ld in loads(0):
                do_load(ld)
                e.alu(1)
        for a in accs:
            e.azero(a)
        e.alu(6)
        kbody()
        e.alu(13 if multi else 1)
        if bi + 1 < nblocks:
            # store_kload0: the next block's step-0 loads between the stores
            lds = loads(0)
            for i in range(max(len(accs), len(lds))):
                if i < len(accs):
                    e.tstore(accs[i], ab, acc=True)
                if i < len(lds):
                    do_load(lds[i])
            e.alu(9)
        else:
            for a in accs:
                e.tstore(a, ab, acc=True)
            e.alu(17 if multi else 1)

    e.loop(nblocks, block)
    mx_drain(e, range(x.accs))
    return e


def ame_mc(m, size, in_fmt, acc_fmt):
    """``ame_mc``: C's 32-row blocks split across the harts, timed barrier to barrier.

    Each hart has its own unit and the dwarfs' L1 has no port limit, so the
    region is the busiest hart's share of row blocks (plus a barrier); on one
    hart it is the ame variant.
    """
    x = need_ame(m)
    M, N, K = dims(size)
    if m.harts <= 1:
        return ame(m, size, in_fmt, acc_fmt)
    bm, _bn, _db = _ame_blocking(x)
    mb = bm * x.m
    blocks = math.ceil(M / mb)
    per = math.ceil(blocks / m.harts)
    e = ame(m, (min(M, per * mb), N, K), in_fmt, acc_fmt)
    e.alu(40)  # the closing barrier (an AMO and a spin on the sense flag)
    return e


def _need_design(m, mode):
    if m.mode != mode or m.design is None:
        raise Unsupported(f"the design has no {mode.upper()} unit")
    return m.design


def ime(m, size, in_fmt, acc_fmt):
    """Zvvm IME: the emitted driver (``framework.naiad.emit.gemm_ime``), costed by the emitter's own
    per-K-step model (the mma unit's occupancy incl. VRF ports against the LSU's tile loads), times
    the K steps of every block, plus each block's C store on the LSU."""
    from framework.naiad.emit import gemm_ime

    d = _need_design(m, "ime")
    M, N, K = dims(size)
    try:
        p = gemm_ime.PairPlan(d, in_fmt, acc_fmt)
    except ValueError as err:
        raise Unsupported(str(err)) from None
    step = p._step_cycles(p.bm, p.bn, p.lmul, p.db)
    blocks = math.ceil(M / p.mb) * math.ceil(N / p.nb)
    steps = math.ceil(K / p.ke)
    c_bytes = p.mb * p.nb * nbytes(acc_fmt)
    bw = m.vector.mem_bytes_per_cycle if m.vector is not None else 32
    e = new_engine(m)
    e.add_cycles(blocks * (steps * step + math.ceil(c_bytes / bw) + 20) + 30)
    return e


def vme(m, size, in_fmt, acc_fmt):
    """Zvt VME: the emitted driver (``framework.naiad.emit.gemm_vme``): its chosen block schedule,
    costed by the emitter's K-loop model, plus the A transpose through the tile state (one row in
    and one column out per TE elements, three issue slots a move: address, TSS, the move)."""
    from framework.naiad.emit import gemm_vme

    d = _need_design(m, "vme")
    M, N, K = dims(size)
    try:
        geo = gemm_vme.Geometry.for_pair(d.matrix, in_fmt, acc_fmt)
        costs = gemm_vme.costs_for(d, geo)
        sched = gemm_vme.choose(geo, costs)
    except ValueError as err:
        raise Unsupported(str(err)) from None
    plan = sched.plan
    steps = max(plan.u, (K // geo.kmax) // plan.u * plan.u)
    blocks = math.ceil(M / plan.mb) * math.ceil(N / plan.nb)
    init_ops = plan.mps + 6
    store_ops = 3 * plan.mps * geo.e + 4 * plan.bm
    nb = steps // plan.u
    if nb <= 3:
        cyc, _ = gemm_vme._simulate(sched, costs, steps, init_ops, store_ops)
    else:
        c2, _ = gemm_vme._simulate(sched, costs, 2 * plan.u, init_ops, store_ops)
        c3, _ = gemm_vme._simulate(sched, costs, 3 * plan.u, init_ops, store_ops)
        cyc = c2 + (nb - 2) * (c3 - c2)
    transpose = 2 * 3 * math.ceil(M * K / geo.e)
    e = new_engine(m)
    e.add_cycles(blocks * cyc + transpose + 30)
    return e


def redundant(m):
    """Variants that are another variant on this design (``choose`` skips them): one hart's ame_mc is ame."""
    return {"ame_mc"} if m.harts <= 1 else set()


VARIANTS = {
    "scalar": scalar,
    "rvv": rvv,
    "ame": ame,
    "ame_mc": ame_mc,
    "ime": ime,
    "vme": vme,
}
