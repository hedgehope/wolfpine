"""jacobi: 8 sweeps of the 5-point stencil on an n x n grid (``kernels/naiad/dwarfs/jacobi``).

* **scalar**: per interior point 4 loads, 3 adds, 1 multiply, 1 store, in one
  dependent chain (``L + 3 fp_add + fp_mul`` exposed per point).
* **rvv**: VL-wide column strips, two rows at a time: 4 unit-stride loads per
  row (no slides), adds and a multiply on the datapath; issue-, datapath- and
  load-latency bound in turn (the compiled loop, ``listings/jacobi_rvv_*``).
* **ame** (banded form, ``tshift`` on): per 16 x 16 output tile 2 tile loads
  (the right neighbour; one 17-row-stride halo load), 5 mmas, 1 tshift, the
  previous tile's astore and an azero; bound by ``max(5 ii, 2 T_load, ...)``
  per tile. A 16-bit grid is re-rounded between sweeps: bf16 by one identity
  mma per tile into a bf16 accumulator, fp16 by a scalar software rounding
  pass over every point (20 instructions a point: why fp16 is 100x slower).
"""

import math

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

ITERS = 8
SWITCH_CFLAGS = {
    "-DJACOBI_TSHIFT=0": ("tshift", False),
    "-DJACOBI_EPILOGUE=1": ("epilogue", True),
    "-DJACOBI_FORM=0": ("form", "window"),
    "-DJACOBI_MCONV=1": ("mconv", True),
    "-DJACOBI_TSHIFT2=1": ("tshift2", True),
}

_CACHE = {}


def switches_from_cflags(cflags):
    out = {}
    for c in cflags:
        if c not in SWITCH_CFLAGS:
            return None
        k, v = SWITCH_CFLAGS[c]
        out[k] = v
    return out


def _listing(name):
    if name not in _CACHE:
        try:
            _CACHE[name] = Program.load(name)
        except FileNotFoundError:
            raise Unsupported(f"no compiled listing {name}") from None
    return _CACHE[name]


def _check(n):
    if (n - 2) % 16 or n < 18:
        raise Unsupported("jacobi needs n = 16 t + 2")


def scalar(m, n, in_fmt, acc_fmt):
    if (in_fmt, acc_fmt) not in (("fp32", "fp32"), ("fp64", "fp64")):
        raise Unsupported("the scalar jacobi runs fp32/fp32 and fp64/fp64")
    # fp64 compiles to the same instruction classes (fld/fadd.d/...): one listing times both.
    P = _listing("jacobi_scalar_fp32_fp32")
    e, vs = new_engine(m), VState()
    r0 = P.find("rdcycle")
    j0 = P.find(r"^j ", after=r0)
    it_tail = P.target(j0)  # sweep header (li t3, 0)
    sweep_j = P.find(r"^j ", after=it_tail)  # into the row loop
    row_head = P.target(sweep_j)  # mul s3, ... : a row's header
    row_j = P.find(r"^j ", after=row_head)  # into the element loop
    elem = P.target(row_j)  # flw ...
    elem_br = P.find(
        r"^bne ", after=elem
    )  # last sweep? (falls through to the out store)
    elem_tail = P.target(elem_br)  # fsw (next), addi...
    elem_beq = P.find(r"^beq ", after=elem_tail)  # row end
    row_tail = P.target(elem_beq)
    row_beq = P.find(r"^beq ", after=row_tail)
    it_inc = P.target(row_beq)
    out_st = P.after(elem_br)  # fsw to out, j
    P.follow(e, P.after(r0), j0, vs)
    e.jump()
    pts = n - 2

    def sweep(it):
        last = it == ITERS - 1
        P.run(e, it_tail, sweep_j, vs)

        def row(r):
            P.run(e, row_head, row_j, vs)

            def point(c):
                if last:
                    P.run(e, elem, elem_br, vs, last_taken=False)
                    P.run(e, out_st, P.after(out_st), vs)
                    P.run(e, P.after(elem_tail), elem_beq, vs, last_taken=c == pts - 1)
                else:
                    P.run(e, elem, elem_br, vs, last_taken=True)
                    P.run(e, elem_tail, elem_beq, vs, last_taken=c == pts - 1)

            e.loop(pts, point)
            P.run(e, row_tail, row_beq, vs, last_taken=r == pts - 1)

        e.loop(pts, row)
        P.run(e, it_inc, P.after(it_inc, 2), vs, last_taken=last)

    e.loop(ITERS, sweep)
    return e


def rvv(m, n, in_fmt, acc_fmt):
    v = need_vector(m)
    _check(n)
    P = _listing(f"jacobi_rvv_{in_fmt}_{acc_fmt}")
    e, vs = new_engine(m), VState()
    r0 = P.find("rdcycle")
    loops = P.loops()
    (oi_h, oi_b), (oo_h, oo_b) = (
        loops[2],
        loops[3],
    )  # last sweep (STORE_OUT): inner, outer
    (ii_h, ii_b), (io_h, io_b) = loops[4], loops[5]  # other sweeps (STORE_IN)
    j0 = P.find(r"^j ", after=r0)
    dispatch = P.target(j0)  # andi / mul / add / bne -> STORE_IN form
    to_in = P.find(r"^bne ", after=dispatch)
    in_entry = P.target(to_in)
    out_entry = P.after(to_in)
    it_inc = P.find(r"^addi ", after=j0)
    it_beq = P.find(r"^beq ", after=it_inc)
    P.follow(e, P.after(r0), j0, vs)
    e.jump()
    # Column strips at the compute format's LMUL-4 VLMAX; two rows per inner iteration.
    sew = 64 if acc_fmt == "fp64" else 32
    vlmax = v.vlen * 4 // sew
    cols = n - 2
    strips = math.ceil(cols / vlmax)
    pairs = (n - 2) // 2

    def form(entry, oh, ob, ih, ib):
        for s in range(strips):
            vs.avl = cols - s * vlmax
            if s == 0:
                P.follow(e, entry, ih, vs)
            else:
                P.run(e, oh, P.insns[P.index[ih] - 1].addr, vs)
            e.loop(pairs, lambda i: P.run(e, ih, ib, vs, last_taken=i < pairs - 1))
            P.run(e, P.after(ib), ob, vs, last_taken=s < strips - 1)
        vs.avl = None
        e.jump()  # j back to the sweep loop

    def sweep(it):
        last = it == ITERS - 1
        P.run(e, dispatch, to_in, vs, last_taken=not last)
        if last:
            form(out_entry, oo_h, oo_b, oi_h, oi_b)
        else:
            form(in_entry, io_h, io_b, ii_h, ii_b)
        P.run(e, it_inc, it_beq, vs, last_taken=last)

    e.loop(ITERS, sweep)
    P.follow(e, P.target(it_beq), "rdcycle", vs)
    return e


def ame(
    m,
    n,
    in_fmt,
    acc_fmt,
    form="banded",
    tshift=True,
    epilogue=False,
    mconv=False,
    tshift2=False,
):
    """The default banded form (+ its switches) from the kernel's C; scalar counts from its listing."""
    x = need_ame(m)
    _check(n)
    if (x.m, x.n, x.k) != (16, 16, 16) or x.tiles < 8:
        raise Unsupported(
            "jacobi_ame is written for a 16 x 16 x 16 unit with at least 8 tiles"
        )
    feats = set(x.features)
    if form == "banded" and tshift and "shift" not in feats:
        raise Unsupported("the banded form's tshift needs the shift feature")
    if epilogue and "epilogue" not in feats:
        raise Unsupported("JACOBI_EPILOGUE needs the epilogue feature")
    if mconv and not {"acc_move", "convert"} <= feats:
        raise Unsupported("JACOBI_MCONV needs acc_move and convert")
    if tshift2 and (form != "window" or not {"shift2", "shift"} <= feats):
        raise Unsupported("JACOBI_TSHIFT2 needs the window form, shift and shift2")
    need_acc_store(x)
    same = in_fmt == acc_fmt
    ib, ab = nbytes(in_fmt), nbytes(acc_fmt)
    tb, atb = 256 * ib, 256 * ab
    ii, ii_acc = x.ii(in_fmt), x.ii(acc_fmt)
    tt = (n - 2) // 16
    e = new_engine(m)
    e.alu(4)

    def store(a, conv):
        """A finished output tile: astore, or (mconv) amov + tconv + a tile store in the input format."""
        if conv:
            e.mop(
                math.ceil(16 / x.acc_to_tile_rows_per_cycle),
                x.acc_to_tile_latency,
                accs_read=(a,),
                tiles_written=(7,),
            )
            e.mop(
                x.epilogue if x.tconv is None else x.tconv,
                1,
                tiles_read=(7,),
                tiles_written=(7,),
            )
            e.tstore(7, tb)
        else:
            e.tstore(a, atb if not same or conv is None else tb, acc=True)

    def sweep_banded(out_acc, conv):
        sbytes = atb if out_acc else (tb if same else atb)
        e.alu(6)
        for t in (0, 1, 2, 3):
            e.tload(t, tb)
        e.alu(2)
        rot = ((4, 5, 6, 0, 2), (5, 6, 4, 1, 0), (6, 4, 5, 2, 1))

        def row(_):
            e.tload(4, tb)
            e.tload(5, tb)
            e.alu(4)

            def bstep(j):
                L, C, R, A, PA = rot[j % 3]
                e.tload(R, tb)
                e.alu(1)
                if conv and j:
                    store(PA, True)
                e.tload(
                    7, tb
                )  # VLOAD: the halo tile (17-row stride), or the upper window
                if j and not conv:
                    e.tstore(PA, sbytes, acc=True)
                e.alu(3)
                e.azero(A)
                e.mma(A, C, 0, ii)
                e.mma(A, L, 2, ii)
                if tshift:
                    e.mma(A, 0, C, ii)
                    e.tshift(7, 7)
                    e.mma(A, 1, 7, ii)
                else:
                    e.mma(A, 1, 7, ii)
                    e.tload(7, tb)
                    e.mma(A, 1, 7, ii)
                e.mma(A, R, 3, ii)
                if epilogue:
                    e.mop(x.epilogue, 1, accs_read=(A,), accs_written=(A,))
                e.alu(
                    2
                )  # the 3-way unrolled loop averages 3 control instructions a step
                e.branch()

            e.loop(tt, bstep, period=3)
            last_a = rot[(tt - 1) % 3][3]
            if conv:
                store(last_a, True)
            else:
                e.tstore(last_a, sbytes, acc=True)
            e.alu(9)

        e.loop(tt, row)

    def sweep_window(out_acc, conv):
        """JACOBI_FORM 0: four shifted windows (WSTEP), or with tshift2 the windows built on the
        unit from rotating left / centre / right tiles and one halo load (W2STEP)."""
        sbytes = atb if out_acc else (tb if same else atb)
        t2 = x.tshift if x.tshift2 is None else x.tshift2

        def fin(a):
            store(a, True) if conv else e.tstore(a, sbytes, acc=True)

        e.alu(6)
        e.tload(0, tb)
        e.alu(2)

        def row(_):
            e.alu(4)
            if tshift2:
                for t in (1, 2, 3, 4):
                    e.tload(t, tb)
                e.alu(4)
                rot = ((1, 2, 3, 0, 2), (2, 3, 1, 1, 0), (3, 1, 2, 2, 1))

                def w2step(j):
                    L, C, R, A, PA = rot[j % 3]
                    e.tshift(5, 4)
                    for td, ta, tb_ in ((6, C, 4), (4, 5, C), (5, C, R), (7, L, C)):
                        e.mop(
                            t2, 1, tiles_read=(ta, tb_), tiles_written=(td,), pipe=None
                        )
                    e.azero(A)
                    e.mma(A, 0, 4, ii)
                    e.mma(A, 0, 6, ii)
                    if j + 1 < tt:
                        e.alu(1)
                        e.tload(L, tb)
                        e.tload(4, tb)
                    if j:
                        fin(PA)
                    e.mma(A, 7, 0, ii)
                    e.mma(A, 5, 0, ii)
                    if epilogue:
                        e.mop(x.epilogue, 1, accs_read=(A,), accs_written=(A,))
                    e.alu(3)
                    e.branch()

                e.loop(tt, w2step, period=3)
                fin(rot[(tt - 1) % 3][3])
            else:

                def wstep(j):
                    a = j % 2
                    for t in (1, 2, 3, 4):
                        e.tload(t, tb)
                        e.alu(1)
                    if j:
                        fin(1 - a)
                    e.azero(a)
                    e.mma(a, 0, 1, ii)
                    e.mma(a, 0, 2, ii)
                    e.mma(a, 3, 0, ii)
                    e.mma(a, 4, 0, ii)
                    if epilogue:
                        e.mop(x.epilogue, 1, accs_read=(a,), accs_written=(a,))
                    e.alu(3)
                    e.branch()

                e.loop(tt, wstep, period=2)
                fin((tt - 1) % 2)
            e.alu(9)

        e.loop(tt, row)

    sweep = sweep_banded if form == "banded" else sweep_window

    def to_bf16():
        e.alu(4)
        e.tload(0, atb)

        def tile(_):
            e.alu(3)
            e.tload(1, atb)
            e.azero(1)
            e.mma(1, 0, 1, ii_acc)
            e.tstore(1, tb, acc=True)
            e.alu(2)
            e.branch()

        e.loop(tt * tt, tile)

    def to_fp16():
        P = _listing("jacobi_ame_fp16_fp32")
        head, br = P.loops()[
            -1
        ]  # the element loop (rounding, store, pointer bumps, row-end beq)
        beq = P.find(r"^beq ", after=head)
        vs = VState()

        def row(_):
            e.alu(11)

            def point(c):
                P.run(e, P.after(beq), br, vs, last_taken=True)
                P.run(e, head, beq, vs, last_taken=c == n - 3)

            e.loop(n - 2, point)

        e.loop(n - 2, row)

    for it in range(ITERS):
        last = it == ITERS - 1
        e.alu(5)
        if same or last:
            sweep(out_acc=last, conv=False)
        elif mconv:
            sweep(out_acc=False, conv=True)
        else:
            sweep(out_acc=True, conv=False)
            if in_fmt == "bf16":
                to_bf16()
            else:
                to_fp16()
    mx_drain(e, range(x.accs))
    return e


VARIANTS = {"scalar": scalar, "rvv": rvv, "ame": ame}
