"""scan: the inclusive prefix sum of n elements (``kernels/naiad/dwarfs/scan``).

* **scalar**: ``s += x[i]; y[i] = s``: per element the load, the dependent add
  and the dependent store expose ``L + fp_add`` cycles plus 4 issue slots
  (14 a element on naiad-dwarfs).
* **rvv**: Hillis-Steele over four LMUL-2 chunks at once: ``log2(VLMAX)``
  rounds of slide-up + masked add per chunk, then the carry chain (a
  slide-down, ``vfmv.f.s`` and a scalar add per chunk) and four stores; per
  ``4 * VLMAX`` elements about ``4 * (2 log2 VLMAX + 4) * occ`` datapath cycles,
  partly serialised by the carry chain's latency.
* **ame**: per 256-element tile ``X U + (Ls X) J`` (two mmas, ``Ls X``
  round-tripped through memory: an astore, then a tile load that must wait
  for it), plus the carry: a stride-0 load of the previous tile's last row
  (waiting for that tile's astore) and one more mma; the chain
  astore -> load -> mma -> astore serialises the tiles, so the per-tile time
  is a latency sum, not a throughput bound.

All three issue their kernel's compiled loop bodies (``listings/scan_*``).
"""

import math

from framework.naiad.costmodel.dwarfs.common import (
    Unsupported,
    need_acc_store,
    need_ame,
    need_vector,
)
from framework.naiad.costmodel.engine import new_engine
from framework.naiad.costmodel.listing import Program, VState

SWITCH_CFLAGS = {
    "-DSCAN_EPILOGUE=1": ("epilogue", True),
    "-DSCAN_MMOV=1": ("mmov", True),
    "-DSCAN_PREFIX=1": ("prefix", True),
}


def switches_from_cflags(cflags):
    out = {}
    for c in cflags:
        if c not in SWITCH_CFLAGS:
            return None
        k, v = SWITCH_CFLAGS[c]
        out[k] = v
    return out


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
        raise Unsupported("the scalar scan runs fp32/fp32 and fp64/fp64")
    P = _listing(f"scan_scalar_{in_fmt}_{acc_fmt}")
    e, vs = new_engine(m), VState()
    r0 = P.find("rdcycle")
    head, br = P.loops()[0]
    P.follow(e, P.after(r0), head, vs)
    e.loop(n, lambda i: P.run(e, head, br, vs))
    P.follow(e, P.after(br), "rdcycle", vs)
    return e


def rvv(m, n, in_fmt, acc_fmt):
    v = need_vector(m)
    P = _listing(f"scan_rvv_{in_fmt}_{acc_fmt}")
    acc_bits = 64 if acc_fmt == "fp64" else 32
    vlmax = v.vlen * 2 // acc_bits  # LMUL-2 accumulators
    rounds = max(1, math.ceil(math.log2(vlmax)))
    e, vs = new_engine(m), VState()
    r0 = P.find("rdcycle")
    (mh, mb), (lh, lb) = P.loops()[:2]
    P.follow(e, P.after(r0), mh, vs)
    e.loop(rounds, lambda i: P.run(e, mh, mb, vs))
    P.follow(e, P.after(mb), lh, vs)
    e.loop(n // (4 * vlmax), lambda i: P.run(e, lh, lb, vs))
    P.follow(e, P.after(lb), "rdcycle", vs)
    return e


def ame(m, n, in_fmt, acc_fmt, epilogue=False, mmov=False, prefix=False):
    x = need_ame(m)
    if (x.m, x.n, x.k) != (16, 16, 16):
        raise Unsupported("scan_ame is written for a 16 x 16 x 16 unit")
    if epilogue or mmov or prefix:
        return _ame_switched(m, n, in_fmt, acc_fmt, epilogue, mmov, prefix)
    need_acc_store(x)
    P = _listing(f"scan_ame_{in_fmt}_{acc_fmt}")
    e, vs = new_engine(m), VState()
    r0 = P.find("rdcycle")
    j0 = P.find(r"^j ", after=r0)
    head = P.find(r"^andi ", after=j0)  # the tile loop's entry (the j's target)
    st_y = P.find(r"mx\.astore as=0", after=r0)
    beq = P.find(r"^beq ", after=st_y)
    st_q = P.find(r"mx\.astore as=1", after=head)
    ld_q = P.find(r"mx\.tload\.a td=5", after=st_q)
    beqz = P.find(r"^beqz ", after=head)
    ld_y = P.find(r"mx\.tload\.a td=6", after=beqz)
    jt = P.find(r"^j ", after=ld_y)
    P.note(st_q, "!q").note(ld_q, "@q").note(st_y, "!y").note(ld_y, "@y")
    P.follow(e, P.after(r0), j0, vs)
    e.jump()  # the j into the loop

    def tile(c):
        if c == 0:
            P.run(e, head, beqz, vs, last_taken=True)
        else:
            P.run(e, head, beqz, vs, last_taken=False)
            P.run(e, P.after(beqz), jt, vs)
        P.run(e, st_y, beq, vs, last_taken=c == n // 256 - 1)

    e.loop(n // 256, tile)
    P.follow(e, P.target(beq), "rdcycle", vs)
    return e


def _ame_switched(m, n, in_fmt, acc_fmt, epilogue, mmov, prefix):
    """The feature-switch forms (``SCAN_EPILOGUE`` / ``SCAN_MMOV`` / ``SCAN_PREFIX``), as abstract bodies."""
    x = m.ame
    feats = set(x.features)
    if epilogue and "epilogue" not in feats:
        raise Unsupported("SCAN_EPILOGUE needs the epilogue feature")
    if (mmov or prefix) and "acc_move" not in feats:
        raise Unsupported("SCAN_MMOV / SCAN_PREFIX need acc_move")
    if prefix and not {"reduce", "shift"} <= feats:
        raise Unsupported("SCAN_PREFIX needs reduce and shift")
    from framework.naiad.costmodel.machine import FORMAT_BYTES

    ib, ab = FORMAT_BYTES[in_fmt], FORMAT_BYTES[acc_fmt]
    tb, atb = 256 * ib, 256 * ab
    e = new_engine(m)
    e.alu(6)
    if prefix:
        e.tload(0, atb)
    else:
        e.tload(0, tb)
        e.alu(5)
        e.tload(1, tb)
        e.tload(2, atb)
    if not epilogue:
        e.tload(3, atb)
    e.alu(6)
    ii_in, ii_acc = x.ii(in_fmt), x.ii(acc_fmt)
    a2t = (math.ceil(16 / x.acc_to_tile_rows_per_cycle), x.acc_to_tile_latency)
    t2a = (math.ceil(16 / x.tile_to_acc_rows_per_cycle), x.tile_to_acc_latency)
    red = math.ceil(16 / x.treduce_rows_per_cycle)

    def tile(c):
        e.alu(1)
        e.tload(4, tb)
        e.alu(1)
        if prefix:
            if in_fmt != acc_fmt:
                e.mop(
                    x.epilogue if x.tconv is None else x.tconv,
                    1,
                    tiles_read=(4,),
                    tiles_written=(4,),
                )
            e.mop(4 * red, x.treduce_latency, tiles_read=(4,), tiles_written=(5,))
            e.mop(red, x.treduce_latency, tiles_read=(4,), tiles_written=(2,))
            e.mop(4 * red, x.treduce_latency, tiles_read=(2,), tiles_written=(2,))
            e.tshift(2, 2)
            e.mop(t2a[0], t2a[1], tiles_read=(5,), accs_written=(0,))
            e.mma(0, 2, 0, ii_acc)
        else:
            e.azero(0)
            e.azero(1)
            e.mma(1, 1, 4, ii_in)
            e.alu(1)
            e.mma(0, 4, 0, ii_in)
            if mmov:
                e.mop(a2t[0], a2t[1], accs_read=(1,), tiles_written=(5,))
            else:
                e.tstore(1, atb, acc=True, token="q")
                e.tload(5, atb, after=("q",))
            e.mma(0, 5, 2, ii_acc)
        e.alu(1)
        if c:
            if epilogue:
                e.load("carry", after=("y",), fp=True)
                e.alu(1)
                e.mop(x.epilogue, 1, accs_read=(0,), accs_written=(0,))
            else:
                e.alu(1)
                e.tload(6, atb, after=("y",))
                e.mma(0, 6, 3, ii_acc)
        e.alu(1)
        e.tstore(0, atb, acc=True, token="y")
        e.alu(4)
        e.branch()

    e.loop(n // 256, tile)
    for a in range(x.accs):
        e.azero(a)
    return e


VARIANTS = {"scalar": scalar, "rvv": rvv, "ame": ame}
