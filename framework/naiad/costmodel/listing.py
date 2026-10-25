"""Issue an assembly listing on the :class:`~framework.naiad.costmodel.engine.Engine`.

A loop body is often clearest as the instructions the kernel actually issues
-- which is what a compiler knows about the code it emits. :func:`parse`
turns objdump-style RISC-V text (addresses optional) into :class:`Insn` records,
and :meth:`Program.run` issues a slice of them on an engine, tracking
``vsetvli`` state so every vector instruction's VL, SEW and LMUL (hence its
datapath occupancy, its transfer size and the registers its operands cover)
is the one the hardware would see.

Supported: RV64I/M, F/D (+ Zfbfmin moves), and the RVV 1.0 subset the dwarf
kernels use (unit-stride / strided / indexed / mask / whole-register memory,
integer and FP arithmetic incl. widening / narrowing / FMA / compares /
merges / slides / gathers / reductions / mask logic / moves). Unknown
mnemonics raise, so a listing never silently mis-times.

Annotations after ``#`` in a line: ``!name`` (a store registers memory token
``name``), ``@name`` (a load waits for token ``name``), ``taken`` /
``nottaken`` (a branch's direction; the default is taken for a backward loop
branch and ``j``, not taken otherwise).
"""

import re
from dataclasses import dataclass, field

from framework.naiad.costmodel.engine import ceil_div

_XREGS = (
    {"zero", "ra", "sp", "gp", "tp"}
    | {f"t{i}" for i in range(7)}
    | {f"s{i}" for i in range(12)}
)
_XREGS |= {f"a{i}" for i in range(8)} | {f"x{i}" for i in range(32)}
_FREGS = (
    {f"ft{i}" for i in range(12)}
    | {f"fs{i}" for i in range(12)}
    | {f"fa{i}" for i in range(8)}
)
_FREGS |= {f"f{i}" for i in range(32)}
_SEW = {"e8": 8, "e16": 16, "e32": 32, "e64": 64}
_LMUL = {"m1": 1, "m2": 2, "m4": 4, "m8": 8, "mf2": 0.5, "mf4": 0.25, "mf8": 0.125}

_BRANCHES = {
    "beq",
    "bne",
    "blt",
    "bge",
    "bltu",
    "bgeu",
    "beqz",
    "bnez",
    "blez",
    "bgez",
    "bltz",
    "bgtz",
    "bgt",
    "ble",
    "bgtu",
    "bleu",
}
_JUMPS = {"j", "jal", "jr", "jalr", "ret", "call", "tail"}
_XLOADS = {"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu"}
_FLOADS = {"flw", "fld", "flh"}
_XSTORES = {"sb", "sh", "sw", "sd"}
_FSTORES = {"fsw", "fsd", "fsh"}


@dataclass
class Insn:
    """One parsed instruction."""

    mnem: str
    ops: list
    text: str
    addr: int | None = None
    notes: set = field(default_factory=set)
    tokens_out: tuple = ()
    tokens_in: tuple = ()


_LINE = re.compile(r"^\s*(?:([0-9a-f]+):)?\s*([a-z][a-z0-9_.]*)\s*(.*)$")


def parse(text):
    """Instructions of ``text``: one per line, ``addr: mnemonic operands`` (``addr:`` optional)."""
    out = []
    for raw in text.splitlines():
        line = raw.split("#", 1)
        body, note = line[0].strip(), (line[1] if len(line) > 1 else "")
        if not body or body.endswith(">:"):
            continue
        m = _LINE.match(body)
        if not m:
            raise ValueError(f"cannot parse listing line {raw!r}")
        addr, mnem, rest = m.groups()
        rest = re.sub(r"<[^>]*>", "", rest).strip()
        ops = [o.strip() for o in rest.split(",")] if rest else []
        notes = set(note.split())
        out.append(
            Insn(
                mnem,
                ops,
                f"{mnem} {rest}".strip(),
                int(addr, 16) if addr else None,
                notes,
                tuple(n[1:] for n in notes if n.startswith("!")),
                tuple(n[1:] for n in notes if n.startswith("@")),
            )
        )
    return out


def _mem_base(op):
    m = re.match(r"^(-?(?:0x)?[0-9a-f]*)\((\w+)\)$", op)
    return m.group(2) if m else None


def _vreg(op):
    m = re.match(r"^v(\d+)(?:\.t)?$", op)
    return int(m.group(1)) if m else None


def _is_x(op):
    return op in _XREGS


def _is_f(op):
    return op in _FREGS


class Program:
    """A parsed listing, run by address ranges (``run(e, start, end)``) or whole."""

    def __init__(self, text):
        self.insns = parse(text)
        self.index = {
            ins.addr: i for i, ins in enumerate(self.insns) if ins.addr is not None
        }

    @classmethod
    def load(cls, name):
        """A listing shipped in ``costmodel/listings/<name>.s``."""
        from pathlib import Path

        return cls(
            (Path(__file__).resolve().parent / "listings" / f"{name}.s").read_text()
        )

    def note(self, addr, *notes):
        """Annotate the instruction at ``addr`` (``!token``, ``@token``, ``taken``, ``nottaken``)."""
        ins = self.insns[self.index[addr]]
        ins.notes |= set(notes)
        ins.tokens_out = tuple(n[1:] for n in ins.notes if n.startswith("!"))
        ins.tokens_in = tuple(n[1:] for n in ins.notes if n.startswith("@"))
        return self

    def find(self, pattern, nth=0, after=None, before=None):
        """Address of the ``nth`` instruction (from ``after``, exclusive) whose text matches regex ``pattern``."""
        lo = 0 if after is None else self.index[after] + 1
        hi = len(self.insns) if before is None else self.index[before]
        hits = [ins.addr for ins in self.insns[lo:hi] if re.search(pattern, ins.text)]
        if len(hits) <= nth:
            raise KeyError(f"listing: no match {nth} for {pattern!r}")
        return hits[nth]

    def after(self, addr, k=1):
        """The address ``k`` instructions after ``addr``."""
        return self.insns[self.index[addr] + k].addr

    def target(self, addr):
        """The branch / jump target of the instruction at ``addr``."""
        return _target(self.insns[self.index[addr]].ops[-1])

    def loops(self, jumps=False):
        """``(head, branch)`` address pairs of every backward branch (and, with ``jumps``, backward ``j``)."""
        out = []
        kinds = _BRANCHES | ({"j"} if jumps else set())
        for ins in self.insns:
            if ins.mnem in kinds and ins.addr is not None:
                t = _target(ins.ops[-1])
                if t is not None and t <= ins.addr and t in self.index:
                    out.append((t, ins.addr))
        return out

    def slice(self, start=None, end=None):
        """Instructions from address ``start`` up to and including address ``end``."""
        i = 0 if start is None else self.index[start]
        j = len(self.insns) - 1 if end is None else self.index[end]
        return self.insns[i : j + 1]

    def run(self, e, start=None, end=None, vstate=None, last_taken=None):
        """Issue ``[start, end]`` on engine ``e``; ``vstate`` carries vtype across calls.

        ``last_taken`` overrides the direction of the last instruction when it is a branch.
        """
        vs = vstate if vstate is not None else VState()
        seg = self.slice(start, end)
        for ins in seg[:-1]:
            issue(e, ins, vs)
        if seg:
            issue(e, seg[-1], vs, taken=last_taken)
        return vs

    def follow(self, e, start, stop, vstate, taken=(), limit=10_000):
        """Issue from ``start`` to ``stop`` (an address, exclusive, or a regex), following ``j`` jumps.

        Conditional branches fall through unless their address is in ``taken``
        (then execution continues at the target). For straight-line prologues
        and epilogues between the loops a formula drives explicitly.
        """
        i = self.index[start]
        for _ in range(limit):
            ins = self.insns[i]
            if (isinstance(stop, int) and ins.addr == stop) or (
                isinstance(stop, str) and re.search(stop, ins.text)
            ):
                return
            jump = ins.mnem == "j" or (ins.mnem in _BRANCHES and ins.addr in taken)
            issue(e, ins, vstate, taken=jump if ins.mnem in _BRANCHES else None)
            if jump:
                i = self.index[_target(ins.ops[-1])]
            else:
                i += 1
        raise RuntimeError("listing.follow: no stop within the limit")

    def steady(self, machine, path, iters=24):
        """Cycles per iteration of ``path`` (``[(start, end, last_taken), ...]``) once in steady state."""
        from framework.naiad.costmodel.engine import Engine

        e, vs = Engine(machine), VState()

        def once():
            for a, b, tk in path:
                self.run(e, a, b, vs, last_taken=tk)

        for _ in range(4):
            once()
        t0 = e.t
        for _ in range(iters):
            once()
        return (e.t - t0) / iters

    def count(self, start=None, end=None):
        return len(self.slice(start, end))


@dataclass
class VState:
    """The vtype / vl a listing runs under."""

    sew: int = 32
    lmul: float = 1
    vl: int = 0
    #: The application vector length a ``vsetvli`` with a register AVL sees (None: VLMAX).
    avl: int | None = None

    def nregs(self, emul):
        return max(1, int(emul))


def _vlmax(e, sew, lmul):
    return int(e.m.vector.vlen * lmul // sew)


def _group(base, n):
    return tuple(("v", base + i) for i in range(int(n)))


def issue(e, ins, vs, taken=None):
    """Issue one instruction on ``e`` (``vs`` is the current vtype, updated by vset).

    ``taken`` overrides a branch's direction (else its notes, else backward = taken).
    """
    if e.log is not None:
        e.label = f"{ins.addr:x} {ins.text}" if ins.addr is not None else ins.text
    mn, ops = ins.mnem, ins.ops
    if mn.startswith("mx."):
        return _matrix(e, ins)
    if mn.startswith("v"):
        return _vector(e, ins, vs)
    if mn in _BRANCHES:
        srcs = [o for o in ops[:-1] if _is_x(o) and o != "zero"]
        if taken is not None:
            pass
        elif "taken" in ins.notes:
            taken = True
        elif "nottaken" in ins.notes:
            taken = False
        else:
            taken = (
                ins.addr is not None
                and _target(ops[-1]) is not None
                and _target(ops[-1]) <= ins.addr
            )
        e.branch(taken, srcs)
        return
    if mn in _JUMPS:
        srcs = (
            [o for o in ops if _is_x(o) and o not in ("zero",)]
            if mn in ("jr", "jalr")
            else []
        )
        dst = (
            ops[0]
            if mn in ("jal", "jalr") and len(ops) > 1 and _is_x(ops[0])
            else ("ra" if mn == "call" else None)
        )
        if dst is not None:
            e.ready[dst] = e.t + 1
        e.branch("nottaken" not in ins.notes, srcs)
        return
    if mn in _XLOADS or mn in _FLOADS:
        e.load(ops[0], [_mem_base(ops[1])], after=ins.tokens_in, fp=mn in _FLOADS)
        return
    if mn in _XSTORES or mn in _FSTORES:
        e.store([ops[0], _mem_base(ops[1])], fp=mn in _FSTORES)
        return
    if mn.startswith(("lr.", "sc.", "amo")):
        e.load(ops[0], [o.strip("()") for o in ops[1:]])
        return
    if mn.startswith(
        (
            "csrr",
            "rdcycle",
            "rdinstret",
            "rdtime",
            "frcsr",
            "fscsr",
            "fsrm",
            "frrm",
            "csrw",
            "csrs",
            "csrc",
        )
    ):
        e.op(
            "csr",
            ops[0] if ops and _is_x(ops[0]) and not mn.startswith("csrw") else None,
        )
        return
    if mn.startswith("f"):
        return _fp(e, ins)
    # integer ALU / mul / div
    dst = ops[0] if ops and _is_x(ops[0]) and ops[0] != "zero" else None
    srcs = [o for o in ops[1:] if _is_x(o) and o != "zero"]
    if mn in ("nop", "fence", "fence.i", "wfi", "ecall", "ebreak"):
        e.alu(1)
        return
    if mn.startswith(("mul", "mulh")):
        e.op("mul", dst, srcs)
    elif mn.startswith(("div", "rem")):
        e.op("div", dst, srcs)
    else:
        e.op("alu", dst, srcs)


def _target(op):
    try:
        return int(op, 16)
    except ValueError:
        return None


def _fp(e, ins):
    mn, ops = ins.mnem, ins.ops
    base = mn.split(".")[0]
    regs = [o for o in ops if _is_f(o) or (_is_x(o) and o != "zero")]
    dst, srcs = (regs[0], regs[1:]) if regs else (None, [])
    if base in ("fadd", "fsub"):
        kind = "fadd"
    elif base == "fmul":
        kind = "fmul"
    elif base in ("fmadd", "fmsub", "fnmadd", "fnmsub"):
        kind = "fma"
    elif base == "fdiv":
        kind = "fdiv"
    elif base == "fsqrt":
        kind = "fsqrt"
    elif base in (
        "fmv",
        "fcvt",
        "fsgnj",
        "fsgnjn",
        "fsgnjx",
        "fmin",
        "fmax",
        "feq",
        "flt",
        "fle",
        "fclass",
        "fneg",
        "fabs",
    ):
        kind = "fmisc"
    else:
        raise ValueError(f"listing: unknown FP mnemonic {mn!r}")
    e.op(kind, dst, srcs)


# -- vector ----------------------------------------------------------------------


def _vector(e, ins, vs):
    mn, ops = ins.mnem, ins.ops
    v = e.m.vector
    if mn in ("vsetvli", "vsetivli", "vsetvl"):
        sew = lmul = None
        for o in ops[2:]:
            if o in _SEW:
                sew = _SEW[o]
            elif o in _LMUL:
                lmul = _LMUL[o]
        if sew is not None:
            vs.sew, vs.lmul = sew, lmul
        vmax = _vlmax(e, vs.sew, vs.lmul)
        if mn == "vsetivli":
            vs.vl = min(int(ops[1], 0), vmax)
        elif ops[1] == "zero":
            vs.vl = vmax if ops[0] != "zero" else min(vs.vl or vmax, vmax)
        else:
            vs.vl = vmax if vs.avl is None else max(0, min(vs.avl, vmax))
        e.vset(ops[0] if ops[0] != "zero" else None)
        return
    sew, lmul, vl = vs.sew, vs.lmul, vs.vl
    ng = max(1, int(lmul))
    masked = any(o == "v0.t" for o in ops)
    ops = [o for o in ops if o != "v0.t"]
    mask_src = (("v", 0),) if masked else ()
    xs = [o.strip("()") for o in ops if _is_x(o.strip("()")) and o != "zero"]
    fs = [o for o in ops if _is_f(o)]
    # memory
    m2 = re.match(r"^v(l|s)(seg\d)?(s|ox|ux)?(?:e|ei)(\d+)(ff)?\.v$", mn)
    if mn in ("vlm.v", "vsm.v"):
        nbytes = ceil_div(vl, 8)
        if mn == "vlm.v":
            e.vload(_group(_vreg(ops[0]), 1), nbytes, xs, after=ins.tokens_in)
        else:
            e.vstore(
                _group(_vreg(ops[0]), 1),
                nbytes,
                xs,
                token=ins.tokens_out[0] if ins.tokens_out else None,
            )
        return
    wm = re.match(r"^v(l|s)(\d)r(?:e(\d+))?\.v$", mn)
    if wm:
        nreg = int(wm.group(2))
        nbytes = nreg * v.vlenb
        if wm.group(1) == "l":
            e.vload(_group(_vreg(ops[0]), nreg), nbytes, xs, after=ins.tokens_in)
        else:
            e.vstore(
                _group(_vreg(ops[0]), nreg),
                nbytes,
                xs,
                token=ins.tokens_out[0] if ins.tokens_out else None,
            )
        return
    if m2:
        ls, seg, mode, eew, _ff = m2.groups()
        eew = int(eew)
        nf = int(seg[3:]) if seg else 1
        indexed = mode in ("ox", "ux")
        data_eew = sew if indexed else eew
        nbytes = vl * nf * data_eew // 8
        dregs = max(1, int(lmul * data_eew / sew)) * nf
        elems = vl * nf if (mode or nf > 1) else 0
        vsrcs = ()
        if indexed:
            vsrcs = _group(_vreg(ops[-1]), max(1, int(lmul * eew / sew)))
        if ls == "l":
            if vsrcs:
                # an indexed load waits for its index vector like an arithmetic source
                c = max(e._at(e.vready, vsrcs), 0)
                e.t = max(e.t, c)
            e.vload(
                _group(_vreg(ops[0]), dregs),
                nbytes,
                xs,
                elems=elems,
                after=ins.tokens_in,
            )
        else:
            e.vstore(
                _group(_vreg(ops[0]), dregs) + vsrcs + mask_src,
                nbytes,
                xs,
                elems=elems,
                token=ins.tokens_out[0] if ins.tokens_out else None,
            )
        return
    # whole-register moves
    mr = re.match(r"^vmv(\d)r\.v$", mn)
    if mr:
        n = int(mr.group(1))
        e.vop(_group(_vreg(ops[0]), n), _group(_vreg(ops[1]), n), regs=n)
        return
    vregs = [_vreg(o) for o in ops if _vreg(o) is not None]
    sdst = None
    kind = "arith"
    wide = False
    d_n, s_ns = ng, None  # dst group size; per-source group sizes
    if mn in ("vfmv.f.s", "vmv.x.s"):
        sdst = ops[0]
        e.vop(None, _group(vregs[0], 1), vl, sew, sdst=sdst)
        return
    if mn in ("vfmv.s.f", "vmv.s.x"):
        e.vop(_group(vregs[0], 1), (), vl, sew, xsrcs=xs + fs)
        return
    if mn in ("vcpop.m", "vfirst.m"):
        e.vop(None, _group(vregs[0], 1) + mask_src, vl, sew, kind="red", sdst=ops[0])
        return
    if mn.startswith(("vfred", "vred", "vfwred", "vwred")):
        wide = mn.startswith(("vfwred", "vwred"))
        e.vop(
            _group(vregs[0], 1),
            _group(vregs[1], ng) + _group(vregs[2], 1) + mask_src,
            vl,
            sew,
            wide=wide,
            kind="red",
        )
        return
    if mn.startswith("vm") and mn.split(".")[0] in (
        "vmand",
        "vmnand",
        "vmandn",
        "vmxor",
        "vmor",
        "vmnor",
        "vmorn",
        "vmxnor",
        "vmnot",
        "vmmv",
        "vmclr",
        "vmset",
    ):
        e.vop(
            _group(vregs[0], 1),
            tuple(r for x in vregs[1:] for r in _group(x, 1)),
            vl,
            sew,
        )
        return
    if mn.split(".")[0] in ("vmsbf", "vmsif", "vmsof"):
        e.vop(_group(vregs[0], 1), _group(vregs[1], 1) + mask_src, vl, sew)
        return
    base = mn.split(".")[0]
    is_cmp = base.startswith(("vmf", "vms", "vmadc", "vmsbc")) and base not in (
        "vmsbf",
        "vmsif",
        "vmsof",
    )
    if base.startswith(("vfw", "vw")) or base == "vfwmaccbf16":
        wide = True
        d_n = max(1, int(2 * lmul))
        wide_vs2 = ".w" in mn
        s_ns = [
            max(1, int(2 * lmul)) if (i == 0 and wide_vs2) else ng
            for i in range(len(vregs) - 1)
        ]
        # operand order in asm: vd, vs2, vs1|rs1 (vfwmacc: vd, rs1|vs1, vs2)
    elif base.startswith(("vfn", "vn")) and base not in (
        "vfnmacc",
        "vfnmsac",
        "vfnmadd",
        "vfnmsub",
        "vnot",
        "vneg",
    ):
        wide = True
        s_ns = [max(1, int(2 * lmul))] + [ng] * (len(vregs) - 2)
    if base in ("vfdiv", "vfrdiv", "vfsqrt", "vdiv", "vdivu", "vrem", "vremu"):
        kind = "div"
    if base.startswith(("vfwcvt", "vfncvt", "vfwcvtbf16", "vfncvtbf16")):
        wide = True
    if is_cmp:
        d_n = 1
    if base in ("vid",):
        e.vop(_group(vregs[0], ng), mask_src, vl, sew)
        return
    dst = _group(vregs[0], d_n)
    srcs = []
    for i, r in enumerate(vregs[1:]):
        srcs += _group(r, s_ns[i] if s_ns and i < len(s_ns) else ng)
    if (
        any(t in base for t in ("macc", "msac", "madd", "msub"))
        or base.endswith("bf16")
        and "macc" in base
    ):
        srcs += list(dst)
    if mn.startswith("vmerge") or mn.startswith("vfmerge"):
        srcs += [("v", 0)]
    e.vop(dst, tuple(srcs) + mask_src, vl, sew, wide=wide, kind=kind, xsrcs=xs + fs)


# -- the AME unit ----------------------------------------------------------------


_TABLE = None


def _matrix(e, ins):
    """``mx.<name> field=value ...`` (``extract`` renders matrix words this way)."""
    global _TABLE
    if _TABLE is None:
        from framework.matrix import encodings

        _TABLE = encodings.load_table()
    name = ins.mnem[3:]
    insn = _TABLE.instructions[name]
    f = dict(o.split("=", 1) for o in " ".join(ins.ops).split() if "=" in o)
    x = e.m.ame
    if x is None:
        raise ValueError(f"listing: {name} on a design without an AME unit")
    kind = insn.exec
    xs = [
        f[k] for k in ("base", "stride", "value", "amount") if k in f and f[k] != "zero"
    ]
    fb = {"fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int32": 4, "tf32": 4}
    token = ins.tokens_out[0] if ins.tokens_out else None
    if kind == "tile_load":
        rows, cols = (x.m, x.k) if insn.params["role"] == "a" else (x.k, x.n)
        e.tload(int(f["td"]), rows * cols * fb[f["fmt"]], xs, after=ins.tokens_in)
    elif kind == "tile_store":
        rows, cols = (x.m, x.k) if insn.params["role"] == "a" else (x.k, x.n)
        e.tstore(int(f["ts"]), rows * cols * fb[f["fmt"]], xs, token=token)
    elif kind == "acc_load":
        e.tload(
            int(f["ad"]), x.m * x.n * fb[f["fmt"]], xs, after=ins.tokens_in, acc=True
        )
    elif kind == "acc_store":
        e.tstore(int(f["as"]), x.m * x.n * fb[f["fmt"]], xs, acc=True, token=token)
    elif kind == "acc_zero":
        e.azero(int(f["ad"]))
    elif kind == "tile_shift":
        e.tshift(int(f["td"]), int(f["ts"]))
    elif kind == "mma":
        e.mma(
            int(f["ad"]),
            int(f["ta"]),
            int(f["tb"]),
            x.ii(f["in"]),
            macs=x.m * x.n * x.k,
        )
    elif kind == "acc_scalar":
        e.mop(x.epilogue, 1, accs_read=(int(f["ad"]),), accs_written=(int(f["ad"]),))
    elif kind == "acc_to_tile":
        e.mop(
            ceil_div(x.m, x.acc_to_tile_rows_per_cycle),
            x.acc_to_tile_latency,
            accs_read=(int(f["as"]),),
            tiles_written=(int(f["td"]),),
        )
    elif kind == "tile_to_acc":
        e.mop(
            ceil_div(x.m, x.tile_to_acc_rows_per_cycle),
            x.tile_to_acc_latency,
            tiles_read=(int(f["ts"]),),
            accs_written=(int(f["ad"]),),
        )
    elif kind == "tile_convert":
        e.mop(
            x.epilogue if x.tconv is None else x.tconv,
            1,
            tiles_read=(int(f["ts"]),),
            tiles_written=(int(f["td"]),),
        )
    elif kind == "tile_reduce":
        b = f.get("b", "0") == "1"
        view_rows, view_cols = (x.k, x.n) if b else (x.m, x.k)
        length = view_rows if f.get("cols", "0") == "1" else view_cols
        import math

        passes = (
            max(1, math.ceil(math.log2(length))) if insn.params.get("prefix") else 1
        )
        e.mop(
            passes * ceil_div(view_rows, x.treduce_rows_per_cycle),
            x.treduce_latency,
            tiles_read=(int(f["ts"]),),
            tiles_written=(int(f["td"]),),
        )
    elif kind == "tile_shift2":
        e.mop(
            x.tshift if x.tshift2 is None else x.tshift2,
            1,
            tiles_read=(int(f["ta"]), int(f["tb"])),
            tiles_written=(int(f["td"]),),
            pipe=None,
        )
    elif kind == "config_read":
        e.op("alu", f.get("rd"))
    else:
        raise ValueError(f"listing: no timing rule for matrix op {name} ({kind})")
