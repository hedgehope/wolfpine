"""Random RVV differential testing: legal random instruction streams, Spike vs Naiad.

    tools/naiad/rvv_fuzz.py --seeds 0:200 [-n 48] [--families int,fp,...] [--only REGEX] [-j N]
    tools/naiad/rvv_fuzz.py --seed 17 --emit            # print seed 17's program
    tools/naiad/rvv_fuzz.py --seed 17 --keep 3,9 --emit # a shrunk sub-program
    tools/naiad/rvv_fuzz.py --list-families

Each seed generates one program: an initial state (all 32 vector registers,
the x and f registers it uses, ``frm``/``vxrm``, a 4 KiB data buffer, all from
the seed), a stream of random but **legal** RVV instructions, and an epilogue
that dumps the final architectural state (x1..x31, ``vl``, ``vtype``,
``vstart``, ``vxsat``, ``vxrm``, ``fcsr``, ``vlenb``, f0..f31, v0..v31 and the
buffer) into a signature region. The program is assembled with clang
(the integrated assembler is llvm-mc) and run on Spike, the reference, and on
the device under test, Naiad (``--target naiad``, the default) or a second
Spike (``--target spike --dut-isa ...``, which is how the harness is checked
without Naiad). Final states are compared field by field.

**Legality.** Every instruction is generated against an explicit vtype
(SEW/LMUL/AVL, so vl is known statically): the stream carries one
:class:`Config` per instruction, and the emitter inserts a ``vsetvli`` only
where the config changes. So any sub-sequence of a program is itself a legal
program, which is what makes shrinking sound. The generator respects:

* SEW in {8, 16, 32, 64}, LMUL in {1/8 .. 8}, SEW <= LMUL * ELEN (Spike sets
  vill otherwise); AVL is never in (VLMAX, 2 VLMAX), where vl is
  implementation-defined;
* EMUL = EEW / SEW * LMUL in [1/8, 8] for every operand (widening, narrowing,
  extension, ``vrgatherei16``, indexed and segment accesses; ``nf * EMUL <= 8``);
* group alignment (base register a multiple of EMUL) and groups inside v0..v31;
* overlap: a destination never overlaps a source of a different EEW (a
  conservative superset of the spec's rules), never overlaps any source for
  slide-up, gather, compress, ``viota``, ``vmsbf``-type and segment-indexed
  instructions, and never overlaps ``v0`` when the instruction is masked;
* memory: every access is aligned and inside the buffer (unit, strided with
  negative and zero strides, indexed via per-instruction index tables, segment,
  whole-register, mask and fault-only-first); unordered indexed stores never
  write one address twice;
* tail/mask policy is ``tu, mu`` (agnostic results are implementation-defined);
  unordered FP reductions (``vfredusum``/``vfwredusum``) are generated only when
  the ``fpred_unordered`` family is asked for, for the same reason.

A program that traps on Spike (its ``mtvec`` handler exits with code 0x7FF) is
reported as ``REF``: a generator bug, never a DUT bug.

**Shrinking.** A failing seed is reduced (``--shrink``, the default when a seed
fails) by delta debugging over the instruction stream: remove chunks while the
failure keeps its kind (a state mismatch, or a DUT exception). The result is
printed as ``--seed S --keep i,j,...`` plus its assembly, and written to
``<out>/seed<S>.min.S``.

**Allowlist.** ``--families`` selects instruction families (``--list-families``),
``--ext`` the extensions (``zvfh`` enables SEW=16 FP; ``zvfbfmin``, ``zvfbfwma``),
``--only`` / ``--exclude`` filter mnemonics by regex, so a run can be limited to
what the DUT implements.

Exit status: 0 if every seed passed, 1 otherwise, 2 on a set-up error.
"""

import argparse
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction as F
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VLEN = 256
VLENB = VLEN // 8
ELEN = 64
BUF_SIZE = 4096
DEFAULT_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")
DEFAULT_DESIGN = REPO_ROOT / "framework" / "naiad" / "designs" / "naiad-rvv.yaml"
MARCH = "rv64imafdv_zicsr_zifencei_zfhmin_zfbfmin_zvfh_zvfbfmin_zvfbfwma"
SPIKE_ISA = MARCH + "_zicntr_zvl256b"
TRAP_EXIT = 0x7FF
DEFAULT_MAX_CYCLES = 5_000_000

SEWS = (8, 16, 32, 64)
LMULS = (F(1, 8), F(1, 4), F(1, 2), F(1), F(2), F(4), F(8))

#: x registers the generator never allocates: t0/t1 (vsetvli), t2 (strides,
#: indices, scratch), s0 (buffer base), s1 (access address), and x0.
X_SCRATCH = (5, 6, 7, 8, 9)
X_POOL = tuple(r for r in range(1, 32) if r not in X_SCRATCH)


def lmul_name(lmul):
    return f"m{lmul}" if lmul >= 1 else f"mf{lmul.denominator}"


def nregs(emul):
    """Registers a group of this EMUL occupies (fractional groups use one)."""
    return max(1, int(emul))


def valid_emul(emul):
    return F(1, 8) <= emul <= 8


# -- vtype --------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    sew: int
    lmul: F
    avl: int

    @property
    def vlmax(self):
        return int(self.lmul * VLEN / self.sew)

    @property
    def vl(self):
        return vl_for(self.avl, self.vlmax)

    def vsetvli(self):
        return [
            f"li t0, {self.avl}",
            f"vsetvli t1, t0, e{self.sew}, {lmul_name(self.lmul)}, tu, mu",
        ]


def vl_for(avl, vlmax):
    if avl <= vlmax:
        return avl
    if avl >= 2 * vlmax:
        return vlmax
    raise ValueError(
        f"AVL {avl} is in (VLMAX, 2 VLMAX) = ({vlmax}, {2 * vlmax}): vl is implementation-defined"
    )


def legal_vtype(sew, lmul):
    return sew in SEWS and lmul in LMULS and sew <= lmul * ELEN


def random_avl(rng, vlmax):
    r = rng.random()
    if r < 0.05:
        return 0
    if r < 0.45:
        return vlmax
    if r < 0.55:
        return 2 * vlmax + rng.randrange(0, 40)
    return rng.randint(1, vlmax)


def random_config(rng, sews=SEWS, lmuls=LMULS):
    for _ in range(64):
        sew, lmul = rng.choice(sews), rng.choice(lmuls)
        if legal_vtype(sew, lmul):
            vlmax = int(lmul * VLEN / sew)
            return Config(sew, lmul, random_avl(rng, vlmax))
    return None


# -- items --------------------------------------------------------------------


@dataclass
class Group:
    """A vector register operand, for the legality checks (and the tests)."""

    role: str  # "vd" (vector dest), "md" (mask dest), "vs" (source), "vm" (v0 as mask)
    base: int
    count: int
    eew: int
    emul: F


@dataclass
class Item:
    """One generated instruction (plus any set-up it needs), self-contained under ``cfg``."""

    name: str
    family: str
    lines: list
    cfg: Config | None = None
    #: The vtype state after this item, when it changes it (vset items): a Config,
    #: "vill", or None (unchanged).
    result: object = None
    groups: list = field(default_factory=list)
    masked: bool = False
    #: The destination may overlap no source (slideup, gather, compress, ...).
    vd_excl: bool = False
    #: Extra .data the item needs (index tables): list of lines.
    data: list = field(default_factory=list)
    #: Memory items: :class:`Access` (every element it may touch, buffer-relative).
    access: object = None


@dataclass
class Access:
    store: bool
    #: Byte offsets into the buffer of each element (each field, for segments).
    offsets: list
    size: int
    #: An unordered indexed store, whose element addresses must all differ.
    unordered: bool = False


# -- the op table ---------------------------------------------------------------
#
# Operand roles, in assembly order:
#   vd / vs <mult>   vector dest / source, EEW = SEW * mult, EMUL = LMUL * mult
#   vr <mult>        a single register (reduction scalar, vmv.s.x), EEW = SEW * mult
#   md / ms          mask dest / mask source, one register
#   x / xs / xd      scalar source (pool reg) / small scalar in t2 / scalar dest
#   f / fd           FP scalar source / dest
#   si5 / ui5        5-bit immediate, signed / unsigned
#   v0               the literal v0 carry/merge operand


@dataclass(frozen=True)
class Op:
    name: str
    family: str
    operands: tuple
    sews: tuple = SEWS
    mask: bool = True  # may be masked (v0.t)
    vd_excl: bool = False
    ext: str = "v"
    #: SEWs that need another extension, e.g. {16: "zvfh"}.
    sew_ext: tuple = ()


def _ops():
    ops = []

    def add(name, family, operands, **kw):
        ops.append(Op(name, family, tuple(operands), **kw))

    V, X, I5, U5 = ("vs", 1), ("x",), ("si5",), ("ui5",)
    D = ("vd", 1)
    # Integer arithmetic.
    for n, forms in {
        "vadd": "vxi", "vsub": "vx", "vrsub": "xi", "vand": "vxi", "vor": "vxi", "vxor": "vxi",
        "vsll": "vxu", "vsrl": "vxu", "vsra": "vxu", "vminu": "vx", "vmin": "vx", "vmaxu": "vx",
        "vmax": "vx", "vmul": "vx", "vmulh": "vx", "vmulhu": "vx", "vmulhsu": "vx", "vdivu": "vx",
        "vdiv": "vx", "vremu": "vx", "vrem": "vx",
    }.items():  # fmt: skip
        fam = "int"
        if "v" in forms:
            add(f"{n}.vv", fam, [D, V, V])
        if "x" in forms:
            add(f"{n}.vx", fam, [D, V, X])
        if "i" in forms:
            add(f"{n}.vi", fam, [D, V, I5])
        if "u" in forms:
            add(f"{n}.vi", fam, [D, V, U5])
    for n in ("vmacc", "vnmsac", "vmadd", "vnmsub"):
        add(f"{n}.vv", "int", [D, V, V])
        add(f"{n}.vx", "int", [D, X, V])
    add("vmerge.vvm", "int", [D, V, V, ("v0",)], mask=False)
    add("vmerge.vxm", "int", [D, V, X, ("v0",)], mask=False)
    add("vmerge.vim", "int", [D, V, I5, ("v0",)], mask=False)
    add("vmv.v.v", "int", [D, V], mask=False)
    add("vmv.v.x", "int", [D, X], mask=False)
    add("vmv.v.i", "int", [D, I5], mask=False)
    # Widening / narrowing / extension.
    W = ("vd", 2)
    W2 = ("vs", 2)
    narrow_sews = (8, 16, 32)
    for n in ("vwaddu", "vwadd", "vwsubu", "vwsub"):
        add(f"{n}.vv", "widen", [W, V, V], sews=narrow_sews)
        add(f"{n}.vx", "widen", [W, V, X], sews=narrow_sews)
        add(f"{n}.wv", "widen", [W, W2, V], sews=narrow_sews)
        add(f"{n}.wx", "widen", [W, W2, X], sews=narrow_sews)
    for n in ("vwmul", "vwmulu", "vwmulsu"):
        add(f"{n}.vv", "widen", [W, V, V], sews=narrow_sews)
        add(f"{n}.vx", "widen", [W, V, X], sews=narrow_sews)
    for n in ("vwmaccu", "vwmacc", "vwmaccsu"):
        add(f"{n}.vv", "widen", [W, V, V], sews=narrow_sews)
        add(f"{n}.vx", "widen", [W, X, V], sews=narrow_sews)
    add("vwmaccus.vx", "widen", [W, X, V], sews=narrow_sews)
    for n in ("vnsrl", "vnsra"):
        add(f"{n}.wv", "narrow", [D, W2, V], sews=narrow_sews)
        add(f"{n}.wx", "narrow", [D, W2, X], sews=narrow_sews)
        add(f"{n}.wi", "narrow", [D, W2, U5], sews=narrow_sews)
    for f_, sews in ((2, (16, 32, 64)), (4, (32, 64)), (8, (64,))):
        for n in ("vzext", "vsext"):
            add(f"{n}.vf{f_}", "ext", [D, ("vs", F(1, f_))], sews=sews)
    # Carry.
    for n, forms in (("vadc", "vxi"), ("vsbc", "vx")):
        add(f"{n}.vvm", "carry", [D, V, V, ("v0",)], mask=False)
        add(f"{n}.vxm", "carry", [D, V, X, ("v0",)], mask=False)
        if "i" in forms:
            add(f"{n}.vim", "carry", [D, V, I5, ("v0",)], mask=False)
    M = ("md",)
    for n, forms in (("vmadc", "vxi"), ("vmsbc", "vx")):
        add(f"{n}.vvm", "carry", [M, V, V, ("v0",)], mask=False)
        add(f"{n}.vxm", "carry", [M, V, X, ("v0",)], mask=False)
        add(f"{n}.vv", "carry", [M, V, V], mask=False)
        add(f"{n}.vx", "carry", [M, V, X], mask=False)
        if "i" in forms:
            add(f"{n}.vim", "carry", [M, V, I5, ("v0",)], mask=False)
            add(f"{n}.vi", "carry", [M, V, I5], mask=False)
    # Integer compares.
    for n, forms in {
        "vmseq": "vxi", "vmsne": "vxi", "vmsltu": "vx", "vmslt": "vx", "vmsleu": "vxi",
        "vmsle": "vxi", "vmsgtu": "xi", "vmsgt": "xi",
    }.items():  # fmt: skip
        if "v" in forms:
            add(f"{n}.vv", "cmp", [M, V, V])
        if "x" in forms:
            add(f"{n}.vx", "cmp", [M, V, X])
        if "i" in forms:
            add(f"{n}.vi", "cmp", [M, V, I5])
    # Fixed point.
    for n, forms in {
        "vsaddu": "vxi", "vsadd": "vxi", "vssubu": "vx", "vssub": "vx", "vaaddu": "vx", "vaadd": "vx",
        "vasubu": "vx", "vasub": "vx", "vsmul": "vx", "vssrl": "vxu", "vssra": "vxu",
    }.items():  # fmt: skip
        if "v" in forms:
            add(f"{n}.vv", "fixed", [D, V, V])
        if "x" in forms:
            add(f"{n}.vx", "fixed", [D, V, X])
        if "i" in forms:
            add(f"{n}.vi", "fixed", [D, V, I5])
        if "u" in forms:
            add(f"{n}.vi", "fixed", [D, V, U5])
    for n in ("vnclipu", "vnclip"):
        add(f"{n}.wv", "fixed", [D, W2, V], sews=narrow_sews)
        add(f"{n}.wx", "fixed", [D, W2, X], sews=narrow_sews)
        add(f"{n}.wi", "fixed", [D, W2, U5], sews=narrow_sews)
    # Integer reductions: vd and vs1 are single registers (element 0).
    for n in (
        "vredsum",
        "vredmaxu",
        "vredmax",
        "vredminu",
        "vredmin",
        "vredand",
        "vredor",
        "vredxor",
    ):
        add(f"{n}.vs", "red", [("vr", 1), V, ("vr", 1)])
    for n in ("vwredsumu", "vwredsum"):
        add(f"{n}.vs", "red", [("vr", 2), V, ("vr", 2)], sews=narrow_sews)
    # Mask instructions.
    for n in ("vmand", "vmnand", "vmandn", "vmxor", "vmor", "vmnor", "vmorn", "vmxnor"):
        add(f"{n}.mm", "mask", [M, ("ms",), ("ms",)], mask=False)
    add("vcpop.m", "mask", [("xd",), ("ms",)])
    add("vfirst.m", "mask", [("xd",), ("ms",)])
    for n in ("vmsbf.m", "vmsif.m", "vmsof.m"):
        add(n, "mask", [M, ("ms",)], vd_excl=True)
    add("viota.m", "mask", [D, ("ms",)], vd_excl=True)
    add("vid.v", "mask", [D])
    # Permutation.
    add("vmv.x.s", "perm", [("xd",), ("vr", 1)], mask=False)
    add("vmv.s.x", "perm", [("vr", 1), X], mask=False)
    add("vslideup.vx", "perm", [D, V, ("xs",)], vd_excl=True)
    add("vslideup.vi", "perm", [D, V, U5], vd_excl=True)
    add("vslidedown.vx", "perm", [D, V, ("xs",)])
    add("vslidedown.vi", "perm", [D, V, U5])
    add("vslide1up.vx", "perm", [D, V, X], vd_excl=True)
    add("vslide1down.vx", "perm", [D, V, X])
    add("vrgather.vv", "perm", [D, V, V], vd_excl=True)
    add("vrgather.vx", "perm", [D, V, ("xs",)], vd_excl=True)
    add("vrgather.vi", "perm", [D, V, U5], vd_excl=True)
    add("vrgatherei16.vv", "perm", [D, V, ("vs", "ei16")], vd_excl=True)
    add("vcompress.vm", "perm", [D, V, ("ms",)], mask=False, vd_excl=True)

    # Floating point. SEW 16 needs Zvfh (Zvfhmin for the plain converts).
    FS, FN = (16, 32, 64), (16, 32)
    h = ((16, "zvfh"),)
    Fx = ("f",)
    for n, forms in {
        "vfadd": "vf", "vfsub": "vf", "vfrsub": "f", "vfmul": "vf", "vfdiv": "vf", "vfrdiv": "f",
        "vfmin": "vf", "vfmax": "vf", "vfsgnj": "vf", "vfsgnjn": "vf", "vfsgnjx": "vf",
    }.items():  # fmt: skip
        if "v" in forms:
            add(f"{n}.vv", "fp", [D, V, V], sews=FS, sew_ext=h)
        if "f" in forms:
            add(f"{n}.vf", "fp", [D, V, Fx], sews=FS, sew_ext=h)
    for n in (
        "vfmacc",
        "vfnmacc",
        "vfmsac",
        "vfnmsac",
        "vfmadd",
        "vfnmadd",
        "vfmsub",
        "vfnmsub",
    ):
        add(f"{n}.vv", "fp", [D, V, V], sews=FS, sew_ext=h)
        add(f"{n}.vf", "fp", [D, Fx, V], sews=FS, sew_ext=h)
    for n in ("vfsqrt.v", "vfrsqrt7.v", "vfrec7.v", "vfclass.v"):
        add(n, "fp", [D, V], sews=FS, sew_ext=h)
    add("vfmerge.vfm", "fp", [D, V, Fx, ("v0",)], sews=FS, sew_ext=h, mask=False)
    add("vfmv.v.f", "fp", [D, Fx], sews=FS, sew_ext=h, mask=False)
    add("vfmv.f.s", "fp", [("fd",), ("vr", 1)], sews=FS, sew_ext=h, mask=False)
    add("vfmv.s.f", "fp", [("vr", 1), Fx], sews=FS, sew_ext=h, mask=False)
    add("vfslide1up.vf", "fp", [D, V, Fx], sews=FS, sew_ext=h, vd_excl=True)
    add("vfslide1down.vf", "fp", [D, V, Fx], sews=FS, sew_ext=h)
    for n in ("vmfeq", "vmfne", "vmflt", "vmfle"):
        add(f"{n}.vv", "fpcmp", [M, V, V], sews=FS, sew_ext=h)
        add(f"{n}.vf", "fpcmp", [M, V, Fx], sews=FS, sew_ext=h)
    for n in ("vmfgt", "vmfge"):
        add(f"{n}.vf", "fpcmp", [M, V, Fx], sews=FS, sew_ext=h)
    for n in (
        "vfcvt.xu.f.v",
        "vfcvt.x.f.v",
        "vfcvt.rtz.xu.f.v",
        "vfcvt.rtz.x.f.v",
        "vfcvt.f.xu.v",
        "vfcvt.f.x.v",
    ):
        add(n, "fpcvt", [D, V], sews=FS, sew_ext=h)
    for n in ("vfwcvt.xu.f.v", "vfwcvt.x.f.v", "vfwcvt.rtz.xu.f.v", "vfwcvt.rtz.x.f.v"):
        add(n, "fpcvt", [W, V], sews=FN, sew_ext=h)
    for n in ("vfwcvt.f.xu.v", "vfwcvt.f.x.v"):
        add(n, "fpcvt", [W, V], sews=(8, 16, 32), sew_ext=((8, "zvfh"),))
    add("vfwcvt.f.f.v", "fpcvt", [W, V], sews=FN, sew_ext=h)
    for n in ("vfncvt.xu.f.w", "vfncvt.x.f.w", "vfncvt.rtz.xu.f.w", "vfncvt.rtz.x.f.w"):
        add(n, "fpcvt", [D, W2], sews=(8, 16, 32), sew_ext=((8, "zvfh"),))
    for n in ("vfncvt.f.xu.w", "vfncvt.f.x.w", "vfncvt.f.f.w", "vfncvt.rod.f.f.w"):
        add(n, "fpcvt", [D, W2], sews=FN, sew_ext=h)
    for n in ("vfwadd", "vfwsub"):
        add(f"{n}.vv", "fpwiden", [W, V, V], sews=FN, sew_ext=h)
        add(f"{n}.vf", "fpwiden", [W, V, Fx], sews=FN, sew_ext=h)
        add(f"{n}.wv", "fpwiden", [W, W2, V], sews=FN, sew_ext=h)
        add(f"{n}.wf", "fpwiden", [W, W2, Fx], sews=FN, sew_ext=h)
    add("vfwmul.vv", "fpwiden", [W, V, V], sews=FN, sew_ext=h)
    add("vfwmul.vf", "fpwiden", [W, V, Fx], sews=FN, sew_ext=h)
    for n in ("vfwmacc", "vfwnmacc", "vfwmsac", "vfwnmsac"):
        add(f"{n}.vv", "fpwiden", [W, V, V], sews=FN, sew_ext=h)
        add(f"{n}.vf", "fpwiden", [W, Fx, V], sews=FN, sew_ext=h)
    for n in ("vfredosum", "vfredmax", "vfredmin"):
        add(f"{n}.vs", "fpred", [("vr", 1), V, ("vr", 1)], sews=FS, sew_ext=h)
    add("vfwredosum.vs", "fpred", [("vr", 2), V, ("vr", 2)], sews=FN, sew_ext=h)
    add(
        "vfredusum.vs", "fpred_unordered", [("vr", 1), V, ("vr", 1)], sews=FS, sew_ext=h
    )
    add(
        "vfwredusum.vs",
        "fpred_unordered",
        [("vr", 2), V, ("vr", 2)],
        sews=FN,
        sew_ext=h,
    )
    # bf16.
    add("vfwcvtbf16.f.f.v", "bf16", [W, V], sews=(16,), ext="zvfbfmin")
    add("vfncvtbf16.f.f.w", "bf16", [D, W2], sews=(16,), ext="zvfbfmin")
    add("vfwmaccbf16.vv", "bf16", [W, V, V], sews=(16,), ext="zvfbfwma")
    add("vfwmaccbf16.vf", "bf16", [W, Fx, V], sews=(16,), ext="zvfbfwma")
    return ops


OPS = _ops()

#: Families generated by builders other than the op table.
EXTRA_FAMILIES = {
    "vset": "vsetvli / vsetivli / vsetvl, including vill and the vl-preserving x0, x0 form",
    "csr": "vector/FP CSR reads and writes (vl, vtype, vlenb, vxrm, vxsat, vcsr, frm, fflags)",
    "load": "unit-stride and fault-only-first loads",
    "store": "unit-stride stores",
    "strided": "strided loads and stores (negative and zero strides; no zero-stride stores)",
    "indexed": "ordered/unordered indexed loads and stores (index tables via vl<n>re<eew>.v)",
    "segment": "unit, strided and indexed segment loads and stores (nf 2..8)",
    "whole": "whole-register loads, stores and moves (vl<n>re<eew>.v, vs<n>r.v, vmv<n>r.v)",
    "maskldst": "mask loads and stores (vlm.v, vsm.v)",
}
FAMILY_HELP = {
    "int": "integer arithmetic, multiply/divide, multiply-add, merge, moves",
    "widen": "widening integer arithmetic",
    "narrow": "narrowing shifts",
    "ext": "vzext / vsext",
    "carry": "vadc / vsbc / vmadc / vmsbc",
    "cmp": "integer compares",
    "fixed": "fixed point (saturating, averaging, vsmul, scaling shifts, vnclip)",
    "red": "integer reductions",
    "mask": "mask logical, vcpop, vfirst, vmsbf/vmsif/vmsof, viota, vid",
    "perm": "vmv.x.s/vmv.s.x, slides, gathers, compress",
    "fp": "FP arithmetic, FMA, sqrt/rec7/rsqrt7/class, merge, moves, FP slides",
    "fpcmp": "FP compares",
    "fpcvt": "FP / integer conversions (single, widening, narrowing)",
    "fpwiden": "widening FP arithmetic and FMA",
    "fpred": "ordered FP reductions and FP min/max reductions",
    "fpred_unordered": "vfredusum / vfwredusum (off by default: order is implementation-defined)",
    "bf16": "Zvfbfmin / Zvfbfwma",
    **EXTRA_FAMILIES,
}
DEFAULT_OFF = frozenset({"fpred_unordered"})
ALL_FAMILIES = tuple(FAMILY_HELP)
DEFAULT_EXTS = ("v", "zvfh", "zvfbfmin", "zvfbfwma")


# -- operand allocation ---------------------------------------------------------


def overlaps(a_base, a_count, b_base, b_count):
    return a_base < b_base + b_count and b_base < a_base + a_count


def pick_group(rng, emul, avoid=(), count=None):
    """A random aligned group base for ``emul`` (``count`` registers) avoiding ``avoid`` ranges."""
    n = count if count is not None else nregs(emul)
    align = nregs(emul)
    choices = [
        b
        for b in range(0, 32 - n + 1, align)
        if not any(overlaps(b, n, lo, cnt) for lo, cnt in avoid)
    ]
    return rng.choice(choices) if choices else None


def scalar_value(rng):
    r = rng.random()
    if r < 0.3:
        return rng.getrandbits(64)
    if r < 0.6:
        return rng.randint(-20, 20) & (2**64 - 1)
    return rng.choice(
        [
            0,
            1,
            2**64 - 1,
            2**63,
            2**63 - 1,
            0x80,
            0x7F,
            0xFF,
            0x8000,
            0x7FFF,
            0xFFFF,
            2**31,
            2**31 - 1,
            2**32 - 1,
        ]
    )


class Generator:
    def __init__(
        self,
        seed,
        *,
        count=48,
        families=None,
        exts=DEFAULT_EXTS,
        only=None,
        exclude=None,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.count = count
        self.families = (
            set(families) if families is not None else set(ALL_FAMILIES) - DEFAULT_OFF
        )
        self.exts = set(exts)
        only_re = re.compile(only) if only else None
        exclude_re = re.compile(exclude) if exclude else None

        def wanted(name):
            return (only_re is None or only_re.search(name)) and (
                exclude_re is None or not exclude_re.search(name)
            )

        self.ops = [
            op
            for op in OPS
            if op.family in self.families and op.ext in self.exts and wanted(op.name)
        ]
        self.extra = [f for f in EXTRA_FAMILIES if f in self.families]
        self.wanted = wanted
        self.last_cfg = None
        self.n_tables = 0

    # -- configs --

    def op_sews(self, op):
        out = []
        for s in op.sews:
            need = dict(op.sew_ext).get(s)
            if need is None or need in self.exts:
                out.append(s)
        return out

    def config_for(self, sews, check):
        """A config (reusing the previous one when it fits) for which ``check(cfg)`` holds."""
        rng = self.rng
        if (
            self.last_cfg is not None
            and self.last_cfg.sew in sews
            and rng.random() < 0.6
            and check(self.last_cfg)
        ):
            return self.last_cfg
        for _ in range(48):
            cfg = random_config(rng, sews)
            if cfg is not None and check(cfg):
                return cfg
        return None

    # -- arithmetic ops --

    def gen_op(self, op):
        rng = self.rng
        sews = self.op_sews(op)
        if not sews:
            return None

        def mult_of(m, sew):
            return F(16, sew) if m == "ei16" else F(m)

        def check(cfg):
            for o in op.operands:
                if o[0] in ("vd", "vs", "vr"):
                    m = mult_of(o[1], cfg.sew)
                    eew = cfg.sew * m
                    if eew not in SEWS:
                        return False
                    if o[0] != "vr" and not valid_emul(cfg.lmul * m):
                        return False
                    if op.family in (
                        "fp",
                        "fpcmp",
                        "fpred",
                        "fpred_unordered",
                        "fpcvt",
                        "fpwiden",
                    ):
                        pass
            return True

        cfg = self.config_for(sews, check)
        if cfg is None:
            return None
        masked = op.mask and rng.random() < 0.4
        groups, pre = [], []
        sources = []
        dest_role = op.operands[0][0]
        # Sources first, then the destination with its exclusions.
        chosen = {}
        for i, o in enumerate(op.operands):
            role = o[0]
            if role in ("vs", "vr"):
                m = mult_of(o[1], cfg.sew)
                emul = cfg.lmul * m if role == "vs" else F(1)
                base = pick_group(rng, emul)
                chosen[i] = f"v{base}"
                g = Group("vs", base, nregs(emul), int(cfg.sew * m), emul)
                if i == 0:  # a reduction/vmv.s.x destination
                    g.role = "vd"
                    chosen[i] = None  # allocate below with the dest rules
                else:
                    sources.append(g)
                    groups.append(g)
            elif role == "ms":
                base = rng.randrange(32)
                chosen[i] = f"v{base}"
                g = Group("vs", base, 1, 1, F(1))
                sources.append(g)
                groups.append(g)
            elif role == "x":
                chosen[i] = f"x{rng.choice(X_POOL)}"
            elif role == "xs":
                pre.append(
                    f"li t2, {rng.choice([0, 1, 2, 3, 5, 7, 8, 13, 31, 64, 255, rng.randrange(600)])}"
                )
                chosen[i] = "t2"
            elif role == "f":
                chosen[i] = f"f{rng.randrange(32)}"
            elif role == "si5":
                chosen[i] = str(rng.randint(-16, 15))
            elif role == "ui5":
                chosen[i] = str(rng.randint(0, 31))
            elif role == "v0":
                chosen[i] = "v0"
            elif role == "xd":
                chosen[i] = f"x{rng.choice(X_POOL)}"
            elif role == "fd":
                chosen[i] = f"f{rng.randrange(32)}"
        # Destination.
        o = op.operands[0]
        uses_v0 = masked or any(x[0] == "v0" for x in op.operands)
        if dest_role in ("vd", "md", "vr"):
            if dest_role == "md":
                emul, eew, count = F(1), 1, 1
            elif dest_role == "vr":
                eew, emul, count = int(cfg.sew * F(o[1])), F(1), 1
            else:
                m = F(o[1])
                emul, eew = cfg.lmul * m, int(cfg.sew * m)
                count = nregs(emul)
            avoid = []
            for s in sources:
                if op.vd_excl or s.eew != eew or dest_role == "md":
                    avoid.append((s.base, s.count))
            # v0: never the destination of a masked / v0-reading instruction,
            # except a reduction's scalar result (allowed by the spec, kept simple: avoided too).
            if uses_v0:
                avoid.append((0, 1))
            base = pick_group(rng, emul, avoid, count)
            if base is None:
                return None
            chosen[0] = f"v{base}"
            groups.insert(
                0, Group("md" if dest_role == "md" else "vd", base, count, eew, emul)
            )
        if uses_v0:  # v0.t, or the v0 carry / merge operand
            groups.append(Group("vm", 0, 1, 1, F(1)))
        args = [chosen[i] for i in range(len(op.operands))]
        line = f"{op.name} " + ", ".join(args) + (", v0.t" if masked else "")
        return Item(
            op.name,
            op.family,
            pre + [line],
            cfg,
            groups=groups,
            masked=masked,
            vd_excl=op.vd_excl,
        )

    # -- memory --

    def _base(self, offset):
        return [f"li t2, {offset}", "add s1, s0, t2"]

    def gen_mem(self, family):
        for _ in range(16):
            item = self._gen_mem(family)
            if item is not None and self.wanted(item.name):
                return item
        return None

    def _gen_mem(self, family):
        rng = self.rng
        store = rng.random() < 0.5
        if family == "load":
            store = False
        if family == "store":
            store = True
        if family == "whole":
            return self._gen_whole()
        if family == "maskldst":
            cfg = random_config(rng)
            nbytes = (cfg.vl + 7) // 8
            off = rng.randrange(0, BUF_SIZE - max(nbytes, 1) + 1)
            reg = rng.randrange(32)
            name = "vsm.v" if store else "vlm.v"
            g = Group("vs" if store else "md", reg, 1, 8, F(1))
            acc = Access(store, [off + i for i in range(nbytes)], 1)
            return Item(
                name,
                family,
                self._base(off) + [f"{name} v{reg}, (s1)"],
                cfg,
                groups=[g],
                access=acc,
            )
        if family == "indexed" or (family == "segment" and rng.random() < 0.3):
            return self._gen_indexed(family, store)
        nf = rng.randint(2, 8) if family == "segment" else 1
        strided = family == "strided" or (family == "segment" and rng.random() < 0.4)
        eew = rng.choice(SEWS)

        def check(cfg):
            emul = F(eew, cfg.sew) * cfg.lmul
            return valid_emul(emul) and nregs(emul) * nf <= 8

        cfg = self.config_for(SEWS, check)
        if cfg is None:
            return None
        emul = F(eew, cfg.sew) * cfg.lmul
        eb = eew // 8
        vl = cfg.vl
        masked = rng.random() < 0.4
        count = nregs(emul) * nf
        avoid = [(0, 1)] if masked and not store else []
        base = pick_group(rng, emul, avoid, count)
        if base is None:
            return None
        seg = f"seg{nf}" if nf > 1 else ""
        lines = []
        if strided:
            span_el = nf * eb
            choices = [0, span_el, 2 * span_el, -span_el, -3 * span_el, eb, 64, -64]
            if store:
                choices = [c for c in choices if c != 0 and abs(c) >= span_el]
            stride = rng.choice(choices)
            if vl > 1 and abs(stride) * (vl - 1) + span_el > BUF_SIZE:
                stride = span_el if stride >= 0 else -span_el
            lo = min(0, stride * (vl - 1)) if vl else 0
            hi = max(0, stride * (vl - 1)) + span_el if vl else span_el
            if hi - lo > BUF_SIZE:
                return None
            off = rng.randrange(-lo, BUF_SIZE - hi + 1)
            off -= off % eb
            if off + lo < 0:
                off += eb
            lines += self._base(off) + [f"li t2, {stride}"]
            name = f"vss{seg}e{eew}.v" if store else f"vls{seg}e{eew}.v"
            op_text = f"{name} v{base}, (s1), t2"
        else:
            span = max(vl, 1) * nf * eb
            if span > BUF_SIZE:
                return None
            off = rng.randrange(0, BUF_SIZE - span + 1)
            off -= off % eb
            ff = not store and nf == 1 and rng.random() < 0.25
            if store:
                name = f"vs{seg}e{eew}.v"
            else:
                name = f"vl{seg}e{eew}{'ff' if ff else ''}.v"
            lines += self._base(off)
            op_text = f"{name} v{base}, (s1)"
        lines.append(op_text + (", v0.t" if masked else ""))
        g = Group("vs" if store else "vd", base, count, eew, emul)
        groups = [g] + ([Group("vm", 0, 1, 1, F(1))] if masked else [])
        step = stride if strided else nf * eb
        offsets = [off + i * step + f * eb for i in range(vl) for f in range(nf)]
        acc = Access(store, offsets, eb)
        return Item(name, family, lines, cfg, groups=groups, masked=masked, access=acc)

    def _gen_whole(self):
        rng = self.rng
        kind = rng.choice(["load", "store", "move"])
        nr = rng.choice([1, 2, 4, 8])
        base = rng.randrange(0, 32, nr)
        if kind == "move":
            src = rng.randrange(0, 32, nr)
            g = [Group("vd", base, nr, 8, F(nr)), Group("vs", src, nr, 8, F(nr))]
            # vmv<nr>r.v reads vtype (its EEW is SEW, and it is illegal when vill is set).
            return Item(
                f"vmv{nr}r.v",
                "whole",
                [f"vmv{nr}r.v v{base}, v{src}"],
                random_config(rng),
                groups=g,
            )
        off = rng.randrange(0, BUF_SIZE - nr * VLENB + 1, 8)
        if kind == "store":
            return Item(
                f"vs{nr}r.v", "whole", self._base(off) + [f"vs{nr}r.v v{base}, (s1)"], None,
                groups=[Group("vs", base, nr, 8, F(nr))], access=Access(True, list(range(off, off + nr * VLENB)), 1),
            )  # fmt: skip
        eew = rng.choice(SEWS)
        return Item(
            f"vl{nr}re{eew}.v", "whole", self._base(off) + [f"vl{nr}re{eew}.v v{base}, (s1)"], None,
            groups=[Group("vd", base, nr, eew, F(nr))],
            access=Access(False, list(range(off, off + nr * VLENB, eew // 8)), eew // 8),
        )  # fmt: skip

    def _gen_indexed(self, family, store):
        rng = self.rng
        nf = rng.randint(2, 8) if family == "segment" else 1
        ieew = rng.choice(SEWS)

        def check(cfg):
            iemul = F(ieew, cfg.sew) * cfg.lmul
            return valid_emul(iemul) and nregs(cfg.lmul) * nf <= 8

        cfg = self.config_for(SEWS, check)
        if cfg is None:
            return None
        sew, vl = cfg.sew, cfg.vl
        eb = sew // 8
        iemul = F(ieew, sew) * cfg.lmul
        icount = nregs(iemul)
        masked = rng.random() < 0.4
        ordered = rng.random() < 0.5
        seg_bytes = nf * eb
        limit = min(BUF_SIZE - seg_bytes, 2**ieew - 1)
        slots = list(range(0, limit + 1, seg_bytes if (store and not ordered) else eb))
        if store and not ordered and len(slots) < vl:
            ordered = True  # not enough distinct addresses for an unordered store
        if store and not ordered:
            offsets = rng.sample(slots, vl)
        else:
            offsets = [rng.choice(slots) for _ in range(vl)]
        nelem = icount * VLENB // (ieew // 8)
        offsets += [0] * (nelem - len(offsets))
        idx = pick_group(rng, iemul)
        dcount = nregs(cfg.lmul) * nf
        avoid = [(idx, icount)]
        if masked and not store:
            avoid.append((0, 1))
        data = pick_group(rng, cfg.lmul, avoid if not store else [], dcount)
        if data is None:
            return None
        if masked and not store and idx == 0:
            return None
        label = f"idx_{self.seed}_{self.n_tables}"
        self.n_tables += 1
        directive = {8: ".byte", 16: ".half", 32: ".word", 64: ".dword"}[ieew]
        table = [".balign 64", f"{label}:"] + [
            f"  {directive} " + ", ".join(str(v) for v in offsets[i : i + 16])
            for i in range(0, len(offsets), 16)
        ]
        seg = f"seg{nf}" if nf > 1 else ""
        kind = "o" if ordered else "u"
        name = f"vs{kind}x{seg}ei{ieew}.v" if store else f"vl{kind}x{seg}ei{ieew}.v"
        lines = [f"la t2, {label}", f"vl{icount}re{ieew}.v v{idx}, (t2)", "mv s1, s0"]
        lines.append(f"{name} v{data}, (s1), v{idx}" + (", v0.t" if masked else ""))
        groups = [
            Group("vs" if store else "vd", data, dcount, sew, cfg.lmul),
            Group("vs", idx, icount, ieew, iemul),
        ] + ([Group("vm", 0, 1, 1, F(1))] if masked else [])
        acc = Access(
            store,
            [o + f * eb for o in offsets[:vl] for f in range(nf)],
            eb,
            unordered=store and not ordered,
        )
        return Item(
            name,
            family,
            lines,
            cfg,
            groups=groups,
            masked=masked,
            vd_excl=True,
            data=table,
            access=acc,
        )

    # -- vset / csr --

    def gen_vset(self):
        rng = self.rng
        cfg = random_config(rng)
        rd = f"x{rng.choice(X_POOL)}"
        form = rng.choice(["vsetvli", "vsetivli", "vsetvl", "x0x0", "vlmax", "vill"])
        tail = f"e{cfg.sew}, {lmul_name(cfg.lmul)}, tu, mu"
        if form == "vsetvli":
            return Item(
                "vsetvli",
                "vset",
                [f"li t0, {cfg.avl}", f"vsetvli {rd}, t0, {tail}"],
                None,
                result=cfg,
            )
        if form == "vsetivli":
            avl = rng.randint(0, 31)
            if cfg.vlmax < avl < 2 * cfg.vlmax:
                avl = cfg.vlmax
            c = Config(cfg.sew, cfg.lmul, avl)
            return Item(
                "vsetivli", "vset", [f"vsetivli {rd}, {avl}, {tail}"], None, result=c
            )
        if form == "vsetvl":
            return Item(
                "vsetvl", "vset", [f"li t0, {cfg.avl}", f"li t2, {vtype_bits(cfg.sew, cfg.lmul)}", f"vsetvl {rd}, t0, t2"],
                None, result=cfg,
            )  # fmt: skip
        if form == "vlmax":
            c = Config(cfg.sew, cfg.lmul, cfg.vlmax)
            return Item(
                "vsetvli", "vset", [f"vsetvli {rd}, x0, {tail}"], None, result=c
            )
        if form == "x0x0":
            # vl-preserving: the new SEW/LMUL ratio equals the old one.
            ratio = F(cfg.sew) / cfg.lmul
            options = [
                (s, lm)
                for s in SEWS
                for lm in LMULS
                if legal_vtype(s, lm) and F(s) / lm == ratio
            ]
            s, lm = rng.choice(options)
            vlmax = int(lm * VLEN / s)
            c = Config(s, lm, cfg.vl if cfg.vl <= vlmax else vlmax)
            line = f"vsetvli x0, x0, e{s}, {lmul_name(lm)}, tu, mu"
            return Item("vsetvli.x0", "vset", [line], cfg, result=c)
        # vill: a reserved vtype through vsetvl (reserved SEW, reserved LMUL, or a set reserved bit).
        bad = rng.choice([0b100 << 3, 0b101 << 3, 0b100, 1 << 8, 1 << 20])
        return Item(
            "vsetvl.vill",
            "vset",
            [f"li t0, {cfg.avl}", f"li t2, {bad}", f"vsetvl {rd}, t0, t2"],
            None,
            result="vill",
        )

    def gen_csr(self):
        rng = self.rng
        rd = f"x{rng.choice(X_POOL)}"
        r = rng.random()
        if r < 0.5:
            csr = rng.choice(
                [
                    "vl",
                    "vtype",
                    "vlenb",
                    "vxrm",
                    "vxsat",
                    "vcsr",
                    "frm",
                    "fflags",
                    "fcsr",
                    "vstart",
                ]
            )
            return Item(f"csrr.{csr}", "csr", [f"csrr {rd}, {csr}"], None)
        what = rng.choice(["vxrm", "vxsat", "vcsr", "frm", "fflags"])
        val = {"vxrm": rng.randint(0, 3), "vxsat": rng.randint(0, 1), "vcsr": rng.randint(0, 7),
               "frm": rng.randint(0, 4), "fflags": rng.choice([0, 0, rng.randint(0, 31)])}[what]  # fmt: skip
        return Item(f"csrwi.{what}", "csr", [f"csrwi {what}, {val}"], None)

    # -- the stream --

    def generate(self):
        rng = self.rng
        state = InitialState.random(rng)
        items = []
        kinds = [("op", None)] * (len(self.ops) > 0) * 6 + [
            ("extra", f) for f in self.extra
        ]
        if not kinds:
            raise ValueError("the allowlist selects no instructions")
        attempts = 0
        while len(items) < self.count and attempts < self.count * 20:
            attempts += 1
            kind, fam = rng.choice(kinds)
            if kind == "op":
                item = self.gen_op(rng.choice(self.ops))
            elif fam == "vset":
                item = self.gen_vset()
            elif fam == "csr":
                item = self.gen_csr()
            else:
                item = self.gen_mem(fam)
            if item is None or not self.wanted(item.name):
                continue
            if item.cfg is not None:
                self.last_cfg = item.cfg
            elif isinstance(item.result, Config):
                self.last_cfg = item.result
            items.append(item)
        return Program(self.seed, state, items)


def vtype_bits(sew, lmul, ta=0, ma=0):
    vsew = SEWS.index(sew)
    vlmul = {F(1, 8): 5, F(1, 4): 6, F(1, 2): 7, F(1): 0, F(2): 1, F(4): 2, F(8): 3}[
        lmul
    ]
    return (ma << 7) | (ta << 6) | (vsew << 3) | vlmul


# -- initial state --------------------------------------------------------------


def _fp_specials(width, rng):
    if width == 16:
        exp_bits, man_bits = rng.choice([(5, 10), (8, 7)])  # fp16 or bf16 patterns
    elif width == 32:
        exp_bits, man_bits = 8, 23
    else:
        exp_bits, man_bits = 11, 52
    emax = (1 << exp_bits) - 1
    sign = rng.getrandbits(1) << (width - 1)
    man_mask = (1 << man_bits) - 1
    kind = rng.choice(
        [
            "zero",
            "inf",
            "qnan",
            "snan",
            "sub",
            "minsub",
            "maxsub",
            "minnorm",
            "maxnorm",
            "one",
            "norm",
            "norm",
            "rand",
        ]
    )
    if kind == "zero":
        v = 0
    elif kind == "inf":
        v = emax << man_bits
    elif kind == "qnan":
        v = (emax << man_bits) | (1 << (man_bits - 1)) | rng.getrandbits(man_bits - 1)
    elif kind == "snan":
        v = (emax << man_bits) | max(1, rng.getrandbits(man_bits - 1))
    elif kind == "sub":
        v = max(1, rng.getrandbits(man_bits))
    elif kind == "minsub":
        v = 1
    elif kind == "maxsub":
        v = man_mask
    elif kind == "minnorm":
        v = 1 << man_bits
    elif kind == "maxnorm":
        v = ((emax - 1) << man_bits) | man_mask
    elif kind == "one":
        v = ((emax >> 1) << man_bits) | rng.choice([0, 0, 1])
    elif kind == "norm":
        bias = emax >> 1
        e = rng.randint(max(1, bias - 12), min(emax - 1, bias + 12))
        v = (e << man_bits) | rng.getrandbits(man_bits)
    else:
        return rng.getrandbits(width)
    return sign | v


@dataclass
class InitialState:
    vregs: bytes  # 32 * VLENB
    xregs: dict  # reg -> value
    fregs: list  # 32 raw 64-bit values
    buf: bytes
    frm: int
    vxrm: int

    @classmethod
    def random(cls, rng):
        v = bytearray()
        for _ in range(32):
            kind = rng.random()
            if kind < 0.4:
                v += rng.randbytes(VLENB)
            elif kind < 0.55:
                small = [rng.randint(-3, 3) & 0xFF for _ in range(VLENB)]
                v += bytes(small)
            else:
                w = rng.choice([16, 32, 64])
                for _ in range(VLENB * 8 // w):
                    v += _fp_specials(w, rng).to_bytes(w // 8, "little")
        x = {r: scalar_value(rng) for r in X_POOL}
        f = []
        for _ in range(32):
            k = rng.random()
            if k < 0.3:
                f.append(_fp_specials(64, rng))
            elif k < 0.6:
                f.append(0xFFFFFFFF00000000 | _fp_specials(32, rng))
            elif k < 0.95:
                f.append(0xFFFFFFFFFFFF0000 | _fp_specials(16, rng))
            else:
                f.append(
                    rng.getrandbits(64)
                )  # not NaN-boxed: reads as the canonical NaN
        buf = bytearray(rng.randbytes(BUF_SIZE))
        for i in range(0, BUF_SIZE, 8):  # sprinkle FP specials through the buffer
            if rng.random() < 0.3:
                w = rng.choice([16, 32, 64])
                buf[i : i + w // 8] = _fp_specials(w, rng).to_bytes(w // 8, "little")
        return cls(bytes(v), x, f, bytes(buf), rng.randint(0, 4), rng.randint(0, 3))


# -- the program ----------------------------------------------------------------

#: Signature layout: x0..x31, CSRs, f0..f31, v0..v31, buffer.
SIG_X = 0
SIG_CSR = 32 * 8
CSR_FIELDS = ("vl", "vtype", "vstart", "vxsat", "vxrm", "fcsr", "vlenb")
SIG_F = SIG_CSR + 8 * 8
SIG_V = SIG_F + 32 * 8
SIG_BUF = SIG_V + 32 * VLENB
SIG_SIZE = SIG_BUF + BUF_SIZE


@dataclass
class Program:
    seed: int
    state: InitialState
    items: list

    def subset(self, keep):
        keep = sorted(set(keep))
        return Program(self.seed, self.state, [self.items[i] for i in keep]), keep

    def body_lines(self, labels=True):
        out, current = [], None
        for k, item in enumerate(self.items):
            if labels:
                out.append(f"fuzz_item_{k}:")
            if item.cfg is not None and item.cfg != current:
                out += ["  " + line for line in item.cfg.vsetvli()]
                current = item.cfg
            out += ["  " + line for line in item.lines]
            if item.result is not None:
                current = item.result
        return out

    def assembly(self):
        s = self.state
        L = [
            f"# rvv_fuzz seed {self.seed}: {len(self.items)} items. Generated by tools/naiad/rvv_fuzz.py.",
            '  .section .text.init, "ax", @progbits',
            "  .globl _start",
            "_start:",
            "  la t0, fuzz_trap",
            "  csrw mtvec, t0",
            f"  li t0, {(1 << 9) | (1 << 13)}  # mstatus.VS = mstatus.FS = Initial",
            "  csrs mstatus, t0",
            "  csrwi fcsr, 0",
            f"  csrwi frm, {s.frm}",
            "  csrwi vcsr, 0",
            f"  csrwi vxrm, {s.vxrm}",
            "  la s0, fuzz_buf",
            "  la t2, fuzz_vinit",
            "  vsetvli t1, x0, e8, m8, tu, mu",
        ]
        for g in range(0, 32, 8):
            L += [f"  vle8.v v{g}, (t2)", f"  addi t2, t2, {8 * VLENB}"]
        L.append("  la t2, fuzz_finit")
        L += [f"  fld f{i}, {8 * i}(t2)" for i in range(32)]
        L.append("  la t2, fuzz_xinit")
        L += [f"  ld x{r}, {8 * r}(t2)" for r in sorted(s.xregs)]
        L.append("fuzz_body:")
        L += self.body_lines()
        L += [
            "fuzz_end:",
            "  la t2, begin_signature",
            *[f"  sd x{r}, {8 * r}(t2)" for r in range(1, 32)],
        ]
        for i, csr in enumerate(CSR_FIELDS):
            L += [f"  csrr t0, {csr}", f"  sd t0, {SIG_CSR + 8 * i}(t2)"]
        L += [f"  fsd f{i}, {SIG_F + 8 * i}(t2)" for i in range(32)]
        L += [
            "  vsetvli t1, x0, e8, m8, tu, mu",
            f"  li t0, {SIG_V}",
            "  add t0, t0, t2",
        ]
        for g in range(0, 32, 8):
            L += [f"  vse8.v v{g}, (t0)", f"  addi t0, t0, {8 * VLENB}"]
        L += [
            "  li a0, 1",
            "fuzz_exit:",
            "  la t5, tohost",
            "1: ld t6, 0(t5)",
            "  bnez t6, 1b",
            "  sd a0, 0(t5)",
            "2: j 2b",
            "  .balign 4",
            "fuzz_trap:",
            f"  li a0, {(TRAP_EXIT << 1) | 1}",
            "  j fuzz_exit",
            "",
            '  .section .tohost, "aw", @progbits',
            "  .balign 64",
            "  .globl tohost",
            "tohost: .dword 0",
            "  .balign 64",
            "  .globl fromhost",
            "fromhost: .dword 0",
            "",
            "  .data",
            "  .balign 64",
            "  .globl begin_signature",
            "begin_signature:",
            f"  .zero {SIG_BUF}",
            "fuzz_buf:",
            *_bytes_directives(s.buf),
            "  .globl end_signature",
            "end_signature:",
            "  .balign 64",
            "fuzz_vinit:",
            *_bytes_directives(s.vregs),
            "  .balign 8",
            "fuzz_finit:",
            *[f"  .dword {v:#x}" for v in s.fregs],
            "fuzz_xinit:",
            *[f"  .dword {s.xregs.get(r, 0):#x}" for r in range(32)],
        ]
        for item in self.items:
            L += item.data
        return "\n".join(L) + "\n"


def _bytes_directives(data):
    return [
        f"  .byte {', '.join(str(b) for b in data[i : i + 32])}"
        for i in range(0, len(data), 32)
    ]


# -- build and run --------------------------------------------------------------


class FuzzError(Exception):
    pass


def build(program, work, cc=DEFAULT_CC, name=None):
    """Assemble and link ``program`` in ``work``; returns the ELF path."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    name = name or f"seed{program.seed}"
    src, elf = work / f"{name}.S", work / f"{name}.elf"
    src.write_text(program.assembly())
    link = REPO_ROOT / "tools" / "naiad" / "vector_env" / "link.ld"
    cmd = [
        cc, "--target=riscv64-unknown-elf", f"-march={MARCH}", "-mabi=lp64d", "-mcmodel=medany",
        "-mno-relax", "-nostdlib", "-static", "-fuse-ld=lld", f"-T{link}", str(src), "-o", str(elf),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FuzzError(f"assembly failed: {proc.stderr.strip()[:2000]}")
    return elf


@dataclass
class SimResult:
    status: str  # "ok", "exit", "trap", "timeout", "crash"
    exit_code: int | None = None
    signature: bytes | None = None
    detail: str = ""
    pc: int | None = None


def _parse_spike_signature(text, size):
    words = text.split()
    data = b"".join(bytes.fromhex(w)[::-1] for w in words)
    return data[:size]


def run_spike_sig(elf, isa=SPIKE_ISA, spike=None, timeout=60):
    from framework.naiad.difftest import find_spike

    spike = spike or find_spike()
    if spike is None:
        raise FuzzError("spike not found")
    with tempfile.TemporaryDirectory(prefix="rvv-fuzz-") as tmp:
        sig = Path(tmp) / "sig"
        cmd = [
            spike,
            f"--isa={isa}",
            f"+signature={sig}",
            "+signature-granularity=8",
            str(elf),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return SimResult("timeout", detail=f"spike timed out after {timeout}s")
        if b"tohost = 2047" in proc.stderr or proc.returncode == TRAP_EXIT:
            return SimResult(
                "trap",
                TRAP_EXIT,
                detail="spike trapped: " + spike_trap_site(spike, isa, elf, timeout),
            )
        if proc.returncode != 0:
            return SimResult(
                "exit",
                proc.returncode,
                detail=proc.stderr.decode(errors="replace")[-300:],
            )
        return SimResult("ok", 0, _parse_spike_signature(sig.read_text(), SIG_SIZE))


def spike_trap_site(spike, isa, elf, timeout=60):
    """The first exception Spike reports for ``elf``, with the instruction at its epc."""
    try:
        proc = subprocess.run(
            [spike, "-l", f"--isa={isa}", str(elf)],
            capture_output=True,
            timeout=timeout,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return "?"
    last = {}
    for line in proc.stderr.splitlines():
        m = re.match(r"core\s+\d+: 0x([0-9a-f]+) \(0x[0-9a-f]+\) (.*)", line)
        if m:
            last[int(m.group(1), 16)] = m.group(2).strip()
            continue
        m = re.search(r"exception (\w+), epc 0x([0-9a-f]+)", line)
        if m:
            epc = int(m.group(2), 16)
            item = _item_at(elf, epc)
            return f"{m.group(1)} at {epc:#x} (item {item}): {last.get(epc, '?')}"
    return "no exception in the log"


_DESIGNS = {}


def run_naiad_sig(elf, design=DEFAULT_DESIGN, max_cycles=DEFAULT_MAX_CYCLES):
    from framework.naiad.builder import build as build_soc
    from framework.naiad.design import load_design

    key = str(design)
    if key not in _DESIGNS:
        _DESIGNS[key] = load_design(design)
    soc = build_soc(_DESIGNS[key])
    if soc.htif is not None:
        soc.htif.console_stream = None
    soc.load(elf)
    try:
        result = soc.run(max_cycles)
    except Exception as e:  # noqa: BLE001 - a DUT crash is a result
        pc = None
        try:
            pc = soc.cores[0].pc_register.read_uint()
        except Exception:  # noqa: BLE001
            pass
        return SimResult("crash", detail=" ".join(str(e).split())[:400], pc=pc)
    if result.exit_code is None:
        return SimResult("timeout", detail=f"no exit in {result.cycles} cycles")
    if result.exit_code != 0:
        return SimResult("exit", result.exit_code)
    begin = soc.image.symbol("begin_signature")
    return SimResult("ok", 0, bytes(soc.memory.read(begin, SIG_SIZE)))


# -- comparison -----------------------------------------------------------------


def decode_signature(sig):
    """The signature as ``{field: value}``: x1..x31, CSRs, f0..f31, v0..v31 (bytes), buf (bytes)."""
    out = {}
    for r in range(1, 32):
        out[f"x{r}"] = struct.unpack_from("<Q", sig, SIG_X + 8 * r)[0]
    for i, name in enumerate(CSR_FIELDS):
        out[name] = struct.unpack_from("<Q", sig, SIG_CSR + 8 * i)[0]
    for i in range(32):
        out[f"f{i}"] = struct.unpack_from("<Q", sig, SIG_F + 8 * i)[0]
    for i in range(32):
        out[f"v{i}"] = sig[SIG_V + VLENB * i : SIG_V + VLENB * (i + 1)]
    out["buf"] = sig[SIG_BUF : SIG_BUF + BUF_SIZE]
    return out


#: Registers the harness itself writes (t0/t1/t2/s1 hold addresses and scratch at the end).
_HARNESS_REGS = {"x5", "x6", "x7"}


def compare_signatures(ref, dut):
    """Human-readable differences between two signatures (empty when equal)."""
    a, b = decode_signature(ref), decode_signature(dut)
    diffs = []
    for key in a:
        if key in _HARNESS_REGS or a[key] == b[key]:
            continue
        if key == "buf":
            bad = [i for i in range(BUF_SIZE) if a[key][i] != b[key][i]]
            diffs.append(
                f"buf: {len(bad)} bytes differ, first at +{bad[0]:#x}: "
                f"spike {a[key][bad[0] : bad[0] + 8].hex()} dut {b[key][bad[0] : bad[0] + 8].hex()}"
            )
        elif key.startswith("v"):
            if key in CSR_FIELDS:
                diffs.append(f"{key}: spike {a[key]:#x} dut {b[key]:#x}")
                continue
            bad = [i for i in range(VLENB) if a[key][i] != b[key][i]]
            diffs.append(
                f"{key}: bytes {bad[0]}..{bad[-1]} differ: spike {a[key][::-1].hex()} dut {b[key][::-1].hex()} (MSB first)"
            )
        else:
            diffs.append(f"{key}: spike {a[key]:#018x} dut {b[key]:#018x}")
    return diffs


@dataclass
class Outcome:
    seed: int
    status: str  # PASS, FAIL, CRASH, TIMEOUT, REF, BUILD
    detail: str = ""
    diffs: list = field(default_factory=list)
    item: int | None = None
    keep: list | None = None

    @property
    def failing(self):
        return self.status in ("FAIL", "CRASH", "TIMEOUT")


def _item_at(elf, pc):
    """The ``fuzz_item_<k>`` an address falls in, from the ELF symbols."""
    from framework.util.elf_loader import read_elf

    image, _ = read_elf(elf)
    labels = sorted(
        (addr, int(name[len("fuzz_item_") :]))
        for name, addr in image.symbols.items()
        if name.startswith("fuzz_item_")
    )
    found = None
    for addr, k in labels:
        if addr <= pc:
            found = k
        else:
            break
    return found


def evaluate(
    program,
    work,
    *,
    target="naiad",
    design=DEFAULT_DESIGN,
    dut_isa=None,
    cc=DEFAULT_CC,
    name=None,
    max_cycles=DEFAULT_MAX_CYCLES,
):
    """Build ``program``, run it on Spike and the DUT, and compare."""
    try:
        elf = build(program, work, cc, name)
    except FuzzError as e:
        return Outcome(program.seed, "BUILD", str(e))
    ref = run_spike_sig(elf)
    if ref.status != "ok":
        return Outcome(
            program.seed, "REF", ref.detail or f"spike: {ref.status} {ref.exit_code}"
        )
    if target == "spike":
        dut = run_spike_sig(elf, isa=dut_isa or SPIKE_ISA)
    else:
        dut = run_naiad_sig(elf, design, max_cycles)
    if dut.status == "crash" or dut.status == "trap":
        item = _item_at(elf, dut.pc) if dut.pc is not None else None
        return Outcome(program.seed, "CRASH", dut.detail, item=item)
    if dut.status == "timeout":
        return Outcome(program.seed, "TIMEOUT", dut.detail)
    if dut.status != "ok":
        return Outcome(program.seed, "FAIL", f"dut exit code {dut.exit_code}")
    diffs = compare_signatures(ref.signature, dut.signature)
    if diffs:
        return Outcome(program.seed, "FAIL", f"{len(diffs)} state fields differ", diffs)
    return Outcome(program.seed, "PASS")


# -- shrinking ------------------------------------------------------------------


def ddmin(n, fails):
    """Indices ``0..n-1`` reduced to a 1-minimal list for which ``fails(list)`` holds.

    Classic delta debugging (Zeller): try removing each of ``k`` chunks; on success
    keep the complement and coarsen, otherwise refine until chunks are single items.
    ``fails(range(n))`` is assumed true.
    """
    current = list(range(n))
    k = 2
    while len(current) >= 2:
        size = max(1, len(current) // k)
        chunks = [current[i : i + size] for i in range(0, len(current), size)]
        reduced = False
        for chunk in chunks:
            rest = [i for i in current if i not in chunk]
            if rest and fails(rest):
                current = rest
                k = max(k - 1, 2)
                reduced = True
                break
        if not reduced:
            # Try each chunk alone (fast path to a single culprit).
            for chunk in chunks:
                if len(chunk) < len(current) and fails(chunk):
                    current = chunk
                    k = 2
                    reduced = True
                    break
        if not reduced:
            if size == 1:
                break
            k = min(len(current), 2 * k)
    return current


def shrink(program, outcome, work, **kw):
    """The smallest sub-program (by :func:`ddmin`) failing the same way as ``outcome``."""
    n = len(program.items)
    calls = [0]

    def same_failure(o):
        return o.status == outcome.status

    def fails(keep):
        calls[0] += 1
        sub, _ = program.subset(keep)
        o = evaluate(sub, work, name=f"seed{program.seed}.shrink", **kw)
        return same_failure(o)

    if outcome.status == "CRASH" and outcome.item is not None:
        # The crashing instruction alone is the usual answer: try it first.
        if fails([outcome.item]):
            keep = [outcome.item]
            return keep, calls[0]
        keep = ddmin(outcome.item + 1, fails)
    else:
        keep = ddmin(n, fails)
    return keep, calls[0]


# -- driver ---------------------------------------------------------------------


def _parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if ":" in part:
            a, b = part.split(":")
            seeds += range(int(a), int(b))
        elif part:
            seeds.append(int(part))
    return seeds


def run_seed(args):
    seed, opts = args
    gen = Generator(
        seed,
        count=opts["count"],
        families=opts["families"],
        exts=opts["exts"],
        only=opts["only"],
        exclude=opts["exclude"],
    )
    program = gen.generate()
    if opts.get("keep") is not None:
        program, _ = program.subset(opts["keep"])
    work = Path(opts["work"])
    kw = dict(
        target=opts["target"],
        design=opts["design"],
        dut_isa=opts["dut_isa"],
        cc=opts["cc"],
    )
    outcome = evaluate(program, work, **kw)
    if outcome.failing and opts["shrink"]:
        keep, calls = shrink(program, outcome, work, **kw)
        outcome.keep = keep
        sub, _ = program.subset(keep)
        final = evaluate(sub, work, name=f"seed{seed}.min", **kw)
        outcome.diffs = final.diffs or outcome.diffs
        outcome.detail = f"{outcome.detail} [shrunk {len(program.items)} -> {len(keep)} items in {calls} runs]"
        if final.status == "CRASH":
            outcome.detail = f"{final.detail} [shrunk {len(program.items)} -> {len(keep)} items in {calls} runs]"
        outcome.minimal = "\n".join(sub.body_lines(labels=False))
    return outcome


def format_outcome(o, verbose=False):
    lines = [f"{o.status:7} seed {o.seed}: {o.detail}"]
    if o.keep is not None:
        lines.append(
            f"        reproduce: --seed {o.seed} --keep {','.join(map(str, o.keep))}"
        )
        minimal = getattr(o, "minimal", "")
        lines += [f"        | {line}" for line in minimal.splitlines()]
    if o.status != "PASS" or verbose:
        lines += [f"        {d}" for d in o.diffs[:8]]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None, help="one seed")
    parser.add_argument(
        "--seeds", default=None, help="seeds: A:B (half-open), or a comma list"
    )
    parser.add_argument(
        "-n", "--count", type=int, default=48, help="instructions per program"
    )
    parser.add_argument(
        "--families",
        default=None,
        help="comma-separated families (default: all but fpred_unordered)",
    )
    parser.add_argument(
        "--ext",
        default=",".join(DEFAULT_EXTS),
        help="extensions: v,zvfh,zvfbfmin,zvfbfwma",
    )
    parser.add_argument(
        "--only", default=None, help="only mnemonics matching this regex"
    )
    parser.add_argument(
        "--exclude", default=None, help="no mnemonics matching this regex"
    )
    parser.add_argument(
        "--keep", default=None, help="comma list of item indices (a shrunk program)"
    )
    parser.add_argument(
        "--emit", action="store_true", help="print the program's assembly and stop"
    )
    parser.add_argument("--target", choices=("naiad", "spike"), default="naiad")
    parser.add_argument("--design", default=str(DEFAULT_DESIGN))
    parser.add_argument(
        "--dut-isa", default=None, help="with --target spike: the DUT Spike's --isa"
    )
    parser.add_argument("--no-shrink", action="store_true")
    parser.add_argument("--cc", default=DEFAULT_CC)
    parser.add_argument(
        "--work", default=str(REPO_ROOT / "kernels" / "naiad" / "build" / "rvv_fuzz")
    )
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--list-families", action="store_true")
    args = parser.parse_args(argv)

    if args.list_families:
        for fam, text in FAMILY_HELP.items():
            off = " (off by default)" if fam in DEFAULT_OFF else ""
            n = sum(op.family == fam for op in OPS)
            print(f"{fam:16} {text}{off}" + (f" [{n} ops]" if n else ""))
        return 0
    families = None
    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
        unknown = [f for f in families if f not in FAMILY_HELP]
        if unknown:
            print(
                f"rvv_fuzz: unknown families {unknown} (see --list-families)",
                file=sys.stderr,
            )
            return 2
    exts = [e.strip() for e in args.ext.split(",") if e.strip()]
    keep = [int(k) for k in args.keep.split(",") if k] if args.keep else None
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        seeds = _parse_seeds(args.seeds)
    else:
        seeds = list(range(16))
    opts = dict(
        count=args.count, families=families, exts=exts, only=args.only, exclude=args.exclude, keep=keep,
        target=args.target, design=args.design, dut_isa=args.dut_isa, cc=args.cc, work=args.work,
        shrink=not args.no_shrink,
    )  # fmt: skip
    if args.emit:
        for seed in seeds:
            program = Generator(
                seed,
                count=args.count,
                families=families,
                exts=exts,
                only=args.only,
                exclude=args.exclude,
            ).generate()
            if keep is not None:
                program, _ = program.subset(keep)
            print(program.assembly())
        return 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        outcomes = list(pool.map(run_seed, [(s, opts) for s in seeds]))
    counts = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
        if o.status != "PASS" or args.verbose:
            print(format_outcome(o, args.verbose))
    print(
        f"\nrvv_fuzz on {args.target}: {len(seeds)} seeds x {args.count} instructions: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    )
    return 0 if counts.get("PASS", 0) == len(seeds) else 1


if __name__ == "__main__":
    sys.exit(main())
