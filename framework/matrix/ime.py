"""IME (Zvvm): the integrated matrix mode's definitions, independent of execution.

Pinned to the RISC-V Integrated Matrix Extension TG's Zvvm draft,
``riscv/integrated-matrix-extension`` @ ``6effbcae`` (2026-08-04),
``src/integrated-matrix.adoc``. Everything here is that draft's, by section:

* **Operation** ("Matrix tile multiplication computation"): ``C += A . B^T``. A
  and B are row-major ``M x K_eff`` tiles in vector register groups (B's rows
  are B^T's columns), C a square ``M x M`` group. No new architectural state:
  only ``vtype`` fields.
* **vtype fields** ("New fields in the Vector Type Register"): ``lambda[2:0]`` at
  ``vtype[XLEN-2:XLEN-4]``, ``bs`` at ``[XLEN-5]``, ``altfmt_A`` at
  ``[XLEN-6]``, ``altfmt_B`` at ``[XLEN-7]``; Zvfbfa's ``altfmt`` (bit 8)
  picks the C format. Only ``vsetvl`` writes the IME fields; ``vsetvli`` keeps
  ``bs``/``altfmt_A``/``altfmt_B``, and ``lambda`` is WARL (:func:`warl_lambda`).
* **Geometry** ("ISA control of matrix tile geometry", the SAIL
  ``decode_gemm_geometry``): ``M = VLEN / (SEW * lambda)``, ``K_eff = lambda * W
  * LMUL``, ``EMUL_C = M / lambda`` in {1, 2, 4, 8, 16}, ``N = VL / (lambda *
  LMUL)``; LMUL in {1, 2, 4, 8} only.
* **Encodings** (the per-instruction wavedrom blocks): the multiply-accumulates in
  OP-V (OPIVV funct6 0x38-0x3b, OPFVV 0x14-0x17, ``vm = 1``), the tile
  loads/stores in LOAD-FP / STORE-FP with ``width = 111``, ``mew = 1``, ``mop``
  00 (order-preserving) / 01 (transposing) and ``nf`` reused as an immediate
  lambda. :data:`INSTRUCTIONS` is the table; the header generator emits it.
* **Legality** ("Encoding map", normative): :func:`encoding_row` is the FP and
  integer maps for the ``vm = 1`` forms; :data:`EXTENSIONS` is ``tbl-extensions``.
* **Register <-> tile layout**: :func:`tile_reg_idx` is the SAIL helper; the
  numpy views (:func:`tile_view` / :func:`write_tile`) are its vectorised form.

**Where this module departs from the draft's text** (each noted where it lives):

* For a widening multiply-accumulate (``W > 1``) with ``LMUL > 1`` the draft's
  SAIL ``mat_A_idx`` calls ``tile_reg_idx(i * K_eff + k, LMUL, lambda, epr_A)``
  at the narrow element width, which puts narrow elements ``k`` and ``k + lambda
  * LMUL`` in the same register. That contradicts its own prose and its tile
  load: a load moves ``lambda * LMUL`` SEW storage elements per row, register
  ``r`` taking storage elements ``[r * lambda, (r + 1) * lambda)``, each one a
  sub-dot-product of ``W`` packed narrow elements ("Element packing in input
  tiles"), and the rounding model's ``LMUL = 1`` steps are registers. We use
  the layout the loads produce, ``tile_reg_idx`` with ``lambda * W`` in place of
  ``lambda``, which agrees with the SAIL whenever ``W = 1`` or ``LMUL = 1``. The
  two differ only in which products share a rounding group (integers, and FP
  with exact grouping, give the same numbers either way).
* Not modelled: microscaling (``vm = 0``, ``bs``, ``v0.scale``), 4-bit inputs
  (Int4, OFP4), OFP8 accumulators (``Zvvofp8mm``), ``psm = 1`` (RVBNA), the
  ``m16`` pair/unpair intrinsics, fflags, and a non-RNE ``frm`` (an FP
  multiply-accumulate under one raises :class:`IMEError`). Those extensions are
  refused by the design validator by name.
"""

from dataclasses import dataclass, field

import numpy as np

XLEN = 64

# -- vtype fields -----------------------------------------------------------------

LAMBDA_SHIFT = XLEN - 4  # lambda[2:0] at vtype[XLEN-2:XLEN-4]
BS_SHIFT = XLEN - 5
ALTFMT_A_SHIFT = XLEN - 6
ALTFMT_B_SHIFT = XLEN - 7
ALTFMT_SHIFT = 8  # Zvfbfa's altfmt: the C format
LAMBDA_MASK = 0x7 << LAMBDA_SHIFT
#: The fields vsetvli / vsetivli keep (bs, altfmt_A, altfmt_B).
KEPT_MASK = (1 << BS_SHIFT) | (1 << ALTFMT_A_SHIFT) | (1 << ALTFMT_B_SHIFT)
#: Every IME bit of vtype (lambda, bs, altfmt_A, altfmt_B).
IME_VTYPE_MASK = LAMBDA_MASK | KEPT_MASK

#: lambda[2:0] encoding: 001..111 = 1, 2, 4, ..., 64; 000 = no selected lambda.
LAMBDA_VALUES = (1, 2, 4, 8, 16, 32, 64)


def decode_lambda(enc):
    """``decode_vlambda``: the lambda a 3-bit encoding selects, or None for 000."""
    return None if enc == 0 else 1 << (enc - 1)


def encode_lambda(lam):
    return lam.bit_length()


def lambda_of(vtype):
    """The selected lambda in ``vtype`` (None when ``lambda[2:0] = 000``)."""
    return decode_lambda((vtype >> LAMBDA_SHIFT) & 0x7)


def permissible_lambdas(vlen, sew):
    """The architecturally permissible lambdas for (VLEN, SEW): ``EMUL_C = VLEN / (SEW * lambda**2)`` in {1..16}.

    Empty outside the IME-legal domain (``VLEN < SEW``).
    """
    out = []
    for lam in LAMBDA_VALUES:
        emul, rem = divmod(vlen, sew * lam * lam)
        if rem == 0 and emul in (1, 2, 4, 8, 16):
            out.append(lam)
    return tuple(out)


def warl_lambda(request, current, supported):
    """The lambda[2:0] encoding a vtype write leaves, by the draft's WARL rules.

    ``request`` is the requested encoding (0: preserve-or-initialize), ``current``
    the field's present encoding, ``supported`` the implementation's lambdas for
    the resulting (VLEN, SEW). Outside the IME-legal domain (nothing supported)
    the field is 000. A zero request keeps a supported current value, else picks
    the largest supported; a nonzero one keeps itself if supported, else the
    largest supported value at most the request, else the smallest supported.
    """
    if not supported:
        return 0
    encs = sorted(encode_lambda(lam) for lam in supported)
    if request == 0:
        return current if current in encs else encs[-1]
    if request in encs:
        return request
    below = [e for e in encs if e <= request]
    return below[-1] if below else encs[0]


# -- the encodings ----------------------------------------------------------------------

OP_V, LOAD_FP, STORE_FP = 0x57, 0x07, 0x27
OPIVV, OPFVV = 0, 1


@dataclass(frozen=True)
class Instruction:
    name: str
    opcode: int
    funct3: int
    #: Multiply-accumulates: funct6; loads/stores: mop (0 order-preserving, 1 transposing).
    code: int
    kind: str  # "int", "fp", "load", "store"
    w: int = 1

    @property
    def funct7(self):
        """``.insn r`` funct7 of the unmasked form with lambda immediate 000 (vm = 1)."""
        if self.kind in ("int", "fp"):
            return (self.code << 1) | 1
        return (1 << 3) | (self.code << 1) | 1  # mew = 1, mop, vm = 1


#: The Zvvm instructions, from the draft's wavedrom encodings.
INSTRUCTIONS = {
    i.name: i
    for i in (
        Instruction("vmmacc.vv", OP_V, OPIVV, 0x38, "int", 1),
        Instruction("vwmmacc.vv", OP_V, OPIVV, 0x39, "int", 2),
        Instruction("vqmmacc.vv", OP_V, OPIVV, 0x3A, "int", 4),
        Instruction("v8wmmacc.vv", OP_V, OPIVV, 0x3B, "int", 8),
        Instruction("vfmmacc.vv", OP_V, OPFVV, 0x14, "fp", 1),
        Instruction("vfwmmacc.vv", OP_V, OPFVV, 0x15, "fp", 2),
        Instruction("vfqmmacc.vv", OP_V, OPFVV, 0x16, "fp", 4),
        Instruction("vf8wmmacc.vv", OP_V, OPFVV, 0x17, "fp", 8),
        Instruction("vmtl.v", LOAD_FP, 7, 0, "load"),
        Instruction("vmttl.v", LOAD_FP, 7, 1, "load"),
        Instruction("vmts.v", STORE_FP, 7, 0, "store"),
        Instruction("vmtts.v", STORE_FP, 7, 1, "store"),
    )
}
_MAC_BY_KEY = {
    (i.funct3, i.code): i for i in INSTRUCTIONS.values() if i.kind in ("int", "fp")
}
_LS_BY_KEY = {
    (i.opcode, i.code): i for i in INSTRUCTIONS.values() if i.kind in ("load", "store")
}


@dataclass(frozen=True)
class Decoded:
    insn: Instruction
    vd: int  # vd (C) / vd (load destination) / vs3 (store source)
    vs1: int = 0  # A (multiply-accumulates)
    vs2: int = 0  # B (multiply-accumulates)
    rs1: int = 0  # base (loads/stores)
    rs2: int = 0  # leading dimension register (loads/stores)
    vm: int = 1
    lam_imm: int = 0  # loads/stores: the nf field, an immediate lambda encoding

    @property
    def transposing(self):
        return self.insn.kind in ("load", "store") and self.insn.code == 1


def in_space(instr):
    """True if ``instr`` is in the encoding space Zvvm claims (RVV 1.0 leaves it reserved)."""
    opcode = instr & 0x7F
    if opcode == OP_V:
        return ((instr >> 12) & 0x7, instr >> 26) in _MAC_BY_KEY
    if opcode in (LOAD_FP, STORE_FP):
        return (instr >> 12) & 0x7 == 7 and (instr >> 28) & 1 and (instr >> 27) & 1 == 0
    return False


def decode(instr):
    """The :class:`Decoded` form of a Zvvm word, or None if it is not one.

    ``vm = 0`` multiply-accumulates (the MX forms) still decode, so the ISA can
    report them as not implemented rather than unknown.
    """
    opcode = instr & 0x7F
    rd, f3 = (instr >> 7) & 0x1F, (instr >> 12) & 0x7
    rs1, rs2 = (instr >> 15) & 0x1F, (instr >> 20) & 0x1F
    vm = (instr >> 25) & 1
    if opcode == OP_V:
        insn = _MAC_BY_KEY.get((f3, instr >> 26))
        if insn is None:
            return None
        return Decoded(insn, rd, vs1=rs1, vs2=rs2, vm=vm)
    if opcode in (LOAD_FP, STORE_FP) and f3 == 7 and (instr >> 28) & 1:
        insn = _LS_BY_KEY.get((opcode, (instr >> 26) & 0x3))
        if insn is None:
            return None
        return Decoded(insn, rd, rs1=rs1, rs2=rs2, vm=vm, lam_imm=instr >> 29)
    return None


def encode(name, vd, a, b, vm=1, lam_imm=0):
    """The 32-bit word for ``name``: ``(vd, vs1, vs2)`` for a multiply-accumulate, ``(vd/vs3, rs1, rs2)`` else."""
    insn = INSTRUCTIONS[name]
    f7 = insn.funct7 & ~1 | vm
    if insn.kind in ("load", "store"):
        f7 |= lam_imm << 4
    return (
        (f7 << 25)
        | (b << 20)
        | (a << 15)
        | (insn.funct3 << 12)
        | (vd << 7)
        | insn.opcode
    )


# -- types and extensions (tbl-extensions) ---------------------------------------------------

#: Integer element types: name -> (bits, signed). The FP ones are framework.matrix.formats names.
INT_TYPES = {
    f"{'u' if u else ''}int{b}": (b, not u) for b in (8, 16, 32, 64) for u in (0, 1)
}
_FP_BITS = {
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
    "fp16": 16,
    "bf16": 16,
    "fp32": 32,
    "fp64": 64,
}


@dataclass(frozen=True)
class Extension:
    name: str
    kind: str  # "int", "fp" or "ls"
    #: Input / accumulator widths in bits and the widening factor (0 for load/store).
    in_bits: int = 0
    acc_bits: int = 0
    #: For FP: the input family ("ofp8", "fp16", "bf16", "fp32", "fp64") and C format.
    in_family: str = ""
    acc_format: str = ""

    @property
    def w(self):
        return self.acc_bits // self.in_bits if self.in_bits else 0


def _int_ext(name, ib, ab):
    return Extension(name, "int", ib, ab, f"i{ib}", f"int{ab}")


def _fp_ext(name, family, ib, acc):
    return Extension(name, "fp", ib, _FP_BITS[acc], family, acc)


#: The modelled computational extensions of tbl-extensions, plus the two tile load/store ones.
EXTENSIONS = {
    e.name: e
    for e in (
        _int_ext("Zvvi8mm", 8, 8),
        _int_ext("Zvvi8i16mm", 8, 16),
        _int_ext("Zvvi8i32mm", 8, 32),
        _int_ext("Zvvi8i64mm", 8, 64),
        _int_ext("Zvvi16mm", 16, 16),
        _int_ext("Zvvi16i32mm", 16, 32),
        _int_ext("Zvvi16i64mm", 16, 64),
        _int_ext("Zvvi32mm", 32, 32),
        _int_ext("Zvvi32i64mm", 32, 64),
        _int_ext("Zvvi64mm", 64, 64),
        _fp_ext("Zvvofp8fp16mm", "ofp8", 8, "fp16"),
        _fp_ext("Zvvofp8bf16mm", "ofp8", 8, "bf16"),
        _fp_ext("Zvvofp8fp32mm", "ofp8", 8, "fp32"),
        _fp_ext("Zvvofp8fp64mm", "ofp8", 8, "fp64"),
        _fp_ext("Zvvfp16mm", "fp16", 16, "fp16"),
        _fp_ext("Zvvfp16fp32mm", "fp16", 16, "fp32"),
        _fp_ext("Zvvfp16fp64mm", "fp16", 16, "fp64"),
        _fp_ext("Zvvbf16mm", "bf16", 16, "bf16"),
        _fp_ext("Zvvbf16fp32mm", "bf16", 16, "fp32"),
        _fp_ext("Zvvbf16fp64mm", "bf16", 16, "fp64"),
        _fp_ext("Zvvfp32mm", "fp32", 32, "fp32"),
        _fp_ext("Zvvfp32fp64mm", "fp32", 32, "fp64"),
        _fp_ext("Zvvfp64mm", "fp64", 64, "fp64"),
        Extension("Zvvmtls", "ls"),
        Extension("Zvvmttls", "ls"),
    )
}
#: Draft extensions this model refuses by name (see the module docstring).
UNMODELLED = (
    "Zvvi4i8mm",
    "Zvvi4i16mm",
    "Zvvi4i32mm",
    "Zvvofp4ofp8mm",
    "Zvvofp4fp16mm",
    "Zvvofp4bf16mm",
    "Zvvofp4fp32mm",
    "Zvvofp8mm",
)
_BY_LOWER = {name.lower(): name for name in (*EXTENSIONS, *UNMODELLED)}


def canonical_extension(name):
    """The draft's spelling of an extension name (case-insensitive), or None if unknown."""
    return _BY_LOWER.get(str(name).lower())


#: Input formats per (family, altfmt bit): the FP input table of the draft.
_FP_INPUT = {
    ("ofp8", 0): "fp8_e4m3",
    ("ofp8", 1): "fp8_e5m2",
    ("16", 0): "fp16",
    ("16", 1): "bf16",
}


@dataclass(frozen=True)
class Row:
    """One encoding-map row: the operand types and the extensions it requires (all of them)."""

    fmt_a: str
    fmt_b: str
    fmt_c: str
    requires: tuple


def encoding_row(kind, w, sew, altfmt_a, altfmt_b, altfmt):
    """The ``vm = 1`` encoding-map row for an instruction, or None (reserved / not modelled).

    ``kind`` is ``"int"`` or ``"fp"``; ``sew`` is the C element width. Implements
    tbl-fp-encoding-map and tbl-int-encoding-map, minus 4-bit inputs and OFP8
    accumulators.
    """
    in_bits, rem = divmod(sew, w)
    if rem or in_bits < 8:
        return None
    if kind == "int":
        name = {
            (8, 8): "Zvvi8mm",
            (8, 16): "Zvvi8i16mm",
            (8, 32): "Zvvi8i32mm",
            (8, 64): "Zvvi8i64mm",
            (16, 16): "Zvvi16mm",
            (16, 32): "Zvvi16i32mm",
            (16, 64): "Zvvi16i64mm",
            (32, 32): "Zvvi32mm",
            (32, 64): "Zvvi32i64mm",
            (64, 64): "Zvvi64mm",
        }.get((in_bits, sew))
        if name is None:
            return None
        fa = f"{'u' if altfmt_a else ''}int{in_bits}"
        fb = f"{'u' if altfmt_b else ''}int{in_bits}"
        return Row(fa, fb, f"int{sew}", (name,))
    # Floating point: the C format from (altfmt, SEW).
    fmt_c = {(16, 0): "fp16", (16, 1): "bf16", (32, 0): "fp32", (64, 0): "fp64"}.get(
        (sew, altfmt)
    )
    if fmt_c is None:
        return (
            None  # 8-bit (OFP8) C is not modelled; altfmt = 1 at SEW 32/64 is reserved
        )
    c_tok = {"fp16": "fp16", "bf16": "bf16", "fp32": "fp32", "fp64": "fp64"}[fmt_c]
    if in_bits == 8:
        fa, fb = _FP_INPUT[("ofp8", altfmt_a)], _FP_INPUT[("ofp8", altfmt_b)]
        return Row(fa, fb, fmt_c, (f"Zvvofp8{c_tok}mm",))
    if in_bits == 16:
        fa, fb = _FP_INPUT[("16", altfmt_a)], _FP_INPUT[("16", altfmt_b)]
        if fa == fb:
            if sew == 16 and fa != fmt_c:
                return None  # fp16 x fp16 -> bf16, bf16 x bf16 -> fp16: reserved
            tok = fa
            return Row(fa, fb, fmt_c, (_ext_name(tok, c_tok),))
        if (
            sew == 16
        ):  # mixed 16-bit inputs, 16-bit C (either format): both single-format extensions
            return Row(fa, fb, fmt_c, ("Zvvfp16mm", "Zvvbf16mm"))
        return Row(fa, fb, fmt_c, (_ext_name("fp16", c_tok), _ext_name("bf16", c_tok)))
    if in_bits == 32:
        return Row("fp32", "fp32", fmt_c, (_ext_name("fp32", c_tok),))
    return Row("fp64", "fp64", fmt_c, ("Zvvfp64mm",))


def _ext_name(in_tok, c_tok):
    return f"Zvv{in_tok}mm" if in_tok == c_tok else f"Zvv{in_tok}{c_tok}mm"


# -- geometry -------------------------------------------------------------------------------


class IMEError(Exception):
    """An IME configuration or instruction this model cannot honour (not an architectural trap)."""


@dataclass(frozen=True)
class Geometry:
    """One multiply-accumulate's tile shape (the SAIL ``gemm_geom``)."""

    sew: int
    w: int
    lmul: int
    lam: int
    m: int  # rows of A and C; also N_max
    n: int  # active columns of C (rows of B)
    k: int  # K_eff, in narrow elements
    emul_c: int

    @property
    def macs(self):
        return self.m * self.n * self.k

    def a_bytes(self):
        return self.m * self.lam * self.lmul * self.sew // 8

    def b_bytes(self):
        return self.n * self.lam * self.lmul * self.sew // 8

    def c_bytes(self):
        return self.m * self.n * self.sew // 8


def gemm_geometry(vlen, sew, lmul, lam, vl, w):
    """The geometry, or a string saying which illegal-instruction condition holds."""
    if lmul not in (1, 2, 4, 8):
        return f"LMUL {lmul} is not one of 1, 2, 4, 8"
    m, rem = divmod(vlen, sew * lam)
    if rem or m < lam:
        return f"lambda {lam} is not legal at VLEN {vlen}, SEW {sew}"
    emul_c = m // lam
    if emul_c not in (1, 2, 4, 8, 16):
        return f"EMUL_C {emul_c} is not one of 1, 2, 4, 8, 16"
    if vl % (lam * lmul):
        return f"VL {vl} is not a multiple of lambda * LMUL = {lam * lmul}"
    n = vl // (lam * lmul)
    if n > m:
        return f"N {n} exceeds M {m}"
    return Geometry(sew, w, lmul, lam, m, n, lam * w * lmul, emul_c)


# -- register <-> tile layout --------------------------------------------------------------------


def tile_reg_idx(i, lmul, lam, elems_per_reg):
    """The draft's SAIL ``tile_reg_idx``: tile element ``i`` -> flat register-group element index."""
    linesize = lam * lmul
    line, elem_in_line = divmod(i, linesize)
    regoff = elem_in_line // lam
    elementoff = line * lam + elem_in_line % lam
    return regoff * elems_per_reg + elementoff


def tile_index_map(lmul, lam, elems_per_reg, count):
    """``tile_reg_idx`` for ``i = 0 .. count - 1`` as a numpy array."""
    i = np.arange(count)
    linesize = lam * lmul
    line, eil = i // linesize, i % linesize
    return (eil // lam) * elems_per_reg + line * lam + eil % lam


def tile_view(span, nregs, rows, width):
    """The row-major ``rows x (nregs * width)`` tile held in a register span (a copy).

    ``span`` is the span's elements (``vreg_bytes(base, nregs).view(dtype)``);
    register ``r`` holds columns ``[r * width, (r + 1) * width)`` of every row,
    ``width`` elements per row per register (``lambda * W`` for A/B, ``lambda``
    for C). This is :func:`tile_reg_idx` with ``lambda`` = ``width``.
    """
    return (
        span.reshape(nregs, rows, width).transpose(1, 0, 2).reshape(rows, nregs * width)
    )


def write_tile(span, tile, nregs, width):
    """Write a row-major tile back into its register span (the inverse of :func:`tile_view`)."""
    rows = tile.shape[0]
    span.reshape(nregs, rows, width)[...] = tile.reshape(rows, nregs, width).transpose(
        1, 0, 2
    )


# -- the design spec --------------------------------------------------------------------------------

PSM_VALUES = (0, 1)
RND_VALUES = ("frm", "rto", "xct")


@dataclass(frozen=True)
class Reduction:
    """One (SEW, W, lambda) entry of the FP accumulation disclosure."""

    g: int = 1
    psm: int = 0
    rnd: str = "frm"


@dataclass(frozen=True)
class IMESpec:
    """A design's ``matrix:`` section in ``mode: ime``.

    ``lambdas`` maps each SEW to the implementation's supported lambdas (every SEW
    of the IME-legal domain at the core's VLEN has an entry); ``extensions`` are
    the draft's names; ``fp_reduction`` maps ``(sew, w, lambda)`` to its disclosed
    :class:`Reduction` (absent keys use the default ``G = 1, psm = 0, rnd = frm``).
    """

    vlen: int
    lambdas: dict = field(hash=False)
    extensions: tuple = ()
    fp_reduction: dict = field(default_factory=dict, hash=False)
    mode: str = "ime"

    def reduction(self, sew, w, lam):
        return self.fp_reduction.get((sew, w, lam), Reduction())

    def has(self, name):
        return name in self.extensions

    @property
    def mac_extensions(self):
        return tuple(e for e in self.extensions if EXTENSIONS[e].kind != "ls")

    @property
    def format_pairs(self):
        """``(in, acc)`` format pairs the listed computational extensions give (same-format inputs)."""
        pairs = []
        for name in self.mac_extensions:
            e = EXTENSIONS[name]
            if e.kind == "int":
                pairs.append((f"int{e.in_bits}", e.acc_format))
            elif e.in_family == "ofp8":
                pairs += [("fp8_e4m3", e.acc_format), ("fp8_e5m2", e.acc_format)]
            else:
                pairs.append((e.in_family, e.acc_format))
        return tuple(pairs)


def parse_ime(raw, cores, error):
    """Validate a ``matrix: {mode: ime, ...}`` mapping against the cores; returns an :class:`IMESpec`.

    ``error`` is the exception class to raise (``DesignError``). Keys:
    ``lambdas`` (``{sew: [lambda, ...]}``, default: the largest permissible
    lambda per SEW, which is what the WARL rule picks on a 000 write),
    ``extensions`` (the draft's names; default ``[Zvvfp32mm, Zvvmtls,
    Zvvmttls]``), ``fp_reduction`` (``{"sew,w,lambda": {G, psm, rnd}}``).
    """
    allowed = {"mode", "lambdas", "extensions", "fp_reduction"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise error(
            f"matrix (mode ime): unknown key(s) {unknown}; expected some of {sorted(allowed)}"
        )
    if cores.xlen != 64 or not cores.has_vector:
        raise error(
            "matrix.mode ime needs an RV64 core with a vector unit ('v' or 'zve64d' in cores.isa)"
        )
    vlen = cores.vlen
    names = raw.get("extensions", ["Zvvfp32mm", "Zvvmtls", "Zvvmttls"])
    if not isinstance(names, list) or not names:
        raise error(
            "matrix.extensions must be a non-empty list of Zvvm extension names"
        )
    exts = []
    for name in names:
        canon = canonical_extension(name)
        if canon is None:
            raise error(
                f"matrix.extensions: {name!r} is not a Zvvm extension of the pinned draft "
                f"(tbl-extensions, plus Zvvmtls / Zvvmttls)"
            )
        if canon in UNMODELLED:
            raise error(
                f"matrix.extensions: {canon} is not modelled (4-bit inputs and OFP8 accumulators are out of scope)"
            )
        if canon in exts:
            raise error(f"matrix.extensions lists {canon} twice")
        exts.append(canon)
    if not any(EXTENSIONS[e].kind != "ls" for e in exts):
        raise error("matrix.extensions names no multiply-accumulate extension")
    for e in exts:
        if EXTENSIONS[e].acc_bits > cores.elen:
            raise error(
                f"matrix.extensions: {e} needs ELEN >= {EXTENSIONS[e].acc_bits}"
            )

    lambdas = {}
    given = raw.get("lambdas") or {}
    if not isinstance(given, dict):
        raise error("matrix.lambdas must be a mapping of SEW -> [lambda, ...]")
    for key in given:
        try:
            sew = int(key)
        except (TypeError, ValueError):
            raise error(f"matrix.lambdas: SEW {key!r} is not an integer") from None
        if sew not in (8, 16, 32, 64) or sew > cores.elen:
            raise error(
                f"matrix.lambdas: SEW {sew} is not one of 8, 16, 32, 64 (<= ELEN)"
            )
    for sew in (8, 16, 32, 64):
        if sew > cores.elen:
            continue
        legal = permissible_lambdas(vlen, sew)
        entry = given.get(sew, given.get(str(sew)))
        if entry is None:
            lambdas[sew] = legal[
                -1:
            ]  # the largest permissible (empty outside the domain)
            continue
        if not isinstance(entry, list) or not entry:
            raise error(f"matrix.lambdas.{sew} must be a non-empty list")
        lams = []
        for lam in entry:
            if not isinstance(lam, int) or lam not in legal:
                raise error(
                    f"matrix.lambdas.{sew}: {lam!r} is not permissible at VLEN {vlen} "
                    f"(EMUL_C = VLEN / (SEW * lambda^2) in 1..16 allows {list(legal)})"
                )
            if lam in lams:
                raise error(f"matrix.lambdas.{sew} lists {lam} twice")
            lams.append(lam)
        lambdas[sew] = tuple(sorted(lams))
    for e in exts:
        ext = EXTENSIONS[e]
        if ext.kind != "ls" and not lambdas.get(ext.acc_bits):
            raise error(
                f"matrix.extensions: {e} accumulates at SEW {ext.acc_bits}, which has no legal lambda at VLEN {vlen}"
            )

    reductions = {}
    raw_red = raw.get("fp_reduction") or {}
    if not isinstance(raw_red, dict):
        raise error(
            "matrix.fp_reduction must be a mapping of 'sew,w,lambda' -> {G, psm, rnd}"
        )
    for key, entry in raw_red.items():
        try:
            sew, w, lam = (int(v) for v in str(key).replace(" ", "").split(","))
        except ValueError:
            raise error(
                f"matrix.fp_reduction: key {key!r} is not 'sew,w,lambda'"
            ) from None
        if lam not in lambdas.get(sew, ()):
            raise error(
                f"matrix.fp_reduction: lambda {lam} is not supported at SEW {sew} ({list(lambdas.get(sew, ()))})"
            )
        if w not in (1, 2, 4, 8) or sew // w < 8:
            raise error(f"matrix.fp_reduction: W {w} is not legal at SEW {sew}")
        if not isinstance(entry, dict) or set(entry) - {"G", "psm", "rnd"}:
            raise error(
                f"matrix.fp_reduction.{key}: expected a mapping with keys from G, psm, rnd"
            )
        g = entry.get("G", 1)
        psm = entry.get("psm", 0)
        rnd = str(entry.get("rnd", "frm")).lower()
        if not isinstance(g, int) or g < 1 or g > lam or g & (g - 1):
            raise error(
                f"matrix.fp_reduction.{key}: G must be a power of two in 1..lambda ({lam}), got {g!r}"
            )
        if psm not in PSM_VALUES:
            raise error(f"matrix.fp_reduction.{key}: psm must be 0 or 1, got {psm!r}")
        if psm == 1:
            raise error(
                f"matrix.fp_reduction.{key}: psm = 1 (an implementation-defined reduction, e.g. RVBNA) is not modelled"
            )
        if rnd not in RND_VALUES:
            raise error(
                f"matrix.fp_reduction.{key}: rnd must be one of {list(RND_VALUES)}, got {rnd!r}"
            )
        reductions[(sew, w, lam)] = Reduction(g, psm, rnd)
    return IMESpec(
        vlen=vlen, lambdas=lambdas, extensions=tuple(exts), fp_reduction=reductions
    )


# -- naiad.h ---------------------------------------------------------------------------------------


def _ident(name):
    return name.upper().replace(".", "_")


def header_lines(spec):
    """The ``naiad.h`` lines for an IME design: mode, VLEN-relative lambdas, extensions, encodings."""
    lines = [
        "",
        "/* matrix unit: ime (Zvvm @ 6effbcae), integrated into the vector unit */",
        "#define NAIAD_HAS_IME 1",
        "#define NAIAD_MATRIX_IME 1",
        '#define NAIAD_MATRIX_MODE "ime"',
    ]
    for name in spec.extensions:
        lines.append(f"#define NAIAD_IME_EXT_{name.upper()} 1")
    for sew, lams in sorted(spec.lambdas.items()):
        if not lams:
            continue
        lines.append(f"#define NAIAD_IME_LAMBDA_MIN_E{sew} {min(lams)}")
        lines.append(f"#define NAIAD_IME_LAMBDA_MAX_E{sew} {max(lams)}")
        mask = sum(1 << encode_lambda(lam) for lam in lams)
        lines.append(
            f"#define NAIAD_IME_LAMBDAS_E{sew} {mask:#x}  /* bit (1 << enc) per supported lambda */"
        )
    lines += [
        "",
        "/* Zvvm vtype fields (RV64) */",
        f"#define NAIAD_IME_VTYPE_LAMBDA_SHIFT {LAMBDA_SHIFT}",
        f"#define NAIAD_IME_VTYPE_BS_SHIFT {BS_SHIFT}",
        f"#define NAIAD_IME_VTYPE_ALTFMT_A_SHIFT {ALTFMT_A_SHIFT}",
        f"#define NAIAD_IME_VTYPE_ALTFMT_B_SHIFT {ALTFMT_B_SHIFT}",
        f"#define NAIAD_IME_VTYPE_ALTFMT_SHIFT {ALTFMT_SHIFT}",
        "",
        "/* Zvvm encodings: .insn r OPCODE, FUNCT3, FUNCT7, vd, vs1|rs1, vs2|rs2 */",
        "#define NAIAD_IME_LS_LAMBDA_SHIFT 4  /* tile load/store funct7: immediate lambda in bits 6:4 */",
    ]
    for insn in INSTRUCTIONS.values():
        stem = f"NAIAD_IME_{_ident(insn.name)}"
        lines += [
            f"#define {stem}_OPCODE {insn.opcode:#x}",
            f"#define {stem}_FUNCT3 {insn.funct3}",
            f"#define {stem}_FUNCT7 {insn.funct7:#x}",
        ]
    return lines


def reference_geometry(vlen, sew, w, lambdas=None, regs=32):
    """The geometry an equal-resource comparison uses: the largest C panel with a single-instruction C store.

    The smallest supported lambda with ``EMUL_C <= 8``, and the largest LMUL with
    one A, one B and the C group in ``regs`` registers. Returns a :class:`Geometry`
    at full VL (``N = M``), or None if nothing fits.
    """
    lams = lambdas if lambdas is not None else permissible_lambdas(vlen, sew)
    for lam in sorted(lams):
        m = vlen // (sew * lam)
        emul_c = m // lam
        if emul_c > 8:
            continue
        lmul = 8
        while lmul > 1 and 2 * lmul + emul_c > regs:
            lmul //= 2
        if 2 * lmul + emul_c > regs:
            continue
        return Geometry(sew, w, lmul, lam, m, m, lam * w * lmul, emul_c)
    return None


def ceil_div(a, b):
    return -(-a // b) if b else 0


__all__ = [
    "EXTENSIONS",
    "INSTRUCTIONS",
    "Geometry",
    "IMEError",
    "IMESpec",
    "Reduction",
    "decode",
    "encode",
    "encoding_row",
    "gemm_geometry",
    "header_lines",
    "parse_ime",
    "permissible_lambdas",
    "tile_reg_idx",
    "warl_lambda",
]
