"""``RV64_V_ISA``: decode RVV 1.0 words and dispatch them to the handlers.

The ISA claims OP-V (0x57) and the vector widths (0, 5, 6, 7) of LOAD-FP /
STORE-FP (0x07 / 0x27); the scalar FP loads and stores use widths 1-3, so the
two never compete for a word and their order in the ISA chain does not matter.
Each word is decoded once into ``(handler, args, text)`` and memoised.

**Declining.** A word RVV 1.0 does not define (a reserved funct6, ``mew = 1``,
an unused unary code) decodes to None and :meth:`RV64_V_ISA.run` returns
False, so a later ISA in the chain or dispatch table (the IME matrix mode,
whose multiply-accumulates are OP-V words such as OPIVV funct6 0x38-0x3b and
OPFVV 0x14-0x17, and whose tile loads set ``mew``) can claim it. Nothing here
raises illegal-instruction for an unknown word: if no ISA claims it the core's
unknown-instruction path reports it. A word that *is* RVV but illegal in the
current configuration (vill, a misaligned group, vstart != 0 for arithmetic,
SEW 16 FP without Zvfh, ...) also returns False, for the same reason.
"""

from functools import partial

from framework.pe.rv.isa.i64_isa import write_reg
from framework.pe.rv.isa.rv_isa import RV_ISA
from framework.pe.rv.isa.v import fpu, integer, mem, permute
from framework.pe.rv.isa.v.state import VectorStateError

# -- OPI (OPIVV = 0, OPIVI = 3, OPIVX = 4) ------------------------------------------------
# funct6 -> (mnemonic stem, forms, handler, extra, immediate is unsigned)

_B = integer.INT_BINARY
_C = integer.INT_COMPARE

_OPI = {
    0x00: ("vadd", "vxi", integer.binary, _B["add"], False),
    0x02: ("vsub", "vx", integer.binary, _B["sub"], False),
    0x03: ("vrsub", "xi", integer.binary, _B["rsub"], False),
    0x04: ("vminu", "vx", integer.binary, _B["minu"], False),
    0x05: ("vmin", "vx", integer.binary, _B["min"], False),
    0x06: ("vmaxu", "vx", integer.binary, _B["maxu"], False),
    0x07: ("vmax", "vx", integer.binary, _B["max"], False),
    0x09: ("vand", "vxi", integer.binary, _B["and"], False),
    0x0A: ("vor", "vxi", integer.binary, _B["or"], False),
    0x0B: ("vxor", "vxi", integer.binary, _B["xor"], False),
    0x10: ("vadc", "vxi", integer.carry, "adc", False),
    0x11: ("vmadc", "vxi", integer.carry, "madc", False),
    0x12: ("vsbc", "vx", integer.carry, "sbc", False),
    0x13: ("vmsbc", "vx", integer.carry, "msbc", False),
    0x18: ("vmseq", "vxi", integer.compare, _C["seq"], False),
    0x19: ("vmsne", "vxi", integer.compare, _C["sne"], False),
    0x1A: ("vmsltu", "vx", integer.compare, _C["sltu"], False),
    0x1B: ("vmslt", "vx", integer.compare, _C["slt"], False),
    0x1C: ("vmsleu", "vxi", integer.compare, _C["sleu"], False),
    0x1D: ("vmsle", "vxi", integer.compare, _C["sle"], False),
    0x1E: ("vmsgtu", "xi", integer.compare, _C["sgtu"], False),
    0x1F: ("vmsgt", "xi", integer.compare, _C["sgt"], False),
    0x20: ("vsaddu", "vxi", integer.fixed, "saddu", False),
    0x21: ("vsadd", "vxi", integer.fixed, "sadd", False),
    0x22: ("vssubu", "vx", integer.fixed, "ssubu", False),
    0x23: ("vssub", "vx", integer.fixed, "ssub", False),
    0x25: ("vsll", "vxi", integer.binary, _B["sll"], True),
    0x27: ("vsmul", "vx", integer.fixed, "smul", False),
    0x28: ("vsrl", "vxi", integer.binary, _B["srl"], True),
    0x29: ("vsra", "vxi", integer.binary, _B["sra"], True),
    0x2A: ("vssrl", "vxi", integer.fixed, "ssrl", True),
    0x2B: ("vssra", "vxi", integer.fixed, "ssra", True),
    0x2C: ("vnsrl", "vxi", integer.narrowing, "nsrl", True),
    0x2D: ("vnsra", "vxi", integer.narrowing, "nsra", True),
    0x2E: ("vnclipu", "vxi", integer.narrowing, "nclipu", True),
    0x2F: ("vnclip", "vxi", integer.narrowing, "nclip", True),
}

# -- OPM (OPMVV = 2, OPMVX = 6) ---------------------------------------------------------------

_OPM = {
    0x08: ("vaaddu", "vx", integer.fixed, "aaddu"),
    0x09: ("vaadd", "vx", integer.fixed, "aadd"),
    0x0A: ("vasubu", "vx", integer.fixed, "asubu"),
    0x0B: ("vasub", "vx", integer.fixed, "asub"),
    0x20: ("vdivu", "vx", integer.binary, _B["divu"]),
    0x21: ("vdiv", "vx", integer.binary, _B["div"]),
    0x22: ("vremu", "vx", integer.binary, _B["remu"]),
    0x23: ("vrem", "vx", integer.binary, _B["rem"]),
    0x24: ("vmulhu", "vx", integer.binary, _B["mulhu"]),
    0x25: ("vmul", "vx", integer.binary, _B["mul"]),
    0x26: ("vmulhsu", "vx", integer.binary, _B["mulhsu"]),
    0x27: ("vmulh", "vx", integer.binary, _B["mulh"]),
    0x29: ("vmadd", "vx", integer.multiply_add, "madd"),
    0x2B: ("vnmsub", "vx", integer.multiply_add, "nmsub"),
    0x2D: ("vmacc", "vx", integer.multiply_add, "macc"),
    0x2F: ("vnmsac", "vx", integer.multiply_add, "nmsac"),
    0x30: ("vwaddu", "vx", integer.widening, "waddu"),
    0x31: ("vwadd", "vx", integer.widening, "wadd"),
    0x32: ("vwsubu", "vx", integer.widening, "wsubu"),
    0x33: ("vwsub", "vx", integer.widening, "wsub"),
    0x34: ("vwaddu.w", "vx", integer.widening, "waddu.w"),
    0x35: ("vwadd.w", "vx", integer.widening, "wadd.w"),
    0x36: ("vwsubu.w", "vx", integer.widening, "wsubu.w"),
    0x37: ("vwsub.w", "vx", integer.widening, "wsub.w"),
    0x38: ("vwmulu", "vx", integer.widening, "wmulu"),
    0x3A: ("vwmulsu", "vx", integer.widening, "wmulsu"),
    0x3B: ("vwmul", "vx", integer.widening, "wmul"),
    0x3C: ("vwmaccu", "vx", integer.widening, "wmaccu"),
    0x3D: ("vwmacc", "vx", integer.widening, "wmacc"),
    0x3E: ("vwmaccus", "x", integer.widening, "wmaccus"),
    0x3F: ("vwmaccsu", "vx", integer.widening, "wmaccsu"),
}

_RED = ("sum", "and", "or", "xor", "minu", "min", "maxu", "max")
_MASK_OPS = ("andn", "and", "or", "xor", "orn", "nand", "nor", "xnor")
_EXT = {
    2: (8, False),
    3: (8, True),
    4: (4, False),
    5: (4, True),
    6: (2, False),
    7: (2, True),
}

# -- OPF (OPFVV = 1, OPFVF = 5) -------------------------------------------------------------------

_OPF = {
    0x00: ("vfadd", "vf", fpu.binary, "add", False),
    0x02: ("vfsub", "vf", fpu.binary, "sub", False),
    0x04: ("vfmin", "vf", fpu.binary, "min", False),
    0x06: ("vfmax", "vf", fpu.binary, "max", False),
    0x08: ("vfsgnj", "vf", fpu.binary, "sgnj", False),
    0x09: ("vfsgnjn", "vf", fpu.binary, "sgnjn", False),
    0x0A: ("vfsgnjx", "vf", fpu.binary, "sgnjx", False),
    0x20: ("vfdiv", "vf", fpu.binary, "div", False),
    0x21: ("vfrdiv", "f", fpu.binary, "div", True),
    0x24: ("vfmul", "vf", fpu.binary, "mul", False),
    0x27: ("vfrsub", "f", fpu.binary, "sub", True),
    0x18: ("vmfeq", "vf", fpu.compare, "eq", None),
    0x19: ("vmfle", "vf", fpu.compare, "le", None),
    0x1B: ("vmflt", "vf", fpu.compare, "lt", None),
    0x1C: ("vmfne", "vf", fpu.compare, "ne", None),
    0x1D: ("vmfgt", "f", fpu.compare, "gt", None),
    0x1F: ("vmfge", "f", fpu.compare, "ge", None),
    0x28: ("vfmadd", "vf", fpu.fma, "madd", None),
    0x29: ("vfnmadd", "vf", fpu.fma, "nmadd", None),
    0x2A: ("vfmsub", "vf", fpu.fma, "msub", None),
    0x2B: ("vfnmsub", "vf", fpu.fma, "nmsub", None),
    0x2C: ("vfmacc", "vf", fpu.fma, "macc", None),
    0x2D: ("vfnmacc", "vf", fpu.fma, "nmacc", None),
    0x2E: ("vfmsac", "vf", fpu.fma, "msac", None),
    0x2F: ("vfnmsac", "vf", fpu.fma, "nmsac", None),
    0x30: ("vfwadd", "vf", fpu.widening, "wadd", None),
    0x32: ("vfwsub", "vf", fpu.widening, "wsub", None),
    0x34: ("vfwadd.w", "vf", fpu.widening, "wadd.w", None),
    0x36: ("vfwsub.w", "vf", fpu.widening, "wsub.w", None),
    0x38: ("vfwmul", "vf", fpu.widening, "wmul", None),
    0x3B: ("vfwmaccbf16", "vf", fpu.widening, "wmaccbf16", None),
    0x3C: ("vfwmacc", "vf", fpu.widening, "wmacc", None),
    0x3D: ("vfwnmacc", "vf", fpu.widening, "wnmacc", None),
    0x3E: ("vfwmsac", "vf", fpu.widening, "wmsac", None),
    0x3F: ("vfwnmsac", "vf", fpu.widening, "wnmsac", None),
}
_FRED = {
    0x01: "usum",
    0x03: "osum",
    0x05: "min",
    0x07: "max",
    0x31: "wusum",
    0x33: "wosum",
}
_FUNARY1 = {0x00: "sqrt", 0x04: "rsqrt7", 0x05: "rec7", 0x10: "class"}

_SUFFIX = {"v": "vv", "x": "vx", "i": "vi", "u": "vi", "f": "vf"}


def _vset(vs, rf, memory, form, rd, rs1, rs2, zimm):
    if form == "vsetvl":
        vtype, avl, src = rf[rs2].read_uint(), rf[rs1].read_uint(), rs1
    elif form == "vsetvli":
        vtype, avl, src = zimm, rf[rs1].read_uint(), rs1
    else:  # vsetivli: AVL is the immediate, never "x0"
        vtype, avl, src = zimm, rs1, 1
    vl = vs.set_vl(rd, src, avl, vtype, "vsetvl" if form == "vsetvl" else "vsetvli")
    if rd:
        write_reg(rf, rd, vl)
    return True


def _decode_cfg(instr, rd, rs1, rs2):
    if not instr >> 31:
        return _vset, ("vsetvli", rd, rs1, 0, (instr >> 20) & 0x7FF), "vsetvli"
    if instr >> 30 == 0x3:
        return _vset, ("vsetivli", rd, rs1, 0, (instr >> 20) & 0x3FF), "vsetivli"
    if (instr >> 25) == 0x40:
        return _vset, ("vsetvl", rd, rs1, rs2, 0), "vsetvl"
    return None


def _decode_opv(instr):
    funct3 = (instr >> 12) & 0x7
    funct6 = instr >> 26
    vm = (instr >> 25) & 1
    vs2 = (instr >> 20) & 0x1F
    src = (instr >> 15) & 0x1F
    vd = (instr >> 7) & 0x1F
    if funct3 == 7:
        return _decode_cfg(instr, vd, src, vs2)

    if funct3 in (0, 3, 4):  # OPI
        form = {0: "v", 3: "i", 4: "x"}[funct3]
        if funct6 == 0x0C and form in "vxi":
            kind = "u" if form == "i" else form
            return permute.gather, (vd, vs2, kind, src, vm), f"vrgather.{_SUFFIX[form]}"
        if funct6 == 0x0E:
            if form == "v":
                return (
                    partial(permute.gather, ei16=True),
                    (vd, vs2, "v", src, vm),
                    "vrgatherei16.vv",
                )
            kind = "u" if form == "i" else "x"
            return (
                permute.slide,
                ("up", vd, vs2, kind, src, vm),
                f"vslideup.{_SUFFIX[form]}",
            )
        if funct6 == 0x0F and form != "v":
            kind = "u" if form == "i" else "x"
            return (
                permute.slide,
                ("down", vd, vs2, kind, src, vm),
                f"vslidedown.{_SUFFIX[form]}",
            )
        if funct6 == 0x17:
            kind = form
            return (
                integer.merge,
                (vd, vs2, kind, src, vm),
                ("vmv.v." if vm else "vmerge.") + _SUFFIX[form],
            )
        if funct6 == 0x27 and form == "i":
            if not vm or src not in (0, 1, 3, 7):
                return None
            return permute.move_whole, (vd, vs2, src + 1), f"vmv{src + 1}r.v"
        if funct6 in (0x30, 0x31) and form == "v":
            name = "wsumu" if funct6 == 0x30 else "wsum"
            return integer.reduction, (name, vd, vs2, src, vm), f"vwred{name[1:]}.vs"
        entry = _OPI.get(funct6)
        if entry is None or form not in entry[1]:
            return None
        stem, _, handler, extra, uimm = entry
        kind = "u" if form == "i" and uimm else form
        return handler, (extra, vd, vs2, kind, src, vm), f"{stem}.{_SUFFIX[form]}"

    if funct3 in (2, 6):  # OPM
        form = "v" if funct3 == 2 else "x"
        if funct6 <= 0x07 and form == "v":
            return (
                integer.reduction,
                (_RED[funct6], vd, vs2, src, vm),
                f"vred{_RED[funct6]}.vs",
            )
        if funct6 in (0x0E, 0x0F) and form == "x":
            name = "1up" if funct6 == 0x0E else "1down"
            return permute.slide, (name, vd, vs2, "x", src, vm), f"vslide{name}.vx"
        if funct6 == 0x10:
            if form == "v":
                if not vm and src == 0:
                    return None
                if src == 0:
                    return permute.mv_x_s, (vd, vs2), "vmv.x.s"
                if src in (0x10, 0x11):
                    name = "cpop" if src == 0x10 else "first"
                    return permute.cpop_first, (name, vd, vs2, vm), f"v{name}.m"
                return None
            if vs2 == 0 and vm:
                return permute.mv_s_x, (vd, src), "vmv.s.x"
            return None
        if funct6 == 0x12 and form == "v":
            if src not in _EXT:
                return None
            factor, sign = _EXT[src]
            return (
                integer.extension,
                (vd, vs2, factor, sign, vm),
                f"v{'s' if sign else 'z'}ext.vf{factor}",
            )
        if funct6 == 0x14 and form == "v":
            if src in (1, 2, 3):
                name = {1: "sbf", 2: "sof", 3: "sif"}[src]
                return permute.set_first, (name, vd, vs2, vm), f"vms{name}.m"
            if src == 0x10:
                return permute.iota, (vd, vs2, vm), "viota.m"
            if src == 0x11 and vs2 == 0:
                return permute.vid, (vd, vm), "vid.v"
            return None
        if funct6 == 0x17 and form == "v":
            if not vm:
                return None
            return permute.compress, (vd, vs2, src), "vcompress.vm"
        if 0x18 <= funct6 <= 0x1F and form == "v":
            if not vm:
                return None
            name = _MASK_OPS[funct6 - 0x18]
            return permute.mask_logical, (name, vd, vs2, src), f"vm{name}.mm"
        entry = _OPM.get(funct6)
        if entry is None or form not in entry[1]:
            return None
        stem, _, handler, extra = entry
        return handler, (extra, vd, vs2, form, src, vm), f"{stem}.{_SUFFIX[form]}"

    # OPF
    form = "v" if funct3 == 1 else "f"
    if funct6 in _FRED and form == "v":
        name = _FRED[funct6]
        return (
            fpu.reduction,
            (name, vd, vs2, src, vm),
            f"vf{'w' if name[0] == 'w' else ''}red{name.lstrip('w')}.vs",
        )
    if funct6 in (0x0E, 0x0F) and form == "f":
        name = "1up" if funct6 == 0x0E else "1down"
        return (
            partial(permute.slide, fp_ok=fpu.fp_ok),
            (name, vd, vs2, "f", src, vm),
            f"vfslide{name}.vf",
        )
    if funct6 == 0x10:
        if not vm:
            return None
        if form == "v" and src == 0:
            return permute.fmv_f_s, (vd, vs2, fpu.fp_ok), "vfmv.f.s"
        if form == "f" and vs2 == 0:
            return permute.fmv_s_f, (vd, src, fpu.fp_ok), "vfmv.s.f"
        return None
    if funct6 == 0x12 and form == "v":
        if src not in fpu.CONVERSIONS:
            return None
        return fpu.convert, (src, vd, vs2, vm), fpu.CONVERSIONS[src][0]
    if funct6 == 0x13 and form == "v":
        op = _FUNARY1.get(src)
        if op is None:
            return None
        return fpu.unary, (op, vd, vs2, vm), f"vf{op}.v"
    if funct6 == 0x17 and form == "f":
        return fpu.merge, (vd, vs2, src, vm), "vfmv.v.f" if vm else "vfmerge.vfm"
    entry = _OPF.get(funct6)
    if entry is None or form not in entry[1]:
        return None
    stem, _, handler, extra, reverse = entry
    args = (extra, vd, vs2, form, src, vm)
    if handler is fpu.binary:
        return partial(fpu.binary, reverse=reverse), args, f"{stem}.{_SUFFIX[form]}"
    return handler, args, f"{stem}.{_SUFFIX[form]}"


def decode(instr):
    """``(handler, args, mnemonic)`` for a vector word, or None if RVV 1.0 does not define it."""
    opcode = instr & 0x7F
    if opcode == 0x57:
        return _decode_opv(instr)
    if opcode in (0x07, 0x27):
        return mem.decode(instr)
    return None


#: The LOAD-FP / STORE-FP widths that are vector (bit w set): 0, 5, 6, 7.
_VEC_WIDTHS = 0b11100001


class RV64_V_ISA(RV_ISA):
    """RVV 1.0 (V / Zve64d, with Zvfh, Zvfbfmin and Zvfbfwma) on the core's :class:`VectorState`.

    The state lives on the register file as ``register_file.vector`` (and on
    the core as ``core.vector``), put there by
    :func:`framework.pe.rv.isa.v.attach`.
    """

    #: Major opcodes this ISA may claim (for an opcode-indexed dispatch table).
    OPCODES = frozenset({0x07, 0x27, 0x57})
    #: The counter hub's instruction class.
    INSTR_CLASS = "vector"
    _decoded = {}

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        opcode = instr & 0x7F
        if opcode != 0x57:
            if (opcode != 0x07 and opcode != 0x27) or not (
                _VEC_WIDTHS >> ((instr >> 12) & 0x7)
            ) & 1:
                return False
        entry = cls._decoded.get(instr)
        if entry is None:
            entry = cls._decoded[instr] = decode(instr) or False
        if entry is False:
            return False
        vs = getattr(register_file, "vector", None)
        if vs is None:
            raise VectorStateError(
                f"vector instruction {instr:#010x} on a core with no vector state; "
                f"attach one with framework.pe.rv.isa.v.attach(core, vlen=..., elen=...)"
            )
        handler, args, text = entry
        if not handler(vs, register_file, memory_space, *args):
            return False
        vs.vstart = 0
        vs.instructions += 1
        if snoop:
            RV_ISA.print_snoop(
                snoop, f"{text} {args}", f"vl={vs.vl} vtype={vs.vtype:#x}"
            )
        return True
