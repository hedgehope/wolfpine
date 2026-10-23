"""F and D floating-point for the RV64 core (FLEN = 64), with full IEEE semantics.

This is the general-purpose RISC-V F / D extension, for Naiad's RV64 hart. It is
unrelated to the Blackhole baby cores, whose partial F unit (no ``fdiv.s`` /
``fsqrt.s``) stays behind ``guard_isa.RV_F_GUARD_ISA``.

``RV64_F_ISA`` claims the single-precision encodings (``flw`` / ``fsw``, OP-FP
and the fused multiply-adds with fmt = S); ``RV64_D_ISA`` the double-precision
ones (``fld`` / ``fsd``, fmt = D) plus the two cross-format conversions
``fcvt.s.d`` / ``fcvt.d.s``. D requires F, which the Naiad design validator
enforces. Arithmetic is exact-then-round-once in :mod:`fd_softfloat`, so every
rounding mode and every fflags bit is the spec's.

* **Registers.** 32 eight-byte f-registers at ``FP_REGISTER_BASE``, and
  ``fcsr`` (``frm`` in bits 7:5, ``fflags`` in 4:0) at ``FCSR_INDEX``, the same
  register ``CSRFile64`` exposes as ``fflags`` / ``frm`` / ``fcsr``.
* **NaN-boxing.** A single in a 64-bit register is held with the upper 32 bits
  all ones. Every single-precision result is written boxed; every
  single-precision operand that is not properly boxed reads as the canonical
  NaN. ``fsw`` and ``fmv.x.w`` are transfers and move the low 32 bits
  unchecked, per the spec.
* **Rounding.** An instruction's rm field picks RNE / RTZ / RDN / RUP / RMM, or
  DYN (``frm``). A reserved rm (5, 6), or DYN with ``frm`` holding 5-7, is an
  illegal instruction: the ISA does not claim it, so the core reports an
  unknown instruction, Wolfpine's stand-in for the trap.
* **Flags** accrue into ``fflags`` (NX / UF / OF / DZ / NV); nothing clears them
  but software.
* **mstatus.FS** is not modelled: Wolfpine models no traps, so FP instructions
  run whatever FS holds, and FS is never set dirty.
"""

from framework.pe.rv.isa import fd_softfloat as sf
from framework.pe.rv.isa.i64_isa import sext32, write_reg
from framework.pe.rv.isa.i_isa import _imm_i, _imm_s
from framework.pe.rv.isa.rv_isa import RV_ISA
from framework.pe.rv.rv32 import FCSR_INDEX, FP_REGISTER_BASE

_M64 = 0xFFFFFFFFFFFFFFFF
_M32 = 0xFFFFFFFF
_BOX32 = 0xFFFFFFFF00000000

#: The opcodes any F / D / Zfbfmin instruction uses: LOAD-FP, STORE-FP, the four
#: fused multiply-adds, and OP-FP.
FP_OPCODES = frozenset({0x07, 0x27, 0x43, 0x47, 0x4B, 0x4F, 0x53})

# Fused multiply-add opcode -> (mnemonic, negate product, negate addend).
_FMA = {
    0x43: ("fmadd", False, False),
    0x47: ("fmsub", False, True),
    0x4B: ("fnmsub", True, False),
    0x4F: ("fnmadd", True, True),
}

# fcvt int-side selector (the rs2 field) -> (width, signed, suffix).
_INT_KINDS = {
    0: (32, True, "w"),
    1: (32, False, "wu"),
    2: (64, True, "l"),
    3: (64, False, "lu"),
}


def fname(idx):
    return f"f{idx}"


def read_f(register_file, idx):
    """The raw 64 bits of f-register ``idx``."""
    return register_file[FP_REGISTER_BASE + idx].read_uint()


def write_f(register_file, idx, value):
    register_file[FP_REGISTER_BASE + idx].write((value & _M64).to_bytes(8, "little"))


def unbox32(raw):
    """A NaN-boxed single's 32 bits, or the canonical NaN if it is not boxed."""
    return raw & _M32 if raw >> 32 == _M32 else sf.F32.canonical_nan


def resolve_rm(register_file, rm):
    """The effective rounding mode for an rm field, or ``None`` if it is illegal."""
    if rm == sf.DYN:
        rm = (register_file[FCSR_INDEX].read_uint() >> 5) & 0x7
    return rm if rm <= sf.RMM else None


def raise_flags(register_file, flags):
    """Accrue ``flags`` into fcsr.fflags."""
    if flags:
        reg = register_file[FCSR_INDEX]
        reg.write((reg.read_uint() | flags).to_bytes(4, "little"))


def _rm_suffix(rm):
    return "" if rm == sf.DYN else f", {sf.ROUNDING_MODE_NAMES[rm]}"


class _FloatISA(RV_ISA):
    """One precision's share of the F / D encodings. Subclasses set the format."""

    #: The fmt field value (0 = S, 1 = D) and load/store width (2 = W, 3 = D).
    FMT_CODE = 0
    WIDTH = 2
    FMT = sf.F32
    SUFFIX = "s"

    @classmethod
    def _read(cls, rf, idx):
        raw = read_f(rf, idx)
        return unbox32(raw) if cls.FMT_CODE == 0 else raw

    @classmethod
    def _write(cls, rf, idx, bits):
        write_f(rf, idx, (_BOX32 | bits) if cls.FMT_CODE == 0 else bits)

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        opcode = instr & 0x7F
        if opcode not in FP_OPCODES:
            return False
        if opcode == 0x53:
            if (instr >> 25) & 0x3 != cls.FMT_CODE:
                return cls._op_fp_other(instr, register_file, snoop)
            return cls._op_fp(instr, register_file, snoop)
        if opcode == 0x07 or opcode == 0x27:
            if (instr >> 12) & 0x7 != cls.WIDTH:
                return False
            if opcode == 0x07:
                return cls._load(instr, register_file, memory_space, snoop)
            return cls._store(instr, register_file, memory_space, snoop)
        if (instr >> 25) & 0x3 != cls.FMT_CODE:
            return False
        return cls._fma(instr, register_file, snoop)

    @classmethod
    def _op_fp_other(cls, instr, rf, snoop):
        """OP-FP with another fmt field; D overrides this for ``fcvt.s.d``."""
        return False

    # -- loads and stores ------------------------------------------------------

    @classmethod
    def _load(cls, instr, rf, mem, snoop):
        rd = (instr >> 7) & 0x1F
        rs1 = (instr >> 15) & 0x1F
        imm = _imm_i(instr)
        addr = (rf[rs1].read_uint() + imm) & _M64
        size = 4 if cls.FMT_CODE == 0 else 8
        cls._write(rf, rd, int.from_bytes(mem.read(addr, size), "little"))
        if snoop:
            name = "flw" if cls.FMT_CODE == 0 else "fld"
            RV_ISA.print_snoop(
                snoop, f"{name} {fname(rd)}, {imm}({cls.get_reg_name(rs1)})", hex(addr)
            )
        return True

    @classmethod
    def _store(cls, instr, rf, mem, snoop):
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        imm = _imm_s(instr)
        addr = (rf[rs1].read_uint() + imm) & _M64
        raw = read_f(rf, rs2)  # a transfer: the low bits, boxed or not
        if cls.FMT_CODE == 0:
            mem.write(addr, (raw & _M32).to_bytes(4, "little"))
        else:
            mem.write(addr, raw.to_bytes(8, "little"))
        if snoop:
            name = "fsw" if cls.FMT_CODE == 0 else "fsd"
            RV_ISA.print_snoop(
                snoop, f"{name} {fname(rs2)}, {imm}({cls.get_reg_name(rs1)})", hex(addr)
            )
        return True

    # -- fused multiply-add ----------------------------------------------------

    @classmethod
    def _fma(cls, instr, rf, snoop):
        rm = resolve_rm(rf, (instr >> 12) & 0x7)
        if rm is None:
            return False
        rd = (instr >> 7) & 0x1F
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        rs3 = instr >> 27
        name, neg_product, neg_addend = _FMA[instr & 0x7F]
        bits, flags = sf.fma(
            cls._read(rf, rs1),
            cls._read(rf, rs2),
            cls._read(rf, rs3),
            cls.FMT,
            rm,
            neg_product,
            neg_addend,
        )
        cls._write(rf, rd, bits)
        raise_flags(rf, flags)
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"{name}.{cls.SUFFIX} {fname(rd)}, {fname(rs1)}, {fname(rs2)}, "
                f"{fname(rs3)}{_rm_suffix((instr >> 12) & 0x7)}",
                hex(bits),
            )
        return True

    # -- OP-FP -----------------------------------------------------------------

    @classmethod
    def _op_fp(cls, instr, rf, snoop):
        funct5 = instr >> 27
        funct3 = (instr >> 12) & 0x7
        rd = (instr >> 7) & 0x1F
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        fmt = cls.FMT
        sfx = cls.SUFFIX
        # Result kinds: an f-register result, an x-register result.
        fres = xres = None
        flags = 0

        if funct5 <= 0x03 or funct5 == 0x0B:  # fadd / fsub / fmul / fdiv / fsqrt
            if funct5 == 0x0B and rs2 != 0:
                return False
            rm = resolve_rm(rf, funct3)
            if rm is None:
                return False
            a = cls._read(rf, rs1)
            if funct5 == 0x0B:
                fres, flags = sf.sqrt(a, fmt, rm)
                text = f"fsqrt.{sfx} {fname(rd)}, {fname(rs1)}"
            else:
                b = cls._read(rf, rs2)
                if funct5 == 0x00:
                    fres, flags = sf.add(a, b, fmt, rm)
                elif funct5 == 0x01:
                    fres, flags = sf.add(a, b, fmt, rm, negate_b=True)
                elif funct5 == 0x02:
                    fres, flags = sf.mul(a, b, fmt, rm)
                else:
                    fres, flags = sf.div(a, b, fmt, rm)
                op = ("fadd", "fsub", "fmul", "fdiv")[funct5]
                text = f"{op}.{sfx} {fname(rd)}, {fname(rs1)}, {fname(rs2)}"
            text += _rm_suffix(funct3)
        elif funct5 == 0x04:  # fsgnj / fsgnjn / fsgnjx
            if funct3 > 2:
                return False
            a = cls._read(rf, rs1)
            b = cls._read(rf, rs2)
            top = fmt.bits - 1
            if funct3 == 0:
                sign = b >> top
            elif funct3 == 1:
                sign = (b >> top) ^ 1
            else:
                sign = (a >> top) ^ (b >> top)
            fres = (a & ((1 << top) - 1)) | (sign << top)
            op = ("fsgnj", "fsgnjn", "fsgnjx")[funct3]
            text = f"{op}.{sfx} {fname(rd)}, {fname(rs1)}, {fname(rs2)}"
        elif funct5 == 0x05:  # fmin / fmax
            if funct3 > 1:
                return False
            fres, flags = sf.minmax(cls._read(rf, rs1), cls._read(rf, rs2), fmt, funct3)
            op = ("fmin", "fmax")[funct3]
            text = f"{op}.{sfx} {fname(rd)}, {fname(rs1)}, {fname(rs2)}"
        elif funct5 == 0x14:  # fle / flt / feq
            if funct3 > 2:
                return False
            op = ("le", "lt", "eq")[funct3]
            xres, flags = sf.compare(cls._read(rf, rs1), cls._read(rf, rs2), fmt, op)
            text = f"f{op}.{sfx} {cls.get_reg_name(rd)}, {fname(rs1)}, {fname(rs2)}"
        elif funct5 == 0x18:  # fcvt.{w,wu,l,lu}.fmt
            if rs2 > 3:
                return False
            rm = resolve_rm(rf, funct3)
            if rm is None:
                return False
            width, signed, isfx = _INT_KINDS[rs2]
            value, flags = sf.to_int(cls._read(rf, rs1), fmt, rm, width, signed)
            # The W forms sign-extend their 32-bit result, unsigned ones included.
            xres = sext32(value) if width == 32 else value & _M64
            text = f"fcvt.{isfx}.{sfx} {cls.get_reg_name(rd)}, {fname(rs1)}{_rm_suffix(funct3)}"
        elif funct5 == 0x1A:  # fcvt.fmt.{w,wu,l,lu}
            if rs2 > 3:
                return False
            rm = resolve_rm(rf, funct3)
            if rm is None:
                return False
            width, signed, isfx = _INT_KINDS[rs2]
            value = rf[rs1].read_uint() & ((1 << width) - 1)
            if signed and value >> (width - 1):
                value -= 1 << width
            fres, flags = sf.from_int(value, fmt, rm)
            text = f"fcvt.{sfx}.{isfx} {fname(rd)}, {cls.get_reg_name(rs1)}{_rm_suffix(funct3)}"
        elif funct5 == 0x1C and rs2 == 0:  # fmv.x.fmt / fclass.fmt
            if funct3 == 0:
                raw = read_f(rf, rs1)  # a transfer: no NaN-box check
                xres = sext32(raw) if cls.FMT_CODE == 0 else raw
                text = f"fmv.x.{'w' if cls.FMT_CODE == 0 else 'd'} {cls.get_reg_name(rd)}, {fname(rs1)}"
            elif funct3 == 1:
                xres = sf.fclass(cls._read(rf, rs1), fmt)
                text = f"fclass.{sfx} {cls.get_reg_name(rd)}, {fname(rs1)}"
            else:
                return False
        elif funct5 == 0x1E and rs2 == 0 and funct3 == 0:  # fmv.fmt.x
            x = rf[rs1].read_uint()
            fres = x & _M32 if cls.FMT_CODE == 0 else x
            text = f"fmv.{'w' if cls.FMT_CODE == 0 else 'd'}.x {fname(rd)}, {cls.get_reg_name(rs1)}"
        else:
            return False

        if fres is not None:
            cls._write(rf, rd, fres)
        elif rd != 0:
            write_reg(rf, rd, xres)
        raise_flags(rf, flags)
        if snoop:
            RV_ISA.print_snoop(snoop, text, hex(fres if fres is not None else xres))
        return True


class RV64_F_ISA(_FloatISA):
    """Single precision, NaN-boxed in the 64-bit f-registers."""


class RV64_D_ISA(_FloatISA):
    """Double precision, plus ``fcvt.s.d`` / ``fcvt.d.s``."""

    FMT_CODE = 1
    WIDTH = 3
    FMT = sf.F64
    SUFFIX = "d"

    @classmethod
    def _op_fp(cls, instr, rf, snoop):
        if instr >> 27 == 0x08:  # fcvt.d.s
            return cls._convert(instr, rf, snoop, src_code=0)
        return super()._op_fp(instr, rf, snoop)

    @classmethod
    def _op_fp_other(cls, instr, rf, snoop):
        if instr >> 27 == 0x08 and (instr >> 25) & 0x3 == 0:  # fcvt.s.d
            return cls._convert(instr, rf, snoop, src_code=1)
        return False

    @classmethod
    def _convert(cls, instr, rf, snoop, src_code):
        """``fcvt.d.s`` (``src_code`` 0) or ``fcvt.s.d`` (1)."""
        if (instr >> 20) & 0x1F != src_code:
            return False
        rm = resolve_rm(rf, (instr >> 12) & 0x7)
        if rm is None:
            return False
        rd = (instr >> 7) & 0x1F
        rs1 = (instr >> 15) & 0x1F
        if src_code == 0:
            bits, flags = sf.convert(unbox32(read_f(rf, rs1)), sf.F32, sf.F64, rm)
            write_f(rf, rd, bits)
            text = f"fcvt.d.s {fname(rd)}, {fname(rs1)}"
        else:
            bits, flags = sf.convert(read_f(rf, rs1), sf.F64, sf.F32, rm)
            write_f(rf, rd, _BOX32 | bits)
            text = f"fcvt.s.d {fname(rd)}, {fname(rs1)}"
        raise_flags(rf, flags)
        if snoop:
            RV_ISA.print_snoop(snoop, text + _rm_suffix((instr >> 12) & 0x7), hex(bits))
        return True
