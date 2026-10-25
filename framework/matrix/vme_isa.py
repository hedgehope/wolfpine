"""VME (Zvt v0.4): the vector-coupled matrix mode as an RV64 ISA executor.

The configuration, move, tile-zero and discard instructions are OP-V words RVV
1.0 leaves reserved (and :class:`~framework.pe.rv.isa.v.RV64_V_ISA` declines),
the tile loads/stores are LOAD-FP / STORE-FP with ``width = 111, mew = 1``
(also declined by RVV), and the multiplies are OP-VE (0x77), which no other ISA
claims. :class:`RV64_VME_ISA` sits after RVV in the core's chain and is offered
exactly those words; one in Zvt's space that this implementation cannot execute
(a reserved type row, an unlisted extension, ``mtwiden = 0``, a misaligned
group, ``vstart != 0`` for a multiply) raises
:class:`~framework.matrix.unit.IllegalMatrixInstruction`, as hardware would trap.

**State** (:class:`VMEUnit`): the ``16 * TE * TE``-byte tile buffer, viewed per
TEW through the draft's punning function (precomputed byte-index maps, so a
tile is one gather and one scatter and every cross-TEW read is bit-exact), and
the ``mtype`` CSR (``tm``, ``tk``, ``mtwiden``), readable at 0xC23. A and B are
views onto the hart's ``VectorState``; tn is ``min(vl, ETE)``. ``vset{i}vl{i}``
write ``mtype = 0`` through ``VectorState.install_vtype_extension``'s hook,
which also makes Zvfbfa's ``altfmt`` (vtype bit 8) legal when a listed
extension needs it (bf16, int8 signedness, OFP8 B type).

Policies: tile elements outside ``[0, tm) x [0, tn)`` (the tail) are left
unchanged, as are tail elements of a ``vtmv.v.t`` destination (both legal under
the agnostic policies); the moves, loads and stores honour ``vstart`` and leave
it 0; a misaligned ``vtle``/``vtse`` address raises
:class:`~framework.matrix.unit.MatrixAccessFault` (the draft's "address-misaligned
or access-fault exception"). Semantics: :mod:`framework.matrix.vme`; numerics:
:mod:`framework.matrix.vme_numerics` and
:func:`~framework.matrix.ime_numerics.int_mac`.
"""

import numpy as np

from framework.matrix import vme
from framework.matrix.formats import get_format
from framework.matrix.ime_numerics import int_mac
from framework.matrix.unit import IllegalMatrixInstruction, MatrixAccessFault, read_rows
from framework.matrix.vme import KMAX, VMEError
from framework.matrix.vme_numerics import (
    zvt_tile_mac,  # also registers tile_mac's "zvt"
)
from framework.pe.rv.isa.i64_isa import write_reg
from framework.pe.rv.isa.rv_isa import RV_ISA
from framework.pe.rv.rv32 import FCSR_INDEX

#: The unit's counters (``hart<N>.matrix.<key>``), in report order.
COUNTER_NAMES = (
    "mma",
    "macs",
    "vrf_a_read_bytes",
    "vrf_b_read_bytes",
    "vec_to_tile_bytes",
    "tile_to_vec_bytes",
    "mem_to_tile_bytes",
    "tile_to_mem_bytes",
    "tile_zero",
    "config",
    "discard",
)
_UINT = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
_INT_DTYPES = {
    "int8": np.dtype("<i1"),
    "uint8": np.dtype("<u1"),
    "int32": np.dtype("<i4"),
}
_MTW = {1: 1, 2: 2, 4: 3}


def _dtype(name):
    return _INT_DTYPES[name] if name in _INT_DTYPES else get_format(name).dtype


class VMEUnit:
    """One hart's Zvt state (tiles + mtype), configuration and counters."""

    mode = "vme"

    def __init__(self, spec, vector_state, xlen=64):
        if xlen != 64:
            raise VMEError("VME is modelled on RV64 only")
        if vector_state.vlen != spec.vlen:
            raise VMEError(
                f"the VME spec is for VLEN {spec.vlen}, the vector unit has {vector_state.vlen}"
            )
        self.spec = spec
        self.vs = vector_state
        self.te = spec.te
        self.xmask = (1 << xlen) - 1
        self.extensions = frozenset(spec.extensions)
        self.tews = frozenset(spec.tews)
        for pair in (
            spec.format_pairs
        ):  # formats resolve here, so a missing ml_dtypes fails at build
            for name in pair:
                _dtype(name)
        self.buf = np.zeros(16 * self.te * self.te, dtype=np.uint8)
        self.maps = {tew: vme.tile_byte_index(self.te, tew) for tew in vme.TEWS}
        mask = (
            (1 << vme.ALTFMT_SHIFT)
            if any(e in self.extensions for e in vme.ALTFMT_EXTENSIONS)
            else 0
        )
        vector_state.install_vtype_extension(mask, self.on_vset)
        self.reset()

    # -- lifecycle ---------------------------------------------------------------

    def reset(self):
        self.buf[:] = 0
        self.tm = self.tk = self.mtwiden = 0
        self.stats = dict.fromkeys(COUNTER_NAMES, 0)

    def counters(self):
        return dict(self.stats)

    def illegal(self, instr, why):
        return IllegalMatrixInstruction(f"illegal VME instruction {instr:#010x}: {why}")

    @property
    def mtype(self):
        return (self.tm << 10) | (self.tk << 5) | self.mtwiden

    def read_csr(self, addr):
        if addr != vme.CSR_MTYPE:
            raise KeyError(addr)
        return self.mtype

    def on_vset(self, requested, old, form):
        """Every ``vset{i}vl{i}`` writes ``mtype = 0``; ``msetmtype`` (form ``"msetmtype"``) sets it itself."""
        if form != "msetmtype":
            self.tm = self.tk = self.mtwiden = 0
        return requested

    # -- the configured geometry ----------------------------------------------------

    @property
    def twiden(self):
        return vme.twiden(self.mtwiden)

    def kmax(self):
        return KMAX.get((self.vs.sew, self.twiden), 1)

    def ete(self, tew):
        return vme.ete(self.te, tew)

    def tile(self, tile, tew):
        """Tile ``tile`` at ``tew`` as an ``ETE x ETE`` array (a copy)."""
        idx = self.maps[tew][tile]
        e = idx.shape[0]
        dt = np.dtype(_UINT[tew])
        return self.buf[idx].view(dt).reshape(e, e)

    def set_tile(self, tile, tew, values, rows, cols):
        """Write ``values`` (``rows x cols`` of TEW-wide elements) to the top-left of ``tile``."""
        idx = self.maps[tew][tile][:rows, :cols]
        raw = np.ascontiguousarray(values).view(np.uint8).reshape(rows, cols, tew // 8)
        self.buf[idx] = raw

    def _tile_number(self, instr, t, tew):
        if t % vme.tile_step(tew):
            raise self.illegal(
                instr,
                f"tile {t} is not a TEW-{tew} tile (multiples of {vme.tile_step(tew)})",
            )
        return t

    # -- execution -----------------------------------------------------------------

    def execute(self, insn, instr, register_file, memory):
        name = insn.name
        if name.startswith("mset"):
            self.configure(insn, instr, register_file)
        elif name in ("vtle", "vtse"):
            self.tile_memory(insn, instr, register_file, memory)
        elif name.startswith("vtmv"):
            self.move(insn, instr, register_file)
        elif name == "vtzero":
            self.zero(insn, instr)
        elif name == "vtdiscard":
            if self.vs.vill:
                raise self.illegal(instr, "vtdiscard with vtype.vill = 1")
            self.stats["discard"] += 1
        else:
            self.multiply(insn, instr, register_file)

    def configure(self, insn, instr, rf):
        vs, name = self.vs, insn.name
        self.stats["config"] += 1
        if name in ("msetmtype", "msetmtypei"):
            if name == "msetmtypei":
                mt = insn.field(instr, "mtwiden")
                vt = (insn.field(instr, "vsew") << 3) | (
                    insn.field(instr, "altfmt") << vme.ALTFMT_SHIFT
                )
                tm = tk = 0
            else:
                raw = rf[insn.field(instr, "rs1")].read_uint()
                vt = rf[insn.field(instr, "rs2")].read_uint()
                mt, tm, tk = raw & 0x3, (raw >> 10) & 0x3FFF, (raw >> 5) & 0x7
            ok = vs.decode_vtype(vt) is not None
            if ok and mt:
                sew = 8 << ((vt >> 3) & 0x7)
                tw = vme.twiden(mt)
                tew = sew * tw
                lmul = vme.config_lmul(sew, tw, self.te, vs.vlen)
                vt = (vt & ~0xC7) | (lmul.bit_length() - 1) | (1 << 6) | (1 << 7)
                if tew > vs.elen or tew not in self.tews or (sew, tw) not in KMAX:
                    ok = False
                else:
                    eve = vs.vlen // sew
                    tk = min(tk, KMAX[(sew, tw)])
                    tm = min(tm, lmul * eve, vme.ete(self.te, tew))
            if not ok:
                vt = 1 << 63  # vill
            # vl = 0 and vtype as though by vsetvl (rs1 nonzero, AVL 0).
            vs.set_vl(0, 1, 0, vt, "msetmtype")
            if vs.vill or not mt:
                self.tm = self.tk = self.mtwiden = 0
            else:
                self.tm, self.tk, self.mtwiden = tm, tk, mt
            return
        value = rf[insn.field(instr, "rs1")].read_uint()
        rd = insn.field(instr, "rd")
        if name == "msettn":
            if not self.mtwiden:
                out = 0 if vs.vill else min(value, vs.vlmax)
            else:
                tew = vs.sew * self.twiden
                out = min(value, (vs.vlen // vs.sew) << vs.lmul_log2, self.ete(tew))
            vs.vl = out
            vs.vstart = 0
        else:
            if not self.mtwiden:
                raise self.illegal(
                    instr,
                    f"{name} with mtwiden = 0 (the matrix unit is not configured)",
                )
            if name == "msettm":
                tew = vs.sew * self.twiden
                out = self.tm = min(
                    value, (vs.vlen // vs.sew) << vs.lmul_log2, self.ete(tew)
                )
            else:
                out = self.tk = min(value, self.kmax())
        if rd:
            write_reg(rf, rd, out)

    def _configured(self, instr, what):
        if self.vs.vill:
            raise self.illegal(instr, f"{what} with vtype.vill = 1")
        if not self.mtwiden:
            raise self.illegal(instr, f"{what} is reserved when mtwiden = 0")

    def _body(self, tew):
        vs = self.vs
        return vs.vstart, min(vs.vl, self.ete(tew))

    def _tss(self, instr, value, tew):
        return vme.decode_tss(value, tew, self.te)

    def tile_memory(self, insn, instr, rf, memory):
        is_load = insn.name == "vtle"
        eee = insn.field(instr, "eee")
        if eee > 3:
            raise self.illegal(instr, f"{insn.name} with a reserved EEW code {eee}")
        tew = 8 << eee
        self._configured(instr, f"{insn.name}{tew}")
        if tew not in self.tews:
            raise self.illegal(
                instr,
                f"{insn.name}{tew} needs Zvt{tew}e, which this design does not list",
            )
        tile, col, index = self._tss(
            instr, rf[insn.field(instr, "rs2")].read_uint(), tew
        )
        base = rf[insn.field(instr, "rs1")].read_uint()
        start, end = self._body(tew)
        eb = tew // 8
        if base % eb:
            raise MatrixAccessFault(
                f"{insn.name}{tew} at {base:#x}: element addresses must be {eb}-byte aligned"
            )
        n = max(0, end - start)
        if n:
            idx = self.maps[tew][tile]
            sel = idx[:, index] if col else idx[index]
            sel = sel[start:end]
            addr = (base + start * eb) & self.xmask
            if is_load:
                self.buf[sel] = read_rows(
                    memory, addr, n * eb, 1, n * eb, self.xmask
                ).reshape(n, eb)
            else:
                memory.write(addr, self.buf[sel].tobytes())
        self.vs.vstart = 0
        self.stats["mem_to_tile_bytes" if is_load else "tile_to_mem_bytes"] += n * eb

    def move(self, insn, instr, rf):
        vs = self.vs
        self._configured(instr, insn.name)
        tew = vs.sew
        if tew not in self.tews:
            raise self.illegal(
                instr,
                f"{insn.name} at TEW {tew} needs Zvt{tew}e, which this design does not list",
            )
        lmul = 1 << max(0, vs.lmul_log2)
        reg = insn.field(instr, "vd" if insn.name == "vtmv.v.t" else "vs2")
        if reg % lmul:
            raise self.illegal(
                instr, f"{insn.name}: v{reg} is not aligned to LMUL = {lmul}"
            )
        tile, col, index = self._tss(
            instr, rf[insn.field(instr, "rs1")].read_uint(), tew
        )
        start, end = self._body(tew)
        n = max(0, end - start)
        if n:
            idx = self.maps[tew][tile]
            sel = (idx[:, index] if col else idx[index])[start:end]
            group = vs.vreg_bytes(reg, lmul).reshape(-1, tew // 8)
            if insn.name == "vtmv.v.t":
                group[start:end] = self.buf[sel]
            else:
                self.buf[sel] = group[start:end]
        vs.vstart = 0
        self.stats[
            "tile_to_vec_bytes" if insn.name == "vtmv.v.t" else "vec_to_tile_bytes"
        ] += n * tew // 8

    def zero(self, insn, instr):
        """``vtzero``: the ``tm x tn`` sub-tile at TEW = SEW (see the module docs of :mod:`framework.matrix.vme`)."""
        vs = self.vs
        self._configured(instr, "vtzero")
        tew = vs.sew
        tile = self._tile_number(instr, insn.field(instr, "mtd"), tew)
        e = self.ete(tew)
        rows, cols = min(self.tm, e), min(vs.vl, e)
        if rows and cols:
            self.buf[self.maps[tew][tile][:rows, :cols]] = 0
        self.stats["tile_zero"] += 1

    def multiply(self, insn, instr, rf):
        vs = self.vs
        if vs.vstart:
            raise self.illegal(instr, f"{insn.name} with vstart = {vs.vstart}")
        self._configured(instr, insn.name)
        sew, tw = vs.sew, self.twiden
        row = vme.arith_row(insn.name, sew, tw, (vs.vtype >> vme.ALTFMT_SHIFT) & 1)
        if row is None:
            raise self.illegal(
                instr, f"{insn.name} at SEW {sew}, TWIDEN {tw}: a reserved type row"
            )
        if row.ext not in self.extensions:
            raise self.illegal(
                instr,
                f"{insn.name} ({row.fmt_a} x {row.fmt_b} -> {row.fmt_c}) needs {row.ext}",
            )
        tew = sew * tw
        tile = self._tile_number(instr, insn.field(instr, "mtd") << 1, tew)
        kmax = KMAX[(sew, tw)]
        stride = 8 // kmax
        lmul = 1 << max(0, vs.lmul_log2)
        for what, reg in (
            ("vs2", insn.field(instr, "vs2")),
            ("vs1", insn.field(instr, "vs1")),
        ):
            if reg % lmul or reg % 8 >= stride:
                raise self.illegal(
                    instr,
                    f"{insn.name}: {what} = v{reg} (needs vs % LMUL == 0 and vs % 8 < 8/KMAX = {stride})",
                )
        e = self.ete(tew)
        tm, tn, tk = min(self.tm, e), min(vs.vl, e), self.tk
        if not (tm and tn and tk):
            return
        da, db = _dtype(row.fmt_a), _dtype(row.fmt_b)
        vs2, vs1 = insn.field(instr, "vs2"), insn.field(instr, "vs1")
        A = np.stack(
            [vs.vreg_bytes(vs2 + r * stride, lmul).view(da)[:tm] for r in range(tk)]
        )
        B = np.stack(
            [vs.vreg_bytes(vs1 + r * stride, lmul).view(db)[:tn] for r in range(tk)]
        )
        C = self.tile(tile, tew)[:tm, :tn].view(_dtype(row.fmt_c))
        if row.fmt_c == "int32":
            new = int_mac(C, A.T, B, 32)
        else:
            if sew >= 32:
                frm = (rf[FCSR_INDEX].read_uint() >> 5) & 0x7
                if frm != 0:
                    raise VMEError(
                        f"{insn.name} under frm = {frm}: only round-to-nearest-even is modelled"
                    )
            new = zvt_tile_mac(
                C,
                A.T,
                B,
                get_format(row.fmt_a),
                get_format(row.fmt_c),
                fmt_b=row.fmt_b,
                kmax=kmax,
            )
        self.set_tile(tile, tew, np.ascontiguousarray(new).view(_UINT[tew]), tm, tn)
        st = self.stats
        st["mma"] += 1
        st["macs"] += tm * tn * tk
        st["vrf_a_read_bytes"] += tk * tm * sew // 8
        st["vrf_b_read_bytes"] += tk * tn * sew // 8


def _disasm(insn, instr):
    ops = ", ".join(f"{k}={insn.field(instr, k)}" for k in insn.operands)
    return f"{insn.name} {ops}"


class RV64_VME_ISA(RV_ISA):
    """Zvt on the hart's :class:`VMEUnit` (``register_file.matrix``); must follow RVV in the chain."""

    #: Major opcodes this ISA may claim (OP-V and LOAD/STORE-FP words RVV declines, and OP-VE).
    OPCODES = vme.OPCODES
    #: The counter hub's instruction class.
    INSTR_CLASS = "matrix"
    _decoded = {}

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        unit = getattr(register_file, "matrix", None)
        if unit is None or getattr(unit, "mode", None) != "vme":
            return False
        insn = cls._decoded.get(instr, 0)
        if insn == 0:
            insn = cls._decoded[instr] = vme.decode(instr)
        if insn is None:
            return False
        unit.execute(insn, instr, register_file, memory_space)
        if snoop:
            cls.print_snoop(
                snoop, _disasm(insn, instr), f"vl={unit.vs.vl} mtype={unit.mtype:#x}"
            )
        return True


def attach(core, unit):
    """Give ``core`` the VME ``unit``: on the register file as ``matrix``, VME after RVV, mtype readable."""
    from framework.pe.rv.isa.v import RV64_V_ISA

    if RV64_V_ISA not in core.isas or getattr(core, "vector", None) is not unit.vs:
        raise VMEError("VME needs the core's vector unit (attach RVV first)")
    core.matrix = unit
    core.register_file.matrix = unit
    csrs = getattr(core, "csrs", None)
    if csrs is not None and hasattr(csrs, "ext_csrs"):
        csrs.ext_csrs[vme.CSR_MTYPE] = lambda: unit.mtype
    core.add_isa(RV64_VME_ISA)
    return unit


__all__ = ["COUNTER_NAMES", "RV64_VME_ISA", "VMEUnit", "attach"]
