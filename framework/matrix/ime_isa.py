"""IME (Zvvm @ ``6effbcae``): the integrated matrix mode as an RV64 ISA executor.

The multiply-accumulates are OP-V words (OPIVV funct6 0x38-0x3b, OPFVV
0x14-0x17) and the tile loads/stores LOAD-FP / STORE-FP words with ``width =
111, mew = 1``: encodings RVV 1.0 leaves reserved, which its ISA declines
(returns False on), so :class:`RV64_IME_ISA` sits *after* RVV in the core's
chain and is offered exactly those words. Every word in that space belongs to
this mode: one this implementation does not have (an unlisted extension, a
lambda it does not support, a misaligned group, ...) raises
:class:`~framework.matrix.unit.IllegalMatrixInstruction`, as hardware would trap.

**No new state.** A, B and C are views onto the hart's vector register file
(``VectorState``); the only architectural addition is the four ``vtype``
fields, made legal with ``VectorState.install_vtype_extension`` and written by
:meth:`IMEUnit.on_vset` under the draft's rules (``vsetvl`` writes all of
them, ``vsetvli`` keeps ``bs`` / ``altfmt_A`` / ``altfmt_B``, ``lambda`` is
WARL; see :func:`framework.matrix.ime.warl_lambda`). Designs that list a bf16
accumulator extension also get Zvfbfa's ``altfmt`` bit (vtype bit 8), which
selects the C format here; the RVV unit itself does not implement Zvfbfa.

Semantics live in :mod:`framework.matrix.ime` (geometry, layout, encoding
map) and the numerics in :mod:`framework.matrix.ime_numerics` (FP, through
``tile_mac(..., numerics="ime")``) and :func:`~framework.matrix.ime_numerics.int_mac`.
Policies: C tile tail columns (``j >= N``) are left untouched (legal for either
``vta``), tile loads leave tail and masked-off elements undisturbed (the
draft's rule), and tile loads/stores honour ``vstart`` and leave it 0.
"""

import numpy as np

from framework.matrix import ime
from framework.matrix.formats import get_format
from framework.matrix.ime import (
    ALTFMT_A_SHIFT,
    ALTFMT_B_SHIFT,
    ALTFMT_SHIFT,
    IME_VTYPE_MASK,
    KEPT_MASK,
    LAMBDA_SHIFT,
    IMEError,
)
from framework.matrix.ime_numerics import int_mac
from framework.matrix.mac import tile_mac
from framework.matrix.unit import IllegalMatrixInstruction, read_rows, write_rows
from framework.pe.rv.isa.rv_isa import RV_ISA
from framework.pe.rv.rv32 import FCSR_INDEX

#: The unit's counters (``hart<N>.matrix.<key>``), in report order.
COUNTER_NAMES = (
    "mma",
    "macs",
    "vrf_a_read_bytes",
    "vrf_b_read_bytes",
    "vrf_c_read_bytes",
    "vrf_c_write_bytes",
    "mem_to_tile_bytes",
    "tile_to_mem_bytes",
)
_UINT = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
#: The bf16-accumulator extensions: they make Zvfbfa's altfmt (vtype bit 8) legal.
_BF16_C = ("Zvvbf16mm", "Zvvofp8bf16mm")


def _dtype(name):
    if name in ime.INT_TYPES:
        bits, signed = ime.INT_TYPES[name]
        return np.dtype(f"<{'i' if signed else 'u'}{bits // 8}")
    return get_format(name).dtype


class IMEUnit:
    """One hart's IME configuration and counters (the state is the hart's VRF)."""

    mode = "ime"

    def __init__(self, spec, vector_state, xlen=64):
        if xlen != 64:
            raise IMEError("IME is RV64 only")
        if vector_state.vlen != spec.vlen:
            raise IMEError(
                f"the IME spec is for VLEN {spec.vlen}, the vector unit has {vector_state.vlen}"
            )
        self.spec = spec
        self.vs = vector_state
        self.xlen = xlen
        self.xmask = (1 << xlen) - 1
        self.supported = {sew: tuple(lams) for sew, lams in spec.lambdas.items()}
        self.extensions = frozenset(spec.extensions)
        # Formats resolve here, so a missing ml_dtypes fails at build time.
        for pair in spec.format_pairs:
            for name in pair:
                _dtype(name)
        mask = IME_VTYPE_MASK
        if any(e in self.extensions for e in _BF16_C):
            mask |= 1 << ALTFMT_SHIFT
        vector_state.install_vtype_extension(mask, self.on_vset)
        self._maps = {}
        self.stats = dict.fromkeys(COUNTER_NAMES, 0)

    # -- lifecycle ---------------------------------------------------------------

    def reset(self):
        self.stats = dict.fromkeys(COUNTER_NAMES, 0)

    def counters(self):
        return dict(self.stats)

    def illegal(self, instr, why):
        return IllegalMatrixInstruction(f"illegal IME instruction {instr:#010x}: {why}")

    # -- vtype -------------------------------------------------------------------

    def on_vset(self, requested, old, form):
        """The vtype a ``vset*`` installs: the draft's IME field rules applied to ``requested``."""
        vsew = (requested >> 3) & 0x7
        if vsew > 3:
            return requested  # vill anyway
        sew = 8 << vsew
        current = (old >> LAMBDA_SHIFT) & 0x7
        if form == "vsetvl":
            kept = requested & KEPT_MASK
            want = (requested >> LAMBDA_SHIFT) & 0x7
        else:  # vsetvli / vsetivli keep bs / altfmt_A / altfmt_B and re-evaluate lambda
            kept = old & KEPT_MASK
            want = 0
        lam = ime.warl_lambda(want, current, self.supported.get(sew, ()))
        return (requested & ~IME_VTYPE_MASK) | kept | (lam << LAMBDA_SHIFT)

    # -- execution -----------------------------------------------------------------

    def execute(self, d, register_file, memory, instr):
        if d.insn.kind in ("int", "fp"):
            self.mma(d, register_file, instr)
        else:
            self.tile_access(d, register_file, memory, instr)

    def _config(self, instr, lam_imm=0):
        """``(sew, lmul, lambda)`` from vtype, or raise for any illegal configuration."""
        vs = self.vs
        if vs.vill:
            raise self.illegal(instr, "vtype is vill")
        if vs.lmul_log2 < 0:
            raise self.illegal(
                instr, "LMUL must be 1, 2, 4 or 8 (fractional LMUL is reserved)"
            )
        sew = vs.sew
        if lam_imm:
            lam = ime.decode_lambda(lam_imm)
            where = "immediate lambda"
        else:
            lam = ime.lambda_of(vs.vtype)
            where = "vtype.lambda"
            if lam is None:
                raise self.illegal(instr, "no lambda selected (vtype.lambda = 000)")
        if lam not in self.supported.get(sew, ()):
            raise self.illegal(
                instr,
                f"{where} {lam} is not supported at SEW {sew} (supported: {list(self.supported.get(sew, ()))})",
            )
        return sew, 1 << vs.lmul_log2, lam

    def mma(self, d, register_file, instr):
        vs, insn = self.vs, d.insn
        if vs.vstart:
            raise self.illegal(instr, f"{insn.name} with vstart = {vs.vstart}")
        if not d.vm:
            raise self.illegal(
                instr,
                f"{insn.name} with vm = 0 ({'reserved' if insn.w == 1 else 'a microscaled form, not modelled'})",
            )
        sew, lmul, lam = self._config(instr)
        geo = ime.gemm_geometry(vs.vlen, sew, lmul, lam, vs.vl, insn.w)
        if isinstance(geo, str):
            raise self.illegal(instr, f"{insn.name}: {geo}")
        for name, reg, n in (
            ("vs1", d.vs1, lmul),
            ("vs2", d.vs2, lmul),
            ("vd", d.vd, geo.emul_c),
        ):
            if reg % n or reg + n > 32:
                raise self.illegal(
                    instr,
                    f"{insn.name}: {name} = v{reg} is not aligned to a group of {n}",
                )
        vt = vs.vtype
        row = ime.encoding_row(
            insn.kind,
            insn.w,
            sew,
            (vt >> ALTFMT_A_SHIFT) & 1,
            (vt >> ALTFMT_B_SHIFT) & 1,
            (vt >> ALTFMT_SHIFT) & 1,
        )
        if row is None:
            raise self.illegal(
                instr,
                f"{insn.name} at SEW {sew}: a reserved or unmodelled encoding-map row",
            )
        missing = [e for e in row.requires if e not in self.extensions]
        if missing:
            raise self.illegal(
                instr,
                f"{insn.name} ({row.fmt_a} x {row.fmt_b} -> {row.fmt_c}) needs {missing}, which this design does not list",
            )
        m, n, w = geo.m, geo.n, insn.w
        A = ime.tile_view(
            vs.vreg_bytes(d.vs1, lmul).view(_dtype(row.fmt_a)), lmul, m, lam * w
        )
        B = ime.tile_view(
            vs.vreg_bytes(d.vs2, lmul).view(_dtype(row.fmt_b)), lmul, m, lam * w
        )[:n]
        span = vs.vreg_bytes(d.vd, geo.emul_c).view(_dtype(row.fmt_c))
        C = ime.tile_view(span, geo.emul_c, m, lam)
        if insn.kind == "fp":
            frm = (register_file[FCSR_INDEX].read_uint() >> 5) & 0x7
            if frm != 0:
                raise IMEError(
                    f"{insn.name} under frm = {frm}: only round-to-nearest-even is modelled"
                )
            red = self.spec.reduction(sew, w, lam)
            new = tile_mac(
                C[:, :n],
                A,
                B.T,
                row.fmt_a,
                row.fmt_c,
                numerics="ime",
                fmt_b=row.fmt_b,
                group=red.g,
                psm=red.psm,
                rnd=red.rnd,
                w=w,
                step=lam * w,
            )
        else:
            new = int_mac(C[:, :n], A, B.T, sew)
        C[:, :n] = new
        ime.write_tile(span, C, geo.emul_c, lam)
        st = self.stats
        st["mma"] += 1
        st["macs"] += geo.macs
        st["vrf_a_read_bytes"] += geo.a_bytes()
        st["vrf_b_read_bytes"] += geo.b_bytes()
        st["vrf_c_read_bytes"] += geo.c_bytes()
        st["vrf_c_write_bytes"] += geo.c_bytes()

    def _index_map(self, lmul, lam, epr, vl):
        key = (lmul, lam, epr, vl)
        m = self._maps.get(key)
        if m is None:
            m = self._maps[key] = ime.tile_index_map(lmul, lam, epr, vl)
        return m

    def tile_access(self, d, register_file, memory, instr):
        vs, insn = self.vs, d.insn
        need = "Zvvmttls" if d.transposing else "Zvvmtls"
        if need not in self.extensions:
            raise self.illegal(
                instr, f"{insn.name} needs {need}, which this design does not list"
            )
        sew, lmul, lam = self._config(instr, d.lam_imm)
        ls = lam * lmul
        vl = vs.vl
        if vl % ls:
            raise self.illegal(
                instr, f"{insn.name}: VL {vl} is not a multiple of lambda * LMUL = {ls}"
            )
        if d.vd % lmul or d.vd + lmul > 32:
            raise self.illegal(
                instr, f"{insn.name}: v{d.vd} is not aligned to LMUL = {lmul}"
            )
        is_load = insn.kind == "load"
        if is_load and not d.vm and d.vd == 0:
            raise self.illegal(instr, f"{insn.name}: a masked load cannot write v0")
        eb = sew // 8
        epr = vs.vlen // sew
        ld = register_file[d.rs2].read_uint()
        if ld == 0:
            ld = epr // lam if d.transposing else ls
        base = register_file[d.rs1].read_uint()
        group = vs.group(d.vd, lmul, _UINT[sew])
        flat = self._index_map(lmul, lam, epr, vl)
        nl = vl // ls
        start = vs.vstart
        if start == 0 and d.vm:
            n_active = vl
            if is_load:
                if d.transposing:
                    raw = read_rows(memory, base, ld * eb, ls, nl * eb, self.xmask)
                    data = raw.view(_UINT[sew]).reshape(ls, nl).T.reshape(-1)
                else:
                    raw = read_rows(memory, base, ld * eb, nl, ls * eb, self.xmask)
                    data = raw.view(_UINT[sew]).reshape(-1)
                group[flat] = data
            else:
                elems = group[flat]
                if d.transposing:
                    rows = np.ascontiguousarray(elems.reshape(nl, ls).T)
                else:
                    rows = elems.reshape(nl, ls)
                write_rows(memory, base, ld * eb, rows.view(np.uint8), self.xmask)
        else:
            active = np.arange(vl) >= start
            if not d.vm:
                active &= vs.mask(0, vl)
            idx = np.nonzero(active)[0]
            n_active = len(idx)
            for i in idx:
                line, k = divmod(int(i), ls)
                off = k * ld + line if d.transposing else line * ld + k
                addr = (base + off * eb) & self.xmask
                if is_load:
                    raw = read_rows(memory, addr, eb, 1, eb, self.xmask)
                    group[flat[i]] = raw.view(_UINT[sew])[0, 0]
                else:
                    memory.write(addr, group[flat[i : i + 1]].tobytes())
        vs.vstart = 0
        key = "mem_to_tile_bytes" if is_load else "tile_to_mem_bytes"
        self.stats[key] += n_active * eb


def _disasm(d):
    if d.insn.kind in ("int", "fp"):
        return f"{d.insn.name} v{d.vd}, v{d.vs1}, v{d.vs2}"
    lam = f", L{ime.decode_lambda(d.lam_imm)}" if d.lam_imm else ""
    mask = "" if d.vm else ", v0.t"
    base = RV_ISA.get_reg_name(d.rs1)
    return f"{d.insn.name} v{d.vd}, ({base}), {RV_ISA.get_reg_name(d.rs2)}{lam}{mask}"


class RV64_IME_ISA(RV_ISA):
    """Zvvm on the hart's :class:`IMEUnit` (``register_file.matrix``); must follow RVV in the chain."""

    #: Major opcodes this ISA may claim (a subset of RVV's; RVV declines these words).
    OPCODES = frozenset({ime.OP_V, ime.LOAD_FP, ime.STORE_FP})
    #: The counter hub's instruction class.
    INSTR_CLASS = "matrix"
    _decoded = {}

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        unit = getattr(register_file, "matrix", None)
        if unit is None or getattr(unit, "mode", None) != "ime":
            return False
        d = cls._decoded.get(instr, 0)
        if d == 0:
            d = cls._decoded[instr] = ime.decode(instr)
        if d is None:
            return False
        unit.execute(d, register_file, memory_space, instr)
        if snoop:
            cls.print_snoop(
                snoop, _disasm(d), f"vl={unit.vs.vl} vtype={unit.vs.vtype:#x}"
            )
        return True


def attach(core, unit):
    """Give ``core`` the IME ``unit``: on the register file as ``matrix``, IME after RVV in the chain."""
    from framework.pe.rv.isa.v import RV64_V_ISA

    if RV64_V_ISA not in core.isas or getattr(core, "vector", None) is not unit.vs:
        raise IMEError("IME needs the core's vector unit (attach RVV first)")
    core.matrix = unit
    core.register_file.matrix = unit
    core.add_isa(RV64_IME_ISA)
    return unit


__all__ = ["COUNTER_NAMES", "IMEUnit", "RV64_IME_ISA", "attach"]
