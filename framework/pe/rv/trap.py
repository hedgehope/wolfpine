"""Machine-mode trap and interrupt delivery for Naiad cores.

Used only by Naiad (``framework/naiad/builder.py``). The Tenstorrent baby cores
(``RV32I`` / ``BabyRISCV``) and the shared interpreters in ``rv32.py`` /
``rv64.py`` are untouched: a Naiad core is a subclass of the plain core with
:class:`TrapMixin` in front, and the mixin costs nothing until a program arms it.

Arming
------

A core is **armed** when its ``mtvec`` is non-zero. While armed, ``clock_tick``
is replaced (on the instance) by :meth:`TrapMixin._trap_tick`; writing ``mtvec``
back to zero, or resetting the core, restores the plain interpreter. With
``mtvec == 0`` every behaviour is exactly what it was before traps existed:
``ecall`` and ``mret`` are nops, ``ebreak`` raises ``RiscvBreakpoint`` (the
device-side-assertion halt), no interrupt is ever taken, and an illegal
instruction, an unknown CSR or an access to an unmapped address ends the run
with an error (``NaiadRunError``). The HTIF exit
path is a store intercept, not a trap, so it is unaffected either way.

What is delivered
-----------------

Synchronous exceptions (``mepc`` = the faulting instruction's pc):

===== ============================ ==========================================
cause name                         raised when / ``mtval``
===== ============================ ==========================================
0     instruction address          a taken branch / jump target not aligned
      misaligned                   to IALIGN (4, or 2 with C); ``mtval`` is the
                                   target. The instruction does not retire and
                                   writes nothing: a ``jal`` / ``jalr`` link
                                   register keeps its old value.
1     instruction access fault     the fetch address is unmapped; ``mtval`` = pc
2     illegal instruction          an unclaimed encoding, or a CSR the core does
                                   not implement / a write to a read-only one;
                                   ``mtval`` is the instruction
3     breakpoint                   ``ebreak``; ``mtval`` = pc
4, 6  load / store-AMO address     a load, store, FP load/store, ``lr`` / ``sc`` /
      misaligned                   AMO whose address is not a multiple of its
                                   size (a Naiad bus otherwise allows it);
                                   ``mtval`` is the address. A vector load /
                                   store traps when an active element's address
                                   is not a multiple of its EEW (SEW for an
                                   indexed access): see **Vector accesses**.
5, 7  load / store access fault    the access hit no mapped block, or a register
                                   that refuses it (read-only); ``mtval`` is
                                   the address
11    environment call from M      ``ecall``
===== ============================ ==========================================

Interrupts (``mcause`` bit XLEN-1 set): machine software (3, the CLINT's
``msip[hart]``) and machine timer (7, ``mtime >= mtimecmp[hart]``), taken
before an instruction when ``mstatus.MIE`` is set and ``mip & mie`` is non-zero
(software before timer, the spec's priority order). ``mip`` is read-only here: it reflects the CLINT, and the
interrupt is level-triggered, so a handler clears it by writing ``msip`` or
``mtimecmp``. Machine external (11, ``MEIP``) comes from a ``plic`` block
(:mod:`framework.ip.plic`), if the design has one, and has the highest priority
(external, software, timer); a design without one never raises it.

Delivery sets ``mepc`` (pc of the interrupted or faulting instruction),
``mcause``, ``mtval`` (0 for an interrupt), ``mstatus.MPIE = MIE``,
``MIE = 0``, ``MPP = 3`` (M-mode only), and jumps to ``mtvec``'s base: direct
mode (``mtvec[0] = 0``) always, vectored mode (``mtvec[0] = 1``) to
``base + 4 * code`` for an interrupt and to ``base`` for an exception. ``mret``
jumps to ``mepc`` and sets ``MIE = MPIE``, ``MPIE = 1``, ``MPP = 3``.

``wfi`` wakes on a pending, enabled interrupt (``mip & mie``, whatever
``mstatus.MIE``); if ``mstatus.MIE`` is set the trap is then taken with
``mepc`` = the instruction after the ``wfi``. See ``framework/ip/clint.py``.

Vector accesses. Before an RVV load / store (unit-stride, strided, indexed,
segment, whole-register) runs, the active elements' addresses are checked
against the element size (the fault-only-first load, the segment forms and the
whole-register forms included; ``vlm.v`` / ``vsm.v`` are byte accesses and
never misalign). The first misaligned element ``i`` raises cause 4 / 6 with
``mtval`` = its address and ``vstart`` = ``i``. Elements before ``i`` *are*
performed (the access is run once with ``vl`` cut to ``i``), so ``mret`` after
the handler fixed the cause re-executes from ``vstart``. Simplifications:
Zvvm / Zvt tile loads and stores are not checked, and the cost model charges
the truncated access, not the whole one. (A unit-stride access is misaligned
at its first element or not at all, so a fault-only-first load never has a
later element to trim ``vl`` to.)

Cost model. Delivery of a trap takes the one cycle of the tick itself; with a
timing model the cost state then holds the handler's first instruction for
``timing.scalar.trap_entry`` cycles, and the instruction after an ``mret`` for
``trap_return`` (see :mod:`framework.pe.rv.timing.core`; both are ``branch``
stalls, charged only when a trap or ``mret`` happens, so a core that never arms
``mtvec`` is unaffected). A trapping ``ecall`` / ``ebreak`` / faulting instruction does not
retire (``minstret`` unchanged), an ``mret`` does.

Implementation caveat: while armed, each instruction is peeked (decoded a
second time from memory, without publishing a memory event) so that ``ecall``,
misalignment and the access-fault address are known; the simulator runs
slower for it, not the simulated clock.
"""

import numpy as np

from framework.pe.rv.isa.v import mem as vmem
from framework.pe.rv.isa.v.common import nregs
from framework.pe.rv.isa.v.state import UINT
from framework.pe.rv.isa.zicsr_isa import CSRError, CSRFile
from framework.pe.rv.rv32 import RV32I, RV32IM
from framework.pe.rv.rv64 import RV64I

CSR_MSTATUS = 0x300
CSR_MIE = 0x304
CSR_MTVEC = 0x305
CSR_MSCRATCH = 0x340
CSR_MEPC = 0x341
CSR_MCAUSE = 0x342
CSR_MTVAL = 0x343
CSR_MIP = 0x344

MSTATUS_MIE = 1 << 3
MSTATUS_MPIE = 1 << 7
MSTATUS_MPP = 3 << 11

IRQ_MSI = 3
IRQ_MTI = 7
IRQ_MEI = 11
MIP_MSIP = 1 << IRQ_MSI
MIP_MTIP = 1 << IRQ_MTI
MIP_MEIP = 1 << IRQ_MEI

CAUSE_FETCH_MISALIGNED = 0
CAUSE_FETCH_ACCESS = 1
CAUSE_ILLEGAL = 2
CAUSE_BREAKPOINT = 3
CAUSE_LOAD_MISALIGNED = 4
CAUSE_LOAD_ACCESS = 5
CAUSE_STORE_MISALIGNED = 6
CAUSE_STORE_ACCESS = 7
CAUSE_ECALL_M = 11

#: The bits of ``mstatus`` an RV32 Naiad core stores (the rest reads as before:
#: FS/VS permanently dirty, see ``zicsr_isa.CSR_READ_ONLY``).
_RV32_MSTATUS_BITS = MSTATUS_MIE | MSTATUS_MPIE | MSTATUS_MPP
_RV32_MSTATUS_BASE = 0x80006600
#: Plain read-what-you-wrote trap CSRs on the RV32 Naiad file.
_RV32_STORAGE = frozenset({CSR_MIE, CSR_MSCRATCH, CSR_MTVAL, CSR_MCAUSE})

_ECALL = 0x00000073
_EBREAK = 0x00100073
_MRET = 0x30200073
_UNKNOWN = ("Unknown instruction",)


class NaiadCSRFile32(CSRFile):
    """The RV32 baby-core CSR file plus the machine trap CSRs a Naiad RV32 hart needs.

    ``mstatus`` keeps its Blackhole reading (``0x80006600``: FS and VS dirty)
    with MIE / MPIE / MPP writable on top; ``mie``, ``mtvec``, ``mscratch``,
    ``mepc``, ``mcause``, ``mtval`` are storage; ``mip`` reflects
    :attr:`mip_source`. ``mie`` / ``mip`` are not in the Blackhole table (the
    baby cores have no interrupts), which is why the plain file rejects them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: ``None`` or a zero-argument callable giving the pending-interrupt bits (``mip``).
        self.mip_source = None
        #: ``None`` or a callable told every new ``mtvec`` value (arms the core).
        self.on_mtvec = None
        self.mepc_mask = ~0x3

    def read(self, addr, pc=None):
        if addr == CSR_MSTATUS:
            return _RV32_MSTATUS_BASE | (
                self.store.get(CSR_MSTATUS, 0) & _RV32_MSTATUS_BITS
            )
        if addr in _RV32_STORAGE or addr in (CSR_MTVEC, CSR_MEPC):
            return self.store.get(addr, 0)
        if addr == CSR_MIP:
            return 0 if self.mip_source is None else self.mip_source() & 0xFFFFFFFF
        return super().read(addr, pc)

    def write(self, addr, value, pc=None):
        value &= 0xFFFFFFFF
        if addr == CSR_MSTATUS:
            self.store[addr] = value & _RV32_MSTATUS_BITS
        elif addr in _RV32_STORAGE:
            self.store[addr] = value
        elif addr == CSR_MTVEC:
            self.store[addr] = value & ~0x2
            if self.on_mtvec is not None:
                self.on_mtvec(self.store[addr])
        elif addr == CSR_MEPC:
            self.store[addr] = value & self.mepc_mask & 0xFFFFFFFF
        elif addr == CSR_MIP:
            pass  # MSIP / MTIP are read-only views of the CLINT; no other bit is implemented
        else:
            super().write(addr, value, pc)


class TrapMixin:
    """Trap delivery for an ``RV32I`` / ``RV64I`` subclass; see the module docstring.

    Put it first in the bases (``class NaiadRV64(TrapMixin, RV64I)``). The
    core's ``csrs`` must be a :class:`NaiadCSRFile32` or a
    :class:`~framework.pe.rv.isa.zicsr64_isa.CSRFile64`; call
    :meth:`enable_traps` once it is in place.
    """

    _trap_cycle = 0

    def enable_traps(self):
        """Hook this core's CSR file so writing ``mtvec`` arms and disarms it."""
        csrs = self.csrs
        csrs.on_mtvec = self._mtvec_written
        if getattr(self, "_rvc", False):
            csrs.mepc_mask = ~0x1  # IALIGN = 16 with the C extension
        self._mtvec_written(csrs.store.get(CSR_MTVEC, 0))

    def _mtvec_written(self, value):
        if value:
            self.__dict__["clock_tick"] = self._trap_tick
        else:
            self.__dict__.pop("clock_tick", None)

    def initialise_core(self):
        self.__dict__.pop("clock_tick", None)  # csrs.reset() clears mtvec: disarmed
        super().initialise_core()

    # -- helpers -----------------------------------------------------------

    def _quiet_read(self, addr, size):
        """``size`` bytes at ``addr`` without a memory event; ``IndexError`` if unmapped."""
        memory = self.visible_memory
        rng, space = memory._locate_memory_space(addr)
        if space is None:
            raise IndexError(f"{hex(addr)} is unmapped")
        return space.read(addr - rng.low, size)

    def _peek(self, pc):
        """``(instruction, raw parcel)`` at ``pc``: the 32-bit form the ISAs see, and what ``mtval`` reports."""
        if getattr(self, "_rvc", False):
            from framework.pe.rv.isa.rvc_isa import expand

            low = int.from_bytes(self._quiet_read(pc, 2), "little")
            if low & 3 != 3:
                return expand(low, self._rvc_zcd) or 0, low
            high = int.from_bytes(self._quiet_read(pc + 2, 2), "little")
            raw = low | high << 16
            return raw, raw
        raw = int.from_bytes(self._quiet_read(pc, 4), "little")
        return raw, raw

    def _effective_address(self, instr):
        rs1 = (instr >> 15) & 0x1F
        base = self.register_file[rs1].read_uint()
        op = instr & 0x7F
        if op in (0x03, 0x07):  # load, FP load
            imm = instr >> 20
        elif op in (0x23, 0x27):  # store, FP store
            imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1F)
            imm &= 0xFFF
        else:  # AMO / lr / sc: the address is rs1
            return base
        if imm & 0x800:
            imm -= 0x1000
        return (base + imm) & ((1 << (8 * self.XLEN_BYTES)) - 1)

    @staticmethod
    def _access_size(instr):
        """``(size, is_store, is_lr)`` of a load/store/AMO the misalignment check covers, else ``None``."""
        op = instr & 0x7F
        funct3 = (instr >> 12) & 7
        if op == 0x03:
            return 1 << (funct3 & 3), False, False
        if op == 0x23:
            return 1 << (funct3 & 3), True, False
        if op in (
            0x07,
            0x27,
        ):  # scalar FP only: width 1..4 (vector widths are 0, 5, 6, 7)
            if 1 <= funct3 <= 4:
                return 1 << funct3, op == 0x27, False  # flh, flw, fld, flq
            return None
        if op == 0x2F and funct3 in (2, 3):
            return 1 << funct3, True, (instr >> 27) == 2
        return None

    def _take(self, cause, epc, tval, interrupt=False, own_cycle=True):
        """Deliver a trap: CSR updates and the jump to the handler.

        ``own_cycle``: the delivery is the tick's whole work (so ``mcycle`` is
        bumped here); False when the tick already counted itself.
        """
        csrs = self.csrs
        store = csrs.store
        bits = 8 * self.XLEN_BYTES
        store[CSR_MEPC] = epc & csrs.mepc_mask & ((1 << bits) - 1)
        store[CSR_MCAUSE] = cause | (1 << (bits - 1) if interrupt else 0)
        store[CSR_MTVAL] = tval & ((1 << bits) - 1)
        status = store.get(CSR_MSTATUS, 0)
        status &= ~(MSTATUS_MIE | MSTATUS_MPIE | MSTATUS_MPP)
        if store.get(CSR_MSTATUS, 0) & MSTATUS_MIE:
            status |= MSTATUS_MPIE
        store[CSR_MSTATUS] = status | MSTATUS_MPP
        mtvec = store.get(CSR_MTVEC, 0)
        target = mtvec & ~0x3
        if interrupt and mtvec & 1:
            target += 4 * cause
        self.pc_register.write(target.to_bytes(self.XLEN_BYTES, "little"))
        cycles = getattr(csrs, "cycles", None)
        if own_cycle and cycles is not None:
            csrs.cycles = cycles + 1
        self._charge("trap_entry")

    def _mret(self):
        store = self.csrs.store
        status = store.get(CSR_MSTATUS, 0)
        new = status & ~(MSTATUS_MIE | MSTATUS_MPIE | MSTATUS_MPP)
        if status & MSTATUS_MPIE:
            new |= MSTATUS_MIE
        store[CSR_MSTATUS] = new | MSTATUS_MPIE | MSTATUS_MPP
        epc = store.get(CSR_MEPC, 0)
        self.pc_register.write(epc.to_bytes(self.XLEN_BYTES, "little"))
        self._charge("trap_return")

    def _charge(self, which):
        """Hold the redirect target for the cost model's ``which`` penalty (no model: free)."""
        cost = self.rv_cost
        charge = getattr(cost, "charge_redirect", None)
        if charge is not None:
            charge(self._trap_cycle, getattr(cost, which, 0))

    def _classify(self, error, instr, raw):
        """``(cause, tval)`` for an exception an instruction raised, or ``None`` to let it propagate."""
        if (type(error) is Exception and error.args == _UNKNOWN) or isinstance(
            error, CSRError
        ):
            return CAUSE_ILLEGAL, raw
        if isinstance(error, IndexError):
            op = instr & 0x7F
            if op in (0x03, 0x07):
                return CAUSE_LOAD_ACCESS, self._effective_address(instr)
            if op in (0x23, 0x27, 0x2F):
                return CAUSE_STORE_ACCESS, self._effective_address(instr)
        return None

    # -- vector misalignment ---------------------------------------------------

    def _vector_misalignment(self, instr):
        """``(element, is_store, address)`` of the first misaligned element of an RVV access, else ``None``."""
        vs = getattr(self, "vector", None)
        if vs is None or vs.vill or instr & 0x7F not in (0x07, 0x27):
            return None
        if (instr >> 12) & 7 not in (0, 5, 6, 7):
            return None
        decoded = vmem.decode(instr)
        if decoded is None:
            return None
        handler, args, _ = decoded
        rf = self.register_file
        start, vl = vs.vstart, vs.vl
        if handler is vmem.whole:
            is_load, nreg, eew, _vd, rs1 = args
            esz = eew // 8
            base = rf[rs1].read_uint()
            if base % esz == 0 or start * esz >= nreg * vs.vlenb:
                return None
            return start, not is_load, (base + start * esz) & ((1 << 64) - 1)
        if handler is not vmem.access or start >= vl:
            return None
        is_load, mode, eew, nf, _vd, rs1, rs2, vm, *_ff = args
        if eew > vs.elen:
            return None
        esz = (vs.sew if mode == "indexed" else eew) // 8
        if esz == 1:
            return None
        base = rf[rs1].read_uint()
        elems = np.arange(start, vl, dtype=np.uint64)
        with np.errstate(over="ignore"):
            if mode == "unit":
                addrs = np.uint64(base) + elems * np.uint64(nf * esz)
            elif mode == "strided":
                addrs = np.uint64(base) + elems * np.uint64(rf[rs2].read_uint())
            else:
                idx_lg = vs.lmul_log2 + (eew.bit_length() - vs.sew.bit_length())
                if not -3 <= idx_lg <= 3:
                    return None
                idx = vs.group(rs2, nregs(idx_lg), UINT[eew])[start:vl]
                addrs = np.uint64(base) + idx.astype(np.uint64)
        bad = addrs % np.uint64(esz) != 0
        if not vm:
            bad &= vs.mask(0, vl)[start:]
        hits = np.flatnonzero(bad)
        if len(hits) == 0:
            return None
        k = int(hits[0])
        return start + k, not is_load, int(addrs[k])

    def _vector_prefix(self, cycle_num, pc, element):
        """Perform the elements of a vector access before its misaligned ``element``; set ``vstart``.

        False when the cost model stalled the access (it is offered again next cycle).
        """
        vs = self.vector
        if element > vs.vstart:
            vl = vs.vl
            vs.vl = element
            try:
                super().clock_tick(cycle_num)
            finally:
                vs.vl = vl
            if self.pc_register.read_uint() == pc:
                return False
            self.pc_register.write(pc.to_bytes(self.XLEN_BYTES, "little"))
            if self.csrs.retired:
                self.csrs.retired -= 1
        vs.vstart = element
        return True

    # -- the armed interpreter ----------------------------------------------

    def _trap_tick(self, cycle_num):
        if not self.active:
            return
        self._trap_cycle = cycle_num
        csrs = self.csrs
        store = csrs.store
        pc_register = self.pc_register
        pc = pc_register.read_uint()

        status = store.get(CSR_MSTATUS, 0)
        if status & MSTATUS_MIE:
            enabled = store.get(CSR_MIE, 0)
            if enabled and csrs.mip_source is not None:
                pending = csrs.mip_source() & enabled
                if pending:
                    if pending & MIP_MEIP:
                        code = IRQ_MEI
                    else:
                        code = IRQ_MSI if pending & MIP_MSIP else IRQ_MTI
                    self._take(code, pc, 0, interrupt=True)
                    return

        try:
            instr, raw = self._peek(pc)
        except IndexError:
            self._take(CAUSE_FETCH_ACCESS, pc, pc)
            return

        if instr == _ECALL:
            self._take(CAUSE_ECALL_M, pc, 0)
            return
        if instr == _EBREAK:
            self._take(CAUSE_BREAKPOINT, pc, pc)
            return
        access = self._access_size(instr)
        if access is not None:
            size, is_store, is_lr = access
            address = self._effective_address(instr)
            if address & (size - 1):
                self._take(
                    CAUSE_STORE_MISALIGNED
                    if is_store and not is_lr
                    else CAUSE_LOAD_MISALIGNED,
                    pc,
                    address,
                )
                return

        # A vector access with a misaligned element: run the elements before it, then trap.
        vfault = self._vector_misalignment(instr)
        link = None
        if instr & 0x7F in (0x6F, 0x67) and (instr >> 7) & 0x1F:
            rd = (
                instr >> 7
            ) & 0x1F  # jal / jalr: kept so a misaligned target can undo the link write
            link = (rd, self.register_file[rd].read_uint())

        try:
            if vfault is not None:
                if self._vector_prefix(cycle_num, pc, vfault[0]):
                    self._take(
                        CAUSE_STORE_MISALIGNED if vfault[1] else CAUSE_LOAD_MISALIGNED,
                        pc,
                        vfault[2],
                    )
                return
            super().clock_tick(cycle_num)
        except Exception as error:
            cause = self._classify(error, instr, raw)
            if cause is None:
                raise
            self._take(cause[0], pc, cause[1])
            return

        new_pc = pc_register.read_uint()
        if new_pc == pc:
            return  # stalled by the cost model, or a branch to itself: nothing to deliver
        if instr == _MRET:
            self._mret()
        elif new_pc & (1 if getattr(self, "_rvc", False) else 3):
            # A taken branch / jump to a target IALIGN forbids: the instruction
            # traps instead, so it neither retires nor writes its link register.
            if link is not None:
                self.register_file[link[0]].write(
                    link[1].to_bytes(self.XLEN_BYTES, "little")
                )
            if csrs.retired:
                csrs.retired -= 1
            self._take(CAUSE_FETCH_MISALIGNED, pc, new_pc, own_cycle=False)


class NaiadRV64(TrapMixin, RV64I):
    """An ``RV64I`` with machine-mode trap delivery (see :mod:`framework.pe.rv.trap`)."""


class NaiadRV32(TrapMixin, RV32I):
    """An ``RV32I`` with machine-mode trap delivery."""


class NaiadRV32IM(TrapMixin, RV32IM):
    """An ``RV32IM`` with machine-mode trap delivery."""
