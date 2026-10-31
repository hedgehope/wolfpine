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
                                   target. The link register of a ``jal`` /
                                   ``jalr`` has already been written (the
                                   spec says it is not); nothing else is.
1     instruction access fault     the fetch address is unmapped; ``mtval`` = pc
2     illegal instruction          an unclaimed encoding, or a CSR the core does
                                   not implement / a write to a read-only one;
                                   ``mtval`` is the instruction
3     breakpoint                   ``ebreak``; ``mtval`` = pc
4, 6  load / store-AMO address     a load, store, FP load/store, ``lr`` / ``sc`` /
      misaligned                   AMO whose address is not a multiple of its
                                   size (a Naiad bus otherwise allows it);
                                   ``mtval`` is the address. Vector accesses
                                   are not checked.
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
``mtimecmp``. Machine external (11) is not modelled; no block raises it.

Delivery sets ``mepc`` (pc of the interrupted or faulting instruction),
``mcause``, ``mtval`` (0 for an interrupt), ``mstatus.MPIE = MIE``,
``MIE = 0``, ``MPP = 3`` (M-mode only), and jumps to ``mtvec``'s base: direct
mode (``mtvec[0] = 0``) always, vectored mode (``mtvec[0] = 1``) to
``base + 4 * code`` for an interrupt and to ``base`` for an exception. ``mret``
jumps to ``mepc`` and sets ``MIE = MPIE``, ``MPIE = 1``, ``MPP = 3``.

``wfi`` wakes on a pending, enabled interrupt (``mip & mie``, whatever
``mstatus.MIE``); if ``mstatus.MIE`` is set the trap is then taken with
``mepc`` = the instruction after the ``wfi``. See ``framework/ip/clint.py``.

Cost model. Delivery of a trap, and ``mret``, are not timed beyond the one
cycle a delivery takes (the timing model's floor; nothing is charged for the
redirect). A trapping ``ecall`` / ``ebreak`` / faulting instruction does not
retire (``minstret`` unchanged), an ``mret`` does.

Implementation caveat: while armed, each instruction is peeked (decoded a
second time from memory, without publishing a memory event) so that ``ecall``,
misalignment and the access-fault address are known; the simulator runs
slower for it, not the simulated clock.
"""

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
MIP_MSIP = 1 << IRQ_MSI
MIP_MTIP = 1 << IRQ_MTI

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

    def _mret(self):
        store = self.csrs.store
        status = store.get(CSR_MSTATUS, 0)
        new = status & ~(MSTATUS_MIE | MSTATUS_MPIE | MSTATUS_MPP)
        if status & MSTATUS_MPIE:
            new |= MSTATUS_MIE
        store[CSR_MSTATUS] = new | MSTATUS_MPIE | MSTATUS_MPP
        epc = store.get(CSR_MEPC, 0)
        self.pc_register.write(epc.to_bytes(self.XLEN_BYTES, "little"))

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

    # -- the armed interpreter ----------------------------------------------

    def _trap_tick(self, cycle_num):
        if not self.active:
            return
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

        try:
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
            # A taken branch / jump to a target IALIGN forbids. (The link register,
            # if the instruction had one, is already written.)
            self._take(CAUSE_FETCH_MISALIGNED, pc, new_pc, own_cycle=False)


class NaiadRV64(TrapMixin, RV64I):
    """An ``RV64I`` with machine-mode trap delivery (see :mod:`framework.pe.rv.trap`)."""


class NaiadRV32(TrapMixin, RV32I):
    """An ``RV32I`` with machine-mode trap delivery."""


class NaiadRV32IM(TrapMixin, RV32IM):
    """An ``RV32IM`` with machine-mode trap delivery."""
