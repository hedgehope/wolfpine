"""Zicsr + Zicntr for the RV64 core, with 64-bit CSRs.

The Blackhole baby cores' CSR file (``zicsr_isa.CSRFile``) is a transcription of
that hardware's CSR table, 32-bit and full of Tenstorrent-specific registers. An
RV64 core is not a baby core, so it gets its own, smaller file, built from the
privileged spec's machine-mode CSR list. Every CSR is XLEN = 64 bits wide, and
Zicntr on RV64 has no ``cycleh`` / ``timeh`` / ``instreth`` halves: those
addresses are unknown here and raise like any other.

What is modelled
----------------

* All six Zicsr instructions, with the spec's side-effect suppression:
  ``csrrw`` with ``rd == x0`` does not read, ``csrrs`` / ``csrrc`` with a zero
  ``rs1`` / ``uimm`` do not write (so ``csrr`` of a read-only CSR is legal).
* Zicntr ``cycle`` (0xC00), ``time`` (0xC01), ``instret`` (0xC02): read-only
  shadows of ``mcycle`` / ``mtime`` / ``minstret``. ``cycle`` counts the core's
  clock ticks since reset (bumped by ``RV64I.clock_tick``); ``instret`` counts
  retired instructions, with the same "retired" definition as the baby cores
  (``RV32I.clock_tick``). ``time`` reads :attr:`CSRFile64.time_source` when one
  is bound (a CLINT's ``mtime``), and otherwise ticks with ``cycle``.
* ``mcycle`` (0xB00) / ``minstret`` (0xB02), writable, modelled as an offset from
  the underlying count so a counter written to N carries on from N.
* ``mhartid`` / ``mvendorid`` / ``marchid`` / ``mimpid`` / ``mconfigptr``
  (read-only), and ``misa`` (WARL, writes ignored), reporting MXL = 2 and the
  letters of the extensions the core was built with.
* Zihpm: ``mhpmcounter3..31`` (0xB03-0xB1F), their read-only shadows
  ``hpmcounter3..31`` (0xC03-0xC1F) and ``mhpmevent3..31`` (0x323-0x33F). A
  counter is bound to an event by the platform (:meth:`CSRFile64.bind_hpm`,
  from a Naiad design's ``counters.hpm``), not by software: ``mhpmeventN`` is
  WARL and reads back ``N`` when counter ``N`` is bound, 0 when not. A bound
  counter reads the event's live value, with the same offset rule as
  ``mcycle``; an unbound one is hardwired to zero, as the spec allows.
* ``fflags`` / ``frm`` / ``fcsr`` as views of the FP register file's ``fcsr``
  entry, for a core built with ``fp_registers=True``.
* ``mstatus``, ``mtvec``, ``mie``, ``mip``, ``mscratch``, ``mepc``, ``mcause``,
  ``mtval``, ``mcounteren``, ``mcountinhibit`` as plain storage. Wolfpine models
  no traps or interrupts, so nothing reads them back but software;
  ``mcountinhibit`` in particular does not stop the counters. The one WARL rule
  kept is ``mepc[1:0]`` reading as zero (IALIGN = 32: Naiad cores have no C).

Refused
-------

* An address not listed here raises :class:`~zicsr_isa.UnknownCSRError` (the
  spec's answer is an illegal-instruction trap, which is not modelled).
* A write to a read-only CSR, i.e. one whose address has bits [11:10] = 0b11
  (the counters' shadows and the ID registers), also raises
  :class:`~zicsr_isa.UnknownCSRError`: the write form of a read-only CSR is an
  encoding no CSR answers for, and the spec traps on it the same way.
"""

from framework.pe.rv.isa.i64_isa import write_reg
from framework.pe.rv.isa.rv_isa import RV_ISA
from framework.pe.rv.isa.zicsr_isa import CSRError, NoCSRsError, UnknownCSRError

_M64 = 0xFFFFFFFFFFFFFFFF

CSR_FFLAGS = 0x001
CSR_FRM = 0x002
CSR_FCSR = 0x003
CSR_MCYCLE = 0xB00
CSR_MINSTRET = 0xB02
CSR_CYCLE = 0xC00
CSR_TIME = 0xC01
CSR_INSTRET = 0xC02
CSR_MISA = 0x301
CSR_MEPC = 0x341
CSR_MHARTID = 0xF14
CSR_MHPMCOUNTER3 = 0xB03
CSR_HPMCOUNTER3 = 0xC03
CSR_MHPMEVENT3 = 0x323
#: The Zihpm counter numbers.
HPM_COUNTERS = range(3, 32)

CSR_NAMES = {
    0x001: "fflags",
    0x002: "frm",
    0x003: "fcsr",
    0x300: "mstatus",
    0x301: "misa",
    0x304: "mie",
    0x305: "mtvec",
    0x306: "mcounteren",
    0x320: "mcountinhibit",
    0x340: "mscratch",
    0x341: "mepc",
    0x342: "mcause",
    0x343: "mtval",
    0x344: "mip",
    0xB00: "mcycle",
    0xB02: "minstret",
    0xC00: "cycle",
    0xC01: "time",
    0xC02: "instret",
    0xF11: "mvendorid",
    0xF12: "marchid",
    0xF13: "mimpid",
    0xF14: "mhartid",
    0xF15: "mconfigptr",
}
for _n in HPM_COUNTERS:
    CSR_NAMES[0xB00 + _n] = f"mhpmcounter{_n}"
    CSR_NAMES[0xC00 + _n] = f"hpmcounter{_n}"
    CSR_NAMES[0x320 + _n] = f"mhpmevent{_n}"
del _n

#: Plain read-what-you-wrote machine CSRs (reset to zero).
_STORAGE = frozenset(
    {0x300, 0x304, 0x305, 0x306, 0x320, 0x340, 0x341, 0x342, 0x343, 0x344}
)

#: ID registers that read as zero: no vendor, architecture, implementation or
#: configuration-structure ID is assigned to this core.
_ZERO_IDS = frozenset({0xF11, 0xF12, 0xF13, 0xF15})


def csr_name(addr):
    return CSR_NAMES.get(addr, hex(addr))


def misa_value(letters):
    """``misa`` for RV64 (MXL = 2) with the given extension letters set."""
    value = 2 << 62
    for letter in letters:
        value |= 1 << (ord(letter.upper()) - ord("A"))
    return value


def _at(pc):
    return "" if pc is None else f" at PC {hex(pc)}"


class CSRFile64:
    """One RV64 hart's CSRs. Held by the core and by its register file."""

    def __init__(self, register_file, hart_id=0, misa=misa_value("I"), core_label="?"):
        self.register_file = register_file
        self.hart_id = hart_id
        self.misa = misa
        self.core_label = core_label
        #: ``None``, or a zero-argument callable returning the platform's
        #: ``mtime`` (e.g. a CLINT's). Without one, ``time`` reads ``cycle``.
        self.time_source = None
        self.store = {}
        #: Clock ticks since reset, bumped by ``RV64I.clock_tick``.
        self.cycles = 0
        #: Retired instructions, bumped by ``RV32I.clock_tick``.
        self.retired = 0
        self._cycle_base = 0
        self._instret_base = 0
        #: Zihpm: counter number -> zero-argument reader of its event.
        self.hpm = {}
        self._hpm_base = {}

    def bind_hpm(self, n, reader):
        """Bind ``mhpmcounter<n>`` (3-31) to ``reader``, a zero-argument callable."""
        if n not in HPM_COUNTERS:
            raise ValueError(f"no Zihpm counter {n}; they are numbered 3-31")
        self.hpm[n] = reader

    def _hpm_read(self, n):
        reader = self.hpm.get(n)
        if reader is None:
            return 0
        return (reader() + self._hpm_base.get(n, 0)) & _M64

    def read(self, addr, pc=None):
        if addr == CSR_CYCLE or addr == CSR_MCYCLE:
            return (self.cycles + self._cycle_base) & _M64
        if addr == CSR_INSTRET or addr == CSR_MINSTRET:
            return (self.retired + self._instret_base) & _M64
        if addr == CSR_TIME:
            source = self.time_source
            return (source() if source is not None else self.cycles) & _M64
        if addr in _STORAGE:
            return self.store.get(addr, 0)
        if addr == CSR_MHARTID:
            return self.hart_id
        if addr == CSR_MISA:
            return self.misa
        if addr in _ZERO_IDS:
            return 0
        low = addr & 0x1F
        if low >= 3 and (addr & ~0x1F) in (0xB00, 0xC00):
            return self._hpm_read(low)
        if low >= 3 and (addr & ~0x1F) == 0x320:
            return low if low in self.hpm else 0
        if addr == CSR_FCSR:
            return self._fcsr(pc).read_uint() & 0xFF
        if addr == CSR_FFLAGS:
            return self._fcsr(pc).read_uint() & 0x1F
        if addr == CSR_FRM:
            return (self._fcsr(pc).read_uint() >> 5) & 0x7
        raise UnknownCSRError(
            f"CSR {hex(addr)} read on RV64 core {self.core_label}{_at(pc)} is not "
            f"one this core implements (see framework/pe/rv/isa/zicsr64_isa.py). "
            f"The spec's answer is an illegal-instruction trap, which Wolfpine "
            f"does not model."
        )

    def write(self, addr, value, pc=None):
        value &= _M64
        if addr >> 10 == 0x3:
            if addr not in CSR_NAMES:
                self.read(addr, pc)  # raises UnknownCSRError
            raise UnknownCSRError(
                f"write to read-only CSR {csr_name(addr)} ({hex(addr)}) on RV64 "
                f"core {self.core_label}{_at(pc)}. The spec's answer is an "
                f"illegal-instruction trap, which Wolfpine does not model."
            )
        if addr in _STORAGE:
            if addr == CSR_MEPC:
                value &= ~0x3  # WARL: no C extension, so IALIGN = 32
            self.store[addr] = value
        elif addr == CSR_MCYCLE:
            self._cycle_base = (value - self.cycles) & _M64
        elif addr == CSR_MINSTRET:
            self._instret_base = (value - self.retired) & _M64
        elif addr == CSR_MISA:
            pass  # WARL: the extension set is fixed at construction
        elif 0xB03 <= addr <= 0xB1F:
            n = addr & 0x1F
            reader = self.hpm.get(n)
            if reader is not None:
                self._hpm_base[n] = (value - reader()) & _M64
        elif 0x323 <= addr <= 0x33F:
            pass  # WARL: the event is fixed by the platform
        elif addr == CSR_FCSR:
            self._fcsr(pc).write((value & 0xFF).to_bytes(4, "little"))
        elif addr == CSR_FFLAGS:
            reg = self._fcsr(pc)
            reg.write(
                ((reg.read_uint() & ~0x1F) | (value & 0x1F)).to_bytes(4, "little")
            )
        elif addr == CSR_FRM:
            reg = self._fcsr(pc)
            new = (reg.read_uint() & ~0xE0) | ((value & 0x7) << 5)
            reg.write(new.to_bytes(4, "little"))
        else:
            self.read(addr, pc)  # raises UnknownCSRError

    def _fcsr(self, pc):
        try:
            return self.register_file["fcsr"]
        except (IndexError, KeyError):
            raise CSRError(
                f"FP CSR accessed on RV64 core {self.core_label}{_at(pc)}, which "
                f"was built without a floating-point register file "
                f"(fp_registers=False)."
            ) from None

    def reset(self):
        self.store = {}
        self.cycles = 0
        self.retired = 0
        self._cycle_base = 0
        self._instret_base = 0
        self._hpm_base = {}


_OP_NAMES = {1: "csrrw", 2: "csrrs", 3: "csrrc", 5: "csrrwi", 6: "csrrsi", 7: "csrrci"}


class RV64_ZICSR_ISA(RV_ISA):
    """The six Zicsr instructions against the core's :class:`CSRFile64`.

    Also provides Zicntr: the counters are entries in the same file. Adding
    this class to an ``RV64I``'s extensions is what allocates that file.
    """

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        if instr & 0x7F != 0x73:
            return False
        funct3 = (instr >> 12) & 0x7
        if funct3 not in _OP_NAMES:
            return False
        csrs = getattr(register_file, "csrs", None)
        if csrs is None:
            raise NoCSRsError(
                f"CSR instruction {hex(instr)} on a core with no CSR file; an "
                f"RV64I allocates one when RV64_ZICSR_ISA is in its extensions."
            )

        addr = instr >> 20
        rd = (instr >> 7) & 0x1F
        rs1 = (instr >> 15) & 0x1F  # uimm[4:0] in the immediate forms
        immediate = funct3 & 0x4
        operand = rs1 if immediate else register_file[rs1].read_uint()
        pc = register_file["pc"].read_uint()

        old = None
        if funct3 & 0x3 == 1:  # csrrw / csrrwi: no read when rd is x0
            if rd != 0:
                old = csrs.read(addr, pc)
            csrs.write(addr, operand, pc)
        else:  # csrrs / csrrc / csrrsi / csrrci: no write when rs1/uimm is 0
            old = csrs.read(addr, pc)
            if rs1 != 0:
                new = (old | operand) if funct3 & 0x3 == 2 else (old & ~operand)
                csrs.write(addr, new, pc)

        if rd != 0:
            write_reg(register_file, rd, old)
        if snoop:
            source = str(rs1) if immediate else cls.get_reg_name(rs1)
            RV_ISA.print_snoop(
                snoop,
                f"{_OP_NAMES[funct3]} {cls.get_reg_name(rd)}, {csr_name(addr)}, {source}",
                None if old is None else f"{csr_name(addr)} was {hex(old)}",
            )
        return True
