"""An RV64 integer core, for designs outside the Tenstorrent tiles (Naiad).

``RV64I`` reuses ``RV32I``'s machinery (fetch fast path, ISA chain, tracing,
cost hook, retire counting) with :attr:`~RV32I.XLEN_BYTES` = 8 and the RV64
base ISA in place of RV32's. The extensions are the separate RV64 classes in
``framework/pe/rv/isa/*64*``, so none of the RV32 handlers branch on XLEN.

A core is built from a list of extension classes, exactly like ``RV32I``; a
design that names its ISA as strings resolves them through
:func:`rv64_extensions`, which reads :data:`RV64_ISA_REGISTRY`.
"""

from framework.pe.rv.isa.a64_isa import RV64_ZAAMO_ISA, RV64_ZALRSC_ISA
from framework.pe.rv.isa.b64_isa import RV64_ZBA_ISA, RV64_ZBB_ISA
from framework.pe.rv.isa.i64_isa import RV64_I_ISA
from framework.pe.rv.isa.m64_isa import RV64_M_ISA
from framework.pe.rv.isa.zicsr64_isa import RV64_ZICSR_ISA, CSRFile64, misa_value
from framework.pe.rv.rv32 import RV32I

#: Extension name -> ISA class(es), lower-case. ``i`` is the base ISA and
#: resolves to nothing (it is always present); ``a`` is Zaamo + Zalrsc, and
#: ``zicntr`` lives in the Zicsr CSR file, so it resolves to the same class.
RV64_ISA_REGISTRY = {
    "i": (),
    "m": (RV64_M_ISA,),
    "a": (RV64_ZAAMO_ISA, RV64_ZALRSC_ISA),
    "zaamo": (RV64_ZAAMO_ISA,),
    "zalrsc": (RV64_ZALRSC_ISA,),
    "zba": (RV64_ZBA_ISA,),
    "zbb": (RV64_ZBB_ISA,),
    "zicsr": (RV64_ZICSR_ISA,),
    "zicntr": (RV64_ZICSR_ISA,),
}


def rv64_extensions(names):
    """Resolve extension names (see :data:`RV64_ISA_REGISTRY`) to ISA classes.

    Order is kept and duplicates dropped. An unknown name raises ``KeyError``
    listing the known ones.
    """
    out = []
    for name in names:
        key = name.lower()
        if key not in RV64_ISA_REGISTRY:
            raise KeyError(
                f"unknown RV64 extension {name!r}; known: {sorted(RV64_ISA_REGISTRY)}"
            )
        for isa in RV64_ISA_REGISTRY[key]:
            if isa not in out:
                out.append(isa)
    return out


#: The ``misa`` letter each extension class contributes. ``A`` needs both
#: halves of the A extension.
_MISA_LETTER = {RV64_I_ISA: "I", RV64_M_ISA: "M"}


class RV64I(RV32I):
    """RV64I core: 64-bit x-registers and PC, FLEN = 64 f-registers.

    ``extensions`` is a list of ISA classes (the base RV64I is implicit and
    first). Including :class:`RV64_ZICSR_ISA` allocates a :class:`CSRFile64`,
    reachable as ``core.csrs``, whose ``mhartid`` is ``core_id``.
    ``fp_registers=True`` allocates 32 eight-byte f-registers plus ``fcsr`` at
    the same indices RV32 uses, for the D extension to address.
    """

    XLEN_BYTES = 8
    FLEN_BYTES = 8
    BASE_ISA = RV64_I_ISA

    def __init__(
        self,
        start_address,
        memory_spaces,
        extensions=None,
        unknown_instr_is_error=False,
        snoop=False,
        core_id=0,
        fp_registers=False,
    ):
        extensions = list(extensions) if extensions is not None else []
        super().__init__(
            start_address,
            memory_spaces,
            extensions,
            unknown_instr_is_error,
            snoop,
            core_id,
            fp_registers,
        )
        # Zalrsc's reservation, (start, end) or None; see a64_isa.
        self.register_file.reservation = None
        self._base_tick = super().clock_tick

        if RV64_ZICSR_ISA in self.isas:
            letters = {_MISA_LETTER[isa] for isa in self.isas if isa in _MISA_LETTER}
            if RV64_ZAAMO_ISA in self.isas and RV64_ZALRSC_ISA in self.isas:
                letters.add("A")
            self.csrs = CSRFile64(
                self.register_file,
                hart_id=core_id,
                misa=misa_value(sorted(letters)),
                core_label=self.core_label,
            )
            self.register_file.csrs = self.csrs

    def clock_tick(self, cycle_num):
        # ``cycle`` / ``mcycle`` count the ticks this core has been clocked
        # while running. Counted here, not in RV32I.clock_tick, so the baby
        # cores' path is untouched.
        csrs = self.csrs
        if csrs is not None and self.active:
            csrs.cycles += 1
        self._base_tick(cycle_num)

    def initialise_core(self):
        super().initialise_core()
        self.register_file.reservation = None


class RV64IM(RV64I):
    def __init__(
        self,
        start_address,
        memory_spaces,
        extensions=None,
        unknown_instr_is_error=False,
        snoop=False,
        core_id=0,
        fp_registers=False,
    ):
        extensions = list(extensions) if extensions is not None else []
        if RV64_M_ISA not in extensions:
            extensions.append(RV64_M_ISA)
        super().__init__(
            start_address,
            memory_spaces,
            extensions,
            unknown_instr_is_error,
            snoop,
            core_id,
            fp_registers,
        )
