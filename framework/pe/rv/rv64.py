"""An RV64 integer core, for designs outside the Tenstorrent tiles (Naiad).

``RV64I`` reuses ``RV32I``'s machinery (fetch fast path, ISA chain, tracing,
cost hook, retire counting) with :attr:`~RV32I.XLEN_BYTES` = 8 and the RV64
base ISA in place of RV32's. The extensions are the separate RV64 classes in
``framework/pe/rv/isa/*64*``, so none of the RV32 handlers branch on XLEN.

A core is built from a list of extension classes, exactly like ``RV32I``; a
design that names its ISA as strings resolves them through
:func:`rv64_extensions`, which reads :data:`RV64_ISA_REGISTRY`.

Two things differ from ``RV32I``'s front end, both in :meth:`RV64I.clock_tick`
(the RV32 baby cores' loop is not touched):

* **Opcode-indexed dispatch.** ``RV32I`` offers every instruction to each ISA
  in turn, so every extension slows every instruction. Here a 128-entry table
  maps the major opcode (``instr & 0x7F``) to the ISAs that may claim it, in
  ``isas`` order, built from each class's ``OPCODES`` (a class without one is
  offered every opcode). The table is rebuilt whenever the ISA chain changes:
  assigning ``core._isa_runs`` (the counter hub's wrappers do) or
  :meth:`RV64I.add_isa` (the matrix unit's attach does).
* **The compressed ISA** (Zca / Zcd, ``rvc_isa.py``), when listed: 16-bit
  parcels at any 2-byte-aligned PC, expanded to their 32-bit instruction and
  dispatched as that.
"""

import struct

from framework.memory.memory import AddressableMemory, SparseAddressableMemory
from framework.pe.pe import ProcessingElement
from framework.pe.rv.isa.a64_isa import RV64_ZAAMO_ISA, RV64_ZALRSC_ISA
from framework.pe.rv.isa.b64_isa import RV64_ZBA_ISA, RV64_ZBB_ISA
from framework.pe.rv.isa.fd_isa import RV64_D_ISA, RV64_F_ISA
from framework.pe.rv.isa.i64_isa import RV64_I_ISA
from framework.pe.rv.isa.m64_isa import RV64_M_ISA
from framework.pe.rv.isa.rvc_isa import (
    C_JALR_MASK,
    C_JALR_MATCH,
    RV64_ZCA_ISA,
    RV64_ZCD_ISA,
    expand,
)
from framework.pe.rv.isa.v import RV64_V_ISA
from framework.pe.rv.isa.zfbfmin_isa import RV64_ZFBFMIN_ISA
from framework.pe.rv.isa.zicsr64_isa import RV64_ZICSR_ISA, CSRFile64, misa_value
from framework.pe.rv.rv32 import RV32I
from framework.trace import EventCategory, InstrEvent
from framework.util.conversion import conv_to_uint32

#: Extension name -> ISA class(es), lower-case. ``i`` is the base ISA and
#: resolves to nothing (it is always present); ``a`` is Zaamo + Zalrsc, and
#: ``zicntr`` and ``zihpm`` live in the Zicsr CSR file, so they resolve to the same class.
#: ``d`` and ``zfbfmin`` need ``f`` as well; the Naiad design validator says so.
RV64_ISA_REGISTRY = {
    "i": (),
    "m": (RV64_M_ISA,),
    "a": (RV64_ZAAMO_ISA, RV64_ZALRSC_ISA),
    "zaamo": (RV64_ZAAMO_ISA,),
    "zalrsc": (RV64_ZALRSC_ISA,),
    "zba": (RV64_ZBA_ISA,),
    "zbb": (RV64_ZBB_ISA,),
    "f": (RV64_F_ISA,),
    "d": (RV64_D_ISA,),
    "zfbfmin": (RV64_ZFBFMIN_ISA,),
    "zicsr": (RV64_ZICSR_ISA,),
    "zicntr": (RV64_ZICSR_ISA,),
    "zihpm": (RV64_ZICSR_ISA,),
    # C = Zca, plus Zcd when D is present (Zcf is RV32-only); see rvc_isa.
    "c": (RV64_ZCA_ISA, RV64_ZCD_ISA),
    "zca": (RV64_ZCA_ISA,),
    "zcd": (RV64_ZCD_ISA,),
    # The vector unit: V (VLEN >= 128) or Zve64d, with its FP sub-extensions as
    # flags on the same ISA class. VLEN/ELEN come from the design; see
    # framework.pe.rv.isa.v.attach.
    "v": (RV64_V_ISA,),
    "zve64d": (RV64_V_ISA,),
    "zvfh": (RV64_V_ISA,),
    "zvfbfmin": (RV64_V_ISA,),
    "zvfbfwma": (RV64_V_ISA,),
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
_MISA_LETTER = {RV64_I_ISA: "I", RV64_M_ISA: "M", RV64_F_ISA: "F", RV64_D_ISA: "D"}

#: Extensions that address the f-registers; any of them allocates the file.
_FP_ISAS = frozenset({RV64_F_ISA, RV64_D_ISA, RV64_ZFBFMIN_ISA})

_M64 = 0xFFFFFFFFFFFFFFFF
_PE_STALL = ProcessingElement.PEStall
_WORD_AT = struct.Struct("<I").unpack_from


class _ISARunList(list):
    """``RV64I._isa_runs``: a list that rebuilds its core's dispatch table when mutated.

    So code written against ``RV32I``'s plain list (``core._isa_runs.append(...)``)
    cannot leave the table stale.
    """

    def __init__(self, runs, core):
        super().__init__(runs)
        self._core = core

    def _changed(self):
        self._core.rebuild_dispatch()


def _mutator(name):
    method = getattr(list, name)

    def mutate(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._changed()
        return self if name == "__iadd__" else result

    mutate.__name__ = name
    return mutate


for _name in (
    "append",
    "extend",
    "insert",
    "remove",
    "pop",
    "clear",
    "sort",
    "reverse",
    "__setitem__",
    "__delitem__",
    "__iadd__",
):
    setattr(_ISARunList, _name, _mutator(_name))
del _name


def dispatch_table(isas, runs):
    """The opcode-indexed dispatch table for an ISA chain: 128 tuples of ``runs``.

    Entry ``op`` holds, in chain order, the run callable of every ISA whose
    ``OPCODES`` contains ``op``, and of every ISA that declares no ``OPCODES``
    (offered everything, as the plain chain would). ``runs[i]`` belongs to
    ``isas[i]``; it may be a wrapper (the counter hub's) rather than ``isa.run``.
    """
    table = []
    for op in range(128):
        claimants = []
        for isa, run in zip(isas, runs):
            opcodes = getattr(isa, "OPCODES", None)
            if opcodes is None or op in opcodes:
                claimants.append(run)
        table.append(tuple(claimants))
    return tuple(table)


class RV64I(RV32I):
    """RV64I core: 64-bit x-registers and PC, FLEN = 64 f-registers.

    ``extensions`` is a list of ISA classes (the base RV64I is implicit and
    first). Including :class:`RV64_ZICSR_ISA` allocates a :class:`CSRFile64`,
    reachable as ``core.csrs``, whose ``mhartid`` is ``core_id``.
    ``fp_registers=True`` allocates 32 eight-byte f-registers plus ``fcsr`` at
    the same indices RV32 uses; listing F, D or Zfbfmin implies it.
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
        fp_registers = fp_registers or any(isa in _FP_ISAS for isa in extensions)
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

        if RV64_ZICSR_ISA in self.isas:
            letters = {_MISA_LETTER[isa] for isa in self.isas if isa in _MISA_LETTER}
            if RV64_ZAAMO_ISA in self.isas and RV64_ZALRSC_ISA in self.isas:
                letters.add("A")
            # misa.C means every compressed form the other letters imply: Zca,
            # and Zcd when D is present.
            if RV64_ZCA_ISA in self.isas and (
                RV64_D_ISA not in self.isas or RV64_ZCD_ISA in self.isas
            ):
                letters.add("C")
            self.csrs = CSRFile64(
                self.register_file,
                hart_id=core_id,
                misa=misa_value(sorted(letters)),
                core_label=self.core_label,
            )
            self.register_file.csrs = self.csrs

    # -- the ISA chain and its dispatch table ------------------------------------

    @property
    def _isa_runs(self):
        """The ISA chain's run callables, ``isas[i]``'s at index ``i``.

        Assigning it (``RV32I.__init__`` and the counter hub's wrappers do) or
        mutating it in place rebuilds the dispatch table; ``isas`` must be in
        step by then, so append to ``isas`` first, or use :meth:`add_isa`.
        """
        return self.__dict__["_isa_run_list"]

    @_isa_runs.setter
    def _isa_runs(self, runs):
        self.__dict__["_isa_run_list"] = _ISARunList(runs, self)
        self.rebuild_dispatch()

    def add_isa(self, isa):
        """Append ``isa`` to the chain (if absent); the dispatch table follows."""
        if isa not in self.isas:
            self.isas.append(isa)
            self._isa_runs.append(isa.run)

    def rebuild_dispatch(self):
        """Recompute the dispatch table and the compressed-ISA switches from the chain."""
        isas = self.isas
        runs = self.__dict__["_isa_run_list"]
        if len(runs) != len(isas):
            raise ValueError(
                f"core {self.core_label}: {len(runs)} ISA run callables for {len(isas)} ISAs"
            )
        self._dispatch = dispatch_table(isas, runs)
        self._rvc = RV64_ZCA_ISA in isas
        zcd = self._rvc and RV64_ZCD_ISA in isas
        if getattr(self, "_rvc_zcd", None) != zcd:
            # parcel -> 32-bit expansion (0 for a reserved encoding), memoised:
            # a program has few distinct compressed parcels.
            self._rvc_cache = {}
        self._rvc_zcd = zcd

    # -- the front end -------------------------------------------------------------

    def _cache_fetch_src(self, addr):
        """``RV32I._cache_fetch_src``, narrowed to a buffer the fast path can read directly.

        ``RV32I`` keeps the leaf memory and pays its ``read`` (bounds checks, a
        chunk lookup, a numpy slice) on every fetch. Here the span is narrowed
        to the leaf's backing array (one chunk of a sparse memory), and
        :attr:`_fetch_src` holds ``(low, last, view, view_base)`` with a
        memoryview of it, so a fetch is one ``struct.unpack_from``. The arrays
        are allocated once and written in place, so the view never goes stale:
        a store into the program is seen by the next fetch. An address whose
        sparse chunk does not exist yet (it reads as zeros) is not cached.
        """
        super()._cache_fetch_src(addr)
        src = self._fetch_src
        if src is None:
            return
        low, last, leaf, base = src
        if isinstance(leaf, SparseAddressableMemory):
            index = (addr - base) // leaf.chunk_size
            chunk = leaf.chunks.get(index)
            if chunk is None:
                self._fetch_src = None
                return
            view_base = base + index * leaf.chunk_size
            low = max(low, view_base)
            last = min(last, view_base + leaf.chunk_size - 4)
            view = memoryview(chunk)
        elif isinstance(leaf, AddressableMemory):
            view_base = base
            view = memoryview(leaf.memory)
        else:
            self._fetch_src = None
            return
        self._fetch_src = (low, last, view, view_base) if low <= last else None

    def _fetch(self, pc_val):
        """The instruction at ``pc_val`` through the full memory dispatch (the slow path).

        With the compressed ISA the first 16-bit parcel is read on its own and
        the second only if the first says the instruction is 32 bits, so a
        compressed instruction in a region's last two bytes, or a 32-bit one
        straddling two regions, fetches correctly. Without it the 4-byte read
        is ``RV32I``'s.
        """
        memory = self.visible_memory
        if self._rvc:
            instr = int.from_bytes(memory.read(pc_val, 2), "little")
            if instr & 3 == 3:
                high = int.from_bytes(memory.read((pc_val + 2) & _M64, 2), "little")
                instr |= high << 16
        else:
            instr = int.from_bytes(memory.read(pc_val, 4), "little")
        self._cache_fetch_src(pc_val)
        return instr

    def clock_tick(self, cycle_num):
        """``RV32I.clock_tick`` with table dispatch, the compressed ISA and ``mcycle``.

        Kept line-for-line in step with ``RV32I.clock_tick`` (fetch fast path,
        cost-model back-pressure, tracing, snoop, retire counting) so the two
        cores behave the same except where noted. A compressed instruction is
        traced, and seen by Spike-style commit logs, as its 16-bit parcel; the
        cost model and the ISAs see its 32-bit expansion.
        """
        if not self.active:
            return
        # ``cycle`` / ``mcycle`` count the ticks this core has been clocked
        # while running.
        csrs = self.csrs
        if csrs is not None:
            csrs.cycles += 1

        register_file = self.register_file
        pc = self.pc_register
        nextpc = self.nextpc_register
        # pc and nextpc are plain RW 8-byte registers that tracing never hooks
        # (``RegisterFile.set_write_recording`` skips them), so their ``value``
        # is read and assigned directly: exactly what ``Register.write`` does
        # for a full-width value, without the call and its checks.
        pc_val = int.from_bytes(pc.value, "little")
        memory = self.visible_memory
        memory.caller_context = (self.unit_id, self.core_label, pc_val)
        bus = self.bus

        src = self._fetch_src
        if (
            src is not None
            and src[0] <= pc_val <= src[1]
            and not bus.enabled
            and not memory.snoop_addresses
        ):
            instr = _WORD_AT(src[2], pc_val - src[3])[0]
        else:
            instr = self._fetch(pc_val)

        if instr & 3 != 3 and self._rvc:
            parcel = instr & 0xFFFF
            instr = self._rvc_cache.get(parcel)
            if instr is None:
                instr = self._rvc_cache[parcel] = expand(parcel, self._rvc_zcd) or 0
            step = 2
        else:
            parcel = instr
            step = 4
        nextpc.value = ((pc_val + step) & _M64).to_bytes(8, "little")

        cost = self.rv_cost
        if cost is not None and not cost.can_issue(instr, cycle_num, register_file):
            return

        trace_instr = self.unit_id is not None and bus.is_enabled(EventCategory.INSTR)
        if trace_instr != register_file.write_recording:
            register_file.set_write_recording(trace_instr)
        if trace_instr:
            register_file.clear_write_record()

        snoop = self.snoop
        if snoop:
            print(f"[{self.core_id}-> {cycle_num}][{hex(pc_val)}] ", end="")
            if step == 2:
                print(f"(c {parcel:#06x}) ", end="")

        # The first ISA to return something truthy claimed the instruction;
        # PEStall (a class, so truthy) means it must be re-offered later.
        actioned = False
        exec_memory = self._exec_memory
        for isa_run in self._dispatch[instr & 0x7F]:
            actioned = isa_run(register_file, exec_memory, snoop, instr)
            if actioned:
                break
        pe_stall = actioned is _PE_STALL

        if not actioned and not pe_stall:
            self.unknown_instructions += 1
            if self.unknown_instr_is_error:
                raise Exception("Unknown instruction")
            if snoop:
                print(f"unknown # {parcel:032b}", end="")
        elif step == 2 and not pe_stall and instr & C_JALR_MASK == C_JALR_MATCH:
            # c.jalr: the expansion linked pc + 4; the compressed form links pc + 2.
            register_file[1].write(((pc_val + 2) & _M64).to_bytes(8, "little"))

        if snoop:
            if pe_stall:
                print("[Stalled]")
            print("")

        if trace_instr:
            wi = register_file.last_write_idx
            if 1 <= wi <= 31:
                reg_idx = wi
                reg_val = conv_to_uint32(register_file.last_write_value)
            else:
                reg_idx = -1
                reg_val = 0
            if cost is None:
                stall_cycles, stall_reason = 0, ""
            else:
                stall_cycles, stall_reason = cost.take_pending_stall()
            bus.publish(
                InstrEvent(
                    cycle=cycle_num,
                    unit_id=self.unit_id,
                    pc=pc_val,
                    instruction=parcel,
                    stalled=pe_stall,
                    reg_write_idx=reg_idx,
                    reg_write_value=reg_val,
                    stall_cycles=stall_cycles,
                    stall_reason=stall_reason,
                )
            )

        if not pe_stall:
            if csrs is not None and actioned:
                csrs.retired += 1  # minstret; see RV32I.clock_tick
            pc.value = nextpc.value

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
