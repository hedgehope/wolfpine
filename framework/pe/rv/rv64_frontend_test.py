"""The RV64 core's front end: opcode-indexed dispatch, and the compressed ISA executed on a core.

Expansion itself is tested in ``isa/rvc_isa_test.py``; this file covers what the
core adds: 2-byte steps, ``c.jalr``'s link, 2-aligned branch targets, fetch
across region boundaries, ``misa.C``, tracing the 16-bit parcel, and the
dispatch table staying in step with the ISA chain (counter-hub wrapping, the
matrix unit's attach).
"""

import random

import pytest

from framework.device.device import DeviceMemory
from framework.memory.memory import DRAM
from framework.memory.memory_map import AddressRange, MemoryMap
from framework.pe.rv.isa.a64_isa import RV64_ZAAMO_ISA, RV64_ZALRSC_ISA
from framework.pe.rv.isa.b64_isa import RV64_ZBA_ISA, RV64_ZBB_ISA
from framework.pe.rv.isa.fd_isa import RV64_D_ISA, RV64_F_ISA
from framework.pe.rv.isa.i64_isa import RV64_I_ISA
from framework.pe.rv.isa.m64_isa import RV64_M_ISA
from framework.pe.rv.isa.rvc_isa import RV64_ZCA_ISA, RV64_ZCD_ISA
from framework.pe.rv.isa.zfbfmin_isa import RV64_ZFBFMIN_ISA
from framework.pe.rv.isa.zicsr64_isa import RV64_ZICSR_ISA
from framework.pe.rv.rv32 import FP_REGISTER_BASE
from framework.pe.rv.rv64 import RV64I, dispatch_table, rv64_extensions
from framework.trace import EventCategory, InstrEvent, get_bus

M64 = (1 << 64) - 1
PC = 0x1000
GC = ["m", "a", "f", "d", "c", "zicsr", "zicntr"]


def _core(names=GC, regions=((0, 0x4000),), pc=PC, error=True):
    mem_map = MemoryMap()
    for low, size in regions:
        mem_map[AddressRange(low, size)] = DRAM(size)
    core = RV64I(
        pc,
        [DeviceMemory(mem_map)],
        rv64_extensions(names),
        unknown_instr_is_error=error,
    )
    core.start()
    return core


def _place(core, addr, parcels):
    """Write a mix of 16-bit (< 0x10000 with low bits != 0b11) and 32-bit words from ``addr``."""
    mem = core.visible_memory
    for word in parcels:
        size = 2 if word & 3 != 3 else 4
        mem.write(addr, word.to_bytes(size, "little"))
        addr += size
    return addr


def _x(core, idx):
    return core.register_file[idx].read_uint()


def _set(core, idx, value):
    core.register_file[idx].write((value & M64).to_bytes(8, "little"))


def _pc(core):
    return core.pc_register.read_uint()


def _step(core, n=1):
    for cycle in range(n):
        core.clock_tick(cycle)


# -- the compressed ISA on a core ----------------------------------------------------


def test_mixed_16_and_32_bit_stream():
    core = _core()
    _place(
        core,
        PC,
        [
            0x4505,  # c.li a0, 1
            0x00150593,  # addi a1, a0, 1        (at PC + 2: a 32-bit word on a 2-aligned PC)
            0x95AA,  # c.add a1, a0
            0x0505,  # c.addi a0, 1
        ],
    )
    _step(core)
    assert (_x(core, 10), _pc(core)) == (1, PC + 2)
    _step(core)
    assert (_x(core, 11), _pc(core)) == (2, PC + 6)
    _step(core)
    assert (_x(core, 11), _pc(core)) == (3, PC + 8)
    _step(core)
    assert (_x(core, 10), _pc(core)) == (2, PC + 10)


def test_c_jalr_links_pc_plus_2_and_c_jr_returns():
    core = _core()
    _place(core, PC, [0x9302])  # c.jalr t1
    _place(core, 0x1802, [0x8082])  # c.jr ra (a 2-aligned target)
    _set(core, 6, 0x1802)
    _step(core)
    assert _pc(core) == 0x1802
    assert _x(core, 1) == PC + 2
    _step(core)
    assert _pc(core) == PC + 2


def test_c_jalr_through_ra_reads_the_old_ra():
    core = _core()
    _place(core, PC, [0x9082])  # c.jalr ra
    _set(core, 1, 0x2000)
    _step(core)
    assert (_pc(core), _x(core, 1)) == (0x2000, PC + 2)


def test_c_j_and_branches_to_2_aligned_targets():
    core = _core()
    _place(core, PC, [0xA009])  # c.j +2
    _step(core)
    assert _pc(core) == PC + 2
    # c.beqz a0, -2: taken when a0 == 0, else falls through by 2.
    core = _core()
    _place(core, PC, [0xDD7D])  # c.beqz a0, -2
    _step(core)
    assert _pc(core) == PC - 2
    core = _core()
    _place(core, PC, [0xDD7D])
    _set(core, 10, 5)
    _step(core)
    assert _pc(core) == PC + 2


def test_32_bit_branch_to_a_2_aligned_target():
    core = _core()
    _place(core, PC, [0x00000363])  # beq x0, x0, +6
    _step(core)
    assert _pc(core) == PC + 6


def test_stack_forms_and_c_ld_c_sd():
    core = _core()
    _set(core, 2, 0x3000)  # sp
    _set(core, 10, 0x1122334455667788)
    _place(
        core,
        PC,
        [
            0xE42A,  # c.sdsp a0, 8(sp)
            0x65A2,  # c.ldsp a1, 8(sp)
            0x0030,  # c.addi4spn a2, sp, 8
            0x6214,  # c.ld a3, 0(a2)
            0x7179,  # c.addi16sp sp, -48
            0x2681,  # c.addiw a3, 0 (sext.w)
        ],
    )
    _step(core, 6)
    assert _x(core, 11) == 0x1122334455667788
    assert _x(core, 12) == 0x3008
    assert _x(core, 2) == 0x3000 - 48
    assert _x(core, 13) == 0x55667788
    assert _pc(core) == PC + 12


def test_c_fld_c_fsd_need_d():
    core = _core()
    _set(core, 2, 0x3000)
    core.visible_memory.write(0x3010, (0x400921FB54442D18).to_bytes(8, "little"))
    _place(core, PC, [0x2542, 0xA42A])  # c.fldsp fa0, 16(sp); c.fsdsp fa0, 8(sp)
    _step(core, 2)
    assert core.register_file[FP_REGISTER_BASE + 10].read_uint() == 0x400921FB54442D18
    assert core.visible_memory.read(0x3008, 8) == (0x400921FB54442D18).to_bytes(
        8, "little"
    )
    # "zca" alone: c.fldsp is reserved, so an unknown instruction.
    core = _core(["m", "f", "d", "zca", "zicsr"], error=False)
    _place(core, PC, [0x2542])
    _step(core)
    assert core.unknown_instructions == 1


def test_illegal_parcel_is_an_unknown_instruction():
    core = _core(error=False)
    _place(core, PC, [0x0000, 0x8002])  # c.unimp, c.jr x0 (reserved)
    _step(core, 2)
    assert core.unknown_instructions == 2
    assert _pc(core) == PC + 4


def test_without_c_a_compressed_parcel_is_unknown():
    core = _core(["m", "zicsr"], error=False)
    _place(core, PC, [0x4505, 0x0001])
    _step(core)
    assert core.unknown_instructions == 1
    assert _x(core, 10) == 0
    assert _pc(core) == PC + 4


def test_compressed_in_the_last_two_bytes_of_a_region_and_a_word_straddling_two():
    # Two adjacent DRAM regions; the fetch fast path caches one at a time.
    core = _core(regions=((0x1000, 0x1000), (0x2000, 0x1000)), pc=0x1FFA)
    mem = core.visible_memory
    mem.write(0x1FFA, (0x00100513).to_bytes(4, "little"))  # addi a0, x0, 1
    mem.write(
        0x1FFE, (0x4589).to_bytes(2, "little")
    )  # c.li a1, 2 (last 2 bytes of region 1)
    mem.write(0x2000, (0x4611).to_bytes(2, "little"))  # c.li a2, 4
    mem.write(0x2002, (0x0001).to_bytes(2, "little"))  # c.nop
    mem.write(0x2004, (0x0001).to_bytes(2, "little"))  # c.nop
    mem.write(0x2006, (0x0001).to_bytes(2, "little"))  # c.nop
    mem.write(0x2008, (0x0001).to_bytes(2, "little"))  # c.nop
    _step(core, 3)
    assert [_x(core, r) for r in (10, 11, 12)] == [1, 2, 4]
    assert _pc(core) == 0x2002
    # A 32-bit word whose halves sit in different regions.
    core = _core(regions=((0x1000, 0x1000), (0x2000, 0x1000)), pc=0x1FFE)
    word = 0x00700693  # addi a3, x0, 7
    core.visible_memory.write(0x1FFE, (word & 0xFFFF).to_bytes(2, "little"))
    core.visible_memory.write(0x2000, (word >> 16).to_bytes(2, "little"))
    _step(core)
    assert (_x(core, 13), _pc(core)) == (7, 0x2002)


def test_misa_has_c():
    def misa(names):
        return _core(names).csrs.read(0x301)

    c_bit = 1 << 2
    assert misa(GC) & c_bit
    assert misa(["m", "zca", "zicsr"]) & c_bit
    assert not misa(["m", "f", "d", "zca", "zicsr"]) & c_bit  # D without Zcd is not C
    assert misa(["m", "f", "d", "zca", "zcd", "zicsr"]) & c_bit
    assert not misa(["m", "zicsr"]) & c_bit


def test_trace_reports_the_16_bit_parcel():
    bus = get_bus()
    events = []
    core = _core()
    core.unit_id = (0, 0)
    _place(
        core, PC, [0x4505, 0x00150593, 0x9302]
    )  # c.li a0, 1; addi a1, a0, 1; c.jalr t1
    _set(core, 6, 0x1800)
    _place(core, 0x1800, [0x0001])
    saved = bus.enabled, {cat: bus._per_category[cat] for cat in EventCategory}
    for cat in EventCategory:
        bus.set_category_enabled(cat, cat == EventCategory.INSTR)
    bus.subscribe(EventCategory.INSTR, events.append)
    bus.enabled = True
    try:
        _step(core, 3)
    finally:
        bus._subscribers[EventCategory.INSTR].remove(events.append)
        bus.enabled = saved[0]
        for cat, on in saved[1].items():
            bus.set_category_enabled(cat, on)
    instrs = [e for e in events if isinstance(e, InstrEvent)]
    assert [
        (e.pc, e.instruction, e.reg_write_idx, e.reg_write_value) for e in instrs
    ] == [
        (PC, 0x4505, 10, 1),
        (PC + 2, 0x00150593, 11, 2),
        (PC + 6, 0x9302, 1, PC + 8),
    ]


def test_cost_model_sees_the_expansion():
    seen = []

    class Probe:
        def can_issue(self, instr, cycle, register_file):
            seen.append(instr)
            return True

        def take_pending_stall(self):
            return 0, ""

        def reset(self):
            pass

    core = _core()
    core.rv_cost = Probe()
    _place(core, PC, [0x4505])
    _step(core)
    assert seen == [0x00100513]


# -- the dispatch table --------------------------------------------------------------

#: Every Naiad-used ISA class with its declared opcodes.
NAIAD_ISAS = [
    RV64_I_ISA,
    RV64_M_ISA,
    RV64_ZAAMO_ISA,
    RV64_ZALRSC_ISA,
    RV64_ZBA_ISA,
    RV64_ZBB_ISA,
    RV64_F_ISA,
    RV64_D_ISA,
    RV64_ZFBFMIN_ISA,
    RV64_ZICSR_ISA,
    RV64_ZCA_ISA,
    RV64_ZCD_ISA,
]


def test_every_naiad_isa_declares_its_opcodes():
    from framework.matrix.ame_isa import RV_AME_ISA

    for isa in NAIAD_ISAS + [RV_AME_ISA]:
        assert isinstance(getattr(isa, "OPCODES", None), frozenset), isa.__name__


@pytest.mark.parametrize("isa", NAIAD_ISAS, ids=lambda i: i.__name__)
def test_an_isa_never_claims_an_opcode_it_does_not_declare(isa):
    """Random words with every undeclared major opcode: ``run`` declines all of them."""
    rng = random.Random(isa.__name__)
    core = _core(
        ["m", "a", "zba", "zbb", "f", "d", "zfbfmin", "zicsr", "zicntr"], error=False
    )
    rf = core.register_file
    for op in range(128):
        if op in isa.OPCODES:
            continue
        for _ in range(64):
            word = (rng.getrandbits(25) << 7) | op
            assert not isa.run(rf, core.visible_memory, False, word), (
                isa.__name__,
                hex(word),
            )


def test_table_matches_the_plain_chain():
    """For random words, the first claimant in the table is the first claimant in the chain."""
    core = _core(
        ["m", "a", "zba", "zbb", "f", "d", "zfbfmin", "zicsr", "zicntr", "c"],
        error=False,
    )
    rng = random.Random(7)
    rf = core.register_file
    for _ in range(20000):
        word = rng.getrandbits(32) | 3
        chain = next((isa for isa in core.isas if _claims(isa, rf, core, word)), None)
        table = next(
            (
                run.__self__
                for run in core._dispatch[word & 0x7F]
                if _claims(run.__self__, rf, core, word)
            ),
            None,
        )
        assert chain is table, hex(word)


def _claims(isa, rf, core, word):
    """Whether ``isa`` claims ``word``; the core's state is restored after."""
    saved = [r.value for r in rf.registers]
    reservation = rf.reservation
    try:
        return bool(isa.run(rf, core.visible_memory, False, word))
    except Exception:
        return True  # raising is claiming (an illegal-instruction stand-in)
    finally:
        for r, v in zip(rf.registers, saved):
            r.value = v
        rf.reservation = reservation


def test_shared_opcodes_keep_chain_order():
    table = dispatch_table(
        [RV64_I_ISA, RV64_M_ISA, RV64_ZBA_ISA, RV64_F_ISA, RV64_ZFBFMIN_ISA],
        ["i", "m", "zba", "f", "zfbfmin"],
    )
    assert table[0x33] == ("i", "m", "zba")
    assert table[0x07] == ("f", "zfbfmin")
    assert table[0x2F] == ()


def test_an_isa_without_opcodes_is_offered_everything():
    class Legacy:
        pass

    table = dispatch_table([RV64_I_ISA, Legacy], ["i", "legacy"])
    assert all("legacy" in entry for entry in table)
    assert table[0x13] == ("i", "legacy")


def test_assigning_or_mutating_isa_runs_rebuilds_the_table():
    core = _core(["m"])

    def wrapped(run):
        def w(*args):
            return run(*args)

        return w

    core._isa_runs = [wrapped(r) for r in core._isa_runs]
    assert core._dispatch[0x13][0] is core._isa_runs[0]

    class Extra:
        OPCODES = frozenset({0x0B})

        @classmethod
        def run(cls, *args):
            return False

    core.isas.append(Extra)
    core._isa_runs.append(Extra.run)  # in place, as ame_isa.attach does
    assert core._dispatch[0x0B] == (Extra.run,)
    assert core._isa_runs == [wrapped_run for wrapped_run in core._isa_runs]


def test_add_isa():
    core = _core(["m"])

    class Extra:
        OPCODES = frozenset({0x5B})

        @classmethod
        def run(cls, *args):
            return False

    core.add_isa(Extra)
    core.add_isa(Extra)
    assert core.isas.count(Extra) == 1
    assert core._dispatch[0x5B] == (Extra.run,)


def test_mismatched_chain_is_refused():
    core = _core(["m"])
    with pytest.raises(ValueError, match="ISA run callables"):
        core._isa_runs = core._isa_runs[:1]
