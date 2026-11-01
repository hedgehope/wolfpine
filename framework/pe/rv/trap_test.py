"""Trap and interrupt delivery on Naiad cores (``framework.pe.rv.trap``).

Programs are hand-assembled words run on a packaged design (``naiad-base-rv64``,
and ``naiad-base`` for RV32), so no compiler is needed. The handler records
``mcause`` / ``mtval`` / ``mepc`` / ``mstatus`` and a count into a results table
at ``RES``; each test reads the table back after the run.
"""

import pytest
import yaml

from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import NaiadRunError, build
from framework.naiad.design import DesignError, load_design, parse_design
from framework.pe.rv.isa.v.state import UINT
from framework.pe.rv.isa.v.vtesting import vmem, vsetivli, vtype
from framework.util.elf_testing import write_elf

BASE = 0x80000000
TOHOST = BASE + 0x1000
RES = BASE + 0x2000
HANDLER = BASE + 0x400
CLINT = 0x02000000
MTIMECMP = CLINT + 0x4000
MTIME = CLINT + 0xBFF8

X0, T0, T1, S0, S1, A0, S2, S3, S5, T3, T4 = 0, 5, 6, 8, 9, 10, 18, 19, 21, 28, 29
T5, T6 = 30, 31
MSTATUS, MIE, MTVEC, MEPC, MCAUSE, MTVAL, MIP = (
    0x300,
    0x304,
    0x305,
    0x341,
    0x342,
    0x343,
    0x344,
)
MIE_BIT = 1 << 3
INTERRUPT64 = 1 << 63

NOP = 0x13
SPIN = 0x6F


def itype(op, rd, f3, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def addi(rd, rs1, imm):
    return itype(0x13, rd, 0, rs1, imm)


def slli(rd, rs1, sh):
    return itype(0x13, rd, 1, rs1, sh)


def lui(rd, imm20):
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | 0x37


def add(rd, rs1, rs2):
    return (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33


def load(f3, rd, rs1, imm):
    return itype(0x03, rd, f3, rs1, imm)


def store(f3, rs1, rs2, imm):
    return (
        ((imm >> 5 & 0x7F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (f3 << 12)
        | ((imm & 0x1F) << 7)
        | 0x23
    )


def jal(rd, offset):
    return (
        ((offset >> 20 & 1) << 31)
        | ((offset >> 1 & 0x3FF) << 21)
        | ((offset >> 11 & 1) << 20)
        | ((offset >> 12 & 0xFF) << 12)
        | (rd << 7)
        | 0x6F
    )


def jalr(rd, rs1, imm):
    return itype(0x67, rd, 0, rs1, imm)


def csrrw(rd, csr, rs1):
    return itype(0x73, rd, 1, rs1, csr)


def csrrs(rd, csr, rs1):
    return itype(0x73, rd, 2, rs1, csr)


def csrr(rd, csr):
    return csrrs(rd, csr, X0)


def csrw(csr, rs1):
    return csrrw(X0, csr, rs1)


def csrs(csr, rs1):
    return csrrs(X0, csr, rs1)


def csrsi(csr, imm):
    return itype(0x73, X0, 6, imm, csr)


def csrci(csr, imm):
    return itype(0x73, X0, 7, imm, csr)


ECALL, EBREAK, MRET, WFI = 0x73, 0x100073, 0x30200073, 0x10500073


class Prog:
    """A word list with the RV64/RV32 differences (register width, big addresses) folded in."""

    def __init__(self, xlen):
        self.xlen = xlen
        self.ld = 3 if xlen == 64 else 2
        self.words = []

    def pc(self):
        return BASE + 4 * len(self.words)

    def emit(self, *words):
        self.words.extend(words)
        return self

    def li(self, rd, value):
        """``rd = value`` for a non-negative address (up to 2**32) on either width."""
        if self.xlen == 32 or value < 0x80000000:
            self.emit(lui(rd, (value + 0x800) >> 12), addi(rd, rd, value & 0xFFF or 0))
            if value & 0xFFF == 0:
                self.words.pop()
            return self
        # lui would sign-extend a value with bit 31 set: build it as 1 << 31 plus the rest.
        rest = value - 0x80000000
        self.emit(addi(rd, X0, 1), slli(rd, rd, 31))
        if rest:
            self.emit(lui(S0, (rest + 0x800) >> 12), add(rd, rd, S0))
            if rest & 0xFFF:
                self.emit(addi(rd, rd, ((rest & 0xFFF) ^ 0x800) - 0x800))
        return self

    def sd(self, rs1, rs2, imm):
        return store(self.ld, rs1, rs2, imm)

    def ldw(self, rd, rs1, imm):
        return load(self.ld, rd, rs1, imm)

    def setup(self, mtvec=HANDLER, with_clint=True):
        self.li(S1, RES)
        if with_clint:
            self.li(S2, MTIMECMP)
            self.li(S3, CLINT)
        if mtvec is not None:
            self.li(T0, mtvec)
            self.emit(csrw(MTVEC, T0))
        return self

    def exit(self):
        self.li(T0, TOHOST)
        self.emit(addi(A0, X0, 1), store(2, T0, A0, 0), SPIN)
        return self


def handler(prog, kind):
    """The handler body for ``kind``: record the trap, then skip / redirect / quiet the source."""
    words = [
        csrr(T3, MCAUSE),
        prog.sd(S1, T3, 0),
        csrr(T3, MTVAL),
        prog.sd(S1, T3, 8),
        csrr(T3, MEPC),
        prog.sd(S1, T3, 16),
        csrr(T3, MSTATUS),
        prog.sd(S1, T3, 24),
        prog.ldw(T3, S1, 32),
        addi(T3, T3, 1),
        prog.sd(S1, T3, 32),
    ]
    if kind == "skip":
        words += [csrr(T3, MEPC), addi(T3, T3, 4), csrw(MEPC, T3)]
    elif kind == "redirect":
        words += [csrw(MEPC, S5)]
    elif kind == "timer":
        words += [addi(T3, X0, -1), prog.sd(S2, T3, 0)]
    elif kind == "soft":
        words += [store(2, S3, X0, 0)]
    return [*words, MRET]


def run(
    tmp_path,
    main,
    handler_words=None,
    design="naiad-base-rv64",
    max_cycles=30000,
    vectors=None,
):
    code = main.words + [NOP] * (0x100 - len(main.words))
    assert len(main.words) <= 0x100
    tail = handler_words or []
    if vectors is not None:  # a vector table at HANDLER, handlers after it
        tail = vectors + tail
    code = code + tail
    blob = b"".join(w.to_bytes(4, "little") for w in code)
    elf = write_elf(
        tmp_path / "prog.elf",
        [(BASE, blob, len(blob)), (TOHOST, bytes(0x1000 + 0x48), 0x1048)],
        BASE,
        {"tohost": TOHOST, "fromhost": TOHOST + 0x40},
        elfclass=main.xlen,
    )
    if isinstance(design, str):
        design = load_design(DESIGNS_DIR / f"{design}.yaml")
    soc = build(design)
    soc.htif.console_stream = None
    soc.load(elf)
    result = soc.run(max_cycles)
    width = 8 if main.xlen == 64 else 4

    def results(i):
        return int.from_bytes(soc.memory.read(RES + 8 * i, width), "little")

    return soc, result, results


def prog64():
    return Prog(64)


def test_ecall_vectors_through_mtvec_and_mret_resumes(tmp_path):
    p = prog64().setup()
    p.emit(csrsi(MSTATUS, 8))  # MIE on, so mret has something to restore
    ecall_pc = p.pc()
    p.emit(ECALL)
    p.emit(csrr(T4, MSTATUS), p.sd(S1, T4, 48))
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "skip"))
    assert result.exit_code == 0
    assert res(0) == 11  # ecall from M-mode
    assert res(1) == 0  # mtval
    assert res(2) == ecall_pc
    assert res(4) == 1
    inside = res(3)
    assert inside & MIE_BIT == 0  # MIE cleared on entry
    assert inside & (1 << 7)  # MPIE = old MIE
    assert inside >> 11 & 3 == 3  # MPP = M
    after = res(6)
    assert after & MIE_BIT  # mret: MIE = MPIE
    assert after & (1 << 7)  # MPIE = 1


def test_illegal_instruction_reports_the_word_in_mtval(tmp_path):
    p = prog64().setup()
    bad_pc = p.pc()
    p.emit(0xFFFFFFFF)
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "skip"))
    assert result.exit_code == 0
    assert (res(0), res(1), res(2)) == (2, 0xFFFFFFFF, bad_pc)


def test_unimplemented_csr_is_an_illegal_instruction(tmp_path):
    p = prog64().setup()
    p.emit(csrr(T4, 0x7C0))  # not a CSR this core has
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "skip"))
    assert result.exit_code == 0
    assert res(0) == 2
    assert res(1) == csrr(T4, 0x7C0)


def test_ebreak_sets_mtval_to_its_pc(tmp_path):
    p = prog64().setup()
    pc = p.pc()
    p.emit(EBREAK)
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "skip"))
    assert (res(0), res(1), res(2)) == (3, pc, pc)


@pytest.mark.parametrize(
    "access, cause",
    [
        ("ld", 4),
        ("lw", 4),
        ("sd", 6),
        ("sh", 6),
    ],
)
def test_misaligned_access_traps_with_the_address(tmp_path, access, cause):
    p = prog64().setup()
    offset = {"ld": 7, "lw": 2, "sd": 7, "sh": 1}[access]
    p.emit(addi(T4, S1, offset))
    insn = {
        "ld": load(3, T3, T4, 0),
        "lw": load(2, T3, T4, 0),
        "sd": store(3, T4, T3, 0),
        "sh": store(1, T4, T3, 0),
    }[access]
    p.emit(insn)
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "skip"))
    assert result.exit_code == 0
    assert (res(0), res(1)) == (cause, RES + offset)


def test_aligned_and_unarmed_misaligned_accesses_do_not_trap(tmp_path):
    # Armed: an aligned ld is fine. Unarmed (mtvec = 0): a misaligned one just works, as it always did.
    p = prog64().setup(mtvec=None)
    p.emit(load(3, T3, S1, 7), p.sd(S1, T3, 8))
    p.emit(csrw(MTVEC, X0))
    p.exit()
    _, result, res = run(tmp_path, p)
    assert result.exit_code == 0
    assert res(4) == 0


@pytest.mark.parametrize("access, cause", [("ld", 5), ("sd", 7)])
def test_access_fault_on_unmapped_address(tmp_path, access, cause):
    p = prog64().setup()
    p.emit(addi(T4, X0, 0x10))
    p.emit(load(3, T3, T4, 0) if access == "ld" else store(3, T4, T3, 0))
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "skip"))
    assert result.exit_code == 0
    assert (res(0), res(1)) == (cause, 0x10)


def test_misaligned_jump_target(tmp_path):
    p = prog64().setup()
    p.li(T4, BASE + 0x202)
    pc = p.pc()
    p.emit(jalr(X0, T4, 0))
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "skip"))
    assert result.exit_code == 0
    assert (res(0), res(1), res(2)) == (0, BASE + 0x202, pc)


def test_fetch_access_fault(tmp_path):
    # Jump to an unmapped address; the handler sends mepc to s5, the instruction after the jump.
    resume = BASE
    for _ in range(
        3
    ):  # s5's load is the same length whatever its value; settle on the address
        p = prog64().setup()
        p.li(S5, resume)
        p.li(T4, 0x10)
        p.emit(jalr(X0, T4, 0))
        resume, previous = p.pc(), resume
        if resume == previous:
            break
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "redirect"))
    assert result.exit_code == 0
    assert (res(0), res(1), res(2)) == (1, 0x10, 0x10)


def test_software_interrupt_waits_for_mie_then_handler_clears_msip(tmp_path):
    p = prog64().setup()
    p.emit(addi(T4, X0, 8), csrw(MIE, T4))  # mie.MSIE
    p.emit(addi(T4, X0, 1), store(2, S3, T4, 0))  # msip[0] = 1
    p.emit(csrr(T4, MIP), p.sd(S1, T4, 56))  # pending in mip even with MIE clear
    p.emit(p.ldw(T3, S1, 32), p.sd(S1, T3, 64))  # count so far: handler has not run
    p.emit(csrsi(MSTATUS, 8))  # MIE on: taken before the next instruction
    p.emit(NOP)
    p.emit(csrci(MSTATUS, 8))
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "soft"))
    assert result.exit_code == 0
    assert res(7) & 8  # mip.MSIP (and MTIP too: mtimecmp resets to 0)
    assert res(8) == 0
    assert res(0) == INTERRUPT64 | 3
    assert res(1) == 0
    assert res(4) == 1


def test_timer_interrupt_wakes_wfi_and_returns_after_it(tmp_path):
    p = prog64().setup()
    p.li(T4, MTIME)
    p.emit(
        p.ldw(T3, T4, 0), addi(T3, T3, 300), p.sd(S2, T3, 0)
    )  # mtimecmp = mtime + 300
    p.emit(addi(T4, X0, 0x80), csrw(MIE, T4))  # mie.MTIE
    p.emit(csrsi(MSTATUS, 8))
    wfi_pc = p.pc()
    p.emit(WFI)
    p.emit(csrci(MSTATUS, 8))
    p.exit()
    soc, result, res = run(tmp_path, p, handler(p, "timer"))
    assert result.exit_code == 0
    assert res(0) == INTERRUPT64 | 7
    assert res(2) == wfi_pc + 4  # resumes after the wfi
    assert res(4) == 1
    core = soc.cores[0]
    assert core.csrs.cycles > 250  # it slept through the wait ...
    assert core.csrs.retired < 150  # ... instead of spinning


def test_interrupt_masked_by_mie_and_mstatus_still_ends_wfi(tmp_path):
    # mtimecmp resets to 0, so the timer is due at once: MTIE alone (MIE clear) ends wfi, no trap.
    p = prog64().setup()
    p.emit(addi(T4, X0, 0x80), csrw(MIE, T4), WFI)
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "timer"))
    assert result.exit_code == 0
    assert res(4) == 0


def test_vectored_mode_sends_interrupts_to_base_plus_four_times_cause(tmp_path):
    p = prog64().setup(mtvec=HANDLER | 1)
    p.li(T4, MTIME)
    p.emit(p.ldw(T3, T4, 0), addi(T3, T3, 100), p.sd(S2, T3, 0))
    p.emit(addi(T4, X0, 0x80), csrw(MIE, T4), csrsi(MSTATUS, 8), WFI)
    p.emit(
        addi(T4, X0, 0), ECALL
    )  # an exception in vectored mode goes to the base entry
    p.exit()
    body = handler(p, "timer")
    skip = handler(p, "skip")
    # Table: entry 0 (exceptions) -> skip, entry 7 (timer) -> timer, the rest -> a spin.
    skip_at = 16 * 4
    timer_at = skip_at + 4 * len(skip)
    table = []
    for n in range(16):
        target = {0: skip_at, 7: timer_at}.get(n)
        table.append(jal(X0, target - 4 * n) if target is not None else SPIN)
    _, result, res = run(tmp_path, p, skip + body, vectors=table)
    assert result.exit_code == 0
    assert res(4) == 2  # one timer entry, one exception entry
    assert res(0) == 11  # the last trap was the ecall


def test_mtvec_arms_and_disarms_the_interpreter(tmp_path):
    p = prog64().setup(mtvec=None)
    p.emit(addi(T0, X0, 0x40), csrw(MTVEC, T0), csrr(T4, MTVEC), p.sd(S1, T4, 0))
    p.emit(
        addi(T0, X0, 0x47), csrw(MTVEC, T0), csrr(T4, MTVEC), p.sd(S1, T4, 8)
    )  # mode 3 is reserved
    p.emit(csrw(MTVEC, X0))
    p.exit()
    soc, result, res = run(tmp_path, p)
    assert result.exit_code == 0
    assert (res(0), res(1)) == (0x40, 0x45)  # mtvec[1] reads zero
    assert "clock_tick" not in vars(soc.cores[0])  # disarmed again


def test_unarmed_core_keeps_the_old_behaviour(tmp_path):
    # ecall / ebreak / mret are nops with no mtvec, and the plain interpreter stays installed.
    p = prog64().setup(mtvec=None)
    p.emit(ECALL, MRET, csrw(MIP, X0), addi(T4, X0, 1))
    p.exit()
    soc, result, _ = run(tmp_path, p)
    assert result.exit_code == 0
    assert "clock_tick" not in vars(soc.cores[0])


def test_ebreak_without_a_handler_ends_the_run_with_an_error(tmp_path):
    p = prog64().setup(mtvec=None)
    p.emit(EBREAK)
    p.exit()
    with pytest.raises(NaiadRunError, match="ebreak"):
        run(tmp_path, p)


def test_illegal_instruction_without_a_handler_ends_the_run_with_an_error(tmp_path):
    p = prog64().setup(mtvec=None)
    p.emit(0xFFFFFFFF)
    p.exit()
    with pytest.raises(NaiadRunError, match="Unknown instruction"):
        run(tmp_path, p)


def test_unmapped_load_without_a_handler_ends_the_run_with_an_error(tmp_path):
    p = prog64().setup(mtvec=None)
    p.emit(addi(T4, X0, 0x10), load(3, T3, T4, 0))
    p.exit()
    with pytest.raises(NaiadRunError, match="does not match any registered"):
        run(tmp_path, p)


def test_reset_disarms(tmp_path):
    p = prog64().setup()
    p.exit()
    soc, result, _ = run(tmp_path, p)
    assert result.exit_code == 0
    assert "clock_tick" in vars(soc.cores[0])  # left armed by the program
    soc.reset()
    assert "clock_tick" not in vars(soc.cores[0])


def test_single_hart_wfi_is_a_nop_without_a_wake_source(tmp_path):
    p = prog64().setup(mtvec=None)
    p.emit(WFI, WFI)
    p.exit()
    _, result, _ = run(tmp_path, p)
    assert result.exit_code == 0


def test_single_hart_wfi_sleeps_when_a_source_is_enabled_and_nothing_wakes_it(tmp_path):
    p = prog64().setup(mtvec=None)
    p.emit(addi(T4, X0, 8), csrw(MIE, T4), WFI)  # MSIE, nobody sets msip
    p.exit()
    _, result, _ = run(tmp_path, p, max_cycles=2000)
    assert result.exit_code is None  # still asleep when the budget ended


def test_rv32_core_has_mie_and_takes_traps(tmp_path):
    p = Prog(32).setup()
    p.emit(addi(T4, X0, 0x88), csrw(MIE, T4), csrr(T4, MIE), p.sd(S1, T4, 48))
    p.emit(csrr(T4, MSTATUS), p.sd(S1, T4, 56))
    ecall_pc = p.pc()
    p.emit(ECALL)
    p.exit()
    soc, result, res = run(tmp_path, p, handler(p, "skip"), design="naiad-base")
    assert result.exit_code == 0
    assert res(6) == 0x88
    assert res(7) & 0x80006600 == 0x80006600  # the baby cores' FS / VS reading is kept
    assert (res(0), res(2), res(4)) == (11, ecall_pc, 1)


def test_rv32_unarmed_ecall_is_a_nop_and_illegal_is_an_error(tmp_path):
    p = Prog(32).setup(mtvec=None)
    p.emit(ECALL)
    p.exit()
    assert run(tmp_path, p, design="naiad-base")[1].exit_code == 0
    q = Prog(32).setup(mtvec=None)
    q.emit(0xFFFFFFFF)
    q.exit()
    with pytest.raises(NaiadRunError, match="Unknown instruction"):
        run(tmp_path, q, design="naiad-base")


# -- jal / jalr to a misaligned target leave rd alone ----------------------------


def test_misaligned_jalr_does_not_write_rd(tmp_path):
    p = prog64().setup()
    p.li(T4, BASE + 0x202)
    p.emit(addi(7, X0, 0x123))  # t2 = 0x123, the jalr's rd
    pc = p.pc()
    p.emit(jalr(7, T4, 0))
    p.emit(p.sd(S1, 7, 48))  # runs after the handler skips the jalr
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "skip"))
    assert result.exit_code == 0
    assert (res(0), res(1), res(2)) == (0, BASE + 0x202, pc)
    assert res(6) == 0x123


def test_misaligned_jal_does_not_write_rd(tmp_path):
    p = prog64().setup()
    p.emit(addi(7, X0, 0x123))
    pc = p.pc()
    p.emit(jal(7, 6))  # pc + 6: not a multiple of 4
    p.emit(p.sd(S1, 7, 48))
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "skip"))
    assert result.exit_code == 0
    assert (res(0), res(1), res(2)) == (0, pc + 6, pc)
    assert res(6) == 0x123


def test_misaligned_jalr_with_rd_equal_rs1_keeps_the_register_on_rv32(tmp_path):
    p = Prog(32).setup()
    p.li(T4, BASE + 0x202)
    p.emit(jalr(T4, T4, 0))
    p.emit(p.sd(S1, T4, 48))
    p.exit()
    _, result, res = run(tmp_path, p, handler(p, "skip"), design="naiad-base")
    assert result.exit_code == 0
    assert res(0) == 0
    assert res(6) == BASE + 0x202


def test_misaligned_jump_does_not_retire(tmp_path):
    def retired(target):
        p = prog64().setup()
        p.li(T4, target)
        p.emit(jalr(X0, T4, 0), NOP)
        p.exit()
        return run(tmp_path, p, handler(p, "skip"))[0].cores[0].csrs.retired

    # The trapping jalr is skipped by the handler either way; only the trap path differs:
    # an aligned jump to the instruction after the nop retires the jalr, a misaligned one does not.
    assert retired(BASE + 0x202) < retired(BASE + 0x200)


# -- vector misalignment -----------------------------------------------------------

VDATA = BASE + 0x3000
VSTART = 0x008


def vload(eew, vd, rs1, **kw):
    return vmem(True, eew, vd, rs1, **kw)


def vstore(eew, vd, rs1, **kw):
    return vmem(False, eew, vd, rs1, **kw)


def vector_prog(base_offset, *ops, vl=4, sew=32, armed=True):
    """vl elements of ``sew`` bits; t4 = VDATA + base_offset, t6 = 6 (a stride), 0x11223344 stored at VDATA."""
    p = prog64().setup(mtvec=HANDLER if armed else None)
    p.li(T3, VDATA)
    p.li(T5, 0x11223344)
    p.emit(store(2, T3, T5, 0), addi(T6, X0, 6))
    p.li(T4, VDATA + base_offset)
    p.emit(vsetivli(0, vl, vtype(sew)), *ops)
    p.exit()
    return p


def vector_run(tmp_path, p, hw=None):
    return run(tmp_path, p, hw or handler(p, "skip"), design="naiad-rvv")


@pytest.mark.parametrize("op,cause", [(vload(32, 2, T4), 4), (vstore(32, 2, T4), 6)])
def test_misaligned_unit_stride_vector_access_traps(tmp_path, op, cause):
    _, result, res = vector_run(tmp_path, vector_prog(2, op))
    assert result.exit_code == 0
    assert (res(0), res(1), res(4)) == (cause, VDATA + 2, 1)


def test_aligned_and_unarmed_vector_accesses_do_not_trap(tmp_path):
    _, result, res = vector_run(tmp_path, vector_prog(4, vload(32, 2, T4)))
    assert (result.exit_code, res(4)) == (0, 0)
    _, result, _ = vector_run(tmp_path, vector_prog(2, vload(32, 2, T4), armed=False))
    assert result.exit_code == 0  # unarmed: a misaligned vector load is just a load


def test_byte_vector_accesses_are_never_misaligned(tmp_path):
    p = vector_prog(1, vload(8, 2, T4), vstore(8, 2, T4), sew=8, vl=8)
    _, result, res = vector_run(tmp_path, p)
    assert (result.exit_code, res(4)) == (0, 0)


def test_strided_access_runs_the_elements_before_the_misaligned_one(tmp_path):
    # Stride 6: element 0 at VDATA is aligned, element 1 at VDATA + 6 is not.
    p = vector_prog(0, vload(32, 2, T4, mop=2, rs2=T6))
    h = handler(p, "skip")[:-1] + [csrr(T3, VSTART), p.sd(S1, T3, 72), MRET]
    soc, result, res = vector_run(tmp_path, p, h)
    assert result.exit_code == 0
    assert (res(0), res(1), res(4)) == (4, VDATA + 6, 1)
    assert res(9) == 1  # vstart = the faulting element
    assert int(soc.cores[0].vector.group(2, 1, UINT[32])[0]) == 0x11223344


def test_indexed_segment_and_whole_register_misalignment(tmp_path):
    whole_load = (1 << 25) | (8 << 20) | (T4 << 15) | (6 << 12) | (2 << 7) | 0x07
    whole_store = (1 << 25) | (8 << 20) | (T4 << 15) | (0 << 12) | (2 << 7) | 0x27
    for op in (
        vload(32, 2, T4, mop=1, rs2=4),  # indexed, v4 = 0: every address is the base
        vload(32, 2, T4, nf=2),  # segment: fields of 4 bytes
        whole_load,  # vl1re32.v
    ):
        _, result, res = vector_run(tmp_path, vector_prog(2, op))
        assert (res(0), res(1), res(4)) == (4, VDATA + 2, 1)
    # vs1r.v moves bytes: never misaligned.
    _, result, res = vector_run(tmp_path, vector_prog(2, whole_store))
    assert (result.exit_code, res(4)) == (0, 0)


def test_masked_off_misaligned_elements_do_not_trap(tmp_path):
    # v0 is zero: every element is masked off, so none is accessed.
    _, result, res = vector_run(tmp_path, vector_prog(2, vload(32, 2, T4, vm=0)))
    assert (result.exit_code, res(4)) == (0, 0)


# -- trap timing --------------------------------------------------------------------


def timed_design(**scalar):
    raw = yaml.safe_load((DESIGNS_DIR / "naiad-base-rv64.yaml").read_text())
    raw["timing"] = {"scalar": scalar}
    return parse_design(raw)


def cycles_of(tmp_path, program, handler_words, **scalar):
    soc, result, _ = run(
        tmp_path, program, handler_words, design=timed_design(**scalar)
    )
    assert result.exit_code == 0
    return soc.clock.current_cycle, soc


def test_trap_entry_and_return_are_charged_when_a_trap_happens(tmp_path):
    p = prog64().setup()
    p.emit(ECALL, NOP, NOP)
    p.exit()
    h = handler(p, "skip")
    free, _ = cycles_of(tmp_path, p, h, trap_entry=0, trap_return=0)
    entry, soc = cycles_of(tmp_path, p, h, trap_entry=5, trap_return=0)
    both, _ = cycles_of(tmp_path, p, h, trap_entry=5, trap_return=7)
    assert entry - free == 5
    assert both - entry == 7
    assert soc.counters()["hart0.timing.stall_branch"] >= 5


def test_trap_options_do_not_matter_without_a_trap(tmp_path):
    p = prog64().setup()  # armed, but nothing traps
    p.emit(NOP, NOP)
    p.exit()
    a, _ = cycles_of(tmp_path, p, None, trap_entry=0, trap_return=0)
    b, _ = cycles_of(tmp_path, p, None, trap_entry=9, trap_return=9)
    assert a == b


def test_trap_timing_has_defaults_and_rejects_negatives():
    spec = timed_design().timing.scalar
    assert (spec.trap_entry, spec.trap_return) == (3, 2)
    with pytest.raises(DesignError):
        timed_design(trap_entry=-1)


# -- RV32 wfi sleeps like RV64's -----------------------------------------------------


def test_rv32_wfi_is_a_nop_without_a_wake_source(tmp_path):
    p = Prog(32).setup(mtvec=None)
    p.emit(WFI, WFI)
    p.exit()
    assert run(tmp_path, p, design="naiad-base")[1].exit_code == 0


def test_rv32_wfi_sleeps_when_a_source_is_enabled_and_nothing_wakes_it(tmp_path):
    p = Prog(32).setup(mtvec=None)
    p.emit(addi(T4, X0, 8), csrw(MIE, T4), WFI)
    p.exit()
    _, result, _ = run(tmp_path, p, design="naiad-base", max_cycles=2000)
    assert result.exit_code is None


def test_rv32_timer_interrupt_wakes_wfi(tmp_path):
    p = Prog(32).setup()
    p.li(T4, MTIME)
    p.emit(p.ldw(T3, T4, 0), addi(T3, T3, 300), p.sd(S2, T3, 0))
    p.emit(addi(T4, X0, 0x80), csrw(MIE, T4), csrsi(MSTATUS, 8))
    wfi_pc = p.pc()
    p.emit(WFI, csrci(MSTATUS, 8))
    p.exit()
    soc, result, res = run(tmp_path, p, handler(p, "timer"), design="naiad-base")
    assert result.exit_code == 0
    assert (res(0), res(2), res(4)) == (0x80000007, wfi_pc + 4, 1)
    assert soc.clock.current_cycle > 250
    assert soc.cores[0].csrs.retired < 150


# -- machine external interrupts (plic) ---------------------------------------------

PLIC = 0x0C000000
MEIE = 1 << 11


def plic_design(schedule=(), **params):
    raw = yaml.safe_load((DESIGNS_DIR / "naiad-base-rv64.yaml").read_text())
    if schedule:
        params["schedule"] = [list(entry) for entry in schedule]
    block = {"name": "plic", "type": "plic", "base": PLIC, "window": 0x4000}
    raw["blocks"].append({**block, "params": params})
    return parse_design(raw)


def plic_handler(prog):
    """Record the trap, then clear every pending source (write ~0 to CLEAR)."""
    h = Prog(prog.xlen)
    h.li(T3, PLIC + 8)
    h.emit(addi(T4, X0, -1), h.sd(T3, T4, 0), store(2, S3, X0, 0))  # and msip = 0
    return handler(prog, "none")[:-1] + h.words + [MRET]


def set_meie(p):
    p.emit(addi(T4, X0, 1), slli(T4, T4, 11), csrw(MIE, T4))


def test_scheduled_external_interrupt_wakes_wfi_and_is_taken(tmp_path):
    p = prog64().setup()
    p.li(T4, PLIC + 0x1000)
    p.emit(addi(T3, X0, 1 << 5), p.sd(T4, T3, 0))  # enable[0] = source 5
    set_meie(p)
    p.emit(csrsi(MSTATUS, 8))
    wfi_pc = p.pc()
    p.emit(WFI, csrci(MSTATUS, 8))
    p.exit()
    soc, result, res = run(
        tmp_path, p, plic_handler(p), design=plic_design([(300, 5)]), max_cycles=5000
    )
    assert result.exit_code == 0
    assert (res(0), res(1), res(2), res(4)) == (INTERRUPT64 | 11, 0, wfi_pc + 4, 1)
    core = soc.cores[0]
    assert core.csrs.cycles > 250
    assert core.csrs.retired < 150
    assert soc.blocks["plic"].pending == 0  # the handler cleared it


def test_external_interrupt_needs_an_enabled_source_and_beats_the_others(tmp_path):
    # Source 2 raised through PENDING: no MEIP until enabled for hart 0. Then, with
    # msip pending too, the external interrupt is the one taken.
    p = prog64().setup()
    p.li(T4, PLIC)
    p.li(S5, PLIC + 0x1000)
    p.emit(addi(T3, X0, 4), p.sd(T4, T3, 0))  # pending |= source 2
    p.emit(csrr(T3, MIP), p.sd(S1, T3, 56))
    p.emit(p.ldw(T3, T4, 0), p.sd(S1, T3, 64))
    p.emit(addi(T3, X0, 4), p.sd(S5, T3, 0))  # enable[0] = source 2
    p.emit(csrr(T3, MIP), p.sd(S1, T3, 72))
    p.emit(addi(T3, X0, -1), p.sd(S2, T3, 0))  # mtimecmp = max: no timer interrupt
    p.emit(addi(T3, X0, 1), store(2, S3, T3, 0))  # msip
    p.emit(addi(T3, X0, 0x8), csrw(MIE, T3))  # MSIE
    p.emit(addi(T3, X0, 1), slli(T3, T3, 11), csrs(MIE, T3))  # | MEIE
    p.emit(csrsi(MSTATUS, 8), NOP, csrci(MSTATUS, 8))
    p.exit()
    _, result, res = run(
        tmp_path, p, plic_handler(p), design=plic_design(), max_cycles=5000
    )
    assert result.exit_code == 0
    assert res(7) & MEIE == 0
    assert res(8) == 4
    assert res(9) & MEIE
    assert res(0) == INTERRUPT64 | 11


def test_wfi_with_meie_is_still_a_nop_without_a_plic(tmp_path):
    p = prog64().setup(mtvec=None)
    set_meie(p)
    p.emit(WFI)
    p.exit()
    assert run(tmp_path, p)[1].exit_code == 0


def test_plic_needs_a_clint_and_validates_params():
    raw = yaml.safe_load((DESIGNS_DIR / "naiad-base-rv64.yaml").read_text())
    raw["blocks"] = [b for b in raw["blocks"] if b["type"] != "clint"]
    raw["blocks"].append(
        {"name": "plic", "type": "plic", "base": PLIC, "window": 0x4000}
    )
    with pytest.raises(DesignError, match="clint"):
        parse_design(raw)
    with pytest.raises(DesignError, match="sources"):
        plic_design(sources=65)
    with pytest.raises(DesignError, match="schedule"):
        plic_design([(10, 40)])
