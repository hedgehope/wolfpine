"""Tests for the Naiad-vs-Spike differential runner.

The comparison logic is tested on synthetic streams, so it runs everywhere. The
end-to-end tests build ``kernels/naiad`` programs with clang and run them on
Spike, and are skipped when either is missing.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from framework.naiad import DESIGNS_DIR
from framework.naiad.difftest import difftest
from framework.pe.rv.difftest import (
    RunOutcome,
    SpikeCommit,
    compare_commits,
    compare_console,
    find_spike,
    parse_spike_commits,
    spike_isa,
)
from framework.pe.rv.isa.isa64_testing import PC, j_type, make_core
from framework.trace import EventCategory, get_bus

REPO = Path(__file__).resolve().parents[2]
KERNELS = REPO / "kernels" / "naiad"
CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")
RV64 = DESIGNS_DIR / "naiad-base-rv64.yaml"

ENTRY = 0x80000000
ADDI_X5 = 0x00128293  # addi t0, t0, 1
LD_T1 = 0x0002B303  # ld t1, 0(t0)
BNEZ_BACK = 0xFE031EE3  # bnez t1, -4
CSRR_CYCLE = 0xC0002573  # csrr a0, cycle
ADDI_A1_A0 = 0x00150593  # addi a1, a0, 1


def _s(pc, instr, rd=-1, value=0, mem=None, store=False):
    return SpikeCommit(pc, instr, rd, value, mem, store)


# -- pure logic --------------------------------------------------------------


def test_spike_isa_adds_zicntr_with_zicsr():
    assert spike_isa("rv64im_zicsr") == "rv64im_zicsr_zicntr"
    assert spike_isa("rv64ima_zicsr_zicntr_zba") == "rv64ima_zicsr_zicntr_zba"
    assert spike_isa("rv32im") == "rv32im"


def test_parse_spike_commit_lines():
    lines = [
        "core   0: 3 0x0000000080000000 (0x00001197) x3  0x0000000080001000\n",
        "core   0: 3 0x000000008000000c (0x0182b283) x5  0x0000000080000000 mem 0x0000000000001018\n",
        "core   0: 3 0x0000000080000024 (0x0002b023) mem 0x0000000080000d90 0x0000000000000000\n",
        "core   0: 3 0x0000000080000028 (0xff5ff06f)\n",
        "core   0: >>>>  main\n",
    ]
    got = list(parse_spike_commits(lines))
    assert got == [
        _s(0x80000000, 0x1197, 3, 0x80001000),
        _s(0x8000000C, 0x0182B283, 5, 0x80000000, 0x1018, False),
        _s(0x80000024, 0x0002B023, -1, 0, 0x80000D90, True),
        _s(0x80000028, 0xFF5FF06F),
    ]


def test_streams_align_on_entry_and_agree():
    boot = [(0x1000, 0x297, 5, 0x1000)]
    body = [(ENTRY, ADDI_X5, 5, 1), (ENTRY + 4, ADDI_X5, 5, 2)]
    spike_boot = [_s(0x1000, 0x297, 5, 0x1000), _s(0x1004, 0x02028593, 11, 0x1020)]
    diff = compare_commits(
        boot + body, spike_boot + [_s(*c) for c in body], entry=ENTRY
    )
    assert diff.ok, diff.report()
    assert diff.matched == 2


def test_value_divergence_is_reported_with_pc_and_values():
    naiad = [(ENTRY, ADDI_X5, 5, 1), (ENTRY + 4, ADDI_X5, 5, 3)]
    spike = [_s(ENTRY, ADDI_X5, 5, 1), _s(ENTRY + 4, ADDI_X5, 5, 2)]
    diff = compare_commits(naiad, spike, entry=ENTRY)
    assert not diff.ok
    assert not diff.tainted
    assert diff.matched == 1
    assert "value" in diff.divergence
    assert "0x0000000080000004" in diff.divergence
    assert "0x0000000000000003" in diff.divergence
    assert "0x0000000000000002" in diff.divergence


VADD_V2 = 0x02430157  # vadd.vv v2, v4, v6
VSETVLI = 0x0112F357  # vsetvli t1, t0, e32, m2, tu, mu
VLE8_V8 = 0x02038407  # vle8.v v8, (t2)


def _vrf(**regs):
    """A 32 x 32-byte register file (VLEN 256) with the given registers set."""
    data = bytearray(32 * 32)
    for name, value in regs.items():
        r = int(name[1:])
        data[r * 32 : (r + 1) * 32] = value.to_bytes(32, "little")
    return bytes(data)


def test_parse_spike_vector_writes():
    v2, v3 = "00" * 31 + "1f", "ab" + "00" * 31
    line = (
        f"core   0: 3 0x0000000080000014 (0x02430157) e32 m2 l16 v2  0x{v2} v3  0x{v3} "
        "c8_vstart 0x0000000000000000\n"
    )
    (c,) = parse_spike_commits([line])
    assert c.rd == -1
    assert c.vregs == ((2, 0x1F), (3, 0xAB << 248))
    # A scalar write after the vtype prefix still parses, with no vector writes.
    (c,) = parse_spike_commits(
        [
            "core   0: 3 0x0000000080000018 (0x42202557) e32 m2 l16 c8_vstart 0x0000000000000000 x10 0x0000000000000007\n"
        ]
    )
    assert (c.rd, c.value, c.vregs) == (10, 7, ())


def test_is_vector_instr():
    from framework.pe.rv.difftest import is_vector_instr

    assert is_vector_instr(VADD_V2)
    assert is_vector_instr(VSETVLI)
    assert is_vector_instr(VLE8_V8)
    assert not is_vector_instr(0x0002B007)  # fld ft0, 0(t0): LOAD-FP, width 3
    assert not is_vector_instr(ADDI_X5)


def test_vector_writes_are_compared():
    naiad = [(ENTRY, ADDI_X5, 5, 1), (ENTRY + 4, VADD_V2, -1, 0)]
    spike = [
        _s(ENTRY, ADDI_X5, 5, 1),
        SpikeCommit(ENTRY + 4, VADD_V2, -1, 0, vregs=((2, 5), (3, 6))),
    ]
    good = [None, _vrf(v2=5, v3=6)]
    diff = compare_commits(naiad, spike, entry=ENTRY, naiad_vregs=good)
    assert diff.ok, diff.report()
    assert diff.matched == 2
    assert diff.vector_writes == 2
    assert "2 vector writes" in diff.report()
    bad = [None, _vrf(v2=5, v3=7)]
    diff = compare_commits(naiad, spike, entry=ENTRY, naiad_vregs=bad)
    assert not diff.ok
    assert diff.matched == 1
    assert "v3" in diff.divergence
    assert "vector value" in diff.divergence
    # Without the register files (a core with no vector unit) only scalars are compared.
    assert compare_commits(naiad, spike, entry=ENTRY).ok


def test_vector_compare_indexes_survive_boot_records_and_resyncs():
    # Records before the entry point shift the indices: the register file used
    # must be the one recorded with the same instruction.
    naiad = [(0x1000, 0x297, 5, 0x1000), (ENTRY, VADD_V2, -1, 0)]
    spike = [
        _s(0x1000, 0x297, 5, 0x1000),
        SpikeCommit(ENTRY, VADD_V2, -1, 0, vregs=((2, 9),)),
    ]
    vrfs = [_vrf(v2=1), _vrf(v2=9)]
    assert compare_commits(naiad, spike, entry=ENTRY, naiad_vregs=vrfs).ok


def test_htif_spin_loop_resyncs():
    # Waiting for tohost to clear: Naiad sees 0 at once, Spike loops three times.
    tohost = 0x80001000
    after = (ENTRY + 8, ADDI_X5, 5, 7)
    naiad = [(ENTRY, LD_T1, 6, 0), (ENTRY + 4, BNEZ_BACK, -1, 0), after]
    spike = [
        _s(ENTRY, LD_T1, 6, 0x0101000000000041, tohost),
        _s(ENTRY + 4, BNEZ_BACK),
        _s(ENTRY, LD_T1, 6, 0x0101000000000041, tohost),
        _s(ENTRY + 4, BNEZ_BACK),
        _s(ENTRY, LD_T1, 6, 0, tohost),
        _s(ENTRY + 4, BNEZ_BACK),
        _s(*after),
    ]
    diff = compare_commits(
        naiad, spike, entry=ENTRY, volatile_ranges=[(tohost, tohost + 8)]
    )
    assert diff.ok, diff.report()
    assert diff.resyncs == 1
    assert diff.matched == 3


def test_counter_reads_taint_their_dependents():
    naiad = [(ENTRY, CSRR_CYCLE, 10, 5), (ENTRY + 4, ADDI_A1_A0, 11, 6)]
    spike = [_s(ENTRY, CSRR_CYCLE, 10, 900), _s(ENTRY + 4, ADDI_A1_A0, 11, 901)]
    diff = compare_commits(naiad, spike, entry=ENTRY)
    assert diff.ok, diff.report()
    assert diff.volatile_reads == 1


def test_console_comparison_ignores_matching_lines():
    naiad = RunOutcome(b"hi\ncycles: 5\nPASS\n", 0)
    spike = RunOutcome(b"hi\ncycles: 900\nPASS\n", 0)
    assert compare_console(naiad, spike)
    assert compare_console(naiad, spike, ignore=[r"^cycles"]) == []
    assert compare_console(naiad, RunOutcome(spike.console, 1), ignore=[r"^cycles"])


def test_instr_event_reports_the_link_register_of_a_jump():
    # jal writes rd and then nextpc; the trace must report rd, as Spike does.
    core, _ = make_core()
    core.visible_memory.write(PC, j_type(16, rd=1).to_bytes(4, "little"))
    core.unit_id = ("test", 0, 0, "hart0")
    events = []
    bus = get_bus()
    bus.subscribe(EventCategory.INSTR, events.append)
    bus.enabled = True
    try:
        core.clock_tick(0)
    finally:
        bus.reset()
    assert [(e.pc, e.reg_write_idx, e.reg_write_value) for e in events] == [
        (PC, 1, PC + 4)
    ]


def test_instr_event_reports_rd_of_an_fp_op_that_also_sets_flags():
    # fcvt.w.d writes rd and then fcsr (inexact); the trace must report rd.
    import struct

    from framework.pe.rv.isa.fd_isa import RV64_D_ISA, RV64_F_ISA
    from framework.pe.rv.isa.zicsr64_isa import RV64_ZICSR_ISA
    from framework.pe.rv.rv32 import FP_REGISTER_BASE

    core, _ = make_core([RV64_F_ISA, RV64_D_ISA, RV64_ZICSR_ISA])
    core.visible_memory.write(
        PC, (0xC2079653).to_bytes(4, "little")
    )  # fcvt.w.d a2, fa5, rtz
    core.register_file[FP_REGISTER_BASE + 15].write(struct.pack("<d", -2.5))
    core.unit_id = ("test", 0, 0, "hart0")
    events = []
    bus = get_bus()
    bus.subscribe(EventCategory.INSTR, events.append)
    bus.enabled = True
    try:
        core.clock_tick(0)
    finally:
        bus.reset()
    assert [(e.reg_write_idx, e.reg_write_value) for e in events] == [
        (12, (-2) & (2**64 - 1))
    ]
    assert core.csrs.read(0x001) & 0x1  # NX was set, after rd


# -- end to end against Spike ------------------------------------------------

needs_tools = pytest.mark.skipif(
    find_spike() is None or not Path(CC).is_file(),
    reason="needs spike and clang (tools/naiad/build_spike.sh)",
)


def _build(tmp_path, prog_dir, design=RV64):
    build = tmp_path / "build"
    subprocess.run(
        [
            "make",
            "-s",
            "-C",
            str(KERNELS),
            f"DESIGN={design}",
            f"PROG={prog_dir.name}",
            f"PROG_DIR={prog_dir}",
            f"BUILD={build}",
            f"NAIAD_CC={CC}",
            f"PYTHON={sys.executable}",
        ],
        check=True,
        capture_output=True,
    )
    return build / f"{prog_dir.name}.elf"


@needs_tools
def test_hello_agrees_with_spike(tmp_path):
    elf = _build(tmp_path, KERNELS / "hello")
    # Spike's cycle counter counts its HTIF spin loops too, so that line differs.
    result = difftest(RV64, elf, ignore=[r"^cycles in main"])
    assert result.ok, result.report()
    assert result.naiad.exit_code == 0
    assert b"PASS" in result.naiad.console


@needs_tools
def test_commit_streams_agree_for_a_counter_free_program(tmp_path):
    prog = tmp_path / "sums"
    prog.mkdir()
    (prog / "sums.c").write_text(
        """
#include "naiad_platform.h"
static long long tri(long long n) { return n ? n + tri(n - 1) : 0; }
int main(void) {
    volatile long long x = -123456789012345LL;
    unsigned long long h = 1469598103934665603ULL;
    for (int i = 0; i < 64; i++) {
        h = (h ^ (unsigned)i) * 1099511628211ULL;
    }
    naiad_printf("%lld %lld %llu\\n", tri(40), x / 7, h % 1000003);
    return (int)(h & 3);
}
"""
    )
    elf = _build(tmp_path, prog)
    result = difftest(RV64, elf, commits=True)
    assert result.ok, result.report()
    assert result.commits.matched > 500


@needs_tools
def test_fpcheck_agrees_with_spike_commit_by_commit(tmp_path):
    elf = _build(
        tmp_path, KERNELS / "fpcheck", design=DESIGNS_DIR / "naiad-fp-rv64.yaml"
    )
    result = difftest(DESIGNS_DIR / "naiad-fp-rv64.yaml", elf, commits=True)
    assert result.ok, result.report()
    assert result.naiad.exit_code == 0
