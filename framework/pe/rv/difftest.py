"""Differential testing against Spike: run an ELF on Spike, and compare consoles and commit streams.

The simulator-independent half of :mod:`framework.naiad.difftest` (which runs
the same ELF on a Naiad design and is the CLI): finding and running Spike
(:func:`run_spike`), comparing two :class:`RunOutcome` s' consoles and exit
codes (:func:`compare_console`), and aligning Spike's ``--log-commits`` stream
with a core's retirement stream (:func:`compare_commits`). See that module's
docstring for what is compared and what the comparison allows for (volatile
reads and their taint, HTIF spin loops, branches on tainted values,
:data:`TAINT_HORIZON`).

Spike is found as ``$NAIAD_SPIKE``, then ``spike`` on ``PATH``, then
``~/.local/bin/spike`` (where ``tools/naiad/build_spike.sh`` installs it).
This module imports nothing heavy at load time, so it is cheap to import from
tools and tests that only need :func:`find_spike`.
"""

import os
import re
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from framework.pe.rv.isa.rvc_isa import expand as expand_rvc

#: Spike's ``--isa`` misses one thing Naiad always has: Naiad's RV64 CSR file
#: implements Zicntr whenever Zicsr is present, and without ``zicntr`` Spike
#: traps on ``rdcycle`` (and, with no trap handler, hangs).
_IMPLIED_WITH_ZICSR = ("zicntr",)

#: Counter CSRs: cycle/time/instret (+ hpmcounter3..31), mcycle/minstret (+ mhpmcounter3..31).
_COUNTER_CSRS = frozenset(range(0xC00, 0xC20)) | frozenset(range(0xB00, 0xB20))

DEFAULT_TIMEOUT = 300
_RESYNC_WINDOW = 64
_RESYNC_MAX_SKIP = 5_000_000
_CONTEXT = 5
#: How far (in retired instructions) a tainted branch makes a later divergence "tainted".
TAINT_HORIZON = 2000


class DifftestError(Exception):
    pass


# -- Spike ------------------------------------------------------------------


def find_spike():
    """Path of the Spike binary, or None (see the module docstring for the search order)."""
    env = os.environ.get("NAIAD_SPIKE")
    if env:
        return env if Path(env).is_file() else None
    found = shutil.which("spike")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "spike"
    return str(local) if local.is_file() else None


def spike_isa(march):
    """Spike's ``--isa`` string for a design's ``-march`` string."""
    parts = march.lower().split("_")
    if "zicsr" in parts:
        parts += [ext for ext in _IMPLIED_WITH_ZICSR if ext not in parts]
    return "_".join(parts)


@dataclass
class RunOutcome:
    """What one simulator made of the program."""

    console: bytes
    #: The HTIF exit code, or None if the run did not exit (timeout / cycle limit).
    exit_code: int | None
    #: Free-form detail for a report (cycles, stderr, the command line).
    detail: str = ""


def run_spike(
    elf,
    isa,
    *,
    spike=None,
    timeout=DEFAULT_TIMEOUT,
    commit_log=None,
    signature=None,
    signature_granularity=None,
    extra_args=(),
):
    """Run ``elf`` on Spike. ``commit_log`` / ``signature`` are paths to write those to."""
    spike = spike or find_spike()
    if spike is None:
        raise DifftestError(
            "spike not found (set NAIAD_SPIKE, or run tools/naiad/build_spike.sh)"
        )
    cmd = [spike, f"--isa={isa}"]
    if commit_log is not None:
        cmd += ["--log-commits", f"--log={commit_log}"]
    if signature is not None:
        cmd.append(f"+signature={signature}")
        if signature_granularity is not None:
            cmd.append(f"+signature-granularity={signature_granularity}")
    cmd += [*extra_args, str(elf)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        return RunOutcome(
            e.stdout or b"", None, f"timed out after {timeout}s: {' '.join(cmd)}"
        )
    detail = " ".join(cmd)
    stderr = proc.stderr.decode(errors="replace").strip()
    if stderr:
        detail += "\n" + stderr
    return RunOutcome(proc.stdout, proc.returncode, detail)


# -- console comparison -----------------------------------------------------


def filter_console(text, ignore=()):
    """``text`` (bytes or str) as a list of lines, minus those matching any ``ignore`` regex."""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    patterns = [re.compile(p) for p in ignore]
    return [
        line for line in text.splitlines() if not any(p.search(line) for p in patterns)
    ]


def compare_console(naiad, spike, ignore=()):
    """A list of human-readable differences between two outcomes (empty if they agree)."""
    problems = []
    if naiad.exit_code != spike.exit_code:
        problems.append(f"exit code: naiad {naiad.exit_code}, spike {spike.exit_code}")
    n_lines, s_lines = (
        filter_console(naiad.console, ignore),
        filter_console(spike.console, ignore),
    )
    if n_lines != s_lines:
        for i in range(max(len(n_lines), len(s_lines))):
            n = n_lines[i] if i < len(n_lines) else "<end of output>"
            s = s_lines[i] if i < len(s_lines) else "<end of output>"
            if n != s:
                problems.append(
                    f"console line {i + 1}:\n  naiad: {n!r}\n  spike: {s!r}"
                )
                break
    return problems


# -- commit-log comparison --------------------------------------------------

_SPIKE_LINE = re.compile(r"core\s+\d+: \d 0x([0-9a-f]+) \(0x([0-9a-f]+)\)(.*)")
_SPIKE_XREG = re.compile(r"\bx(\d+)\s+0x([0-9a-f]+)")
_SPIKE_VREG = re.compile(r"\bv(\d+)\s+0x([0-9a-f]+)")
_SPIKE_MEM = re.compile(r"\bmem 0x([0-9a-f]+)(?: 0x([0-9a-f]+))?")


@dataclass(frozen=True)
class SpikeCommit:
    pc: int
    instr: int
    rd: int
    value: int
    mem_addr: int | None = None
    #: True for a store (Spike logs the value), False for a load or no access.
    mem_store: bool = False
    #: Vector registers Spike logs as written, ``((reg, value), ...)`` with each
    #: value the whole register as an integer (byte 0 least significant). Spike
    #: logs the registers an arithmetic instruction's body touched, but not a
    #: vector load's destination.
    vregs: tuple = ()


def parse_spike_commits(lines):
    """Yield :class:`SpikeCommit` s from ``spike --log-commits`` output lines."""
    for line in lines:
        m = _SPIKE_LINE.match(line)
        if m is None:
            continue
        rest = m.group(3)
        rd, value = -1, 0
        x = _SPIKE_XREG.search(rest)
        if x is not None:
            rd, value = int(x.group(1)), int(x.group(2), 16)
        mem = _SPIKE_MEM.search(rest)
        addr = int(mem.group(1), 16) if mem else None
        vregs = tuple((int(r), int(v, 16)) for r, v in _SPIKE_VREG.findall(rest))
        yield SpikeCommit(
            int(m.group(1), 16),
            int(m.group(2), 16),
            rd,
            value,
            addr,
            bool(mem and mem.group(2)),
            vregs,
        )


def _sources(instr):
    """The x-registers ``instr`` reads (for taint propagation)."""
    opcode = instr & 0x7F
    rs1, rs2 = (instr >> 15) & 0x1F, (instr >> 20) & 0x1F
    if opcode in (0x33, 0x3B, 0x2F, 0x23, 0x63):  # OP, OP-32, AMO, STORE, BRANCH
        return (rs1, rs2)
    if opcode in (0x13, 0x1B, 0x03, 0x67):  # OP-IMM, OP-IMM-32, LOAD, JALR
        return (rs1,)
    if opcode == 0x73 and ((instr >> 12) & 0x7) in (
        1,
        2,
        3,
    ):  # csrrw/s/c (register forms)
        return (rs1,)
    return ()


def is_vector_instr(instr):
    """True for an RVV instruction that may write a vector register (OP-V, or a vector load)."""
    opcode = instr & 0x7F
    return opcode == 0x57 or (opcode == 0x07 and (instr >> 12) & 0x7 in (0, 5, 6, 7))


def _is_counter_read(instr):
    return (
        instr & 0x7F == 0x73
        and (instr >> 12) & 0x7 != 0
        and (instr >> 20) in _COUNTER_CSRS
    )


def _is_branch(instr):
    return instr & 0x7F in (
        0x63,
        0x67,
    )  # BRANCH, JALR (jal's target is not data-dependent)


@dataclass
class CommitDiff:
    """The outcome of a commit-stream comparison."""

    matched: int = 0
    #: None when the streams agree; otherwise a one-line summary of the first divergence.
    divergence: str | None = None
    #: True when the divergence follows (within TAINT_HORIZON) a branch on a tainted value.
    tainted: bool = False
    #: pc of the most recent branch on a tainted value, if any.
    tainted_branch_pc: int | None = None
    context: list = field(default_factory=list)
    resyncs: int = 0
    volatile_reads: int = 0
    #: Vector register writes compared (Spike-logged registers checked against Naiad's).
    vector_writes: int = 0

    @property
    def ok(self):
        return self.divergence is None

    def report(self):
        lines = [
            f"commit streams: {self.matched} instructions matched, {self.resyncs} HTIF spin resyncs, "
            f"{self.volatile_reads} volatile reads"
            + (f", {self.vector_writes} vector writes" if self.vector_writes else "")
        ]
        if self.divergence is None:
            lines.append("commit streams agree")
            return "\n".join(lines)
        lines.append(
            ("tainted " if self.tainted else "") + "divergence: " + self.divergence
        )
        if self.tainted:
            lines.append(
                f"  (shortly after a branch at {self.tainted_branch_pc:#x} on a value derived from a "
                f"counter/MMIO/HTIF read, so probably not a core bug)"
            )
        if self.context:
            lines.append("preceding (matched) instructions:")
            lines += [
                f"  {_fmt(pc, instr, rd, value)}"
                for pc, instr, rd, value in self.context
            ]
        return "\n".join(lines)


def _fmt(pc, instr, rd, value):
    text = f"pc {pc:#018x} instr {instr:#010x}"
    return text + (f" x{rd} = {value:#018x}" if rd >= 0 else "")


def compare_commits(
    naiad_commits, spike_commits, *, entry, volatile_ranges=(), naiad_vregs=None
):
    """Compare Naiad ``(pc, instr, rd, value)`` tuples with :class:`SpikeCommit` s.

    Both streams are aligned on ``entry`` (the first record at that pc). ``volatile_ranges``
    is a list of ``(low, high)`` half-open address ranges whose loads are not compared.
    ``naiad_vregs``, parallel to ``naiad_commits`` (:attr:`framework.naiad.difftest.NaiadRun.vregs`), turns on
    the vector comparison: every vector register Spike logs as written is compared
    with Naiad's register file after the same instruction.
    """
    diff = CommitDiff()
    index = [-1]  # index into naiad_commits of the record last taken

    def _naiad():
        for k, c in enumerate(naiad_commits):
            index[0] = k
            yield c

    naiad = _naiad()
    spike = iter(spike_commits)
    n = next((c for c in naiad if c[0] == entry), None)

    def vreg_mismatch(k, s):
        if naiad_vregs is None or not s.vregs or k >= len(naiad_vregs):
            return None
        vrf = naiad_vregs[k]
        if vrf is None:
            return f"v{s.vregs[0][0]}: spike logs a vector write, naiad's instruction is not a vector one"
        vlenb = len(vrf) // 32
        for reg, value in s.vregs:
            got = int.from_bytes(vrf[reg * vlenb : (reg + 1) * vlenb], "little")
            if got != value:
                return f"v{reg}: naiad {got:#0{2 * vlenb + 2}x}, spike {value:#0{2 * vlenb + 2}x}"
        return None

    s = next((c for c in spike if c.pc == entry), None)
    if n is None or s is None:
        side = "naiad" if n is None else "spike"
        diff.divergence = f"{side} never reached the entry point {entry:#x}"
        return diff

    def volatile_addr(addr):
        return addr is not None and any(lo <= addr < hi for lo, hi in volatile_ranges)

    tainted_regs = set()
    tainted_mem = set()
    recent = deque(maxlen=_RESYNC_WINDOW)
    context = deque(maxlen=_CONTEXT)
    last_tainted_branch = None  # (index, pc)

    def stop(message):
        diff.divergence = message
        if (
            last_tainted_branch is not None
            and diff.matched - last_tainted_branch[0] <= TAINT_HORIZON
        ):
            diff.tainted = True
            diff.tainted_branch_pc = last_tainted_branch[1]

    while n is not None:
        if s is None:
            stop(f"spike's stream ended; naiad went on to {_fmt(*n)}")
            break
        pc, instr, rd, value = n
        if (s.pc, s.instr) != (pc, instr):
            # An HTIF spin loop: Spike repeats pcs it just executed; skip them.
            resynced = None
            if s.pc in recent:
                skipped = 0
                t = s
                while t is not None and t.pc in recent and skipped < _RESYNC_MAX_SKIP:
                    if (t.pc, t.instr) == (pc, instr):
                        resynced = t
                        break
                    t = next(spike, None)
                    skipped += 1
                if (
                    resynced is None
                    and t is not None
                    and (t.pc, t.instr) == (pc, instr)
                ):
                    resynced = t
            if resynced is None:
                stop(
                    f"control flow: naiad {_fmt(*n)}, spike {_fmt(s.pc, s.instr, s.rd, s.value)}"
                )
                break
            s = resynced
            diff.resyncs += 1

        # The analyses below read 32-bit fields; a compressed instruction is
        # compared as its 16-bit parcel (both streams log that) but analysed as
        # its expansion.
        word = instr if instr & 3 == 3 else (expand_rvc(instr) or 0)
        sources = _sources(word)
        taint = any(r in tainted_regs for r in sources)
        if s.mem_addr is not None and not s.mem_store:
            if volatile_addr(s.mem_addr) or s.mem_addr in tainted_mem:
                taint = True
                diff.volatile_reads += volatile_addr(s.mem_addr)
        if _is_counter_read(word):
            taint = True
            diff.volatile_reads += 1
        if s.mem_store and s.mem_addr is not None:
            data_reg = (word >> 20) & 0x1F
            if word & 0x7F == 0x23 and data_reg in tainted_regs:
                tainted_mem.add(s.mem_addr)
            else:
                tainted_mem.discard(s.mem_addr)

        if rd != s.rd:
            stop(
                f"destination: naiad {_fmt(*n)}, spike {_fmt(s.pc, s.instr, s.rd, s.value)}"
            )
            break
        if rd >= 0:
            if taint:
                tainted_regs.add(rd)
            else:
                if value != s.value:
                    stop(
                        f"value: naiad {_fmt(*n)}, spike {_fmt(s.pc, s.instr, s.rd, s.value)}"
                    )
                    break
                tainted_regs.discard(rd)
        if s.vregs:
            bad = vreg_mismatch(index[0], s)
            if bad is not None:
                stop(f"vector value at pc {pc:#x} instr {instr:#010x}: {bad}")
                break
            if naiad_vregs is not None:
                diff.vector_writes += len(s.vregs)
        if taint and _is_branch(word):
            last_tainted_branch = (diff.matched, pc)
        diff.matched += 1
        recent.append(pc)
        context.append(n)
        n = next(naiad, None)
        s = next(spike, None)

    diff.context = list(context)
    return diff
