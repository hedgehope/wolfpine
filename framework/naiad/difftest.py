"""Differential testing: run one ELF on a Naiad design and on Spike, and compare.

    python -m framework.naiad.difftest <design.yaml> <prog.elf> [--commits] [--ignore REGEX]...

The default comparison is the program's console output (HTIF putchar /
``SYS_write``) and its exit code. Lines matching an ``--ignore`` regex are
dropped from both consoles first, for output that legitimately differs (a cycle
count, say: Spike's ``cycle`` counts its own instructions, including the ones it
spends spinning on ``tohost``).

``--commits`` also compares the two retirement streams instruction by
instruction: Spike's ``--log-commits`` against Naiad's ``InstrEvent`` s (pc,
instruction word, destination register and value). Both streams are aligned on
the ELF entry point, since the two boot ROMs differ (Spike's also loads ``a1``
with its DTB address and ``a0`` with ``mhartid``). The first divergence is
reported with its pc, instruction and rd values. Three things differ between
the two simulators without either being wrong, and the comparison allows for
them:

* **volatile reads**: counter CSRs (``cycle``/``time``/``instret`` and their
  M-mode and HPM forms) and loads from ``tohost``/``fromhost`` or a non-memory
  block (CLINT, perf counters). Their values are not compared; the destination
  register is marked *tainted*, and taint propagates through register sources
  and (by Spike's logged address) through memory, so values computed from them
  are not compared either.
* **HTIF spin loops**: Spike services ``tohost`` only every few thousand
  instructions, so a program waiting for it to clear loops many times on Spike
  and not at all on Naiad. On a mismatch, Spike records are skipped while they
  stay on pcs just executed (the loop body) until it reaches Naiad's pc again.
* **a branch on a tainted value**, which may legitimately go another way, and
  whose effects (a loop count, say) taint cannot follow. A divergence within
  :data:`TAINT_HORIZON` instructions of such a branch is reported as *tainted*:
  likely a consequence of the volatile read, not a core bug, but still a
  divergence (the comparison stops there).

Spike is found as ``$NAIAD_SPIKE``, then ``spike`` on ``PATH``, then
``~/.local/bin/spike`` (where ``tools/naiad/build_spike.sh`` installs it).
This module imports nothing heavy at load time, so it is cheap to import from
tools and tests that only need :func:`find_spike`.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

#: Spike's ``--isa`` misses one thing Naiad always has: Naiad's RV64 CSR file
#: implements Zicntr whenever Zicsr is present, and without ``zicntr`` Spike
#: traps on ``rdcycle`` (and, with no trap handler, hangs).
_IMPLIED_WITH_ZICSR = ("zicntr",)

#: Counter CSRs: cycle/time/instret (+ hpmcounter3..31), mcycle/minstret (+ mhpmcounter3..31).
_COUNTER_CSRS = frozenset(range(0xC00, 0xC20)) | frozenset(range(0xB00, 0xB20))

#: Blocks whose contents are program memory: loads from anything else are volatile.
_MEMORY_BLOCK_TYPES = frozenset({"dram", "scratchpad"})

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


# -- Naiad ------------------------------------------------------------------


@dataclass
class NaiadRun:
    outcome: RunOutcome
    soc: object
    #: ``(pc, instr, rd, value)`` per retired instruction when traced, else None.
    commits: list | None = None


def run_naiad(design, elf, *, max_cycles=None, trace=False):
    """Run ``elf`` on a Naiad design (a path or a loaded ``Design``)."""
    from framework.naiad.builder import build
    from framework.naiad.design import load_design
    from framework.naiad.run import DEFAULT_MAX_CYCLES
    from framework.trace import EventCategory, get_bus

    if isinstance(design, (str, Path)):
        design = load_design(design)
    soc = build(design)
    if soc.htif is not None:
        soc.htif.console_stream = None
    soc.load(elf)

    commits = None
    if not trace:
        result = soc.run(max_cycles or DEFAULT_MAX_CYCLES)
    else:
        commits = []

        def on_instr(event):
            if not event.stalled:
                commits.append(
                    (
                        event.pc,
                        event.instruction,
                        event.reg_write_idx,
                        event.reg_write_value,
                    )
                )

        for core in soc.cores:
            # Cores only publish InstrEvents once they have a unit id.
            core.unit_id = ("naiad", design.name, core.core_id, f"hart{core.core_id}")
        bus = get_bus()
        saved_enabled = bus.enabled
        saved_categories = {cat: bus._per_category[cat] for cat in EventCategory}
        for cat in EventCategory:
            bus.set_category_enabled(cat, cat == EventCategory.INSTR)
        bus.subscribe(EventCategory.INSTR, on_instr)
        bus.enabled = True
        try:
            result = soc.run(max_cycles or DEFAULT_MAX_CYCLES)
        finally:
            _unsubscribe(bus, EventCategory.INSTR, on_instr)
            bus.enabled = saved_enabled
            for cat, on in saved_categories.items():
                bus.set_category_enabled(cat, on)
    detail = (
        f"{result.cycles} cycles, {result.counters.get('instret', '?')} instructions"
    )
    return NaiadRun(RunOutcome(result.console, result.exit_code, detail), soc, commits)


def _unsubscribe(bus, cat, callback):
    # The bus has no unsubscribe; drop exactly the callback this module added.
    with bus._lock:
        subs = bus._subscribers[cat]
        if callback in subs:
            subs.remove(callback)


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
        yield SpikeCommit(
            int(m.group(1), 16),
            int(m.group(2), 16),
            rd,
            value,
            addr,
            bool(mem and mem.group(2)),
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

    @property
    def ok(self):
        return self.divergence is None

    def report(self):
        lines = [
            f"commit streams: {self.matched} instructions matched, {self.resyncs} HTIF spin resyncs, "
            f"{self.volatile_reads} volatile reads"
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


def compare_commits(naiad_commits, spike_commits, *, entry, volatile_ranges=()):
    """Compare Naiad ``(pc, instr, rd, value)`` tuples with :class:`SpikeCommit` s.

    Both streams are aligned on ``entry`` (the first record at that pc). ``volatile_ranges``
    is a list of ``(low, high)`` half-open address ranges whose loads are not compared.
    """
    diff = CommitDiff()
    naiad = iter(naiad_commits)
    spike = iter(spike_commits)
    n = next((c for c in naiad if c[0] == entry), None)
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

        sources = _sources(instr)
        taint = any(r in tainted_regs for r in sources)
        if s.mem_addr is not None and not s.mem_store:
            if volatile_addr(s.mem_addr) or s.mem_addr in tainted_mem:
                taint = True
                diff.volatile_reads += volatile_addr(s.mem_addr)
        if _is_counter_read(instr):
            taint = True
            diff.volatile_reads += 1
        if s.mem_store and s.mem_addr is not None:
            data_reg = (instr >> 20) & 0x1F
            if instr & 0x7F == 0x23 and data_reg in tainted_regs:
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
        if taint and _is_branch(instr):
            last_tainted_branch = (diff.matched, pc)
        diff.matched += 1
        recent.append(pc)
        context.append(n)
        n = next(naiad, None)
        s = next(spike, None)

    diff.context = list(context)
    return diff


def volatile_ranges(soc):
    """Address ranges whose loads :func:`compare_commits` should not compare, for a built SoC."""
    ranges = []
    for block in soc.blocks.values():
        if block.mapped and block.spec.type not in _MEMORY_BLOCK_TYPES:
            ranges.append((block.base, block.base + block.size))
    image = soc.image
    for name in ("tohost", "fromhost"):
        if image is not None and name in image.symbols:
            ranges.append((image.symbols[name], image.symbols[name] + 8))
    return ranges


# -- the whole thing --------------------------------------------------------


@dataclass
class DiffResult:
    naiad: RunOutcome
    spike: RunOutcome
    problems: list
    commits: CommitDiff | None = None

    @property
    def ok(self):
        return not self.problems and (self.commits is None or self.commits.ok)

    def report(self):
        lines = [
            f"naiad: exit {self.naiad.exit_code} ({self.naiad.detail})",
            f"spike: exit {self.spike.exit_code}",
        ]
        if self.problems:
            lines += self.problems
        else:
            lines.append("console and exit code agree")
        if self.commits is not None:
            lines.append(self.commits.report())
        lines.append("PASS" if self.ok else "FAIL")
        return "\n".join(lines)


def difftest(
    design,
    elf,
    *,
    ignore=(),
    commits=False,
    max_cycles=None,
    timeout=DEFAULT_TIMEOUT,
    spike=None,
    isa=None,
    log_dir=None,
):
    """Run ``elf`` on ``design`` and on Spike; return a :class:`DiffResult`."""
    from framework.naiad.design import load_design

    if isinstance(design, (str, Path)):
        design = load_design(design)
    isa = isa or spike_isa(design.cores.march)
    naiad = run_naiad(design, elf, max_cycles=max_cycles, trace=commits)

    with tempfile.TemporaryDirectory(prefix="naiad-difftest-") as tmp:
        log = None
        if commits:
            log = Path(log_dir or tmp) / f"{Path(elf).stem}.spike.commits"
            log.parent.mkdir(parents=True, exist_ok=True)
        spike_out = run_spike(elf, isa, spike=spike, timeout=timeout, commit_log=log)
        problems = compare_console(naiad.outcome, spike_out, ignore)
        commit_diff = None
        if commits:
            with open(log, errors="replace") as f:
                commit_diff = compare_commits(
                    naiad.commits,
                    parse_spike_commits(f),
                    entry=naiad.soc.image.entry,
                    volatile_ranges=volatile_ranges(naiad.soc),
                )
    return DiffResult(naiad.outcome, spike_out, problems, commit_diff)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m framework.naiad.difftest",
        description="Run an ELF on a Naiad design and on Spike, and compare console, exit code and (optionally) commits.",
    )
    parser.add_argument("design", help="Naiad design YAML")
    parser.add_argument("elf", help="program ELF (built against the design)")
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="REGEX",
        help="drop console lines matching REGEX before comparing (repeatable)",
    )
    parser.add_argument(
        "--commits",
        action="store_true",
        help="also compare the per-instruction commit streams",
    )
    parser.add_argument("--max-cycles", type=lambda s: int(s, 0), default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Spike wall-clock limit (s)",
    )
    parser.add_argument(
        "--spike", default=None, help="Spike binary (default: see module docs)"
    )
    parser.add_argument(
        "--isa",
        default=None,
        help="override Spike's --isa (default: from the design's march)",
    )
    parser.add_argument("--log-dir", default=None, help="keep Spike's commit log here")
    args = parser.parse_args(argv)

    try:
        result = difftest(
            args.design,
            args.elf,
            ignore=args.ignore,
            commits=args.commits,
            max_cycles=args.max_cycles,
            timeout=args.timeout,
            spike=args.spike,
            isa=args.isa,
            log_dir=args.log_dir,
        )
    except Exception as e:  # noqa: BLE001 - a CLI: report any failure as an error exit
        print(f"difftest: error: {e}", file=sys.stderr)
        return 2
    print(result.report())
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
