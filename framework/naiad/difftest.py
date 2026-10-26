"""Differential testing: run one ELF on a Naiad design and on Spike, and compare.

    python -m framework.naiad.difftest <design.yaml> <prog.elf> [--commits] [--ignore REGEX]...

The default comparison is the program's console output (HTIF putchar /
``SYS_write``) and its exit code. Lines matching an ``--ignore`` regex are
dropped from both consoles first, for output that legitimately differs (a cycle
count, say: Spike's ``cycle`` counts its own instructions, including the ones it
spends spinning on ``tohost``).

``--commits`` also compares the two retirement streams instruction by
instruction: Spike's ``--log-commits`` against Naiad's ``InstrEvent`` s (pc,
instruction word, destination register and value; on a core with a vector unit,
also every vector register Spike logs as written, against Naiad's register file
after that instruction). Both streams are aligned on
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

A multi-hart design runs Spike with ``-p<count>`` (same CLINT at
``0x2000000``, same hart numbering) and compares console and exit code only;
``--commits`` is single-hart.

Spike is found as ``$NAIAD_SPIKE``, then ``spike`` on ``PATH``, then
``~/.local/bin/spike`` (where ``tools/naiad/build_spike.sh`` installs it).
The Spike side and the comparisons live in :mod:`framework.pe.rv.difftest`
(re-exported here); this module adds the Naiad run and the CLI.
"""

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from framework.pe.rv.difftest import (  # noqa: F401  (re-exported)
    DEFAULT_TIMEOUT,
    TAINT_HORIZON,
    CommitDiff,
    DifftestError,
    RunOutcome,
    SpikeCommit,
    compare_commits,
    compare_console,
    filter_console,
    find_spike,
    is_vector_instr,
    parse_spike_commits,
    run_spike,
    spike_isa,
)

#: Blocks whose contents are program memory: loads from anything else are volatile.
_MEMORY_BLOCK_TYPES = frozenset({"dram", "scratchpad"})


# -- Naiad ------------------------------------------------------------------


@dataclass
class NaiadRun:
    outcome: RunOutcome
    soc: object
    #: ``(pc, instr, rd, value)`` per retired instruction when traced, else None.
    commits: list | None = None
    #: Parallel to ``commits`` on a core with a vector unit: the register file
    #: (``bytes``, 32 * VLENB) after each vector instruction, None after the rest.
    vregs: list | None = None


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
        vector = getattr(soc.cores[0], "vector", None)
        vregs = [] if vector is not None else None

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
                if vregs is not None:
                    vregs.append(
                        vector.vrf.tobytes()
                        if is_vector_instr(event.instruction)
                        else None
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
    run = NaiadRun(RunOutcome(result.console, result.exit_code, detail), soc, commits)
    if trace:
        run.vregs = vregs
    return run


def _unsubscribe(bus, cat, callback):
    # The bus has no unsubscribe; drop exactly the callback this module added.
    with bus._lock:
        subs = bus._subscribers[cat]
        if callback in subs:
            subs.remove(callback)


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
    harts = design.cores.count
    if commits and harts > 1:
        raise DifftestError(
            f"--commits compares one retirement stream; design '{design.name}' has {harts} harts "
            f"(compare the console and exit code only)"
        )
    naiad = run_naiad(design, elf, max_cycles=max_cycles, trace=commits)

    with tempfile.TemporaryDirectory(prefix="naiad-difftest-") as tmp:
        log = None
        if commits:
            log = Path(log_dir or tmp) / f"{Path(elf).stem}.spike.commits"
            log.parent.mkdir(parents=True, exist_ok=True)
        spike_out = run_spike(
            elf,
            isa,
            spike=spike,
            timeout=timeout,
            commit_log=log,
            extra_args=[f"-p{harts}"] if harts > 1 else (),
        )
        problems = compare_console(naiad.outcome, spike_out, ignore)
        commit_diff = None
        if commits:
            with open(log, errors="replace") as f:
                commit_diff = compare_commits(
                    naiad.commits,
                    parse_spike_commits(f),
                    entry=naiad.soc.image.entry,
                    volatile_ranges=volatile_ranges(naiad.soc),
                    naiad_vregs=naiad.vregs,
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
