"""Run an ELF on a Naiad design: the CLI behind ``python -m framework.naiad``.

The program's console (HTIF putchar / ``SYS_write``) streams to stdout as it
runs; the report (cycles, counters, ``--dump`` hexdumps) goes to stderr, so
stdout is exactly what the program printed. The process exits with the
program's exit code, or 124 if ``--max-cycles`` ran out first.

The report lists every counter-hub event (``framework.ip.counters``);
``--counters-json PATH`` also writes them, with the run's outcome, as JSON for
the experiment harness, and ``--count-all`` turns on the on-demand families
(instruction classes, per-block bus bytes) the design does not ask for.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import NaiadBuildError, NaiadRunError, build
from framework.naiad.design import (
    DesignError,
    _UniqueKeyLoader,
    load_design,
    parse_design,
)
from framework.naiad.ip.htif import HTIFError
from framework.util.elf_loader import ELFLoadError

EXIT_TIMEOUT = 124
EXIT_ERROR = 2
DEFAULT_MAX_CYCLES = 50_000_000
DEFAULT_DESIGN = "naiad-dwarfs"


def design_names():
    """The packaged designs, by name (``framework/naiad/designs/<name>.yaml``)."""
    return sorted(p.stem for p in DESIGNS_DIR.glob("*.yaml"))


def resolve_design(spec):
    """A design given as a packaged name or a YAML path, as a path."""
    path = Path(spec)
    if path.suffix in (".yaml", ".yml") or path.exists():
        return path
    packaged = DESIGNS_DIR / f"{spec}.yaml"
    if packaged.exists():
        return packaged
    raise DesignError(
        f"no design {spec!r}: not a file, and not one of {', '.join(design_names())}"
    )


def apply_set(raw, text):
    """Apply one ``key.path=value`` override to the raw design mapping (value parsed as YAML).

    A path segment names a mapping key, or a list element by its ``name`` (``blocks.l1.latency=8``).
    Overrides that move a block change the memory map an ELF was linked against: keep to timing.
    """
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise DesignError(f"--set {text!r}: expected key.path=value")
    value = yaml.safe_load(value)
    node = raw
    parts = key.split(".")
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if isinstance(node, list):
            named = [e for e in node if isinstance(e, dict) and e.get("name") == part]
            if not named:
                raise DesignError(f"--set {key}: no list element named {part!r}")
            if last:
                raise DesignError(f"--set {key}: cannot replace a whole list element")
            node = named[0]
        elif isinstance(node, dict):
            if last:
                node[part] = value
            else:
                node = node.setdefault(part, {})
        else:
            raise DesignError(
                f"--set {key}: {'.'.join(parts[:i])} is not a mapping or list"
            )
    return raw


def load_design_spec(spec, sets=()):
    """Load a design by name or path, with ``--set`` overrides applied."""
    path = resolve_design(spec)
    if not sets:
        return load_design(path), path
    try:
        with open(path) as f:
            raw = yaml.load(f, Loader=_UniqueKeyLoader)
    except OSError as e:
        raise DesignError(f"cannot read {path}: {e.strerror}") from None
    for text in sets:
        apply_set(raw, text)
    return parse_design(raw, path), path


def run_program(
    design_path,
    elf_path,
    max_cycles=DEFAULT_MAX_CYCLES,
    console_stream=None,
    count_all=False,
):
    """Build ``design_path`` (a path or a loaded design), load ``elf_path`` and run it; returns ``(soc, RunResult)``."""
    design = (
        load_design(design_path)
        if isinstance(design_path, (str, Path))
        else design_path
    )
    soc = build(design, count_all=count_all)
    if soc.htif is not None:
        soc.htif.console_stream = console_stream
    soc.load(elf_path)
    return soc, soc.run(max_cycles)


def _parse_dump(soc, text):
    """``addr:len``, where ``addr`` is an integer or an ELF symbol name."""
    where, sep, length = text.rpartition(":")
    if not sep:
        raise ValueError(f"--dump '{text}': expected addr:len")
    if where in soc.image.symbols:
        addr = soc.image.symbols[where]
    else:
        addr = int(where, 0)
    return addr, int(length, 0)


def counters_json(soc, result, design_path, elf_path):
    """The run's outcome and every counter, as the dict ``--counters-json`` writes."""
    return {
        "design": soc.design.name,
        "design_path": str(design_path),
        "elf": str(elf_path),
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "cycles": result.cycles,
        "counters": result.counters,
    }


def hexdump(memory, addr, length, out):
    for row in range(addr, addr + length, 16):
        n = min(16, addr + length - row)
        data = memory.read(row, n)
        hexes = " ".join(f"{b:02x}" for b in data)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in data)
        print(f"  {row:08x}  {hexes:<47}  {text}", file=out)


def _add_run_options(parser):
    parser.add_argument(
        "--max-cycles", type=lambda s: int(s, 0), default=DEFAULT_MAX_CYCLES
    )
    parser.add_argument(
        "--dump",
        action="append",
        default=[],
        metavar="ADDR:LEN",
        help="hexdump memory after the run (ADDR may be a symbol)",
    )
    parser.add_argument(
        "--counters-json",
        metavar="PATH",
        help="write the run's outcome and every counter event to PATH as JSON",
    )
    parser.add_argument(
        "--count-all",
        action="store_true",
        help="also count the on-demand events (instruction classes, bus bytes per block)",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m framework.naiad", description="Run an ELF on a Naiad design."
    )
    parser.add_argument("design", help="Naiad design YAML")
    parser.add_argument(
        "elf",
        help="program to load (an ELF linked with the design's generated linker.ld/crt0.S)",
    )
    _add_run_options(parser)
    args = parser.parse_args(argv)
    return _run(args, args.design)


def target_main(argv=None, prog="wolfpine run naiad"):
    """``wolfpine run naiad [--design D] [--set k=v ...] [options] prog.elf``."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run an ELF on a Naiad design. Build it against the design's generated files: "
        "python -m framework.naiad.gen <design.yaml> -o <dir> writes naiad.h, linker.ld, crt0.S and naiad.mk.",
    )
    parser.add_argument(
        "--design",
        default=DEFAULT_DESIGN,
        help=f"a packaged design's name or a design YAML path (default {DEFAULT_DESIGN}; --list-designs)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a design field, e.g. timing.scalar.issue_width=2 (value parsed as YAML; repeatable)",
    )
    parser.add_argument(
        "--list-designs", action="store_true", help="list the packaged designs and exit"
    )
    parser.add_argument(
        "elf",
        nargs="?",
        help="program to load (an ELF linked with the design's generated linker.ld/crt0.S)",
    )
    _add_run_options(parser)
    args = parser.parse_args(argv)
    if args.list_designs:
        for name in design_names():
            print(name)
        return 0
    if args.elf is None:
        parser.error("the ELF to run is required")
    try:
        design, path = load_design_spec(args.design, args.set)
    except (DesignError, OSError) as e:
        print(f"naiad: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    return _run(args, design, path)


def _run(args, design, design_path=None):
    design_path = design if design_path is None else design_path
    report = sys.stderr
    try:
        soc, result = run_program(
            design,
            args.elf,
            args.max_cycles,
            console_stream=sys.stdout,
            count_all=args.count_all,
        )
    except (
        DesignError,
        NaiadBuildError,
        ELFLoadError,
        HTIFError,
        NaiadRunError,
        OSError,
    ) as e:
        sys.stdout.flush()
        print(f"naiad: error: {e}", file=report)
        return EXIT_ERROR
    sys.stdout.flush()

    if result.console and not result.console.endswith(b"\n"):
        print(file=report)
    if result.timed_out:
        print(
            f"naiad: no exit after {result.cycles} cycles (--max-cycles)", file=report
        )
    else:
        print(
            f"naiad: exit code {result.exit_code} after {result.cycles} cycles",
            file=report,
        )
    for name, value in result.counters.items():
        print(f"naiad:   {name} = {value}", file=report)
    if args.counters_json:
        try:
            with open(args.counters_json, "w") as f:
                json.dump(
                    counters_json(soc, result, design_path, args.elf), f, indent=2
                )
                f.write("\n")
        except OSError as e:
            print(f"naiad: error: --counters-json: {e}", file=report)
            return EXIT_ERROR
    for text in args.dump:
        try:
            addr, length = _parse_dump(soc, text)
        except ValueError as e:
            print(f"naiad: {e}", file=report)
            return EXIT_ERROR
        print(f"naiad: dump {hex(addr)}+{hex(length)}", file=report)
        hexdump(soc.memory, addr, length, report)

    if result.timed_out:
        return EXIT_TIMEOUT
    # A process exit status is 8 bits; never let a non-zero code wrap to 0.
    return (result.exit_code & 0xFF) or (1 if result.exit_code else 0)
