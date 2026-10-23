"""Run an ELF on a Naiad design: the CLI behind ``python -m framework.naiad``.

The program's console (HTIF putchar / ``SYS_write``) streams to stdout as it
runs; the report (cycles, counters, ``--dump`` hexdumps) goes to stderr, so
stdout is exactly what the program printed. The process exits with the
program's exit code, or 124 if ``--max-cycles`` ran out first.

The report lists every counter-hub event (``framework.naiad.counters``);
``--counters-json PATH`` also writes them, with the run's outcome, as JSON for
the experiment harness, and ``--count-all`` turns on the on-demand families
(instruction classes, per-block bus bytes) the design does not ask for.
"""

import argparse
import json
import sys

from framework.naiad.builder import NaiadBuildError, NaiadRunError, build
from framework.naiad.design import DesignError, load_design
from framework.naiad.ip.htif import HTIFError
from framework.util.elf_loader import ELFLoadError

EXIT_TIMEOUT = 124
EXIT_ERROR = 2
DEFAULT_MAX_CYCLES = 50_000_000


def run_program(
    design_path,
    elf_path,
    max_cycles=DEFAULT_MAX_CYCLES,
    console_stream=None,
    count_all=False,
):
    """Build ``design_path``, load ``elf_path`` and run it; returns ``(soc, RunResult)``."""
    soc = build(load_design(design_path), count_all=count_all)
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m framework.naiad", description="Run an ELF on a Naiad design."
    )
    parser.add_argument("design", help="Naiad design YAML")
    parser.add_argument(
        "elf",
        help="program to load (an ELF linked with the design's generated linker.ld/crt0.S)",
    )
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
    args = parser.parse_args(argv)

    report = sys.stderr
    try:
        soc, result = run_program(
            args.design,
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
                    counters_json(soc, result, args.design, args.elf), f, indent=2
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
