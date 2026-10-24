"""Run riscv-vector-tests (chipsalliance) on Naiad's RVV unit, or on Spike.

    tools/naiad/vectortest.py [--filter REGEX] [-j N] [--sim naiad|spike] [-v]

The tests are built by ``tools/naiad/build_vector_tests.sh`` (VLEN 256, XLEN 64,
ELEN 64) into ``$NAIAD_VECTOR_TESTS`` (default
``~/projects/riscv/riscv-vector-tests/out/naiad-v256x64``). Each one is a
self-checking ELF: Spike computed every expected value when the test was built,
and the ``tools/naiad/vector_env`` environment exits with code 0 on success or
with the number of the first failing ``TEST_CASE``. So no reference run is
needed here; ``--sim spike`` re-runs the suite on Spike to check the harness.

A failing test is reduced to its first differing check: the instruction under
test (with its VL / LMUL / SEW / mask configuration, from the generator's
comments in the stage-2 source), the stored result's register and 64-bit word,
the expected value, and the value Naiad produced (``t0``, which the check
loaded, is still live at the exit).

Each source file holds one instruction's tests, split into ``<insn>-<k>`` parts;
the table has one row per instruction. Statuses:

``PASS``     every part passed
``FAIL``     a check differed (the first failing part is shown)
``UNIMPL``   Naiad raised on an instruction it does not decode (not yet implemented)
``ERROR``    any other simulator exception
``TIMEOUT``  no exit within ``--max-cycles``

Exit status: 0 if every selected test passed, 1 otherwise, 2 on a set-up error.
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TESTS = Path(
    os.environ.get(
        "NAIAD_VECTOR_TESTS",
        Path.home()
        / "projects"
        / "riscv"
        / "riscv-vector-tests"
        / "out"
        / "naiad-v256x64",
    )
)
DEFAULT_DESIGN = REPO_ROOT / "framework" / "naiad" / "designs" / "naiad-rvv.yaml"
#: Spike's ISA for the built suite (build_vector_tests.sh: MARCH + zicntr + zvl256b).
SPIKE_ISA = (
    "rv64imafdv_zicsr_zifencei_zfhmin_zfbfmin_zvfh_zvfbfmin_zvfbfwma_zicntr_zvl256b"
)
VLEN = 256
DEFAULT_MAX_CYCLES = 20_000_000

STATUSES = ("PASS", "FAIL", "UNIMPL", "ERROR", "TIMEOUT")
_UNIMPL = re.compile(
    r"unknown instruction|not implemented|NotImplemented|illegal instruction|unsupported",
    re.I,
)


def instruction_of(test):
    """``vadd_vv-3`` -> ``vadd.vv``."""
    return test.rsplit("-", 1)[0].replace("_", ".")


# -- failure analysis ---------------------------------------------------------

_CASE = re.compile(r"^\s*TEST_CASE\((\d+),\s*(\w+),\s*(0x[0-9a-fA-F]+|-?\d+)")


def explain_failure(source, number):
    """What check ``number`` of a stage-2 source tests: a dict, or None if not found."""
    try:
        lines = Path(source).read_text(errors="replace").splitlines()
    except OSError:
        return None
    at = next(
        (
            i
            for i, line in enumerate(lines)
            if (m := _CASE.match(line)) and int(m.group(1)) == number
        ),
        None,
    )
    if at is None:
        return None
    m = _CASE.match(lines[at])
    info = {"line": at + 1, "expected": int(m.group(3), 0), "check_reg": m.group(2)}
    # Position of this check among the run of TEST_CASEs it belongs to.
    first = at
    while first > 0 and _CASE.match(lines[first - 1]):
        first -= 1
    info["word"] = at - first
    if m.group(2) != "t0" or "vxsat" in lines[at]:
        info["what"] = "vxsat" if "vxsat" in lines[at] else lines[at].strip()
    begin = end = None
    for i in range(first - 1, -1, -1):
        if end is None and "TEST END" in lines[i]:
            end = i
        if "TEST BEGIN" in lines[i]:
            begin = i
            break
    if begin is not None and end is not None:
        info["code"] = [line.strip() for line in lines[begin + 1 : end] if line.strip()]
    for i in range(first - 1, -1, -1):
        if lines[i].startswith("# Generating tests for"):
            info["config"] = lines[i][len("# Generating tests for ") :].strip()
            break
    # The stored group: "vse<eew>.v vN, (a0)" between TEST END and the checks.
    if end is not None:
        for line in lines[end:first]:
            st = re.match(r"\s*vs(?:e\d+|m)\.v\s+v(\d+)", line)
            if st:
                info["stored"] = int(st.group(1))
    return info


def describe(info, actual=None):
    if not info:
        return ""
    parts = []
    if "config" in info:
        parts.append(info["config"])
    if "code" in info:
        parts.append(" ; ".join(info["code"][-3:]))
    if "what" in info:
        parts.append(f"check {info['what']}")
    elif "stored" in info:
        words = VLEN // 64
        reg = info["stored"] + info["word"] // words
        parts.append(f"v{reg} dword {info['word'] % words}")
    text = f"expected {info['expected'] & (2**64 - 1):#x}"
    if actual is not None:
        text += f", got {actual:#x}"
    parts.append(text)
    return " | ".join(parts)


# -- one test -----------------------------------------------------------------

_DESIGN_CACHE = {}


def _design(path):
    if path not in _DESIGN_CACHE:
        from framework.naiad.design import load_design

        _DESIGN_CACHE[path] = load_design(path)
    return _DESIGN_CACHE[path]


def run_on_naiad(elf, design_path, max_cycles):
    """``(status, exit_code, detail, t0)`` for one ELF on Naiad."""
    from framework.naiad.builder import build

    soc = build(_design(design_path))
    if soc.htif is not None:
        soc.htif.console_stream = None
    soc.load(elf)
    try:
        result = soc.run(max_cycles)
    except Exception as e:  # noqa: BLE001 - a simulator crash is the test's result
        text = " ".join(str(e).split())
        status = "UNIMPL" if _UNIMPL.search(text) else "ERROR"
        return status, None, text[:300], None
    if result.exit_code is None:
        return "TIMEOUT", None, f"no exit in {result.cycles} cycles", None
    t0 = None
    try:
        t0 = soc.cores[0].register_file[5].read_uint()
    except Exception:  # noqa: BLE001 - only a nicety for the report
        pass
    return (
        ("PASS" if result.exit_code == 0 else "FAIL"),
        result.exit_code,
        f"{result.cycles} cycles",
        t0,
    )


def run_on_spike(elf, spike, timeout):
    try:
        proc = subprocess.run(
            [spike, f"--isa={SPIKE_ISA}", str(elf)],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None, f"spike timed out after {timeout}s", None
    code = proc.returncode
    # The exit status keeps 8 bits; Spike's "*** FAILED *** (tohost = N)" has all of them.
    m = re.search(rb"tohost = (\d+)", proc.stderr)
    if code and m:
        code = int(m.group(1))
    return ("PASS" if code == 0 else "FAIL"), code, "", None


def run_one(args):
    test, tests_dir, sim, design_path, spike, max_cycles, timeout = args
    elf = Path(tests_dir) / "bin" / test
    if sim == "spike":
        status, code, detail, actual = run_on_spike(elf, spike, timeout)
    else:
        status, code, detail, actual = run_on_naiad(elf, design_path, max_cycles)
    if status == "FAIL" and code:
        info = explain_failure(Path(tests_dir) / "stage2" / f"{test}.S", code)
        detail = f"check {code}: " + describe(info, actual)
    return {
        "test": test,
        "insn": instruction_of(test),
        "status": status,
        "detail": detail,
    }


# -- the table ------------------------------------------------------------------


def summarise(results):
    """Per instruction: ``(status, parts passed, parts, first failing result)``."""
    rank = {s: i for i, s in enumerate(STATUSES)}
    rows = {}
    for r in sorted(results, key=lambda r: (r["insn"], _part(r["test"]))):
        row = rows.setdefault(
            r["insn"], {"status": "PASS", "passed": 0, "parts": 0, "first": None}
        )
        row["parts"] += 1
        if r["status"] == "PASS":
            row["passed"] += 1
        else:
            if row["first"] is None:
                row["first"] = r
            # The worst status wins, FAIL being worse than UNIMPL only if seen first.
            if row["status"] == "PASS" or rank[r["status"]] < rank[row["status"]]:
                row["status"] = r["status"]
    return rows


def _part(test):
    try:
        return int(test.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def print_table(rows, verbose=False, out=sys.stdout):
    width = max([len(k) for k in rows] + [11])
    print(
        f"{'instruction':{width}}  {'status':7} {'parts':>9}  first failure", file=out
    )
    for insn, row in rows.items():
        if not verbose and row["status"] == "PASS":
            continue
        first = row["first"]
        note = f"{first['test']}: {first['detail']}" if first else ""
        print(
            f"{insn:{width}}  {row['status']:7} {row['passed']:>4}/{row['parts']:<4}  {note}",
            file=out,
        )
    totals = {s: sum(1 for r in rows.values() if r["status"] == s) for s in STATUSES}
    print(
        f"\n{len(rows)} instructions: "
        + ", ".join(f"{totals[s]} {s}" for s in STATUSES),
        file=out,
    )
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tests", type=Path, default=DEFAULT_TESTS, help="build_vector_tests.sh output"
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="only tests whose name (e.g. vadd_vv-0) matches this regex",
    )
    parser.add_argument("--sim", choices=("naiad", "spike"), default="naiad")
    parser.add_argument(
        "--design",
        default=str(DEFAULT_DESIGN),
        help="Naiad design (default naiad-rvv.yaml)",
    )
    parser.add_argument("--spike", default=None)
    parser.add_argument(
        "--max-cycles", type=lambda s: int(s, 0), default=DEFAULT_MAX_CYCLES
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="list passing instructions too"
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="also write per-test results here"
    )
    args = parser.parse_args(argv)

    bin_dir = args.tests / "bin"
    if not bin_dir.is_dir():
        print(
            f"vectortest: no {bin_dir}; run tools/naiad/build_vector_tests.sh",
            file=sys.stderr,
        )
        return 2
    spike = None
    if args.sim == "spike":
        from framework.naiad.difftest import find_spike

        spike = args.spike or find_spike()
        if spike is None:
            print("vectortest: spike not found", file=sys.stderr)
            return 2
    pattern = re.compile(args.filter) if args.filter else None
    tests = sorted(
        p.name for p in bin_dir.iterdir() if not pattern or pattern.search(p.name)
    )
    if not tests:
        print("vectortest: no tests selected", file=sys.stderr)
        return 2
    jobs = [
        (
            t,
            str(args.tests),
            args.sim,
            args.design,
            spike,
            args.max_cycles,
            args.timeout,
        )
        for t in tests
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run_one, jobs, chunksize=4))
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["test", "insn", "status", "detail"])
            w.writeheader()
            w.writerows(results)
    print(f"riscv-vector-tests on {args.sim} ({len(tests)} tests)")
    totals = print_table(summarise(results), args.verbose)
    return 0 if all(totals[s] == 0 for s in STATUSES if s != "PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
