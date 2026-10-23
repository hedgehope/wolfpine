"""Run riscv-arch-test (ACT4) RV64 suites on Naiad, with Spike as the reference.

    tools/naiad/archtest.py [--suite DIR] [--ext I,M,...] [--filter REGEX] [-j N]

For every test ``.S`` under ``<suite>/tests/rv64i/<Ext>/``, this script:

1. compiles it in **signature mode** (no ``RVTEST_SELFCHECK``: each
   ``RVTEST_SIGUPD`` stores its result instead of comparing it) with clang,
   against a small model of its own (``rvmodel_macros.h``, ``rvtest_config.h``
   and a ``link.ld`` at 0x8000_0000, written into the work directory);
2. runs the ELF on Spike with ``+signature=`` and on Naiad, dumping Naiad's
   ``[begin_signature, end_signature)`` in Spike's format;
3. compares the two signatures and the exit codes.

ACT4 normally takes its expected values from the Sail model and needs UDB
(Ruby) to generate the configuration header; neither is needed here, because
Spike computes the reference signature from the same ELF. The suite's ``env/``
headers are used unchanged, except ``sail_macros.h`` (which overrides the
model's HTIF macros for Sail's benefit) is replaced by an empty one.

The model is machine-mode only with no trap handler (``STANDARD_SM_SUPPORTED``
is left undefined), since Naiad models no traps; tests that need traps or
privilege modes are not selected. The design is ``naiad-fp-rv64`` plus Zba,
Zbb and Zfbfmin, so F and D (FLEN = 64) run too. A test runs when its
``REQUIRED_EXTENSIONS`` are all in :data:`SUPPORTED`.

Exit status: 0 if every selected test passed, 1 otherwise.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.naiad.difftest import (  # noqa: E402
    find_spike,
    run_naiad,
    run_spike,
    spike_isa,
)

DEFAULT_SUITE = Path(
    os.environ.get(
        "NAIAD_ARCH_TEST", Path.home() / "projects" / "riscv" / "riscv-arch-test"
    )
)
DEFAULT_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")
DEFAULT_WORK = REPO_ROOT / "kernels" / "naiad" / "build" / "archtest"
BASE_DESIGN = REPO_ROOT / "framework" / "naiad" / "designs" / "naiad-fp-rv64.yaml"

#: The suites run by default: every RV64 extension Naiad implements.
DEFAULT_EXTENSIONS = (
    "I",
    "M",
    "Zaamo",
    "Zalrsc",
    "Zba",
    "Zbb",
    "Zicsr",
    "Zicntr",
    "Zmmul",
    "F",
    "D",
    "Zfbfmin",
)

#: ACT extension names (``REQUIRED_EXTENSIONS``) the Naiad core implements.
SUPPORTED = frozenset(
    {
        "I",
        "M",
        "Zmmul",
        "Zaamo",
        "Zalrsc",
        "A",
        "Zba",
        "Zbb",
        "Zicsr",
        "Zicntr",
        "F",
        "D",
        "Zfbfmin",
    }
)

#: Naiad's ISA list for the archtest design (design extension names).
DESIGN_ISA = ["i", "m", "a", "f", "d", "zba", "zbb", "zfbfmin", "zicsr", "zicntr"]

SIGNATURE_GRANULARITY = 8

RVTEST_CONFIG_H = """\
// rvtest_config.h for Naiad (RV64, M-mode only, no traps). Written by tools/naiad/archtest.py.
// Stands in for the UDB-generated header ACT4 would produce.
#define UDB_MXLEN 64
#define UDB_MXLEN_64
#define I_SUPPORTED
#define M_SUPPORTED
#define ZMMUL_SUPPORTED
#define A_SUPPORTED
#define ZAAMO_SUPPORTED
#define ZALRSC_SUPPORTED
#define ZBA_SUPPORTED
#define ZBB_SUPPORTED
#define F_SUPPORTED
#define D_SUPPORTED
#define ZFBFMIN_SUPPORTED
#define ZICSR_SUPPORTED
#define ZICNTR_SUPPORTED
// time is a real CSR (CLINT mtime), so no emulation through RVMODEL_MTIME_ADDRESS.
#define UDB_TIME_CSR_IMPLEMENTED
#define UDB_NUM_PMP_ENTRIES 0
"""

# HTIF on the ELF's tohost/fromhost, which both Spike and Naiad speak. Every
# command waits for the previous one to be consumed (Spike polls tohost only
# every few thousand instructions) and is a single 64-bit store.
RVMODEL_MACROS_H = """\
// rvmodel_macros.h for Naiad (and Spike): HTIF console and exit. Written by tools/naiad/archtest.py.
#ifndef _RVMODEL_MACROS_H
#define _RVMODEL_MACROS_H

#define RVMODEL_DATA_SECTION \\
        .pushsection .tohost,"aw",@progbits;                \\
        .balign 64; .global tohost; tohost: .dword 0;        \\
        .balign 64; .global fromhost; fromhost: .dword 0;    \\
        .popsection

#define RVMODEL_HALT_WITH(_CODE) \\
  li x1, _CODE            ;\\
  la t0, tohost           ;\\
  1: ld t1, 0(t0)         ;\\
  bnez t1, 1b             ;\\
  sd x1, 0(t0)            ;\\
  2: j 2b                 ;

#define RVMODEL_HALT_PASS RVMODEL_HALT_WITH(1)
#define RVMODEL_HALT_FAIL RVMODEL_HALT_WITH(3)

// HTIF device 1 (console), cmd 1 (write): tohost = 1 << 56 | 1 << 48 | char.
#define RVMODEL_IO_WRITE_STR(_R1, _R2, _R3, _STR_PTR) \\
1:                                 ;\\
  lbu _R1, 0(_STR_PTR)             ;\\
  beqz _R1, 3f                     ;\\
  la _R2, tohost                   ;\\
2:                                 ;\\
  ld _R3, 0(_R2)                   ;\\
  bnez _R3, 2b                     ;\\
  li _R3, 0x0101                   ;\\
  slli _R3, _R3, 48                ;\\
  or _R1, _R1, _R3                 ;\\
  sd _R1, 0(_R2)                   ;\\
  addi _STR_PTR, _STR_PTR, 1       ;\\
  j 1b                             ;\\
3:

#define RVMODEL_ACCESS_FAULT_ADDRESS 0x00000000
#define RVMODEL_MTIME_ADDRESS  0x0200BFF8
#define RVMODEL_MTIMECMP_ADDRESS 0x02004000
#define RVMODEL_INTERRUPT_LATENCY 4096
#define RVMODEL_TIMER_INT_SOON_DELAY 1000
#define RVMODEL_MAX_CYCLES_PER_TIMER_TICK 5000

#endif // _RVMODEL_MACROS_H
"""

SAIL_MACROS_H = """\
// Replaced by tools/naiad/archtest.py: the signature ELF runs on Spike and Naiad,
// so it keeps the model's own macros (rvmodel_macros.h) rather than Sail's.
"""

LINK_LD = """\
/* Written by tools/naiad/archtest.py: ACT4's spike link.ld, sized to Naiad's 1 GiB DRAM. */
RAM_ORIGIN = 0x80000000;
RAM_LENGTH = 0x40000000;
TEST_BASE = RAM_ORIGIN;
NUM_HARTS = 1;
STACK_SIZE = 0x20000;

OUTPUT_ARCH( "riscv" )
ENTRY(rvtest_entry_point)

MEMORY
{
  ram (rwx) : ORIGIN = RAM_ORIGIN, LENGTH = RAM_LENGTH
}

PROVIDE(__stack_size = STACK_SIZE);
PROVIDE(__num_harts = NUM_HARTS);

SECTIONS
{
  .text.init   TEST_BASE : { *(.text.init) } > ram
  .text.rvtest . : { *(.text.rvtest) *(.text.rvtest.*) } > ram

  . = ALIGN(0x4000);
  .rodata . : { *(.rodata) *(.rodata.*) *(.srodata) *(.srodata.*) } > ram
  .data   . : { *(.data) *(.data.*) *(.sdata) *(.sdata.*) } > ram

  . = ALIGN(16);
  .bss . : {
    __bss_start = .;
    *(.sbss) *(.sbss.*) *(.bss) *(.bss.*) *(COMMON)
    . = ALIGN(16);
    __bss_end = .;
  } > ram

  . = ALIGN(16);
  __stack_bottom = .;
  . += __stack_size * __num_harts;
  __stack_top = .;

  . = ALIGN(0x1000);
  .text.rvmodel . : { *(.text.rvmodel) *(.text.rvmodel.*) *(.text) *(.text.*) } > ram

  . = ALIGN(0x1000);
  _end = .;
}
"""

_CONFIG_LINE = re.compile(r"^#\s*(REQUIRED_EXTENSIONS|MARCH):\s*(.*)$", re.M)


def test_config(path):
    """``(required_extensions, march)`` from a test's ``START_TEST_CONFIG`` block."""
    head = path.read_text(errors="replace")[:4096]
    found = dict(_CONFIG_LINE.findall(head))
    required = re.findall(r"'([^']+)'", found.get("REQUIRED_EXTENSIONS", ""))
    return required, found.get("MARCH", "").strip()


def prepare_work_dir(suite, work):
    """Write the model headers, link.ld, a copy of the suite's env/ and the design into ``work``."""
    env = work / "env"
    if env.exists():
        shutil.rmtree(env)
    shutil.copytree(suite / "tests" / "env", env)
    (env / "sail_macros.h").write_text(SAIL_MACROS_H)
    (env / "rvtest_config.h").write_text(RVTEST_CONFIG_H)
    (env / "rvmodel_macros.h").write_text(RVMODEL_MACROS_H)
    (work / "link.ld").write_text(LINK_LD)

    import yaml

    design = yaml.safe_load(BASE_DESIGN.read_text())
    design["name"] = "naiad-archtest-rv64"
    design["cores"]["isa"] = DESIGN_ISA
    path = work / "naiad-archtest-rv64.yaml"
    path.write_text(yaml.safe_dump(design, sort_keys=False))
    return path


def test_flen(march):
    """FLEN the test is built for, from its ``MARCH`` (as ACT4 derives it): D = 64, else 32."""
    parts = march.lower().split("_")
    singles = re.sub(r"^rv\d+", "", parts[0])
    if "q" in singles or "q" in parts[1:]:
        return 128
    if "d" in singles or "g" in singles or "d" in parts[1:]:
        return 64
    return 32


def compile_test(cc, work, source, march, out):
    cmd = [
        cc,
        "--target=riscv64-unknown-elf",
        f"-march={march}",
        "-mabi=lp64",
        "-mcmodel=medany",
        "-mno-relax",
        "-nostdlib",
        "-static",
        "-fuse-ld=lld",
        "-O0",
        f"-T{work / 'link.ld'}",
        f"-I{work / 'env'}",
        "-Wl,--defsym=__stack_size=128",
        f"-DTEST_FLEN={test_flen(march)}",
        f'-DTEST_FILE="{source.name}"',
        str(source),
        "-o",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.strip().splitlines()[-1]
            if proc.stderr.strip()
            else "compile failed"
        )


def format_signature(data, granularity=SIGNATURE_GRANULARITY):
    """``data`` as Spike's ``+signature`` file: one ``granularity``-byte word per line, MSB first."""
    lines = []
    for i in range(0, len(data), granularity):
        lines.append(data[i : i + granularity][::-1].hex())
    return "\n".join(lines) + "\n"


def run_one(args):
    """Compile and run one test on both simulators; returns a result dict (picklable)."""
    source, design_path, work, cc, spike, max_cycles, timeout = args
    source = Path(source)
    name = source.stem
    ext = source.parent.name
    out_dir = Path(work) / "out" / ext
    out_dir.mkdir(parents=True, exist_ok=True)
    elf = out_dir / f"{name}.elf"
    result = {"ext": ext, "name": name, "status": "PASS", "detail": ""}
    required, march = test_config(source)
    try:
        compile_test(cc, Path(work), source, march, elf)
    except RuntimeError as e:
        return {**result, "status": "BUILD", "detail": str(e)}

    from framework.naiad.design import load_design

    design = load_design(design_path)
    spike_sig = out_dir / f"{name}.spike.sig"
    spike_out = run_spike(
        elf,
        spike_isa(design.cores.march),
        spike=spike,
        timeout=timeout,
        signature=spike_sig,
        signature_granularity=SIGNATURE_GRANULARITY,
    )
    try:
        naiad = run_naiad(design, elf, max_cycles=max_cycles)
    except Exception as e:  # noqa: BLE001 - one test's crash is that test's result
        return {
            **result,
            "status": "ERROR",
            "detail": f"naiad: {str(e).splitlines()[0][:200]}",
        }
    soc = naiad.soc
    begin, end = soc.image.symbol("begin_signature"), soc.image.symbol("end_signature")
    naiad_sig_text = format_signature(soc.memory.read(begin, end - begin))
    (out_dir / f"{name}.naiad.sig").write_text(naiad_sig_text)

    if spike_out.exit_code is None or not spike_sig.exists():
        return {
            **result,
            "status": "REF",
            "detail": f"spike: {spike_out.detail.splitlines()[-1][:200]}",
        }
    if naiad.outcome.exit_code is None:
        return {**result, "status": "TIMEOUT", "detail": naiad.outcome.detail}
    spike_lines = spike_sig.read_text().split()
    naiad_lines = naiad_sig_text.split()
    if spike_lines != naiad_lines:
        bad = [i for i, (a, b) in enumerate(zip(naiad_lines, spike_lines)) if a != b]
        if len(naiad_lines) != len(spike_lines):
            detail = f"signature length: naiad {len(naiad_lines)} words, spike {len(spike_lines)}"
        else:
            i = bad[0]
            detail = (
                f"{len(bad)} signature words differ; first at {begin + i * SIGNATURE_GRANULARITY:#x}: "
                f"naiad {naiad_lines[i]}, spike {spike_lines[i]}"
            )
        return {**result, "status": "FAIL", "detail": detail}
    if naiad.outcome.exit_code != spike_out.exit_code:
        return {
            **result,
            "status": "FAIL",
            "detail": f"exit code: naiad {naiad.outcome.exit_code}, spike {spike_out.exit_code}",
        }
    return {**result, "detail": f"{len(naiad_lines)} words; {naiad.outcome.detail}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--suite",
        type=Path,
        default=DEFAULT_SUITE,
        help="riscv-arch-test checkout ($NAIAD_ARCH_TEST)",
    )
    parser.add_argument(
        "--ext",
        default=",".join(DEFAULT_EXTENSIONS),
        help="comma-separated suite directories",
    )
    parser.add_argument(
        "--filter", default=None, help="only tests whose name matches this regex"
    )
    parser.add_argument(
        "--work", type=Path, default=DEFAULT_WORK, help="build/output directory"
    )
    parser.add_argument("--cc", default=DEFAULT_CC, help="clang ($NAIAD_CC)")
    parser.add_argument("--spike", default=None)
    parser.add_argument("--max-cycles", type=lambda s: int(s, 0), default=20_000_000)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print every test, not only failures",
    )
    args = parser.parse_args(argv)

    spike = args.spike or find_spike()
    if spike is None:
        print("archtest: spike not found (tools/naiad/build_spike.sh)", file=sys.stderr)
        return 2
    if not Path(args.cc).is_file():
        print(f"archtest: compiler {args.cc} not found", file=sys.stderr)
        return 2
    rv64 = args.suite / "tests" / "rv64i"
    if not rv64.is_dir():
        print(
            f"archtest: no {rv64} (clone riscv-non-isa/riscv-arch-test, or pass --suite)",
            file=sys.stderr,
        )
        return 2

    args.work.mkdir(parents=True, exist_ok=True)
    design_path = prepare_work_dir(args.suite, args.work)
    pattern = re.compile(args.filter) if args.filter else None
    jobs, skipped = [], {}
    for ext in [e.strip() for e in args.ext.split(",") if e.strip()]:
        for source in sorted((rv64 / ext).glob("*.S")):
            if pattern and not pattern.search(source.stem):
                continue
            required, _ = test_config(source)
            missing = [r for r in required if r not in SUPPORTED]
            if missing:
                skipped.setdefault(ext, []).append(
                    f"{source.stem} (needs {', '.join(missing)})"
                )
                continue
            jobs.append(
                (
                    str(source),
                    str(design_path),
                    str(args.work),
                    args.cc,
                    spike,
                    args.max_cycles,
                    args.timeout,
                )
            )

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run_one, jobs))

    table = {}
    for r in results:
        row = table.setdefault(r["ext"], {})
        row[r["status"]] = row.get(r["status"], 0) + 1
        if args.verbose or r["status"] != "PASS":
            print(f"{r['status']:7} {r['ext']}/{r['name']}: {r['detail']}")
    for ext, names in skipped.items():
        table.setdefault(ext, {})["SKIP"] = len(names)
        if args.verbose:
            for n in names:
                print(f"SKIP    {ext}/{n}")

    statuses = ["PASS", "FAIL", "ERROR", "TIMEOUT", "BUILD", "REF", "SKIP"]
    print()
    print(f"{'suite':10}" + "".join(f"{s:>8}" for s in statuses))
    for ext, row in table.items():
        print(f"{ext:10}" + "".join(f"{row.get(s, 0):>8}" for s in statuses))
    total = {s: sum(row.get(s, 0) for row in table.values()) for s in statuses}
    print(f"{'total':10}" + "".join(f"{total[s]:>8}" for s in statuses))
    failed = sum(v for s, v in total.items() if s not in ("PASS", "SKIP"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
