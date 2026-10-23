import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import build
from framework.naiad.design import load_design
from framework.naiad.gen import generate
from framework.naiad.run import main
from framework.util.elf_testing import write_elf

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS = REPO_ROOT / "kernels" / "naiad"
BASE_DESIGN = DESIGNS_DIR / "naiad-base.yaml"
NAIAD_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")
TOHOST = 0x80001000


def _lui(rd, imm20):
    return (imm20 << 12) | (rd << 7) | 0x37


def _addi(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (rd << 7) | 0x13


def _sw(rs2, rs1, imm):
    return (
        ((imm >> 5) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (2 << 12)
        | ((imm & 0x1F) << 7)
        | 0x23
    )


T0, T1, A0 = 5, 6, 10
SPIN = 0x0000006F  # j .


def _soc(tmp_path, program, symbols=None):
    """naiad-base with ``program`` (a list of RV32 words) at the DRAM base."""
    code = b"".join(w.to_bytes(4, "little") for w in program)
    symbols = (
        {"tohost": TOHOST, "fromhost": TOHOST + 0x40} if symbols is None else symbols
    )
    elf = write_elf(
        tmp_path / "prog.elf",
        [(0x80000000, code, len(code)), (TOHOST, bytes(0x48), 0x48)],
        0x80000000,
        symbols,
    )
    soc = build(load_design(BASE_DESIGN))
    soc.htif.console_stream = None
    soc.load(elf)
    return soc


def test_bootrom_reaches_entry_and_htif_prints_and_exits(tmp_path):
    soc = _soc(
        tmp_path,
        [
            _lui(T0, TOHOST >> 12),
            _lui(T1, 0x01010),  # device 1, cmd 1 in the high word
            _addi(A0, 0, ord("N")),
            _sw(T1, T0, 4),  # high word first ...
            _sw(A0, T0, 0),  # ... then the low-word doorbell
            _addi(A0, 0, (7 << 1) | 1),
            _sw(A0, T0, 0),  # exit(7)
            SPIN,
        ],
    )
    result = soc.run(1000)
    assert result.exit_code == 7
    assert result.console == b"N"
    # Three boot ROM instructions, then seven of the program's; the exit store halts the core.
    assert result.cycles == 10
    assert result.counters["instret"] == 10


def test_htif_handshake_and_syscall_proxy(tmp_path):
    soc = _soc(tmp_path, [SPIN])
    soc.reset()
    mem = soc.memory
    mem.write(TOHOST + 4, (0x01010000).to_bytes(4, "little"))
    mem.write(TOHOST, ord("x").to_bytes(4, "little"))
    assert soc.htif.console == bytearray(b"x")
    assert mem.read(TOHOST, 8) == bytes(8)  # tohost cleared for the next command
    # A console write is not acked through fromhost (Spike's bcd device does not either).
    assert mem.read(TOHOST + 0x40, 8) == bytes(8)

    # SYS_write(1, buf, 3) through a magic_mem block, then SYS_exit(5).
    magic, buf = 0x80002000, 0x80002100
    mem.write(buf, b"abc")
    for i, word in enumerate([64, 1, buf, 3]):
        mem.write(magic + 8 * i, word.to_bytes(8, "little"))
    mem.write(TOHOST, magic.to_bytes(8, "little"))
    assert soc.htif.console == bytearray(b"xabc")
    assert int.from_bytes(mem.read(magic, 8), "little") == 3
    for i, word in enumerate([93, 5]):
        mem.write(magic + 8 * i, word.to_bytes(8, "little"))
    mem.write(TOHOST, magic.to_bytes(8, "little"))
    assert soc.htif.exit_code == 5
    assert soc.halted


def test_elf_without_tohost_is_refused(tmp_path):
    with pytest.raises(Exception, match="tohost"):
        _soc(tmp_path, [SPIN], symbols={})


def test_clint_and_perfcounters(tmp_path):
    soc = _soc(tmp_path, [SPIN])
    soc.run(50)
    clint, perf = 0x02000000, 0x02010000
    assert int.from_bytes(soc.memory.read(clint + 0xBFF8, 8), "little") == 49
    soc.memory.write(clint + 0xBFF8, (1000).to_bytes(8, "little"))
    assert int.from_bytes(soc.memory.read(clint + 0xBFF8, 4), "little") == 1000
    soc.memory.write(clint + 0x4000, (0x1234).to_bytes(4, "little"))
    assert int.from_bytes(soc.memory.read(clint + 0x4000, 8), "little") == 0x1234
    assert int.from_bytes(soc.memory.read(perf + 0, 8), "little") == 49  # cycle
    assert int.from_bytes(soc.memory.read(perf + 8, 4), "little") == 50  # instret


def test_generated_files_carry_banner_and_addresses(tmp_path):
    paths = generate(load_design(BASE_DESIGN), tmp_path)
    assert sorted(p.name for p in paths) == [
        "crt0.S",
        "linker.ld",
        "naiad.h",
        "naiad.mk",
    ]
    header = (tmp_path / "naiad.h").read_text()
    assert "DO NOT EDIT" in header
    assert "#define NAIAD_DRAM_BASE 0x80000000UL" in header
    assert "#define NAIAD_XLEN 32" in header
    assert "ORIGIN = 0x80000000" in (tmp_path / "linker.ld").read_text()
    assert "sw    zero, 0(t0)" in (tmp_path / "crt0.S").read_text()
    rv64 = tmp_path / "rv64"
    generate(load_design(DESIGNS_DIR / "naiad-base-rv64.yaml"), rv64)
    crt0 = (rv64 / "crt0.S").read_text()
    assert "sd    zero, 0(t0)" in crt0
    assert "addi  t0, t0, 8" in crt0
    assert "NAIAD_MABI := lp64" in (rv64 / "naiad.mk").read_text()
    # The ld/bnez wait before the exit store: Spike consumes tohost lazily.
    assert "bnez  t1, 4b" in crt0
    assert "csrs  mstatus" not in crt0


def test_fp_design_crt0_turns_the_fpu_on(tmp_path):
    generate(load_design(DESIGNS_DIR / "naiad-fp-rv64.yaml"), tmp_path)
    crt0 = (tmp_path / "crt0.S").read_text()
    # mstatus.FS = Initial before main, or FP code traps on Spike.
    assert "li    t0, 1 << 13\n  csrs  mstatus, t0" in crt0
    assert crt0.index("csrs  mstatus") < crt0.index("call  main")


# -- end to end: clang -> ELF -> Naiad --------------------------------------

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)


def _make(build_dir, **variables):
    args = [
        "make",
        "-C",
        str(KERNELS),
        f"BUILD={build_dir}",
        f"PYTHON={sys.executable}",
        f"NAIAD_CC={NAIAD_CC}",
    ]
    args += [f"{k}={v}" for k, v in variables.items()]
    subprocess.run(args, check=True, capture_output=True, text=True)


@needs_clang
def test_hello_builds_runs_and_passes(tmp_path, capsys):
    _make(tmp_path)
    assert main([str(BASE_DESIGN), str(tmp_path / "hello.elf")]) == 0
    out, err = capsys.readouterr()
    assert "hello from naiad: design naiad-base, rv32" in out
    assert "primes below 1000: 168 (expect 168) ok" in out
    assert out.endswith("PASS\n")
    assert "exit code 0" in err


@needs_clang
def test_exit_code_is_the_programs(tmp_path, capsys):
    prog = tmp_path / "ret3"
    prog.mkdir()
    (prog / "ret3.c").write_text(
        '#include "naiad_platform.h"\nint main(void) { naiad_puts("bye\\n"); return 3; }\n'
    )
    _make(tmp_path / "build", PROG="ret3", PROG_DIR=prog)
    assert (
        main(
            [
                str(BASE_DESIGN),
                str(tmp_path / "build" / "ret3.elf"),
                "--dump",
                "tohost:8",
            ]
        )
        == 3
    )
    out, err = capsys.readouterr()
    assert out == "bye\n"
    assert "exit code 3" in err


def test_cli_reports_timeout(tmp_path, capsys):
    code = b"".join(w.to_bytes(4, "little") for w in [SPIN])
    elf = write_elf(
        tmp_path / "spin.elf",
        [(0x80000000, code, 4)],
        0x80000000,
        {"tohost": TOHOST, "fromhost": TOHOST + 0x40},
    )
    assert main([str(BASE_DESIGN), str(elf), "--max-cycles", "100"]) == 124
    assert "no exit after 100 cycles" in capsys.readouterr().err


def test_cli_rejects_elf_of_the_wrong_class(tmp_path, capsys):
    elf = write_elf(tmp_path / "x.elf", [(0x80000000, b"\0" * 4, 4)], 0x80000000)
    assert main([str(DESIGNS_DIR / "naiad-base-rv64.yaml"), str(elf)]) == 2
    assert "ELF32" in capsys.readouterr().err


@needs_clang
def test_hello_builds_runs_and_passes_on_rv64(tmp_path, capsys):
    _make(tmp_path, DESIGN=DESIGNS_DIR / "naiad-base-rv64.yaml")
    assert (
        main([str(DESIGNS_DIR / "naiad-base-rv64.yaml"), str(tmp_path / "hello.elf")])
        == 0
    )
    out, err = capsys.readouterr()
    assert "hello from naiad: design naiad-base-rv64, rv64" in out
    assert "signed rem: -6 (expect -6) ok" in out
    assert out.endswith("PASS\n")


def test_console_stream_receives_output(tmp_path):
    soc = _soc(tmp_path, [SPIN])
    stream = io.StringIO()
    soc.htif.console_stream = stream
    soc.reset()
    soc.memory.write(TOHOST + 4, (0x01010000).to_bytes(4, "little"))
    soc.memory.write(TOHOST, ord("\n").to_bytes(4, "little"))
    assert stream.getvalue() == "\n"
