"""The ``wolfpine`` CLI's environment arithmetic, against a fake tt-metal checkout."""

import json
import os
import subprocess
import sys

import pytest

from framework import cli
from framework.naiad import DESIGNS_DIR
from framework.sim import simulator_dir
from framework.util.elf_testing import write_elf


def _fake_tt_metal(tmp_path):
    root = tmp_path / "tt-metal"
    (root / "tt_metal").mkdir(parents=True)
    (root / "build_Release" / "lib").mkdir(parents=True)
    (root / "build_Release" / "lib" / cli.LIB_MARKER).touch()
    return root


def _eval(lines, env):
    """Apply ``shell_lines`` output to ``env`` the way a POSIX shell would."""
    script = "\n".join([*lines, "env -0"])
    out = subprocess.run(
        ["sh", "-c", script], env=env, capture_output=True, check=True
    ).stdout
    return dict(item.split("=", 1) for item in out.decode().split("\0") if item)


def test_sim_target_points_tt_metal_at_the_packaged_simulator(tmp_path):
    tt_metal = _fake_tt_metal(tmp_path)
    env = cli.target_env(
        "blackhole", {"LD_LIBRARY_PATH": "/opt/x"}, tt_metal=str(tt_metal)
    )
    assert env["TT_METAL_SIMULATOR"] == str(simulator_dir("blackhole"))
    assert (simulator_dir("blackhole") / "run.sh").exists()
    assert env["TT_METAL_SLOW_DISPATCH_MODE"] == "1"
    assert env["TT_METAL_RUNTIME_ROOT"] == env["TT_METAL_HOME"] == str(tt_metal)
    assert env["LD_LIBRARY_PATH"] == f"{tt_metal}/build_Release/lib:/opt/x"
    assert env["WOLFPINE_PYTHON"] == sys.executable


def test_hw_target_drops_the_simulator(tmp_path):
    tt_metal = _fake_tt_metal(tmp_path)
    env = cli.target_env(
        "hw", {"TT_METAL_SIMULATOR": "/x", "TT_METAL_RUNTIME_ROOT": str(tt_metal)}
    )
    assert "TT_METAL_SIMULATOR" not in env
    assert env["LD_LIBRARY_PATH"] == f"{tt_metal}/build_Release/lib"


def test_env_switches_without_stacking_and_off_restores(tmp_path, monkeypatch):
    tt_metal = _fake_tt_metal(tmp_path)
    base = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "LD_LIBRARY_PATH": "/opt/x",
        "TT_METAL_SIMULATOR": "/old",
    }
    env = dict(base)
    for target in ("wormhole", "blackhole", "hw", "wormhole"):
        monkeypatch.setattr(os, "environ", env)
        lines = []
        monkeypatch.setattr("builtins.print", lambda line, **_: lines.append(line))
        cli.main(["env", "--tt-metal", str(tt_metal), target])
        env = _eval(lines, env)
    assert env["LD_LIBRARY_PATH"] == f"{tt_metal}/build_Release/lib:/opt/x"
    assert env["TT_METAL_SIMULATOR"] == str(simulator_dir("wormhole"))

    monkeypatch.setattr(os, "environ", env)
    lines = []
    monkeypatch.setattr("builtins.print", lambda line, **_: lines.append(line))
    cli.main(["env", "off"])
    restored = _eval(lines, env)
    for var in (*cli.MANAGED, cli.SAVED):
        assert restored.get(var) == base.get(var), var


def test_missing_checkout_is_a_usage_error(tmp_path):
    with pytest.raises(cli.UsageError, match="no tt-metal checkout"):
        cli.resolve_tt_metal(
            None, {"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path)}
        )
    with pytest.raises(cli.UsageError, match="not a tt-metal checkout"):
        cli.tt_metal_lib_dir(tmp_path)


# -- run naiad -----------------------------------------------------------------

_TOHOST = 0x80001000


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


def _exit_elf(tmp_path, code):
    """An RV32 ELF for naiad-base that prints 'N' and exits with ``code`` through HTIF."""
    t0, t1, a0 = 5, 6, 10
    words = [
        _lui(t0, _TOHOST >> 12),
        _lui(t1, 0x01010),  # device 1, cmd 1 in the high word
        _addi(a0, 0, ord("N")),
        _sw(t1, t0, 4),  # high word first ...
        _sw(a0, t0, 0),  # ... then the low-word doorbell
        _addi(a0, 0, (code << 1) | 1),
        _sw(a0, t0, 0),  # exit(code)
        0x0000006F,  # j .
    ]
    text = b"".join(w.to_bytes(4, "little") for w in words)
    return write_elf(
        tmp_path / "exit.elf",
        [(0x80000000, text, len(text)), (_TOHOST, bytes(0x48), 0x48)],
        0x80000000,
        {"tohost": _TOHOST, "fromhost": _TOHOST + 0x40},
    )


def test_run_naiad_runs_an_elf_without_tt_metal(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("TT_METAL_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("TT_METAL_HOME", raising=False)
    elf = _exit_elf(tmp_path, 5)
    assert cli.main(["run", "naiad", "--design", "naiad-base", str(elf)]) == 5
    out, err = capsys.readouterr()
    assert out == "N"
    assert "exit code 5" in err


def test_run_naiad_applies_set_overrides(tmp_path, capsys):
    elf = _exit_elf(tmp_path, 0)
    js = tmp_path / "c.json"
    args = [
        "run",
        "naiad",
        "--design",
        "naiad-base",
        "--set",
        "name=renamed",
        "--counters-json",
        str(js),
        str(elf),
    ]
    assert cli.main(args) == 0
    assert json.loads(js.read_text())["design"] == "renamed"


def test_run_naiad_takes_a_design_path_and_lists_designs(tmp_path, capsys):
    elf = _exit_elf(tmp_path, 0)
    assert (
        cli.main(
            ["run", "naiad", "--design", str(DESIGNS_DIR / "naiad-base.yaml"), str(elf)]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["run", "naiad", "--list-designs"]) == 0
    names = capsys.readouterr().out.split()
    assert "naiad-base" in names
    assert "naiad-dwarfs" in names


def test_run_naiad_reports_bad_designs_and_overrides(tmp_path, capsys):
    elf = _exit_elf(tmp_path, 0)
    assert cli.main(["run", "naiad", "--design", "no-such-design", str(elf)]) == 2
    assert "no design 'no-such-design'" in capsys.readouterr().err
    assert (
        cli.main(
            ["run", "naiad", "--design", "naiad-base", "--set", "nonsense", str(elf)]
        )
        == 2
    )
    assert "expected key.path=value" in capsys.readouterr().err
    assert (
        cli.main(
            [
                "run",
                "naiad",
                "--design",
                "naiad-base",
                "--set",
                "blocks.nope.latency=3",
                str(elf),
            ]
        )
        == 2
    )
    assert "no list element named 'nope'" in capsys.readouterr().err


def test_run_naiad_needs_an_elf(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "naiad"])
    assert "the ELF to run is required" in capsys.readouterr().err
