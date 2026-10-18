"""The ``wolfpine`` CLI's environment arithmetic, against a fake tt-metal checkout."""

import os
import subprocess
import sys

import pytest

from framework import cli
from framework.sim import simulator_dir


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
