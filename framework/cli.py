"""``wolfpine`` — point tt-metal at the simulator (or at hardware) without hand-exporting.

Running a tt-metal program against Wolfpine needs a handful of environment
variables, one of which (``TT_METAL_SIMULATOR``) is the path of a directory
inside this package — not something a ``pip install`` user can be expected to
know. This command works them out (see ``EXAMPLES``).

``run`` scopes the variables to one child process, so nothing leaks into the
shell. ``env`` prints POSIX ``export``/``unset`` lines; the first one records the
original values of every variable it touches (in ``_WOLFPINE_SAVED``), later
ones compute from those originals, and ``env off`` puts them back — so
switching between targets never accumulates ``LD_LIBRARY_PATH`` entries.
"""

import argparse
import json
import os
import pathlib
import shlex
import sys

import yaml

from framework.sim import ARCHES, simulator_dir

EXAMPLES = """\
examples:
  wolfpine config --tt-metal /path/to/tt-metal   remember the checkout (once)
  wolfpine run wormhole ./build/one              one program, simulated
  wolfpine run hw ./build/one                    the same, on real hardware
  wolfpine run blackhole                         a subshell with the simulator on
  eval "$(wolfpine env blackhole)"               turn it on in this shell
  eval "$(wolfpine env off)"                     restore what was there before
  wolfpine doctor                                show what would be used

options such as --tt-metal go before the target: everything after it is the command.
"""

HW = "hw"
TARGETS = (*ARCHES, HW)

#: Every variable ``env``/``run`` may change, and so every one ``env off`` restores.
MANAGED = (
    "TT_METAL_RUNTIME_ROOT",
    "TT_METAL_HOME",
    "LD_LIBRARY_PATH",
    "TT_METAL_SIMULATOR",
    "TT_METAL_SLOW_DISPATCH_MODE",
    "WOLFPINE_PYTHON",
)
SAVED = "_WOLFPINE_SAVED"

#: Where a tt-metal build puts its shared libraries, in preference order. The
#: plain ``build/`` tree is what ``build_metal.sh`` produces; the suffixed ones
#: are CMake-preset builds.
LIB_DIRS = (
    "build/lib",
    "build_Release/lib",
    "build_RelWithDebInfo/lib",
    "build_Debug/lib",
)
LIB_MARKER = "libtt_metal.so"


class UsageError(Exception):
    """A problem the user can fix; printed without a traceback."""


def config_path(env=None):
    env = os.environ if env is None else env
    base = env.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return pathlib.Path(base) / "wolfpine" / "config.yaml"


def load_config(env=None):
    path = config_path(env)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def original_env(env):
    """``env`` with every managed variable put back to its pre-``wolfpine env`` value."""
    raw = env.get(SAVED)
    if not raw:
        return dict(env)
    out = dict(env)
    for var, value in json.loads(raw).items():
        if value is None:
            out.pop(var, None)
        else:
            out[var] = value
    out.pop(SAVED, None)
    return out


def resolve_tt_metal(flag, env):
    """``(path, where it came from)`` for the tt-metal checkout to use.

    The current environment wins over the config file, so a shell that already
    points at a checkout (a venv, or an earlier ``wolfpine env``) keeps it.
    """
    for candidate, source in (
        (flag, "--tt-metal"),
        (env.get("TT_METAL_RUNTIME_ROOT"), "$TT_METAL_RUNTIME_ROOT"),
        (env.get("TT_METAL_HOME"), "$TT_METAL_HOME"),
        (load_config(env).get("tt_metal"), str(config_path(env))),
    ):
        if candidate:
            return pathlib.Path(candidate).expanduser().resolve(), source
    raise UsageError(
        "no tt-metal checkout found. Pass --tt-metal /path/to/tt-metal, export "
        "TT_METAL_RUNTIME_ROOT, or save one with `wolfpine config --tt-metal /path/to/tt-metal`."
    )


def tt_metal_lib_dir(root):
    """The directory of ``root``'s built shared libraries."""
    if not (root / "tt_metal").is_dir():
        raise UsageError(
            f"{root} is not a tt-metal checkout (no tt_metal/ directory in it)"
        )
    for rel in LIB_DIRS:
        if (root / rel / LIB_MARKER).exists():
            return root / rel
    raise UsageError(
        f"{root} has no built {LIB_MARKER} in any of {', '.join(LIB_DIRS)}; build tt-metal first"
    )


def target_env(target, env, tt_metal=None):
    """The environment a program should see for ``target``, derived from ``env``.

    ``env`` may carry an earlier ``wolfpine env``; it is computed from the
    originals underneath, so targets replace each other rather than stack.
    """
    root, _ = resolve_tt_metal(tt_metal, env)
    lib = str(tt_metal_lib_dir(root))
    out = original_env(env)

    out["TT_METAL_RUNTIME_ROOT"] = str(root)
    out["TT_METAL_HOME"] = str(root)
    paths = [
        p for p in out.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p and p != lib
    ]
    out["LD_LIBRARY_PATH"] = os.pathsep.join([lib, *paths])

    if target == HW:
        out.pop("TT_METAL_SIMULATOR", None)
        out.pop("WOLFPINE_PYTHON", None)
    else:
        out["TT_METAL_SIMULATOR"] = str(simulator_dir(target))
        # The direct launch flow is the only one the simulator models; this
        # makes EnqueueProgram fall back to it.
        out["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
        # run.sh would otherwise use whatever `python3` is on PATH, which need
        # not be the interpreter Wolfpine and its dependencies are installed in.
        out["WOLFPINE_PYTHON"] = sys.executable
    return out


def shell_lines(before, after):
    """POSIX shell lines turning the managed variables of ``before`` into ``after``."""
    lines = []
    for var in (*MANAGED, SAVED):
        if var in after and after[var] != before.get(var):
            lines.append(f"export {var}={shlex.quote(after[var])}")
        elif var not in after and var in before:
            lines.append(f"unset {var}")
    return lines


def cmd_env(args):
    env = dict(os.environ)
    if args.target == "off":
        if SAVED not in env:
            print("# wolfpine env: nothing to restore", file=sys.stderr)
            return 0
        after = original_env(env)
    else:
        after = target_env(args.target, env, args.tt_metal)
        saved = env.get(SAVED) or json.dumps({var: env.get(var) for var in MANAGED})
        after[SAVED] = saved
    for line in shell_lines(env, after):
        print(line)
    return 0


def cmd_run(args):
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    env = target_env(args.target, dict(os.environ), args.tt_metal)
    env.pop(SAVED, None)
    if not command:
        shell = os.environ.get("SHELL") or "/bin/sh"
        print(
            f"wolfpine: {shell} with target {args.target}; exit to leave",
            file=sys.stderr,
        )
        command = [shell]
    try:
        os.execvpe(command[0], command, env)
    except OSError as e:
        raise UsageError(f"cannot run {command[0]!r}: {e.strerror}") from None


def cmd_config(args):
    path = config_path()
    config = load_config()
    if args.tt_metal:
        root = pathlib.Path(args.tt_metal).expanduser().resolve()
        tt_metal_lib_dir(root)  # refuse to save a path that would not work
        config["tt_metal"] = str(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config))
        print(f"saved tt_metal: {root} to {path}")
    elif config:
        print(f"# {path}")
        print(yaml.safe_dump(config), end="")
    else:
        print(
            f"no config yet ({path}); set one with `wolfpine config --tt-metal /path/to/tt-metal`"
        )
    return 0


def cmd_doctor(args):
    ok = True

    def report(label, value, good=True):
        nonlocal ok
        ok &= good
        print(f"{'  ' if good else '!!'} {label:<22} {value}")

    report("python", sys.executable)
    for arch in ARCHES:
        run_sh = simulator_dir(arch) / "run.sh"
        executable = os.access(run_sh, os.X_OK)
        report(
            f"simulator ({arch})",
            simulator_dir(arch) if executable else f"{run_sh} is not executable",
            executable,
        )

    env = os.environ
    try:
        root, source = resolve_tt_metal(args.tt_metal, env)
        report("tt-metal", f"{root}  (from {source})")
        report("tt-metal libs", tt_metal_lib_dir(root))
    except UsageError as e:
        report("tt-metal", e, False)

    current = env.get("TT_METAL_SIMULATOR")
    state = (
        f"simulator: {current}" if current else "hardware (TT_METAL_SIMULATOR unset)"
    )
    report("this shell targets", state)
    if current and not pathlib.Path(current, "run.sh").exists():
        report("", f"{current}/run.sh does not exist", False)
    if current and env.get("TT_METAL_SLOW_DISPATCH_MODE") != "1":
        report(
            "",
            "TT_METAL_SLOW_DISPATCH_MODE is not 1; EnqueueProgram will not run on the simulator",
            False,
        )
    return 0 if ok else 1


def build_parser():
    ap = argparse.ArgumentParser(
        prog="wolfpine",
        description="Point tt-metal at the Wolfpine simulator, or at hardware, without hand-exporting variables.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    tt_metal = argparse.ArgumentParser(add_help=False)
    tt_metal.add_argument(
        "--tt-metal",
        metavar="PATH",
        help="tt-metal checkout (default: env, then config)",
    )

    run = sub.add_parser(
        "run",
        parents=[tt_metal],
        help="run one command (default: a subshell) with tt-metal pointed at a target",
    )
    run.add_argument("target", choices=TARGETS)
    run.add_argument(
        "command", nargs=argparse.REMAINDER, help="the program and its arguments"
    )
    run.set_defaults(func=cmd_run)

    env = sub.add_parser(
        "env",
        parents=[tt_metal],
        help='print shell exports for a target; use as eval "$(wolfpine env <target>)"',
    )
    env.add_argument(
        "target",
        choices=(*TARGETS, "off"),
        help="'off' restores the variables to before",
    )
    env.set_defaults(func=cmd_env)

    config = sub.add_parser(
        "config", help="show, or with --tt-metal save, the default tt-metal checkout"
    )
    config.add_argument(
        "--tt-metal", metavar="PATH", help="tt-metal checkout to remember"
    )
    config.set_defaults(func=cmd_config)

    doctor = sub.add_parser(
        "doctor", parents=[tt_metal], help="show what would be used, and what is wrong"
    )
    doctor.set_defaults(func=cmd_doctor)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except UsageError as e:
        print(f"wolfpine: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
