#!/usr/bin/env bash
# Launched by UMD's tt_SimulationDevice. NNG_SOCKET_ADDR is already in env.
# The arch is this script's directory name, so the Wormhole and Blackhole
# copies are identical; driver/<arch>/run.sh is a symlink to it.
#
# Opt-in env knobs (UMD inherits the parent env, so set these before running
# the tt-metal program):
#   WOLFPINE_PYTHON=<path>    interpreter to run the server with (default
#                             python3); `wolfpine run`/`wolfpine env` set it to
#                             the one Wolfpine is installed into
#   TT_SIM_RECORD=<path>      record every wire message to this file
#   TT_SIM_LOG_PROTOCOL=1     print every wire message to stderr
#   TT_SIM_CYCLES_PER_POLL=N  cycles to run after each message (default 100)
#   TT_SIM_MOCK_TENSIX=1      skip building the device; every core is NullCore
#   TT_SIM_TENSIX_COORDS=1-1,2-1  PIN the worker set to exactly these physical
#                                 coords. Unset, Wolfpine builds one worker up
#                                 front (1-1 WH, 1-2 BH) and materialises any
#                                 other worker the program launches on (or a
#                                 peer addresses) on demand.
#   TT_SIM_TENSIX_CORES=N         pin N workers at default coords (column-major
#                                 over tt-metal's compute grid) instead of
#                                 naming them. Mutually exclusive with _COORDS.
#   TT_SIM_RUN_TAG=<tag>      stamp <tag> into the server's command line, so the
#                             test script that set it can clean up exactly the
#                             servers it caused to start (driver/sim_procs.sh).
set -euo pipefail

# Resolve symlinks so driver/<arch>/run.sh lands here too.
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ARCH="$(basename "$HERE")"
# The directory holding `framework/` (repo root, or site-packages when
# installed) must be importable so `framework.sim.<arch>` resolves.
export PYTHONPATH="$HERE/../../..:${PYTHONPATH:-}"

extra=()
[ -n "${TT_SIM_RECORD:-}" ] && extra+=(--record "$TT_SIM_RECORD")
[ -n "${TT_SIM_LOG_PROTOCOL:-}" ] && extra+=(--log-protocol)
[ -n "${TT_SIM_CYCLES_PER_POLL:-}" ] && extra+=(--cycles-per-poll "$TT_SIM_CYCLES_PER_POLL")
[ -n "${TT_SIM_MOCK_TENSIX:-}" ] && extra+=(--mock-tensix)
[ -n "${TT_SIM_RUN_TAG:-}" ] && extra+=(--run-tag "$TT_SIM_RUN_TAG")

echo "[run.sh] starting Wolfpine $ARCH server on $NNG_SOCKET_ADDR" >&2
exec "${WOLFPINE_PYTHON:-python3}" -u -m "framework.sim.$ARCH" "${extra[@]}" "$@"
