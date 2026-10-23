#!/usr/bin/env bash
# Build the Spike reference simulator (and the dtc it needs) into a user prefix.
#
# Spike is the golden model for Naiad's RV64 core (tools/naiad/difftest.py,
# tools/naiad/archtest.py). Neither needs root: both install under $PREFIX.
#
# riscv-isa-sim's only release tags (v1.0.0, v1.1.0) predate Zicntr, Zaamo and
# Zalrsc as ISA-string extensions, so Spike is pinned to a master commit.
#
#   tools/naiad/build_spike.sh            # build + install into ~/.local
#   PREFIX=/opt/x SRC=/tmp/b tools/naiad/build_spike.sh
set -euo pipefail

PREFIX=${PREFIX:-$HOME/.local}
SRC=${SRC:-$HOME/projects/riscv}
JOBS=${JOBS:-$(nproc)}

DTC_REPO=https://github.com/dgibson/dtc.git
DTC_REV=v1.8.1
SPIKE_REPO=https://github.com/riscv-software-src/riscv-isa-sim.git
SPIKE_REV=5a760667 # master, 2026-10-15 (before softfloat became a submodule)

fetch() { # repo rev dir
  if [ ! -d "$3/.git" ]; then git clone --quiet "$1" "$3"; fi
  git -C "$3" fetch --quiet --tags origin
  git -C "$3" checkout --quiet "$2"
}

# dtc: Spike calls it at configure time and at run time to compile its device tree.
fetch "$DTC_REPO" "$DTC_REV" "$SRC/dtc"
make -C "$SRC/dtc" -j"$JOBS" NO_PYTHON=1 NO_YAML=1 NO_VALGRIND=1 PREFIX="$PREFIX" install-bin
export PATH="$PREFIX/bin:$PATH"

fetch "$SPIKE_REPO" "$SPIKE_REV" "$SRC/riscv-isa-sim"
mkdir -p "$SRC/riscv-isa-sim/build"
cd "$SRC/riscv-isa-sim/build"
../configure --prefix="$PREFIX"
make -j"$JOBS"
make install

"$PREFIX/bin/dtc" --version
"$PREFIX/bin/spike" --help 2>&1 | head -1
