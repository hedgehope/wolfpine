#!/usr/bin/env bash
# Build riscv-vector-tests (chipsalliance) for Naiad: VLEN 256, XLEN 64, ELEN 64.
#
# The suite is a Go generator plus a two-stage flow: stage 1 emits each test with
# a custom-0 "magic" word after every instruction under test; a patched Spike
# (pspike, linked against the Spike that tools/naiad/build_spike.sh installs)
# runs it and prints the expected register-group contents as TEST_CASEs; the
# merger splices them in, giving self-checking stage-2 tests. Upstream builds
# with riscv64-unknown-elf-gcc and riscv-test-env's "p" environment; here:
#
# * the compiler is clang 20.1.7 (no gcc needed);
# * the environment is tools/naiad/vector_env (M-mode, no traps, HTIF through
#   tohost), because p's trap-vector / mret boot cannot run on Naiad. A failing
#   check N exits with code N, so the runner can name it;
# * the march has no C (Naiad's RV64 core decodes no RVC yet) and no vector
#   crypto, only what the naiad-rvv design lists: V + Zvfh + Zvfbfmin + Zvfbfwma
#   (Zfhmin/Zfbfmin for the fmv.h.x the fp16/bf16 tests use to load scalars).
#
# Every stage-2 binary is then run once on (stock) Spike, which must pass it.
#
#   tools/naiad/build_vector_tests.sh                 # everything
#   PATTERN='^vadd' tools/naiad/build_vector_tests.sh # a subset (regex on the test name)
#   FROM=compile tools/naiad/build_vector_tests.sh    # only recompile stage 2 + recheck
#                                                     # (after an edit to vector_env/)
#
# Output: $SRC/out/naiad-v256x64/{stage2,bin}/ (~6.5 GB); bin/ is what
# tools/naiad/vectortest.py runs. A list of tests Spike failed (should be empty)
# is written to $OUT/spike-failures.txt.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HOME/projects/riscv/riscv-vector-tests}
PREFIX=${PREFIX:-$HOME/.local}
JOBS=${JOBS:-$(nproc)}
CC=${NAIAD_CC:-/opt/llvm/20.1.7/bin/clang}
SPIKE=${NAIAD_SPIKE:-$PREFIX/bin/spike}
PATTERN=${PATTERN:-.*}
VLEN=256
XLEN=64
OUT=${OUT:-$SRC/out/naiad-v${VLEN}x${XLEN}}
ENV_DIR=$HERE/vector_env

REPO=https://github.com/chipsalliance/riscv-vector-tests.git
REV=f76bff121dce91b6b23e1d37be77a5e44af914c3 # master, 2026-05-24
GO_VERSION=go1.27.3                          # into $PREFIX/go, no root needed

MARCH=rv64imafdv_zicsr_zifencei_zfhmin_zfbfmin_zvfh_zvfbfmin_zvfbfwma
SPIKE_ISA=${MARCH}_zicntr_zvl${VLEN}b
CFLAGS=(--target=riscv64-unknown-elf -march=$MARCH -mabi=lp64d -static -mcmodel=medany
  -fvisibility=hidden -nostdlib -nostartfiles -I"$ENV_DIR" -I"$SRC/macros/general" -T"$ENV_DIR/link.ld")

# -- Go ---------------------------------------------------------------------
if [ ! -x "$PREFIX/go/bin/go" ]; then
  tmp=$(mktemp -d)
  curl -sL -o "$tmp/go.tgz" "https://go.dev/dl/$GO_VERSION.linux-amd64.tar.gz"
  sum=$(curl -s 'https://go.dev/dl/?mode=json&include=all' | python3 -c \
    "import json,sys; print(next(f['sha256'] for r in json.load(sys.stdin) for f in r['files'] if f['filename']=='$GO_VERSION.linux-amd64.tar.gz'))")
  echo "$sum  $tmp/go.tgz" | sha256sum -c -
  tar -C "$PREFIX" -xzf "$tmp/go.tgz"
  rm -rf "$tmp"
fi
export PATH=$PREFIX/go/bin:$PATH
export GOPATH=${GOPATH:-$PREFIX/gopath} GOCACHE=${GOCACHE:-$HOME/.cache/go-build}

# -- generator, merger, pspike ----------------------------------------------
if [ ! -d "$SRC/.git" ]; then git clone --quiet "$REPO" "$SRC"; fi
git -C "$SRC" checkout --quiet "$REV"
git -C "$SRC" submodule update --init --quiet
(cd "$SRC" && go build -o build/generator && go build -o build/merger merger/merger.go)
g++ -std=c++17 -O2 -I"$PREFIX/include" -L"$PREFIX/lib" "$SRC/pspike/pspike.cc" \
  -lriscv -lfesvr -Wl,-rpath,"$PREFIX/lib" -o "$SRC/build/pspike"

export CC OUT SRC SPIKE SPIKE_ISA
export CFLAGS_STR="${CFLAGS[*]}"

if [ "${FROM:-all}" = all ]; then
# -- stage 1 ----------------------------------------------------------------
rm -rf "$OUT"
mkdir -p "$OUT"/{stage1,bin1,patch,stage2,bin}
(cd "$SRC" && build/generator -VLEN $VLEN -XLEN $XLEN -split=10000 -integer=false \
  -pattern="$PATTERN" -testfloat3level=2 -repeat=1 -stage1output "$OUT/stage1/" \
  -configs configs/ -march "gv_zvfh_zvfbfmin_zvfbfwma")
git -C "$SRC" checkout --quiet Makefrag # the generator rewrites it; we do not use it

ls "$OUT/stage1" | sed 's/\.S$//' >"$OUT/tests.txt"
echo "$(wc -l <"$OUT/tests.txt") tests"

# shellcheck disable=SC2016
xargs -P "$JOBS" -I{} sh -c '$CC $CFLAGS_STR "$OUT/stage1/{}.S" -o "$OUT/bin1/{}"' <"$OUT/tests.txt"
# shellcheck disable=SC2016
xargs -P "$JOBS" -I{} sh -c '"$SRC/build/pspike" --isa=$SPIKE_ISA "$OUT/bin1/{}" >"$OUT/patch/{}.patch"' <"$OUT/tests.txt"

# -- stage 2 ----------------------------------------------------------------
"$SRC/build/merger" -stage1output "$OUT/stage1/" -stage2output "$OUT/stage2/" -stage2patch "$OUT/patch/"
# Stage-1 intermediates are ~3.5 GB and not needed again (KEEP_STAGE1=1 keeps them).
[ "${KEEP_STAGE1:-0}" = 1 ] || rm -rf "$OUT/stage1" "$OUT/bin1" "$OUT/patch"
fi
rm -f "$OUT/bin/"*
# shellcheck disable=SC2016
xargs -P "$JOBS" -I{} sh -c '$CC $CFLAGS_STR "$OUT/stage2/{}.S" -o "$OUT/bin/{}"' <"$OUT/tests.txt"

# -- check every test on Spike ----------------------------------------------
: >"$OUT/spike-failures.txt"
# shellcheck disable=SC2016
xargs -P "$JOBS" -I{} sh -c 'timeout 120 "$SPIKE" --isa=$SPIKE_ISA "$OUT/bin/{}" >/dev/null 2>&1 || echo "{} $?" >>"$OUT/spike-failures.txt"' <"$OUT/tests.txt"
{
  echo "riscv-vector-tests $REV, VLEN=$VLEN XLEN=$XLEN, march $MARCH"
  echo "generator: -VLEN $VLEN -XLEN $XLEN -split=10000 -testfloat3level=2 -repeat=1 -pattern='$PATTERN' -march gv_zvfh_zvfbfmin_zvfbfwma"
  echo "cc: $("$CC" --version | head -1)"
  echo "spike: $SPIKE --isa=$SPIKE_ISA"
} >"$OUT/BUILD_INFO"
echo "$(wc -l <"$OUT/tests.txt") tests built in $OUT/bin; Spike failures: $(wc -l <"$OUT/spike-failures.txt")"
