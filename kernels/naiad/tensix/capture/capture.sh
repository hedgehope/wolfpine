#!/usr/bin/env bash
# Re-capture the Naiad Tensix GEMM images from a live tt-metal run on the
# Wolfpine Blackhole simulator. Needs a built tt-metal (TT_METAL_HOME) and the
# Wolfpine venv on PATH; run in a normal shell (UMD spawns the sim server).
#
#   kernels/naiad/tensix/capture/capture.sh            # both images
#   kernels/naiad/tensix/capture/capture.sh fp32       # just one
#
# For each output format it builds src/ (once), runs the capture program with
# TT_SIM_RECORD, keeps only the worker's messages (framework.naiad.tensix_image
# --filter) as framework/naiad/tensix_images/bh_gemm_bf16_<out>.trace, and
# regenerates the C image header kernels/naiad/include/tensix_gemm_<out>.h.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
: "${TT_METAL_HOME:?set TT_METAL_HOME to a built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_SIMULATOR="$REPO/driver/blackhole"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_SIM_TENSIX_COORDS=1-2
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
FIDELITY="${TENSIXGEMM_FIDELITY:-HiFi2}"
SRC="$HERE/src"
cmake -B "$SRC/build" -S "$SRC" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$SRC/build" -j > /dev/null
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
for out in "${@:-fp32 bf16}"; do
  for o in $out; do
    echo "[capture] bf16 -> $o ($FIDELITY)"
    (cd "$SRC" && TENSIXGEMM_OUT="$o" TENSIXGEMM_FIDELITY="$FIDELITY" TT_SIM_RECORD="$TMP/$o.trace" \
      ./build/tensixgemm | tee "$TMP/$o.out" | grep -E "TENSIXGEMM|Completed|Failure")
    python3 -m framework.naiad.tensix_image filter "$TMP/$o.trace" --core 1,2 \
      -o "$REPO/framework/naiad/tensix_images/bh_gemm_bf16_$o.trace" \
      --note "$(grep TENSIXGEMM_CAPTURE "$TMP/$o.out")"
    python3 -m framework.naiad.tensix_image header \
      "$REPO/framework/naiad/tensix_images/bh_gemm_bf16_$o.trace" --name "gemm_$o" \
      -o "$REPO/kernels/naiad/include/tensix_gemm_$o.h"
  done
done
