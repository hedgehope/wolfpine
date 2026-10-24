#!/bin/sh
# Run every kernel of the K1 pack and print one combined output (stdout), in the
# format tools/naiad/k1/compare.py reads. Build first with `make` (and `make xsmt`
# for the optional IME kernel, which this runs too when it exists).
#
#   ./run_k1.sh > k1.out
#   TASKSET="taskset -c 0" ./run_k1.sh > k1.out   # pin to one core for steadier timings
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HERE/build/linux}
echo "# k1-pack run $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(uname -n) kernel $(uname -r)"
grep -m1 -E '^(isa|model name|uarch)' /proc/cpuinfo 2>/dev/null | sed 's/^/# cpu: /'
status=0
for k in axpy dot gemm reduce gather xsmt_gemm rvv_check; do
  if [ ! -x "$BIN/$k" ]; then
    case $k in xsmt_gemm | rvv_check) ;; *) echo "# MISSING $k (run make)"; status=1 ;; esac
    continue
  fi
  if [ "$k" = rvv_check ]; then
    # Its own format (one "name: ok|MISMATCH" line per check): kept as comments.
    out=$(${TASKSET:-} "$BIN/$k" 2>&1); rc=$?
    printf '%s\n' "$out" | sed 's/^/# rvv_check: /'
    [ $rc -eq 0 ] || { echo "# FAILED rvv_check exit $rc"; status=1; }
    continue
  fi
  ${TASKSET:-} "$BIN/$k" || { echo "# FAILED $k exit $?"; status=1; }
done
exit $status
