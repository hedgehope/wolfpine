#!/usr/bin/env bash
# One card session for the PAPER's evaluation section: every silicon-only cell
# of docs/plans/evalplan.md, in one batch.
#
# This is the HARDWARE side, and it is a sibling of run_card_session.sh rather
# than a replacement: that script collects the perfbench COMPONENT slopes
# (evalplan experiment 2a), this one collects everything else the card owns --
# the ladder's hardware column, its device cycles, the differential optests'
# hardware arm, and the observability facts the capability matrix asserts.
# Neither runs the other. Run run_card_session.sh first if 2a still needs data;
# the 2026-08-09 Blackhole session already banked most of it.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   perfbench/run_paper_session.sh                # everything that applies
#   perfbench/run_paper_session.sh --list         # what would run, and why
#   perfbench/run_paper_session.sh sku ladder     # just those steps
#   perfbench/run_paper_session.sh --resume       # skip steps already done
#
# Options:
#   --arch wormhole|blackhole  skip auto-detection and assert the part
#   --out DIR                  results directory
#                              (default: ~/tt_traces/paper-session-<arch>)
#   --optests DIR              the repo's optests/ tree (default: <repo>/optests,
#                              then ~/optests). Ships separately from perfbench/.
#   --grid WxH                 core-grid override for the multi-core rungs,
#                              as TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE
#                              (default 3,4 -- the gap-free 4x5 sub-block)
#   --rungs a,b,c              restrict the ladder to these rungs (substring
#                              match on the rung name). Useful to resume a
#                              ladder one rung at a time, and to validate the
#                              session block without paying for rung 5.
#   --list                     list steps, their parts and cost, then exit
#   --resume                   skip any step whose output is already in --out
#   --dry-run                  print the commands without running them
#   --sim                      VALIDATION ONLY -- run against Wolfpine at smoke
#                              sizes. Every output is stamped NOT-A-MEASUREMENT.
#   -h, --help                 this text
#
# Runtime: about 35 minutes on a card once tt-metal and the optests are built;
# `--list` prints the per-step breakdown. Nothing here writes a card's flash,
# changes any clock, or needs root.
#
# What this session is FOR, in one line: a silicon measurement is CORROBORATION
# of the simulator, never provenance for it. Nothing collected here may be
# turned into a cost-model entry; see docs/plans/cost-model.md's house rules.
#
# SPDX-License-Identifier: Apache-2.0

set -u

PB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$PB/.." && pwd)"
# shellcheck source=build_provenance.sh
. "$PB/build_provenance.sh"

ARCH=""
OUT=""
OPTESTS=""
# TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE, and the evidence for BOTH settings.
#
# It restricts tt-metal to a gap-free 4x5 sub-block. On 2026-08-10 it was
# defaulted OFF on hardware, on the reasoning that it is a simulator
# convenience and would cut a 120-worker part to 20. That reasoning was wrong
# for THIS part, and the card said so:
#
#   with 3,4     rung 5 matmul_multi_core PASS (PCC 0.982)
#   without      rung 5 dies in tt-metal with YAML::TypedBadConversion<int>
#
# The card is HARVESTED -- addressed worker columns {1..7, 10..14}, a gap at
# 8..9 -- so the full-grid path is not the well-trodden one here, and 0.74
# throws parsing a descriptor rather than reporting anything useful. The
# override is therefore a HARVESTING workaround on silicon as well as a
# descriptor one under the simulator, and it stays on by default.
#
# It is not free: it is why rung 5's PCC reads 0.982 rather than the ~0.9999
# the same program reaches elsewhere, because the work is spread over 20 cores
# instead of 120. Say so wherever that number is quoted. Use --no-grid to run
# the full grid and watch it fail.
GRID="3,4"
GRID_EXPLICIT=0
OBS_PROG=""
RUNGS=""
DO_LIST=0
DO_RESUME=0
DO_DRY=0
DO_SIM=0
WANT=()

# ----------------------------------------------------------------- step table
# name | parts | est.min | which evalplan / paper cell it fills
STEPS=(
  "sku|both|1|the part's own identity: tile count, AICLK, fw bundle, KMD. Fills the paper's setup TODOs and settles the n150-vs-Blackhole SKU question"
  "ladder|both|6|the five benchmark rungs on silicon: runs/does-not-run, self-check verdict, stdout, wall clock. Table 'functional' HW column + table 'sim_perf' slowdown denominator"
  "ladder-prof|both|10|the same rungs under TT_METAL_DEVICE_PROFILER=1, banking profile_log_device.csv. Table 'e2e_timing' HW column. Banked NOW because the Wolfpine half is one unimplemented NoC register away and a second card booking is expensive"
  "optests|both|12|every optests/<name> on silicon, capturing its OPDIFF_RESULT hex. The third arm of the differential: turns the local Wolfpine-vs-ttsim agreement into agreement against HARDWARE. Table 'functional' bit-exact column + the 1b census"
  "obs|both|4|what hardware introspection actually delivers, run rather than cited: watcher on a deliberately wedged launch, and DPRINT from a compute kernel. Table 'obs_matrix' HW column"
)

# Cells this session cannot fill because they need the other part. Not stuck --
# Wormhole is a planned follow-on and every step below already runs there.
DEFERRED_TO_WORMHOLE=(
  "every hardware column of every table, for Wormhole. The paper evaluates one part; the simulator supports two. There is no Wormhole card on this site, so the Wormhole rows of tables 'functional', 'e2e_timing' and 'sim_perf' have no hardware ground truth and must be declared absent rather than inferred from Blackhole."
  "the cross-arch half of the 1b optests census. Wolfpine-vs-ttsim agreement is already measured on BOTH arches locally; only the hardware arm is Blackhole-only."
)

step_field() { # name, field-index (1-based)
  local p
  for p in "${STEPS[@]}"; do
    case "$p" in "$1|"*) printf '%s' "$p" | cut -d'|' -f"$2"; return 0 ;; esac
  done
  return 1
}

step_names() {
  local p
  for p in "${STEPS[@]}"; do printf '%s\n' "${p%%|*}"; done
}

# ------------------------------------------------------------------ arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="${2:-}"; shift 2 || exit 2 ;;
    --out) OUT="${2:-}"; shift 2 || exit 2 ;;
    --optests) OPTESTS="${2:-}"; shift 2 || exit 2 ;;
    --grid) GRID="${2:-}"; GRID_EXPLICIT=1; shift 2 || exit 2 ;;
    --no-grid) GRID=""; GRID_EXPLICIT=1; shift ;;
    --rungs) RUNGS="${2:-}"; shift 2 || exit 2 ;;
    --obs-prog) OBS_PROG="${2:-}"; shift 2 || exit 2 ;;
    --list) DO_LIST=1; shift ;;
    --resume) DO_RESUME=1; shift ;;
    --dry-run) DO_DRY=1; shift ;;
    --sim) DO_SIM=1; shift ;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *)
      if step_field "$1" 1 >/dev/null; then WANT+=("$1"); else
        echo "unknown step '$1'; try --list" >&2; exit 2
      fi
      shift ;;
  esac
done

case "${ARCH:-}" in
  ""|wormhole|blackhole) ;;
  wh) ARCH=wormhole ;;
  bh) ARCH=blackhole ;;
  *) echo "--arch must be wormhole or blackhole" >&2; exit 2 ;;
esac

if [ "$DO_LIST" = 1 ]; then
  printf '%-14s %-10s %6s  %s\n' STEP PART MIN "FILLS"
  total=0
  for p in "${STEPS[@]}"; do
    n="$(printf '%s' "$p" | cut -d'|' -f1)"
    a="$(printf '%s' "$p" | cut -d'|' -f2)"
    m="$(printf '%s' "$p" | cut -d'|' -f3)"
    d="$(printf '%s' "$p" | cut -d'|' -f4)"
    printf '%-14s %-10s %6s  ' "$n" "$a" "$m"
    printf '%s\n' "$d" | fold -s -w 60 | sed '2,$s/^/                              /'
    total=$((total + m))
  done
  echo
  echo "run time once tt-metal and the optests are built: ~${total} min."
  echo "Cold, add the tt-metal build and ~1 min per optests program."
  echo
  echo "This session does NOT collect the perfbench component slopes (the"
  echo "paper's table 'perfbench'). That is run_card_session.sh, and the"
  echo "2026-08-09 Blackhole session already banked tensixbench x2,"
  echo "riscvbench x6, nocbench x2, nocreadbench and dramratebench."
  echo
  echo "Awaiting the planned Wormhole follow-on (every step below already runs"
  echo "there -- pass --arch wormhole -- so it is a hardware booking, not work):"
  for d in "${DEFERRED_TO_WORMHOLE[@]}"; do
    printf '  - %s\n' "$d" | fold -s -w 74 | sed '2,$s/^/    /'
  done
  echo
  echo "Cannot be collected here at all, by construction:"
  echo "  the ttsim hardware comparison   (ttsim IS a simulator; its wall clock"
  echo "                                   is a HOST measurement, taken at home)"
  echo "  Wolfpine's own cycle counts       (taken at home; only the HW column of"
  echo "                                   the e2e table comes off a card)"
  exit 0
fi

# ------------------------------------------------------------------ the world
: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# Slow dispatch throughout. Wolfpine supports only the direct launch path, and the
# paper's claim is that the SAME binary runs on all three platforms, so the
# hardware runs must use the same flow. This is not a simulator concession
# leaking onto silicon; it is the control.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"

if [ "$DO_SIM" = 1 ]; then
  # Validation path: exercise the session block end to end without a card.
  # NOT a measurement, and every artefact says so.
  [ -n "$ARCH" ] || ARCH=blackhole
  case "$ARCH" in
    blackhole) SIM_COORDS=1-2 ;;
    wormhole) SIM_COORDS=1-1 ;;
  esac
  export TT_METAL_SIMULATOR="$REPO/driver/$ARCH"
  export TT_SIM_TENSIX_COORDS="${TT_SIM_TENSIX_COORDS:-$SIM_COORDS}"
  VENV="${TT_SIM_VENV:-$REPO/../venv}"
  [ -x "$VENV/bin/python3" ] && export PATH="$VENV/bin:$PATH"
  echo "!! --sim: running against Wolfpine at smoke sizes. NOT A MEASUREMENT."
elif [ -n "${TT_METAL_SIMULATOR:-}" ]; then
  # A dry run executes nothing, and inspecting the session block off-card is
  # exactly what it is for -- on this site the venv exports TT_METAL_SIMULATOR
  # in every shell, so refusing a dry run would make it unusable at home. Warn
  # and continue; refuse anything that would actually touch a device.
  if [ "$DO_DRY" = 1 ]; then
    echo "!! TT_METAL_SIMULATOR is set ($TT_METAL_SIMULATOR); --dry-run runs" >&2
    echo "!! nothing, so this is a warning. A real run would refuse here." >&2
  else
    echo "TT_METAL_SIMULATOR is set ($TT_METAL_SIMULATOR)." >&2
    echo "This script is for a real card. Unset it, or pass --sim if you meant" >&2
    echo "to validate the session block against Wolfpine." >&2
    exit 2
  fi
fi

PE="$TT_METAL_HOME/build/programming_examples"

# ------------------------------------------------------------ arch detection
# Same instrument run_card_session.sh uses, for the same reason: nocbench
# --dump-grid asks the DEVICE what it is rather than trusting an env var, and
# the sku step wants the dump anyway.
detect_arch() {
  local grid="$1"
  ( cd "$PB/nocbench/src" || exit 2
    bp_check_build "$PB/nocbench/src" "$([ -x build/nocbench ] && echo skip-build || echo build)" || exit 1
    [ -x build/nocbench ] || {
      cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 || exit 1
      cmake --build build -j >/dev/null 2>&1 || exit 1
      bp_record_build "$PB/nocbench/src"; }
    ./build/nocbench --dump-grid --out "$grid" >/dev/null 2>&1 ) || return 1
  sed -n 's/.*arch=\([a-z0-9_]*\).*/\1/p' "$grid" | head -1
}

if [ -z "$ARCH" ]; then
  if [ "$DO_DRY" = 1 ]; then
    ARCH=unknown
  else
    echo "== detecting the part"
    DETECT_TMP="$(mktemp -d "${TMPDIR:-/tmp}/paper-session-detect.XXXXXX")" || exit 2
    ARCH="$(detect_arch "$DETECT_TMP/nocbench-grid.csv")"
    if [ -z "$ARCH" ]; then
      rm -rf "$DETECT_TMP"
      echo "could not detect the part; pass --arch" >&2
      exit 1
    fi
    echo "   arch=$ARCH"
  fi
fi

[ -n "$OUT" ] || OUT="$HOME/tt_traces/paper-session-${ARCH}"
mkdir -p "$OUT" || exit 2
GRIDDUMP="$OUT/nocbench-grid.csv"
if [ -n "${DETECT_TMP:-}" ]; then
  [ -s "$DETECT_TMP/nocbench-grid.csv" ] && cp "$DETECT_TMP/nocbench-grid.csv" "$GRIDDUMP"
  rm -rf "$DETECT_TMP"
fi

# optests/ ships separately from perfbench/ -- two rsyncs, not one. Rather than
# fail the whole session, locate it and let the optests step skip itself with a
# message naming the exact command that would fix it.
if [ -z "$OPTESTS" ]; then
  for c in "$REPO/optests" "$HOME/optests"; do
    [ -d "$c" ] && { OPTESTS="$c"; break; }
  done
fi

LOG="$OUT/session.log"
[ "$DO_DRY" = 1 ] || : > "$LOG"

say() { echo "$@"; [ "$DO_DRY" = 1 ] || echo "$@" >> "$LOG"; }

VERDICTS=()
verdict() { VERDICTS+=("$1|$2|$3"); say "   VERDICT $1: $2 -- $3"; }
skip() { verdict "$1" SKIPPED "$2"; }

# --------------------------------------------------------------- the ladder
# One row per paper benchmark rung. Fields:
#   rung | kind | binary | dir | extra env (space separated K=V) | timeout s
#
# `kind` is `upstream` for a stock programming_examples binary and `intree` for
# one of this repo's own examples/<name>/src builds. FOUR of the five rungs are
# upstream programs run UNMODIFIED, which is the paper's claim; rung 4 is in
# tree because upstream's single-core matmul is hard-coded to 640^3 with no
# runtime or link-time size knob and does not complete inside any sane budget.
# The 640^3 run is kept as its own row (`matmul-single-640`) precisely so the
# timeout is reported rather than hidden -- it is a result about throughput.
LADDER=(
  "1-add2int|upstream|metal_example_add_2_integers_in_riscv||600"
  "2-loopback|upstream|metal_example_loopback||600"
  "3-eltwise|upstream|metal_example_eltwise_binary||900"
  "4-matmul-single|intree|six||900"
  "5-matmul-multi|upstream|metal_example_matmul_multi_core|GRIDOVERRIDE|1800"
  "6-matmul-single-640|upstream|metal_example_matmul_single_core||1800"
)

ladder_env() { # the placeholder in the table -> a real assignment
  case "$1" in
    GRIDOVERRIDE) printf 'TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=%s' "$GRID" ;;
    "") ;;
    *) printf '%s' "$1" ;;
  esac
}

# Worker coordinates the simulator needs materialised, under --sim only. A card
# needs none of this: every core exists.
sim_coords_for() { # rung name
  case "$ARCH" in
    blackhole) local one=1-2 ;;
    *) local one=1-1 ;;
  esac
  case "$1" in
    5-matmul-multi)
      case "$ARCH" in
        blackhole) printf '1-2,2-2,3-2,4-2,1-3,2-3,3-3,4-3,1-4,2-4,3-4,4-4,1-5,2-5,3-5,4-5,1-6,2-6,3-6,4-6' ;;
        *) printf '1-1,2-1,3-1,4-1,1-2,2-2,3-2,4-2,1-3,2-3,3-3,4-3,1-4,2-4,3-4,4-4,1-5,2-5,3-5,4-5' ;;
      esac ;;
    *) printf '%s' "$one" ;;
  esac
}

# A rung's own verdict on itself. Every ladder program either self-checks (exit
# non-zero on mismatch) or prints a success line; nothing here decides
# correctness by the absence of an error string.
ladder_ok() { # logfile, rc
  local log="$1" rc="$2"
  if [ "$rc" = 124 ]; then printf 'TIMEOUT|no result inside the budget'; return; fi
  if [ "$rc" != 0 ]; then printf 'FAILED|exit %s' "$rc"; return; fi
  # grep is line-based, so `.` is the right "rest of the line" atom here; an
  # earlier revision used `[^\n]`, which in a POSIX bracket expression means
  # "not a backslash and not the letter n" and quietly returned an empty match
  # for `Test Passed`. A verdict line that renders empty reads as a pass with no
  # evidence, which is the failure this whole file exists to prevent.
  if grep -qiE 'Completed successfully|Test Passed|PCC|Result = ' "$log" 2>/dev/null; then
    printf 'PASS|%s' "$(grep -m1 -oiE 'Completed successfully.{0,40}|Test Passed|PCC.{0,30}|Result = .{0,30}' "$log")"
  else
    printf 'PASS|exit 0, no success line printed (this program self-checks by exit status)'
  fi
}

build_intree() { # example name
  local dir="$REPO/examples/$1/src"
  [ -d "$dir" ] || { say "   no in-tree example at $dir"; return 1; }
  [ "$DO_DRY" = 1 ] && { echo "   + build examples/$1"; return 0; }
  if [ -x "$dir/build/$1" ]; then
    local newer
    newer="$(find "$dir" -path "$dir/build" -prune -o -type f \
               \( -name '*.cpp' -o -name '*.h' -o -name 'CMakeLists.txt' \) \
               -newer "$dir/build/$1" -print 2>/dev/null | head -1)"
    [ -z "$newer" ] && return 0
    say "   rebuilding $1 (source newer than the binary: $(basename "$newer"))"
  else
    say "   building $1 (first time)"
  fi
  # A CMakeCache.txt records the ABSOLUTE path it was generated in, so a tree
  # that arrived by rsync from another machine makes cmake refuse outright. That
  # cost a card session on 2026-08-09. A build tree is derived and always safe
  # to discard.
  if [ -f "$dir/build/CMakeCache.txt" ] && \
     ! grep -qxF "CMAKE_CACHEFILE_DIR:INTERNAL=$dir/build" "$dir/build/CMakeCache.txt" 2>/dev/null; then
    say "   discarding a build tree generated elsewhere (rsync'd CMakeCache)"
    rm -rf "$dir/build"
  fi
  # A tree generated HERE but against a different tt-metal survives the check
  # above and still yields a binary that cannot run -- cmake reuses the cache,
  # so the headers come from one checkout and the library from the other. Which
  # checkout was meant is an operator question, so it refuses rather than
  # discarding. See build_provenance.sh.
  local bp_out bp_rc=0
  bp_out="$(bp_check_build "$dir")" || bp_rc=1
  say "$bp_out"
  [ "$bp_rc" -eq 0 ] || return 1
  ( cd "$dir" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 \
      && cmake --build build -j >/dev/null 2>&1 ) || {
    say "   build failed; retrying from a clean tree"
    rm -rf "$dir/build"
    ( cd "$dir" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null \
        && cmake --build build -j >/dev/null ) || return 1
  }
  bp_record_build "$dir"
  touch "$dir/build/$1" 2>/dev/null
}

# Run one rung. $2 selects the flavour: "" (plain) or "prof" (device profiler).
run_rung() {
  local row="$1" flavour="${2:-}"
  local name kind bin envph tmo dir prog extra t0 t1 rc log wall
  name="$(printf '%s' "$row" | cut -d'|' -f1)"
  kind="$(printf '%s' "$row" | cut -d'|' -f2)"
  bin="$(printf '%s' "$row" | cut -d'|' -f3)"
  envph="$(printf '%s' "$row" | cut -d'|' -f4)"
  tmo="$(printf '%s' "$row" | cut -d'|' -f5)"
  extra="$(ladder_env "$envph")"

  if [ "$kind" = intree ]; then
    dir="$REPO/examples/$bin/src"; prog="./build/$bin"
  else
    dir="$PE"; prog="./$bin"
  fi

  # The 640^3 single-core rung exists to be reported as a timeout on Wolfpine; on
  # a card it is seconds, so it is run and its number kept. Under --sim it would
  # burn the whole validation budget for a known answer.
  if [ "$DO_SIM" = 1 ] && [ "$name" = 6-matmul-single-640 ]; then
    say "   $name: SKIPPED under --sim (known TIMEOUT-slow; 8000 tile-matmuls)"
    return
  fi

  # --rungs restricts the ladder. A restricted ladder is announced in the log,
  # because a table built from a partial ladder with no note is a table with
  # silently missing rows.
  if [ -n "$RUNGS" ]; then
    local want=0 r
    IFS=',' read -r -a _rl <<< "$RUNGS"
    for r in "${_rl[@]}"; do case "$name" in *"$r"*) want=1 ;; esac; done
    [ "$want" = 1 ] || { say "   $name: not in --rungs $RUNGS"; return; }
  fi

  local suffix=""; [ -n "$flavour" ] && suffix=".$flavour"
  log="$OUT/rung.$name$suffix.out"

  if [ "$DO_DRY" = 1 ]; then
    echo "   + (cd $dir && $extra $prog)  -> $log"
    return
  fi
  if [ "$kind" = intree ]; then
    build_intree "$bin" || { verdict "$name$suffix" FAILED "in-tree build failed"; return; }
  elif [ ! -x "$dir/$bin" ]; then
    verdict "$name$suffix" SKIPPED "no $bin in $PE -- build tt-metal's programming_examples"
    return
  fi

  t0=$(date +%s.%N)
  ( cd "$dir" || exit 2
    [ -n "$extra" ] && export "${extra?}"
    if [ "$flavour" = prof ]; then
      export TT_METAL_DEVICE_PROFILER=1
      export TT_METAL_PROFILER_DIR="$OUT/prof.$name"
      mkdir -p "$TT_METAL_PROFILER_DIR"
    fi
    if [ "$DO_SIM" = 1 ]; then export TT_SIM_TENSIX_COORDS="$(sim_coords_for "$name")"; fi
    timeout "$tmo" $prog ) >"$log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')"
  printf '%s,%s,%s,%s\n' "$name" "$rc" "$wall" "$flavour" >> "$OUT/ladder$suffix.csv"
  local line; line="$(ladder_ok "$log" "$rc")"
  say "   $name: ${line%%|*} in ${wall}s -- ${line#*|}"
  verdict "$name$suffix" "${line%%|*}" "${line#*|} (${wall}s)"
}

# ------------------------------------------------------------------- steps
step_sku() {
  # The paper says its results are measured on an "n150 Blackhole". n150 is a
  # WORMHOLE SKU. Rather than argue from memory, record what the part says about
  # itself and let the number of workers, the AICLK and the board type settle it.
  local f="$OUT/sku.txt"
  if [ "$DO_DRY" = 1 ]; then echo "   + tt-smi -ls; nocbench --dump-grid; UMD banner"; return; fi
  if [ "$DO_SIM" = 1 ]; then
    # Deliberately not merely "degenerate": a simulator has no part number, no
    # AICLK and no firmware bundle, so there is nothing here that could be a
    # reading. Exercise the file-writing path, then say so.
    : > "$f"; echo "SIM SMOKE -- a simulator has no SKU. NOT A MEASUREMENT." >> "$f"
    skip sku "a simulator has no part identity; this step exists to be run on a card"
    return
  fi
  : > "$f"
  {
    echo "== date $(date -Is)"
    echo "== uname"; uname -a
    echo "== tt-smi -ls"; (tt-smi -ls 2>&1 || echo "(tt-smi not on PATH)")
    echo "== tt-smi -s"; (tt-smi -s 2>&1 || echo "(tt-smi not on PATH)")
    echo "== lspci Tenstorrent"; (lspci -nn 2>/dev/null | grep -i tenstorrent || echo "(none)")
    echo "== kmd version"; (cat /sys/module/tenstorrent/version 2>/dev/null || echo "(unknown)")
  } >> "$f" 2>&1
  # `--dump-grid` probes the first logical row and column of the WHOLE compute
  # grid. Against Wolfpine only the tiles named in TT_SIM_TENSIX_COORDS exist, so
  # the launch waits forever on cores that do not -- the same trap
  # run_card_session.sh documents for its congestion probes. Never run it under
  # --sim; a simulator has no SKU to report anyway.
  if [ "$DO_SIM" = 1 ]; then
    say "   grid dump skipped under --sim (--dump-grid probes the full grid and hangs)"
  else
    [ -s "$GRIDDUMP" ] || ( cd "$PB/nocbench/src" && ./build/nocbench --dump-grid --out "$GRIDDUMP" ) >>"$LOG" 2>&1
  fi
  {
    echo "== nocbench --dump-grid header"
    grep -m5 '^#' "$GRIDDUMP" 2>/dev/null
    echo "== worker rows in the dump: $(grep -cvE '^#|^log_x' "$GRIDDUMP" 2>/dev/null)"
  } >> "$f" 2>&1
  # The UMD banner carries the firmware bundle, the KMD version, the logical
  # grid and the AICLK -- every one of them a thing the paper's setup section
  # has a TODO for. The cheapest program that prints it is the first ladder rung.
  if [ -x "$PE/metal_example_add_2_integers_in_riscv" ]; then
    ( cd "$PE" && timeout 300 ./metal_example_add_2_integers_in_riscv ) >"$OUT/sku-banner.out" 2>&1
    {
      echo "== UMD banner (from a live launch)"
      grep -iE "firmware bundle|KMD version|logical grid|Opening local chip|architecture:|AICLK|clock" \
        "$OUT/sku-banner.out" | head -20
    } >> "$f" 2>&1
  fi
  local workers clk
  workers="$(grep -cvE '^#|^log_x' "$GRIDDUMP" 2>/dev/null)"
  clk="$(grep -m1 -oE 'clock_mhz=[0-9]+' "$GRIDDUMP" 2>/dev/null | cut -d= -f2)"
  say "   workers in the grid dump: ${workers:-<none>}   clock: ${clk:-<unread>} MHz"
  say ""
  say "   *********************************************************************"
  say "   * SKU. The paper says 'n150 Blackhole'. n150 is a WORMHOLE part."
  say "   * Read sku.txt's tt-smi board type before quoting any SKU name. If"
  say "   * tt-smi is not installed, quote the part by what was MEASURED --"
  say "   * '<workers>-Tensix Blackhole at <clk> MHz' -- and not by a SKU."
  say "   *********************************************************************"
  say ""
  if [ -z "$workers" ] || [ "$workers" -lt 2 ] 2>/dev/null; then
    verdict sku SUSPECT "the grid dump has ${workers:-no} worker rows; nothing here identifies the part"
  elif grep -qiE 'p1[05]0|n[13]50|board' "$f" 2>/dev/null; then
    verdict sku MEANINGFUL "the part named itself; sku.txt carries a board type AND $workers workers"
  else
    verdict sku COLLECTED "$workers workers at ${clk:-?} MHz recorded, but no tt-smi board type -- quote the measurement, not a SKU name"
  fi
}

step_ladder() {
  : > "$OUT/ladder.csv"
  echo "rung,rc,wall_s,flavour" >> "$OUT/ladder.csv"
  local row
  for row in "${LADDER[@]}"; do run_rung "$row" ""; done
}

step_ladder_prof() {
  # The device profiler is tt-metal's OWN instrument and needs no source change:
  # every kernel built with PROFILE_KERNEL=1 emits the firmware's BRISC-KERNEL /
  # TRISC-KERNEL zones. It works on silicon today. It does NOT work against
  # Wolfpine today -- the instrumented firmware reads NIU register offset 0x14,
  # which framework/network/tt_noc.py does not implement, and the server dies. That
  # is a bounded fix; this step banks the HARDWARE half now so the fix does not
  # need a second card booking to be usable.
  : > "$OUT/ladder.prof.csv"
  echo "rung,rc,wall_s,flavour" >> "$OUT/ladder.prof.csv"
  local row
  for row in "${LADDER[@]}"; do run_rung "$row" prof; done
  if [ "$DO_DRY" = 1 ]; then return; fi
  local n
  n="$(find "$OUT" -name 'profile_log_device.csv' 2>/dev/null | wc -l)"
  say "   profile_log_device.csv files written: $n"
  if [ "${n:-0}" -eq 0 ]; then
    verdict ladder-prof FAILED "the profiler produced no device CSV. Without it the e2e cycle table has no hardware column; check that tt-metal was built with ENABLE_TRACY=ON"
  else
    verdict ladder-prof MEANINGFUL "$n device profile CSV(s); each holds per-core zone timestamps in DEVICE CYCLES"
  fi
}

step_optests() {
  if [ -z "$OPTESTS" ] || [ ! -d "$OPTESTS" ]; then
    skip optests "no optests/ tree. It ships separately from perfbench/: rsync -av --exclude 'build/' optests/ <card-box>:~/optests/  (then re-run with --optests ~/optests)"
    return
  fi
  local csv="$OUT/optests.$ARCH.csv" n=0 ok=0 d name log rc
  if [ "$DO_DRY" = 1 ]; then
    echo "   + for each $OPTESTS/*/src: cmake build, run, capture OPDIFF_RESULT -> $csv"
    return
  fi
  echo "op,rc,hexlen,result" > "$csv"
  for d in "$OPTESTS"/*/; do
    name="$(basename "$d")"
    [ -d "$d/src" ] || continue
    n=$((n + 1))
    if ! bp_check_build "$d/src" "$([ -x "$d/src/build/$name" ] && echo skip-build || echo build)"; then
      printf '%s,stale-build-tree,0,\n' "$name" >> "$csv"; continue
    fi
    if [ ! -x "$d/src/build/$name" ]; then
      ( cd "$d/src" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 \
          && cmake --build build -j >/dev/null 2>&1 ) || {
        printf '%s,build-failed,0,\n' "$name" >> "$csv"; continue; }
      bp_record_build "$d/src"
    fi
    log="$OUT/optest.$name.out"
    # A per-program env file (e.g. optests/where needs TT_METAL_DISABLE_SFPLOADMACRO)
    # is sourced exactly as optests/diff.sh does it, so the card runs the same
    # configuration the local differential does.
    ( cd "$d/src" || exit 2
      # shellcheck disable=SC1090
      [ -f "$d/env" ] && . "$d/env"
      if [ "$DO_SIM" = 1 ]; then export TT_SIM_TENSIX_COORDS="${TT_SIM_TENSIX_COORDS:-$SIM_COORDS}"; fi
      timeout 900 "./build/$name" ) >"$log" 2>&1
    rc=$?
    local hex; hex="$(grep -o 'OPDIFF_RESULT:[0-9a-f]*' "$log" | head -1 | cut -d: -f2)"
    printf '%s,%s,%s,%s\n' "$name" "$rc" "${#hex}" "$hex" >> "$csv"
    [ -n "$hex" ] && ok=$((ok + 1))
    say "   $name: rc=$rc  $(( ${#hex} / 8 )) elements"
  done
  say "   $ok of $n optests programs produced an OPDIFF_RESULT"
  if [ "$ok" -eq 0 ]; then
    verdict optests FAILED "no program produced an OPDIFF_RESULT; the hex dump is the whole point of this step"
  elif [ "$ok" -lt "$n" ]; then
    verdict optests SUSPECT "$ok of $n produced a result; the rest have no hardware arm and their rows must say so, not be dropped"
  else
    verdict optests MEANINGFUL "all $n programs produced a hex dump; the three-way compare can be done at home"
  fi
}

step_obs() {
  # The capability matrix's hardware column, RUN rather than cited. Two facts
  # are worth a card: what the watcher actually prints, and whether DPRINT
  # reaches the host under slow dispatch. Both are claims the paper makes about
  # hardware; neither should be taken from a README.
  #
  # THE PROGRAM MUST BE ONE THAT PASSES ON THIS PART. Until 2026-08-10 this
  # step ran eltwise_binary, which is precisely the rung that fails on the one
  # available Blackhole -- so both invocations aborted and the step banked 4.3 MB
  # of crash trace twice while measuring nothing. A introspection facility
  # demonstrated on a program that dies tells you about the death, not the
  # facility. The default is therefore the first LADDER RUNG KNOWN TO PASS here,
  # and --obs-prog overrides it.
  #
  # Note what this can and cannot reach. DPRINT from a data-movement kernel
  # (BRISC/NCRISC) is fully demonstrated. DPRINT from a COMPUTE kernel is not
  # reachable on this part at all, because every compute rung currently returns
  # wrong results -- say so in the matrix rather than leaving the cell blank.
  local prog="${OBS_PROG:-}"
  if [ -z "$prog" ]; then
    for cand in metal_example_loopback metal_example_add_2_integers_in_riscv; do
      [ -x "$PE/$cand" ] && { prog="$cand"; break; }
    done
  fi
  if [ "$DO_DRY" = 1 ]; then
    echo "   + TT_METAL_WATCHER=10 $prog; TT_METAL_DPRINT_CORES=0,0 $prog"
    return
  fi
  [ -n "$prog" ] && [ -x "$PE/$prog" ] || {
    skip obs "no passing programming example found in $PE (tried loopback, add_2_integers_in_riscv); pass --obs-prog"; return; }
  say "   using $prog (a rung that PASSES on this part)"
  ( cd "$PE" && TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
      timeout 600 "./$prog" ) >"$OUT/obs-watcher.out" 2>&1
  local wrc=$?
  ( cd "$PE" && TT_METAL_DPRINT_CORES=0,0 TT_METAL_DPRINT_RISCVS=BR \
      timeout 600 "./$prog" ) >"$OUT/obs-dprint.out" 2>&1
  local prc=$?
  # The watcher ANNOUNCES its log path on stdout ("Watcher log file: ..."), so
  # take it from there rather than guessing the tree layout. Guessing cost a
  # card run on 2026-08-10: the log was under
  # build_Release/programming_examples/generated/watcher/, not $TT_METAL_HOME/
  # generated, so the step reported watcher=0 while the watcher had in fact
  # attached both devices and written a perfectly good log.
  local wlog
  wlog="$(grep -oE 'Watcher log file: [^ ]+' "$OUT/obs-watcher.out" | tail -1 | cut -d' ' -f4)"
  if [ -n "$wlog" ] && [ -s "$wlog" ]; then
    cp "$wlog" "$OUT/watcher.log" 2>/dev/null
  else
    # Fall back to a search, widened to the whole tree.
    find "$TT_METAL_HOME" -name 'watcher.log' -newermt '-10 minutes' \
      -exec cp {} "$OUT/watcher.log" \; 2>/dev/null
  fi
  local w=0 p=0
  [ -s "$OUT/watcher.log" ] && w=1
  grep -qiE 'DPRINT|Device [0-9]+, Core' "$OUT/obs-dprint.out" 2>/dev/null && p=1
  say "   $prog: watcher rc=$wrc log=$w    dprint rc=$prc output=$p"
  if [ "$wrc" != 0 ] || [ "$prc" != 0 ]; then
    verdict obs SUSPECT "$prog did not exit 0 under introspection (watcher rc=$wrc, dprint rc=$prc). A facility demonstrated on a failing program measures the failure, not the facility -- pick another with --obs-prog"
  elif [ "$w" = 1 ] || [ "$p" = 1 ]; then
    verdict obs COLLECTED "watcher=$w dprint=$p on $prog -- fill the matrix's HW column from these files, not from documentation. DPRINT here is from a DATA-MOVEMENT kernel; the compute-kernel case is unreachable on this part"
  else
    verdict obs SUSPECT "$prog ran clean but neither the watcher log nor DPRINT output appeared; the HW column cannot be asserted from this run"
  fi
}

# --------------------------------------------------------------------- driver
applies() { # step name -> 0 if it runs on this part
  local parts; parts="$(step_field "$1" 2)"
  [ "$parts" = both ] && return 0
  [ "$parts" = "$ARCH" ] && return 0
  return 1
}

done_already() {
  [ "$DO_RESUME" = 1 ] || return 1
  case "$1" in
    sku) [ -s "$OUT/sku.txt" ] ;;
    ladder) [ -s "$OUT/ladder.csv" ] ;;
    ladder-prof) [ -s "$OUT/ladder.prof.csv" ] ;;
    optests) ls "$OUT"/optests.*.csv >/dev/null 2>&1 ;;
    obs) [ -s "$OUT/obs-watcher.out" ] ;;
    *) return 1 ;;
  esac
}

SIM_NOTE=""
[ "$DO_SIM" = 1 ] && SIM_NOTE=" (SIM SMOKE -- NOT A MEASUREMENT)"

if [ ${#WANT[@]} -eq 0 ]; then
  while IFS= read -r n; do WANT+=("$n"); done < <(step_names)
fi

say "== paper session: arch=$ARCH out=$OUT$SIM_NOTE"
say "   started $(date -Is)"
if [ -n "$GRID" ]; then
  say "   grid override for multi-core rungs: TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=$GRID"
else
  say "   grid override: NONE (--no-grid). On the harvested card this makes"
  say "   the multi-core rungs die in tt-metal with YAML::TypedBadConversion."
fi
say "   optests tree: ${OPTESTS:-<not found>}"
say ""

for name in "${WANT[@]}"; do
  if ! applies "$name"; then
    skip "$name" "needs a $(step_field "$name" 2) part; this is $ARCH"
    continue
  fi
  if done_already "$name"; then skip "$name" "--resume: output already in $OUT"; continue; fi
  say "== $name  ($(step_field "$name" 3) min)  $(step_field "$name" 4)"
  if [ "$DO_DRY" = 1 ]; then echo "   + step_${name//-/_}"; fi
  "step_${name//-/_}"
  say ""
done

[ "$DO_DRY" = 1 ] && exit 0

# ------------------------------------------------------------------- handover
bad=0
for v in ${VERDICTS[@]+"${VERDICTS[@]}"}; do
  rest="${v#*|}"
  case "${rest%%|*}" in
    FAILED|SUSPECT|UNCLEAR) bad=$((bad + 1)) ;;
    TIMEOUT) [ "$DO_SIM" = 1 ] || bad=$((bad + 1)) ;;
  esac
done

{
  echo
  echo "============ PAPER SESSION SUMMARY ($ARCH)$SIM_NOTE ============"
  for v in ${VERDICTS[@]+"${VERDICTS[@]}"}; do
    n="${v%%|*}"; rest="${v#*|}"; s="${rest%%|*}"; t="${rest#*|}"
    printf '%-22s %-12s %s\n' "$n" "$s" "$t"
  done
  echo
  echo "STATUSES: MEANINGFUL  the step produced the artefact it exists for."
  echo "          PASS        a ladder rung ran and self-checked clean."
  echo "          COLLECTED   files written; the analysis box decides what they say."
  echo "          TIMEOUT     no result inside the budget. On a CARD that is a"
  echo "                      broken run: silicon is not slow. Investigate."
  echo "          SKIPPED     did not apply, or a dependency was missing."
  echo "          FAILED / SUSPECT / UNCLEAR  needs a look before you leave."
  echo
  echo "BEFORE YOU SEND ANYTHING BACK:"
  echo "  1. No step should read FAILED, SUSPECT, UNCLEAR or TIMEOUT."
  echo "  2. sku.txt: if it carries no tt-smi board type, the paper must quote"
  echo "     the part by measurement (workers x AICLK), not by a SKU name. The"
  echo "     draft's 'n150 Blackhole' is wrong twice over if so: n150 is a"
  echo "     WORMHOLE SKU."
  echo "  3. optests.$ARCH.csv: a zero-length hex is a program that produced no"
  echo "     result, not a program that agreed with anything."
  echo "  4. rung.*.out: read rung 5's PCC. A matmul that 'passed' with a PCC"
  echo "     the log does not print has not been checked."
  echo "  5. Every number here is CORROBORATION of the simulator, never"
  echo "     provenance for a cost-model entry. Do not bank any of it into"
  echo "     framework/perf/datasets/ without the full contract header"
  echo "     (device=, firmware_bundle=19.6.0, kmd=2.10.0, flags, ONE RUN ON"
  echo "     ONE CARD, a valid line)."
  echo "  6. Send the WHOLE directory back, from the analysis box:"
  echo "       rsync -av <card-box>:$OUT/ ./paper-session-$ARCH/"
  echo "     ($(ls "$OUT" 2>/dev/null | wc -l) entries: the CSVs, the per-rung .out logs,"
  echo "      the profiler directories, session.log)"
  echo
  echo "STILL OPEN AFTER A CLEAN RUN -- awaiting the planned Wormhole follow-on:"
  for d in "${DEFERRED_TO_WORMHOLE[@]}"; do
    printf '  - %s\n' "$d" | fold -s -w 74 | sed '2,$s/^/    /'
  done
  echo
  echo "  finished $(date -Is)"
  [ "$bad" -gt 0 ] && echo "  ($bad step(s) need a look)"
  true
} | tee -a "$LOG"

[ "$bad" -eq 0 ]
