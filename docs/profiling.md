# Profiling and analysing kernels on Wolfpine

This is the guide to every way of getting performance and behavioural insight
out of a tt-metal program running on Wolfpine. For each mode it covers what to
install, how to turn the mode on, and the commands that turn its output into an
answer.

**It applies to Wormhole and Blackhole alike.** Tracing is wired into the
architecture-agnostic device base and the shared wire bridge, so every variable
below works the same on both. The few real differences are coordinates and
which hardware counters exist; they are collected in
[§7](#7-wormhole-and-blackhole-differences). Every command in this guide was
run against both architectures on tt-metal 0.74.

> **Read the numbers as a floor.** Wolfpine is not cycle-accurate. Every
> modelled cycle is charged at the low end of a published bound, so use the
> output for *relative* attribution ("this loop dominates", "the unpacker
> waits on the NoC"), not as a cycle promise. The
> [timing-model statement](simulator-overview.md#what-the-timing-model-claims-and-what-it-does-not)
> states what the model claims, and
> [`cost-model-caveats-for-consumers.md`](cost-model-caveats-for-consumers.md)
> lists where it is known to be wrong.

## Contents

1. [Before you start](#1-before-you-start) — what to install, how settings reach the simulator, the cost model
2. [Which mode answers my question?](#2-which-mode-answers-my-question)
3. [The ranked report — `TT_SIM_PROFILE`](#3-the-ranked-report--tt_sim_profile) — start here
4. [Wolfpine's trace writers](#4-wolfpines-trace-writers) — timeline, counters, NoC, raw events, memory, coverage, commitlog, invariants, state dumps
5. [tt-metal's own instrumentation, on the simulator](#5-tt-metals-own-instrumentation-on-the-simulator) — device profiler zones, NoC events, perf counters, DPRINT
6. [Human-readable diagnostics and wire capture](#6-human-readable-diagnostics-and-wire-capture)
7. [Wormhole and Blackhole differences](#7-wormhole-and-blackhole-differences)
8. [Recipes](#8-recipes)
9. [Further reading](#9-further-reading)

---

## 1. Before you start

### 1.1 What to install

You need the base setup for running anything, plus only the tools for the
modes you will use.

| For | Install | Notes |
| --- | --- | --- |
| **Everything** | `pip install wolfpine` (or `pip install -e .` from a checkout) | Brings numpy, pyarrow, pyelftools and the rest. The tracing code is always present and costs nothing while it is off. |
| **Everything** | A built [tt-metal](https://github.com/tenstorrent/tt-metal) checkout | `wolfpine config --tt-metal /path/to/tt-metal`, then check it with `wolfpine doctor`. See the [README quick start](../README.md#quick-start). |
| Timeline (§4.1) | Nothing | Open <https://ui.perfetto.dev> in a browser. The trace is loaded locally and never uploaded. |
| Parquet counters / NoC (§4.2, §4.3), zone CSVs (§5.1) | DuckDB: `pip install duckdb`, or the [DuckDB CLI](https://duckdb.org/docs/installation/) | Or pandas/Polars (`pip install pandas`). pyarrow is already installed with Wolfpine. |
| Raw events (§4.4) | Nothing (Python), or `jq` | `apt install jq` / `brew install jq` |
| Memory hotspots (§4.5) | KCachegrind / QCachegrind | `apt install kcachegrind`, or `brew install qcachegrind` on macOS. `callgrind_annotate` (from `valgrind`) works for text output. |
| Source coverage (§4.6) | `lcov` for `genhtml` | `apt install lcov` / `brew install lcov`. Or the VS Code [Coverage Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) extension. |
| Commitlog diffing (§4.7) | [Spike](https://github.com/riscv-software-src/riscv-isa-sim) | Only for RISC-V differential testing. Build it from source; some distributions package it as `riscv-isa-sim`. |
| tt-metal device profiler (§5.1, §5.2) | Nothing extra | Compiled into tt-metal by default (`ENABLE_TRACY=ON`). A tt-metal built with `./build_metal.sh --disable-profiler` has nothing to turn on. |

### 1.2 How the settings reach the simulator

Every mode is an environment variable set in the shell that launches your
tt-metal program. UMD spawns the simulator as a child process, so the variable
reaches it with no code change, no tt-metal change and no rebuild. Both launch
styles work:

```bash
# With the wolfpine command: the TT_SIM_* variables pass straight through.
TT_SIM_TRACE_PERFETTO=/tmp/run/timeline.json.gz wolfpine run blackhole ./your_program

# Or switch the current shell to the target, then run the binary directly. (To export
# the variables by hand, see the runbook, section 1.1.)
eval "$(wolfpine env wormhole)"
TT_SIM_TRACE_PERFETTO=/tmp/run/timeline.json.gz ./your_program
```

The rest of this guide writes `wolfpine run <arch> ./your_program`. Substitute
`wormhole` or `blackhole`; nothing else changes.

**Four rules that save a confusing afternoon:**

- **Use a fresh output path per run.** Writers overwrite their outputs without
  asking.
- **Outputs are finalised when the simulator exits**, which can be a second or
  two after your program returns. A script that reads them should wait for the
  server to go, or retry. `python3 -m framework.trace.report` already does.
- **Run profiled runs one at a time.** `TT_SIM_PROFILE` works out which kernel
  ELF ran by searching tt-metal's shared build cache. When two runs JIT-compile
  into that cache at the same moment, every unit can fall back to an unverified
  guess (`recent`), and kernel names can come from the *other* run. Run alone,
  the same programs verify every ELF. Concurrency is safe for everything else in
  this guide.
- **The in-repo examples must be run from their `src/` directory.** Their
  kernel paths are relative, and from anywhere else tt-metal aborts in
  `CreateKernel`. For example: `cd examples/four/src && ./build/four`.

### 1.3 Turn the cost model on

```bash
export TT_SIM_COST_MODEL=1
```

Without it every op retires in the cycle it was issued and nothing stalls, so
there is nothing to rank: RV stall counters and backend occupancy are *absent*,
not zero, and every Perfetto slice is one cycle wide. That run is still useful
for correctness and event counts, but whenever the question is "where did the
time go", set the cost model. Every artefact records which regime produced it;
[`trace-schema.md` §3.4](trace-schema.md#34-the-two-timing-regimes) has the
details.

### 1.4 Pick a small, pinned run where you can

Wall time scales with **simulated cycles × materialised Tensix tiles**. With
every writer on, the single-tile `examples/one` takes about 9 s on either
architecture. A 16-core GEMM can take hours. Shrink the problem to the
smallest size that still shows the behaviour you care about. Pin the worker set
with `TT_SIM_TENSIX_COORDS` whenever a mode would otherwise build the whole
grid: the device profiler and DPRINT both do, because their readback touches
every core
([runbook §1.3](running-tt-metal-on-the-simulator.md#13-multiple-tensix-tiles)).

---

## 2. Which mode answers my question?

| Question | Mode | Section |
| --- | --- | --- |
| "Where did my kernel's cycles go?" | `TT_SIM_PROFILE` | [§3](#3-the-ranked-report--tt_sim_profile) |
| "Which function / source line is hot?" | `TT_SIM_PROFILE` (`hotspots.json`) | [§3.2](#32-source-attribution) |
| "What ran when — do unpack, math and pack overlap?" | Perfetto timeline | [§4.1](#41-timeline--tt_sim_trace_perfetto) |
| "How many of each Tensix op, stall and NoC byte?" | Counter Parquet | [§4.2](#42-performance-counters--tt_sim_trace_counters) |
| "Which link or DRAM channel carried the bytes, and how long did each transaction take?" | NoC Parquet | [§4.3](#43-noc-transactions--tt_sim_trace_noc) |
| "Is this phase issue-bound or transfer-bound, per core?" | tt-metal NoC events + `noc_events` | [§5.2](#52-noc-event-trace--tt_metal_device_profiler_noc_events) |
| "How long did each kernel zone take?" | tt-metal device profiler | [§5.1](#51-device-profiler-zones--tt_metal_device_profiler) |
| "What do the hardware perf counters say?" | `TT_METAL_PROFILE_PERF_COUNTERS` | [§5.3](#53-tensix-performance-counters--tt_metal_profile_perf_counters) |
| "Which instruction touches memory most?" | Callgrind memory trace | [§4.5](#45-memory-hotspots--tt_sim_trace_memory) |
| "Print a value from inside my kernel" | DPRINT | [§5.4](#54-kernel-printf--tt_metal_dprint_cores) |
| "Why is it hanging / which instruction went wrong?" | `TT_SIM_DIAG_*`, `TT_SIM_LOG_PROTOCOL` | [§6](#6-human-readable-diagnostics-and-wire-capture) |
| "Is the RISC-V model right?" | Commitlog + Spike | [§4.7](#47-risc-v-commitlog--tt_sim_trace_commitlog) |
| "Something none of the above expresses" | Raw JSONL events | [§4.4](#44-raw-event-stream--tt_sim_trace) |

The `TT_SIM_*` modes (§3, §4) are Wolfpine's own and see *inside* the model:
per-unit occupancy, stall reasons, per-hop NoC latency. The `TT_METAL_*` modes
(§5) are tt-metal's, and they produce **the same artefacts a real card does**.
That makes them the way to compare simulator against silicon, and the way to
reuse tooling you already have. The two families compose freely.

---

## 3. The ranked report — `TT_SIM_PROFILE`

**Start here.** One variable switches on the right collectors and leaves a
ranked, explained report behind.

```bash
export TT_SIM_COST_MODEL=1
TT_SIM_PROFILE=/tmp/myrun wolfpine run blackhole ./your_program
```

When the simulator exits:

| File | What it is |
| --- | --- |
| `report.md` | Human-readable ranked report. **Read this.** Its layout is not frozen, so do not parse it. |
| `report.json` | The same numbers, machine-readable and versioned (`schema_version`). |
| `hotspots.json` | Per-PC and per-function cycle attribution, with source file and line. |
| `profile.json` | Run metadata: cost-model regime, and which ELF was used for each unit and how it was chosen. |
| `counters/` | The raw counter Parquet dataset ([§4.2](#42-performance-counters--tt_sim_trace_counters)). |

The report opens by saying what the numbers are and are not. It then shows:

- the shared resources the model can price (NoC flight, Tensix backend
  occupancy);
- a per-transaction NoC latency split into issue → injection → arrival →
  service;
- every cycle-bearing counter ranked by its share of the run;
- the source functions the cycles were spent in;
- an **attribution coverage** table saying how much of the run the model could
  not name.

Nothing is quietly dropped.

Re-render at any time without re-running the simulator:

```bash
python3 -m framework.trace.report /tmp/myrun --top 40       # rewrite report.md / report.json
python3 -m framework.trace.report /tmp/myrun --stdout | less
```

### 3.1 Reading it

- **Shares do not sum to 100 %.** Units run concurrently, so each row is a
  share of the whole span. A core can be *charged* more stall cycles than the
  run is long, because a charge is not a delivered cycle: the span is set by
  when cores wait for each other.
- **The `unattributed` column is the size of the model's gap, not idle time.**
  Anything the cost model does not price contributes nothing rather than a
  guess.
- For "how much of *this core's* span went to the NoC", use the NoC event
  decomposition ([§5.2](#52-noc-event-trace--tt_metal_device_profiler_noc_events)).
  It partitions one core's elapsed time and is validated against silicon. The
  report's NoC row is an aggregate occupancy across links.

### 3.2 Source attribution

Function and line names come from the DWARF in the ELFs tt-metal builds. A
default build already has it, under `~/.cache/tt-metal-cache/<build-hash>/`
(override the location with `TT_METAL_CACHE`). Wolfpine finds the ELFs itself
and then *proves* each choice by comparing the ELF's bytes with what is
resident in simulated L1. Kernels are relocated at load, and the comparison
recovers that load bias too.

`profile.json`'s `elfs[].how` says how each unit's ELF was chosen:

| `how` | Meaning |
| --- | --- |
| `verified` | Byte-for-byte match against resident memory (firmware). |
| `relocated +0x…` | The same proof for a kernel, plus the recovered load offset. |
| `no match` | The device was readable and no candidate matches, so the unit is deliberately left unattributed. This is correct for a core that ran no kernel. |
| `recent` | The newest candidate by mtime, **unconfirmed**. Treat its names as a guess, and see the concurrency rule in §1.2. |
| `explicit` | You named the ELF yourself (below). |

To override discovery:

```bash
export TT_SIM_PROFILE_ELFS="BRISC:/path/brisc.elf,TRISC2:/path/trisc2.elf@0x8b00"
```

Each entry is `<UNIT>:<path>`, comma-separated, with an optional `@<bias>`
added to every address in that ELF.

Other knobs:

- `TT_SIM_PROFILE_LABEL` names the run in the report title.
- `TT_SIM_PROFILE_INTERVAL` sets the counter flush cadence in cycles (default 100).

**Replaying a captured wire trace** attributes firmware but often not kernels:
the kernel ELF that produced the recorded bytes may be gone from the cache, or
may have been built for the other architecture.

The full field-by-field contract for `report.json` and `hotspots.json` is
[`trace-schema.md` §9](trace-schema.md#9-the-ranked-report-and-its-json). Build
tooling on those two files, never on `report.md`.

---

## 4. Wolfpine's trace writers

Each writer is an independent subscriber to the simulator's event bus, so any
combination can be on at once ([§8.1](#81-everything-on)). Each output is
specified field by field in [`trace-schema.md`](trace-schema.md), which is the
stable, versioned contract to build tools on. This section is the how-to.

| Variable | Output | Size for `examples/one` |
| --- | --- | --- |
| `TT_SIM_TRACE_PERFETTO=<file.json.gz>` | Chrome Trace / Perfetto timeline | ~120 KB |
| `TT_SIM_TRACE_COUNTERS=<dir>` | Counter Parquet dataset | ~30 KB |
| `TT_SIM_TRACE_NOC=<dir>` | NoC transaction Parquet dataset | ~16 KB |
| `TT_SIM_TRACE=<file.jsonl>` | Raw event stream, plus `.ids.json` sidecar | ~9–11 MB |
| `TT_SIM_TRACE_MEMORY=<file>` | Callgrind memory trace | ~100 KB |
| `TT_SIM_TRACE_LCOV=<file>` + `TT_SIM_TRACE_LCOV_ELFS` | LCOV line counts | — |
| `TT_SIM_TRACE_COMMITLOG=<dir>` | One Spike-format commitlog per baby core | ~800 KB |
| `TT_SIM_TRACE_INVARIANTS=<file>` | Architectural-rule violations, JSONL | empty when clean |
| `TT_SIM_TRACE_STATE_DUMP=<dir>` | Device state snapshots | see §4.9 |

Every output keys a unit the same way: `(chip, core_y, core_x, unit)`, where
`unit` is `BRISC`, `NCRISC`, `TRISC0`–`2`, `NOC0`/`NOC1`, a Tensix backend unit
(`MATRIX`, `SFPU`, `PACKER`, …) or `UNKNOWN`. Coordinates are
architecture-specific, so never join a Wormhole dataset with a Blackhole one
([§7](#7-wormhole-and-blackhole-differences)).

### 4.1 Timeline — `TT_SIM_TRACE_PERFETTO`

**What:** a zoomable timeline with one process per tile and one track per unit
(the baby cores, `MATRIX`, `SFPU`, `PACKER`, `UNPACKER`, `MOVER`, `THCON`,
`SYNC`, `NOC0`/`NOC1`). A `<thread> stall` track sits beside each Tensix
thread. NoC transactions are async slices with request → response arrows.

**Why:** the fastest way to see what the kernel actually does over time. It
answers questions like "is the SFPU idle while the unpacker waits?" or "did
pack overlap math?".

```bash
TT_SIM_COST_MODEL=1 TT_SIM_TRACE_PERFETTO=/tmp/run/timeline.json.gz \
    wolfpine run wormhole ./your_program
```

1. Open <https://ui.perfetto.dev> and drag the file onto the page. A `.gz`
   extension turns on compression, which Perfetto reads natively.
2. **Time axis: one trace microsecond is one simulated cycle.** What the UI
   labels "µs" is a cycle.
3. For SQL, open **Query (SQL)** in the sidebar. Canned queries (top slices by
   duration, per-unit event counts, NoC round-trip latency) are in
   [`framework/trace/queries/README.md`](../framework/trace/queries/README.md),
   and the table layout is documented at
   <https://perfetto.dev/docs/analysis/sql-tables>.
4. A NoC transaction's request → response arrow is drawn between the two
   tiles' NoC tracks. Click a slice to see its arguments, including
   `timing_model` wherever a width was not modelled.

Slice widths are modelled cycles with the cost model on. With it off, every
width is 1, and that is the truth rather than a placeholder. The regime is
stamped in the trace (**Info and stats** → `otherData.cost_model`). Memory
events are left out for volume; use §4.5 for those.

### 4.2 Performance counters — `TT_SIM_TRACE_COUNTERS`

**What:** a Hive-partitioned Parquet dataset (`chip=<N>/kernel_id=<N>/*.parquet`)
in long format, with one row per `(cycle, unit, counter_name)`. `value` is the
**increment since the previous flush**, so `SUM(value)` is the run total. Never
take `MAX`.

**Why:** utilisation, stall reasons, op counts and byte counts per unit, in one
SQL query.

```bash
TT_SIM_COST_MODEL=1 TT_SIM_TRACE_COUNTERS=/tmp/run/counters/ \
    wolfpine run blackhole ./your_program
# Optional flush cadence in cycles (default 100):
#   TT_SIM_TRACE_COUNTERS_INTERVAL=50
```

```bash
# Top counters across the run
duckdb -c "
  SELECT counter_name, SUM(value) AS total
  FROM read_parquet('/tmp/run/counters/**/*.parquet', hive_partitioning=true)
  GROUP BY counter_name ORDER BY total DESC LIMIT 15"

# Where each core's RV stalls went, by reason (stall_cycles is their total, so leave it out)
duckdb -c "
  SELECT core_x, core_y, unit, counter_name, SUM(value) AS cycles
  FROM read_parquet('/tmp/run/counters/**/*.parquet', hive_partitioning=true)
  WHERE starts_with(counter_name, 'stall_') AND counter_name <> 'stall_cycles'
  GROUP BY ALL ORDER BY cycles DESC"
```

From Python (`pip install duckdb`, or pandas):

```python
import duckdb
duckdb.sql("""SELECT unit, counter_name, SUM(value) AS total
              FROM read_parquet('/tmp/run/counters/**/*.parquet', hive_partitioning=true)
              GROUP BY ALL ORDER BY total DESC""").show()
```

More canned queries are in
[`framework/trace/queries/counters.sql`](../framework/trace/queries/counters.sql).
They read the relative path `counters/`, so run them from the directory that
contains it: `cd /tmp/run && duckdb -c ".read /path/to/wolfpine/framework/trace/queries/counters.sql"`.

**Traps worth knowing**, each covered in detail in
[`trace-schema.md` §4 and §7](trace-schema.md#4-performance-counters-parquet--the-main-dataset):

- **The counter set is open.** Rank by pattern (`stall_*`, `tensix_stall_*`,
  `*_cycles`), not by a fixed list.
- **Do not sum a total with its parts.** For example, `stall_cycles` already
  includes `stall_load_use`.
- **`cycle` is a flush boundary, not an event time.** Memory counters are
  reliable as run totals but not as a time series.
- **`kernel_id` is always `0` today.** Nothing publishes kernel-boundary events
  in the tt-metal flow, so a multi-launch run is one partition. For per-launch
  spans use the device profiler's zones (§5.1).
- **The dataset does not record the cost-model regime.** Keep `profile.json`
  or note `TT_SIM_COST_MODEL` alongside it.

### 4.3 NoC transactions — `TT_SIM_TRACE_NOC`

**What:** a Parquet dataset (`chip=<N>/*.parquet`) with one row per NoC event.
Columns:

- `cycle`, `core_x`, `core_y`, `unit`
- `phase` (`request`/`response`), `txn_type` (`read`/`write`)
- `src_x`, `src_y`, `dst_x`, `dst_y`, `size_bytes`, `txn_id`
- `issue_cycle`, `arrival_cycle`, `flight_cycles`
- the three latency legs: `issue_to_injection_cycles`,
  `injection_to_arrival_cycles`, `arrival_to_service_cycles`
- `cost_model`

**Why:** which pairs of tiles exchanged the bytes, how long each transaction
spent queued, in flight and at the endpoint, and whether DRAM was the
bottleneck.

```bash
TT_SIM_COST_MODEL=1 TT_SIM_TRACE_NOC=/tmp/run/noc/ wolfpine run wormhole ./your_program
```

```bash
# Bytes and transactions per (source, destination)
duckdb -c "
  SELECT src_x, src_y, dst_x, dst_y, txn_type, SUM(size_bytes) AS bytes, COUNT(*) AS n
  FROM read_parquet('/tmp/run/noc/**/*.parquet', hive_partitioning=true)
  WHERE phase = 'response'
  GROUP BY ALL ORDER BY bytes DESC"

# Where the latency went, per leg
duckdb -c "
  SELECT txn_type, COUNT(*) AS n,
         AVG(issue_to_injection_cycles)   AS queued,
         AVG(injection_to_arrival_cycles) AS transit,
         AVG(arrival_to_service_cycles)   AS at_endpoint
  FROM read_parquet('/tmp/run/noc/**/*.parquet', hive_partitioning=true)
  WHERE phase = 'response' GROUP BY txn_type"
```

`arrival_to_service_cycles` is charged only at a DRAM channel; it is zero at
every other endpoint by construction. See
[`trace-schema.md` §4.4a and §5](trace-schema.md#5-noc-transactions-parquet).

### 4.4 Raw event stream — `TT_SIM_TRACE`

**What:** every event the simulator publishes, one JSON object per line. The
categories are `instr`, `stall`, `dispatch`, `compute`, `noc`, `mem`, `sync`
and `counter`. A sidecar `<file>.ids.json` decodes each `unit_id`. Every other
writer is derived from this same stream.

**Why:** ad-hoc questions none of the structured outputs answer. Expect about
10 MB for even the smallest program, and size runs accordingly.

```bash
TT_SIM_TRACE=/tmp/run/events.jsonl wolfpine run blackhole ./your_program
```

```bash
# Events per category
jq -r .category /tmp/run/events.jsonl | sort | uniq -c

# Hottest PCs on BRISC
python3 -c "
import json, collections
c = collections.Counter(e['pc'] for e in map(json.loads, open('/tmp/run/events.jsonl'))
                        if e['category'] == 'instr' and e['unit_id'][3] == 'BRISC')
print([(hex(pc), n) for pc, n in c.most_common(10)])"

# Or SQL straight over the JSONL
duckdb -c "SELECT category, COUNT(*) FROM read_json_auto('/tmp/run/events.jsonl') GROUP BY 1"
```

Every `pc` is a **runtime** address. Kernels are relocated at load, so convert
with the `bias` from `profile.json` before looking a PC up in an ELF
([`trace-schema.md` §7.2](trace-schema.md#72-kernel-pcs-are-relocated)).

### 4.5 Memory hotspots — `TT_SIM_TRACE_MEMORY`

**What:** a Callgrind-format file with `Dr`/`Dw` (data read/write) counts
bucketed by region, PC and address. Accesses with no PC (NoC-driven or internal
engines) are grouped under a `<region>_no_pc` function.

**Why:** a call-cost-tree view of which instructions touch L1 and MMIO hardest.

```bash
TT_SIM_TRACE_MEMORY=/tmp/run/memory.callgrind wolfpine run wormhole ./your_program

kcachegrind /tmp/run/memory.callgrind            # or qcachegrind on macOS
callgrind_annotate /tmp/run/memory.callgrind     # text, from valgrind
```

The functions are named `<region>_pc_0x<pc>`, not symbol names. Map a hot PC to
source with `hotspots.json` (§3.2).

### 4.6 Source coverage — `TT_SIM_TRACE_LCOV`

**What:** an LCOV file where each line's count is the instructions retired at
that line. It works with `genhtml`, Codecov and editor coverage gutters.

```bash
FW=~/.cache/tt-metal-cache/<build-hash>/firmware
TT_SIM_TRACE_LCOV=/tmp/run/coverage.lcov \
TT_SIM_TRACE_LCOV_ELFS=$FW/brisc/brisc.elf,$FW/ncrisc/ncrisc.elf,$FW/trisc0/trisc0.elf,$FW/trisc1/trisc1.elf,$FW/trisc2/trisc2.elf \
    wolfpine run blackhole ./your_program

genhtml /tmp/run/coverage.lcov -o /tmp/run/cov_html && xdg-open /tmp/run/cov_html/index.html
```

**Limitation: this attributes firmware, not your kernels.**
`TT_SIM_TRACE_LCOV_ELFS` loads each ELF at its link address. tt-metal relocates
kernels at load, so kernel PCs never match their ELF and kernel source files do
not appear. Firmware is not relocated and attributes correctly. **For
kernel-level source attribution, use `TT_SIM_PROFILE`'s `hotspots.json`
(§3.2)**, which recovers the load bias per ELF. `profile.json` lists the exact
firmware ELFs a run used, which is the easiest way to fill in `<build-hash>`
above. Source paths are the absolute paths the ELF was built with, so view the
report on the machine that ran the build, or rewrite the `SF:` lines.

### 4.7 RISC-V commitlog — `TT_SIM_TRACE_COMMITLOG`

**What:** one file per baby core type (`brisc.commitlog`, `ncrisc.commitlog`,
`trisc0..2.commitlog`) in Spike's `--log-commits` format:

```
core   0: 3 0x000039e0 (0xffb001b7) x 3 0xffb00000
```

**Why:** differential testing of the RISC-V model against
[Spike](https://github.com/riscv-software-src/riscv-isa-sim).

```bash
TT_SIM_TRACE_COMMITLOG=/tmp/run/commitlog/ wolfpine run wormhole ./your_program

spike --log-commits ./pure_rv_test.elf 2> /tmp/spike.log
python3 -m framework.trace.diff_spike /tmp/run/commitlog/brisc.commitlog /tmp/spike.log
# -> "files match (N lines)", or the first divergence with context
```

The diff is meaningful only for pure RV32IM code: Spike knows nothing of Tensix
or NoC MMIO. Files are per core *type*, not per tile, so pin a single tile
(`TT_SIM_TENSIX_COORDS`) when you want one core's stream.

### 4.8 Architectural invariants — `TT_SIM_TRACE_INVARIANTS`

**What:** checks over the event stream, with violations written as JSONL. The
active checks are PC alignment, memory-access alignment and NoC request/response
pairing. A fourth, kernel-lifecycle ordering, needs the lifecycle events that
the tt-metal flow does not publish, so it cannot fire today.

```bash
TT_SIM_TRACE_INVARIANTS=/tmp/run/invariants.jsonl wolfpine run blackhole ./your_program
# For CI: also fail the run on the first violation
TT_SIM_TRACE_INVARIANTS_STRICT=1 ...

jq -r .invariant /tmp/run/invariants.jsonl | sort | uniq -c   # empty file = clean
```

This is separate from the always-on NoC alignment, multicast-order and
semaphore checks, which raise immediately. They are described in the
[runbook §4.1](running-tt-metal-on-the-simulator.md#41-server--protocol).

### 4.9 State dumps — `TT_SIM_TRACE_STATE_DUMP`

**What:** JSON snapshots of every baby core's registers and PC, plus NoC
counters, intended to be taken at each kernel boundary. A companion
`python3 -m framework.trace.diff_state a.json b.json` reports the first
divergence between two dumps.

**Status: produces no files in the tt-metal flow today.** Dumps are triggered by
kernel-lifecycle events, and only the removed standalone Python driver ever
published those. The variable is accepted and the directory is created, but it
stays empty. For cross-run regression checks in the meantime, compare counter
totals (§4.2) or the device profiler's zone spans (§5.1).

---

## 5. tt-metal's own instrumentation, on the simulator

tt-metal's built-in instrumentation works unmodified against Wolfpine and
writes **the same files a card does**. That is what makes a simulator-versus-
silicon comparison possible, and it lets existing tooling run on simulator
output.

**Where the files land.** tt-metal writes to `$TT_METAL_PROFILER_DIR/.logs/`,
or to `$TT_METAL_HOME/generated/profiler/.logs/` when that is unset. Nothing is
written next to your program. Set a fresh `TT_METAL_PROFILER_DIR` per run: the
default directory is shared, so concurrent runs overwrite each other.

**Pin the worker set.** The profiler's readback touches every core, and on
Wolfpine a host read to a released core materialises it. Unpinned, a one-core
program becomes a full-grid one (measured: 80 workers instead of 1, 92 s
against 27 s). Pin it:

```bash
export TT_SIM_TENSIX_COORDS=1-1      # Wormhole, logical (0,0)
export TT_SIM_TENSIX_COORDS=1-2      # Blackhole, logical (0,0)
```

Reads to pinned-out cores are answered harmlessly; the server warns that it
"NullCore-swallowed" them, which is expected here.

### 5.1 Device profiler zones — `TT_METAL_DEVICE_PROFILER`

**What:** `profile_log_device.csv`, with a `ZONE_START`/`ZONE_END` pair per
firmware and kernel zone per RISC per core. Any `DeviceZoneScopedN("name")`
markers you add to your kernels appear here too.

```bash
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=/tmp/prof \
TT_SIM_TENSIX_COORDS=1-2 TT_SIM_COST_MODEL=1 \
    wolfpine run blackhole ./your_program
# -> /tmp/prof/.logs/profile_log_device.csv
```

Zone spans in cycles, per core and RISC (the CSV's first line is an `ARCH:`
banner, hence `skip=1`):

```bash
duckdb -c "
  WITH z AS (
    SELECT core_x, core_y, \"RISC processor type\" AS risc, \"zone name\" AS zone,
           type AS kind, \"time[cycles since reset]\" AS t
    FROM read_csv('/tmp/prof/.logs/profile_log_device.csv', skip=1, normalize_names=false)
    WHERE type IN ('ZONE_START', 'ZONE_END'))
  SELECT core_x, core_y, risc, zone,
         MAX(t) FILTER (WHERE kind = 'ZONE_END') - MIN(t) FILTER (WHERE kind = 'ZONE_START') AS cycles
  FROM z GROUP BY ALL ORDER BY core_x, core_y, risc, zone"
```

That query is for a single launch. A program that launches several times
repeats each zone, so group by `"run host ID"` as well, or split the rows at
the `*-KERNEL` zone boundaries.

**Keep the numbers in cycles.** The simulator reports `CHIP_FREQ[MHz]: 0`
because it has no clock frequency. tt-metal's own post-processors
(`tools/tracy/process_device_log.py` and friends, which need tt-metal's Python
environment) convert cycles to time using that field. They are not verified
against simulator output; if you use them, expect the conversion to time to be
meaningless.

On silicon the profiler itself perturbs what it measures. Compare an
instrumented card run only with an instrumented simulator run.

### 5.2 NoC event trace — `TT_METAL_DEVICE_PROFILER_NOC_EVENTS`

**What:** one 8-byte record per NoC transaction issued by BRISC/NCRISC, written
by tt-metal's own device-side recorder into `noc_trace_dev<d>_ID<n>.json`. It
needs no change to your program. Wolfpine ships a decomposer that partitions
each core's kernel span by mechanism.

**Why:** this is the strongest instrument Wolfpine has. It answers "is this
phase bound by command issue or by bytes on the wire?" per core, and it is
validated against real silicon on both architectures.

```bash
TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 TT_METAL_PROFILER_DIR=/tmp/nocev \
TT_SIM_TENSIX_COORDS=1-1 TT_SIM_COST_MODEL=1 \
    wolfpine run wormhole ./your_program
# -> /tmp/nocev/.logs/noc_trace_dev0_ID0.json  (+ profile_log_device.csv, topology.json)

python3 -m framework.perf.noc_events --sim /tmp/nocev/.logs/noc_trace_dev0_ID0.json \
    --decompose-only --report /tmp/nocev/report.txt --json /tmp/nocev/report.json
```

The output gives, per core, buckets that sum exactly to the kernel zone span:

```
BRISC @ 1,2
  kernel zone span = 1115 cycles
  mechanism           sim cycles   % of span
  prologue *                  57       5.11 %
  issue                      145      13.00 %
  read_wait                   58       5.20 %
  write_wait                  28       2.51 %
  other_wait                   0       0.00 %
  local                      827      74.17 %
  profiler                     0       0.00 %
```

| Bucket | Meaning |
| --- | --- |
| `issue` | NoC command issue: from an issuing event to the next event. |
| `read_wait` / `write_wait` | Data transfer: the barrier spin, `START` to `END`. |
| `other_wait` | Semaphore waits, full barriers, flushes. |
| `local` | The core's own work between a barrier's end and the next event. |
| `profiler` | The profiler flushing its own buffer. Subtract it for the uninstrumented span. |
| `prologue` | From `ZONE_START` to the first NoC event. **This is not setup cost:** a blocking `cb_wait_front` emits no event, so a core late in a pipeline books its whole upstream wait here. |

To compare against silicon, pass the card's capture of the same binary with
`--card <file>` instead of `--decompose-only`.
The [runbook §4.3a](running-tt-metal-on-the-simulator.md#43a-where-your-cycles-went-the-noc-event-trace)
has the full caveats: multi-launch files, stamps taken at issue, and how to
size a run.

For Wolfpine's own *modelled* per-transaction cycles beside tt-metal's
records, also capture `TT_SIM_TRACE_NOC` (§4.3) in the same run and pass it as
`--sim-internal <dir>`.

### 5.3 Tensix performance counters — `TT_METAL_PROFILE_PERF_COUNTERS`

**What:** the hardware `RISCV_DEBUG_REG_PERF_CNT_*` counters, programmed and
read by tt-metal exactly as on silicon. The values are written as `TS_DATA` rows
in `profile_log_device.csv`. Bit 5 (`32`) selects the `INSTRN_THREAD` bank,
which is the bank Wolfpine sources.

```bash
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILE_PERF_COUNTERS=32 \
TT_METAL_PROFILER_DIR=/tmp/pc TT_SIM_TENSIX_COORDS=1-1 TT_SIM_COST_MODEL=1 \
    wolfpine run wormhole ./your_program

# Non-zero counters
grep -o '"counter type":"[A-Z_0-9]*";"ref cnt":[0-9]*;"value":[1-9][0-9]*' \
    /tmp/pc/.logs/profile_log_device.csv | sort -u
```

On `examples/four` this gives values such as `THREAD_INSTRUCTIONS_{0,1,2}`,
`THREAD_STALLS_*`, `WAITING_FOR_SRCA_VALID` and `WAITING_FOR_NONZERO_SEM_2`.

- Counters Wolfpine tracks: `THREAD_STALLS_*`, `THREAD_INSTRUCTIONS_*`,
  `WAITING_FOR_NONZERO_SEM_*`, `WAITING_FOR_NONFULL_SEM_*`,
  `WAITING_FOR_SRC{A,B}_VALID`, `WAITING_FOR_SRC{A,B}_CLEAR`, and `ref_cnt`.
- Every other counter reads `0` and prints a warning naming it. Set
  `TT_SIM_STRICT_PERF_COUNTERS=1` to make that an error instead.
- A data-movement-only program has no Tensix thread activity, so every counter
  reads 0.

The counters are a functional register model; none of them feeds the cost
model. See
[`driver/wormhole/README.md`](../driver/wormhole/README.md#tensix-performance-counters).

### 5.4 Kernel printf — `TT_METAL_DPRINT_CORES`

`DPRINT` works as on hardware, and prints arrive on the host's stdout:

```bash
TT_METAL_DPRINT_CORES=0,0 TT_SIM_TENSIX_COORDS=1-1 \
    wolfpine run wormhole ./metal_example_hello_world_datamovement_kernel
# 0:0-0:BR: My logical coordinates are 0,0
```

Pin the worker set: DPRINT initialisation touches every core, and unpinned it
builds the whole grid (8 s → 45 s on Wormhole). The LLK sanitizer rides on the
same channel; set `TT_METAL_LLK_SANITIZER=1` *together with*
`TT_METAL_DPRINT_CORES`. Details are in the
[runbook §4.7](running-tt-metal-on-the-simulator.md#47-dprint--kernel-printf-and-the-llk-sanitizer).

---

## 6. Human-readable diagnostics and wire capture

These print text to stderr rather than writing artefacts. They are the tools
for "why is it wrong" and "why is it hanging" rather than "where did the time
go", and they compose with everything above.

**Per-component traces — `TT_SIM_DIAG_*`** (truthy = `1/true/yes/on`):

| Variable | Traces |
| --- | --- |
| `TT_SIM_DIAG_BRISC`, `_NCRISC`, `_TRISC0`, `_TRISC1`, `_TRISC2` | Each retired RISC-V instruction, disassembled: `[<core>-> <cycle>][<pc>] <disasm>` |
| `TT_SIM_DIAG_NOC0`, `_NOC1` | NoC transactions |
| `TT_SIM_DIAG_CO_ISSUED`, `_CO_CONFIG`, `_CO_UNPACK`, `_CO_PACK`, `_CO_FPU`, `_CO_SFPU`, `_CO_THCON` | Tensix coprocessor activity, e.g. `Issued <OP> to <UNIT> from thread <n>` |
| `TT_SIM_DIAG_TRISC`, `_NOC`, `_CO`, `_ALL` | Aggregates. An individual variable overrides its aggregate, so `TT_SIM_DIAG_ALL=1 TT_SIM_DIAG_NCRISC=0` means everything except NCRISC. |

```bash
TT_SIM_DIAG_BRISC=1 TT_SIM_DIAG_CO_SFPU=1 wolfpine run blackhole ./your_program 2> diag.log
```

These are verbose: a trivial program produces thousands of lines. Narrow to the
one core and the one unit you suspect, and pin a single tile.

**Server and wire:**

| Variable | Effect |
| --- | --- |
| `TT_SIM_LOG_PROTOCOL=1` | Print every host ↔ device wire message. The first tool for a hang: it shows which core and address the host is polling. |
| `TT_SIM_RECORD=<file>` | Record every wire message, with read-reply data, to a text trace that can be replayed later (`driver/wormhole/replay.py`; the `server/*_replay_test.py` guards show the socket-free form). |
| `TT_SIM_DEADLOCK` / `TT_SIM_UNIT_STALL` | The deadlock watchdog and wedged-unit detector are on by default and print a diagnosis when the device stops making progress ([runbook §4.4–4.5](running-tt-metal-on-the-simulator.md#44-deadlock-watchdog)). |

---

## 7. Wormhole and Blackhole differences

Every mode in this guide runs on both architectures. What differs:

| | Wormhole | Blackhole |
| --- | --- | --- |
| Simulator target | `wolfpine run wormhole`, or `TT_METAL_SIMULATOR=…/framework/sim/wormhole` | `wolfpine run blackhole`, or `…/framework/sim/blackhole` |
| Logical worker (0,0) pinned as | `TT_SIM_TENSIX_COORDS=1-1` | `TT_SIM_TENSIX_COORDS=1-2` |
| Worker grid | 8×10, 72 default workers | 17×12 NoC, 130 default workers |
| DRAM channels | 6 | 8 |
| `core_x`/`core_y` in traces | Wormhole NoC coordinates | Blackhole NoC coordinates |
| DRAM → L1 NoC congruence | 32 B | 64 B |
| RISC-V `mcycle`/`minstret` per-zone decomposition (`perfbench/retirebench`) | Not available: Wormhole documents no CSRs | Available |

Coordinates, unit names and counter values are **not comparable across
architectures**, so never join a Wormhole dataset with a Blackhole one. To
compare the same program on both, run it twice and compare the *reports*, for
example `report.json` totals or the `noc_events` decompositions.

---

## 8. Recipes

### 8.1 Everything on

```bash
#!/usr/bin/env bash
# profile-run.sh <arch> <outdir> <program> [args...]
# Runs a tt-metal program on Wolfpine with every Wolfpine writer enabled.
set -euo pipefail
arch=$1 OUT=$2; shift 2
rm -rf "$OUT" && mkdir -p "$OUT"
export TT_SIM_COST_MODEL=1
export TT_SIM_PROFILE="$OUT/profile"            # report + counters + hotspots
export TT_SIM_TRACE_PERFETTO="$OUT/timeline.json.gz"
export TT_SIM_TRACE_NOC="$OUT/noc/"
export TT_SIM_TRACE_MEMORY="$OUT/memory.callgrind"
export TT_SIM_TRACE_INVARIANTS="$OUT/invariants.jsonl"
# Heavier, opt in when you need them:
# export TT_SIM_TRACE="$OUT/events.jsonl"         # ~10 MB even for a tiny program
# export TT_SIM_TRACE_COMMITLOG="$OUT/commitlog/"
wolfpine run "$arch" "$@"
echo "artefacts in $OUT; start with $OUT/profile/report.md"
```

```bash
cd examples/four/src && ./profile-run.sh blackhole /tmp/four-bh ./build/four
```

With every writer on, including the raw stream and commitlog, `examples/one`
completes in 8–9 s on either architecture.

### 8.2 Simulator view and tt-metal view of one run

The two families compose, so a single run can feed both the internal report and
the silicon-comparable NoC decomposition:

```bash
TT_SIM_COST_MODEL=1 TT_SIM_PROFILE=/tmp/both/profile TT_SIM_TRACE_NOC=/tmp/both/noc/ \
TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 TT_METAL_PROFILER_DIR=/tmp/both/metal \
TT_SIM_TENSIX_COORDS=1-2 \
    wolfpine run blackhole ./your_program

python3 -m framework.perf.noc_events --decompose-only \
    --sim /tmp/both/metal/.logs/noc_trace_dev0_ID0.json --sim-internal /tmp/both/noc/
```

### 8.3 Before and after a kernel change

```bash
for tag in before after; do          # rebuild your kernel between the two
  TT_SIM_COST_MODEL=1 TT_SIM_PROFILE=/tmp/cmp-$tag wolfpine run wormhole ./your_program
done
duckdb -c "
  SELECT counter_name,
         SUM(value) FILTER (WHERE filename LIKE '%cmp-before%') AS before,
         SUM(value) FILTER (WHERE filename LIKE '%cmp-after%')  AS after
  FROM read_parquet(['/tmp/cmp-before/counters/**/*.parquet',
                     '/tmp/cmp-after/counters/**/*.parquet'],
                    hive_partitioning=true, filename=true)
  GROUP BY counter_name ORDER BY before DESC NULLS LAST LIMIT 20"
```

Run the two sequentially, not in parallel (§1.2).

---

## 9. Further reading

- [`docs/trace-schema.md`](trace-schema.md) — the versioned contract for every
  output: field meanings, units, what is frozen, and the traps.
- [`docs/running-tt-metal-on-the-simulator.md`](running-tt-metal-on-the-simulator.md)
  — the runbook: environment, multi-tile grids, every server knob, and the
  NoC-event and DPRINT sections in full.
- [`docs/cost-model-caveats-for-consumers.md`](cost-model-caveats-for-consumers.md)
  — where the cycle model is known to be wrong, and by how much.
- [`framework/trace/README.md`](../framework/trace/README.md) — the
  implementer's side: the event bus, and how to add a publish call or a writer.
- [`framework/trace/queries/`](../framework/trace/queries/) — canned Perfetto
  and DuckDB SQL.
- [`docs/simulator-wall-clock.md`](simulator-wall-clock.md) — profiling the
  *simulator* rather than your kernel: where Wolfpine's own host time goes.
