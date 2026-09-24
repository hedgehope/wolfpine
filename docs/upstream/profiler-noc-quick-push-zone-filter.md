# DRAFT — not filed. Read §0 before deciding whether to file.

**Target:** [`tenstorrent/tt-metal`](https://github.com/tenstorrent/tt-metal)
(`0.74.0` / `c49bb7625e`).
**Status:** draft only, 2026-08-19. Nothing has been filed, pushed or sent.

---

## §0. For our reviewer — what this does and does not claim

It claims one thing, and it is a string mismatch: the device profiler emits its
flush zone as `PROFILER-NOC-QUICK-PUSH`, the NoC-trace JSON writer filters
`PROFILER-NOC-QUICK-SEND`, and so the profiler's own zone is written into every
`noc_trace_*.json` whose kernel fills the marker buffer. The filter's existence
is the evidence that this is unintended — nobody writes an exclusion for a zone
they want in the artefact.

It does **not** claim that any *timestamp* or NoC event is wrong; the leak is one
extra `ZONE_START`/`ZONE_END` pair per flush, not corrupted data. It does not
claim a performance cost. And it does not claim to know the history: we cannot
see from a grafted checkout when or why the two names diverged, so "the zone was
renamed and the filter did not follow" is our reading of the two lines, not
something we have confirmed.

Nothing here depends on tt-sim being right. The two lines are read out of the
tt-metal tree, and the only measurement involved is counting records in a
`noc_trace` JSON the vendor's own converter wrote.

The compiler team hit this independently before we did, and said we should file
it. Their reading, which is also ours: *"the filter string occurring exactly
once in the whole tt-metal tree, in the filter itself, is about as clean a
diagnosis as these get."*

---

## §1. Ask

Change the filter in `convertNocTracePacketsToJson` to name the zone the device
actually emits, so the device profiler's own flush zone stops appearing in
`noc_trace_*.json`. One string; a shared constant would stop it recurring.

## §2. What the code says

The NoC-trace JSON writer separates zone endpoints from NoC events and drops a
short list of zones that are instrumentation rather than kernel windows:

```cpp
// tt_metal/impl/profiler/profiler.cpp:778-784, inside
// convertNocTracePacketsToJson (declared at :757)
if (isMarkerAZoneEndpoint(marker)) {
    if (marker.marker_name != "SYNC-ZONE-SENDER" && marker.marker_name != "SYNC-ZONE-RECEIVER" &&
        marker.marker_name != "PROFILER-NOC-QUICK-SEND" && !marker.marker_name.ends_with("-FW") &&
        (!marker.marker_name.ends_with("-KERNEL") || marker.risc == tracy::RiscType::BRISC ||
         marker.risc == tracy::RiscType::NCRISC)) {
        zones_by_op[program_execution_uid].push_back(marker);
    }
}
```

`"PROFILER-NOC-QUICK-SEND"` occurs **exactly once in the entire tt-metal tree**,
and that occurrence is the filter itself:

```
$ grep -rn "PROFILER-NOC-QUICK-SEND" .
tt_metal/impl/profiler/profiler.cpp:780:  ... marker.marker_name != "PROFILER-NOC-QUICK-SEND" ...
```

The zone the device emits is `PROFILER-NOC-QUICK-PUSH`:

```cpp
// tt_metal/tools/profiler/kernel_profiler.hpp:409, inside quick_push() (:390)
SrcLocNameToHash("PROFILER-NOC-QUICK-PUSH");
mark_time_at_index_inlined(wIndex, hash);
wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
...
mark_time_at_index_inlined(wIndex, get_const_id(hash, ZONE_END));   // :420
```

Those two are the only `PROFILER-NOC-QUICK-*` strings anywhere in the tree, so
the filter matches nothing and the surviving markers are added back at `:810-815`
and written out with `"zone"` / `"zone_phase"` fields at `:827-842` like any
kernel zone.

## §3. Why it matters — a consumer counting kernel launches sees N+1

`kernel_profiler::quick_push()` brackets itself in that zone, and
`recordNocEvent` calls `flush_to_dram_if_full` **before** it records
(`tt_metal/tools/profiler/noc_event_profiler.hpp:101` and `:109`, the two
arms), so the push happens *inside* the enclosing `BRISC-KERNEL` /
`NCRISC-KERNEL` zone every time the per-RISC L1 marker buffer fills mid-kernel. Any tool that treats a
`ZONE_START` as a kernel launch — which is the obvious reading of a file whose
only other zones are `*-KERNEL` pairs — sees several windows where there is one.

That is not hypothetical: it is how we met the bug. Our NoC-event decomposition
refused **32 of 32 streams** of a 16-core `gemm_256_check` capture as "2 to 5
`ZONE_START`, expected exactly one". All 32 in fact carry exactly one `*-KERNEL`
pair, kernel `ZONE_START` the first record and `ZONE_END` the last, with 0–4
strictly nested flushes inside. The workload was fine; the instrument was
describing itself.

**How often it fires.** The per-RISC L1 marker vector is 512 `uint32`
(`PROFILER_L1_OPTIONAL_MARKER_COUNT` 250 + 4 guaranteed + 2 program-id, times
`PROFILER_L1_MARKER_UINT32_SIZE` 2 —
`tt_metal/hostdevcommon/api/hostdevcommon/profiler_common.h:92-99`), `wIndex`
starts and resets at `CUSTOM_MARKERS` = 12 (`kernel_profiler.hpp:119`, `:466`),
`recordNocEvent` records under `DoingDispatch::DISPATCH` which reserves
`DISPATCH_HEADROOM_SIZE` = 16 (`:72-74`), and a recorded event costs 4 `uint32`
in the default build (`NON_DROPPING` is false unless `DEVICE_DEBUG_DUMP`,
`:94-101`). That is a flush roughly every 120 recorded NoC events on a RISC, and
a capture we hold agrees: the first `PROFILER-NOC-QUICK-PUSH` `ZONE_START` is
record 122 of a stream whose record 0 is the `BRISC-KERNEL` `ZONE_START`, i.e.
after 121 recorded events.

So this affects any `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` capture of a kernel
that issues more than of order a hundred transactions on one RISC — which is
most real kernels and few microbenchmarks, and is why it can sit unnoticed.

Two records from that capture, verbatim, showing the zone in the artefact:

```json
{ "proc": "BRISC", "sx": 1, "sy": 2, "timestamp": 901232,
  "zone": "PROFILER-NOC-QUICK-PUSH", "zone_phase": "ZONE_START" },
{ "proc": "BRISC", "sx": 1, "sy": 2, "timestamp": 901254,
  "zone": "PROFILER-NOC-QUICK-PUSH", "zone_phase": "ZONE_END" }
```

## §4. Proposed change

```diff
--- a/tt_metal/impl/profiler/profiler.cpp
+++ b/tt_metal/impl/profiler/profiler.cpp
@@ -778,7 +778,7 @@
                 if (isMarkerAZoneEndpoint(marker)) {
                     if (marker.marker_name != "SYNC-ZONE-SENDER" && marker.marker_name != "SYNC-ZONE-RECEIVER" &&
-                        marker.marker_name != "PROFILER-NOC-QUICK-SEND" && !marker.marker_name.ends_with("-FW") &&
+                        marker.marker_name != "PROFILER-NOC-QUICK-PUSH" && !marker.marker_name.ends_with("-FW") &&
                         (!marker.marker_name.ends_with("-KERNEL") || marker.risc == tracy::RiscType::BRISC ||
                          marker.risc == tracy::RiscType::NCRISC)) {
```

Worth more than the one-line fix: the zone name is a bare literal on both sides,
which is what let them drift. A shared `constexpr` in a header both
`kernel_profiler.hpp` and `profiler.cpp` already reach (the marker name is
passed through `SrcLocNameToHash`, so it must stay a literal at the emission
site, but the *host* side can compare against a named constant) would turn the
next rename into a compile error rather than a silent behaviour change.

We have no view on whether you would rather keep the zone in the JSON and
document it. If so, the fix is the mirror image — delete the dead filter clause
and say in the NoC-tracing documentation that `PROFILER-NOC-QUICK-PUSH` pairs
are instrumentation and must be skipped — and this ticket is then a
documentation one. Either resolution is fine by us; what does not work is the
current state, where the code says the zone is excluded and the artefact
contains it.

## §5. Reproducing

**Without running anything.** Both facts are static:

```bash
grep -rn "PROFILER-NOC-QUICK-SEND" .   # 1 hit: the filter, profiler.cpp:780
grep -rn "PROFILER-NOC-QUICK-PUSH" .   # 1 hit: the emitter, kernel_profiler.hpp:409
```

**With hardware.** Build any data-movement kernel that issues more than ~120 NoC
transactions on a single RISC, run it with `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1`,
and grep the resulting `.logs/noc_trace_dev<N>_ID<M>.json` for
`PROFILER-NOC-QUICK-PUSH`. Every match is a zone the filter above was written to
remove. Kernels below that threshold produce a clean file, so a short benchmark
will not show it.

## §6. Verified vs inferred

**Verified by reading the named code, against `0.74.0` / `c49bb7625e`:**

- `profiler.cpp:780` filters `"PROFILER-NOC-QUICK-SEND"`, and that string occurs
  exactly once in the tree — an unrestricted `grep -rc` over the checkout
  (build directories included) returns one file with one hit.
- `kernel_profiler.hpp:409` emits `"PROFILER-NOC-QUICK-PUSH"`, and `:420` its
  `ZONE_END`. These are the only two `PROFILER-NOC-QUICK-*` strings in the tree.
- Zone markers that survive the filter are re-added (`:810-815`) and emitted as
  `"zone"` / `"zone_phase"` records (`:827-842`).
- `recordNocEvent` calls `flush_to_dram_if_full` before recording
  (`noc_event_profiler.hpp:101`, `:109`), so a push nests inside the kernel zone.
- The buffer arithmetic in §3, from `profiler_common.h:92-99` and
  `kernel_profiler.hpp:72-74`, `:94-101`, `:119`, `:165-176`, `:466`.
- The zone name is emitted by real device firmware, not only by our simulator: a
  Wormhole card session's `.logs/zone_src_locations.log` carries the compiler
  `#pragma message: PROFILER-NOC-QUICK-PUSH,<...>/kernel_profiler.hpp,409,KERNEL_PROFILER`.
- The JSON records quoted in §3 are copied verbatim from a capture written by
  tt-metal's own converter.

**A caveat we would rather state than be asked**: the capture in §3 reached us
from a downstream consumer and is held here as a simulator-backed run — our
simulator executes tt-metal's unmodified device-profiler firmware, and the JSON
was written by tt-metal's own host-side converter. We have **not** reproduced
the JSON on a card ourselves: the kernels in our own card sessions all sit below
the flush threshold and produce clean files. That is why this report rests on
the two source lines rather than on the artefact. The filter is host-side and
the emitting firmware is the same either way, so nothing in the mechanism
depends on which backend ran the kernel — but if you want a silicon artefact
before acting, it is a kernel of ~150 transactions away and we are happy to
capture one.

**Inferred, not proven:**

- That the divergence is a rename the filter did not follow. The checkout we have
  is grafted, so we cannot search history; the two lines are all we can see.
- That excluding the zone is still the intent. We read that off the filter's
  existence and its neighbours (`SYNC-ZONE-*`, `-FW`), not off any statement.
- That nothing else in the tree already compensates downstream. We checked the
  JSON writer and the profiler tests; we did not audit every consumer of
  `noc_trace_*.json`.
- **Line numbers are release-specific.** Everything above is `0.74.0`; the
  strings are the stable part of this report, not the line numbers.
