# Blackhole card session, 2026-08-10 — the paper's evaluation runs

Two runs of `perfbench/run_paper_session.sh` on the one available Blackhole
part, nine minutes apart. Banked here because the originals lived on a ramdisk
(`/mnt/ramdisk/tt_traces/`) that does not survive a reboot.

**ONE RUN, ON ONE CARD, by one operator, on one day.** Every figure below is
`corroboration`, never provenance: nothing here may become a cost-model entry.
These are *not* `tt_sim/perf/datasets/` files, carry no dataset contract header,
and are not read by any test — they are session evidence for
`docs/plans/evalplan.md` and the paper's evaluation section.

## The part

| fact | value | source |
| --- | --- | --- |
| architecture | `blackhole` | `session.log` |
| logical worker grid | 12 × 10 = 120 workers | `Slow dispatch mode: Using full logical grid (12, 10)` |
| SoC grid | 17 × 12 | `nocbench-grid.csv` |
| worker columns (physical NoC) | 1–7, 10–14 | `nocbench-grid.csv` |
| AICLK | 1350 MHz | `prof.*.profile_log_device.csv` header |
| DRAM banks | 8 | kernel build defines |
| firmware bundle | 19.6.0 | `session.log` |
| KMD | 2.10.0 | `session.log` |
| PCI identity | `1e52:b140`, two functions | `sku.txt` |
| board SKU | **unobtainable** — `tt-smi` is not installed on the card host | `sku.txt` |
| host | HPE ProLiant DL385 Gen10 Plus, `nextgenio-amd02` | `session.log` |

The SKU question is settled **by absence**, not pending: no `tt-smi`, so no
board-type string exists in any session. Quote the part by measurement.

## The two runs

* `paper-1/` — 09:51, **with** `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4`.
* `paper-2/` — 09:57, **without** it; also carries the optests hardware hex.

## Results, as measured

| rung | paper-1 | paper-2 | note |
| --- | --- | --- | --- |
| `add_2_integers_in_riscv` | PASS 2.5 s | PASS 2.4 s | |
| `loopback` | PASS 3.7 s | PASS 2.4 s | |
| `eltwise_binary` | FAIL (rc 134) | FAIL (rc 134) | 64 230 / 65 536 elements wrong |
| `six` (128³ matmul) | not run | FAIL (rc 1) | PCC 0.3067 vs 0.97 |
| `matmul_multi_core` | PASS, PCC 0.9821 | FAIL (rc 134) | see the override note |
| `matmul_single_core` (640³) | FAIL (rc 134) | FAIL (rc 134) | PCC 0.9228 vs 0.97 |

**Three of six pass on silicon.** The two that pass unconditionally are the two
that move data without computing on it.

### The compute-grid override is a *silicon* workaround

Without it, tt-metal 0.74 throws `YAML::TypedBadConversion<int>` resolving the
core descriptor for this harvested part and aborts before touching the device —
which is why `matmul_multi_core` fails in `paper-2` for a reason that has
nothing to do with the matmul. With it, the profiler shows the work confined to
**40 of the 120 workers** (not 20): per-core kernel-zone durations separate
cleanly into 40 cores at 85–92 k cycles and 80 at 60–70 k. The PCC 0.9821 of
`paper-1` was measured under that restriction and must not be quoted without it.

## Device cycles

Metric: **the maximum, over every core and every RISC-V processor, of a
`*-KERNEL` zone's duration**, in device cycles. A whole-run span across cores is
useless on the multi-core rung — tt-metal writes launch messages to the 120
workers serially, ~28 700 cycles apart, so that span reads 3.68 M cycles and
measures host serialisation. Ten of the 120 cores also report a wall-clock
counter offset by ~1.5e13 cycles and must be excluded from any cross-core span.

| rung | paper-1 | paper-2 |
| --- | --- | --- |
| `add_2_integers_in_riscv` | 910 | 911 |
| `loopback` | 44 288 | 44 334 |
| `eltwise_binary` | 41 595 | 41 313 |
| `six` | — | 72 730 |
| `matmul_multi_core` | 92 452 | — |
| `matmul_single_core` | 9 038 207 | 9 047 308 |

Reproduces to 0.1 % across the two sessions except `eltwise_binary` at 0.7 %.
The `eltwise_binary`, `six` and `matmul_single_core` counts time computations
that produced the **wrong answer**.

## The `eltwise_binary` failure — signature, not root cause

See `eltwise-displacement.md`. The output buffer holds the *correct* result
displaced by exactly 512 bfloat16 elements (1024 bytes = half a tile = two of
the four 16×16 faces). Root cause **open**; do not write it up as a simulator
fidelity gap.

## The optests hardware arm

`paper-2/optests.blackhole.csv` holds `OPDIFF_RESULT` hex for four programs
(the step did not reach the fifth, `matmuluntilize`). Compared byte-for-byte
against local tt-sim and ttsim v1.9.6 runs of the same programs:

| program | tt-sim vs ttsim | vs hardware |
| --- | --- | --- |
| `dramtop` | identical (512 elements) | **identical** — bit-exact three ways |
| `matmulblock` | identical (12 288) | 12 288 / 12 288 differ (100 %) |
| `matmulidx` | identical (5 120) | 4 096 / 5 120 differ (80 %) |
| `matmultranspose` | identical (3 072) | 2 048 / 3 072 differ (66.7 %) |
| `matmuluntilize` | identical (2 048) | no hardware arm |

The three matmul programs also fail their own analytical goldens on the card
(`rc=1`), so the hardware arm is not a usable reference for them. Not one
differing element is within 1 ULP: the value *sets* differ, so this is not a
format or rounding difference.
