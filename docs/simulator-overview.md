# How Wolfpine is put together

A tour of the repository and the simulator library, and a precise statement of
what the timing model claims. For installing and running, start at the
[README](../README.md); for profiling, see [`profiling.md`](profiling.md).

## Contents

- [Repository layout](#repository-layout)
- [What the timing model claims, and what it does not](#what-the-timing-model-claims-and-what-it-does-not)
- [Key parts of the simulator](#key-parts-of-the-simulator)

## Repository layout

| Directory | What is in it |
| --- | --- |
| [`framework/`](../framework) | The simulator library, and the only part shipped in the `wolfpine` package. See [Key parts](#key-parts-of-the-simulator) below. |
| [`driver/`](../driver) | Per-architecture drivers and low-level example scripts (detailed below). |
| [`examples/`](../examples) | Arch-agnostic example tt-metal programs. Each validates its own result, so together they double as a test suite. |
| [`optests/`](../optests) | Differential op tests against the vendor reference simulator. |
| [`perfbench/`](../perfbench) | Timing programs that run unchanged on a card and on Wolfpine, used to check the cost model against silicon. |
| [`docs/`](.) | User documentation; the [README](../README.md#documentation) indexes it. |
| [`ROADMAP.md`](../ROADMAP.md) | What is done, what is next, and the evidence behind the timing model. |

The [driver](../driver) directory has one entry per target, each individually documented:

* [wormhole](../driver/wormhole) is where you likely want to go to. It is the Wormhole driver: the simulator target (`run.sh` and the SoC descriptors, linked from [`framework/sim/wormhole`](../framework/sim/wormhole)), the wire-trace replay tests, and the Wormhole-specific documentation. Workers are materialised on demand, so a program gets exactly the grid it launches on, up to the full 72. The example tt-metal programs it runs live in [`examples`](../examples); each validates its own results, so they double as a test suite.
* [blackhole](../driver/blackhole) is the Blackhole driver: the same wire bridge and machinery as wormhole (all shared in [`framework/bridge`](../framework/bridge)), pointed at a Blackhole device instead. Point `TT_METAL_SIMULATOR` at `driver/blackhole` to route execution there. Workers are materialised on demand by the wire bridge, so a program gets exactly the grid it launches on, up to the full 130; see its README.
* [simple](../driver/simple) are very basic examples, demonstrating the memory subsystem and running codes on a vanilla RV32IM CPU.

## What the timing model claims, and what it does not

The simulator carries a cost model of **documented provenance**: every cycle
figure is traceable to the ISA documentation or a vendor source, there are
**zero un-sourced estimates** (asserted by test), and it has been checked
against real silicon. Stated precisely:

> Runs real tt-metal programs unmodified on both architectures, on the full
> default grid, with a timing model of documented provenance, corroborated
> against silicon at the **slope** and **launch** level on both parts, and
> **by mechanism** for NoC-bound work on both parts.

Four caveats belong with that sentence rather than under it, because three of
them are permanent:

- **It is a floor.** Every bound is charged at its low end, by rule, so it
  under-predicts wherever hardware sits above a documented minimum.
- **Mechanism-level checking is one leg of three.** The RV-bound leg is
  Blackhole-only (Wormhole documents no CSRs, so there is no retired-instruction
  count) and currently fails its bar; the Tensix mechanism leg cannot be built
  at all from the counters the hardware exposes.
- **That failure has a single, sourced cause** — integer divide, where the
  documentation gives only a 6–33 cycle range and no algorithm, so a
  magnitude-dependent term cannot be sourced at any provenance.
- **Energy is ranking-level only**, and barely beats a model that knows nothing
  but the cycle count. Absolute joules are out of reach of the instrument.

The full evidence, including what was proven *impossible* rather than merely
left undone, is in [`ROADMAP.md`](../ROADMAP.md).


## Key parts of the simulator

The simulator library is [`framework/`](../framework). It never imports from
`driver/`, and it is **architecture-agnostic**: a target is an architecture
profile plus a thin device class, so Wormhole and Blackhole share all of the
code below. Everything is modelled from the public
[Tenstorrent ISA documentation](https://github.com/tenstorrent/tt-isa-documentation).

**Describing and building a device**

* [`arch/`](../framework/arch) — architecture profiles. `ArchProfile`
  (../[`profile.py`](../framework/arch/profile.py)) is the single source of every
  per-architecture constant: NoC grid and coordinate scheme, maximum burst, L1
  size, DRAM channel coordinates, default worker set. `WORMHOLE_PROFILE` and
  `BLACKHOLE_PROFILE` supply the values. A new architecture starts here.
* [`device/`](../framework/device) — assembles a device from tiles.
  [`tt_device.py`](../framework/device/tt_device.py) is the arch-agnostic
  `TT_Device`, which builds the NoC directories and wires tracing.
  [`tiles.py`](../framework/device/tiles.py) builds each `TensixTile`, `DRAMTile`
  and `EthTile` from the profile, and plumbs a Tensix tile's whole memory map
  (NoC registers, Tensix config, TDMA, soft reset, mailboxes, per-core local
  memory) per the
  [baby RISC-V memory map](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/README.md).
  [`wormhole.py`](../framework/device/wormhole.py) and
  [`blackhole.py`](../framework/device/blackhole.py) are the thin per-arch
  subclasses. [`clock.py`](../framework/device/clock.py) and
  [`reset.py`](../framework/device/reset.py) step every component in lockstep,
  using an event-driven pump that skips cycles in which nothing can happen.
  [`deadlock.py`](../framework/device/deadlock.py) is the watchdog that diagnoses
  a device that has stopped making progress.
* [`memory/`](../framework/memory) — `MemoryMap`, an interval map from address
  ranges to anything that implements the `mem_mapable` interface; this is how
  every register block and memory is attached.

**Inside a Tensix tile**

* [`pe/rv/`](../framework/pe/rv) — the baby RISC-V cores. The ISA is
  pluggable: [`isa/`](../framework/pe/rv/isa) holds RV32I, M, A, B, Zicsr and
  Zfh plus Tenstorrent's
  [`.ttinsn` extension](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/PushTensixInstruction.md#ttinsn-instruction-set-extension),
  composed in [`rv32.py`](../framework/pe/rv/rv32.py).
  [`babyriscv.py`](../framework/pe/rv/babyriscv.py) specialises it for BRISC,
  NCRISC and TRISC0–2, including the
  [soft-reset](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/SoftReset.md)
  start-PC logic. [`cost.py`](../framework/pe/rv/cost.py) is the RV timing model,
  and [`spin.py`](../framework/pe/rv/spin.py) parks cores spinning in firmware
  poll loops so idle workers cost almost nothing.
* [`pe/tensix/`](../framework/pe/tensix) — the
  [Tensix coprocessor](https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/TensixCoprocessor).
  [`frontend.py`](../framework/pe/tensix/frontend.py) decodes, and
  [`backend.py`](../framework/pe/tensix/backend.py) dispatches to the per-unit
  backends in [`backends/`](../framework/pe/tensix/backends): matrix (../FPU), vector
  (../SFPU), unpacker, packer, mover, ThCon, config, sync and misc. The instruction
  set and backend configuration layout are **data**
  (../[`tensix_instructions.yaml`](../framework/pe/tensix/tensix_instructions.yaml),
  `tensix_backend_cfg*.yaml`), so adding an instruction is mostly a YAML edit.
  Blackhole's extra instructions live in the same table.
* [`misc/`](../framework/misc) — mailboxes, Tensix sync hardware (../`ttsync`), tile
  control, and the `RISCV_DEBUG_REG_PERF_CNT_*` performance counters.

**Between tiles**

* [`network/`](../framework/network) — the NoC. [`tt_noc.py`](../framework/network/tt_noc.py)
  is the NoC interface unit (../NUI) on each tile, for both NoCs, with per-hop
  latency, link contention and multicast. Alongside it are the hardware rules,
  which are checked rather than ignored: address congruence
  (../[`alignment.py`](../framework/network/alignment.py)), multicast rectangle order,
  coordinate translation, and attributing a failing transfer to the code that
  issued it.

**Talking to tt-metal**

* [`bridge/`](../framework/bridge) — the arch-agnostic bridge to UMD's simulation
  wire protocol: protocol decoding, the transport, the fabric that routes each
  host read or write to the right tile, stand-in cores for tiles that are not
  built, and the cycle-pumping `Device` wrapper.
  [`materialise.py`](../framework/bridge/materialise.py) builds exactly the workers
  a program launches on, on demand.
* [`sim/wormhole`](../framework/sim/wormhole) and [`sim/blackhole`](../framework/sim/blackhole)
  — the packaged `TT_METAL_SIMULATOR` targets: `run.sh`, the SoC and cluster
  descriptors, and the server entry point. These are what `wolfpine run` points
  tt-metal at.
* [`cli.py`](../framework/cli.py) — the `wolfpine` command.

**Timing and observability**

* [`perf/`](../framework/perf) — the cost model. Cycle costs are tables with the
  provenance of every number attached
  (../[`unit_costs.yaml`](../framework/perf/unit_costs.yaml) and
  [`tensix_instruction_costs.yaml`](../framework/pe/tensix/tensix_instruction_costs.yaml)),
  and [`model.py`](../framework/perf/model.py) turns them into charged cycles when
  `TT_SIM_COST_MODEL` is set. The rest of the directory holds the tools that
  check the model against silicon, such as
  [`noc_events.py`](../framework/perf/noc_events.py).
* [`trace/`](../framework/trace) — a typed publish/subscribe event bus that the
  components above publish to, the writers that turn it into Perfetto, Parquet,
  Callgrind, LCOV and the other outputs, and the ranked bottleneck report. See
  [`docs/profiling.md`](profiling.md) to use it and
  [`framework/trace/README.md`](../framework/trace/README.md) to extend it.
