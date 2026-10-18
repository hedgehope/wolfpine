# Wolfpine

**A framework for simulating AI accelerators, in pure Python.**

Wolfpine is a hackable software framework for building cycle-approximate
simulators of AI accelerator hardware. The core — the memory map, the
interconnect, the RISC-V cores, the clock and reset fabric, the cost and
trace machinery — is architecture-agnostic; a target is described by an
*architecture profile* plus a thin device class, not by a fork.

The framework ships with backends for the Tenstorrent architecture,
**Wormhole** and **Blackhole**, built from the public
[Tenstorrent ISA Documentation](https://github.com/tenstorrent/tt-isa-documentation/tree/main).
Both run real tt-metal programs unmodified, on the full default worker grid —
72 workers on Wormhole, 130 on Blackhole. Note that it is not cycle accurate
(or intended to be). The idea was to build a simulator from public
documentation, so developers can experiment with writing accelerator code
without needing physical hardware — and, because it is Python all the way
down, poke at simulator state directly from a driver script.

## Contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Using Wolfpine as a simulator for tt-metal](#using-wolfpine-as-a-simulator-for-tt-metal)
- [Repository overview](#repository-overview)
- [What the timing model claims, and what it does not](#what-the-timing-model-claims-and-what-it-does-not)
- [Key parts of the simulator](#key-parts-of-the-simulator)

Profiling a kernel: [`driver/wormhole/docs/profiling.md`](driver/wormhole/docs/profiling.md)
walks through every output the simulator can produce.
Building tooling on those outputs: [`docs/trace-schema.md`](docs/trace-schema.md)
is the stable, versioned schema — field meanings, units and what you may
rely on.

## Quick start

You need Python ≥ 3.10 and a built [tt-metal](https://github.com/tenstorrent/tt-metal)
checkout. Install Wolfpine from PyPI and tell it where tt-metal is:

```bash
pip install wolfpine
wolfpine config --tt-metal /path/to/tt-metal    # once; saved to ~/.config/wolfpine
wolfpine doctor                                  # check what it found
```

Then run any tt-metal program against the simulator with `wolfpine run <target>`, where the
target is `wormhole`, `blackhole`, or `hw` for real hardware. The program needs no changes:

```bash
cd /path/to/tt-metal/build/programming_examples
wolfpine run wormhole  ./metal_example_add_2_integers_in_compute   # simulated Wormhole
wolfpine run blackhole ./metal_example_add_2_integers_in_compute   # simulated Blackhole
wolfpine run hw        ./metal_example_add_2_integers_in_compute   # the same binary on hardware
```

`wolfpine run` sets tt-metal's environment variables (`TT_METAL_SIMULATOR`,
`TT_METAL_SLOW_DISPATCH_MODE`, `TT_METAL_RUNTIME_ROOT`, `LD_LIBRARY_PATH`) for that one
command only, so nothing is left behind in your shell. Leave out the command to get a
subshell with the target switched on (`exit` to leave it). To switch your current shell
instead — handy when building your own programs with CMake — `eval` the exports; `off`
puts back exactly what was there before:

```bash
eval "$(wolfpine env blackhole)"   # or wormhole, or hw
cmake -B build -S . && cmake --build build && ./build/my_program
eval "$(wolfpine env off)"
```

Run `wolfpine --help` for the full list of commands.

## Installation

Wolfpine is pure Python and requires Python ≥ 3.10. It depends on numpy, pyyaml, pynng,
flatbuffers, pyarrow and pyelftools, which pip installs for you.

**From PyPI** — the simulator library plus the `wolfpine` command, everything needed to run
tt-metal programs:

```bash
pip install wolfpine
```

**From source** — for the example programs, the low-level driver scripts and the test
suite, or to hack on the simulator itself:

```bash
git clone https://github.com/hedgehope/wolfpine.git
cd wolfpine
pip install -e .           # add [dev] for the ruff + pytest tooling: pip install -e .[dev]
```

Both install the `framework` simulator library and the `wolfpine` command. The example
drivers under `driver/` are not part of the package; they run from the repository root, so
keep the repo root on your `PYTHONPATH` when invoking them directly:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Using Wolfpine as a simulator for tt-metal

tt-metal's UMD has a "simulation" chip backend: point `TT_METAL_SIMULATOR` at a simulator
directory and UMD launches Wolfpine in place of real silicon, so an ordinary tt-metal program
runs against the simulator over a socket — no code changes, exactly as it would run on
hardware. The simulator directories ship inside the package (`framework/sim/wormhole` and
`framework/sim/blackhole`), which is why the [`wolfpine` command](#quick-start) is the easy
way in: it finds them for you.

`wolfpine` looks for tt-metal in this order: `--tt-metal` (given before the target, e.g.
`wolfpine run --tt-metal /path wormhole ./prog`), then `$TT_METAL_RUNTIME_ROOT` or
`$TT_METAL_HOME`, then the path saved by `wolfpine config`. If you would rather export
the variables yourself, these are the ones it manages:

```bash
export TT_METAL_RUNTIME_ROOT=/path/to/tt-metal          # your built tt-metal checkout
export TT_METAL_SIMULATOR="$(python3 -c 'from framework.sim import simulator_dir; print(simulator_dir("wormhole"))')"
export TT_METAL_SLOW_DISPATCH_MODE=1                    # the only launch path the sim models
export LD_LIBRARY_PATH="$TT_METAL_RUNTIME_ROOT/build/lib:$LD_LIBRARY_PATH"
```

In a repository checkout `driver/wormhole` and `driver/blackhole` link to the same simulator
directories, so `TT_METAL_SIMULATOR=$PWD/driver/wormhole` works too.

A set of ready-to-run example programs, plus a build-and-run test suite, lives under
[`examples`](examples). For the full walkthrough — building the
examples, the `TT_SIM_TENSIX_COORDS` grid-pinning knob, diagnostics, structured tracing and
the deadlock watchdog — see **[`driver/wormhole/README.md`](driver/wormhole/README.md)**.

## Repository overview

The simulator implementation is in the [framework](https://github.com/hedgehope/wolfpine/tree/main/framework) directory, with the [driver](https://github.com/hedgehope/wolfpine/tree/main/driver) directory providing a range of examples that illustrate running the simulator. These are individually documented, but to summarise:

* [wormhole](https://github.com/hedgehope/wolfpine/tree/main/driver/wormhole) is where you likely want to go to. It provides an implementation of a Wormhole, currently with one DRAM tile and one Tensix block (although this is easy to expand, although will likely be slow!) It holds a set of example tt-metal programs that are built against a local tt-metal checkout and run against the simulator — exactly as they would run on hardware, only the device is the simulator. Each validates its own results, so the examples double as a test suite. See its README for how to build and run them.
* [blackhole](https://github.com/hedgehope/wolfpine/tree/main/driver/blackhole) is the Blackhole driver: the same wire bridge and machinery as wormhole (all shared in [`framework/bridge`](https://github.com/hedgehope/wolfpine/tree/main/framework/bridge)), pointed at a Blackhole device instead. Point `TT_METAL_SIMULATOR` at `driver/blackhole` to route execution there. Workers are materialised on demand by the wire bridge, so a program gets exactly the grid it launches on, up to the full 130; see its README.
* [simple](https://github.com/hedgehope/wolfpine/tree/main/driver/simple) are very basic examples, demonstrating the memory subsystem and running codes on a vanilla RV32IM CPU.

### What the timing model claims, and what it does not

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
left undone, is in [`ROADMAP.md`](ROADMAP.md).

Optional diagnostic information can be provided by each RISC-V baby core, the NoC, the Tensix co-processor and memory, enabling tracing of the execution of a program. This is currently at the instruction and architectural state level, but could be enhanced in the future to provide feedback to developers around potential code bottlenecks or other issues.

This is written in Python, mainly to make it easy for people to hackaround and experiment with things. If you want to add some functionality, or fix a bug, then please feel free to go ahead and raise a PR. This is still work in progress, so also please raise issues etc as you find them!

## Key parts of the simulator

There are a few of key components which are worth highlighting:

* [tt_device.Wormhole](https://github.com/hedgehope/wolfpine/blob/93da242e8a1a26160afaca43b0772bebc88b9171/tt_sim/device/tt_device.py#L111) creates the Wormhole, currently with a single DRAM tile and a tensix tile. Here you can see the NoC coordinates specified of each, and also the booleans provided to _TensixTile_ are whether to report diagnostic information (the first five for each RISC-V baby core, then next two for each NoC and then the separate Tensix co-processor choices). In the [wormhole](https://github.com/hedgehope/wolfpine/tree/main/driver/wormhole) tt-metal flow these are set from the `TT_SIM_DIAG_*` environment variables (see that directory's README).
* [tt_device.TensixTile](https://github.com/hedgehope/wolfpine/blob/93da242e8a1a26160afaca43b0772bebc88b9171/tt_sim/device/tt_device.py#L186) plumbs everything together within a Tensix tile, setting all the memory addresses and ranges for each individual component. These are all based on the ISA documentation [memory map](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/README.md). 
* [tt_noc](https://github.com/hedgehope/wolfpine/blob/main/framework/network/tt_noc.py) is the implementation of the NoC, it is not yet complete with all the functionality but is sufficient to communicate between tiles and, for example, read and write between DRAM and the Tensix tile.
* [rv](https://github.com/hedgehope/wolfpine/tree/main/framework/pe/rv) provides the RV32IM implementation with [.ttinsn extension](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/PushTensixInstruction.md#ttinsn-instruction-set-extension). This is fairly self explanatory, providing a pluggable approach to combining ISAs. The [BabyRISCV](https://github.com/hedgehope/wolfpine/blob/main/framework/pe/rv/babyriscv.py) ties this together for the baby RISC-V cores in the Tensix tile, for instance determining the initial PC value after a soft reset as per [here](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/SoftReset.md).
* [tensix](https://github.com/hedgehope/wolfpine/tree/main/framework/pe/tensix) is the implementation of the Tensix coprocessor as per [here](https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/TensixCoprocessor). This is not fully complete, but enough to run a range of codes that use the matrix, vector and scalar units (and all the associated unit implementation required to enable this). 
