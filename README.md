# Wolfpine

<img src="docs/wolfpine_logo.png" alt="Wolfpine logo" align="right" width="260">

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
- [How it works with tt-metal](#how-it-works-with-tt-metal)
- [Profiling your kernels](#profiling-your-kernels)
- [Documentation](#documentation)
- [Contributing](#contributing)

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

## How it works with tt-metal

tt-metal's UMD has a "simulation" chip backend: point `TT_METAL_SIMULATOR` at a simulator
directory and UMD launches Wolfpine in place of real silicon, so an ordinary tt-metal program
runs against it over a socket, exactly as it would on hardware. The simulator directories
ship inside the package, and `wolfpine run` / `wolfpine env` set that variable and the
handful of others tt-metal needs. To set them by hand, or for multi-tile grids,
diagnostics and every other knob, see the
[runbook](docs/running-tt-metal-on-the-simulator.md).

Ready-to-run example programs, plus a build-and-run test suite, live in
[`examples`](examples). Wolfpine is not cycle-accurate: its timing model is a
documented **floor**, every cycle charged at the low end of a published bound; see
[what it claims](docs/simulator-overview.md#what-the-timing-model-claims-and-what-it-does-not).

## Profiling your kernels

Every profiling and tracing mode is an environment variable, needs no code change, and
works on both Wormhole and Blackhole. The quickest answer to "where did my kernel's cycles
go?":

```bash
TT_SIM_COST_MODEL=1 TT_SIM_PROFILE=/tmp/myrun wolfpine run blackhole ./your_program
less /tmp/myrun/report.md
```

Beyond that ranked report there are Perfetto timelines, Parquet datasets, KCachegrind and
LCOV outputs, and tt-metal's own device profiler, NoC event trace, performance counters and
`DPRINT`. **[`docs/profiling.md`](docs/profiling.md)** covers each one: what to install,
how to turn it on, and how to read the output.

## Documentation

| Document | Read it for |
| --- | --- |
| [`docs/running-tt-metal-on-the-simulator.md`](docs/running-tt-metal-on-the-simulator.md) | **The runbook.** Environment, multi-tile grids, every server knob, diagnostics, the deadlock watchdog, `DPRINT`, limitations |
| [`docs/profiling.md`](docs/profiling.md) | **Profiling your kernels.** Every tracing and profiling mode, what to install, and example commands |
| [`docs/simulator-overview.md`](docs/simulator-overview.md) | How Wolfpine is put together: repository layout, the key parts of the simulator, and what the timing model claims |
| [`docs/trace-schema.md`](docs/trace-schema.md) | The stable, versioned schema for building tools on the trace outputs |
| [`examples/README.md`](examples/README.md) | The example tt-metal programs and their test suite |


## Contributing

This is written in Python, mainly to make it easy for people to hack around and experiment
with things. If you want to add some functionality, or fix a bug, then please feel free to
go ahead and raise a PR. This is still work in progress, so also please raise issues etc as
you find them!
