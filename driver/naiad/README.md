# Naiad driver

Naiad is Wolfpine's standalone RISC-V reference SoC: one or more RV32/RV64 harts
(optionally with F/D, RVV, and an attached AME / IME / VME matrix unit) plus generic
IP blocks (boot ROM, CLINT, HTIF, DMA, scratchpads, caches, a perfcounters block),
assembled from a YAML **design** file. It is not a Tenstorrent target, and there is
no tt-metal host program: you compile a freestanding C program against the design and
run the ELF to its HTIF exit. The SoC target lives in [`framework/naiad`](../../framework/naiad)
(design parser, builder, generators, runner); the generic IP lives in `framework/ip`;
the packaged designs are in [`framework/naiad/designs`](../../framework/naiad/designs).
The program-side API (console, exit, counters, harts, barrier) is
`framework/naiad/platform/naiad_platform.h`.

## Building and running

You need `make` and clang with the RISC-V target (default `/opt/llvm/20.1.7/bin/clang`;
override with `NAIAD_CC=`). From the repository root:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH

make -C driver/naiad                 # build every example into driver/naiad/build/<ex>/<ex>.elf
make -C driver/naiad run             # build and run every example
make -C driver/naiad run-ex3         # build and run one
make -C driver/naiad ex3             # just build it
pytest driver/naiad/examples_test.py # build + run each, assert PASS (skipped without clang)
```

The Makefile has no compiler flags of its own: each example is built by
`kernels/naiad/Makefile`, which runs `python -m framework.naiad.gen <design.yaml> -o <dir>`
to write the design's `naiad.h`, `linker.ld`, `crt0.S` and `naiad.mk` (the `-march` /
`-mabi` come from `naiad.mk`) and emits `naiad_gemm.h` for designs with a matrix unit.

To run an ELF by hand, use the `run naiad` target of the Wolfpine CLI (`wolfpine run naiad`
when Wolfpine is pip-installed):

```bash
python -m framework.cli run naiad --design naiad-dma driver/naiad/build/ex3/ex3.elf
```

The program's console goes to stdout; the report (exit code, cycles, every counter
event) goes to stderr, and the process exits with the program's exit code (124 if
`--max-cycles` ran out). Other options: `--dump ADDR:LEN` (ADDR may be a symbol),
`--counters-json PATH`, `--count-all`.

### Picking a design and overriding fields

`--design` takes a packaged design's name (`--list-designs` lists them) or a path to
a design YAML; the default is `naiad-dwarfs`. An ELF must be run on the design it was
built for (or one with the same memory map and ISA). `--set key.path=value`
(repeatable, value parsed as YAML) overrides one field at run time; a path segment
names a mapping key, or a list element by its `name`:

```bash
# ex4 with the cycle model turned on: cycles and IPC change, instret does not.
python -m framework.cli run naiad --design naiad-counters --set timing.scalar.load=4 driver/naiad/build/ex4/ex4.elf
# ex5 with a slower L1 (about 40k -> 109k cycles for the AME GEMM).
python -m framework.cli run naiad --design naiad-ame-timed --set timing.memory.l1.latency=30 driver/naiad/build/ex5/ex5.elf
```

Keep `--set` to timing and other fields the program does not bake in: moving a block
changes the memory map the ELF was linked against, and the hart count is compiled into
`crt0.S` (ex6 therefore ships its own design file rather than using
`--set cores.count=4`).

## The examples

Each example is a short C program that prints what it checks and ends with `PASS` or
`FAIL`; the exit code is 0 only on `PASS`.

| | Design | Shows |
|---|---|---|
| [ex1](ex1/ex1.c) | `naiad-base` (RV32IM) | Console and exit codes over HTIF (`tohost`/`fromhost`); what crt0 and the loader set up (.bss, .data, .rodata, stack); `naiad_exit` from a helper function. |
| [ex2](ex2/ex2.c) | `naiad-base-rv64` | The CLINT: `mtime` tracking the clock, arming `mtimecmp`, `mie.MTIE`, and the `while (mtime < target) wfi;` idiom. Naiad models wake-up, not traps: no interrupt handler runs, and on one hart `wfi` is a nop, so the loop polls. |
| [ex3](ex3/ex3.c) | `naiad-dma` | The DMA engine: a 1D DRAM-to-scratchpad copy and a 2D tile gather, each polled to `DONE` and verified word by word, with the transfer's cycle cost. |
| [ex4](ex4/ex4.c) | `naiad-counters` | Reading the cost model's numbers from a program: `mcycle`/`minstret`, `naiad_counter(EVENT)` through Zihpm, and the perfcounters block, around a dot product. The loop's instruction and DRAM byte counts are checked exactly. |
| [ex5](ex5/ex5.c) | `naiad-ame-timed` | An int8 to int32 GEMM on the AME unit through the emitted `naiad_gemm.h` (ragged sizes, padded edge tiles), checked against a scalar loop in the program, with both timed. [`ex5.py`](ex5/ex5.py) is a host driver that feeds the same ELF numpy inputs through `framework.naiad.kernel_io` and checks C on the host. |
| [ex6](ex6/ex6.c) | [`ex6/naiad-quad.yaml`](ex6/naiad-quad.yaml) (4 x RV64IMA) | Several harts: crt0's msip release, each hart sleeping on its own timer (here `wfi` really sleeps), per-hart partial sums, an AMO, and `naiad_barrier()`. |

To run ex5's host side by hand: `python3 driver/naiad/ex5/ex5.py driver/naiad/build/ex5/ex5.elf --shape 48x33x60 --seed 1`.

## Where the research harness lives

These examples are the introduction. The Naiad studies use the same build and run
machinery at scale:

- `kernels/naiad/` holds the bring-up and feature kernels (`hello`, `dma`, `counters`,
  `gemm_ame`, `multicore`, `wfi_timer`, ...) and, under `kernels/naiad/dwarfs/`, the
  dwarf kernels (gemm, jacobi, fft, spmv, scan, reduce, ...), each in scalar, RVV and
  matrix-unit variants.
- `experiments/naiad/` is the experiment harness: sweeps (`experiments/naiad/sweeps/*.yaml`)
  derive design points from `naiad-dwarfs` by overriding keys, `python -m experiments.naiad.run`
  builds and runs them through `framework.naiad.kernel_io`, and its tests are under
  `experiments/naiad/tests/`.
- `tools/naiad/` has the cross-checking tools: Spike build scripts, the RISC-V
  architecture-test and vector-test runners, the RVV fuzzer, and native runs on a
  SpacemiT K1.
