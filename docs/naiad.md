# Naiad: a RISC-V reference SoC

Naiad is Wolfpine's standalone RISC-V system-on-chip. It is not a Tenstorrent target and
there is no tt-metal host program: you describe a SoC in a YAML **design** file, compile a
freestanding C program against it, and run the ELF in the simulator until the program
exits.

A design picks:

- **the cores**: one or more RV32 or RV64 harts, with any of M, A, F, D, the RVV vector
  extension (at a VLEN you choose) and the Zb*/Zfh*/Zvf* sub-extensions;
- **a matrix unit** attached to every hart, in one of three styles:
  - **AME**, an attached matrix unit with its own tile and accumulator registers, in
    Naiad's own encodings or as the AME TG's Ztt v0.6 draft ISA itself (`mode: ztt`);
  - **IME**, the RISC-V Integrated Matrix Extension draft (Zvvm), which computes on vector
    registers;
  - **VME**, the Zvt v0.4 vector-matrix draft, which uses a separate tile buffer fed from
    vector registers;
- **the IP blocks on the bus**: a boot ROM, a CLINT, an optional PLIC, HTIF for the
  console and exit, scratchpads, DRAM, caches, a DMA engine, a perfcounters block, and even
  a whole Blackhole Tensix tile behind MMIO;
- **an optional cycle model** (`timing:`) that times the scalar pipeline, the vector unit,
  the matrix unit and every memory path.

Everything is Python, as with the rest of Wolfpine, so any part of a run can be inspected or
changed from a driver script. Naiad is also the platform the matrix-unit research runs on:
the Berkeley-dwarf kernels under `kernels/naiad/dwarfs/` come in scalar, RVV and
matrix-unit variants, and an experiment harness sweeps them across designs.

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Writing and building a program](#writing-and-building-a-program)
- [Running a program](#running-a-program)
- [Designs](#designs)
- [The cycle model](#the-cycle-model)
- [Traps and interrupts](#traps-and-interrupts)
- [Several harts](#several-harts)
- [Driving a kernel from Python](#driving-a-kernel-from-python)
- [Checking against Spike](#checking-against-spike)
- [The experiment harness](#the-experiment-harness)
- [Where things live](#where-things-live)
- [Limitations](#limitations)

## Requirements

- Python ≥ 3.10 and Wolfpine itself (`pip install -e .` from a checkout). Add the `naiad`
  extra (`pip install -e .[naiad]`, which pulls in `ml_dtypes`) for the bf16 and fp8
  formats, and the `experiments` extra for the sweep figures (matplotlib).
- **clang with the RISC-V target, and lld.** The Makefiles default to
  `/opt/llvm/20.1.7/bin/clang`; point `NAIAD_CC` at yours to override it. Any LLVM 20 or
  later works, and the matrix and Zvf* intrinsics need a recent one.
- **Optional:** [Spike](https://github.com/riscv-software-src/riscv-isa-sim) for the
  differential tests (`tools/naiad/build_spike.sh` builds and installs it to
  `~/.local/bin`).

Nothing Naiad does needs tt-metal.

## Quick start

From the repository root:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH

make -C driver/naiad run-ex1                  # build and run the first example
make -C driver/naiad run                      # build and run all seven
pytest driver/naiad/examples_test.py          # the same, as a test suite
```

The examples in [`driver/naiad/`](../driver/naiad/README.md) are the guided tour. Each is a
short C program on a packaged design that prints what it checks and ends with `PASS`:

| Example | Shows |
|---|---|
| `ex1` | console and exit over HTIF, and what `crt0` sets up |
| `ex2` | the CLINT timer |
| `ex3` | the DMA engine |
| `ex4` | reading the cycle model's counters from a program |
| `ex5` | an int8 GEMM on the AME matrix unit |
| `ex6` | four harts with a barrier |
| `ex7` | traps and interrupts |

To run an ELF yourself:

```bash
wolfpine run naiad --design naiad-dma driver/naiad/build/ex3/ex3.elf
# from a checkout without the console script:
python -m framework.cli run naiad --design naiad-dma driver/naiad/build/ex3/ex3.elf
```

## Writing and building a program

A Naiad program is freestanding C (no libc) built against files generated from its design:

```bash
python -m framework.naiad.gen framework/naiad/designs/naiad-base-rv64.yaml -o gen/
```

This writes four files:

| File | Contents |
|---|---|
| `naiad.h` | every block's base address and the design's parameters as `#define`s, e.g. `NAIAD_HAS_DMA`, the VLEN and the matrix shape |
| `linker.ld` | the memory map, with the program in the block named by `program.memory` |
| `crt0.S` | the reset code: stack per hart, `.bss`/`.data`, releasing the secondary harts, the call to `main` |
| `naiad.mk` | the `-march`, `-mabi` and code model for clang |

Designs with a matrix unit also get `naiad_gemm.h`, an emitted GEMM for that unit's shape.

The easiest way to build is the kernels Makefile. It regenerates those files whenever the
design changes, compiles every `.c` file in the program's directory together with the
platform's small libc, and links:

```bash
make -C kernels/naiad PROG=hello DESIGN=framework/naiad/designs/naiad-base-rv64.yaml
make -C kernels/naiad PROG_DIR=$PWD/myprog DESIGN=path/to/my-design.yaml run
```

The program-side API is
[`framework/naiad/platform/naiad_platform.h`](../framework/naiad/platform/naiad_platform.h):

- **Console:** `naiad_putchar`, `naiad_puts` and `naiad_printf`. `naiad_printf` handles the
  usual integer, string and char conversions, with width and `-`.
- **Exit:** `naiad_exit(code)`; returning from `main` does the same.
- **Harts:** `naiad_hartid`, `naiad_barrier`, `naiad_lock` and `naiad_unlock`.
- **Counters:** `naiad_cycles`, `naiad_instret`, `naiad_mtime`, `naiad_counter(EVENT)`
  (through Zihpm) and `naiad_perf_read`.

The matrix units' instructions are wrapped as inline functions in
`kernels/naiad/include/matrix.h`.

An ELF must run on the design it was built for, or on one with the same memory map and ISA.

## Running a program

```bash
wolfpine run naiad [--design D] [--set KEY=VALUE]... [--max-cycles N]
                   [--dump ADDR:LEN] [--counters-json PATH] [--count-all] prog.elf
```

- **Output:** stdout is exactly what the program printed. The report goes to stderr: exit
  code, cycles and every counter event.
- **Exit code:** the process exits with the program's exit code, or 124 if `--max-cycles`
  (default 50M) ran out first.
- **`--design`:** a packaged design's name (`--list-designs` lists them) or a path to a
  design YAML. The default is `naiad-dwarfs`.
- **`--set key.path=value`:** overrides one design field for this run; it is repeatable and
  the value is parsed as YAML. A path segment names a mapping key, or a list element by its
  `name` (`blocks.l1.params.size=…`). Keep to fields the ELF does not bake in, such as
  timing. Moving a block changes the memory map the ELF was linked against, and the hart
  count is compiled into `crt0.S`.
- **`--dump ADDR:LEN`:** hexdumps memory after the run; `ADDR` may be an ELF symbol.
- **`--counters-json PATH`:** writes the outcome and every counter event as JSON.
- **`--count-all`:** also counts the on-demand event families (instruction classes, bus
  bytes per block).

```bash
# ex4 with the cycle model on: cycles and IPC change, instret does not.
wolfpine run naiad --design naiad-counters --set timing.scalar.load=4 driver/naiad/build/ex4/ex4.elf
# ex5 with a slower L1.
wolfpine run naiad --design naiad-ame-timed --set timing.memory.l1.latency=30 driver/naiad/build/ex5/ex5.elf
```

## Designs

The packaged designs live in
[`framework/naiad/designs/`](../framework/naiad/designs). Each file opens with comments
that explain its choices and, for the timed ones, where every number comes from.

| Design | What it is |
|---|---|
| `naiad-base` / `naiad-base-rv64` | the minimum: one RV32IM / RV64IM hart, boot ROM, CLINT, HTIF, RAM |
| `naiad-rv64gc`, `naiad-fp-rv64` | RV64 with the general-purpose and floating-point extensions |
| `naiad-dma`, `naiad-counters` | add the DMA engine / the perfcounters block |
| `naiad-rvv`, `naiad-rvv-timed` | an RVV 1.0 vector unit, untimed / timed |
| `naiad-ame`, `naiad-ame-timed` | the attached matrix unit, untimed / timed (calibrated against the Tensix as Wolfpine models it) |
| `naiad-ime`, `naiad-ime-timed` | the integrated (Zvvm) matrix unit |
| `naiad-vme`, `naiad-vme-timed` | the vector-matrix (Zvt) unit |
| `naiad-ztt` | an AME unit restricted to the Ztt profile |
| `naiad-ztt06` | the AME TG's Ztt v0.6 draft ISA itself (`mode: ztt`): its encodings, registers, datatype descriptors and CSRs; untimed |
| `naiad-dwarfs` | the base of every experiment: RV64GCV at VLEN 256 (a SpacemiT K1 X60's vector shape), F/D, a 16×16×16 AME unit, 1.5 MiB L1, DRAM, timed |
| `naiad-tensix` | `naiad-dwarfs` plus a Blackhole Tensix tile and a DMA engine on the bus |

A design has five top-level sections:

```yaml
name: my-design
cores:
  count: 1                 # harts
  xlen: 64                 # 32 or 64
  isa: [i, m, a, f, d, v, zicsr, zicntr]
  vlen: 256                # with v (or zve64d)
  elen: 64
  reset_pc: 0x1000
program:
  memory: l1               # the block the program, data and stacks go in
  stack_size: 64KiB
matrix:                    # optional: one unit per hart
  mode: ame                # ame | ime | vme
  m: 16
  n: 16
  k: 16
  tiles: 8
  accs: 4
  formats: {in: [fp32, bf16, int8], acc: [fp32, int32]}
  features: [transpose, shift, epilogue]
timing: { ... }            # optional: see below; without it the run is untimed
blocks:
  - {name: bootrom, type: bootrom, base: 0x1000, window: 0x1000}
  - {name: clint, type: clint, base: 0x02000000, window: 0x10000}
  - {name: l1, type: scratchpad, base: 0x10000000, window: 2MiB, params: {size: 0x180000}}
  - {name: dram, type: dram, base: 0x80000000, window: 1GiB}
  - {name: htif, type: htif}
```

### Blocks

Each block has a `name`, a `type`, a `base` and a `window` (the address space reserved for
it), plus optional `params`. What the block actually maps must fit inside its window. The
HTIF block takes neither `base` nor `window`, because it lives on the ELF's
`tohost`/`fromhost` symbols. New IP is added by registering a class in
[`framework/naiad/registry.py`](../framework/naiad/registry.py) and naming its type in a
design; the builder never special-cases a block by name.

| Type | Block |
|---|---|
| `bootrom` | the reset vector; jumps to `crt0` |
| `scratchpad`, `dram` | memories; `timing.memory` gives each its latency and bandwidth |
| `cache` | a cache in front of a memory block (unmapped: it takes no address) |
| `clint` | `mtime`, `mtimecmp` and `msip` per hart |
| `plic` | PLIC-lite: up to 64 level-triggered external interrupt sources, with `PENDING`, `CLEAR` and per-hart `ENABLE` registers; `params: {schedule: [[cycle, source], ...]}` raises sources at fixed cycles. Needs a `clint` |
| `htif` | console and exit, the protocol Spike speaks |
| `perfcounters` | memory-mapped counters for the cost model's events |
| `dma` | a multi-channel 1D/2D DMA engine with descriptors and a `DONE` poll |
| `tensix` | a Blackhole Tensix tile behind MMIO: its L1 window and control registers |

### The matrix unit

`matrix.mode` picks the unit:

- **`ame`** — the attached matrix unit: tile registers and accumulators of its own, loaded
  from and stored to memory, with `mma` instructions in the custom opcode space. The
  encodings are data-driven from
  [`framework/matrix/encodings.yaml`](../framework/matrix/encodings.yaml).
  - **Features.** `matrix.features` adds optional instruction groups: `transpose`,
    `shift`, `shift2`, `epilogue`, `semiring` (min-plus and max-plus `mma`), `acc_move`,
    `convert`, `mma_transpose`, `reduce`, `ewise`, `ewise_tx` and `halo`.
  - **The Ztt profile.** `profile: ztt` restricts the unit to the Ztt draft's instruction
    set, so it rejects the features Ztt does not have. It keeps Naiad's own encodings; for
    the draft's encodings use `mode: ztt`.
  - **Accumulator loads and stores** are on unless `acc_mem: false`.
- **`ime`** — the RISC-V Integrated Matrix Extension TG's Zvvm draft: `C += A·Bᵀ` computed
  on vector register groups, with the tile geometry set through new `vtype` fields
  (`lambda`, `bs`, `altfmt`). The core needs the vector extension.
- **`vme`** — the Zvt v0.4 draft: a tile buffer of its own, fed with row and column moves
  from vector registers.
- **`ztt`** — the RISC-V AME TG's Ztt v0.6 draft as written: its 138 custom-1 encodings, so
  a Ztt binary runs unchanged. See [The Ztt mode](#the-ztt-mode) below.

The supported formats are fp64, fp32, fp16, bf16, fp8 and int8. A program checks a format
pair at run time through `mcfg`, the unit's configuration query, and the generated
`naiad.h` lists the formats the design has.

### The Ztt mode

`matrix.mode: ztt` implements the AME TG's draft `riscv/riscv-attached-matrix-extension`
tag `v0.6` (commit `8d27acd`). Its encodings live in
[`framework/matrix/ztt_encodings.yaml`](../framework/matrix/ztt_encodings.yaml), imported
from the draft's `INSTRUCTION_ENCODINGS.md` by `tools/naiad/ztt_import.py <checkout>`.
Definitions are in [`ztt.py`](../framework/matrix/ztt.py) and execution in
[`ztt_isa.py`](../framework/matrix/ztt_isa.py).

```yaml
matrix:
  mode: ztt
  nelem: 256          # AME_NELEM: N x N squares, N = 16
  unit_bits: 8        # AME_UNIT_DATATYPE_SIZE: int8 is one M register, bf16 two, fp32 four
  m_regs: 16          # AME_NUM_M_REGS: 16 or 32
  acc_regs: 8         # AME_NUM_ACC_REGS: 1, 2, 4, 8 or 16
  datatypes: [int8, uint8, int32, fp16, bf16, fp32, fp64]
  matmul: [[int8, int8, int32], ...]   # optional; default int8/uint8 -> int32, narrow floats -> fp32, fp64
  backend: dedicated  # or shared: one backend, and one owner at a time, for every hart
```

- **State and access.** The unit has M registers with `Md` descriptors, accumulators with
  `Ad` descriptors, and the seven CSRs at the draft's provisional addresses. An instruction
  traps unless `mstatus.MS` is non-Off and the hart has run `ame.acquire`.
  `kernels/naiad/include/ztt.h`'s `ztt_enable()` does both.
- **Checks, in the draft's order:**
  1. An illegal register index traps.
  2. An unsupported datatype or operation tuple sets `amestatus.UN` and changes nothing
     else; it does not trap.
  3. A misaligned register group traps.
  4. A misaligned base address raises cause 4 or 6.
  5. An unmapped segment raises cause 5 or 7, with `mtval` the faulting address. Stores are
     checked in full first, so a faulting one writes nothing.
- **Implemented:**
  - acquire and release, and setting and reading datatypes;
  - every load and store form, zeroing, and the moves (`mmov.*`, `mbcast.m.x`, `mmove*`);
  - the six matrix multiplies;
  - `mconv.ew`; add, sub, mul, max and min, element-wise and with a scalar;
  - row and column shifts, reductions and prefixes.

  Every other instruction decodes and reports UN, as the draft allows.
- **Numerics:**
  - **Matrix multiplies** fold N products into the accumulator in order, unfused (each
    product rounded, then added), and do not set `amefflags`.
  - **Element results** round to nearest even and accrue `amefflags`.
  - **Integers** wrap, or saturate and set `amexsat`.
- **Naiad's interpretations of the draft:**
  - The draft gives `mstatus.MS` no bit position; Naiad puts it at bits 26:25
    (`ms_shift`), with SD.
  - `amestype`, `amefflags` and `amexsat` are writable, though their provisional addresses
    are in the read-only CSR range.
  - Packed datatypes are not modelled.
- **Not modelled:** timing (the mode is untimed), address translation and custom datatypes.

`kernels/naiad/ztt_gemm` runs blocked int8 and bf16 GEMMs with the draft's instructions and
checks them against scalar references.

## The cycle model

A design without a `timing:` section runs **untimed**: every instruction takes one cycle and
memory is free, which is the fastest way to check a program's correctness. Adding the
section (or any `--set timing.<field>`) turns the cycle model on, using the defaults for
every field you do not set. Note that **any one** `timing.*` override turns the **whole**
model on, and the run says so on stderr. To vary a single field in isolation, start from a
design that already has a `timing:` section, such as one of the `*-timed` designs or
`naiad-dwarfs`.

The model is in-order. A hart issues one instruction a cycle by default, with
per-instruction-class latencies and scoreboarding; the vector unit and the matrix unit run
decoupled behind their own queues and pipes, and every memory block has a latency and a
bandwidth. `naiad-dwarfs.yaml` is the reference for every field, with a sourced value and
comment for each. The sections are:

| Section | Sets |
|---|---|
| `timing.scalar` | `alu`, `mul`, `div`, `load`, `store`, `branch_taken`, `csr`, `fp_add`/`fp_mul`/`fp_fma`/`fp_div`/`fp_sqrt`, `issue_width`, `trap_entry`, `trap_return` |
| `timing.memory` | per block name or block type: `latency`, `bandwidth` (bytes a cycle, `null` = unlimited), plus the ordering-hazard switches below |
| `timing.vector` | `datapath_bits`, `startup`, `div_ii`, `mem_startup`, `mem_bytes_per_cycle`, `indexed_elements_per_cycle`, `chaining` |
| `timing.matrix` | `macs_per_cycle`, `format_passes`, `pipeline_depth`, `acc_forwarding`, `overlap`, the tile load and store bandwidths and latencies, and per-instruction costs |

Several behaviours are **opt-in** and off by default, so that changing one is a controlled
experiment:

| Option | What it turns on |
|---|---|
| `timing.scalar.issue_width: 2` | dual issue: an ALU instruction pairs with anything else; memory, mul/div, FP and vector/matrix dispatch take one pipe each. A branch ends the issue group, and a RAW hazard or a busy unit blocks the pair |
| `timing.vector.chaining: true` | a dependent vector instruction starts as soon as the first element group of its source is ready, rather than waiting for the whole result |
| `timing.memory.hazards: true` | all four memory-ordering hazards below |
| `timing.memory.scalar_stores` | a load waits for an overlapping in-flight scalar store |
| `timing.memory.dma_coherence` | a core access waits for an overlapping DMA transfer to complete |
| `timing.memory.war`, `timing.memory.waw` | write-after-read and write-after-write ordering for matrix-unit and DMA transfers |

The cycle model is a model, not RTL. The numbers in `naiad-dwarfs` are calibrated against
Wolfpine's Tensix cost tables where the Tensix has a counterpart, and every uncalibrated
value says so in the design file.

### Counting events

Every run counts:

- cycles and retired instructions;
- stall cycles by reason;
- bytes per path (memory to tile, tile to memory, memory to and from the accumulators, per
  block);
- matrix `mma`s and MACs.

A program reads these counts through Zihpm (`naiad_counter`) or the perfcounters block. The
host gets them through the stderr report or `--counters-json`. To time only part of a
program, read `mcycle` around it and leave the difference in an output array, as the dwarf
kernels do (`kernels/naiad/include/dwarf_util.h`).

## Traps and interrupts

Naiad harts deliver machine-mode traps
([`framework/pe/rv/trap.py`](../framework/pe/rv/trap.py)), on RV32 and RV64 alike.
Delivery is **armed by writing a non-zero `mtvec`**. Until then nothing changes: `ecall`
and `mret` are nops, and a fault such as an illegal instruction or an unmapped access ends
the run with an error. Once armed:

- **Exceptions:** illegal instruction, `ecall`, `ebreak`, misaligned loads, stores, AMOs and
  jump targets, misaligned vector element accesses (with `vstart` set to the faulting
  element), and access faults on unmapped addresses. They set `mepc`, `mcause`, `mtval` and
  `mstatus.MIE/MPIE/MPP`, and `mret` returns. A trapping jump does not write its link
  register.
- **Interrupts**, level-triggered and taken when `mstatus.MIE` and the matching `mie` bit
  are set:
  - machine software (`msip`) and machine timer (`mtime >= mtimecmp`) from the CLINT;
  - machine external (cause 11) from a `plic` block, at the highest priority.

  Vectored `mtvec` mode works.
- **`wfi`** sleeps until an enabled interrupt is pending.
- **Cost:** trap entry and `mret` are charged as a pipeline redirect
  (`timing.scalar.trap_entry`, default 3 cycles, and `trap_return`, default 2). A program
  that never traps pays nothing.

`driver/naiad/ex7` is the worked example, and agrees with Spike.

## Several harts

`cores.count: N` gives N identical harts sharing the bus. `crt0` gives each its own stack
and holds the secondary harts until hart 0 releases them through `msip`. The platform
header provides `naiad_hartid()`, `naiad_barrier()` and a lock, and the A extension's AMOs
work across harts. Under the cycle model the harts contend for each memory block's
bandwidth. The hart count is compiled into `crt0.S`, so a multicore program needs a design
file of its own; `driver/naiad/ex6/naiad-quad.yaml` is an example.

## Driving a kernel from Python

[`framework/naiad/kernel_io.py`](../framework/naiad/kernel_io.py) is the host side of a
simple contract: the kernel reads its inputs from global arrays and leaves its outputs in
others, each named by an ELF symbol. `run_kernel` builds the design, loads the ELF, writes
numpy arrays over the input symbols, runs to the exit and returns the output arrays together
with the run's counters, so a kernel's data is generated and checked in numpy rather than in
C. `driver/naiad/ex5/ex5.py` is a worked example:

```bash
python3 driver/naiad/ex5/ex5.py driver/naiad/build/ex5/ex5.elf --shape 48x33x60 --seed 1
```

The simulator itself is a Python object, so a driver can equally build the SoC with
`framework.naiad.builder.build`, step it, and read registers and memory between steps.

## Checking against Spike

```bash
python -m framework.naiad.difftest <design.yaml> <prog.elf> [--commits] [--ignore REGEX]...
```

This runs the same ELF on Naiad and on Spike and compares the console output and the exit
code. With `--commits` it compares the two retirement streams instruction by instruction
(pc, instruction word, destination register and value, and on a vector core every vector
register Spike logs), and reports the first divergence. Values derived from counters and
MMIO reads are excluded from the comparison, since the two simulators legitimately differ
there.

Spike is found as `$NAIAD_SPIKE`, then on `PATH`, then in `~/.local/bin`. Spike has neither
Zvvm nor Zvt, so programs that use the IME or VME units cannot be diff-tested; those units
are tested against their drafts in `framework/matrix/`.

`tools/naiad/` also runs the RISC-V architecture tests and the vector tests, and holds an
RVV fuzzer and scripts for native runs on a SpacemiT K1 board.

## The experiment harness

The harness in [`experiments/naiad/`](../experiments/naiad) runs the dwarf kernels
(`kernels/naiad/dwarfs/`: gemm, jacobi, fft, spmv, scan, reduce, nbody, apsp, tensorprod)
across design points. Each dwarf has a `spec.py` that lists its sizes, formats and variants
(`scalar`, `rvv`, `ame`, `ime`, `vme`, and alternative formulations), and generates and
checks its data against an fp64 golden.

A **sweep** (`experiments/naiad/sweeps/*.yaml`) is a list of studies. Each study picks dwarfs,
variants, sizes and formats, and derives design points from `naiad-dwarfs` by overriding
keys (a Cartesian product of value lists) without ever moving an address. `normalise:` sets
the matrix unit's shape and timing for a target peak, so that a study can sweep the ratio
of matrix to vector peak throughput directly. The header of
[`experiments/naiad/sweep.py`](../experiments/naiad/sweep.py) documents every key.

```bash
python -m experiments.naiad.run experiments/naiad/sweeps/dwarfs_tier2.yaml --jobs 8
python -m experiments.naiad.run dwarfs_tier2 --quick           # smallest sizes, a few ratios
python -m experiments.naiad.run dwarfs_tier2 --set timing.memory.hazards=true
python -m experiments.naiad.plot dwarfs_tier2                  # figures into out/<sweep>/figures/
python -m experiments.naiad.all                                # the headline sweeps, then every figure
```

- **Resuming:** each point's result is stored the moment it finishes
  (`experiments/naiad/out/<sweep>/points/`), so an interrupted sweep resumes where it
  stopped. `--force` reruns everything and `--retry-failed` reruns the failed points.
- **Outputs:** `results.csv` (one row per point), `crossover.csv` (the matrix:vector ratio
  at which a reformulation starts to beat the baseline), `precision.csv` and `meta.json`.
- **Figures:** `plot` draws speedup against the ratio, bytes per path, accuracy per format,
  redundancy (matrix FLOPs over useful FLOPs) and coupling.

**The cost model.** [`framework/naiad/costmodel`](../framework/naiad/costmodel) predicts a
kernel's cycles from its loop nest and the design's `timing:` section, without simulating
it. `choose` picks the fastest implementation and `crossover` predicts the ratio at which
one starts to pay; this is the decision a compiler would make. The model is checked against
the simulator with:

```bash
python -m experiments.naiad.costmodel_validate [--sweeps ...] [--extra DIR ...]
```

which reports the median and maximum error, how often the predicted choice matches the
simulated one, and the predicted against the simulated crossovers.

## Where things live

| Path | Contents |
|---|---|
| `framework/naiad/` | design parser, builder, generators (`gen/`), runner, `kernel_io`, difftest, the packaged designs, the platform header and libc, the GEMM emitters (`emit/`) and the cost model (`costmodel/`) |
| `framework/ip/` | the generic IP blocks: CLINT, PLIC, HTIF, DMA, caches, memories, perfcounters, the Tensix block |
| `framework/pe/rv/` | the RISC-V cores, their ISA modules (`isa/`, with RVV in `isa/v/`), traps, and the cycle model (`timing/`) |
| `framework/matrix/` | the AME, IME, VME and Ztt units: encodings, ISA executors, numerics, formats, timing |
| `kernels/naiad/` | bring-up and feature kernels, the shared `include/matrix.h`, and the dwarfs |
| `driver/naiad/` | the seven guided examples |
| `experiments/naiad/` | the sweep harness, its sweeps and tests |
| `tools/naiad/` | Spike, architecture-test and vector-test runners, the RVV fuzzer, K1 tools |

## Limitations

- **Machine mode only.** There is no S- or U-mode, no virtual memory and no PMP, and the
  PLIC is a minimal one, not the RISC-V PLIC register map.
- **The cycle model is a model, not RTL.** It is in-order with at most dual issue, and has no
  branch predictor or instruction cache. The matrix units' drafts specify behaviour, not
  timing, so their timing is a parametric model calibrated where there is something to
  calibrate against.
- **Tile instructions are not checked for misalignment.** That covers the Zvvm and Zvt tile
  loads and stores.
- **Draft extensions are pinned.** IME follows the Zvvm draft, VME the Zvt v0.4 draft and
  Ztt the AME v0.6 draft, at the commits named in `framework/matrix/ime.py`, `vme.py` and
  `ztt_encodings.yaml`, and will move as the drafts do.
