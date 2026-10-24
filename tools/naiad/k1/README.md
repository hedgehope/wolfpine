# K1 kernel pack: Naiad's RVV kernels on real RVV 1.0 hardware

The SpacemiT K1 (8x X60/"A60" cores, RV64GCV, VLEN 256, RVV 1.0) is the real-hardware
check for Naiad's vector unit. This pack builds the **same RVV-intrinsics sources** for
three targets and diffs their results:

| target | build | runs on |
|---|---|---|
| Naiad | freestanding, through `kernels/naiad/Makefile` against `naiad-rvv.yaml` | Naiad (`make run-naiad`) and Spike (`make run-spike`) |
| Linux | hosted, `-DNAIAD_PLATFORM_LINUX` (`framework/naiad/platform/naiad_platform_linux.h`) | the K1 board, or `qemu-riscv64` (`make CROSS=1 run-qemu`) |

## Kernels

`kernels/<name>/<name>.c`, sharing `kernels/include/k1common.h`:

| kernel | what it exercises | cases |
|---|---|---|
| `axpy` | `vfmacc.vf`, fp32 at LMUL 2 and fp64 at LMUL 1 | n = 1, 13, 64, 257 |
| `dot` | `vfredosum` per strip (exact), `vfwmul` + fp64 ordered sum (exact), vector accumulator + `vfredusum` (tolerance) | n = 1, 31, 32, 33, 200, 1000 |
| `gemm` | the RVV outer-product micro-kernel (MR = 4, NR = 16, `vfmacc.vf` / `vmacc.vx`) driving a 37x45x33 GEMM with M/N edges, fp32 and int32 | 1 |
| `reduce` | `vredsum`/`vredmax`/`vredminu`, `vwredsum` (int16), `vfredosum`/`vfredmax`/`vfredmin`, `vfredusum` (tolerance) | n = 1, 17, 32, 33, 999 |
| `gather` | `vlse32` (matrix column), `vluxei32`/`vloxei32` (permutation, repeats), `vsoxei32` scatter, `vlseg2e32`/`vsseg2e32`, `vrgather.vv`, `vcompress.vm` | 1 |
| `rvv_check` (optional) | C1's self-checking RVV test (`kernels/naiad/rvv_check`); needs Zvfh, Zvfbfmin and Zvfbfwma | `make rvv_check` |
| `xsmt_gemm` (optional) | int8 GEMM on SpacemiT's IME instructions (`smt.vmadot`), see below | `make xsmt` |

Inputs come from a fixed LCG and are exactly representable, and every strip-mined loop
asks for `vl = min(remaining, VLMAX)`: for an AVL in (VLMAX, 2 VLMAX) the spec lets
hardware choose any vl from ceil(AVL/2) up to VLMAX, which would change an accumulating
kernel's rounding. So every `R` result must match **bit for bit** on any conforming
RVV 1.0 implementation with VLEN = 256. The one exception is `vfredusum`, whose order is
implementation-defined; its results are `A` lines, compared with a tolerance.

### Output format

```
# kernel axpy vlenb 32 clock ns        comment (kernel header: VLENB and the timer unit)
R axpy_f32_n13 4 3e8a1c20              exact result: case, index, hex bit pattern
A dot_vacc 2 41a2b000 f32              tolerance result (vfredusum)
T gemm_f32_37x45x33 123456 ns 200      time for <reps> calls (not compared)
# FAILED gather exit 139               a kernel that did not exit 0
```

`compare.py REF OTHER [--rtol 1e-5] [--times]` prints OK or DIFF (with the count and the
first differing element) per case, and `--times` prints the timings per call side by side.
It uses only the Python standard library, so it runs on the board too.

## Running on the board: the exact steps

On the host (this repo):

```bash
make -C tools/naiad/k1 pack                  # -> tools/naiad/k1/k1-pack.tar.gz
scp tools/naiad/k1/k1-pack.tar.gz k1:
```

On the K1 (any Linux with an RVV 1.0 toolchain: GCC >= 13 or clang >= 16 for the
`__riscv_` v1.0 intrinsics; Bianbu's and Debian's packaged compilers work):

```bash
grep -m1 isa /proc/cpuinfo                   # must list v (rv64imafdcv...)
tar xzf k1-pack.tar.gz && cd k1
make                                         # build/linux/{axpy,dot,gemm,reduce,gather}
make rvv_check                               # optional; needs zvfh_zvfbfmin_zvfbfwma
make xsmt                                    # optional; needs GCC >= 16.1 / LLVM >= 22.1
TASKSET="taskset -c 0" ./run_k1.sh > k1.out  # pinned to one core for steadier timings
python3 compare.py k1.out k1.out             # sanity: parses, all OK
```

`make CC=clang` picks another compiler, `MARCH=rv64gcv_zvl256b` adds flags. Timing:
`make TIMER=perf` uses a perf-event cycle counter (`kernel.perf_event_paranoid <= 2`,
the usual default); `make TIMER=rdcycle` reads the CSR directly, which Linux >= 6.6
traps unless `sudo sysctl kernel.perf_user_access=2`. The default is
`CLOCK_MONOTONIC_RAW` nanoseconds. Rebuild with `make clean` after changing `TIMER`.

Back on the host:

```bash
scp k1:k1/k1.out .
make -C tools/naiad/k1 run-naiad             # -> tools/naiad/k1/build/naiad.out (slow: Python)
make -C tools/naiad/k1 run-spike             # -> tools/naiad/k1/build/spike.out
python3 tools/naiad/k1/compare.py tools/naiad/k1/build/spike.out k1.out --times
python3 tools/naiad/k1/compare.py tools/naiad/k1/build/naiad.out k1.out --times
```

The Naiad ELFs are built for `naiad-rvv.yaml` (`DESIGN=...` for another design). The
timings are not comparable as numbers (Naiad reports its modelled cycles, the board
nanoseconds or cycles), but their ratios across kernels are a sanity check of the
vector timing model.

## Cross-compiling and QEMU (no board needed)

There is no system riscv64 Linux toolchain here, but clang 20.1.7 cross-compiles with a
sysroot made from Ubuntu's cross packages, which need no root:

```bash
mkdir -p ~/.local/share/riscv64-sysroot-debs && cd ~/.local/share/riscv64-sysroot-debs
apt-get download libc6-riscv64-cross libc6-dev-riscv64-cross linux-libc-dev-riscv64-cross \
                 libgcc-12-dev-riscv64-cross libgcc-s1-riscv64-cross
mkdir -p ../riscv64-xroot && for d in *.deb; do dpkg-deb -x "$d" ../riscv64-xroot; done
make -C tools/naiad/k1 CROSS=1               # static binaries; XROOT=... for another sysroot
```

Ubuntu 22.04's `qemu-user-static` (6.2) only implements the RVV 0.7.1 draft. Debian
bookworm's 7.2 runs RVV 1.0, and its static binary works on this host: download
`qemu-user-static_7.2+dfsg-7+deb12u18+b3_amd64.deb` from packages.debian.org and
`dpkg-deb -x` it into `~/.local/share/qemu-bookworm-deb/x` (the `QEMU` default). Then:

```bash
make -C tools/naiad/k1 CROSS=1 run-qemu      # -> build/qemu.out (-cpu rv64,v=true,vlen=256,...)
```

QEMU 7.2 has no Zvfh or Zvfbfmin, so `rvv_check` dies there with SIGILL; the five core
kernels run.

**Validated (2026-10-24):** the Linux build cross-compiles with clang 20.1.7 against
this sysroot, and its QEMU output matches the Naiad build run on Spike: 38/38 cases,
including the `A` cases bit for bit. That is two independent RVV 1.0 implementations,
and a freestanding build against a hosted one, agreeing, so the kernels and harness are
deterministic as intended. `xsmt_gemm` builds and passes with `XSMT_EMULATE=1` under
QEMU (it tests the harness, not the instruction). Neither the native K1 build nor the
real `smt.vmadot` path has been run, because there is no board here.

## SpacemiT XSMTVDot (`xsmtvdot`): what the vendor instructions are

This summarises the matrix-extension survey (`docs/plans/matrix-ext-drafts.md` §4.1,
whose sources are cited here), plus what was found while writing `xsmt/xsmt_gemm.c`.

- **They are matrix instructions, not dot products:** IME-style tile matrix
  multiply-accumulate on vector registers, C <- C + A x B^T. `vs1` holds the A tile,
  `vs2` holds B^T, and `vd` holds the accumulator in and out. They are a fixed-lambda,
  LMUL = 1 subset of the community Zvvm (IME) draft, which the doc maps them onto
  ([SpacemiT IME doc v0.6, 2026-04-13](https://github.com/spacemit-com/docs-ai/blob/main/en/architecture/ime_extension.md)).
- **Data types, K1:** int8 x int8 -> int32 only. Signedness is in the mnemonic:
  `smt.vmadot` (signed x signed), `smt.vmadotu` (unsigned x unsigned), `smt.vmadotsu`
  (A signed, B unsigned) and `smt.vmadotus` (A unsigned, B signed). The accumulator is
  always signed int32. There are also "slide" forms, `smt.vmadot{1,2,3}[u|su|us]`, which
  offset the A window and are aimed at convolutions; these need an even `vs1`. The doc
  names the K1 profile `Xsmti8i32mm` / `Xsmti8i32mm_slide` and calls the core "A60".
  The A100 (K3, VLEN 1024) adds int4, 4:2 sparsity, block-scaled forms and fp16/bf16 ->
  fp32 (`smt.vfwmadot`). **There is no FP on the K1.**
- **Geometry and register use, K1 (VLEN 256, lambda = 2):** a 4x8x4 tile. A is 4x8 int8,
  row-major, in one register (32 bytes). B^T is 4x8 int8 (B column-major) in one register.
  C is 4x4 int32 in the pair `vd`, `vd+1`, so `vd` must be even and must not overlap
  `vs1` or `vs2`. Only LMUL = 1 is legal (any other LMUL raises an illegal instruction),
  lambda is fixed rather than set in `vtype`, and `vm` is not a mask.
- **Encoding:** custom-1, `MATCH_SMT_VMADOT = 0x8200302b`, `MASK = 0x9e0070ff`. The mask
  includes instruction bit 7, so `vd` is even by encoding
  ([binutils series 61431](https://patchwork.sourceware.org/project/binutils/list/?series=61431)).
  This collides with Naiad's custom-1 AME table (our funct3 = 3 is `mcfg`), which is
  harmless because binaries are built per design.
- **Toolchain:** binutils 2.47, GCC 16.1 (`-march=..._xsmtvdot`) and LLVM 22.1 (23.1
  for XSMTVDotII). Clang 20.1.7 here rejects `xsmtvdot`, and the local LLVM 22.1.2 is
  x86-only, so the IME kernel is a **native build on the board** with a new enough
  compiler.
- **Open, for the board run.** These are new findings, not in the survey:
  - **The C tile layout.** The doc calls C row-major in §1.5 but "implementation-defined"
    in §2.7.3. `xsmt_gemm` probes one tile against a scalar reference and prints
    `# xsmt C layout: row-major | column-major | neither` before using the layout it
    found.
  - **The `vtype` the instructions want.** The doc gives no `vsetvli` sequence. The
    kernel uses `e8, m1, vl = 32` (`-DXSMT_E32` switches to `e32, m1`).
  - **The operand syntax.** It is taken to be `smt.vmadot vd, vs1, vs2`. The A100 forms
    take a trailing type (`, i8`); if the K1 assembler wants one too, set
    `#define XSMT_SUFFIX ", i8"` at the top of `xsmt/xsmt_gemm.c`.
- **For the talk:** the K1 is a real-hardware proxy for IME **geometry and data
  movement** (A and B in vector registers, C in a register pair, 4x8x4 per instruction)
  at int8. It is not a proxy for FP numerics. `xsmt_gemm` reports IME and scalar GEMM
  times for a 16x16x64 problem as a first data point.
