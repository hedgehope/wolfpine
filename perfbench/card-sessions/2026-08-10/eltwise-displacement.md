# `eltwise_binary` on the card: a half-tile displacement, root cause open

The upstream `metal_example_eltwise_binary` computes `dst[i] = a[i] + (-1.0)`
over 64 tiles of bfloat16, `a` uniform on `[0, 1)`, and checks every element
against a float golden with `eps = 1e-2`. It **passes on tt-sim and on ttsim**
and **fails on this Blackhole card**, reproducibly.

Six invocations were examined: the ladder and profiler runs of both sessions,
plus the watcher and device-print runs of `paper-1`. All six fail with
64 189–64 325 of 65 536 elements outside tolerance.

## What the output actually is

The mismatch log prints `expected` and `got` per element, so both arrays are
recoverable. Converting `expected` to bfloat16 with **round-to-nearest-even**
(the packer's rounding, not truncation):

```
fraction of indices where  got[i] == bf16_rne(expected[i - 512])
    paper-1 ladder        0.9120
    paper-1 watcher       0.9130
    paper-2 ladder        0.9108
    chance level          0.004
```

512 bfloat16 elements = **1024 bytes = half a 32×32 tile = two of the four
16×16 faces a tile is stored as**.

The displacement is uniform: 0.900–0.917 across all eight 128-element bands of
the tile, and 0.885–0.936 across all 64 tiles, with no structure in the
residual. Truncating instead of rounding drops the agreement to 0.745, which is
how the rounding mode was identified.

## What it is not

* **Not a tolerance problem.** Errors are O(0.5) on 98 % of elements against an
  `eps` of 0.01.
* **Not flaky.** Six invocations, two sessions, with and without the
  compute-grid override, under the watcher and under device print — all the
  same.
* **Not a stale buffer.** Each invocation's output is uncorrelated (0.4 %,
  i.e. chance) with every other invocation's input *and* output.
* **Not a permutation of the answer at tile granularity.** Per-tile value
  multisets do not match under any bijection: mean self-overlap 0.779 against a
  mean best-overlap of 0.786, both at the chance level for 1024 samples over
  256 distinct values.
* **Not a global permutation.** Whole-array multiset overlap is 0.982, twelve
  standard deviations below the 0.9949 ± 0.0003 a true permutation gives under
  the same censoring.

## Where it sits

Everything on this card that only *moves* data is bit-exact — `loopback`,
`add_2_integers_in_riscv`, and the `dramtop` optest, the last of which matches
both simulators byte for byte. Everything that routes data through the Tensix
unpack/compute/pack path returns wrong or degraded results: `eltwise_binary`
here, `six` at PCC 0.3067, `matmul_single_core` at 0.9228, `matmul_multi_core`
at 0.9821, and three of four matmul optests failing their own goldens.

That pattern, plus the fact that both simulators reproduce the programs' own
analytical goldens, is why this is **not** written up as a simulator fidelity
gap. A part that computed `A·B` wrongly would not ship. The likelier
explanations are a tt-metal build, firmware-bundle or harvesting interaction
specific to this host, and distinguishing them needs a second card or a rebuilt
tt-metal — neither of which was available.

**Status: signature characterised, root cause open.**
