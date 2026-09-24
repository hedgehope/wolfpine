# DRAFT — not filed. Read §0 before deciding whether to file.

**Target:** [`tenstorrent/tt-isa-documentation`](https://github.com/tenstorrent/tt-isa-documentation).
**Status:** draft only, 2026-08-18. Nothing has been filed, pushed or sent.

---

## §0. For our reviewer — what this does and does not claim

It claims the published **6–33 cycle** range for integer divide is contradicted
at its top by three of our instruments, and that the docs give no rule by which
to reconcile that. It does **not** claim a hardware defect: the excess is small
and consistent, and we cannot see from outside whether the published figure
counts only the divider's occupancy while our measurement also catches a fixed
issue or retirement cost, or whether the true maximum is 34.

The second ask — *state the rule, not just the range* — is the one that matters
to us and the one worth filing for. The first is a courtesy.

Nothing here depends on tt-sim being right: the numbers are card measurements,
and two of the three come from a harness that instruments the divide directly
rather than through the simulator.

---
**Files:** `WormholeB0/TensixTile/BabyRISCV/README.md:49`,
`BlackholeA0/TensixTile/BabyRISCV/README.md:54` — identical wording in both.

## What the documentation says

> Divide instructions (`div`, `divu`, `rem`, `remu`) occupy the Integer Unit for
> a variable number of cycles [...] The number of cycles is dependent upon the
> values of the dividend and the divisor:
> * [...]
> * In all other cases, **between six and 33 cycles** are required, dependent
>   upon the magnitude of the dividend.

## What we measure

Three independent instruments put a large-dividend `divu` **at or fractionally
above** the stated maximum of 33:

| instrument | part | cycles per divide |
| --- | --- | --- |
| `perfbench/retirebench`, 29-bit dividend | Blackhole p150 | **33.10** |
| `perfbench/riscvbench`, separately instrumented | Blackhole p150 | **33.001 / 33.004** |
| `perfbench/riscvbench` | Wormhole n300 | **33.03** |

The Blackhole figures come from a session preserved at
`perfbench/card-sessions/2026-08-18-bh-retirebench/`, three repeats, in which
both divide zones are **bit-identical across every repeat** while every other
zone moves — so the divider is an exactly reproducible function of its operands
and this is not measurement scatter.

Method for the 33.10: a zone of 13 repetitions × 16 `divu` instructions with a
29-bit dividend, bracketed by `mcycle` reads, with the non-divide instructions
in the zone charged at their retired count. The 33.001/33.004 figures are from a
different harness that instruments the divide directly and agrees to 0.1 %.

## Why this may be a documentation issue rather than a hardware one

The excess is small and consistent, which is what one would expect if the
published figure counts the divider's own occupancy while a measurement also
sees a fixed issue or retirement overhead — or if the true maximum is 34 and 33
is off by one. We cannot distinguish those from outside: the documentation gives
a range but no algorithm, no radix, no per-bit rule and no worked example, so
there is nothing to reconcile a measurement against.

## What would help

1. **Confirm or correct the 33-cycle upper bound.** Three instruments on two
   architectures read at or just past it.
2. **State the rule, not just the range.** "Dependent upon the magnitude of the
   dividend" is the only mechanism given, and the obvious candidates are
   *excluded* by our two measured points: `cycles = bits + k` needs a single `k`
   and gets 2.018 at 12 bits and 4.043 at 29; the affine law through both points
   reads 1.71 cycles at one bit (below the documented floor of six) and 36.40 at
   32 (above the documented cap). Something else is going on, and only the
   vendor can say what.

Point 2 is the one that matters to us. Without it a simulator can only charge
the documented floor of six, which under-predicts a 29-bit divide by more than
5x; a magnitude-dependent term cannot be sourced from any published material, so
it cannot enter a provenance-tracked cost model at all.
