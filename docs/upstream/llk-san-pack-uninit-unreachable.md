# DRAFT — not filed. Read §0 before deciding whether to file.

**Target:** [`tenstorrent/tt-metal`](https://github.com/tenstorrent/tt-metal) —
`tt_metal/tt-llk/common/sanitizer/`.
**Status:** draft only, 2026-08-20. Nothing has been filed, pushed or sent.
**Origin:** reported to us by a compiler team using `llk::san` as a CI gate. They
have no channel to Tenstorrent; we do. The findings are theirs, verified by us.

---

## §0. For our reviewer — what this claims, and what is second-hand

The core claim is **verified first-hand in the 0.74 checkout**: `_llk_pack_uninit_`
is defined in both LLK trees and called by nothing. Six occurrences in the whole
tree — two definitions and four doxygen references. There is no
`llk_pack_uninit` and no `pack_uninit()` wrapper.

What is **theirs, not ours**: that this makes `llk::san` emit an unavoidable
ERROR on a real corpus, and the n300 run that confirmed it by prediction. We have
no card, and until this week `llk::san` could not start against our simulator at
all. We are relaying a measurement we cannot independently reproduce, and it
should be read that way.

The two secondary gaps (§3) are diagnostic-quality issues, not defects. They are
included because they made the primary one harder to identify than it needed to
be, and both are already flagged in Tenstorrent's own source comments.

---

## 1. `Operation::Pack` declares an uninit that no API can perform

`llk::san`'s FSM models each execution unit as
`INITIAL → CONFIGURED → INITIALIZED[Op] → EXECUTED[Op]`.
`Operation::Pack` is declared `ExpectUninit::Yes`, so from `EXECUTED[Pack]` the
only permitted transitions are `EXECUTED[Pack]` or `UNINITIALIZED[Pack]`.

**No tt-metal API can produce that uninit.** `_llk_pack_uninit_()` exists —
`tt_llk_blackhole/llk_lib/llk_pack.h:528` and
`tt_llk_wormhole_b0/llk_lib/llk_pack.h:436` — and a tree-wide grep finds **zero
callers**: only those two definitions and four doxygen `@ref`/`@note` mentions.
There is no `llk_pack_uninit` wrapper and no `pack_uninit()` in the compute API.

**Consequence:** any kernel that packs and then reconfigures the packer trips an
ERROR with no way to clear it. This is not exotic — stock TTNN's
`convert_to_chw.cpp` hits the same rule with no tilize involved.

The source already knows. On the line setting the flag:

> `sstanisic todo: contract cannot be enforced if Pack has an uninit, without killing performance`

So the constraint appears to have been added with the performance objection
recorded but the reachability consequence unresolved.

## 2. How it was established

By **prediction**, not inspection alone: the reporters wrote down what inserting
the only uninit-shaped packer call a kernel *can* make should do to the FSM, then
ran it on an n300. The message mutated exactly as predicted rather than clearing
— which is what distinguishes "no API can do this" from "we used the wrong API".

## 3. Two diagnostic gaps that made it harder than necessary

Both flagged in Tenstorrent's own source, both cheap:

- **`tilize.h` carries no `LLK_SAN_FUNCTION()`**, so the recorded callstack for a
  violation reached through it is `UNKNOWN`.
- **`_print_fsm_transition` prints only `.type`, never the operation**, so a
  transition report does not say which `Operation` it concerns — also marked
  `sstanisic todo`.

## 4. What would resolve it

In rough order of cost:

1. **Export an uninit.** A `pack_uninit()` compute-API wrapper over the existing
   `_llk_pack_uninit_` would make the declared contract satisfiable. The
   performance objection in the source comment presumably explains why this has
   not happened.
2. **Or relax `ExpectUninit` for `Pack`** to match what the API can express, so
   the sanitizer stops reporting an ERROR no user can act on.
3. **Or document it** as a known false positive with the shape to ignore — the
   cheapest, and it at least stops the gate being unusable for kernels that
   reconfigure the packer.

Either of the first two would let a consumer adopt `llk::san` as a gate without
an allowlist entry that suppresses a whole class of real violations alongside
this one.

## 5. One unrelated note, if a maintainer is already in this file

`TT_METAL_LLK_SANITIZER=1` **alone** fails at kernel build time —
`#error "LLK_SAN_ENABLE is set but neither ENABLE_LLK_ASSERT nor DEBUG_PRINT_ENABLED is defined"`.
It needs `TT_METAL_DPRINT_CORES` set alongside it. That is presumably intended,
but it is not stated where a first-time user would look, and it presents as a
build failure rather than a configuration message.
