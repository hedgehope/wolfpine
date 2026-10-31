"""Static scheduling of straight-line emitted code for the dual-issue front end.

``timing.scalar.issue_width: 2`` (:mod:`framework.pe.rv.timing.core`) issues the
next instruction in the same cycle only if it is independent of the one before
(a RAW on its result fails it, every latency being at least a cycle) and they
do not both need the memory port or the vector / matrix dispatch port. An
emitter that lays out a loop body for one instruction a cycle therefore leaves
the second slot empty wherever it put a dependent instruction right behind its
producer. :func:`schedule` reorders such a body, given each statement's
register reads and writes: a list scheduler for exactly that front end, one
that fills the second slot with the earliest instruction that is independent of
the first and (when the first is a vector / matrix / memory op) is plain scalar
ALU work.

The statements must be ones the compiler cannot move on its own (``asm
volatile`` keeps their relative order), so the order returned is the order
issued. The result is the emitted program order, never a different program:
every register dependence of the input (read after write, write after read,
write after write) is kept.

Only the issue rules are modelled, as ``core.DualIssueCostState`` states them:
a vector / matrix / memory op ("vec") occupies the one vec port, so two vec
ops never pair, and ``vec_gap`` is the cycles a unit holds between consecutive
vec ops (a tile row's transfer time on the load / store unit).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Op:
    """One statement: its text, the registers (any hashable names) it reads and writes, and its port."""

    text: str
    reads: tuple = ()
    writes: tuple = ()
    #: A vector / matrix / memory op: takes the vec port, never pairs with another.
    vec: bool = False


def _dependences(ops):
    """Per op: ``(raw, war, waw)`` lists of earlier op indices it must follow."""
    last_write = {}
    readers = {}  # reg -> ops that read its current value
    deps = []
    for i, op in enumerate(ops):
        raw = {last_write[r] for r in op.reads if r in last_write}
        war = {j for r in op.writes for j in readers.get(r, ())} - {i}
        waw = {last_write[r] for r in op.writes if r in last_write}
        deps.append((raw, war, waw))
        for r in op.reads:
            readers.setdefault(r, []).append(i)
        for r in op.writes:
            last_write[r] = i
            readers[r] = []
    return deps


def schedule(ops, vec_gap=1, width=2):
    """``ops`` in issue order for ``width`` slots a cycle; returns ``(ordered ops, cycles)``.

    ``cycles`` is the model's own count (every latency one cycle, a vec op every ``vec_gap``).
    """
    deps = _dependences(ops)
    n = len(ops)
    where = {}  # op index -> (cycle, slot)
    order = []
    remaining = list(range(n))
    vec_free = 0
    prev_vec = -1  # the vec ops keep their relative order
    cycle = 0

    def eligible(i, now, slot, first):
        raw, war, waw = deps[i]
        if any(j not in where or where[j][0] >= now for j in raw | waw):
            return False
        for j in (
            war
        ):  # the reader must have issued, in an earlier cycle or earlier in this one
            if j not in where or (where[j][0] == now and where[j][1] >= slot):
                return False
        if ops[i].vec:
            if now < vec_free:
                return False
            # keep the vec ops in program order
            if any(ops[k].vec and k not in where for k in range(i)):
                return False
            if slot and first is not None and ops[first].vec:
                return False
        return True

    while remaining:
        picked = []
        for slot in range(width):
            first = picked[0] if picked else None
            for i in remaining:
                if i in picked:
                    continue
                # a later op may overtake earlier ones only if it is independent of them (checked by deps)
                if eligible(i, cycle, slot, first):
                    where[i] = (cycle, slot)
                    picked.append(i)
                    break
        for i in picked:
            remaining.remove(i)
            order.append(i)
            if ops[i].vec:
                vec_free = cycle + vec_gap
                prev_vec = i
        cycle += 1
    del prev_vec
    return [ops[i] for i in order], cycle
