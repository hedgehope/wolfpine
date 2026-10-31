"""The dual-issue list scheduler (``framework/naiad/emit/dual_sched.py``): it must keep every register
dependence and the order of the vec ops, and fill the second slot as the core's pairing rules allow."""

import random

import pytest

from framework.naiad.emit.dual_sched import Op, schedule


def _stream(n, vec_first=True):
    """``n`` (tile op, pointer advance, index advance) triples on one pointer and one index register."""
    ops = []
    for k in range(n):
        ops.append(Op(f"V{k}", ("p", "i"), (), True))
        ops.append(Op(f"P{k}", ("p",), ("p",)))
        ops.append(Op(f"I{k}", ("i",), ("i",)))
    return ops


def _respects_dependences(original, ordered):
    pos = {o.text: k for k, o in enumerate(ordered)}
    for a in range(len(original)):
        for b in range(a + 1, len(original)):
            x, y = original[a], original[b]
            conflict = (
                set(x.writes) & set(y.reads)  # RAW
                or set(x.reads) & set(y.writes)  # WAR
                or set(x.writes) & set(y.writes)  # WAW
            )
            if conflict and pos[x.text] > pos[y.text]:
                return False
    return True


def test_one_stream_is_bound_by_its_own_advances():
    """One pointer and one index register: the next tile op waits for both advances, 2 cycles each
    however the slots fill (the scheduler must not do worse than program order, which is also 2)."""
    ops = _stream(32)
    ordered, cycles = schedule(ops)
    assert sorted(o.text for o in ordered) == sorted(o.text for o in ops)
    assert _respects_dependences(ops, ordered)
    assert cycles <= 2 * 32 + 1


def test_two_streams_interleave_the_advances_of_the_other():
    a = [Op(f"A{k}", ("pa", "ia"), (), True) for k in range(8)]
    ops = []
    for k in range(8):
        ops += [a[k], Op(f"pa{k}", ("pa",), ("pa",)), Op(f"ia{k}", ("ia",), ("ia",))]
        ops += [
            Op(f"B{k}", ("pb", "ib"), (), True),
            Op(f"pb{k}", ("pb",), ("pb",)),
            Op(f"ib{k}", ("ib",), ("ib",)),
        ]
    ordered, cycles = schedule(ops)
    assert _respects_dependences(ops, ordered)
    assert cycles <= 1.5 * 16 + 4


def test_vec_ops_never_pair_and_keep_their_order():
    ops = _stream(12)
    ordered, _ = schedule(ops)
    vec = [o.text for o in ordered if o.vec]
    assert vec == [f"V{k}" for k in range(12)]


def test_a_unit_that_holds_longer_than_a_cycle_sets_the_pace():
    ops = _stream(16)
    _, cycles = schedule(ops, vec_gap=3)
    assert 3 * 15 <= cycles <= 3 * 16 + 2


def test_single_width_is_program_order_issue():
    ops = _stream(6)
    ordered, cycles = schedule(ops, width=1)
    assert cycles >= len(ops)
    assert _respects_dependences(ops, ordered)


@pytest.mark.parametrize("seed", range(20))
def test_random_streams_keep_every_dependence(seed):
    rnd = random.Random(seed)
    regs = ["a", "b", "c", "d", "e"]
    ops = []
    for k in range(40):
        ops.append(
            Op(
                f"o{k}",
                tuple(rnd.sample(regs, rnd.randint(0, 2))),
                tuple(rnd.sample(regs, rnd.randint(0, 1))),
                rnd.random() < 0.3,
            )
        )
    ordered, cycles = schedule(ops)
    assert len(ordered) == len(ops)
    assert _respects_dependences(ops, ordered)
    assert [o.text for o in ordered if o.vec] == [o.text for o in ops if o.vec]
    assert cycles <= len(
        ops
    )  # never worse than one instruction a cycle with no stalls modelled
