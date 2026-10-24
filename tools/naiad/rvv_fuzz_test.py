"""Tests for tools/naiad/rvv_fuzz.py: generator legality, the shrinker, and Spike runs.

    ~/projects/riscv/venv/bin/python -m pytest tools/naiad/rvv_fuzz_test.py      # all (~1 min)
    ... -k "not spike"                                                           # pure-Python only

Not part of ``pytest framework``. The legality checks re-derive the RVV 1.0
rules from each item's recorded operand groups and memory accesses, independently
of the generator's own allocation code. The ``spike`` tests also run programs on
Spike, which enforces much of the rest (it traps on misaligned groups, reserved
vtypes, EMUL out of range, ...); they skip without Spike or clang.
"""

import os
import random
import shutil
from fractions import Fraction as F
from pathlib import Path

import pytest

from tools.naiad import rvv_fuzz as fz

ALL = list(fz.FAMILY_HELP)
CC = os.environ.get("NAIAD_CC", fz.DEFAULT_CC)


def _have_tools():
    from framework.naiad.difftest import find_spike

    return find_spike() is not None and Path(CC).is_file()


spike = pytest.mark.skipif(not _have_tools(), reason="needs spike and clang")


def overlaps(a, b):
    return a.base < b.base + b.count and b.base < a.base + a.count


def check_item(item):
    """Every legality rule the generator promises, checked from the item's metadata."""
    problems = []
    cfg = item.cfg
    if cfg is not None:
        if not fz.legal_vtype(cfg.sew, cfg.lmul):
            problems.append(f"illegal vtype e{cfg.sew} lmul {cfg.lmul}")
        if cfg.vlmax < cfg.avl < 2 * cfg.vlmax:
            problems.append(f"AVL {cfg.avl} in (VLMAX, 2 VLMAX)")
    dests = [g for g in item.groups if g.role in ("vd", "md")]
    sources = [g for g in item.groups if g.role == "vs"]
    for g in item.groups:
        if g.base + g.count > 32 or g.base < 0:
            problems.append(f"group v{g.base}+{g.count} leaves the register file")
        if g.role == "vm":
            continue
        if g.eew != 1 and g.eew not in fz.SEWS:
            problems.append(f"EEW {g.eew}")
        if g.role in ("vd", "vs") and g.count > 1 or g.emul > 1:
            if not F(1, 8) <= g.emul <= 8:
                problems.append(f"EMUL {g.emul}")
            if g.base % fz.nregs(g.emul):
                problems.append(f"v{g.base} not aligned to EMUL {g.emul}")
        if g.count > 8:
            problems.append(f"group of {g.count} registers")
    for d in dests:
        if any(g.role == "vm" for g in item.groups) and d.base == 0:
            problems.append(f"{item.name}: destination v{d.base} overlaps the mask v0")
        for s in sources:
            if not overlaps(d, s):
                continue
            if item.vd_excl:
                problems.append(
                    f"{item.name}: v{d.base} overlaps source v{s.base} (no overlap allowed)"
                )
            elif d.eew != s.eew:
                problems.append(
                    f"{item.name}: v{d.base} (EEW {d.eew}) overlaps v{s.base} (EEW {s.eew})"
                )
    acc = item.access
    if acc is not None:
        for off in acc.offsets:
            if off < 0 or off + acc.size > fz.BUF_SIZE:
                problems.append(f"{item.name}: access at +{off} outside the buffer")
                break
            if off % acc.size:
                problems.append(
                    f"{item.name}: misaligned {acc.size}-byte access at +{off}"
                )
                break
        if acc.unordered and len(set(acc.offsets)) != len(acc.offsets):
            problems.append(f"{item.name}: unordered store writes an address twice")
    return problems


@pytest.mark.parametrize("seed", range(250))
def test_generated_items_are_legal(seed):
    program = fz.Generator(seed, count=80, families=ALL).generate()
    assert program.items
    bad = [
        (k, it.lines, p) for k, it in enumerate(program.items) for p in check_item(it)
    ]
    assert not bad, bad[:5]


def test_every_op_and_family_is_reachable():
    seen_ops, seen_fams = set(), set()
    for seed in range(400):
        for item in fz.Generator(seed, count=80, families=ALL).generate().items:
            seen_ops.add(item.name)
            seen_fams.add(item.family)
    missing = {op.name for op in fz.OPS} - seen_ops
    assert not missing, sorted(missing)
    assert set(ALL) <= seen_fams


def test_vl_for_rejects_the_implementation_defined_band():
    assert fz.vl_for(5, 8) == 5
    assert fz.vl_for(16, 8) == 8
    assert fz.vl_for(100, 8) == 8
    with pytest.raises(ValueError, match="implementation-defined"):
        fz.vl_for(9, 8)


def test_vl_preserving_vsetvli_keeps_the_ratio():
    for seed in range(300):
        for item in fz.Generator(seed, count=60, families=["vset"]).generate().items:
            if item.name == "vsetvli.x0":
                before, after = item.cfg, item.result
                assert F(before.sew) / before.lmul == F(after.sew) / after.lmul
                assert after.vl == before.vl


def test_seeds_are_reproducible():
    a = fz.Generator(11, count=40).generate().assembly()
    b = fz.Generator(11, count=40).generate().assembly()
    c = fz.Generator(12, count=40).generate().assembly()
    assert a == b
    assert a != c


def test_allowlist_restricts_the_stream():
    items = fz.Generator(3, count=60, families=["int", "cmp"]).generate().items
    assert {i.family for i in items} <= {"int", "cmp"}
    items = fz.Generator(3, count=60, only=r"^vadd\.").generate().items
    assert items
    assert all(i.name.startswith("vadd.") for i in items)
    items = fz.Generator(3, count=60, families=["fp"], exts=["v"]).generate().items
    assert items
    assert all(i.cfg.sew in (32, 64) for i in items)
    with pytest.raises(ValueError, match="selects no instructions"):
        fz.Generator(3, count=60, families=["bf16"], exts=["v"]).generate()


def test_unordered_fp_reductions_are_off_by_default():
    names = {
        i.name for s in range(100) for i in fz.Generator(s, count=60).generate().items
    }
    assert "vfredusum.vs" not in names
    assert "vfredosum.vs" in names


def test_subsets_emit_the_vsetvli_each_item_needs():
    program = fz.Generator(5, count=60, families=ALL).generate()
    rng = random.Random(0)
    for _ in range(20):
        sub, _ = program.subset(rng.sample(range(len(program.items)), 10))
        current = None
        lines = sub.body_lines(labels=False)
        it = iter(lines)
        for item in sub.items:
            line = next(it)
            if item.cfg is not None and item.cfg != current:
                assert line.strip() == f"li t0, {item.cfg.avl}"
                assert (
                    next(it)
                    .strip()
                    .startswith(
                        f"vsetvli t1, t0, e{item.cfg.sew}, {fz.lmul_name(item.cfg.lmul)}"
                    )
                )
                current = item.cfg
                line = next(it)
            assert line.strip() == item.lines[0]
            for _ in item.lines[1:]:
                next(it)
            if item.result is not None:
                current = item.result


# -- the shrinker -----------------------------------------------------------------


def test_ddmin_finds_a_pair():
    calls = []

    def fails(keep):
        calls.append(len(keep))
        return {3, 17} <= set(keep)

    assert sorted(fz.ddmin(40, fails)) == [3, 17]
    assert len(calls) < 120


def test_ddmin_finds_a_single_culprit_quickly():
    calls = []

    def fails(keep):
        calls.append(1)
        return 29 in keep

    assert fz.ddmin(64, fails) == [29]
    assert len(calls) < 30


def test_ddmin_result_is_one_minimal():
    rng = random.Random(1)
    for _ in range(20):
        need = set(rng.sample(range(50), rng.randint(1, 4)))
        keep = fz.ddmin(50, lambda k, need=need: need <= set(k))
        assert set(keep) == need


def test_signature_decode_and_compare():
    sig = bytearray(fz.SIG_SIZE)
    other = bytearray(sig)
    other[fz.SIG_X + 8 * 10] = 1  # x10
    other[fz.SIG_V + fz.VLENB * 3 + 5] = 0xFF  # v3 byte 5
    other[fz.SIG_BUF + 100] = 7
    other[fz.SIG_X + 8 * 5] = 9  # t0: harness scratch, ignored
    diffs = fz.compare_signatures(bytes(sig), bytes(other))
    assert len(diffs) == 3
    assert diffs[0].startswith("x10")
    assert any(d.startswith("v3: bytes 5..5") for d in diffs)
    assert any(d.startswith("buf: 1 bytes differ, first at +0x64") for d in diffs)
    assert fz.compare_signatures(bytes(sig), bytes(sig)) == []


# -- on Spike ---------------------------------------------------------------------


@spike
def test_programs_run_cleanly_on_spike(tmp_path):
    for seed in range(40):
        program = fz.Generator(seed, count=100, families=ALL).generate()
        o = fz.evaluate(program, tmp_path, target="spike")
        assert o.status == "PASS", (seed, o.status, o.detail)


@spike
def test_shrunk_subsets_stay_legal_on_spike(tmp_path):
    program = fz.Generator(77, count=100, families=ALL).generate()
    rng = random.Random(77)
    for _ in range(10):
        sub, _ = program.subset(
            rng.sample(range(len(program.items)), rng.randint(1, 20))
        )
        assert fz.evaluate(sub, tmp_path, target="spike").status == "PASS"


@spike
def test_a_dut_difference_is_found_and_shrunk(tmp_path):
    # The "DUT" is Spike without Zvfh, so its first fp16 vector op traps: the
    # fuzzer must report a crash and shrink the program to that one instruction.
    dut_isa = fz.SPIKE_ISA.replace("_zvfh", "")
    gen = fz.Generator(4, count=40, families=["fp", "int"], exts=["v", "zvfh"])
    program = gen.generate()
    assert any(i.family == "fp" and i.cfg.sew == 16 for i in program.items)
    kw = dict(target="spike", dut_isa=dut_isa)
    outcome = fz.evaluate(program, tmp_path, **kw)
    assert outcome.status == "CRASH"
    keep, calls = fz.shrink(program, outcome, tmp_path, **kw)
    assert len(keep) == 1
    item = program.items[keep[0]]
    assert item.family == "fp"
    assert item.cfg.sew == 16


@spike
def test_a_state_difference_is_shrunk(tmp_path):
    # A DUT with VLEN 512 disagrees on vlenb whatever runs, and on any result whose
    # vl depends on VLMAX; the shrinker must still terminate with a small program.
    kw = dict(target="spike", dut_isa=fz.SPIKE_ISA.replace("zvl256b", "zvl512b"))
    program = fz.Generator(9, count=30, families=["int"]).generate()
    outcome = fz.evaluate(program, tmp_path, **kw)
    assert outcome.status == "FAIL"
    assert any(d.startswith("vlenb") for d in outcome.diffs)
    keep, _ = fz.shrink(program, outcome, tmp_path, **kw)
    assert len(keep) == 1


@pytest.mark.skipif(shutil.which("true") is None, reason="posix")
def test_cli_lists_families(capsys):
    assert fz.main(["--list-families"]) == 0
    out = capsys.readouterr().out
    assert "fpred_unordered" in out
    assert "off by default" in out


# -- on Naiad (the RVV unit itself; skipped until a vector design exists) ----------

naiad = pytest.mark.skipif(
    not (_have_tools() and fz.DEFAULT_DESIGN.is_file()),
    reason="needs spike, clang and naiad-rvv.yaml",
)


@naiad
def test_naiad_matches_spike_on_a_few_seeds(tmp_path):
    for seed in range(8):
        program = fz.Generator(seed, count=60).generate()
        o = fz.evaluate(program, tmp_path)
        assert o.status == "PASS", (seed, o.status, o.detail, o.diffs[:3])


@naiad
def test_difftest_commit_mode_compares_vector_writes_on_naiad(tmp_path):
    from framework.naiad.difftest import difftest

    elf = fz.build(fz.Generator(1, count=60).generate(), tmp_path)
    result = difftest(fz.DEFAULT_DESIGN, elf, commits=True, isa=fz.SPIKE_ISA)
    assert result.ok, result.report()
    assert result.commits.vector_writes > 20
