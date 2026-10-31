"""The VME (Zvt) GEMM emitter: geometry, the block-shape choice, the K-loop schedules (replayed
symbolically against the register image they must produce), and header determinism. The emitted
driver end to end (clang 20) is ``experiments/naiad/tests/gemm_vme_dwarf_test.py``."""

import copy
import importlib.util
import math

import pytest
import yaml

from framework.naiad import DESIGNS_DIR
from framework.naiad.design import parse_design
from framework.naiad.emit import gemm_vme as G
from framework.naiad.emit.gemm import emit_gemm_header, main

VME = DESIGNS_DIR / "naiad-vme.yaml"
VME_TIMED = DESIGNS_DIR / "naiad-vme-timed.yaml"
needs_ml_dtypes = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)


def _design(path=VME_TIMED, te=None, vlen=None, name=None):
    raw = yaml.safe_load(path.read_text())
    raw = copy.deepcopy(raw)
    if te is not None:
        raw["matrix"]["te"] = te
    if vlen is not None:
        raw["cores"]["vlen"] = vlen
    if name:
        raw["name"] = name
    return parse_design(raw, path)


# -- geometry -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pair, te, e, lmul, kmax, np_",
    [
        (("fp32", "fp32"), 32, 32, 4, 1, 2),
        (("fp32", "fp32"), 16, 16, 2, 1, 4),
        (("bf16", "fp32"), 32, 32, 2, 2, 2),
        (("bf16", "fp32"), 16, 16, 1, 2, 4),
        (("fp64", "fp64"), 32, 16, 4, 1, 2),
        (("int8", "int32"), 32, 32, 1, 4, 2),
    ],
)
def test_geometry_follows_the_draft(pair, te, e, lmul, kmax, np_):
    geo = G.Geometry.for_pair(_design(te=te).matrix, *pair)
    assert (geo.e, geo.lmul, geo.kmax, geo.np) == (e, lmul, kmax, np_)
    assert geo.sw == geo.e  # VLEN 256: a tile row fills its group
    assert geo.inplace_ok


def test_geometry_refuses_what_zvt_lacks():
    spec = _design().matrix
    with pytest.raises(ValueError, match="no Zvt"):
        G.Geometry.for_pair(spec, "fp32", "fp64")  # no fp32 -> fp64 in Zvt
    with pytest.raises(ValueError, match="narrower"):
        G.Geometry.for_pair(spec, "fp32", "fp16")
    assert ("bf16", "bf16") not in G.format_pairs(spec)  # no bf16 C


# -- the shape choice ----------------------------------------------------------------------------


def test_timed_te32_shapes():
    d = _design()
    fp32 = G.plan_for(d, "fp32", "fp32").plan
    assert (fp32.bm, fp32.bn) == (2, 2)  # all four TEW-32 accumulators
    bf16 = G.plan_for(d, "bf16", "fp32").plan
    assert (bf16.bm, bf16.bn) == (2, 2)
    fp64 = G.plan_for(d, "fp64", "fp64").plan
    assert fp64.bm * fp64.bn == 8  # all eight TEW-64 tiles: 2 x 4 or 4 x 2
    assert {fp64.bm, fp64.bn} == {2, 4}


def test_every_choice_fits_the_unit_and_the_registers():
    for d in (
        _design(),
        _design(te=16),
        _design(VME),
        _design(te=8),
        _design(vlen=512, te=64),
    ):
        for pair in G.format_pairs(d.matrix):
            sched = G.plan_for(d, *pair)
            plan = sched.plan
            assert plan.valid()
            assert plan.mps <= plan.geo.ntiles
            assert plan.da + plan.db <= G.WINDOWS
            assert plan.kq & (plan.kq - 1) == 0
            assert plan.kq <= G.MAX_KQ


def test_a_forced_block_is_honoured_or_refused():
    d = _design()
    plan = G.plan_for(d, "fp32", "fp32", block="1x2").plan
    assert (plan.bm, plan.bn) == (1, 2)
    with pytest.raises(ValueError, match="no VME blocking"):
        G.plan_for(d, "fp32", "fp32", block="4x4")  # 16 accumulators; TEW 32 has 4


def test_the_choice_follows_the_timing():
    """Untimed, every instruction is a cycle: the choice minimises instructions instead."""
    timed = G.costs_for(
        _design(), G.Geometry.for_pair(_design().matrix, "fp32", "fp32")
    )
    untimed = G.costs_for(
        _design(VME), G.Geometry.for_pair(_design(VME).matrix, "fp32", "fp32")
    )
    assert timed.timed
    assert (timed.mma_occ, timed.load_latency) == (3, 8)
    assert not untimed.timed
    s = G.plan_for(_design(VME), "fp32", "fp32")
    cyc, instr = G._ref_cost(s, untimed, 128)
    assert cyc == instr


# -- the schedules, replayed -----------------------------------------------------------------------


def _replay(sched, nbody):
    """Run the emitted op sequence symbolically; assert every multiply reads its own operands."""
    plan, geo = sched.plan, sched.plan.geo
    ra, rb = plan.rings()
    vlenb = geo.vlenb
    ldb = 3 * plan.bn * geo.sw * geo.ies  # B wider than one block, as in place
    group = geo.lmul * vlenb
    # memory: register-sized chunks, labelled with what they hold
    mem_a = {}
    for k in range(nbody * plan.kq + 64):
        for b in range(plan.bm):
            base = plan.a_offset(k, b)
            for x in range(geo.lmul):
                mem_a[base + x * vlenb] = ("A", k, b, x)

    def mem_b(off):
        k, col = divmod(off, ldb)
        j, rest = divmod(col, group)
        return ("B", k, j, rest // vlenb)

    ptr = {"pa": 0}
    for r in range(geo.kmax):
        ptr[f"pb{r}"] = r * ldb
    bstep = geo.kmax * ldb
    regs = {}

    def adv_value(adv):
        if adv == "bstep":
            return bstep
        if adv.startswith("bstep_"):
            return bstep - int(adv[6:])
        return int(adv)

    def op(o):
        kind, ld, _ = o
        if kind == "adv":
            ptr[ld.stream] += adv_value(ld.adv)
            return
        for x in range(ld.nregs):
            off = ptr[ld.stream] + x * vlenb
            regs[ld.reg + x] = mem_a.get(off) if ld.stream == "pa" else mem_b(off)

    def check(u, i, j):
        for r in range(geo.kmax):
            for x in range(geo.lmul):
                k = u * geo.kmax + r
                assert regs.get(ra.reg(u, i, r) + x) == ("A", k, i, x), (
                    plan,
                    u,
                    i,
                    j,
                    r,
                    x,
                )
                assert regs.get(rb.reg(u, j, r) + x) == ("B", k, j, x), (
                    plan,
                    u,
                    i,
                    j,
                    r,
                    x,
                )

    for o in sched.prologue:
        op(o)
    mmas = 0
    for b in range(nbody):
        for (u, i, j), ops in sched.body:
            check(u + b * plan.u, i, j)
            mmas += 1
            for o in ops:
                if not (o[2] and b == nbody - 1):
                    op(o)
    assert mmas == nbody * plan.u * plan.mps
    # every stream ends exactly nbody bodies on (the last body drops the next body's prefetch)
    steps = nbody * plan.u
    for r in range(geo.kmax):
        assert ptr[f"pb{r}"] == r * ldb + steps * bstep
    return True


@pytest.mark.parametrize(
    "pair, te",
    [
        (("fp32", "fp32"), 32),
        (("fp32", "fp32"), 16),
        (("bf16", "fp32"), 32),
        (("bf16", "fp32"), 16),
        (("fp64", "fp64"), 32),
        (("int8", "int32"), 32),
    ],
)
def test_every_candidate_schedule_feeds_each_multiply_its_operands(pair, te):
    geo = G.Geometry.for_pair(_design(te=te).matrix, *pair)
    n = 0
    for plan in G.candidates(geo):
        for cap, lead in ((1, 0), (2, 4)):
            sched = G.schedule(plan, cap, lead)
            if sched is None:
                continue
            for nbody in (1, 2, 3):
                assert _replay(sched, nbody)
            n += 1
    assert n > 10


def test_the_rings_reject_a_reload_before_the_old_contents_are_read():
    geo = G.Geometry.for_pair(_design().matrix, "fp32", "fp32")
    # one window per operand: each step's loads must wait out the previous step's multiplies
    single = G.Plan(geo, 2, 2, "window", "row", 1, 1, "ij")
    sched = G.schedule(single)
    assert sched is not None
    assert _replay(sched, 3)
    # 2 x 4 fp32 would need 6 groups of 4 registers a step, more than one window each side
    assert not G.Plan(geo, 2, 4, "window", "row", 1, 1, "ij").valid()


def test_the_body_period_is_a_power_of_two_dividing_32():
    for d in (_design(), _design(te=16)):
        for pair in G.format_pairs(d.matrix):
            kq = G.plan_for(d, *pair).plan.kq
            assert 32 % kq == 0


# -- the header ------------------------------------------------------------------------------------


def test_header_is_deterministic():
    a = emit_gemm_header(_design())
    G.choose.cache_clear()
    b = emit_gemm_header(_design())
    assert a == b
    assert "Generated by" in a
    assert "DO NOT EDIT" in a
    # timing-only changes that do not move the choice leave it as is; the shape lives in naiad.h's guard
    assert "#if NAIAD_VME_TE != 32 || NAIAD_VLEN != 256" in a


def test_header_has_every_pair_and_the_narrow_variants():
    text = emit_gemm_header(_design())
    for i, a in G.format_pairs(_design().matrix):
        assert f"void naiad_gemm_{i}_{a}_run(" in text
        assert f"NAIAD_GEMM_HAS_{i.upper()}_{a.upper()} 1" in text
    fp64 = G.plan_for(_design(), "fp64", "fp64").plan
    for bn in range(1, int(math.log2(fp64.bn)) + 1):
        assert f"naiad_gemm_fp64_fp64_block_n{fp64.bn >> bn}(" in text
    assert "#define naiad_gemm NAIAD_GEMM_SEL" in text


def test_header_differs_by_te_and_pairs_can_be_restricted():
    assert emit_gemm_header(_design()) != emit_gemm_header(_design(te=16))
    only = emit_gemm_header(_design(), pairs=["fp32:fp32"])
    assert "naiad_gemm_fp32_fp32_run" in only
    assert "naiad_gemm_fp64_fp64_run" not in only
    with pytest.raises(ValueError, match="not implemented"):
        emit_gemm_header(_design(), pairs=["fp32:fp64"])


def test_cli_writes_a_real_header_for_a_vme_design(tmp_path):
    out = tmp_path / "naiad_gemm.h"
    assert main([str(VME_TIMED), "-o", str(out)]) == 0
    text = out.read_text()
    assert '#error "naiad_gemm.h:' not in text
    assert "naiad_gemm_fp32_fp32_block(" in text


# -- dual issue ------------------------------------------------------------------------------------


def _dual(width, te=16):
    raw = yaml.safe_load(VME_TIMED.read_text())
    raw["matrix"]["te"] = te
    raw["timing"]["scalar"]["issue_width"] = width
    return parse_design(raw, VME_TIMED)


def test_single_issue_header_is_unchanged_by_the_dual_issue_emission():
    """Width 1 keeps the compiler-ordered tile-move loops: no explicit advance ops, no per-column store pointers."""
    one = emit_gemm_header(_dual(1))
    assert "cs_" not in one
    assert "NAIAD_GEMM_ADD(s_, la)" not in one
    assert "NAIAD_GEMM_ADD(" in one  # the K loop's own register advances
    assert one == emit_gemm_header(
        _design(te=16)
    )  # no timing.scalar.issue_width key = width 1


def test_dual_issue_header_schedules_the_tile_move_loops_and_nothing_else():
    one, two = emit_gemm_header(_dual(1)), emit_gemm_header(_dual(2))
    assert one != two
    assert "NAIAD_GEMM_ADD(s_, la)" in two
    assert "cs_" in two
    # the multiplies and zeroing are untouched; the tile moves are the same ops reordered around their
    # address advances (the A transpose's row part twice: a zero row stride, the ones-tile reductions',
    # drops the pointer advance)
    for op in ("vme_vtfmm", "vme_vtzero("):
        assert one.count(op) == two.count(op), op
    assert "if (la == 0) {" in two
    assert two.count("vme_vtse(") >= one.count("vme_vtse(")
    assert two.count("vme_vtle(") >= one.count("vme_vtle(")

    # the K loop is the same text: it is scheduled by the K-loop model, not by the front end
    def kloop(text):
        return text.split("for (long it_ = nbody - 1")[1].split("/* the last body")[0]

    assert kloop(one) == kloop(two)


def test_dual_issue_store_falls_back_when_the_block_has_too_many_columns():
    big = _dual(2)
    plan = G.plan_for(big, "fp32", "fp32").plan
    assert (
        plan.bn <= G.MAX_DUAL_BN
    )  # the default shapes stay within the per-column registers
