"""The dwarf cost model: engine rules against hand-derived counts, the listing interpreter, and
predictions against simulator cycles recorded on naiad-dwarfs.yaml."""

import dataclasses
import statistics
from pathlib import Path

import pytest

from framework.naiad import costmodel
from framework.naiad.costmodel import engine, listing
from framework.naiad.costmodel.dwarfs.common import Unsupported
from framework.naiad.costmodel.machine import AME, Machine, Memory, Scalar, Vector

DESIGNS = Path(__file__).resolve().parents[1] / "designs"
DWARFS_DESIGN = DESIGNS / "naiad-dwarfs.yaml"


def mach(**kw):
    base = dict(
        scalar=Scalar(
            alu=1, load=2, fp_add=2, fp_fma=2, fp_div=20, fp_sqrt=20, branch_taken=0
        ),
        memory=Memory(latency=6),
        vector=Vector(
            vlen=256,
            datapath_bits=256,
            startup=4,
            mem_startup=2,
            mem_bytes_per_cycle=32,
        ),
        mode="ame",
        ame=AME(
            macs_per_cycle=2048,
            format_passes={"fp32": 4},
            pipeline_depth=5,
            tile_load_latency=2,
            tile_store_latency=1,
            tile_load_bandwidth=64,
            tile_store_bandwidth=64,
        ),
    )
    base.update(kw)
    return Machine(**base)


# -- engine rules -------------------------------------------------------------------------


def test_load_use_chain_exposes_the_load_latency():
    """flw; addi; fadd (reads the load); bne: 2 + L + 1 issue slots an element, as the scalar reduce."""
    m = mach()
    e = engine.Engine(m)

    def body(_):
        e.load("x", fp=True)
        e.alu(1)
        e.op("fadd", "s", ("s", "x"))
        e.branch()

    e.loop(100, body)
    assert e.cycles - 1 == 100 * (m.load_latency + 2)


def test_unpipelined_divide_serialises():
    e = engine.Engine(mach())
    for _ in range(5):
        e.op("fdiv", None)
    assert e.cycles - 1 == 4 * 20 + 1


def test_mma_chain_forwards_but_astore_waits_for_the_pipeline():
    m = mach()
    e = engine.Engine(m)
    ii = m.ame.ii("fp32")  # 16^3 * 4 / 2048 = 8
    assert ii == 8
    for _ in range(4):
        e.mma(0, 1, 2, ii)
    c = e.tstore(0, 1024, acc=True)
    # 4 back-to-back mmas from cycle 1, the last result ready ii + depth - 1 after it issues
    assert c == 1 + 3 * ii + ii + m.ame.pipeline_depth - 1


def test_tile_load_latency_and_store_to_load_raw():
    m = mach()
    e = engine.Engine(m)
    e.tstore(0, 1024, acc=True, token="y")  # T = 16, complete at 1 + 1 + 16 - 1
    c = e.tload(1, 1024, after=("y",))
    assert c == 17
    c2 = e.mma(0, 1, 1, 8)
    assert c2 == c + m.ame.tile_load_latency + m.memory.latency + 16 - 1


def test_vector_no_chaining_and_lsu():
    m = mach()
    e = engine.Engine(m)
    e.vload("a", 128)  # 4 cycles on the LSU, ready 2 + 6 + 4 later
    c = e.vop("b", ("a",), 32, 32)  # occupancy 4
    assert c == 1 + 12
    c2 = e.vop("c", ("b",), 32, 32)
    assert c2 == c + 4 + 4


def test_loop_fast_forward_matches_full_replay():
    m = mach()

    def run(trips, warm=None):
        e = engine.Engine(m)

        def body(i):
            e.tload(i % 2, 1024)
            e.alu(2)
            e.mma(0, i % 2, 2, 8)
            e.branch()

        if warm is None:
            for i in range(trips):
                body(i)
        else:
            e.loop(trips, body, period=2)
        return e.cycles

    assert run(200, warm=True) == run(200)


def test_bound_engine_is_a_lower_bound():
    d = costmodel.machine(DWARFS_DESIGN)
    for dwarf, size in (("gemm", 64), ("jacobi", 66), ("scan", 4096)):
        s = costmodel.predict(dwarf, size, "fp32", "fp32", d)
        b = costmodel.predict(dwarf, size, "fp32", "fp32", d, model="bound")
        assert set(s) == set(b)
        for v in s:
            assert b[v] <= s[v] * 1.001, (dwarf, v)


# -- listings -------------------------------------------------------------------------------


def test_listing_tracks_vtype_and_widening():
    text = """
    10: vsetvli t0, zero, e16, m2, ta, ma
    14: vle16.v v8, (a0)
    18: vfwmaccbf16.vf v16, fa5, v8
    1c: vsetvli zero, zero, e32, m4, ta, ma
    20: vfadd.vv v16, v16, v16
    """
    m = mach()
    e = engine.Engine(m)
    e.log = []
    vs = listing.Program(text).run(e)
    assert (vs.sew, vs.lmul, vs.vl) == (32, 4, 32)
    t = {lab.split()[0]: c for c, lab in e.log}
    # vle16 at LMUL 2: 64 B, 2 cycles; the widening FMA occupies 32 * 32 / 256 = 4 cycles
    assert t["18"] == t["14"] + 2 + 6 + 2
    assert t["20"] == t["18"] + 4 + 4


def test_listing_matrix_words_and_tokens():
    text = """
    0: mx.astore as=0 base=a0 stride=a1 fmt=fp32
    4: mx.tload.a td=5 base=a0 stride=a1 fmt=fp32 t=0
    """
    P = listing.Program(text).note(0, "!q").note(4, "@q")
    e = engine.Engine(mach())
    e.log = []
    P.run(e)
    assert e.log[1][0] == 1 + 1 + 16 - 1  # waits for the astore to complete


def test_shipped_listings_parse():
    for path in sorted((Path(listing.__file__).parent / "listings").glob("*.s")):
        P = listing.Program(path.read_text())
        assert P.insns, path.name
        P.find("rdcycle")


# -- API ------------------------------------------------------------------------------------


def test_choose_and_unsupported():
    c = costmodel.choose("gemm", 64, "fp32", "fp32", DWARFS_DESIGN)
    assert c.variant == "ame"
    assert c.margin > 1
    assert c.reformulation_speedup > 10
    nbody = costmodel.choose("nbody", (64, "uniform"), "fp32", "fp32", DWARFS_DESIGN)
    assert nbody.variant == "rvv"
    assert nbody.reformulation_speedup < 1
    m = costmodel.machine(DWARFS_DESIGN)
    plain = m.with_(mode=None, ame=None)
    assert "ame" not in costmodel.predict("gemm", 64, "fp32", "fp32", plain)
    assert "scalar" not in costmodel.predict("gemm", 64, "bf16", "fp32", plain)
    with pytest.raises(Unsupported):
        costmodel.choose(
            "gemm", 64, "fp32", "fp32", plain.with_(vector=None), variants=["rvv"]
        )


def test_crossover_moves_with_the_unit():
    """Half the vector datapath: the matrix peak at which AME catches RVV drops (in MACs a cycle)."""
    m = costmodel.machine(DWARFS_DESIGN)
    x1, s1 = costmodel.crossover("gemm", 64, design=m)
    half = m.with_(vector=dataclasses.replace(m.vector, datapath_bits=128))
    x2, s2 = costmodel.crossover("gemm", 64, design=half)
    assert s1 == s2 == "ok"
    assert 0.5 < x1 < 1.2
    assert x2 * half.vector.datapath_bits < x1 * m.vector.datapath_bits


def test_peak_axis_changes_only_the_matrix_variants():
    m = costmodel.machine(DWARFS_DESIGN)
    weak = costmodel.with_matrix_peak(m, 64)
    a = costmodel.predict("reduce", 16384, "fp32", "fp32", m)
    b = costmodel.predict("reduce", 16384, "fp32", "fp32", weak)
    assert a["rvv"] == b["rvv"]
    assert a["scalar"] == b["scalar"]
    assert b["ame"] > 5 * a["ame"]


# -- against the simulator -------------------------------------------------------------------

#: Region cycles of the current simulator on naiad-dwarfs.yaml (the cost-model validation sweep's
#: ``sizes`` and ``formats_small`` studies): (dwarf, variant, size, in, acc, cycles).
RECORDED = [
    ("apsp", "ame", 32, "fp32", "fp32", 894),
    ("apsp", "ame", 64, "fp32", "fp32", 4919),
    ("apsp", "ame", 128, "fp32", "fp32", 33505),
    ("apsp", "rvv", 32, "fp32", "fp32", 16515),
    ("apsp", "rvv", 64, "fp32", "fp32", 127875),
    ("apsp", "rvv", 128, "fp32", "fp32", 1010947),
    ("apsp", "scalar", 32, "fp32", "fp32", 201030),
    ("apsp", "scalar", 64, "fp32", "fp32", 1741379),
    ("apsp", "scalar", 128, "fp32", "fp32", 13780099),
    ("fft", "ame", 256, "fp32", "fp32", 31548),
    ("fft", "ame", 1024, "fp32", "fp32", 125619),
    ("fft", "ame", 4096, "fp32", "fp32", 990585),
    ("fft", "ame_gauss", 256, "fp32", "fp32", 75281),
    ("fft", "ame_gauss", 1024, "fp32", "fp32", 299881),
    ("fft", "ame_gauss", 4096, "fp32", "fp32", 1836689),
    ("fft", "rvv", 256, "fp32", "fp32", 11483),
    ("fft", "rvv", 1024, "fp32", "fp32", 64056),
    ("fft", "rvv", 4096, "fp32", "fp32", 254440),
    ("fft", "scalar", 256, "fp32", "fp32", 154500),
    ("fft", "scalar", 1024, "fp32", "fp32", 727532),
    ("fft", "scalar", 4096, "fp32", "fp32", 3449475),
    ("gemm", "ame", 32, "fp32", "fp32", 226),
    ("gemm", "ame", 64, "fp32", "fp32", 1317),
    ("gemm", "ame", 128, "fp32", "fp32", 9229),
    ("gemm", "rvv", 32, "fp32", "fp32", 5233),
    ("gemm", "rvv", 64, "fp32", "fp32", 40307),
    ("gemm", "rvv", 128, "fp32", "fp32", 319055),
    ("gemm", "scalar", 32, "fp32", "fp32", 65069),
    ("gemm", "scalar", 64, "fp32", "fp32", 505805),
    ("gemm", "scalar", 128, "fp32", "fp32", 3988909),
    ("jacobi", "ame", 34, "fp32", "fp32", 3092),
    ("jacobi", "ame", 66, "fp32", "fp32", 9510),
    ("jacobi", "ame", 130, "fp32", "fp32", 32590),
    ("jacobi", "rvv", 34, "fp32", "fp32", 9421),
    ("jacobi", "rvv", 66, "fp32", "fp32", 37141),
    ("jacobi", "rvv", 130, "fp32", "fp32", 147877),
    ("jacobi", "scalar", 34, "fp32", "fp32", 184989),
    ("jacobi", "scalar", 66, "fp32", "fp32", 732317),
    ("jacobi", "scalar", 130, "fp32", "fp32", 2914461),
    ("nbody", "ame", [64, "uniform"], "fp32", "fp32", 298934),
    ("nbody", "ame", [256, "uniform"], "fp32", "fp32", 4760440),
    ("nbody", "ame", [384, "uniform"], "fp32", "fp32", 10703564),
    ("nbody", "ame_fgemm", [64, "uniform"], "fp32", "fp32", 266794),
    ("nbody", "ame_fgemm", [256, "uniform"], "fp32", "fp32", 4214873),
    ("nbody", "ame_fgemm", [384, "uniform"], "fp32", "fp32", 9484005),
    ("nbody", "rvv", [64, "uniform"], "fp32", "fp32", 17723),
    ("nbody", "rvv", [256, "uniform"], "fp32", "fp32", 276689),
    ("nbody", "rvv", [384, "uniform"], "fp32", "fp32", 620847),
    ("nbody", "scalar", [64, "uniform"], "fp32", "fp32", 289936),
    ("nbody", "scalar", [256, "uniform"], "fp32", "fp32", 4649493),
    ("nbody", "scalar", [384, "uniform"], "fp32", "fp32", 10456340),
    ("reduce", "ame", 4096, "fp32", "fp32", 439),
    ("reduce", "ame", 16384, "fp32", "fp32", 1387),
    ("reduce", "ame", 65536, "fp32", "fp32", 5179),
    ("reduce", "rvv", 4096, "fp32", "fp32", 621),
    ("reduce", "rvv", 16384, "fp32", "fp32", 2253),
    ("reduce", "rvv", 65536, "fp32", "fp32", 8789),
    ("reduce", "scalar", 4096, "fp32", "fp32", 40967),
    ("reduce", "scalar", 16384, "fp32", "fp32", 163847),
    ("reduce", "scalar", 65536, "fp32", "fp32", 655367),
    ("scan", "ame", 4096, "fp32", "fp32", 1948),
    ("scan", "ame", 16384, "fp32", "fp32", 7611),
    ("scan", "ame", 65536, "fp32", "fp32", 30268),
    ("scan", "rvv", 4096, "fp32", "fp32", 8537),
    ("scan", "rvv", 16384, "fp32", "fp32", 33881),
    ("scan", "rvv", 65536, "fp32", "fp32", 135262),
    ("scan", "scalar", 4096, "fp32", "fp32", 57349),
    ("scan", "scalar", 16384, "fp32", "fp32", 229381),
    ("scan", "scalar", 65536, "fp32", "fp32", 917509),
    ("spmv", "ame", [128, 0.5], "fp32", "fp32", 1302),
    ("spmv", "ame", [256, 0.25], "fp32", "fp32", 2584),
    ("spmv", "ame", [512, 0.125], "fp32", "fp32", 5138),
    ("spmv", "rvv", [128, 0.5], "fp32", "fp32", 26098),
    ("spmv", "rvv", [256, 0.25], "fp32", "fp32", 52194),
    ("spmv", "rvv", [512, 0.125], "fp32", "fp32", 104386),
    ("spmv", "scalar", [128, 0.5], "fp32", "fp32", 730151),
    ("spmv", "scalar", [256, 0.25], "fp32", "fp32", 1460286),
    ("spmv", "scalar", [512, 0.125], "fp32", "fp32", 2920559),
    ("tensorprod", "ame", 8, "fp32", "fp32", 1237),
    ("tensorprod", "ame", 10, "fp32", "fp32", 4584),
    ("tensorprod", "ame", 12, "fp32", "fp32", 5158),
    ("tensorprod", "ame", 16, "fp32", "fp32", 6630),
    ("tensorprod", "rvv", 8, "fp32", "fp32", 37782),
    ("tensorprod", "rvv", 10, "fp32", "fp32", 84770),
    ("tensorprod", "rvv", 12, "fp32", "fp32", 142297),
    ("tensorprod", "rvv", 16, "fp32", "fp32", 344745),
    ("tensorprod", "scalar", 8, "fp32", "fp32", 241367),
    ("tensorprod", "scalar", 10, "fp32", "fp32", 847418),
    ("tensorprod", "scalar", 12, "fp32", "fp32", 1109080),
    ("tensorprod", "scalar", 16, "fp32", "fp32", 3370896),
    ("apsp", "ame", 32, "fp64", "fp64", 1840),
    ("apsp", "ame", 32, "bf16", "bf16", 533),
    ("apsp", "rvv", 32, "fp64", "fp64", 32707),
    ("apsp", "scalar", 32, "fp64", "fp64", 201030),
    ("fft", "ame", 256, "fp64", "fp64", 32479),
    ("fft", "ame", 256, "bf16", "fp32", 53954),
    ("fft", "ame", 256, "fp16", "fp32", 67119),
    ("fft", "ame_gauss", 256, "fp64", "fp64", 76298),
    ("fft", "ame_gauss", 256, "bf16", "fp32", 123443),
    ("fft", "ame_gauss", 256, "fp16", "fp32", 183389),
    ("fft", "rvv", 256, "fp64", "fp64", 19619),
    ("fft", "rvv", 256, "bf16", "fp32", 11567),
    ("fft", "rvv", 256, "fp16", "fp32", 11391),
    ("fft", "scalar", 256, "fp64", "fp64", 155525),
    ("gemm", "ame", 32, "fp64", "fp64", 494),
    ("gemm", "ame", 32, "bf16", "fp32", 152),
    ("gemm", "ame", 32, "fp16", "fp32", 152),
    ("gemm", "ame", 32, "bf16", "bf16", 120),
    ("gemm", "rvv", 32, "fp64", "fp64", 10463),
    ("gemm", "rvv", 32, "bf16", "fp32", 5230),
    ("gemm", "rvv", 32, "fp16", "fp32", 5171),
    ("gemm", "rvv", 32, "bf16", "bf16", 5333),
    ("gemm", "scalar", 32, "fp64", "fp64", 65069),
    ("jacobi", "ame", 34, "fp64", "fp64", 8140),
    ("jacobi", "ame", 34, "bf16", "fp32", 2961),
    ("jacobi", "ame", 34, "fp16", "fp32", 212574),
    ("jacobi", "rvv", 34, "fp64", "fp64", 18709),
    ("jacobi", "rvv", 34, "bf16", "fp32", 9015),
    ("jacobi", "rvv", 34, "fp16", "fp32", 8599),
    ("jacobi", "scalar", 34, "fp64", "fp64", 184988),
    ("nbody", "ame", [64, "uniform"], "fp64", "fp64", 294783),
    ("nbody", "ame", [64, "uniform"], "bf16", "fp32", 302291),
    ("nbody", "ame", [64, "uniform"], "fp16", "fp32", 307101),
    ("nbody", "ame_fgemm", [64, "uniform"], "fp64", "fp64", 271500),
    ("nbody", "ame_fgemm", [64, "uniform"], "bf16", "fp32", 314678),
    ("nbody", "ame_fgemm", [64, "uniform"], "fp16", "fp32", 336936),
    ("nbody", "rvv", [64, "uniform"], "fp64", "fp64", 44334),
    ("nbody", "rvv", [64, "uniform"], "bf16", "fp32", 17795),
    ("nbody", "rvv", [64, "uniform"], "fp16", "fp32", 17795),
    ("nbody", "scalar", [64, "uniform"], "fp64", "fp64", 289938),
    ("reduce", "ame", 4096, "fp64", "fp64", 1046),
    ("reduce", "ame", 4096, "bf16", "fp32", 290),
    ("reduce", "ame", 4096, "fp16", "fp32", 290),
    ("reduce", "rvv", 4096, "fp64", "fp64", 1164),
    ("reduce", "rvv", 4096, "bf16", "fp32", 628),
    ("reduce", "rvv", 4096, "fp16", "fp32", 619),
    ("reduce", "scalar", 4096, "fp64", "fp64", 40967),
    ("scan", "ame", 4096, "fp64", "fp64", 3805),
    ("scan", "ame", 4096, "bf16", "fp32", 1739),
    ("scan", "ame", 4096, "fp16", "fp32", 1739),
    ("scan", "rvv", 4096, "fp64", "fp64", 14408),
    ("scan", "rvv", 4096, "bf16", "fp32", 9122),
    ("scan", "rvv", 4096, "fp16", "fp32", 9122),
    ("scan", "scalar", 4096, "fp64", "fp64", 57349),
    ("spmv", "ame", [128, 0.5], "fp64", "fp64", 2630),
    ("spmv", "ame", [128, 0.5], "bf16", "fp32", 1034),
    ("spmv", "ame", [128, 0.5], "fp16", "fp32", 1034),
    ("spmv", "rvv", [128, 0.5], "fp64", "fp64", 57274),
    ("spmv", "rvv", [128, 0.5], "bf16", "fp32", 22258),
    ("spmv", "rvv", [128, 0.5], "fp16", "fp32", 25650),
    ("spmv", "scalar", [128, 0.5], "fp64", "fp64", 754727),
    ("tensorprod", "ame", 8, "fp64", "fp64", 2407),
    ("tensorprod", "ame", 8, "bf16", "fp32", 1143),
    ("tensorprod", "ame", 8, "fp16", "fp32", 1143),
    ("tensorprod", "rvv", 8, "fp64", "fp64", 51993),
    ("tensorprod", "rvv", 8, "bf16", "fp32", 39699),
    ("tensorprod", "rvv", 8, "fp16", "fp32", 36738),
    ("tensorprod", "scalar", 8, "fp64", "fp64", 243474),
]
#: Points the formulas know to be off more than the default tolerance, with their bound.
LOOSE = {
    ("tensorprod", "scalar", 10): 0.25,
    ("apsp", "scalar", 32): 0.15,
    ("apsp", "ame", 32): 0.16,
}


def _key(size):
    return tuple(size) if isinstance(size, list) else size


@pytest.fixture(scope="module")
def recorded_errors():
    out = []
    for dwarf, variant, size, i, a, sim in RECORDED:
        pred = costmodel.predict(
            dwarf, _key(size), i, a, DWARFS_DESIGN, variants=[variant]
        )[variant]
        out.append(((dwarf, variant, _key(size), i, a), pred / sim - 1))
    return out


def test_recorded_points_within_tolerance(recorded_errors):
    bad = []
    for (dwarf, variant, size, i, a), err in recorded_errors:
        tol = LOOSE.get((dwarf, variant, size), 0.12)
        if abs(err) > tol:
            bad.append((dwarf, variant, size, i, a, round(err, 3)))
    assert not bad


def test_recorded_points_median_error(recorded_errors):
    assert statistics.median(abs(e) for _, e in recorded_errors) < 0.02


def test_recorded_decisions():
    """For every recorded problem, the predicted-fastest variant is the simulated fastest."""
    by = {}
    for dwarf, variant, size, i, a, sim in RECORDED:
        by.setdefault((dwarf, _key(size), i, a), {})[variant] = sim
    wrong = []
    for (dwarf, size, i, a), sims in by.items():
        if len(sims) < 2:
            continue
        pred = costmodel.predict(dwarf, size, i, a, DWARFS_DESIGN, variants=list(sims))
        if min(pred, key=pred.get) != min(sims, key=sims.get):
            wrong.append((dwarf, size, i, a))
    assert not wrong


# -- jacobi radius r > 1 and the halo switch (tloadh) -------------------------------------


def _with_halo():
    m = costmodel.machine(DWARFS_DESIGN)
    return dataclasses.replace(
        m, ame=dataclasses.replace(m.ame, features=(*m.ame.features, "halo"))
    )


@pytest.mark.parametrize(
    "size, halo, sim",
    [
        ((68, 2), False, 9558),
        ((68, 2), True, 9502),
        ((72, 4), False, 9649),
        ((72, 4), True, 9502),
    ],
)
def test_jacobi_radius_ame_tracks_the_simulator(size, halo, sim):
    """Cycles recorded on naiad-dwarfs at 2048 MACs/cycle (the halo runs add the halo feature)."""
    got = costmodel.predict(
        "jacobi",
        size,
        design=_with_halo(),
        variants=["ame"],
        switches={"halo": True} if halo else None,
    )["ame"]
    assert got == pytest.approx(sim, rel=0.03)


def test_jacobi_halo_needs_the_feature_and_a_radius():
    ame = costmodel.dwarf_module("jacobi").ame
    with pytest.raises(Unsupported):
        ame(costmodel.machine(DWARFS_DESIGN), (68, 2), "fp32", "fp32", halo=True)
    with pytest.raises(Unsupported):
        ame(_with_halo(), 66, "fp32", "fp32", halo=True)
    # the baselines are radius 1 only: they are left out, not mispredicted
    assert set(costmodel.predict("jacobi", (68, 2), design=DWARFS_DESIGN)) == {"ame"}


# -- RVV chaining (timing.vector.chaining) ---------------------------------------------------
#
# The cases are ``framework/pe/rv/timing/vector_test.py``'s chaining microbenchmarks restated on
# the engine: the same dependent pairs, the same analytic gaps between their issue cycles.


def chained(m=None):
    m = m or mach()
    return m.with_(vector=dataclasses.replace(m.vector, chaining=True))


def issue_cycles(m, text):
    e = engine.Engine(m)
    e.log = []
    listing.Program(text).run(e)
    return [c for c, _ in e.log]


def _pair_gap(lmul, m, first="vadd.vv v8, v16, v24", second="vadd.vv v0, v8, v8"):
    text = f"""
    0: vsetvli t0, zero, e32, m{lmul}, ta, ma
    4: {first}
    8: {second}
    """
    c = issue_cycles(m, text)
    return c[2] - c[1]


@pytest.mark.parametrize(
    "lmul,unchained,chained_gap",
    [(1, 5, 5), (2, 6, 5), (4, 8, 5), (8, 12, 8)],
)
def test_chaining_dependent_arithmetic_on_the_first_element_group(
    lmul, unchained, chained_gap
):
    assert _pair_gap(lmul, mach()) == unchained  # occupancy + startup
    # chained: 1 + startup, held by the datapath (occupancy) when that is longer
    assert _pair_gap(lmul, chained()) == chained_gap


def test_chaining_streams_a_dependent_sequence_at_the_first_group_rate():
    text = "0: vsetvli t0, zero, e32, m4, ta, ma\n" + "\n".join(
        f"{4 * (i + 1)}: vadd.vv v8, v8, v16" for i in range(6)
    )
    c = issue_cycles(chained(), text)
    assert [b - a for a, b in zip(c[1:], c[2:])] == [5] * 5
    c = issue_cycles(mach(), text)
    assert [b - a for a, b in zip(c[1:], c[2:])] == [8] * 5


def test_chaining_into_a_faster_consumer_waits_for_the_last_group():
    """vfdiv at LMUL 2 (2 beats of div_ii) into a vfadd: one cycle better than unchained."""
    div, add = "vfdiv.vv v8, v16, v24", "vfadd.vv v0, v8, v8"
    assert _pair_gap(2, mach(), div, add) == 2 * 8 + 4
    assert _pair_gap(2, chained(), div, add) == 2 * 8 + 4 - 1


@pytest.mark.parametrize("lmul", [1, 8])
def test_chaining_a_load_into_its_use(lmul):
    m = mach()
    t = 8 * lmul * 4 // 32  # LMUL registers over a 32 B LSU
    fixed = m.vector.mem_startup + m.memory.latency
    text = f"""
    0: vsetvli t0, zero, e32, m{lmul}, ta, ma
    4: vle32.v v8, (a0)
    8: vfadd.vv v16, v8, v8
    """
    c = issue_cycles(m, text)
    assert c[2] - c[1] == fixed + t
    c = issue_cycles(chained(m), text)
    assert c[2] - c[1] == fixed + 1


def test_chaining_leaves_what_the_hardware_reads_whole():
    """A reduction's scalar operand, vrgather's table and a slide-down's source are not chained."""
    for first, second in (
        (
            "vfadd.vv v8, v16, v24",
            "vfredusum.vs v0, v16, v11",
        ),  # vs1 (the last register) is the scalar operand
        ("vfadd.vv v8, v16, v24", "vrgather.vv v0, v8, v24"),
        ("vfadd.vv v8, v16, v24", "vslidedown.vi v0, v8, 1"),
    ):
        assert _pair_gap(4, chained(), first, second) == _pair_gap(
            4, mach(), first, second
        ), second
    # ... but a reduction's vector source and a slide-up source chain
    assert _pair_gap(
        4, chained(), "vfadd.vv v8, v16, v24", "vfredusum.vs v0, v8, v1"
    ) < _pair_gap(4, mach(), "vfadd.vv v8, v16, v24", "vfredusum.vs v0, v8, v1")
    # a reduction's result does not stream: its consumer waits for the whole tree
    red = "vfredusum.vs v0, v16, v8"
    assert _pair_gap(4, chained(), red, "vfadd.vv v1, v0, v0") == _pair_gap(
        4, mach(), red, "vfadd.vv v1, v0, v0"
    )


def test_chaining_keeps_one_datapath_and_one_lsu():
    """Independent ops still serialise on the datapath; chaining only lifts the RAW wait."""
    text = """
    0: vsetvli t0, zero, e32, m4, ta, ma
    4: vfadd.vv v8, v16, v24
    8: vfadd.vv v0, v12, v20
    """
    for m in (mach(), chained()):
        c = issue_cycles(m, text)
        assert c[2] - c[1] == 4


def test_chained_store_waits_for_its_data_stream():
    text = """
    0: vsetvli t0, zero, e32, m8, ta, ma
    4: vfadd.vv v8, v16, v24
    8: vse32.v v8, (a0)
    """
    assert (lambda c: c[2] - c[1])(issue_cycles(mach(), text)) == 8 + 4
    assert (lambda c: c[2] - c[1])(issue_cycles(chained(), text)) == 1 + 4


def test_chaining_never_slows_a_schedule_and_off_is_unchanged():
    base = costmodel.machine(DWARFS_DESIGN)
    assert base.vector.chaining is False
    on = base.with_(vector=dataclasses.replace(base.vector, chaining=True))
    for dwarf, size in (("gemm", 64), ("jacobi", 66), ("scan", 4096), ("reduce", 4096)):
        off = costmodel.predict(dwarf, size, "fp32", "fp32", base, variants=["rvv"])
        net = costmodel.predict(dwarf, size, "fp32", "fp32", on, variants=["rvv"])
        assert net["rvv"] <= off["rvv"], dwarf
    # the scalar and matrix variants do not see it
    a = costmodel.predict("gemm", 64, "fp32", "fp32", base, variants=["scalar", "ame"])
    b = costmodel.predict("gemm", 64, "fp32", "fp32", on, variants=["scalar", "ame"])
    assert a == b


def test_machine_reads_chaining_from_the_design(tmp_path):
    import yaml

    raw = yaml.safe_load(DWARFS_DESIGN.read_text())
    raw.setdefault("timing", {}).setdefault("vector", {})["chaining"] = True
    path = tmp_path / "chain.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert costmodel.machine(path).vector.chaining is True


#: Region cycles of the RVV variants on naiad-dwarfs.yaml with ``timing.vector.mem_bytes_per_cycle: 64`` and
#: ``chaining: true``, mid size, fp32 (the rvv_chaining sweep's ``rvv_control`` study):
#: (dwarf, size, chained cycles, unchained cycles).
RECORDED_CHAINED = [
    ("apsp", 64, 123523, 127619),
    ("fft", 1024, 60696, 61832),
    ("gemm", 64, 40231, 40307),
    ("jacobi", 66, 31509, 33045),
    ("nbody", (256, "clustered"), 248001, 270529),
    ("reduce", 16384, 2248, 2251),
    ("scan", 16384, 32853, 33625),
    ("spmv", (256, 0.25), 50370, 50370),
    ("tensorprod", 12, 141337, 141721),
]


def test_chained_rvv_predictions_track_the_simulator():
    """Chaining on, the schedule model predicts the simulated RVV region within the unchained model's own
    error, and it sees the chained-vs-unchained gain the simulator measures (jacobi -4.6%, nbody -8.3%)."""
    base = costmodel.machine(DWARFS_DESIGN)
    off = base.with_(vector=dataclasses.replace(base.vector, mem_bytes_per_cycle=64))
    on = off.with_(vector=dataclasses.replace(off.vector, chaining=True))
    for dwarf, size, chained_sim, unchained_sim in RECORDED_CHAINED:
        p_on = costmodel.predict(dwarf, size, "fp32", "fp32", on, variants=["rvv"])[
            "rvv"
        ]
        p_off = costmodel.predict(dwarf, size, "fp32", "fp32", off, variants=["rvv"])[
            "rvv"
        ]
        assert abs(p_on / chained_sim - 1) < 0.035, (dwarf, p_on, chained_sim)
        assert p_on <= p_off, dwarf
        # the predicted gain tracks the measured one to within a percentage point
        assert abs((p_on / p_off) - (chained_sim / unchained_sim)) < 0.012, dwarf


# -- dual issue (``Scalar.issue_width`` 2) ------------------------------------------------------------


def dual_mach(**kw):
    m = mach(**kw)
    return m.with_(scalar=dataclasses.replace(m.scalar, issue_width=2))


def test_issue_width_defaults_to_single_issue():
    assert Scalar().issue_width == 1
    e = engine.Engine(mach())
    e.alu(1)
    e.alu(1)
    assert e.cycles == 3  # one a cycle from cycle 1


def test_dual_issue_pairs_independent_alu_work():
    e = engine.Engine(dual_mach())
    e.alu(1)
    e.alu(1)
    e.alu(1)
    e.alu(1)
    assert e.cycles == 3  # two pairs: cycles 1 and 2
    e = engine.Engine(dual_mach())
    e.alu(5)  # two ALU pipes: ceil(5 / 2) cycles
    assert e.cycles == 1 + 3
    assert engine.Engine(mach()).t == 1


def test_dual_issue_raw_and_single_pipes_do_not_pair():
    # a RAW on slot 0's result fails the pair (every latency is >= 1)
    e = engine.Engine(dual_mach())
    e.alu(1, dst="x")
    e.alu(1, srcs=("x",))
    assert e.cycles == 3
    # one memory pipe: two scalar loads never pair, a load pairs with ALU work
    e = engine.Engine(dual_mach())
    e.load("a")
    e.load("b")
    assert e.cycles == 3
    e = engine.Engine(dual_mach())
    e.load("a")
    e.alu(1)
    assert e.cycles == 2
    # one vector / matrix dispatch port; a vector op pairs with scalar ALU work after it
    e = engine.Engine(dual_mach())
    e.vset()
    e.vset()
    assert e.cycles == 3
    e = engine.Engine(dual_mach())
    e.vset()
    e.alu(1)
    assert e.cycles == 2
    # a vector load takes the memory pipe too: it never pairs with a scalar load
    e = engine.Engine(dual_mach())
    e.vload("v", 32)
    e.load("a")
    assert e.cycles == 3


def test_dual_issue_branch_ends_the_issue_group():
    e = engine.Engine(dual_mach())
    e.branch(taken=False)
    e.alu(1)
    assert e.cycles == 3  # nothing issues after a branch in its cycle
    e = engine.Engine(dual_mach())
    e.alu(1)
    e.branch(taken=False)  # a branch may be slot 1
    assert e.cycles == 2


def test_dual_issue_fast_forward_matches_full_replay():
    def run(trips, warm=None):
        e = engine.Engine(dual_mach())

        def body(_):
            e.load("x", fp=True)
            e.alu(2)
            e.op("fadd", "s", ("s", "x"))
            e.vset()
            e.alu(1)
            e.branch()

        e.loop(trips, body, warm=warm, measure=None)
        return e.cycles

    assert run(300) == run(300, warm=400)  # fast-forwarded vs replayed in full


def test_dual_issue_bound_engine_halves_the_issue_demand():
    one = engine.BoundEngine(mach())
    two = engine.BoundEngine(dual_mach())
    for e in (one, two):
        e.alu(40)
    assert one.cycles == 40
    assert two.cycles == 20


def test_machine_reads_issue_width_from_the_design(tmp_path):
    import yaml

    assert costmodel.machine(DWARFS_DESIGN).scalar.issue_width == 1
    raw = yaml.safe_load(DWARFS_DESIGN.read_text())
    raw.setdefault("timing", {}).setdefault("scalar", {})["issue_width"] = 2
    path = tmp_path / "dual.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert costmodel.machine(path).scalar.issue_width == 2


def test_dual_issue_never_slows_a_schedule_and_single_is_unchanged():
    base = costmodel.machine(DWARFS_DESIGN)
    two = base.with_(scalar=dataclasses.replace(base.scalar, issue_width=2))
    for dwarf, size in (("gemm", 64), ("jacobi", 66), ("scan", 4096), ("reduce", 4096)):
        one = costmodel.predict(dwarf, size, "fp32", "fp32", base)
        dual = costmodel.predict(dwarf, size, "fp32", "fp32", two)
        assert set(one) == set(dual)
        for v in one:
            assert dual[v] <= one[v], (dwarf, v)
        assert dual["scalar"] < one["scalar"], dwarf


#: Region cycles on naiad-dwarfs.yaml with ``timing.vector.mem_bytes_per_cycle: 64`` and the AME at 2048
#: MACs/cycle, mid size, fp32 (the dual_issue sweep's ``crossover`` study): (dwarf, size) -> variant ->
#: (issue_width 1 cycles, issue_width 2 cycles).
RECORDED_DUAL = {
    ("apsp", 64): {
        "ame": (4919, 4697),
        "rvv": (127619, 106243),
        "scalar": (1741379, 1597762),
    },
    ("fft", 1024): {
        "ame": (125619, 104044),
        "ame_gauss": (299881, 248728),
        "rvv": (61832, 52623),
        "scalar": (727532, 622875),
    },
    ("gemm", 64): {
        "ame": (1317, 1253),
        "rvv": (40307, 36859),
        "scalar": (505805, 486519),
    },
    ("jacobi", 66): {
        "ame": (9510, 9045),
        "rvv": (33045, 30391),
        "scalar": (732317, 660063),
    },
    ("nbody", (256, "clustered")): {
        "ame": (4760440, 4389926),
        "ame_fgemm": (4214873, 3990887),
        "rvv": (270529, 264212),
        "scalar": (4649493, 4385803),
    },
    ("reduce", 16384): {
        "ame": (1387, 1354),
        "rvv": (2251, 2112),
        "scalar": (163847, 147461),
    },
    ("scan", 16384): {
        "ame": (7611, 7356),
        "rvv": (33625, 30797),
        "scalar": (229381, 196611),
    },
    ("spmv", (256, 0.25)): {
        "ame": (2584, 2503),
        "rvv": (50370, 46674),
        "scalar": (1460286, 1357052),
    },
    ("tensorprod", 12): {
        "ame": (5158, 4635),
        "rvv": (141721, 114118),
        "scalar": (1109080, 1026090),
    },
}


def _dual_machine(width):
    base = costmodel.machine(DWARFS_DESIGN)
    m = base.with_(
        scalar=dataclasses.replace(base.scalar, issue_width=width),
        vector=dataclasses.replace(base.vector, mem_bytes_per_cycle=64),
    )
    return costmodel.with_matrix_peak(m, 2048)


def test_dual_issue_predictions_track_the_simulator():
    """The schedule model at issue_width 2 predicts the dual-issue simulator, and the width-1 predictions are
    what they were. tensorprod's AME streams keep a compiled address-arithmetic count whose dependences the
    model does not see (they pair as if independent), so its dual-issue gain is overpredicted (-13%)."""
    one, two = _dual_machine(1), _dual_machine(2)
    errs = []
    for (dwarf, size), variants in RECORDED_DUAL.items():
        names = list(variants)
        p1 = costmodel.predict(dwarf, size, "fp32", "fp32", one, variants=names)
        p2 = costmodel.predict(dwarf, size, "fp32", "fp32", two, variants=names)
        for v, (sim1, sim2) in variants.items():
            assert abs(p1[v] / sim1 - 1) < 0.03, (dwarf, v, "width 1")
            err = p2[v] / sim2 - 1
            limit = 0.15 if (dwarf, v) == ("tensorprod", "ame") else 0.06
            assert abs(err) < limit, (dwarf, v, p2[v], sim2)
            assert p2[v] <= p1[v], (dwarf, v)
            errs.append(abs(err))
    assert statistics.median(errs) < 0.015
