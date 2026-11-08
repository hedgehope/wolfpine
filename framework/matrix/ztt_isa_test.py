"""Ztt v0.6 execution: every implemented instruction against numpy, the draft's check order, ownership, CSRs.

Each test builds a real ``RV64I`` with Zicsr over a DRAM, attaches a :class:`ZttHart`, and
executes Ztt words by placing them at the PC and clocking the core. References are
written independently of the executor: numpy for integers (exact, then wrapped) and an
explicit per-k loop in fp32 for the matrix multiply's documented unfused order.
"""

import importlib.util

import numpy as np
import pytest

from framework.device.device import DeviceMemory
from framework.matrix import ztt
from framework.matrix.ztt import descriptor, encode
from framework.matrix.ztt_isa import (
    IMPLEMENTED,
    ZttBackend,
    ZttHart,
    ZttIllegal,
    ZttMemoryFault,
    ZttTrap,
    attach,
)
from framework.memory.memory import DRAM
from framework.memory.memory_map import AddressRange, MemoryMap
from framework.pe.rv.isa.isa64_testing import ALL, csr, reg, set_reg
from framework.pe.rv.isa.zicsr_isa import CSRError
from framework.pe.rv.rv64 import RV64I

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)

PC = 0x1000
MEM = 0x40000
A_ADDR, B_ADDR, C_ADDR = 0x8000, 0x9000, 0xA000
T0, T1, T2 = 5, 6, 7
MS = 1 << ztt.DEFAULT_MS_SHIFT
N = 16
RNG = np.random.default_rng(5)
TYPES = ("int8", "uint8", "int32", "int16", "fp16", "bf16", "fp32", "fp64")


def spec(**kw):
    kw.setdefault("datatypes", TYPES)
    kw.setdefault("matmul", ztt.default_matmul(kw["datatypes"]))
    return ztt.ZttSpec(**kw)


class ZCore:
    def __init__(self, s=None, backend=None, hart_id=0):
        mm = MemoryMap()
        mm[AddressRange(0, MEM)] = DRAM(MEM)
        self.core = RV64I(
            PC, [DeviceMemory(mm)], list(ALL), unknown_instr_is_error=True
        )
        self.core.start()
        self.backend = backend or ZttBackend(s or spec())
        self.hart = attach(self.core, ZttHart(self.backend, hart_id, None))
        self.cycle = 0

    @property
    def mem(self):
        return self.core.visible_memory

    def x(self, i, value=None):
        if value is not None:
            set_reg(self.core, i, value)
        return reg(self.core, i)

    def exec(self, word, **regs):
        for name, value in regs.items():
            self.x(int(name[1:]), value)
        pc = reg(self.core, "pc")
        self.mem.write(pc, word.to_bytes(4, "little"))
        self.core.clock_tick(self.cycle)
        self.cycle += 1

    def csrr(self, addr):
        self.exec(csr(2, addr, rd=T2, rs1=0))
        return self.x(T2)

    def csrw(self, addr, value):
        self.exec(csr(1, addr, rd=0, rs1=T2), x7=value)

    def enable(self, acquire=True):
        self.exec(csr(2, 0x300, rd=0, rs1=T0), x5=MS)
        if acquire:
            self.exec(encode("ame.acquire", rd=T1, rs1=0))
            assert self.x(T1) == 1
        return self

    def settyp(self, md, name, **kw):
        self.exec(encode("msettyp", md=md, rs1=T0), x5=descriptor(name, **kw))

    def asettyp(self, ad, name, **kw):
        self.exec(encode("asettyp", ad=ad, rs1=T0), x5=descriptor(name, **kw))

    def un(self):
        return self.backend.amestatus & 1

    def clear_un(self):
        self.backend.amestatus = 0

    def put(self, addr, array):
        self.mem.write(addr, np.ascontiguousarray(array).tobytes())

    def get(self, addr, dtype, shape=(N, N)):
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return np.frombuffer(bytes(self.mem.read(addr, n)), dtype).reshape(shape)

    def load(self, md, name, array, addr=A_ADDR):
        self.settyp(md, name)
        self.put(addr, array)
        self.exec(encode("mls.rm", md=md, rs1=T1), x6=addr)

    def store(self, ms, addr=C_ADDR):
        self.exec(encode("mss.rm", ms1=ms, rs1=T1), x6=addr)

    def acc_out(self, acc, name, dtype, via=12):
        """acc -> m<via> (same type) -> memory -> array."""
        self.settyp(via, name)
        self.exec(encode("mmov.m.a", md=via, acc=acc))
        self.store(via)
        return self.get(C_ADDR, dtype)


def ints(lo, hi, dtype):
    return RNG.integers(lo, hi, (N, N)).astype(dtype)


def floats(dtype, scale=4.0):
    import ml_dtypes

    dt = {
        "bf16": ml_dtypes.bfloat16,
        "fp16": np.float16,
        "fp32": np.float32,
        "fp64": np.float64,
    }[dtype]
    return (RNG.standard_normal((N, N)) * scale).astype(dt)


# -- state access, ownership, CSRs ---------------------------------------------------------------


def test_ms_off_makes_every_instruction_and_csr_illegal():
    z = ZCore()
    with pytest.raises(ZttIllegal, match="MS Off"):
        z.exec(encode("ame.acquire", rd=T1, rs1=0))
    with pytest.raises(CSRError, match="MS Off"):
        z.csrr(0xCC0)


def test_without_ownership_only_acquire_release_and_capability_csrs_work():
    z = ZCore().enable(acquire=False)
    with pytest.raises(ZttIllegal, match="without owning"):
        z.exec(encode("mzero.2d.acc", acc=0))
    assert z.csrr(0xCC0) == N
    assert z.csrr(0xCC1) == 8
    assert z.csrr(0xCC3) == 0
    with pytest.raises(CSRError, match="without owning"):
        z.csrr(0x800)
    z.exec(encode("ame.release"))  # a non-owner's release is a no-op
    z.exec(encode("ame.acquire", rd=T1, rs1=0))
    assert z.x(T1) == 1
    assert z.csrr(0xCC3) == 1


def test_acquire_binds_initial_state_and_sets_ms_initial():
    z = ZCore().enable()
    z.load(0, "int8", ints(-5, 5, np.int8))
    assert (z.core.csrs.read(0x300) >> ztt.DEFAULT_MS_SHIFT) & 3 == ztt.MS_DIRTY
    assert z.core.csrs.read(0x300) >> 63  # SD
    z.exec(
        encode("ame.acquire", rd=T1, rs1=0)
    )  # already owned: GRANTED, nothing changes
    assert z.x(T1) == 1
    assert z.backend.md[0] == descriptor("int8")
    z.exec(encode("ame.release"))
    assert z.csrr(0xCC3) == 0
    assert (z.core.csrs.read(0x300) >> ztt.DEFAULT_MS_SHIFT) & 3 == ztt.MS_INITIAL
    z.exec(encode("ame.acquire", rd=T1, rs1=0))
    assert z.backend.md[0] == 0
    assert not z.backend.m.any()


@pytest.mark.parametrize(
    "desc",
    [1 << 6, 2, 3, 1 | (5 << 2)],
    ids=["reserved", "mode 2", "mode 3", "class 5"],
)
def test_bad_acquire_descriptor(desc):
    z = ZCore().enable(acquire=False)
    z.exec(encode("ame.acquire", rd=T1, rs1=T0), x5=desc)
    assert z.x(T1) == ztt.ACQ_BAD_DESC << 1
    assert z.csrr(0xCC3) == 0


def test_a_shared_backend_has_one_owner():
    backend = ZttBackend(spec(backend="shared"))
    h0, h1 = (
        ZCore(backend=backend, hart_id=0).enable(),
        ZCore(backend=backend, hart_id=1).enable(acquire=False),
    )
    h1.exec(encode("ame.acquire", rd=T1, rs1=0))
    assert h1.x(T1) == ztt.ACQ_BUSY << 1
    h1.exec(encode("ame.acquire", rd=T1, rs1=T0), x5=1)  # bounded wait
    assert h1.x(T1) == ztt.ACQ_TIMEOUT << 1
    h0.exec(encode("ame.release"))
    h1.exec(encode("ame.acquire", rd=T1, rs1=0))
    assert h1.x(T1) == 1
    assert h1.csrr(0xCC3) == 1
    assert h0.csrr(0xCC3) == 0


def test_ms_cannot_go_off_while_owned():
    z = ZCore().enable()
    with pytest.raises(CSRError, match="ame.release first"):
        z.exec(csr(3, 0x300, rd=0, rs1=T0), x5=3 << ztt.DEFAULT_MS_SHIFT)


def test_context_csrs():
    z = ZCore().enable()
    z.exec(encode("mexp2.ew", md=0, ms1=0))  # not implemented: UN
    assert z.csrr(0x800) == 1
    z.csrw(0x800, 1)  # writing one leaves it as it is
    assert z.csrr(0x800) == 1
    z.csrw(0x800, 0)
    assert z.csrr(0x800) == 0
    z.csrw(
        0xCC2, 0x12345678
    )  # amestype keeps any encoding (0xCC2 is writable despite its range)
    assert z.csrr(0xCC2) == 0x12345678
    z.csrw(0xCC4, 0xFF)
    assert z.csrr(0xCC4) == 0x1F
    with pytest.raises(CSRError, match="read-only"):
        z.csrw(0xCC0, 1)


# -- the check order ------------------------------------------------------------------------------


def test_register_indices_are_checked_first():
    z = ZCore().enable()
    with pytest.raises(ZttIllegal, match="m16"):
        z.exec(
            encode("mexp2.ew", md=16, ms1=0)
        )  # even for an instruction that would report UN
    with pytest.raises(ZttIllegal, match="acc8"):
        z.exec(encode("mzero.2d.acc", acc=8))
    with pytest.raises(ZttIllegal, match="different registers"):
        z.exec(encode("mrowzip.ew", ms1=3, ms2=3))
    assert z.un() == 0


def test_unsupported_tuple_before_group_alignment():
    z = ZCore(spec(datatypes=("int8", "int32", "fp32"))).enable()
    z.settyp(2, "bf16")  # unsupported: UN, Md unchanged
    assert z.un() == 1
    assert z.backend.md[2] == 0
    with pytest.raises(ZttIllegal, match="4-register fp32 group"):
        z.settyp(2, "fp32")
    z.settyp(4, "fp32")
    assert z.backend.md[4] == descriptor("fp32")


def test_msettyp_clears_the_whole_group_and_keeps_sticky_descriptors():
    z = ZCore().enable()
    z.backend.m[:] = 0xAB
    z.backend.md[5] = descriptor("int8")
    z.settyp(4, "fp32")
    assert not z.backend.m[4:8].any()
    assert z.backend.m[8].all()
    assert z.backend.md[5] == descriptor("int8")  # sticky, though grouped under m4
    z.exec(encode("mgettyp", rd=T1, ms1=4))
    assert z.x(T1) == descriptor("fp32")


def test_every_other_instruction_reports_un():
    z = ZCore().enable()
    skipped = set(ztt.TABLE) - IMPLEMENTED
    assert "mexp2.ew" in skipped
    assert "mmulacc.2d" not in skipped
    for name in sorted(skipped):
        insn = ztt.TABLE[name]
        ops = {
            op: (1 if op == "ms2" else 0) for op in insn.operands if op not in ("rd",)
        }
        z.clear_un()
        z.exec(encode(name, **ops))
        assert z.un() == 1, name


# -- memory ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, dtype",
    [("int8", np.int8), ("bf16", "bf16"), ("fp32", np.float32), ("fp64", np.float64)],
)
def test_row_and_column_major_round_trips(name, dtype):
    z = ZCore().enable()
    a = floats(name) if name in ("bf16", "fp32", "fp64") else ints(-100, 100, np.int8)
    z.load(0, name, a)
    z.store(0)
    assert np.array_equal(z.get(C_ADDR, a.dtype).view(np.uint8), a.view(np.uint8))
    z.exec(
        encode("mss.cm", ms1=0, rs1=T1), x6=C_ADDR
    )  # column-major in memory: the transpose
    assert np.array_equal(
        z.get(C_ADDR, a.dtype).view(np.uint8), np.ascontiguousarray(a.T).view(np.uint8)
    )
    z.put(B_ADDR, a)
    z.exec(encode("mls.cm", md=0, rs1=T1), x6=B_ADDR)
    z.store(0)
    assert np.array_equal(
        z.get(C_ADDR, a.dtype).view(np.uint8), np.ascontiguousarray(a.T).view(np.uint8)
    )


def test_strided_and_transposed_strided():
    z = ZCore().enable()
    stride = 40
    big = ints(-100, 100, np.int8)
    for i in range(N):
        z.put(A_ADDR + i * stride, big[i])
    z.settyp(0, "int8")
    z.exec(encode("mls.st", md=0, rs1=T1, rs2=T2), x6=A_ADDR, x7=stride)
    z.store(0)
    assert np.array_equal(z.get(C_ADDR, np.int8), big)
    z.settyp(1, "int8")
    z.exec(
        encode("mls.tst", md=1, rs1=T1, rs2=T2), x6=A_ADDR, x7=stride
    )  # segments are columns
    z.store(1)
    assert np.array_equal(z.get(C_ADDR, np.int8), big.T)
    z.exec(encode("mss.tst", ms1=0, rs1=T1, rs2=T2), x6=B_ADDR, x7=stride)
    got = np.stack([z.get(B_ADDR + j * stride, np.int8, (N,)) for j in range(N)])
    assert np.array_equal(got, big.T)


def test_overlapping_strided_store_keeps_the_last_segment():
    z = ZCore().enable()
    a = ints(-100, 100, np.int8)
    z.load(0, "int8", a)
    z.exec(encode("mss.st", ms1=0, rs1=T1, rs2=T2), x6=C_ADDR, x7=0)
    assert np.array_equal(z.get(C_ADDR, np.int8, (N,)), a[N - 1])


def test_whole_register_transfers_are_the_row_major_image():
    z = ZCore().enable()
    a = floats("fp32")
    z.load(4, "fp32", a)  # m4-m7
    z.exec(encode("mss.1r", ms1=5, rs1=T1), x6=C_ADDR)
    assert bytes(z.mem.read(C_ADDR, 256)) == a.tobytes()[256:512]
    z.backend.md[9] = 0  # .1r needs no datatype
    z.exec(encode("mls.1r", md=9, rs1=T1), x6=C_ADDR)
    assert bytes(z.backend.m[9]) == a.tobytes()[256:512]


def test_misaligned_base():
    z = ZCore().enable()
    z.settyp(4, "fp32")
    with pytest.raises(ZttTrap) as e:
        z.exec(encode("mls.rm", md=4, rs1=T1), x6=A_ADDR + 2)
    assert (e.value.rv_cause, e.value.rv_tval) == (4, A_ADDR + 2)
    with pytest.raises(ZttTrap) as e:
        z.exec(encode("mss.rm", ms1=4, rs1=T1), x6=A_ADDR + 2)
    assert e.value.rv_cause == 6
    z.settyp(0, "int8")
    z.exec(encode("mls.rm", md=0, rs1=T1), x6=A_ADDR + 1)  # int8: any byte address


def test_faults_write_nothing():
    z = ZCore().enable()
    a = floats("fp32")
    z.load(4, "fp32", a)
    edge = MEM - 512  # half the square past the end of memory
    before = bytes(z.mem.read(edge, 512))
    with pytest.raises(ZttMemoryFault) as e:
        z.exec(encode("mss.rm", ms1=4, rs1=T1), x6=edge)
    assert (e.value.rv_cause, e.value.rv_tval) == (7, MEM)
    assert bytes(z.mem.read(edge, 512)) == before
    image = z.backend.m.copy()
    with pytest.raises(ZttMemoryFault) as e:
        z.exec(encode("mls.rm", md=4, rs1=T1), x6=edge)
    assert e.value.rv_cause == 5
    assert np.array_equal(z.backend.m, image)
    with pytest.raises(
        ZttMemoryFault
    ) as e:  # a strided store whose last row is unmapped
        z.exec(encode("mss.st", ms1=4, rs1=T1, rs2=T2), x6=MEM - 15 * 1024, x7=1024)
    assert e.value.rv_tval == MEM
    assert bytes(z.mem.read(MEM - 15 * 1024, 64)) == bytes(64)


# -- matrix multiply ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ta, tb",
    [("int8", "int8"), ("uint8", "int8"), ("int8", "uint8"), ("uint8", "uint8")],
)
@pytest.mark.parametrize(
    "name",
    [
        "mmulacc.2d",
        "mmulaccneg.2d",
        "mmulatacc.2d",
        "mmulataccneg.2d",
        "mmulbtacc.2d",
        "mmulbtaccneg.2d",
    ],
)
def test_integer_matmul(name, ta, tb):
    z = ZCore().enable()
    a = ints(-128, 128, np.int8) if ta == "int8" else ints(0, 256, np.uint8)
    b = ints(-128, 128, np.int8) if tb == "int8" else ints(0, 256, np.uint8)
    c0 = RNG.integers(-(1 << 31), 1 << 31, (N, N)).astype(np.int32)
    z.load(0, ta, a)
    z.load(1, tb, b, B_ADDR)
    z.load(4, "int32", c0, C_ADDR)
    z.asettyp(2, "int32")
    z.exec(encode("mmov.a.m", acc=2, ms=4))
    z.exec(encode(name, acc=2, ms1=0, ms2=1))
    A, B = a.astype(np.int64), b.astype(np.int64)
    if name.startswith("mmulat"):
        A = A.T
    if name.startswith("mmulbt"):
        B = B.T
    prod = A @ B
    want = (c0.astype(np.int64) + (-prod if "neg" in name else prod)).astype(
        np.int32
    )  # wraps
    assert np.array_equal(z.acc_out(2, "int32", np.int32), want)
    assert z.hart.stats["macs"] == N**3


def _fp32_reference(c, a, b, neg=False):
    """The documented fold: from the accumulator, q = 0..N-1, each product rounded to fp32, then added."""
    c = c.astype(np.float32).copy()
    for q in range(N):
        p = np.multiply.outer(a[:, q].astype(np.float32), b[q, :].astype(np.float32))
        c = (c - p) if neg else (c + p)
    return c


@pytest.mark.parametrize("fmt", ["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("neg", [False, True])
def test_float_matmul_is_the_unfused_ordered_fold(fmt, neg):
    z = ZCore().enable()
    a, b = floats(fmt), floats(fmt)
    c0 = floats("fp32", 100.0)
    g = 4 if fmt == "fp32" else 2
    z.load(0, fmt, a)
    z.load(g, fmt, b, B_ADDR)
    z.load(8, "fp32", c0, C_ADDR)
    z.asettyp(0, "fp32")
    z.exec(encode("mmov.a.m", acc=0, ms=8))
    z.exec(encode("mmulaccneg.2d" if neg else "mmulacc.2d", acc=0, ms1=0, ms2=g))
    got = z.acc_out(0, "fp32", np.float32)
    assert np.array_equal(
        got.view(np.uint32), _fp32_reference(c0, a, b, neg).view(np.uint32)
    )


def test_matmul_tuples_outside_the_list_are_unsupported():
    z = ZCore().enable()
    z.load(0, "int8", ints(-5, 5, np.int8))
    z.load(2, "bf16", floats("bf16"), B_ADDR)
    z.asettyp(0, "int32")
    z.exec(encode("mmulacc.2d", acc=0, ms1=0, ms2=2))
    assert z.un() == 1
    assert not z.backend.acc[0].any()
    z.clear_un()
    z.load(1, "int8", ints(-5, 5, np.int8), B_ADDR)
    z.asettyp(1, "int32", sat=True)  # a saturating accumulator
    z.exec(encode("mmulacc.2d", acc=1, ms1=0, ms2=1))
    assert z.un() == 1


def test_mzero_and_moves():
    z = ZCore().enable()
    a = ints(-100, 100, np.int32)
    z.load(4, "int32", a)
    z.asettyp(1, "int32")
    z.exec(encode("mmov.a.m", acc=1, ms=4))
    z.settyp(8, "int32")
    z.exec(encode("mmov.m.m", md=8, ms=4))
    z.store(8)
    assert np.array_equal(z.get(C_ADDR, np.int32), a)
    z.exec(encode("mzero.2d.acc", acc=1))
    assert np.array_equal(z.acc_out(1, "int32", np.int32), np.zeros((N, N), np.int32))
    z.settyp(0, "int8")
    z.exec(encode("mmov.m.a", md=0, acc=1))  # int8 vs int32: UN, m0 unchanged
    assert z.un() == 1


def test_mbcast_and_mmove():
    z = ZCore().enable()
    z.settyp(4, "fp32")
    z.exec(
        encode("mbcast.m.x", md=4, rs1=T0, rs2=T1),
        x5=-3 & ((1 << 64) - 1),
        x6=descriptor("int32"),
    )
    z.store(4)
    assert np.array_equal(z.get(C_ADDR, np.float32), np.full((N, N), -3.0, np.float32))
    z.settyp(0, "int8")
    z.exec(
        encode("mmove32.m.x", md=0, rs2=T0, rs1=T1), x5=0xDEADBEEF, x6=5 + 64
    )  # pos taken mod 64
    assert bytes(z.backend.m[0, 20:24]) == (0xDEADBEEF).to_bytes(4, "little")
    z.exec(encode("mmove16.x.m", rd=T2, ms2=0, rs1=T1), x6=11)
    assert z.x(T2) == 0xDEAD


def test_mmove_wider_than_a_row_is_illegal():
    z = ZCore(spec(nelem=4, datatypes=("int8", "int32"))).enable()  # rows of 2 bytes
    with pytest.raises(ZttIllegal, match="narrower"):
        z.exec(encode("mmove32.x.m", rd=T2, ms2=0, rs1=0))


# -- element-wise, conversions, reductions --------------------------------------------------------


def test_integer_elementwise_wraps_or_saturates():
    z = ZCore().enable()
    a, b = np.full((N, N), 100, np.int8), np.full((N, N), 100, np.int8)
    for r, name, kw in ((0, "int8", {}), (2, "int8", {"sat": True})):
        z.settyp(r, name, **kw)
        z.put(A_ADDR, a)
        z.exec(encode("mls.rm", md=r, rs1=T1), x6=A_ADDR)
        z.settyp(r + 1, name, **kw)
        z.put(B_ADDR, b)
        z.exec(encode("mls.rm", md=r + 1, rs1=T1), x6=B_ADDR)
    z.exec(encode("madd.ew", md=0, ms1=0, ms2=1))
    z.store(0)
    assert (z.get(C_ADDR, np.int8) == -56).all()
    assert z.backend.amexsat == 0
    z.exec(encode("madd.ew", md=2, ms1=2, ms2=3))
    z.store(2)
    assert (z.get(C_ADDR, np.int8) == 127).all()
    assert z.backend.amexsat == 1


def test_msub_is_b_minus_a():
    z = ZCore().enable()
    z.load(0, "int32", np.full((N, N), 7, np.int32))
    z.load(4, "int32", np.full((N, N), 2, np.int32), B_ADDR)
    z.exec(encode("msub.ew", md=8, ms1=0, ms2=4))  # m8 is untyped: UN
    assert z.un() == 1
    z.settyp(8, "int32")
    z.exec(encode("msub.ew", md=8, ms1=0, ms2=4))
    z.store(8)
    assert (z.get(C_ADDR, np.int32) == -5).all()


def test_fp_flags_and_canonical_nan():
    z = ZCore().enable()
    a = np.zeros((N, N), np.float32)
    b = np.zeros((N, N), np.float32)
    a[0, 0], b[0, 0] = 1.0, 2.0**-30  # inexact
    a[0, 1], b[0, 1] = 3e38, 3e38  # overflow
    a[0, 2], b[0, 2] = np.inf, -np.inf  # invalid
    z.load(0, "fp32", a)
    z.load(4, "fp32", b, B_ADDR)
    z.settyp(8, "fp32")
    z.exec(encode("madd.ew", md=8, ms1=0, ms2=4))
    z.store(8)
    got = z.get(C_ADDR, np.float32)
    assert got[0, 0] == 1.0
    assert np.isinf(got[0, 1])
    assert got.view(np.uint32)[0, 2] == 0x7FC00000
    assert z.backend.amefflags == 0x10 | 0x04 | 0x01  # NV, OF, NX
    z.backend.amefflags = 0
    z.exec(encode("mmul.ew", md=8, ms1=0, ms2=0))
    assert z.backend.amefflags & 0x04  # 3e38 squared overflows


def test_fp_min_max_signed_zero():
    z = ZCore().enable()
    z.load(0, "fp32", np.full((N, N), -0.0, np.float32))
    z.load(4, "fp32", np.full((N, N), 0.0, np.float32), B_ADDR)
    z.settyp(8, "fp32")
    z.exec(encode("mmax.ew", md=8, ms1=0, ms2=4))
    z.store(8)
    assert not np.signbit(z.get(C_ADDR, np.float32)).any()
    z.exec(encode("mmin.ew", md=8, ms1=0, ms2=4))
    z.store(8)
    assert np.signbit(z.get(C_ADDR, np.float32)).all()


def test_scalar_forms_use_amestype():
    z = ZCore().enable()
    z.load(4, "fp32", np.full((N, N), 1.5, np.float32))
    z.settyp(8, "fp32")
    z.exec(encode("madd.ew.x", md=8, rs1=T0, ms2=4), x5=2)  # amestype uninitialised: UN
    assert z.un() == 1
    z.csrw(0xCC2, descriptor("int32"))
    z.exec(encode("msub.ew.x", md=8, rs1=T0, ms2=4), x5=2)  # B - c
    z.store(8)
    assert (z.get(C_ADDR, np.float32) == -0.5).all()


def test_conversions():
    import ml_dtypes

    z = ZCore().enable()
    a = floats("fp32")
    a[0, :4] = [300.7, -2.5, 2.5, np.nan]
    z.load(4, "fp32", a)
    z.settyp(0, "bf16")
    z.exec(encode("mconv.ew", md=0, ms1=4))
    z.store(0)
    assert np.array_equal(
        z.get(C_ADDR, ml_dtypes.bfloat16)[1:].view(np.uint16),
        a[1:].astype(ml_dtypes.bfloat16).view(np.uint16),
    )
    z.settyp(1, "int8", sat=True)  # rounding RNU
    z.exec(encode("mconv.ew", md=1, ms1=4))
    z.store(1)
    assert list(z.get(C_ADDR, np.int8)[0, :4]) == [127, -2, 3, 0]
    assert z.backend.amexsat == 1
    assert z.backend.amefflags & 0x10
    z.settyp(2, "int8", sat=True, rnd=ztt.INT_RNE)
    z.exec(encode("mconv.ew", md=2, ms1=4))
    z.store(2)
    assert list(z.get(C_ADDR, np.int8)[0, 1:3]) == [-2, 2]
    z.backend.amefflags = 0
    z.load(8, "int32", np.full((N, N), 16777217, np.int32))
    z.settyp(12, "fp32")
    z.exec(encode("mconv.ew", md=12, ms1=8))
    z.store(12)
    assert (z.get(C_ADDR, np.float32) == 16777216.0).all()
    assert z.backend.amefflags == 0x01


def test_reductions_and_prefixes():
    z = ZCore().enable()
    a = RNG.integers(-1000, 1000, (N, N)).astype(np.int32)
    z.load(4, "int32", a)
    z.settyp(8, "int32")
    z.exec(encode("mreduceadd.row", md=8, ms1=4))
    z.store(8)
    assert np.array_equal(
        z.get(C_ADDR, np.int32), np.repeat(a.sum(1, keepdims=True), N, 1)
    )
    z.exec(encode("mprefixadd.col", md=8, ms1=4))
    z.store(8)
    assert np.array_equal(z.get(C_ADDR, np.int32), np.cumsum(a, 0))
    z.exec(encode("mreducemin.col", md=8, ms1=4))
    z.store(8)
    assert np.array_equal(
        z.get(C_ADDR, np.int32), np.repeat(a.min(0, keepdims=True), N, 0)
    )


def test_float_reduction_folds_in_order_in_the_destination_type():
    z = ZCore().enable()
    a = np.zeros((N, N), np.float32)
    a[0, :3] = [1e8, 1.0, -1e8]  # in fp32, in order: (1e8 + 1) - 1e8 = 0
    a[1, 0] = np.nan
    z.load(4, "fp32", a)
    z.settyp(8, "fp32")
    z.exec(encode("mreduceadd.row", md=8, ms1=4))
    z.store(8)
    got = z.get(C_ADDR, np.float32)
    assert (got[0] == 0.0).all()
    assert (got.view(np.uint32)[1] == 0x7FC00000).all()
    z.settyp(
        8, "fp64"
    )  # a widening reduction, fp32 -> fp64: exact (fp64 groups start at m0 or m8)
    z.exec(encode("mreduceadd.row", md=8, ms1=4))
    z.store(8)
    assert (z.get(C_ADDR, np.float64)[0] == 1.0).all()


def test_row_and_column_shift():
    z = ZCore().enable()
    a = ints(-100, 100, np.int32)
    z.load(4, "int32", a)
    z.settyp(8, "int32")
    z.exec(encode("mrowshift.ew.x", md=8, rs1=T0, ms2=4), x5=3)
    z.store(8)
    want = np.zeros_like(a)
    want[: N - 3] = a[3:]
    assert np.array_equal(z.get(C_ADDR, np.int32), want)
    z.exec(encode("mcolshift.ew.x", md=8, rs1=T0, ms2=4), x5=-2 & 0xFFFFFFFF)
    z.store(8)
    want = np.zeros_like(a)
    want[:, 2:] = a[:, : N - 2]
    assert np.array_equal(z.get(C_ADDR, np.int32), want)
