"""Emit the GEMM micro-kernel and blocked driver for a design's matrix unit.

``python -m framework.naiad.emit.gemm <design.yaml> -o naiad_gemm.h [--block 2x2]
[--pairs fp32:fp32,bf16:fp32] [--single-buffer]``

The output is one C header (``naiad_gemm.h``, written by ``kernels/naiad/Makefile``
next to the design's ``naiad.h``) holding, for every (input, accumulator) format
pair the design implements, a family of functions named ``naiad_gemm_<in>_<acc>``:

``..._ukernel(kc, A, lda, B, ldb, C, ldc, accumulate)``
    The register-blocked micro-kernel: ``C[MB x NB] = (accumulate ? C : 0) +
    A[MB x kc] . B[kc x NB]`` with ``MB = BM * M`` and ``NB = BN * N`` (the
    blocking times the unit's tile shape) and ``kc`` a multiple of the tile K.
    Static inline. ``BM x BN`` accumulators stay resident across the K loop; with
    enough tile registers the loop is double-buffered (the loads of step s+1
    are interleaved with the mma of step s), otherwise single-buffered.
``..._init`` / ``..._kloop`` / ``..._store`` (and ``_kload0`` + ``_kbody`` = ``_kloop``, ``_store_kload0``)
    The micro-kernel's phases, for kernels that build their own blocking:
    ``_kload0`` issues step 0's loads (they need no accumulator, so they can go
    before ``_init``), and ``_store_kload0`` interleaves one block's stores with
    the next block's step-0 loads, which the driver uses across full blocks.
``naiad_gemm_<in>_<acc>(m, n, k, A, lda, B, ldb, C, ldc)``
    The blocked driver, ``C = A . B``; ``..._acc`` is ``C += A . B``. Any m, n, k.
    **Edges are padded, not computed in scalar code**: an output block that is
    full in M and N runs the micro-kernel straight from the caller's arrays,
    with a K remainder packed into zero-padded staging tiles for one extra
    step; a block cut by the M or N edge packs every step's partial tiles into
    the staging tiles, accumulates into a staging C block, and copies the valid
    part out. Every product therefore goes through the unit in the pair's
    formats, and the padding shows up as extra MACs and scalar packing.

Strides (``lda``, ``ldb``, ``ldc``) are in elements. Element types are
``naiad_<fmt>_t`` from ``dwarf_util.h`` (fp16/bf16 as raw ``uint16_t`` bits).
Defining ``NAIAD_GEMM_IN`` and ``NAIAD_GEMM_ACC`` (format names, e.g. ``fp32``)
before the include also gives the unsuffixed ``naiad_gemm``,
``naiad_gemm_acc``, ``naiad_gemm_ukernel`` and the ``naiad_gemm_in_t`` /
``naiad_gemm_acc_t`` types for that pair.

Every encoding (format codes, opcodes) comes from ``naiad.h`` through
``matrix.h``; the emitted shape and blocking are checked against ``naiad.h`` at
compile time, so a stale header fails to build rather than computing garbage.

**Dual issue.** For a design with ``timing.scalar.issue_width: 2`` the K loop (``_kbody``) keeps one
persistent pointer per tile and advances it inside the tile-load statement that reads it
(``mx_tload_a_adv`` / ``mx_tload_b_adv``: the load, then one ``add`` in the same asm), so the advance
pairs with the next matrix op instead of an address computation sitting in front of every load. Single
issue emits the text it always did.

The attached mode (AME) is this module; an IME design gets the IME micro-kernel
(:mod:`framework.naiad.emit.gemm_ime`, same API) and a VME design the Zvt
driver (:mod:`framework.naiad.emit.gemm_vme`: the driver half of the API, with
a block shape chosen per format pair from the design). When no header can be
emitted (no matrix unit, or no pair the emitter can type), the CLI writes one
that ``#error``-s on inclusion, so kernels that do not include it still build.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

#: C storage type per format (dwarf_util.h's naiad_<fmt>_t) and its size in bytes.
C_TYPES = {
    "fp64": ("naiad_fp64_t", 8),
    "fp32": ("naiad_fp32_t", 4),
    "fp16": ("naiad_fp16_t", 2),
    "bf16": ("naiad_bf16_t", 2),
    "int8": ("naiad_int8_t", 1),
    "int32": ("naiad_int32_t", 4),
}
_FLOAT = {"fp64", "fp32", "fp16", "bf16"}
MODES = ("ame", "ime", "vme")


@dataclass(frozen=True)
class Blocking:
    """Register blocking: ``bm x bn`` accumulators, double-buffered tile loads or not."""

    bm: int
    bn: int
    double_buffer: bool

    def tiles_needed(self):
        return (self.bm + self.bn) * (2 if self.double_buffer else 1)


def _issue_width(design):
    timing = getattr(design, "timing", None)
    scalar = getattr(timing, "scalar", None)
    return getattr(scalar, "issue_width", 1) or 1


@dataclass(frozen=True)
class GemmShape:
    """What the emitter needs from a design's ``matrix:`` section."""

    mode: str
    m: int
    n: int
    k: int
    tiles: int
    accs: int
    in_formats: tuple
    acc_formats: tuple
    name: str = "design"
    #: ``timing.scalar.issue_width``: at 2 the K loop walks persistent pointers advanced in the
    #: tile-load statements (``mx_tload_*_adv``), so the advance pairs with a matrix op.
    issue_width: int = 1

    @classmethod
    def from_design(cls, design):
        """From a ``Design`` (or a path to one); raises ValueError without a matrix unit."""
        if isinstance(design, (str, Path)):
            from framework.naiad.design import load_design

            design = load_design(design)
        mx = design.matrix
        if mx is None:
            raise ValueError(f"design '{design.name}' has no matrix unit")
        return cls(
            mode=mx.mode,
            m=mx.m,
            n=mx.n,
            k=mx.k,
            tiles=mx.tiles,
            accs=mx.accs,
            in_formats=tuple(mx.in_formats),
            acc_formats=tuple(mx.acc_formats),
            name=design.name,
            issue_width=_issue_width(design),
        )


def choose_blocking(tiles, accs, double_buffer=None):
    """The default blocking: the most accumulators in the squarest block the tiles can feed.

    Maximises ``bm * bn <= accs`` (each loaded tile is reused ``bn`` or ``bm``
    times), breaking ties by the fewest loads per step (``bm + bn``); then
    double-buffers when ``2 * (bm + bn) <= tiles``, unless ``double_buffer`` says.
    """
    best = None
    for bm in range(1, accs + 1):
        for bn in range(1, accs // bm + 1):
            if bm + bn > tiles:
                continue
            key = (bm * bn, -(bm + bn), -abs(bm - bn), -bm)  # then the wider block
            if best is None or key > best[0]:
                best = (key, bm, bn)
    if best is None:
        raise ValueError(
            f"a unit with {tiles} tiles and {accs} accumulators cannot hold one block"
        )
    _, bm, bn = best
    if double_buffer is None:
        double_buffer = 2 * (bm + bn) <= tiles
    blocking = Blocking(bm, bn, double_buffer)
    check_blocking(blocking, tiles, accs)
    return blocking


def check_blocking(blocking, tiles, accs):
    if blocking.bm < 1 or blocking.bn < 1:
        raise ValueError(f"blocking {blocking.bm}x{blocking.bn} must be at least 1x1")
    if blocking.bm * blocking.bn > accs:
        raise ValueError(
            f"blocking {blocking.bm}x{blocking.bn} needs {blocking.bm * blocking.bn} accumulators; the unit has {accs}"
        )
    if blocking.tiles_needed() > tiles:
        raise ValueError(
            f"blocking {blocking.bm}x{blocking.bn}{' double-buffered' if blocking.double_buffer else ''} "
            f"needs {blocking.tiles_needed()} tiles; the unit has {tiles}"
        )


def parse_block(text):
    """``"2x2"`` -> ``(2, 2)``."""
    try:
        bm, bn = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"blocking {text!r} is not of the form BMxBN") from None
    return bm, bn


def format_pairs(shape):
    """Every (in, acc) pair the unit implements that the emitter can type: float with float, int8 with int32."""
    pairs = []
    for i in shape.in_formats:
        for a in shape.acc_formats:
            if i not in C_TYPES or a not in C_TYPES:
                continue
            if (i in _FLOAT) != (a in _FLOAT):
                continue
            if C_TYPES[a][1] < C_TYPES[i][1]:
                continue  # an accumulator narrower than its inputs
            pairs.append((i, a))
    return pairs


# -- C emission ---------------------------------------------------------------


def _interleave(first, second):
    out = []
    for i in range(max(len(first), len(second))):
        if i < len(first):
            out.append(first[i])
        if i < len(second):
            out.append(second[i])
    return out


class _PairEmitter:
    """The functions for one (in, acc) pair."""

    def __init__(self, shape, blocking, in_fmt, acc_fmt):
        self.s, self.b = shape, blocking
        self.in_fmt, self.acc_fmt = in_fmt, acc_fmt
        self.p = f"naiad_gemm_{in_fmt}_{acc_fmt}"
        self.it, self.ies = C_TYPES[in_fmt]
        self.at, self.aes = C_TYPES[acc_fmt]
        self.IN = f"NAIAD_MFMT_{in_fmt.upper()}"
        self.ACC = f"NAIAD_MFMT_{acc_fmt.upper()}"

    def acc(self, i, j):
        return i * self.b.bn + j

    def a_tile(self, buf, i):
        return buf * (self.b.bm + self.b.bn) + i

    def b_tile(self, buf, j):
        return buf * (self.b.bm + self.b.bn) + self.b.bm + j

    def loads(self, buf, a, b):
        """Tile loads for one K step into buffer ``buf`` from pointers named ``a``, ``b``."""
        la = [
            f"mx_tload_a({self.a_tile(buf, i)}, {self.IN}, {a} + {i * self.s.m} * lda, sa);"
            for i in range(self.b.bm)
        ]
        lb = [
            f"mx_tload_b({self.b_tile(buf, j)}, {self.IN}, {b} + {j * self.s.n}, sb);"
            for j in range(self.b.bn)
        ]
        return _interleave(la, lb)

    def loads_adv(self, buf):
        """The K loop's loads at ``issue_width`` 2: one persistent pointer per tile (``pa<i>`` / ``pb<j>``),
        each advanced to the next K step by the load that reads it."""
        la = [
            f"mx_tload_a_adv({self.a_tile(buf, i)}, {self.IN}, pa{i}, sa, abump);"
            for i in range(self.b.bm)
        ]
        lb = [
            f"mx_tload_b_adv({self.b_tile(buf, j)}, {self.IN}, pb{j}, sb, bbump);"
            for j in range(self.b.bn)
        ]
        return _interleave(la, lb)

    def mmas(self, buf, guard=False):
        out = []
        for i in range(self.b.bm):
            for j in range(self.b.bn):
                mma = (
                    f"mx_mma({self.acc(i, j)}, {self.a_tile(buf, i)}, {self.b_tile(buf, j)}, "
                    f"{self.IN}, {self.ACC});"
                )
                out.append(f"if (rv[{i}] > 0 && cv[{j}] > 0) {mma}" if guard else mma)
        return out

    def _emit_kbody_dual(self, w):
        """The K loop for the dual-issue front end: one persistent pointer per tile, advanced to the next
        K step inside the tile-load statement that reads it (``mx_tload_*_adv``), so the advance pairs
        with the matrix op after the load and no address computation sits in front of a load."""
        s, b = self.s, self.b
        w(
            f"    const long abump = {s.k * self.ies}, bbump = {s.k} * sb; /* bytes to the next K step */"
        )
        # the pointers name step 1: step 0 was loaded by _kload0, and every load advances its pointer
        for i in range(b.bm):
            w(f"    const char *pa{i} = (const char *)(A + {i * s.m} * lda + {s.k});")
        for j in range(b.bn):
            w(f"    const char *pb{j} = (const char *)(B + {s.k} * ldb + {j * s.n});")
        if b.double_buffer:
            w(
                "    /* Double-buffered: buffer 0 holds even steps, buffer 1 odd ones; the loads of"
            )
            w("       the next step issue between the mma of this one. */")
            w("    long s = 0;")
            w("    for (; s + 2 <= steps; s += 2) {")
            for line in _interleave(self.loads_adv(1), self.mmas(0)):
                w(f"        {line}")
            w("        if (s + 2 < steps) {")
            for line in _interleave(self.loads_adv(0), self.mmas(1)):
                w(f"            {line}")
            w("        } else {")
            for line in self.mmas(1):
                w(f"            {line}")
            w("        }")
            w("    }")
            w("    if (s < steps) {")
            for line in self.mmas(0):
                w(f"        {line}")
            w("    }")
        else:
            w("    for (long s = 0; s < steps; s++) {")
            w("        if (s) {")
            for line in self.loads_adv(0):
                w(f"            {line}")
            w("        }")
            for line in self.mmas(0):
                w(f"        {line}")
            w("    }")

    def emit(self):
        s, b, p = self.s, self.b, self.p
        MB, NB = b.bm * s.m, b.bn * s.n
        it, at = self.it, self.at
        L = []
        w = L.append
        w(f"/* ---- {self.in_fmt} in, {self.acc_fmt} accumulate ---- */")
        w(
            f"#if defined(NAIAD_MATRIX_IN_{self.in_fmt.upper()}) && defined(NAIAD_MATRIX_ACC_{self.acc_fmt.upper()})"
        )
        w(f"#define NAIAD_GEMM_HAS_{self.in_fmt.upper()}_{self.acc_fmt.upper()} 1")
        w(f"typedef {it} {p}_in_t;")
        w(f"typedef {at} {p}_acc_t;")
        w("")
        # init
        w(
            f"/* Accumulators <- C[{MB} x {NB}] (row stride ldc) if accumulate, else zero. */"
        )
        w(
            f"NAIAD_GEMM_INLINE void {p}_init(const {at} *C, long ldc, int accumulate) {{"
        )
        w(f"    const long sc = ldc * (long)sizeof({at});")
        w("    if (accumulate) {")
        for i in range(b.bm):
            for j in range(b.bn):
                w(
                    f"        mx_aload({self.acc(i, j)}, {self.ACC}, C + {i * s.m} * ldc + {j * s.n}, sc);"
                )
        w("    } else {")
        for i in range(b.bm):
            for j in range(b.bn):
                w(f"        mx_azero({self.acc(i, j)});")
        w("    }")
        w("    (void)sc;")
        w("}")
        w("")
        # store, and store fused with the next block's first loads
        stores = [
            f"mx_astore({self.acc(i, j)}, {self.ACC}, C + {i * s.m} * ldc + {j * s.n}, sc);"
            for i in range(b.bm)
            for j in range(b.bn)
        ]
        w(f"/* C[{MB} x {NB}] <- accumulators. */")
        w(f"NAIAD_GEMM_INLINE void {p}_store({at} *C, long ldc) {{")
        w(f"    const long sc = ldc * (long)sizeof({at});")
        for line in stores:
            w(f"    {line}")
        w("}")
        w("")
        w(
            "/* _store, with the next block's step-0 tile loads (as _kload0) issued between the stores,"
        )
        w("   so the store and load paths overlap across the block boundary. */")
        w(
            f"NAIAD_GEMM_INLINE void {p}_store_kload0({at} *C, long ldc, const {it} *A, long lda, "
            f"const {it} *B, long ldb) {{"
        )
        w(f"    const long sc = ldc * (long)sizeof({at});")
        w(
            f"    const long sa = lda * (long)sizeof({it}), sb = ldb * (long)sizeof({it});"
        )
        for line in _interleave(stores, self.loads(0, "A", "B")):
            w(f"    {line}")
        w("}")
        w("")
        # kload0 / kbody / kloop
        w(
            "/* Step 0's tiles into buffer 0 (the first half of _kloop, split out so it can issue"
        )
        w("   before _init or under the previous block's stores). */")
        w(
            f"NAIAD_GEMM_INLINE void {p}_kload0(const {it} *A, long lda, const {it} *B, long ldb) {{"
        )
        w(
            f"    const long sa = lda * (long)sizeof({it}), sb = ldb * (long)sizeof({it});"
        )
        for line in self.loads(0, "A", "B"):
            w(f"    {line}")
        w("}")
        w("")
        w(
            f"/* Accumulators += A[{MB} x steps*{s.k}] . B[steps*{s.k} x {NB}], steps >= 1, with step 0"
        )
        w("   already in buffer 0 (_kload0); full tiles straight from A and B. */")
        w(
            f"NAIAD_GEMM_INLINE void {p}_kbody(long steps, const {it} *A, long lda, const {it} *B, long ldb) {{"
        )
        w(
            f"    const long sa = lda * (long)sizeof({it}), sb = ldb * (long)sizeof({it});"
        )
        if s.issue_width >= 2:
            self._emit_kbody_dual(w)
        else:
            w(f"    const long bstep = {s.k} * ldb;")
            w(f"    const {it} *a = A, *b = B;")
            if b.double_buffer:
                w(
                    "    /* Double-buffered: buffer 0 holds even steps, buffer 1 odd ones; the loads of"
                )
                w("       the next step issue between the mma of this one. */")
                w("    long s = 0;")
                w("    for (; s + 2 <= steps; s += 2) {")
                w(f"        const {it} *a1 = a + {s.k}, *b1 = b + bstep;")
                for line in _interleave(self.loads(1, "a1", "b1"), self.mmas(0)):
                    w(f"        {line}")
                w("        if (s + 2 < steps) {")
                w(f"            const {it} *a2 = a + {2 * s.k}, *b2 = b + 2 * bstep;")
                for line in _interleave(self.loads(0, "a2", "b2"), self.mmas(1)):
                    w(f"            {line}")
                w("        } else {")
                for line in self.mmas(1):
                    w(f"            {line}")
                w("        }")
                w(f"        a += {2 * s.k};")
                w("        b += 2 * bstep;")
                w("    }")
                w("    if (s < steps) {")
                for line in self.mmas(0):
                    w(f"        {line}")
                w("    }")
            else:
                w("    for (long s = 0; s < steps; s++) {")
                w("        if (s) {")
                for line in self.loads(0, "a", "b"):
                    w(f"            {line}")
                w("        }")
                for line in self.mmas(0):
                    w(f"        {line}")
                w(f"        a += {s.k};")
                w("        b += bstep;")
                w("    }")
        w("}")
        w("")
        w(f"/* Accumulators += A[{MB} x steps*{s.k}] . B[steps*{s.k} x {NB}]. */")
        w(
            f"NAIAD_GEMM_INLINE void {p}_kloop(long steps, const {it} *A, long lda, const {it} *B, long ldb) {{"
        )
        w("    if (steps > 0) {")
        w(f"        {p}_kload0(A, lda, B, ldb);")
        w(f"        {p}_kbody(steps, A, lda, B, ldb);")
        w("    }")
        w("}")
        w("")
        # ukernel
        w(
            f"/* C[{MB} x {NB}] = (accumulate ? C : 0) + A[{MB} x kc] . B[kc x {NB}]; kc a multiple of {s.k}. */"
        )
        w(
            f"NAIAD_GEMM_INLINE void {p}_ukernel(long kc, const {it} *A, long lda, const {it} *B, long ldb, "
            f"{at} *C, long ldc, int accumulate) {{"
        )
        w(f"    const long steps = kc / {s.k};")
        w("    if (steps > 0) {")
        w(
            f"        {p}_kload0(A, lda, B, ldb); /* before _init: the loads need no accumulator */"
        )
        w("    }")
        w(f"    {p}_init(C, ldc, accumulate);")
        w("    if (steps > 0) {")
        w(f"        {p}_kbody(steps, A, lda, B, ldb);")
        w("    }")
        w(f"    {p}_store(C, ldc);")
        w("}")
        w("")
        # staging + packed step
        w(
            "/* Zero-padded staging for edge tiles (one per tile, so no load reads a buffer being refilled). */"
        )
        w(f"static {it} {p}_stage_a[{b.bm}][{s.m * s.k}] NAIAD_GEMM_STAGE;")
        w(f"static {it} {p}_stage_b[{b.bn}][{s.k * s.n}] NAIAD_GEMM_STAGE;")
        w(f"static {at} {p}_stage_c[{MB * NB}] NAIAD_GEMM_STAGE;")
        w("")
        w(
            "/* One K step of at most one tile: mv valid rows, nv valid columns, kv valid K; partial tiles"
        )
        w(
            "   are packed into the staging tiles, empty ones skipped. Single-buffered (edges only). */"
        )
        w(
            f"static inline __attribute__((unused)) void {p}_kstep_packed(long mv, long nv, long kv, "
            f"const {it} *A, long lda, const {it} *B, long ldb) {{"
        )
        w(
            f"    const long sa = lda * (long)sizeof({it}), sb = ldb * (long)sizeof({it});"
        )
        w(f"    long rv[{b.bm}], cv[{b.bn}];")
        for i in range(b.bm):
            w(f"    rv[{i}] = mv - {i * s.m} < {s.m} ? mv - {i * s.m} : {s.m};")
            w(f"    if (rv[{i}] == {s.m} && kv == {s.k}) {{")
            w(
                f"        mx_tload_a({self.a_tile(0, i)}, {self.IN}, A + {i * s.m} * lda, sa);"
            )
            w(f"    }} else if (rv[{i}] > 0) {{")
            w(
                f"        naiad_pack_tile({p}_stage_a[{i}], {s.m}, {s.k}, A + {i * s.m} * lda, lda, rv[{i}], kv, "
                f"{self.ies});"
            )
            w(
                f"        mx_tload_a({self.a_tile(0, i)}, {self.IN}, {p}_stage_a[{i}], {s.k * self.ies});"
            )
            w("    }")
        for j in range(b.bn):
            w(f"    cv[{j}] = nv - {j * s.n} < {s.n} ? nv - {j * s.n} : {s.n};")
            w(f"    if (cv[{j}] == {s.n} && kv == {s.k}) {{")
            w(f"        mx_tload_b({self.b_tile(0, j)}, {self.IN}, B + {j * s.n}, sb);")
            w(f"    }} else if (cv[{j}] > 0) {{")
            w(
                f"        naiad_pack_tile({p}_stage_b[{j}], {s.k}, {s.n}, B + {j * s.n}, ldb, kv, cv[{j}], "
                f"{self.ies});"
            )
            w(
                f"        mx_tload_b({self.b_tile(0, j)}, {self.IN}, {p}_stage_b[{j}], {s.n * self.ies});"
            )
            w("    }")
        for line in self.mmas(0, guard=True):
            w(f"    {line}")
        w("}")
        w("")
        # driver
        w(
            "/* C = (accumulate ? C : 0) + A[m x k] . B[k x n], any m, n, k; edges padded (see the header). */"
        )
        w(
            f"static __attribute__((unused, noinline)) void {p}_run(long m, long n, long k, const {it} *A, long lda, "
            f"const {it} *B, long ldb, {at} *C, long ldc, int accumulate) {{"
        )
        w(
            f"    const long ksteps = k / {s.k}, kfull = ksteps * {s.k}, ktail = k - kfull;"
        )
        w("    int loaded = 0; /* this block's step 0 already sits in buffer 0 */")
        w(f"    for (long i0 = 0; i0 < m; i0 += {MB}) {{")
        w(f"        const long mv = m - i0 < {MB} ? m - i0 : {MB};")
        w(f"        for (long j0 = 0; j0 < n; j0 += {NB}) {{")
        w(f"            const long nv = n - j0 < {NB} ? n - j0 : {NB};")
        w(f"            const {it} *a = A + i0 * lda;")
        w(f"            const {it} *b = B + j0;")
        w(f"            {at} *c = C + i0 * ldc + j0;")
        w(f"            if (mv == {MB} && nv == {NB}) {{")
        w("                if (!loaded && ksteps) {")
        w(f"                    {p}_kload0(a, lda, b, ldb);")
        w("                }")
        w(f"                {p}_init(c, ldc, accumulate);")
        w("                if (ksteps) {")
        w(f"                    {p}_kbody(ksteps, a, lda, b, ldb);")
        w("                }")
        w("                if (ktail) {")
        w(
            f"                    {p}_kstep_packed({MB}, {NB}, ktail, a + kfull, lda, b + kfull * ldb, ldb);"
        )
        w("                }")
        w(
            "                /* The next block, in loop order: if it is full too, start its loads under these stores. */"
        )
        w(f"                long i1 = i0, j1 = j0 + {NB};")
        w("                if (j1 >= n) {")
        w("                    j1 = 0;")
        w(f"                    i1 += {MB};")
        w("                }")
        w(f"                loaded = ksteps && i1 + {MB} <= m && j1 + {NB} <= n;")
        w("                if (loaded) {")
        w(
            f"                    {p}_store_kload0(c, ldc, A + i1 * lda, lda, B + j1, ldb);"
        )
        w("                } else {")
        w(f"                    {p}_store(c, ldc);")
        w("                }")
        w("            } else {")
        w("                if (accumulate) {")
        w(
            f"                    naiad_pack_tile({p}_stage_c, {MB}, {NB}, c, ldc, mv, nv, {self.aes});"
        )
        w("                }")
        w(f"                {p}_init({p}_stage_c, {NB}, accumulate);")
        w(f"                for (long k0 = 0; k0 < k; k0 += {s.k}) {{")
        w(
            f"                    {p}_kstep_packed(mv, nv, k - k0 < {s.k} ? k - k0 : {s.k}, a + k0, lda, "
            "b + k0 * ldb, ldb);"
        )
        w("                }")
        w(f"                {p}_store({p}_stage_c, {NB});")
        w(
            f"                naiad_unpack_tile(c, ldc, {p}_stage_c, {NB}, mv, nv, {self.aes});"
        )
        w("            }")
        w("        }")
        w("    }")
        w("}")
        w("")
        w("/* C = A . B */")
        w(
            f"static inline __attribute__((unused)) void {p}(long m, long n, long k, const {it} *A, long lda, "
            f"const {it} *B, long ldb, {at} *C, long ldc) {{"
        )
        w(f"    {p}_run(m, n, k, A, lda, B, ldb, C, ldc, 0);")
        w("}")
        w("")
        w("/* C += A . B */")
        w(
            f"static inline __attribute__((unused)) void {p}_acc(long m, long n, long k, const {it} *A, long lda, "
            f"const {it} *B, long ldb, {at} *C, long ldc) {{"
        )
        w(f"    {p}_run(m, n, k, A, lda, B, ldb, C, ldc, 1);")
        w("}")
        w(f"#endif /* {self.in_fmt} x {self.acc_fmt} */")
        w("")
        return L


def _banner(shape, blocking, mode):
    return [
        "/*",
        f" * naiad_gemm.h: GEMM micro-kernel + blocked driver for design '{shape.name}'",
        f" * ({mode}, {shape.m}x{shape.n}x{shape.k} tiles, {shape.tiles} tiles, {shape.accs} accumulators).",
        " * Generated by python -m framework.naiad.emit.gemm; see that module for the API.",
        " * DO NOT EDIT: regenerate from the design file instead.",
        " */",
    ]


def _error_header(name, why):
    return "\n".join(
        [
            "/*",
            f" * naiad_gemm.h for design '{name}': no micro-kernel could be emitted.",
            " * Generated by python -m framework.naiad.emit.gemm. DO NOT EDIT.",
            " */",
            "#ifndef NAIAD_GEMM_H",
            "#define NAIAD_GEMM_H",
            f'#error "naiad_gemm.h: {why}"',
            "#endif",
            "",
        ]
    )


def emit_gemm_header(design, *, mode=None, block=None, double_buffer=None, pairs=None):
    """The ``naiad_gemm.h`` text for ``design`` (a ``Design``, path, or :class:`GemmShape`).

    ``mode`` overrides the design's (``ame``). An IME design gets
    :mod:`framework.naiad.emit.gemm_ime`'s header (same API), a VME design
    :mod:`framework.naiad.emit.gemm_vme`'s. ``block`` is ``(bm, bn)`` or ``"BMxBN"`` (default
    :func:`choose_blocking`); ``double_buffer`` forces the K-loop buffering;
    ``pairs`` restricts the (in, acc) format pairs (default :func:`format_pairs`).
    """
    if not isinstance(design, GemmShape):
        if isinstance(design, (str, Path)):
            from framework.naiad.design import load_design

            design = load_design(design)
        if (
            design.matrix is not None
            and design.matrix.mode == "ime"
            and mode in (None, "ime")
        ):
            from framework.naiad.emit.gemm_ime import emit_ime_gemm_header

            return emit_ime_gemm_header(design, block=block, pairs=pairs)
        if (
            design.matrix is not None
            and design.matrix.mode == "vme"
            and mode in (None, "vme")
        ):
            from framework.naiad.emit.gemm_vme import emit_vme_gemm_header

            return emit_vme_gemm_header(design, block=block, pairs=pairs)
        if design.matrix is not None and design.matrix.mode == "ztt":
            raise NotImplementedError(
                "no GEMM driver is emitted for a Ztt design yet (kernels/naiad/include/ztt.h has the instructions)"
            )
    shape = design if isinstance(design, GemmShape) else GemmShape.from_design(design)
    mode = mode or shape.mode
    if mode not in MODES:
        raise ValueError(f"unknown matrix mode {mode!r}; expected one of {MODES}")
    if mode == "ime":
        raise NotImplementedError(
            "the IME micro-kernel is emitted from an IME design (framework.naiad.emit.gemm_ime), not an AME shape"
        )
    if mode != "ame":
        raise NotImplementedError(
            f"the {mode.upper()} driver is emitted from a {mode.upper()} design "
            f"(framework.naiad.emit.gemm_{mode}), not an AME shape"
        )
    if block is None:
        blocking = choose_blocking(shape.tiles, shape.accs, double_buffer)
    else:
        bm, bn = parse_block(block) if isinstance(block, str) else block
        if double_buffer is None:
            double_buffer = 2 * (bm + bn) <= shape.tiles
        blocking = Blocking(bm, bn, double_buffer)
        check_blocking(blocking, shape.tiles, shape.accs)
    available = format_pairs(shape)
    if pairs is None:
        pairs = available
    else:
        pairs = [tuple(p.split(":")) if isinstance(p, str) else tuple(p) for p in pairs]
        missing = [p for p in pairs if p not in available]
        if missing:
            raise ValueError(
                f"format pair(s) {missing} not implemented by this unit; have {available}"
            )
    s, b = shape, blocking
    MB, NB = b.bm * s.m, b.bn * s.n
    L = _banner(s, b, mode)
    L += [
        "#ifndef NAIAD_GEMM_H",
        "#define NAIAD_GEMM_H",
        "",
        '#include "dwarf_util.h"',
        '#include "matrix.h"',
        "",
        "#ifndef NAIAD_MATRIX_AME",
        '#error "naiad_gemm.h was emitted for an AME unit, but naiad.h describes another mode: re-emit"',
        "#endif",
        f"#if NAIAD_MATRIX_M != {s.m} || NAIAD_MATRIX_N != {s.n} || NAIAD_MATRIX_K != {s.k} || "
        f"NAIAD_MATRIX_TILES < {b.tiles_needed()} || NAIAD_MATRIX_ACCS < {b.bm * b.bn}",
        '#error "naiad_gemm.h does not match this design\'s naiad.h (tile shape or register counts): re-emit"',
        "#endif",
        "",
        f"#define NAIAD_GEMM_BM {b.bm}  /* accumulator block: BM x BN tiles */",
        f"#define NAIAD_GEMM_BN {b.bn}",
        f"#define NAIAD_GEMM_MB {MB}  /* rows of C per micro-kernel call */",
        f"#define NAIAD_GEMM_NB {NB}  /* columns of C per micro-kernel call */",
        f"#define NAIAD_GEMM_KB {s.k}  /* kc granularity */",
        f"#define NAIAD_GEMM_DOUBLE_BUFFER {int(b.double_buffer)}",
        f"#define NAIAD_GEMM_TILES_USED {b.tiles_needed()}",
        "",
        "#define NAIAD_GEMM_INLINE static inline __attribute__((always_inline, unused))",
        "#define NAIAD_GEMM_STAGE __attribute__((aligned(64), unused))",
        "",
        "/* NAIAD_GEMM_FN(fp32, fp32) -> naiad_gemm_fp32_fp32 (append _acc, _ukernel, ... by hand). */",
        "#define NAIAD_GEMM_FN_(in, acc) naiad_gemm_##in##_##acc",
        "#define NAIAD_GEMM_FN(in, acc) NAIAD_GEMM_FN_(in, acc)",
        "",
    ]
    for in_fmt, acc_fmt in pairs:
        L += _PairEmitter(s, b, in_fmt, acc_fmt).emit()
    L += [
        "/* The unsuffixed names, for the pair named by NAIAD_GEMM_IN / NAIAD_GEMM_ACC. */",
        "#if defined(NAIAD_GEMM_IN) && defined(NAIAD_GEMM_ACC)",
        "#define NAIAD_GEMM_SEL_(in, acc, f) naiad_gemm_##in##_##acc##f",
        "#define NAIAD_GEMM_SEL(in, acc, f) NAIAD_GEMM_SEL_(in, acc, f)",
        "#define naiad_gemm NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, )",
        "#define naiad_gemm_acc NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _acc)",
        "#define naiad_gemm_ukernel NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _ukernel)",
        "#define naiad_gemm_init NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _init)",
        "#define naiad_gemm_kloop NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _kloop)",
        "#define naiad_gemm_store NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _store)",
        "typedef NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _in_t) naiad_gemm_in_t;",
        "typedef NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _acc_t) naiad_gemm_acc_t;",
        "#endif",
        "",
        "#endif /* NAIAD_GEMM_H */",
        "",
    ]
    return "\n".join(L)


def write_gemm_header(design, out_path, **kwargs):
    """Write :func:`emit_gemm_header` to ``out_path`` (only if it changed); returns the path."""
    out_path = Path(out_path)
    text = emit_gemm_header(design, **kwargs)
    if not out_path.exists() or out_path.read_text() != text:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m framework.naiad.emit.gemm",
        description="Emit naiad_gemm.h (GEMM micro-kernel + blocked driver) for a design.",
    )
    parser.add_argument("design", help="Naiad design YAML")
    parser.add_argument("-o", "--out", required=True, help="header to write")
    parser.add_argument(
        "--block", help="register blocking BMxBN (default: the most accumulators)"
    )
    parser.add_argument(
        "--pairs",
        help="comma-separated in:acc format pairs (default: all the unit has)",
    )
    parser.add_argument(
        "--single-buffer", action="store_true", help="do not double-buffer the K loop"
    )
    args = parser.parse_args(argv)
    from framework.naiad.design import DesignError, load_design

    try:
        design = load_design(args.design)
    except DesignError as e:
        print(f"naiad: {args.design}: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_gemm_header(
            design,
            out,
            block=args.block,
            pairs=args.pairs.split(",") if args.pairs else None,
            double_buffer=False if args.single_buffer else None,
        )
    except (NotImplementedError, ValueError) as e:
        # Only kernels that include the header should fail on such a design.
        out.write_text(_error_header(design.name, str(e).replace('"', "'")))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
