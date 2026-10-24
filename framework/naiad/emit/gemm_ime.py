"""The IME (Zvvm) mode of the GEMM emitter: ``naiad_gemm.h`` for an integrated matrix unit.

Called by :func:`framework.naiad.emit.gemm.emit_gemm_header` for a design in
``mode: ime``; the API is the AME emitter's (``naiad_gemm_<in>_<acc>``,
``..._acc``, ``..._ukernel``, the unsuffixed names under ``NAIAD_GEMM_IN`` /
``NAIAD_GEMM_ACC``), so a kernel written against it builds for either mode.

**Geometry per (in, acc) pair.** C at ``SEW`` = the accumulator's width, ``W =
SEW / in width``; lambda is the smallest the design supports at that SEW with
``EMUL_C <= 8`` (the largest C panel that one tile store covers, so the highest
compute intensity ``M / 2``): ``M = VLEN / (SEW * lambda)``. The micro-kernel
keeps ``BM x BN`` C groups resident; each K step (``K_eff = lambda * W * LMUL``
narrow elements) loads ``BM`` A tiles and ``BN`` B tiles and issues ``BM * BN``
multiply-accumulates, all in **one asm statement** that starts with its own
``vsetvl``, so no compiler-inserted ``vsetvli`` can land between them. The
blocking and LMUL are the ones that fit the 32 vector registers (and the asm
operand limit) with the most MACs per modelled step cycle under the design's
timing (or per instruction when it is untimed); ``block="BMxBN"`` forces one.

**Operands.** A row-major is already an A tile (``vmtl.v``, LD = lda / W); B
row-major needs ``B_tile[j][k] = B[k][j]``, which the transposing ``vmttl.v``
gives for ``W = 1`` straight from B (LD = ldb). For ``W > 1`` the W rows of
each storage element are first interleaved into a panel (RVV unit-stride
loads and a W-field segment store, once per (column block, K chunk), reused
by every row block), and the same transposing load reads the panel. **Edges
are padded**: a block cut by M or N, a K tail, or an unaligned A goes through
zero-padded staging buffers (scalar copies), so every product goes through
the unit. Needs Zvvmtls and Zvvmttls.
"""

import math

from framework.matrix import ime

#: Format -> (C storage type from dwarf_util.h, bytes); only these pairs are emitted.
C_TYPES = {
    "fp64": ("naiad_fp64_t", 8),
    "fp32": ("naiad_fp32_t", 4),
    "fp16": ("naiad_fp16_t", 2),
    "bf16": ("naiad_bf16_t", 2),
    "int8": ("naiad_int8_t", 1),
    "int32": ("naiad_int32_t", 4),
}
#: Narrow elements per K chunk staged at once (a multiple of every K_eff).
KC = 256
_ASM_OPERANDS = 30


def _lmul_options():
    return (8, 4, 2, 1)


class PairPlan:
    """Geometry, blocking and encodings for one (in, acc) pair."""

    def __init__(self, design, in_fmt, acc_fmt, block=None):
        spec = design.matrix
        self.in_fmt, self.acc_fmt = in_fmt, acc_fmt
        self.vlen = spec.vlen
        self.in_bytes = C_TYPES[in_fmt][1]
        self.sew = 8 * C_TYPES[acc_fmt][1]
        self.w = self.sew // (8 * self.in_bytes)
        kind = "int" if in_fmt.startswith("int") else "fp"
        self.insn = next(
            i for i in ime.INSTRUCTIONS.values() if i.kind == kind and i.w == self.w
        )
        self.afa = self.afb = 1 if in_fmt == "bf16" else 0
        self.altfmt = 1 if acc_fmt == "bf16" else 0
        row = ime.encoding_row(kind, self.w, self.sew, self.afa, self.afb, self.altfmt)
        if row is None or row.fmt_a != in_fmt or row.fmt_c != acc_fmt:
            raise ValueError(f"no Zvvm encoding-map row for {in_fmt} -> {acc_fmt}")
        self.extensions = row.requires
        lams = [
            lam
            for lam in spec.lambdas.get(self.sew, ())
            if self.vlen // (self.sew * lam * lam) <= 8
        ]
        if not lams:
            raise ValueError(
                f"{in_fmt} -> {acc_fmt}: no supported lambda at SEW {self.sew} has EMUL_C <= 8"
            )
        self.lam = min(lams)
        self.m = self.vlen // (self.sew * self.lam)
        self.emul_c = self.m // self.lam
        self.timing = getattr(getattr(design, "timing", None), "matrix", None)
        self.vtiming = getattr(getattr(design, "timing", None), "vector", None)
        self.load_latency = self._load_latency(design)
        self.bm, self.bn, self.lmul, self.db = self._choose(block)
        self.ke = self.lam * self.w * self.lmul
        self.mb, self.nb = self.bm * self.m, self.bn * self.m

    # -- blocking -------------------------------------------------------------------------

    def _load_latency(self, design):
        """A tile load's latency into the program memory (startup + the region's), for the blocking model."""
        timing = getattr(design, "timing", None)
        if timing is None or timing.vector is None:
            return 0
        prog = design.program_memory
        return timing.vector.mem_startup + timing.region(prog.name, prog.type).latency

    def _fits(self, bm, bn, lmul, db):
        regs = bm * bn * self.emul_c + (bm + bn) * lmul * (2 if db else 1)
        operands = bm * bn + (3 if db else 2) * (bm + bn) + 4
        return regs <= 32 and operands <= _ASM_OPERANDS

    def _step_cycles(self, bm, bn, lmul, db):
        """Modelled cycles of one K step: the LSU's loads against the mma unit's work.

        Software-pipelined (``db``), the next step's loads overlap this step's
        mma; otherwise each step also waits out its own loads' latency.
        """
        from framework.matrix.ime_timing import IMETimingSpec

        geo = ime.Geometry(
            self.sew,
            self.w,
            lmul,
            self.lam,
            self.m,
            self.m,
            self.lam * self.w * lmul,
            self.emul_c,
        )
        if self.timing is None:  # untimed: one instruction per cycle
            return bm + bn + bm * bn
        spec = (
            self.timing if isinstance(self.timing, IMETimingSpec) else IMETimingSpec()
        )
        dp = self.vtiming.datapath_bits if self.vtiming is not None else self.vlen
        bw = (
            self.vtiming.mem_bytes_per_cycle
            if self.vtiming is not None
            else self.vlen / 8
        )
        occ, _, _, t_write = spec.mma_occupancy(geo, self.in_fmt, dp)
        ls = self.lam * lmul
        tile_bytes = self.m * ls * self.sew // 8
        seg = spec.tile_segments_per_cycle
        a = max(1, math.ceil(tile_bytes / bw), math.ceil(self.m / seg))
        b = max(1, math.ceil(tile_bytes / bw), math.ceil(ls / seg))
        lsu, mma = bm * a + bn * b, bm * bn * occ
        if db:
            # The tile loads write the VRF through the port the mma writes C through, and the
            # timing model gives each class the port in turn: loads and C write-backs serialise.
            return max(mma, lsu + bm * bn * t_write) + 1
        return lsu + self.load_latency + mma

    def _choose(self, block):
        best = None
        cands = []
        for lmul in _lmul_options():
            for db in (True, False):
                shapes = (
                    [block]
                    if block is not None
                    else [(bm, bn) for bm in range(1, 9) for bn in range(1, 9)]
                )
                cands += [
                    (bm, bn, lmul, db)
                    for bm, bn in shapes
                    if self._fits(bm, bn, lmul, db)
                ]
        if not cands:
            raise ValueError(
                f"{self.in_fmt} -> {self.acc_fmt}: blocking {block} does not fit 32 vector registers"
                if block
                else f"{self.in_fmt} -> {self.acc_fmt}: no blocking fits 32 vector registers"
            )
        for bm, bn, lmul, db in cands:
            macs = bm * bn * self.m * self.m * self.lam * self.w * lmul
            rate = macs / self._step_cycles(bm, bn, lmul, db)
            key = (rate, db, lmul, bm * bn, -abs(bm - bn), bn)
            if best is None or key > best[0]:
                best = (key, bm, bn, lmul, db)
        return best[1:]

    # -- C emission ----------------------------------------------------------------------------

    @property
    def prefix(self):
        return f"naiad_gemm_{self.in_fmt}_{self.acc_fmt}"

    def vt(self, lmul):
        return f"IME_VTYPE({self.sew}, {lmul}, {self.lam}, {self.altfmt}, {self.afa}, {self.afb})"

    def emit(self):
        p, s = self.prefix, self
        in_t, acc_t = C_TYPES[s.in_fmt][0], C_TYPES[s.acc_fmt][0]
        st = f"uint{s.sew}_t"  # storage element of a tile load
        vab = f"vuint{s.sew}m{s.lmul}_t"
        vc = f"vuint{s.sew}m{s.emul_c}_t"
        eb = s.sew // 8
        mma = f"NAIAD_IME_{s.insn.name.upper().replace('.', '_')}"
        L = [
            f"/* -- {s.in_fmt} -> {s.acc_fmt}: {s.insn.name}, SEW {s.sew}, W {s.w}, lambda {s.lam}: "
            f"{s.m}x{s.m} C tiles (EMUL_C {s.emul_c}), LMUL {s.lmul}, K_eff {s.ke}, {s.bm}x{s.bn} block"
            f"{', software-pipelined' if s.db else ''} -- */",
            "",
        ]
        for e in s.extensions:
            L += [
                f"#ifndef NAIAD_IME_EXT_{e.upper()}",
                f'#error "naiad_gemm.h: {p} needs {e}: re-emit"',
                "#endif",
            ]
        L += [
            f"#if !(NAIAD_IME_LAMBDAS_E{s.sew} & (1 << {ime.encode_lambda(s.lam)}))",
            f'#error "naiad_gemm.h: {p} needs lambda {s.lam} at SEW {s.sew}: re-emit"',
            "#endif",
            f"typedef {in_t} {p}_in_t;",
            f"typedef {acc_t} {p}_acc_t;",
            f"#define {p.upper()}_MB {s.mb}",
            f"#define {p.upper()}_NB {s.nb}",
            f"#define {p.upper()}_KE {s.ke}",
            f"static {in_t} {p}_stage_a[{s.mb * KC}] NAIAD_GEMM_STAGE;",
            f"static {in_t} {p}_stage_b[{KC * s.nb}] NAIAD_GEMM_STAGE; /* a B panel: KC/W x NB storage elements */",
            f"static {acc_t} {p}_stage_c[{s.mb * s.nb}] NAIAD_GEMM_STAGE;",
            "",
            *self._ukernel(p, st, vab, vc, eb, mma),
            *self._pack_b(p, in_t, st),
            *self._driver(p, in_t, acc_t),
        ]
        return L

    def _asm(self, mma, load=None, step=None, use=None, vsetvl=False, indent="    "):
        """One asm statement: optionally vsetvl, the tile loads of ``step`` into set ``load``,
        then every multiply-accumulate on set ``use`` (sets are the suffixes ``x`` / ``y``)."""
        s = self
        cs = [(i, j) for i in range(s.bm) for j in range(s.bn)]
        ls = s.lam * s.lmul
        L = []
        body, outs, ins = [], [f'[c{i}{j}] "+vr"(c{i}{j})' for i, j in cs], []
        if vsetvl:
            body.append('"vsetvl %[t], zero, %[vt]\\n\\t"')
            outs.append('[t] "=&r"(t)')
            ins.append('[vt] "r"(vt_k)')
        if load is not None:
            vl_ = 'NAIAD_STR(NAIAD_IME_VMTL_V_OPCODE) ", 7, " NAIAD_STR(NAIAD_IME_VMTL_V_FUNCT7)'
            tl_ = 'NAIAD_STR(NAIAD_IME_VMTTL_V_OPCODE) ", 7, " NAIAD_STR(NAIAD_IME_VMTTL_V_FUNCT7)'
            body += [
                f'".insn r " {vl_} ", %[a{i}{load}], %[pa{i}], %[lda]\\n\\t"'
                for i in range(s.bm)
            ]
            body += [
                f'".insn r " {tl_} ", %[b{j}{load}], %[pb{j}], %[ldb]\\n\\t"'
                for j in range(s.bn)
            ]
            outs += [f'[a{i}{load}] "=&vr"(a{i}{load})' for i in range(s.bm)]
            outs += [f'[b{j}{load}] "=&vr"(b{j}{load})' for j in range(s.bn)]
            ins += [
                f'[pa{i}] "r"(ap + {i * s.m}L * lda_s + ({step}) * {ls})'
                for i in range(s.bm)
            ]
            ins += [
                f'[pb{j}] "r"(bp + ({step}) * {ls} * ldb_s + {j * s.m})'
                for j in range(s.bn)
            ]
            ins += ['[lda] "r"(lda_s)', '[ldb] "r"(ldb_s)']
        if use is not None:
            body += [
                f'".insn r " NAIAD_STR({mma}_OPCODE) ", " NAIAD_STR({mma}_FUNCT3) ", " NAIAD_STR({mma}_FUNCT7) '
                f'", %[c{i}{j}], %[a{i}{use}], %[b{j}{use}]\\n\\t"'
                for i, j in cs
            ]
            if use != load:
                ins += [f'[a{i}{use}] "vr"(a{i}{use})' for i in range(s.bm)]
                ins += [f'[b{j}{use}] "vr"(b{j}{use})' for j in range(s.bn)]
        L.append(f"{indent}__asm__ volatile(")
        L += [f"{indent}    {line}" for line in body]
        L += [
            f"{indent}    : {', '.join(outs)}",
            f"{indent}    : {', '.join(ins) if ins else ''}",
            f'{indent}    : "memory");',
        ]
        return L

    def _ukernel(self, p, st, vab, vc, eb, mma):
        s = self
        cs = [(i, j) for i in range(s.bm) for j in range(s.bn)]
        sets = ("x", "y") if s.db else ("x",)
        tiles = [f"a{i}{z}" for z in sets for i in range(s.bm)] + [
            f"b{j}{z}" for z in sets for j in range(s.bn)
        ]
        L = [
            "/* C[MB x NB] (= or +=) A . B for one block; kc a multiple of K_eff. a / b are storage-element",
            " * pointers: A row-major (lda_s storage elements a row), B as a K-major row-major source of",
            " * storage elements (ldb_s a row; B itself when W = 1, an interleaved panel otherwise).",
            " * Software-pipelined: step s + 1 is loaded before step s multiplies (two tile sets). */"
            if s.db
            else " * One step at a time. */",
            f"NAIAD_GEMM_INLINE void {p}_ukernel(long kc, const void *a, long lda_s, const void *b, long ldb_s,",
            "                                  void *c, long ldc, int accumulate) {",
            f"    const unsigned long vt_k = {s.vt(s.lmul)}, vt_c = {s.vt(s.emul_c)};",
            f"    {vc} {', '.join(f'c{i}{j}' for i, j in cs)};",
            f"    {vab} {', '.join(tiles)};",
            f"    {st} *cp = ({st} *)c;",
            f"    const {st} *ap = (const {st} *)a, *bp = (const {st} *)b;",
            f"    const long n = kc / {s.ke};",
            "    unsigned long t;",
            "    if (accumulate) {",
            "        ime_vsetvlmax(vt_c);",
        ]
        L += [
            f"        ime_vmtl(c{i}{j}, cp + {i * s.m}L * ldc + {j * s.m}, ldc);"
            for i, j in cs
        ]
        L += ["    } else {"]
        L += [f"        ime_vzero(c{i}{j}, vt_c);" for i, j in cs]
        L += ["    }"]
        if s.db:
            L += self._asm(mma, load="x", step="0", vsetvl=True)
            L += ["    long s = 0;", "    for (; s + 2 < n; s += 2) {"]
            L += self._asm(mma, load="y", step="s + 1", use="x", indent="        ")
            L += self._asm(mma, load="x", step="s + 2", use="y", indent="        ")
            L += ["    }", "    if (s + 1 < n) {"]
            L += self._asm(mma, load="y", step="s + 1", use="x", indent="        ")
            L += self._asm(mma, use="y", indent="        ")
            L += ["    } else {"]
            L += self._asm(mma, use="x", indent="        ")
            L += ["    }"]
        else:
            L += ["    for (long s = 0; s < n; s++) {"]
            L += self._asm(
                mma, load="x", step="s", use="x", vsetvl=True, indent="        "
            )
            L += ["    }"]
        L += ["    (void)t;", "    ime_vsetvlmax(vt_c);"]
        L += [
            f"    ime_vmts(c{i}{j}, cp + {i * s.m}L * ldc + {j * s.m}, ldc);"
            for i, j in cs
        ]
        L += ["}", ""]
        return L

    def _pack_b(self, p, in_t, st):
        """B[kc x nb] (rows k0.., ldb) -> the zero-padded panel ``stage_b``: kcp/W rows of NB storage elements."""
        s = self
        L = [
            f"/* B[kc x nb] (row stride ldb) -> {p}_stage_b: ceil(kc / K_eff) * K_eff / W rows of NB storage",
            " * elements, row q holding B rows W q .. W q + W - 1 interleaved (zero past kc and nb). */",
            f"NAIAD_GEMM_INLINE void {p}_pack_b(const {in_t} *B, long ldb, long kc, long nb) {{",
            f"    long kcp = (kc + {s.ke - 1}) / {s.ke} * {s.ke};",
            f"    {in_t} *d = {p}_stage_b;",
        ]
        if s.w == 1:
            L += [
                "    for (long k = 0; k < kcp; k++) {",
                "        if (k < kc) {",
                f"            naiad_du_copy(d + k * {s.nb}, B + k * ldb, nb, {s.in_bytes});",
                f"            if (nb < {s.nb}) naiad_zero(d + k * {s.nb} + nb, ({s.nb} - nb) * {s.in_bytes});",
                "        } else {",
                f"            naiad_zero(d + k * {s.nb}, {s.nb * s.in_bytes});",
                "        }",
                "    }",
                "}",
                "",
            ]
            return L
        bits = 8 * s.in_bytes
        lm = 2 if s.in_bytes == 2 else 1
        vt = f"vuint{bits}m{lm}_t"
        sfx = f"u{bits}m{lm}"
        L += [
            f"    naiad_zero(d, kcp / {s.w} * {s.nb} * {s.w} * {s.in_bytes});",
            f"    for (long q = 0; q < kcp / {s.w}; q++) {{",
            f"        uint{bits}_t *row = (uint{bits}_t *)(d + q * {s.nb * s.w});",
            "        for (long j = 0; j < nb;) {",
            f"            size_t vl = __riscv_vsetvl_e{bits}m{lm}(nb - j);",
        ]
        for t in range(s.w):
            L += [
                f"            long k{t} = {s.w} * q + {t};",
                f"            {vt} v{t} = k{t} < kc ? __riscv_vle{bits}_v_{sfx}((const uint{bits}_t *)(B + k{t} * ldb + j), vl)",
                f"                                : __riscv_vmv_v_x_{sfx}(0, vl);",
            ]
        fields = ", ".join(f"v{t}" for t in range(s.w))
        L += [
            f"            __riscv_vsseg{s.w}e{bits}_v_{sfx}x{s.w}(row + {s.w} * j, __riscv_vcreate_v_{sfx}x{s.w}({fields}), vl);",
            "            j += vl;",
            "        }",
            "    }",
            "}",
            "",
        ]
        return L

    def _driver(self, p, in_t, acc_t):
        s = self
        es, ec = s.in_bytes, C_TYPES[s.acc_fmt][1]
        direct_b = "nb == NB && kc == kcp" if s.w == 1 else "0"
        return [
            f"NAIAD_GEMM_INLINE void {p}_impl(long m, long n, long k, const {in_t} *A, long lda, const {in_t} *B,",
            f"                              long ldb, {acc_t} *C, long ldc, int acc) {{",
            f"    enum {{ MB = {s.mb}, NB = {s.nb}, KE = {s.ke}, W = {s.w}, KC = {KC} }};",
            "    for (long j0 = 0; j0 < n; j0 += NB) {",
            "        long nb = n - j0 < NB ? n - j0 : NB;",
            "        for (long k0 = 0; k0 < k; k0 += KC) {",
            "            long kc = k - k0 < KC ? k - k0 : KC;",
            "            long kcp = (kc + KE - 1) / KE * KE;",
            "            const void *bsrc;",
            "            long ldbs;",
            f"            if ({direct_b}) {{ /* the transposing load reads B itself */",
            "                bsrc = B + k0 * ldb + j0;",
            "                ldbs = ldb;",
            "            } else {",
            f"                {p}_pack_b(B + k0 * ldb + j0, ldb, kc, nb);",
            f"                bsrc = {p}_stage_b;",
            "                ldbs = NB;",
            "            }",
            "            for (long i0 = 0; i0 < m; i0 += MB) {",
            "                long mb = m - i0 < MB ? m - i0 : MB;",
            "                int accumulate = acc || k0 > 0;",
            f"                const {in_t} *asrc = A + i0 * lda + k0;",
            "                long ldas = lda / W;",
            f"                if (mb < MB || kc < kcp || lda % W || ((uintptr_t)asrc & ({s.sew // 8} - 1))) {{",
            f"                    naiad_pack_tile({p}_stage_a, MB, kcp, asrc, lda, mb, kc, {es});",
            f"                    asrc = {p}_stage_a;",
            "                    ldas = kcp / W;",
            "                }",
            "                if (mb == MB && nb == NB) {",
            f"                    {p}_ukernel(kcp, asrc, ldas, bsrc, ldbs, C + i0 * ldc + j0, ldc, accumulate);",
            "                } else {",
            "                    if (accumulate) {",
            f"                        naiad_pack_tile({p}_stage_c, MB, NB, C + i0 * ldc + j0, ldc, mb, nb, {ec});",
            "                    }",
            f"                    {p}_ukernel(kcp, asrc, ldas, bsrc, ldbs, {p}_stage_c, NB, accumulate);",
            f"                    naiad_unpack_tile(C + i0 * ldc + j0, ldc, {p}_stage_c, NB, mb, nb, {ec});",
            "                }",
            "            }",
            "        }",
            "    }",
            "}",
            "",
            "/* C = A . B (m x k times k x n; strides in elements). */",
            f"NAIAD_GEMM_INLINE void {p}(long m, long n, long k, const {in_t} *A, long lda, const {in_t} *B,",
            f"                         long ldb, {acc_t} *C, long ldc) {{",
            f"    {p}_impl(m, n, k, A, lda, B, ldb, C, ldc, 0);",
            "}",
            "",
            "/* C += A . B. */",
            f"NAIAD_GEMM_INLINE void {p}_acc(long m, long n, long k, const {in_t} *A, long lda, const {in_t} *B,",
            f"                             long ldb, {acc_t} *C, long ldc) {{",
            f"    {p}_impl(m, n, k, A, lda, B, ldb, C, ldc, 1);",
            "}",
            "",
        ]


def format_pairs(design):
    """The (in, acc) pairs the design's extensions give that the emitter can type (same-format inputs)."""
    pairs = []
    for i, a in design.matrix.format_pairs:
        if i in C_TYPES and a in C_TYPES and (i, a) not in pairs:
            pairs.append((i, a))
    return pairs


def emit_ime_gemm_header(design, *, block=None, pairs=None):
    """The ``naiad_gemm.h`` text for an IME ``design`` (see the module docstring)."""
    from framework.naiad.emit.gemm import parse_block

    spec = design.matrix
    for need in ("Zvvmtls", "Zvvmttls"):
        if need not in spec.extensions:
            raise ValueError(f"the IME GEMM micro-kernel needs {need} (tile loads)")
    if isinstance(block, str):
        block = parse_block(block)
    available = format_pairs(design)
    if pairs is None:
        pairs = available
    else:
        pairs = [tuple(p.split(":")) if isinstance(p, str) else tuple(p) for p in pairs]
        missing = [p for p in pairs if p not in available]
        if missing:
            raise ValueError(
                f"format pair(s) {missing} not implemented by this unit; have {available}"
            )
    plans = []
    for i, a in pairs:
        try:
            plans.append(PairPlan(design, i, a, block))
        except ValueError:
            if len(pairs) == 1:
                raise
    if not plans:
        raise ValueError("no format pair of this IME unit can be emitted")
    L = [
        "/*",
        f" * naiad_gemm.h: GEMM micro-kernel + blocked driver for design '{design.name}'",
        f" * (ime, VLEN {spec.vlen}; Zvvm @ 6effbcae).",
        " * Generated by python -m framework.naiad.emit.gemm; see framework/naiad/emit/gemm_ime.py.",
        " * DO NOT EDIT: regenerate from the design file instead.",
        " */",
        "#ifndef NAIAD_GEMM_H",
        "#define NAIAD_GEMM_H",
        "",
        '#include "dwarf_util.h"',
        '#include "ime.h"',
        "",
        "#ifndef NAIAD_MATRIX_IME",
        '#error "naiad_gemm.h was emitted for an IME unit, but naiad.h describes another mode: re-emit"',
        "#endif",
        f"#if NAIAD_VLEN != {spec.vlen}",
        '#error "naiad_gemm.h does not match this design\'s VLEN: re-emit"',
        "#endif",
        "",
        "#define NAIAD_STR_(x) #x",
        "#define NAIAD_STR(x) NAIAD_STR_(x)",
        "#define NAIAD_GEMM_INLINE static inline __attribute__((always_inline, unused))",
        "#define NAIAD_GEMM_STAGE __attribute__((aligned(64), unused))",
        "#define NAIAD_GEMM_FN_(in, acc) naiad_gemm_##in##_##acc",
        "#define NAIAD_GEMM_FN(in, acc) NAIAD_GEMM_FN_(in, acc)",
        "",
    ]
    for plan in plans:
        L += plan.emit()
    L += [
        "/* The unsuffixed names, for the pair named by NAIAD_GEMM_IN / NAIAD_GEMM_ACC. */",
        "#if defined(NAIAD_GEMM_IN) && defined(NAIAD_GEMM_ACC)",
        "#define NAIAD_GEMM_SEL_(in, acc, f) naiad_gemm_##in##_##acc##f",
        "#define NAIAD_GEMM_SEL(in, acc, f) NAIAD_GEMM_SEL_(in, acc, f)",
        "#define naiad_gemm NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, )",
        "#define naiad_gemm_acc NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _acc)",
        "#define naiad_gemm_ukernel NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _ukernel)",
        "typedef NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _in_t) naiad_gemm_in_t;",
        "typedef NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _acc_t) naiad_gemm_acc_t;",
        "#endif",
        "",
        "#endif /* NAIAD_GEMM_H */",
        "",
    ]
    return "\n".join(L)
