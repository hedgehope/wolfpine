"""The RVV architectural state: one numpy vector register file plus the vector CSRs.

VRF accessor API (what the IME matrix mode and tests use)
---------------------------------------------------------

The register file is :attr:`VectorState.vrf`, **one** preallocated
``numpy.uint8`` array of shape ``(32, VLENB)``: row ``v`` is vector register
``v``, bytes in memory order (element ``i`` of SEW ``s`` occupies bytes
``i*s/8 .. (i+1)*s/8 - 1``, little-endian). Every other view of the state is a
numpy *view* into that array, never a copy and never a ``Register`` object, so
a write through any view is a write to the register file.

``vreg_group(base, lmul, sew_or_dtype) -> numpy view``
    The register group ``v[base] .. v[base + max(1, lmul) - 1]`` as one flat
    1-D array of the element type. ``lmul`` is 1, 2, 4, 8 or a fraction
    (0.5, 0.25, 0.125, or ``Fraction``); a fractional group is one register.
    ``sew_or_dtype`` is an element width in bits (8/16/32/64, giving unsigned
    integers), a numpy dtype (``np.float32``, ``np.int16``, ml_dtypes'
    ``bfloat16``, ...), or a format name (``"fp32"``, ``"bf16"``, ``"int8"``,
    ``"u16"``). ``base`` must be a multiple of the group size. The view covers
    the whole group (VLMAX elements at LMUL >= 1); slice it with ``[:vl]``
    for the body.
``vreg_bytes(base, nregs=1) -> numpy uint8 view``
    ``nregs`` consecutive registers as one flat byte array.
``mask(reg=0, n=None) -> bool array``
    The first ``n`` bits (default VLEN) of mask register ``reg``, as a fresh
    boolean array (bit ``i`` of the mask is bit ``i % 8`` of byte ``i // 8``).
``set_mask(reg, bits, start=0)``
    Write a boolean array into mask register ``reg`` from bit ``start``.

The configuration a vector instruction sees is :attr:`sew`, :attr:`lmul_log2`
(-3..3), :attr:`vl`, :attr:`vstart`, :attr:`vlmax`, :attr:`vta`, :attr:`vma`
and :attr:`vill`, all derived from :attr:`vtype` by :meth:`set_vtype`.

CSRs
----

``vstart`` (0x008), ``vxsat`` (0x009), ``vxrm`` (0x00A), ``vcsr`` (0x00F),
``vl`` (0xC20), ``vtype`` (0xC21) and ``vlenb`` (0xC22), with the spec's
semantics: ``vl``/``vtype``/``vlenb`` are read-only (only ``vset{i}vl{i}``
change the first two), ``vcsr`` is ``{vxrm, vxsat}``, and ``vstart`` keeps
``log2(VLEN)`` bits as Spike does. ``CSRFile64`` forwards these addresses
here once :func:`~framework.pe.rv.isa.v.attach` has bound a state.

``vtype`` has ``vill`` (bit 63) set, and every other bit clear, after reset
and whenever ``vset*`` asks for an unsupported configuration: an LMUL outside
1/8..8, a SEW above ``ELEN * min(1, LMUL)`` (Spike's rule), any reserved bit,
or ``vsetvli x0, x0`` changing VLMAX.

Extensions that add ``vtype`` fields (Zvvm's ``lambda``/``bs``/``altfmt_A/B``
at bits 62:57, Zvfbfa's ``altfmt`` at bit 8) do not edit this decoder: they
call :meth:`VectorState.install_vtype_extension` with the bits they define
and, optionally, a write-rule hook. Every field is read through
:meth:`VectorState.vtype_fields`.
"""

from fractions import Fraction

import numpy as np

CSR_VSTART = 0x008
CSR_VXSAT = 0x009
CSR_VXRM = 0x00A
CSR_VCSR = 0x00F
CSR_VL = 0xC20
CSR_VTYPE = 0xC21
CSR_VLENB = 0xC22

VECTOR_CSRS = {
    CSR_VSTART: "vstart",
    CSR_VXSAT: "vxsat",
    CSR_VXRM: "vxrm",
    CSR_VCSR: "vcsr",
    CSR_VL: "vl",
    CSR_VTYPE: "vtype",
    CSR_VLENB: "vlenb",
}

VILL = 1 << 63

#: Element width (bits) -> little-endian unsigned / signed / float dtypes.
UINT = {
    8: np.dtype("<u1"),
    16: np.dtype("<u2"),
    32: np.dtype("<u4"),
    64: np.dtype("<u8"),
}
SINT = {
    8: np.dtype("<i1"),
    16: np.dtype("<i2"),
    32: np.dtype("<i4"),
    64: np.dtype("<i8"),
}
FLOAT = {16: np.dtype("<f2"), 32: np.dtype("<f4"), 64: np.dtype("<f8")}

_NAMED = {
    "fp64": FLOAT[64],
    "fp32": FLOAT[32],
    "fp16": FLOAT[16],
    **{f"int{w}": SINT[w] for w in SINT},
    **{f"i{w}": SINT[w] for w in SINT},
    **{f"uint{w}": UINT[w] for w in UINT},
    **{f"u{w}": UINT[w] for w in UINT},
}


class VectorStateError(RuntimeError):
    """A vector instruction ran on a core with no vector state attached, or the accessor API was misused."""


def _dtype(sew_or_dtype):
    if isinstance(sew_or_dtype, (int, np.integer)) and not isinstance(
        sew_or_dtype, bool
    ):
        if int(sew_or_dtype) not in UINT:
            raise VectorStateError(
                f"no element width {sew_or_dtype}; expected 8, 16, 32 or 64"
            )
        return UINT[int(sew_or_dtype)]
    if isinstance(sew_or_dtype, str):
        name = sew_or_dtype.lower()
        if name in _NAMED:
            return _NAMED[name]
        if name in ("bf16", "bfloat16"):
            import ml_dtypes

            return np.dtype(ml_dtypes.bfloat16)
        raise VectorStateError(f"unknown element format {sew_or_dtype!r}")
    return np.dtype(sew_or_dtype)


def _lmul_regs(lmul):
    lmul = Fraction(lmul)
    if lmul not in (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), 1, 2, 4, 8):
        raise VectorStateError(f"LMUL {lmul} is not one of 1/8, 1/4, 1/2, 1, 2, 4, 8")
    return max(1, int(lmul))


class VectorState:
    """One hart's vector unit: the register file, the vector CSRs and the unit's counters.

    ``extensions`` is the design's ISA list (only the vector names matter):
    ``zvfh`` enables SEW=16 FP arithmetic, ``zvfbfmin`` the bf16 converts and
    ``zvfbfwma`` ``vfwmaccbf16``.
    """

    #: The counter hub names these events ``hart<N>.vector.<key>``.
    counter_prefix = "vector"

    def __init__(self, vlen=128, elen=64, extensions=("v",)):
        if vlen < 64 or vlen & (vlen - 1) or vlen > 65536:
            raise VectorStateError(f"VLEN {vlen} is not a power of two in 64-65536")
        if elen not in (32, 64) or elen > vlen:
            raise VectorStateError(f"ELEN {elen} must be 32 or 64 and at most VLEN")
        self.vlen = vlen
        self.elen = elen
        self.vlenb = vlen // 8
        names = {str(e).lower() for e in extensions}
        self.zvfh = "zvfh" in names
        self.zvfbfmin = "zvfbfmin" in names
        self.zvfbfwma = "zvfbfwma" in names
        self.vrf = np.zeros((32, self.vlenb), dtype=np.uint8)
        #: vtype bits above bit 7 that installed extensions define (install_vtype_extension).
        self.vtype_ext_mask = 0
        self.vtype_hooks = []
        #: Flat byte view of the whole file, for group slicing without a reshape.
        self._flat = self.vrf.reshape(-1)
        # Counters (fixed key set; see counters()).
        self.instructions = 0
        self.elements = 0
        self.mem_to_vec_bytes = 0
        self.vec_to_mem_bytes = 0
        self.reset()

    # -- configuration ---------------------------------------------------------

    def reset(self):
        self.vrf[:] = 0
        self.vstart = 0
        self.vxrm = 0
        self.vxsat = 0
        self.vl = 0
        self._set_vill()
        self.instructions = 0
        self.elements = 0
        self.mem_to_vec_bytes = 0
        self.vec_to_mem_bytes = 0

    def _set_vill(self):
        self.vtype = VILL
        self.vill = True
        self.sew = 8
        self.lmul_log2 = 0
        self.vlmax = 0
        self.vta = self.vma = 0

    def install_vtype_extension(self, mask, on_vset=None):
        """Let an extension (IME's Zvvm, Zvfbfa) define ``vtype`` bits above RVV 1.0's eight.

        ``mask`` is the extra bits that stop counting as reserved (never bit
        XLEN-1, ``vill``); masks from several extensions accumulate. ``on_vset``,
        if given, is ``on_vset(requested, old_vtype, form) -> vtype`` and runs
        on every ``vset*`` before the value is decoded, so the extension can
        apply its own write rules (WARL fields, fields ``vsetvli`` preserves):
        ``form`` is ``"vsetvl"`` or ``"vsetvli"`` (``vsetivli`` too). It may
        return any value; reserved bits left in it still set vill. With nothing
        installed every bit above bit 7 is reserved, as RVV 1.0 says.
        """
        if mask & VILL:
            raise VectorStateError("vill (vtype bit 63) cannot be made legal")
        self.vtype_ext_mask |= mask
        if on_vset is not None:
            self.vtype_hooks.append(on_vset)

    def vtype_fields(self, vtype):
        """The RVV 1.0 fields of ``vtype``: ``(vlmul, vsew, vta, vma, reserved)``.

        ``reserved`` is the bits that are neither RVV 1.0's nor an installed
        extension's (see :meth:`install_vtype_extension`); nonzero means vill.
        Extension fields are the caller's to extract (``vtype & mask``).
        """
        reserved = vtype & ~(0xFF | self.vtype_ext_mask) & ((1 << 64) - 1)
        return (
            vtype & 0x7,
            (vtype >> 3) & 0x7,
            (vtype >> 6) & 1,
            (vtype >> 7) & 1,
            reserved,
        )

    def decode_vtype(self, vtype):
        """``(sew, lmul_log2, vlmax, vta, vma)`` for a vtype value, or None if unsupported."""
        vlmul, vsew, vta, vma, reserved = self.vtype_fields(vtype)
        if reserved:
            return None  # vill, or a bit no installed extension defines
        if vlmul == 4:
            return None
        lmul_log2 = vlmul - 8 if vlmul & 4 else vlmul
        if vsew > 3:
            return None
        sew = 8 << vsew
        # Spike: SEW <= ELEN * min(LMUL, 1).
        if sew > (self.elen >> -lmul_log2 if lmul_log2 < 0 else self.elen):
            return None
        vlmax = (
            (self.vlen << lmul_log2) // sew
            if lmul_log2 >= 0
            else (self.vlen // sew) >> -lmul_log2
        )
        return sew, lmul_log2, vlmax, vta, vma

    def set_vl(self, rd, rs1, avl, vtype, form="vsetvl"):
        """``vset{i}vl{i}``: install ``vtype`` and set ``vl``, following Spike; returns the new vl.

        ``rs1`` is the AVL register number (or, for ``vsetivli``, anything
        nonzero): ``rs1 = x0`` with ``rd != x0`` asks for VLMAX, and with
        ``rd = x0`` keeps the current vl (illegal, so vill, if VLMAX changes).
        Otherwise ``vl = min(AVL, VLMAX)``.
        """
        old_vlmax = self.vlmax
        for hook in self.vtype_hooks:
            vtype = hook(vtype, self.vtype, form) & ((1 << 64) - 1)
        decoded = self.decode_vtype(vtype)
        if decoded is not None and rd == 0 and rs1 == 0 and decoded[2] != old_vlmax:
            decoded = None
        if decoded is None:
            self._set_vill()
            self.vl = 0
        else:
            self.vtype = vtype
            self.vill = False
            self.sew, self.lmul_log2, self.vlmax, self.vta, self.vma = decoded
            if rs1 != 0:
                self.vl = min(avl, self.vlmax)
            elif rd != 0:
                self.vl = self.vlmax
            # rd == rs1 == 0: keep vl (VLMAX is unchanged, so it still fits)
        self.vstart = 0
        return self.vl

    # -- CSRs --------------------------------------------------------------------

    def read_csr(self, addr):
        if addr == CSR_VL:
            return self.vl
        if addr == CSR_VTYPE:
            return self.vtype
        if addr == CSR_VLENB:
            return self.vlenb
        if addr == CSR_VSTART:
            return self.vstart
        if addr == CSR_VXRM:
            return self.vxrm
        if addr == CSR_VXSAT:
            return self.vxsat
        if addr == CSR_VCSR:
            return (self.vxrm << 1) | self.vxsat
        raise KeyError(addr)

    def write_csr(self, addr, value):
        """Write a writable vector CSR (vl / vtype / vlenb are read-only and never reach here)."""
        if addr == CSR_VSTART:
            self.vstart = value & (self.vlen - 1)
        elif addr == CSR_VXRM:
            self.vxrm = value & 0x3
        elif addr == CSR_VXSAT:
            self.vxsat = value & 0x1
        elif addr == CSR_VCSR:
            self.vxsat = value & 0x1
            self.vxrm = (value >> 1) & 0x3
        else:
            raise KeyError(addr)

    # -- the accessor API ----------------------------------------------------------

    def vreg_group(self, base, lmul, sew_or_dtype):
        """The register group at ``base`` of ``lmul`` registers as a flat element view (see the module docs)."""
        nregs = _lmul_regs(lmul)
        dtype = _dtype(sew_or_dtype)
        if not 0 <= base < 32 or base % nregs or base + nregs > 32:
            raise VectorStateError(
                f"v{base} does not start a group of {nregs} registers"
            )
        return self.group(base, nregs, dtype)

    def vreg_bytes(self, base, nregs=1):
        """``nregs`` registers from ``base`` as one flat ``uint8`` view."""
        if not 0 <= base < 32 or nregs < 1 or base + nregs > 32:
            raise VectorStateError(f"no register range v{base}..v{base + nregs - 1}")
        b = self.vlenb
        return self._flat[base * b : (base + nregs) * b]

    def group(self, base, nregs, dtype):
        """The unchecked form of :meth:`vreg_group` the ISA uses: ``nregs`` registers as ``dtype``."""
        b = self.vlenb
        return self._flat[base * b : (base + nregs) * b].view(dtype)

    def mask(self, reg=0, n=None):
        bits = np.unpackbits(self.vrf[reg], bitorder="little").view(np.bool_)
        return bits if n is None else bits[:n]

    def set_mask(self, reg, bits, start=0):
        current = np.unpackbits(self.vrf[reg], bitorder="little")
        current[start : start + len(bits)] = bits
        self.vrf[reg] = np.packbits(current, bitorder="little")

    # -- reporting -----------------------------------------------------------------

    def counters(self):
        """Hub events: vector instructions retired, body elements processed, bytes each way to memory."""
        return {
            "instructions": self.instructions,
            "elements": self.elements,
            "mem_to_vec_bytes": self.mem_to_vec_bytes,
            "vec_to_mem_bytes": self.vec_to_mem_bytes,
        }

    @property
    def lmul(self):
        return Fraction(2) ** self.lmul_log2
