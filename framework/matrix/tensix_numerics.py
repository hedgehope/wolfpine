"""Tensix FPU numerics for ``tile_mac`` (Naiad B6, gate G3).

``tile_mac(..., numerics="tensix_bh" | "tensix_wh")`` reproduces, bit for bit,
what the Tensix matrix unit (the FPU) leaves in Dst for the same operands. The
arithmetic is not re-derived here: every MVMUL goes through the in-tree model's
own batched datapath, ``MatrixUnit._fpu_group_sums_batch`` and
``MatrixUnit._fpu_accumulate_batch`` in ``framework/pe/tensix/backends/matrix.py``
(static, pure functions, pinned against ttsim by ``fpu_accumulate_test.py``), so
there is one source of truth. What this module adds is the decomposition of an
M x N x K product into the MVMULs a tt-metal matmul issues, and the conversions
in and out of the 32-bit patterns the datapath works on. It needs no
``ml_dtypes``; only returning a bf16 *array* does, as for the IEEE path.

The datapath, in short (the docstrings in matrix.py have the detail): each
product is truncated to a fidelity-phase mantissa slice (5 bits of SrcA, 7 of
SrcB per phase); the 16 K lanes form two groups of eight, each aligned to its
largest exponent with round-half-up and summed as integers; the two group sums
and Dst are aligned to the largest of the three exponents, with every term
rounded to bf16 *before* the add when Dst is 16-bit; then renormalised and
rounded into Dst. An exponent of zero is zero (FTZ, operands and Dst alike),
nothing is infinite (exponent 255 is a saturated finite value), and on Wormhole
an accumulation that lands on exactly -1 unit renormalises 27 binades too high
(``negOneRenormBug``; Blackhole fixed it).

Decomposition (:func:`mvmul_schedule`)
--------------------------------------
``MVMUL`` computes ``Dst[i][j] += sum_k SrcB[i][k] * SrcA[k][j]`` over K = 16
lanes, for up to 8 Dst rows x 16 columns. tt-llk loads a matmul's in0 into SrcB
and in1 into SrcA, so for Naiad's ``C + A @ B``:

- ``A`` (M x K) is **SrcB** -- its rows are Dst rows, its columns the lanes;
- ``B`` (K x N) is **SrcA** -- its rows are the lanes, its columns Dst columns;
- ``C`` is Dst: fp32 accumulation is ``ALU_ACC_CTRL_Fp32_enabled`` (32-bit
  Dst), bf16 accumulation the 16-bit Dst.

Each Dst element is independent of every other, so M and N tile freely (the
8-row / 16-column MVMUL footprint changes nothing numerically) and are batched
whole here. K splits into chunks of 16 (zero-padded: a zero lane contributes
nothing, exactly as a zero-padded tile does), one MVMUL per chunk *per fidelity
phase*, each a full round into Dst. The order is tt-llk's
``matmul_configure_mop``: the fidelity phases are the MOP's inner loop around a
whole tile's MVMULs, so a ``matmul_tiles`` call over a 32-wide K tile runs
phase 0 over both 16-chunks, then phase 1 over both, and so on, and successive
K tiles follow one another. ``phase_k`` is that sweep width (default 32, one
tt-metal tile; 16 puts the phases innermost, per MVMUL). LoFi is phase 0 only;
HiFi2/3/4 run phases 0-1/0-2/0-3, phase bit 0 selecting SrcA's low mantissa
slice and bit 1 SrcB's.

Inputs are rounded to ``in_fmt`` round-to-nearest-even first, exactly as the
IEEE path rounds them, so the two modes see the same operand values and differ
only in the datapath. bf16 operands enter Src exactly; tf32 operands exactly
too (Src's 19-bit datum is tf32). A NaN or infinite operand has exponent 255 and
is a huge finite number to the FPU.

Validation
----------
``tensix_numerics_test.py`` holds this bit-exact against the whole Tensix
backend (real MVMUL words issued to a ``TensixCoProcessor``, both arches, both
Dst widths, every fidelity, K 16-64, mixed / cancelling / denormal /
saturating operands), and against ttsim's packed output of a real tt-metal
``matmul_tiles`` K loop (``optests/matmulrand``, all 16 arch x fidelity x Dst
combinations).

Where it does not match, by construction
----------------------------------------
- **Formats.** fp16 (and the BFP*_b formats that unpack to it), int8, fp8 and
  fp32 inputs are rejected, as are fp16/fp64 accumulators: the in-tree model
  runs a real-number model for fp16/int8, so there is nothing exact to match.
  An fp32 CB reaches Src as tf32 on Blackhole (the implied SrcA format) but as
  7-bit-mantissa bf16 on Wormhole, whose configured SrcA format stays FP32 --
  choose ``in_fmt`` to match.
- **Getting values into Src.** Operands are rounded RNE here; how a tt-metal
  host or the unpacker narrowed them (truncation, BFP shared exponents) is
  upstream of the datapath and not modelled. ``C`` goes into Dst as its exact
  bits, as an unpack-to-Dst of the same format would put it.
- **Getting results out.** The result is Dst. Packing a 32-bit Dst to a bf16
  CB rounds half-up (not RNE) and flushes denormals
  (``DataFormatConversions.FP32InDstToBF16``); that is a separate step, applied
  in the ttsim test, not by ``tile_mac``. The saturated Dst value (exponent 255)
  reads back as an IEEE infinity.
- **Order.** ``phase_k=32`` is ``matmul_tiles`` / ``matmul_block``'s order.
  Another LLK could interleave phases differently. With a 16-bit Dst the
  order is visible (ttsim disagrees with ``phase_k=16`` on ~20-25% of
  elements); a 32-bit Dst packed to bf16 mostly hides it.
- **Mixed Src styles.** One ``in_fmt`` sets both operands' style, as SrcA's
  format does for an MVMUL. A tf32 datum read under the bf16 style loses its low
  three mantissa bits (as MOVB2D also drops them for non-tf32 formats); that
  mixed case is not a ``tile_mac`` input and is not modelled.
"""

import numpy as np

from framework.matrix.formats import _f32_to_tf32, _to_f32_sticky, get_format

#: The Tensix numerics ``tile_mac`` accepts, and whether each has Wormhole's
#: -1 renormalisation quirk.
TENSIX_NUMERICS = {"tensix_bh": False, "tensix_wh": True}

#: tt-metal's ``MathFidelity`` levels -> number of fidelity phases run.
FIDELITY_PHASES = {"lofi": 1, "hifi2": 2, "hifi3": 3, "hifi4": 4}

#: tt-metal's ComputeKernelConfig default.
DEFAULT_FIDELITY = "hifi4"

#: K per MVMUL, and the default fidelity sweep (one 32-wide tt-metal tile).
MVMUL_K = 16
DEFAULT_PHASE_K = 32

#: The formats the datapath is modelled bit-exactly for (8-bit-exponent Src,
#: and the two Dst widths). fp16/int8 take matrix.py's real-number model.
IN_FORMATS = ("bf16", "tf32")
ACC_FORMATS = ("fp32", "bf16")


def fidelity_phases(fidelity):
    """The number of phases for a fidelity name (``"LoFi"``, ``"HiFi4"``...), case-insensitive."""
    key = str(fidelity).lower()
    if key not in FIDELITY_PHASES:
        raise ValueError(
            f"unknown Tensix fidelity {fidelity!r}; expected one of "
            f"{', '.join(n for n in ('LoFi', 'HiFi2', 'HiFi3', 'HiFi4'))}"
        )
    return FIDELITY_PHASES[key]


def check_formats(numerics, in_fmt, acc_fmt):
    """Raise ``ValueError`` unless ``in_fmt``/``acc_fmt`` are modelled exactly under ``numerics``."""
    in_f, acc_f = get_format(in_fmt), get_format(acc_fmt)
    if in_f.name not in IN_FORMATS:
        raise ValueError(
            f"{numerics} numerics models only {'/'.join(IN_FORMATS)} inputs bit-exactly, "
            f"not {in_f.name} (the Tensix FPU's fp16/int8 paths are not modelled exactly)"
        )
    if acc_f.name not in ACC_FORMATS:
        raise ValueError(
            f"{numerics} numerics accumulates in {'/'.join(ACC_FORMATS)} (32- or 16-bit Dst) "
            f"only, not {acc_f.name}"
        )


def mvmul_schedule(K, fidelity=DEFAULT_FIDELITY, phase_k=DEFAULT_PHASE_K):
    """The ``(k0, phase)`` of every MVMUL, in issue order, for a K-deep product.

    Each MVMUL covers lanes ``k0 .. k0 + 15`` (past K they are zero padding).
    """
    phases = fidelity_phases(fidelity)
    if phase_k < MVMUL_K or phase_k % MVMUL_K:
        raise ValueError(
            f"phase_k must be a positive multiple of {MVMUL_K}, got {phase_k!r}"
        )
    kpad = -(-K // MVMUL_K) * MVMUL_K
    for block in range(0, kpad, phase_k):
        for phase in range(phases):
            for k0 in range(block, min(block + phase_k, kpad), MVMUL_K):
                yield k0, phase


def exp_prod_adj(phase):
    """The product-exponent bias for a phase: -127, less each slice's scale."""
    return -127 - (5 if phase & 1 else 0) - (7 if phase & 2 else 0)


# -- conversions to and from the datapath's 32-bit patterns -------------------


def _is_bf16_dtype(dtype):
    return dtype.kind == "V" and dtype.itemsize == 2 and dtype.name == "bfloat16"


def _rne_to_bf16_bits(bits):
    """fp32 patterns (int64) rounded to bf16 (RNE), kept in the top 16 bits; NaN stays NaN."""
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return np.where(nan, (bits & 0xFFFF0000) | 0x00400000, rounded)


def to_fp32_bits(x, fmt):
    """``x`` rounded to ``fmt`` (bf16, tf32 or fp32; RNE) as int64 fp32 bit patterns.

    The rounding is the IEEE path's (``Format.round``) done on bits, so a bf16
    array, or a float array of any width, needs no ``ml_dtypes``.
    """
    f = get_format(fmt)
    x = np.asarray(x)
    if _is_bf16_dtype(x.dtype):
        bits = x.view(np.uint16).astype(np.int64) << 16
        return (
            bits
            if f.name in ("bf16", "fp32")
            else to_fp32_bits(bits.astype(np.uint32).view(np.float32), f)
        )
    if f.name == "fp32":
        f32 = get_format("fp32").round(x)
    else:
        f32 = _to_f32_sticky(x)
        if f.name == "tf32":
            f32 = _f32_to_tf32(f32)
    bits = np.asarray(f32, dtype=np.float32).view(np.uint32).astype(np.int64)
    if f.name == "bf16":
        bits = _rne_to_bf16_bits(bits)
    return bits


def from_fp32_bits(bits, acc_fmt):
    """Dst contents (fp32 patterns) as an array of ``acc_fmt``'s storage dtype.

    The saturated value (exponent 255, no mantissa) reads as an IEEE infinity.
    """
    f = get_format(acc_fmt)
    bits = np.asarray(bits, dtype=np.int64)
    if f.name == "bf16":
        return (bits >> 16).astype(np.uint16).view(f.dtype)
    return bits.astype(np.uint32).view(np.float32)


# -- the datapath ---------------------------------------------------------------


def tensix_mac_bits(
    dst,
    a,
    b,
    *,
    dst32=True,
    fidelity=DEFAULT_FIDELITY,
    wormhole=False,
    phase_k=DEFAULT_PHASE_K,
):
    """Run ``dst + a @ b`` through the FPU datapath, all as int64 fp32 bit patterns.

    ``a`` is (M, K) (SrcB), ``b`` (K, N) (SrcA) and ``dst`` (M, N) the initial
    Dst; every operand must already be representable in its Src format (bf16 or
    tf32 bits in fp32 layout) and ``dst`` in the Dst width (``dst32`` False:
    bf16 bits). Returns the final Dst as fp32 patterns, (M, N).
    """
    from framework.pe.tensix.backends.matrix import MatrixUnit

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    M, K = a.shape
    N = b.shape[1]
    kpad = -(-K // MVMUL_K) * MVMUL_K
    # Lanes lead (the axis the datapath reduces): SrcA as (lane, 1, N), SrcB as
    # (lane, M, 1), so the products broadcast to (lane, M, N) -- exactly the
    # shapes perform_mvmul_exact hands the same two functions.
    srcA = np.zeros((kpad, 1, N), np.int64)
    srcA[:K, 0, :] = b
    srcB = np.zeros((kpad, M, 1), np.int64)
    srcB[:K, :, 0] = a.T
    out = np.array(np.broadcast_to(np.asarray(dst, np.int64), (M, N)))
    for k0, phase in mvmul_schedule(K, fidelity, phase_k):
        out = MatrixUnit._fpu_accumulate_batch(
            MatrixUnit._fpu_group_sums_batch(
                srcA[k0 : k0 + MVMUL_K],
                srcB[k0 : k0 + MVMUL_K],
                phase,
                exp_prod_adj(phase),
            ),
            out,
            dst32,
            wormhole,
        )
    return out


def tensix_tile_mac(C, A, B, in_fmt, acc_fmt, *, numerics, fidelity=None, phase_k=None):
    """``tile_mac``'s Tensix strategy: ``C + A @ B`` as the Tensix FPU computes it."""
    if numerics not in TENSIX_NUMERICS:
        raise ValueError(
            f"unknown Tensix numerics {numerics!r}; expected one of {list(TENSIX_NUMERICS)}"
        )
    check_formats(numerics, in_fmt, acc_fmt)
    acc_f = get_format(acc_fmt)
    a = to_fp32_bits(A, in_fmt)
    b = to_fp32_bits(B, in_fmt)
    M, N = a.shape[0], b.shape[1]
    dst = np.zeros((M, N), np.int64) if C is None else to_fp32_bits(C, acc_f)
    out = tensix_mac_bits(
        dst,
        a,
        b,
        dst32=acc_f.name == "fp32",
        fidelity=DEFAULT_FIDELITY if fidelity is None else fidelity,
        wormhole=TENSIX_NUMERICS[numerics],
        phase_k=DEFAULT_PHASE_K if phase_k is None else int(phase_k),
    )
    return from_fp32_bits(out, acc_f)
