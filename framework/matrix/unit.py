"""The matrix unit's architectural state and operations, independent of coupling mode.

:class:`MatrixUnit` owns what the instructions act on: ``tiles`` input tile
registers and ``accs`` accumulators, preallocated as raw bytes (numpy
``uint8``), never ``Register`` objects. A tile is sized for the larger of the
M x K ("a") and K x N ("b") shapes in the widest input format the unit
implements; an accumulator is M x N in the widest accumulator format. The bytes
are untyped: each instruction's format field says how to read them, through
:meth:`MatrixUnit.tile_view` / :meth:`MatrixUnit.acc_view`, which are numpy
views onto the storage (one numpy call per operation, no per-element loops).

Everything an implementation does *not* have -- a format the design does not
list, an optional feature it leaves out, a tile or accumulator index past the
end, an encoding no instruction claims -- raises
:class:`IllegalMatrixInstruction`, as hardware would trap. :meth:`resolve` is
the one place those checks live, so the IME/VME ISAs get the same gating by
calling it.

Functional only: nothing here counts cycles. The counters are traffic and work
(bytes per path, mma issues, MACs), for the run report and the later timing
model to read.
"""

import numpy as np

from framework.matrix.encodings import DECODER
from framework.matrix.formats import get_format
from framework.matrix.mac import tile_mac

#: The unit's counters, in report order.
COUNTER_NAMES = (
    "mem_to_tile_bytes",
    "tile_to_mem_bytes",
    "mem_to_acc_bytes",
    "acc_to_mem_bytes",
    "acc_to_tile_bytes",
    "tile_to_acc_bytes",
    "mma",
    "macs",
)


class IllegalMatrixInstruction(Exception):
    """A matrix instruction this unit does not implement (hardware: illegal-instruction trap)."""


class MatrixAccessFault(Exception):
    """A tile/accumulator load or store that touches unmapped memory."""


def _compute_dtype(fmt):
    """The dtype ``tile_mac`` computes in for an accumulator of ``fmt``."""
    if not fmt.is_float:
        return np.int64
    return np.float64 if fmt.name == "fp64" else np.float32


def _wide_dtype(fmt):
    """Where a tile op computes on ``fmt`` elements: fp64 for floats (one add of two
    fp32-or-narrower values rounded to fp64, then to ``fmt``, is the correctly
    rounded sum, since 53 >= 2 * 24 + 2), int64 for ints (wrapped by ``fmt.round``)."""
    return np.float64 if fmt.is_float else np.int64


#: ``treduce`` / ``tprefix`` op codes (the rs2 field) -> the binary fold.
_REDUCE_OPS = (np.add, np.maximum, np.minimum)
_PREFIX_OPS = (np.add, np.maximum)


class MatrixUnit:
    """One hart's matrix unit: shape, implemented formats/features, and state.

    ``in_formats`` / ``acc_formats`` are format names (``framework.matrix.formats``);
    every one must have a code in the decoder's table. Resolving a format's
    dtype is what imports ``ml_dtypes``, so constructing a unit that lists
    bf16/fp8 without it installed raises ``ImportError`` here, at build time.
    """

    def __init__(
        self,
        *,
        m,
        n,
        k,
        tiles,
        accs,
        in_formats,
        acc_formats,
        features=(),
        acc_mem=True,
        k_chunk=None,
        numerics="ieee",
        fidelity=None,
        xlen=64,
        decoder=DECODER,
    ):
        self.m, self.n, self.k = m, n, k
        self.num_tiles, self.num_accs = tiles, accs
        self.decoder = decoder
        table = decoder.table
        self.in_formats = {table.formats[name]: get_format(name) for name in in_formats}
        self.acc_formats = {
            table.formats[name]: get_format(name) for name in acc_formats
        }
        unknown = sorted(set(features) - set(table.features))
        if unknown:
            raise ValueError(
                f"unknown matrix feature(s) {unknown}; known: {sorted(table.features)}"
            )
        # aload/astore (the "acc_mem" feature) are on unless acc_mem=False: a
        # Ztt-like unit has no memory <-> accumulator path (see encodings.yaml).
        self.features = frozenset(features) | ({"acc_mem"} if acc_mem else set())
        self.k_chunk = k_chunk
        # The summation strategy every mma uses (``tile_mac``'s ``numerics``);
        # a Tensix one is checked here against every format the unit lists, so
        # an unmodelled pairing fails at build time, not on the first mma.
        self.numerics = numerics
        self.mac_options = {} if fidelity is None else {"fidelity": fidelity}
        if numerics != "ieee":
            from framework.matrix.tensix_numerics import (
                TENSIX_NUMERICS,
                check_formats,
                fidelity_phases,
            )

            if numerics not in TENSIX_NUMERICS:
                raise ValueError(
                    f"unknown matrix numerics {numerics!r}; expected ieee or one of {list(TENSIX_NUMERICS)}"
                )
            for in_f in self.in_formats.values():
                for acc_f in self.acc_formats.values():
                    check_formats(numerics, in_f, acc_f)
            if fidelity is not None:
                fidelity_phases(fidelity)
        elif fidelity is not None:
            raise ValueError("fidelity applies only to tensix numerics")
        self.xlen = xlen
        self.xmask = (1 << xlen) - 1

        in_bytes = max(f.dtype.itemsize for f in self.in_formats.values())
        acc_bytes = max(f.dtype.itemsize for f in self.acc_formats.values())
        self.tiles = np.zeros((tiles, max(m, k), max(k, n) * in_bytes), np.uint8)
        self.accs = np.zeros((accs, m, n * acc_bytes), np.uint8)
        self.stats = dict.fromkeys(COUNTER_NAMES, 0)
        self._config = {
            "M": m,
            "N": n,
            "K": k,
            "TILES": tiles,
            "ACCS": accs,
            "IN_FORMATS": sum(1 << c for c in self.in_formats),
            "ACC_FORMATS": sum(1 << c for c in self.acc_formats),
            "FEATURES": sum(1 << table.features[f] for f in self.features),
        }
        self._config_by_selector = {
            table.config[name]: value
            for name, value in self._config.items()
            if name in table.config
        }

    # -- lifecycle ---------------------------------------------------------

    def reset(self):
        self.tiles.fill(0)
        self.accs.fill(0)
        self.stats = dict.fromkeys(COUNTER_NAMES, 0)

    def counters(self):
        return dict(self.stats)

    # -- decode-time gating ------------------------------------------------

    def illegal(self, instr, why):
        return IllegalMatrixInstruction(
            f"illegal matrix instruction {instr:#010x}: {why}"
        )

    def resolve(self, insn, ops, instr):
        """Check a decoded instruction against this implementation; return its operands.

        Format codes become :class:`~framework.matrix.formats.Format` objects;
        indices and flags stay ints. Raises :class:`IllegalMatrixInstruction`
        for anything not implemented.
        """
        if insn.feature is not None and insn.feature not in self.features:
            raise self.illegal(
                instr,
                f"{insn.name} needs the '{insn.feature}' feature, which this unit does not have",
            )
        out = dict(ops)
        for op in insn.operands:
            value = ops[op.name]
            if op.kind == "tile":
                if value >= self.num_tiles:
                    raise self.illegal(
                        instr,
                        f"{insn.name}: tile {value} (this unit has {self.num_tiles})",
                    )
            elif op.kind == "acc":
                if value >= self.num_accs:
                    raise self.illegal(
                        instr,
                        f"{insn.name}: accumulator {value} (this unit has {self.num_accs})",
                    )
            elif op.kind == "in_fmt":
                out[op.name] = self._format(
                    self.in_formats, value, "input", insn, instr
                )
            elif op.kind == "acc_fmt":
                out[op.name] = self._format(
                    self.acc_formats, value, "accumulator", insn, instr
                )
            elif (
                op.kind == "flag"
                and value
                and op.feature is not None
                and op.feature not in self.features
            ):
                raise self.illegal(
                    instr,
                    f"{insn.name}.{op.name} needs the '{op.feature}' feature, which this unit does not have",
                )
        return out

    def _format(self, implemented, code, role, insn, instr):
        fmt = implemented.get(code)
        if fmt is None:
            name = self.decoder.table.format_names.get(code, f"code {code}")
            have = ", ".join(f.name for f in implemented.values())
            raise self.illegal(
                instr,
                f"{insn.name}: {name} is not an implemented {role} format (have: {have})",
            )
        return fmt

    # -- views -------------------------------------------------------------

    def shape(self, role):
        """``(rows, cols)`` of an ``"a"`` (M x K) or ``"b"`` (K x N) tile."""
        return (self.m, self.k) if role == "a" else (self.k, self.n)

    def tile_bytes(self, t, role, fmt):
        """Tile ``t``'s ``role`` shape as ``(rows, cols * itemsize)`` raw bytes (a view)."""
        rows, cols = self.shape(role)
        return self.tiles[t, :rows, : cols * fmt.dtype.itemsize]

    def tile_view(self, t, role, fmt):
        """Tile ``t``'s ``role`` shape read as ``fmt`` elements (a writable view)."""
        return self.tile_bytes(t, role, fmt).view(fmt.dtype)

    def acc_bytes(self, d, fmt):
        return self.accs[d, :, : self.n * fmt.dtype.itemsize]

    def acc_view(self, d, fmt):
        return self.acc_bytes(d, fmt).view(fmt.dtype)

    # -- operations --------------------------------------------------------

    def load_tile(self, t, role, fmt, data, transpose=False):
        """Fill tile ``t`` from raw ``data`` (``uint8``, one row per memory row).

        ``data`` is the ``role`` shape, or its transpose when ``transpose``.
        Bytes outside the shape become zero.
        """
        rows, cols = self.shape(role)
        eb = fmt.dtype.itemsize
        if transpose:
            data = (
                data.reshape(cols, rows, eb).transpose(1, 0, 2).reshape(rows, cols * eb)
            )
        self.tiles[t].fill(0)
        self.tiles[t, :rows, : cols * eb] = data
        self.stats["mem_to_tile_bytes"] += data.size

    def load_acc(self, d, fmt, data):
        self.acc_bytes(d, fmt)[...] = data
        self.stats["mem_to_acc_bytes"] += data.size

    def zero_acc(self, d):
        self.accs[d].fill(0)

    def mma(
        self, d, a, b, in_fmt, acc_fmt, semiring="plus_times", instr=0, transpose=None
    ):
        """``acc[d] = acc[d] (+) tile[a] (x) tile[b]`` through :func:`tile_mac`.

        ``transpose="a"`` reads tile ``a`` as the K x M block at its top left and
        multiplies by its transpose (``mma.at``); ``"b"`` reads tile ``b`` as an
        N x K block, transposed (``mma.bt``).
        """
        if in_fmt.is_float != acc_fmt.is_float:
            raise self.illegal(
                instr,
                f"mma cannot pair {in_fmt.name} inputs with a {acc_fmt.name} accumulator",
            )
        if semiring != "plus_times" and self.numerics != "ieee":
            raise self.illegal(
                instr,
                f"{semiring} mma has no {self.numerics} datapath (plus_times only)",
            )
        C = self.acc_view(d, acc_fmt)
        if transpose == "a":
            A = self._block(a, self.k, self.m, in_fmt, instr, "mma.at").T
        else:
            A = self.tile_view(a, "a", in_fmt)
        if transpose == "b":
            B = self._block(b, self.n, self.k, in_fmt, instr, "mma.bt").T
        else:
            B = self.tile_view(b, "b", in_fmt)
        C[...] = tile_mac(
            C,
            A,
            B,
            in_fmt,
            acc_fmt,
            k_chunk=self.k_chunk,
            semiring=semiring,
            numerics=self.numerics,
            **self.mac_options,
        )
        self.stats["mma"] += 1
        self.stats["macs"] += self.m * self.n * self.k

    def acc_scalar(self, d, fmt, bits, op):
        """``acc[d] = acc[d] * s`` (``op="mul"``) or ``+ s`` (``"add"``), rounded once to ``fmt``.

        ``s`` is the low ``fmt``-width bits of ``bits`` read as one ``fmt`` element.
        """
        eb = fmt.dtype.itemsize
        s = np.frombuffer(
            (bits & ((1 << (8 * eb)) - 1)).to_bytes(eb, "little"), fmt.dtype
        )[0]
        C = self.acc_view(d, fmt)
        cdt = _compute_dtype(fmt)
        with np.errstate(over="ignore", invalid="ignore"):
            wide, s = C.astype(cdt), cdt(s)
            C[...] = fmt.round(wide * s if op == "mul" else wide + s)

    def shift_tile(self, td, ts, role, fmt, amount, cols):
        """``td = ts`` shifted by ``amount`` rows (or columns), zero-filled."""
        rows, ncols = self.shape(role)
        eb = fmt.dtype.itemsize
        src = self.tile_bytes(ts, role, fmt).reshape(rows, ncols, eb)
        out = np.zeros_like(src)
        axis = 1 if cols else 0
        n = src.shape[axis]
        if abs(amount) < n:
            keep = slice(0, n - abs(amount))
            moved = slice(abs(amount), n)
            dst_sl, src_sl = (moved, keep) if amount >= 0 else (keep, moved)
            index = (slice(None),) * axis
            out[index + (dst_sl,)] = src[index + (src_sl,)]
        self.tile_bytes(td, role, fmt)[...] = out.reshape(rows, ncols * eb)

    # -- the Ztt-tracking operations (custom-2) ------------------------------

    def _block(self, t, rows, cols, fmt, instr, what):
        """Tile ``t``'s top-left ``rows x cols`` block of ``fmt`` elements (a view)."""
        eb = fmt.dtype.itemsize
        if rows > self.tiles.shape[1] or cols * eb > self.tiles.shape[2]:
            raise self.illegal(
                instr,
                f"{what}: a {rows} x {cols} {fmt.name} block does not fit a tile of this "
                f"{self.m} x {self.n} x {self.k} unit",
            )
        return self.tiles[t, :rows, : cols * eb].view(fmt.dtype)

    def _acc_shaped_view(self, role, fmt, instr, what):
        """Check that ``role``'s view is M x N and ``fmt`` a tile format too."""
        if self.shape(role) != (self.m, self.n):
            raise self.illegal(
                instr,
                f"{what}: the {role} view is {self.shape(role)}, not the accumulator's ({self.m}, {self.n})",
            )
        if self.decoder.table.formats[fmt.name] not in self.in_formats:
            raise self.illegal(
                instr, f"{what}: {fmt.name} is not an implemented input (tile) format"
            )

    def _write_tile(self, td, role, fmt, data):
        """Tile ``td``'s ``role`` view = ``data`` (elements of ``fmt``); other bytes zero."""
        self.tiles[td].fill(0)
        self.tile_view(td, role, fmt)[...] = data

    def acc_to_tile(self, td, a, fmt, role, instr=0):
        """``amov.t.a``: tile ``td``'s ``role`` view = accumulator ``a``, bit for bit."""
        self._acc_shaped_view(role, fmt, instr, "amov.t.a")
        data = self.acc_bytes(a, fmt).copy()
        self.tiles[td].fill(0)
        self.tile_bytes(td, role, fmt)[...] = data
        self.stats["acc_to_tile_bytes"] += data.size

    def tile_to_acc(self, d, ts, fmt, role, instr=0):
        """``amov.a.t``: accumulator ``d`` = tile ``ts``'s ``role`` view, bit for bit."""
        self._acc_shaped_view(role, fmt, instr, "amov.a.t")
        data = self.tile_bytes(ts, role, fmt).copy()
        self.accs[d].fill(0)
        self.acc_bytes(d, fmt)[...] = data
        self.stats["tile_to_acc_bytes"] += data.size

    def convert_tile(self, td, ts, src, dst, role, instr=0):
        """``tconv``: tile ``td``'s view = tile ``ts``'s ``src`` view converted to ``dst`` (RNE)."""
        if src.is_float != dst.is_float:
            raise self.illegal(
                instr, f"tconv: cannot convert {src.name} to {dst.name} (float <-> int)"
            )
        x = self.tile_view(ts, role, src).astype(_wide_dtype(src))
        with np.errstate(over="ignore", invalid="ignore"):
            out = dst.round(x)
        self._write_tile(td, role, dst, out)

    def reduce_tile(self, td, ts, fmt, role, cols, op, prefix, instr=0):
        """``treduce`` / ``tprefix`` along rows (``cols=0``) or columns of ``ts``'s view.

        Sequential in index order, every step rounded to ``fmt``; a reduction
        is broadcast along the reduced dimension.
        """
        ops = _PREFIX_OPS if prefix else _REDUCE_OPS
        if op >= len(ops):
            raise self.illegal(
                instr,
                f"{'tprefix' if prefix else 'treduce'}: op {op} (0-{len(ops) - 1} are defined)",
            )
        fn = ops[op]
        x = self.tile_view(ts, role, fmt).astype(_wide_dtype(fmt))
        if not cols:
            x = x.T  # walk the rows' elements as the leading axis
        out = np.empty_like(x)
        run = x[0]
        out[0] = run
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(1, x.shape[0]):
                run = fmt.round(fn(run, x[i])).astype(x.dtype)
                out[i] = run
        if not prefix:
            out[...] = run
        self._write_tile(td, role, fmt, out if cols else out.T)

    def shift2_tile(self, td, ta, tb, esz, cols, amount, instr=0):
        """``tshift2``: the R x R window ``amount`` rows (columns) into ``ta ++ tb``."""
        r = self.m
        if not self.m == self.n == self.k:
            raise self.illegal(
                instr,
                f"tshift2 needs a square unit, not {self.m} x {self.n} x {self.k}",
            )
        eb = 1 << esz
        widest = max(f.dtype.itemsize for f in self.in_formats.values())
        if eb > widest:
            raise self.illegal(
                instr,
                f"tshift2: {eb}-byte elements are wider than the widest input format ({widest} bytes)",
            )
        if amount > r:
            raise self.illegal(instr, f"tshift2: amount {amount} > {r}")
        a = self.tiles[ta, :r, : r * eb].reshape(r, r, eb)
        b = self.tiles[tb, :r, : r * eb].reshape(r, r, eb)
        axis = 1 if cols else 0
        window = np.concatenate([a, b], axis=axis)
        index = (slice(None),) * axis + (slice(amount, amount + r),)
        out = window[index].reshape(r, r * eb)
        self.tiles[td].fill(0)
        self.tiles[td, :r, : r * eb] = out

    def config_word(self, selector, instr=0):
        if selector not in self._config_by_selector:
            raise self.illegal(instr, f"mcfg: unknown selector {selector}")
        return self._config_by_selector[selector]


# -- memory <-> raw rows ----------------------------------------------------


def read_rows(memory, base, stride, rows, row_bytes, xmask):
    """``rows`` rows of ``row_bytes`` from ``base`` at a byte ``stride``, as ``uint8 (rows, row_bytes)``.

    A dense block (stride == row_bytes) is one read; otherwise one read per row.
    """
    if stride == row_bytes:
        chunks = [memory.read(base, rows * row_bytes)]
    else:
        chunks = [
            memory.read((base + r * stride) & xmask, row_bytes) for r in range(rows)
        ]
    buf = b"".join(bytes(c) for c in chunks)
    if len(buf) != rows * row_bytes:
        raise MatrixAccessFault(
            f"matrix load of {rows} x {row_bytes} bytes at {base:#x} (stride {stride}) reaches unmapped memory"
        )
    return np.frombuffer(buf, np.uint8).reshape(rows, row_bytes)


def write_rows(memory, base, stride, data, xmask):
    """Store ``uint8 (rows, row_bytes)`` ``data`` from ``base`` at a byte ``stride``."""
    rows, row_bytes = data.shape
    if stride == row_bytes:
        memory.write(base, np.ascontiguousarray(data).tobytes())
        return
    for r in range(rows):
        memory.write((base + r * stride) & xmask, data[r].tobytes())
