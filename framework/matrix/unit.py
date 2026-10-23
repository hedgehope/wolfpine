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
        k_chunk=None,
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
        self.features = frozenset(features)
        self.k_chunk = k_chunk
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

    def mma(self, d, a, b, in_fmt, acc_fmt, semiring="plus_times", instr=0):
        """``acc[d] = acc[d] (+) tile[a] (x) tile[b]`` through :func:`tile_mac`."""
        if in_fmt.is_float != acc_fmt.is_float:
            raise self.illegal(
                instr,
                f"mma cannot pair {in_fmt.name} inputs with a {acc_fmt.name} accumulator",
            )
        C = self.acc_view(d, acc_fmt)
        A = self.tile_view(a, "a", in_fmt)
        B = self.tile_view(b, "b", in_fmt)
        C[...] = tile_mac(
            C, A, B, in_fmt, acc_fmt, k_chunk=self.k_chunk, semiring=semiring
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
