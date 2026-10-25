"""Shared pieces of the per-dwarf formulas: format widths, vector geometry, AME helpers."""

import math

from framework.naiad.costmodel.machine import FORMAT_BYTES


class Unsupported(ValueError):
    """The design cannot run this variant (no unit of the right kind, or a format it lacks)."""


def nbytes(fmt):
    return FORMAT_BYTES[fmt]


def compute_fmt(in_fmt, acc_fmt):
    """The arithmetic format the scalar / RVV baselines compute in (fp64 for fp64, else fp32)."""
    return "fp64" if "fp64" in (in_fmt, acc_fmt) else "fp32"


def vlmax(machine, sew_bits, lmul):
    return machine.vector.vlen * lmul // sew_bits


def need_vector(machine):
    if machine.vector is None:
        raise Unsupported("the design has no vector unit")
    return machine.vector


def need_ame(machine, *formats):
    if machine.mode != "ame" or machine.ame is None:
        raise Unsupported("the design has no AME matrix unit")
    return machine.ame


def strips(n, vl):
    """Strip-mined trip count."""
    return max(1, math.ceil(n / vl))


def need_acc_store(x):
    """Accumulator <-> memory: ``aload``/``astore``, or their ``matrix.h`` lowering through ``acc_move``."""
    if not x.acc_mem and "acc_move" not in x.features:
        raise Unsupported(
            "the kernel stores accumulators: the unit needs acc_mem or acc_move"
        )


def mx_drain(e, accs):
    """``naiad_mx_drain()``: an ``azero`` of every accumulator, then (without ``acc_mem``) the scratch fence."""
    for a in accs:
        e.azero(a)
    x = e.m.ame
    if x is not None and not x.acc_mem and "acc_move" in x.features:
        import math

        for i in range(4 if x.tiles >= 16 else 1):
            e.mop(
                math.ceil(x.m / x.acc_to_tile_rows_per_cycle),
                x.acc_to_tile_latency,
                accs_read=(0,),
                tiles_written=(("scratch", x.tiles - 1 - i),),
            )


def need_features(x, *features):
    missing = [f for f in features if f not in x.features]
    if missing:
        raise Unsupported(f"the kernel needs matrix feature(s) {missing}")


def ewise(e, fmt, accs_read, acc_written, tiles_read=(), tx=False):
    """One element-wise accumulator op (custom-3) on the unit's ``ewise`` pipe (or ``mma``)."""
    x = e.m.ame
    e.mop(
        x.ewise_ii(fmt, tx),
        x.ewise_tx_latency if tx else x.ewise_latency,
        tiles_read=tiles_read,
        accs_read=accs_read,
        accs_written=(acc_written,),
        pipe="mma" if x.ewise_pipe == "mma" else "ewise",
    )


def amov_to_tile(e, acc, tile):
    """``amov.t.a``: accumulator -> tile on the mma pipe."""
    x = e.m.ame
    e.mop(
        math.ceil(x.m / x.acc_to_tile_rows_per_cycle),
        x.acc_to_tile_latency,
        accs_read=(acc,),
        tiles_written=(tile,),
    )
