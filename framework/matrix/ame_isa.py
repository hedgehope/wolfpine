"""AME: the attached matrix unit as an RV ISA executor (custom-0 / custom-1).

The core offers each fetched word to this class like any other extension
(``run(register_file, memory_space, snoop, instr)``, True when claimed). The
unit is reached through ``register_file.matrix``, the same way the CSR
executor reaches ``register_file.csrs``; a core without one never claims a
word. Decoding comes from the encodings table (``encodings.yaml``), the
implemented-or-illegal checks from :meth:`MatrixUnit.resolve`, and the
numerics from ``tile_mac``, so what is left here is the GPR and memory side of
each instruction: one ``read``/``write`` per row (one per block when the rows
are dense), never per element.

Every word in the table's opcodes belongs to the matrix unit, so a word there
that decodes to nothing is an illegal instruction rather than an unknown one.
"""

from framework.matrix.unit import read_rows, write_rows
from framework.pe.rv.isa.rv_isa import RV_ISA


def _gpr(register_file, index):
    return register_file[index].read_uint()


def _signed(value, unit):
    return value - (1 << unit.xlen) if value >> (unit.xlen - 1) else value


def _tile_load(unit, insn, ops, register_file, memory, instr):
    role, fmt, t = insn.params["role"], ops["fmt"], ops.get("t", 0)
    rows, cols = unit.shape(role)
    if t:
        rows, cols = cols, rows
    base, stride = (
        _gpr(register_file, ops["base"]),
        _signed(_gpr(register_file, ops["stride"]), unit),
    )
    data = read_rows(memory, base, stride, rows, cols * fmt.dtype.itemsize, unit.xmask)
    unit.load_tile(ops["td"], role, fmt, data, transpose=bool(t))


def _tile_store(unit, insn, ops, register_file, memory, instr):
    data = unit.tile_bytes(ops["ts"], insn.params["role"], ops["fmt"])
    base, stride = (
        _gpr(register_file, ops["base"]),
        _signed(_gpr(register_file, ops["stride"]), unit),
    )
    write_rows(memory, base, stride, data, unit.xmask)
    unit.stats["tile_to_mem_bytes"] += data.size


def _acc_load(unit, insn, ops, register_file, memory, instr):
    fmt = ops["fmt"]
    base, stride = (
        _gpr(register_file, ops["base"]),
        _signed(_gpr(register_file, ops["stride"]), unit),
    )
    unit.load_acc(
        ops["ad"],
        fmt,
        read_rows(
            memory, base, stride, unit.m, unit.n * fmt.dtype.itemsize, unit.xmask
        ),
    )


def _acc_store(unit, insn, ops, register_file, memory, instr):
    data = unit.acc_bytes(ops["as"], ops["fmt"])
    base, stride = (
        _gpr(register_file, ops["base"]),
        _signed(_gpr(register_file, ops["stride"]), unit),
    )
    write_rows(memory, base, stride, data, unit.xmask)
    unit.stats["acc_to_mem_bytes"] += data.size


def _acc_zero(unit, insn, ops, register_file, memory, instr):
    unit.zero_acc(ops["ad"])


def _tile_shift(unit, insn, ops, register_file, memory, instr):
    role = "b" if ops["b"] else "a"
    amount = _signed(_gpr(register_file, ops["amount"]), unit)
    unit.shift_tile(ops["td"], ops["ts"], role, ops["fmt"], amount, bool(ops["cols"]))


def _mma(unit, insn, ops, register_file, memory, instr):
    unit.mma(
        ops["ad"],
        ops["ta"],
        ops["tb"],
        ops["in"],
        ops["acc"],
        insn.params["semiring"],
        instr,
    )


def _config_read(unit, insn, ops, register_file, memory, instr):
    value = unit.config_word(ops["sel"], instr)
    register_file[ops["rd"]].write(
        (value & unit.xmask).to_bytes(unit.xlen // 8, "little")
    )


def _acc_scalar(unit, insn, ops, register_file, memory, instr):
    unit.acc_scalar(
        ops["ad"], ops["fmt"], _gpr(register_file, ops["value"]), insn.params["op"]
    )


#: ``exec`` names in the table -> executor.
EXECUTORS = {
    "tile_load": _tile_load,
    "tile_store": _tile_store,
    "acc_load": _acc_load,
    "acc_store": _acc_store,
    "acc_zero": _acc_zero,
    "tile_shift": _tile_shift,
    "mma": _mma,
    "config_read": _config_read,
    "acc_scalar": _acc_scalar,
}


def _disasm(insn, ops):
    parts = []
    for op in insn.operands:
        value = ops[op.name]
        if op.kind == "gpr":
            parts.append(RV_ISA.get_reg_name(value))
        elif op.kind in ("in_fmt", "acc_fmt"):
            parts.append(f"{op.name}={value.name}")
        elif op.kind == "tile":
            parts.append(f"t{value}")
        elif op.kind == "acc":
            parts.append(f"acc{value}")
        else:
            parts.append(f"{op.name}={value}")
    return f"{insn.name} {', '.join(parts)}"


class RV_AME_ISA(RV_ISA):
    #: Major opcodes this ISA may claim (``RV64I``'s dispatch table offers it no others):
    #: the four custom opcodes, a superset of whatever ``encodings.yaml`` assigns.
    OPCODES = frozenset({0x0B, 0x2B, 0x5B, 0x7B})

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        unit = getattr(register_file, "matrix", None)
        if unit is None or instr & 0x7F not in unit.decoder.opcodes:
            return False
        decoded = unit.decoder.decode(instr)
        if decoded is None:
            raise unit.illegal(
                instr,
                "no matrix instruction has this encoding (reserved bits set, or an unassigned funct3)",
            )
        insn, raw_ops = decoded
        ops = unit.resolve(insn, raw_ops, instr)
        EXECUTORS[insn.exec](unit, insn, ops, register_file, memory_space, instr)
        if snoop:
            cls.print_snoop(snoop, _disasm(insn, ops))
        return True


def attach(core, unit):
    """Give ``core`` the matrix ``unit``: its state on the register file, AME last in the ISA chain.

    Appending is safe because every other extension declines custom-0/custom-1.
    ``isas`` and the pre-bound ``_isa_runs`` are kept in step.
    """
    core.matrix = unit
    core.register_file.matrix = unit
    if RV_AME_ISA not in core.isas:
        core.isas.append(RV_AME_ISA)
        core._isa_runs.append(RV_AME_ISA.run)
    return unit
