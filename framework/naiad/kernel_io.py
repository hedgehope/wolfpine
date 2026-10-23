"""Run a Naiad kernel with host-supplied inputs and read its outputs back.

The kernels behind the dwarf study take their operands from, and leave their
results in, global arrays named by ELF symbols. This module is the host side of
that contract: :func:`run_kernel` builds the design, loads the ELF, writes each
input array over its symbol, runs to the HTIF exit, and returns each output
array plus the run's counters. The experiment harness and every kernel's
correctness test go through it, so a kernel's data is generated and checked in
numpy (against an fp64 golden) rather than in C.

Memory survives ``NaiadSoC.reset``, so inputs written after the load are what
the program sees. Sizes are not checked against the ELF (the loader keeps
symbol values, not sizes): a kernel declares each array at its largest size
and reads the problem size from a scalar input.
"""

from dataclasses import dataclass, field

import numpy as np

from framework.naiad.builder import build
from framework.naiad.design import load_design

DEFAULT_MAX_CYCLES = 200_000_000


@dataclass
class KernelRun:
    exit_code: int | None
    cycles: int
    console: str
    outputs: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.exit_code == 0


def run_kernel(
    design,
    elf_path,
    inputs=None,
    outputs=None,
    max_cycles=DEFAULT_MAX_CYCLES,
    count_all=False,
):
    """Run ``elf_path`` on ``design`` (a path or a loaded ``Design``).

    ``inputs`` maps a symbol to a value: a numpy array (written as its raw
    bytes, so pick the dtype the kernel declares) or a Python int (written as
    a little-endian 64-bit word). ``outputs`` maps a symbol to
    ``(dtype, shape)``; each is read back after the run as a fresh array.
    """
    if not hasattr(design, "cores"):
        design = load_design(design)
    soc = build(design, count_all=count_all)
    image = soc.load(elf_path)
    for name, value in (inputs or {}).items():
        if isinstance(value, (int, np.integer)):
            data = int(value).to_bytes(8, "little", signed=value < 0)
        else:
            data = np.ascontiguousarray(value).tobytes()
        soc.memory.write(image.symbol(name), data)
    result = soc.run(max_cycles)
    read = {}
    for name, (dtype, shape) in (outputs or {}).items():
        dtype = np.dtype(dtype)
        count = int(np.prod(shape))
        raw = bytes(soc.memory.read(image.symbol(name), count * dtype.itemsize))
        read[name] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return KernelRun(
        exit_code=result.exit_code,
        cycles=result.cycles,
        console=result.console.decode(errors="replace"),
        outputs=read,
        counters=result.counters,
    )
