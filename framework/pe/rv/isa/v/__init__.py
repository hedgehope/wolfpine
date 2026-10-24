"""RVV 1.0 for the Naiad RV64 core: V / Zve64d, Zvfh, Zvfbfmin, Zvfbfwma.

Modules:

* :mod:`.state` -- :class:`VectorState`: the register file (one
  ``uint8[32, VLENB]`` numpy array; LMUL groups and typed element views are
  views into it), the vector CSRs and the unit's counters. **Its docstring is
  the VRF accessor API** (``vreg_group``, ``vreg_bytes``, ``mask``) that the
  IME matrix mode builds on.
* :mod:`.isa` -- :class:`RV64_V_ISA`, the decoder and dispatcher.
* :mod:`.mem`, :mod:`.integer`, :mod:`.fpu`, :mod:`.permute` -- the handlers.
* :mod:`.fp` -- exact vector floating point (softfloat semantics, numpy fast paths).

Policies (the spec leaves these to the implementation; each matches Spike):

* **Tail and mask agnostic are undisturbed.** ``vta``/``vma`` are recorded in
  ``vtype`` and otherwise ignored: tail elements, masked-off elements and the
  tail of a mask result keep their old values, which is legal for either
  setting and what Spike does.
* **vstart.** Arithmetic with ``vstart != 0`` is illegal (Spike's default);
  loads and stores start at ``vstart``. Every executed vector instruction
  leaves ``vstart = 0``. Nothing traps in this core, so no instruction ever
  stops part-way and leaves a nonzero ``vstart`` behind; see :mod:`.mem` for
  what a memory fault does (it raises, except in fault-only-first loads).
* **Reductions.** Integer reductions are order-free. ``vfredusum`` and
  ``vfredosum`` (and the widening forms) are both summed **sequentially in
  element order**, Spike's order, so the unordered sum's NaN, signed-zero
  and rounding results match Spike bit for bit. Unordered differs only when
  no element is active: Spike then canonicalises a NaN start value, as here.
* **FP exceptions** accrue into ``fflags``; no trap is taken.
* ``mstatus.VS`` is not modelled (the generated crt0 sets it for Spike's sake).
"""

from framework.pe.rv.isa.v.isa import RV64_V_ISA
from framework.pe.rv.isa.v.state import VectorState, VectorStateError

__all__ = ["RV64_V_ISA", "VectorState", "VectorStateError", "attach"]


def attach(core, vlen=128, elen=64, extensions=("v",)):
    """Give ``core`` (an ``RV64I`` built with :class:`RV64_V_ISA`) its vector state; returns it.

    Sets ``core.vector`` and ``core.register_file.vector`` (where the ISA looks),
    binds the vector CSRs into the core's ``CSRFile64``, and sets ``misa.V``
    when the design lists the full V extension.
    """
    state = VectorState(vlen=vlen, elen=elen, extensions=extensions)
    core.vector = state
    core.register_file.vector = state
    csrs = getattr(core, "csrs", None)
    if csrs is not None and hasattr(csrs, "vector"):
        csrs.vector = state
        if "v" in {str(e).lower() for e in extensions}:
            csrs.misa |= 1 << (ord("V") - ord("A"))
    return state
