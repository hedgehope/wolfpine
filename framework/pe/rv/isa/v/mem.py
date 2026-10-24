"""Vector loads and stores: unit-stride, strided, indexed, segment, whole-register, mask, fault-only-first.

Memory is reached only through ``mem.read(addr, size)`` / ``mem.write(addr,
data)``, so the bus, the counters and the HTIF intercept see every byte. The
element addresses of an access are computed as one numpy array and then
**coalesced into contiguous runs**, one ``read``/``write`` per run: a unit-stride
access (unmasked, or a mask with few holes) is one call, and only strided or
indexed accesses with gaps pay a call per element. Masked-off elements are
never accessed.

Loads and stores honour ``vstart`` (elements below it are skipped), the only
instructions here that do; ``vstart`` is cleared afterwards. Nothing in this
core traps, so a fault mid-instruction never leaves a nonzero ``vstart``:
an access to unmapped memory raises the memory system's error out of the
instruction, except in a fault-only-first load, where a failing element
after the first trims ``vl`` to its index as the spec allows (a failing first
element raises).

Ordered and unordered indexed accesses behave the same (elements in order).
"""

import numpy as np

from framework.pe.rv.isa.v.common import (
    M64,
    aligned,
    nregs,
    overlapped,
    overlapped_widen,
    size,
)
from framework.pe.rv.isa.v.state import UINT

_EEW = {0: 8, 5: 16, 6: 32, 7: 64}


def _runs(addrs, esz):
    """Split an address array into ``(start index, end index)`` runs of consecutive elements."""
    n = len(addrs)
    if n == 0:
        return []
    if n == 1:
        return [(0, 1)]
    breaks = np.flatnonzero(np.diff(addrs) != np.uint64(esz)) + 1
    edges = [0, *breaks.tolist(), n]
    return list(zip(edges[:-1], edges[1:]))


def _read(mem, addrs, esz):
    """Bytes of ``len(addrs)`` elements of ``esz`` bytes, as one uint8 array."""
    out = np.empty(len(addrs) * esz, dtype=np.uint8)
    lst = addrs.tolist()
    for lo, hi in _runs(addrs, esz):
        out[lo * esz : hi * esz] = np.frombuffer(
            bytes(mem.read(lst[lo], (hi - lo) * esz)), dtype=np.uint8
        )
    return out


def _write(mem, addrs, esz, data):
    lst = addrs.tolist()
    for lo, hi in _runs(addrs, esz):
        mem.write(lst[lo], data[lo * esz : hi * esz].tobytes())


def _check_data_group(vs, vd, emul_lg, nf):
    if not -3 <= emul_lg <= 3:
        return False
    n = nregs(emul_lg)
    return aligned(vd, emul_lg) and nf * n <= 8 and vd + nf * n <= 32


def access(vs, rf, mem, is_load, mode, eew, nf, vd, rs1, rs2, vm, ff=False):
    """Unit-stride (mode 'unit'), strided ('strided') or indexed ('indexed') vector access.

    ``eew`` is the instruction's width: the data EEW for unit/strided, the
    index EEW for indexed (whose data EEW is SEW). ``nf`` is the field count
    (1 for a plain access).
    """
    if vs.vill:
        return False
    sew, lg = vs.sew, vs.lmul_log2
    if mode == "indexed":
        data_eew, data_lg = sew, lg
        idx_lg = lg + (eew.bit_length() - sew.bit_length())
        if eew > vs.elen or not -3 <= idx_lg <= 3 or not aligned(rs2, idx_lg):
            return False
        if not _check_data_group(vs, vd, data_lg, nf):
            return False
        if is_load:
            dn = size(data_lg)
            for f in range(nf):
                seg = vd + f * nregs(data_lg)
                if eew > sew and seg != rs2 and overlapped(seg, dn, rs2, size(idx_lg)):
                    return False
                if eew < sew:
                    if (overlapped if idx_lg < 0 else overlapped_widen)(
                        seg, dn, rs2, size(idx_lg)
                    ):
                        return False
                if nf >= 2 and overlapped(seg, dn, rs2, size(idx_lg)):
                    return False
    else:
        data_eew = eew
        data_lg = lg + (eew.bit_length() - sew.bit_length())
        if eew > vs.elen or not _check_data_group(vs, vd, data_lg, nf):
            return False
    if is_load and not vm and vd == 0:
        return False
    vl, start = vs.vl, vs.vstart
    esz = data_eew // 8
    n_regs = nregs(data_lg)
    if start >= vl:
        vs.vstart = 0
        return True
    base = rf[rs1].read_uint()
    count = vl - start
    if mode == "unit" and nf == 1 and vm and not ff:
        # The common case: one contiguous span, one bus access.
        group = vs.group(vd, n_regs, UINT[data_eew])
        addr = (base + start * esz) & M64
        nbytes = count * esz
        if is_load:
            group[start:vl] = np.frombuffer(
                bytes(mem.read(addr, nbytes)), dtype=UINT[data_eew]
            )
            vs.mem_to_vec_bytes += nbytes
        else:
            mem.write(addr, group[start:vl].tobytes())
            vs.vec_to_mem_bytes += nbytes
        vs.elements += count
        vs.vstart = 0
        return True
    elems = np.arange(start, vl, dtype=np.uint64)
    if mode == "unit":
        elem_addr = np.uint64(base) + elems * np.uint64(nf * esz)
    elif mode == "strided":
        stride = rf[rs2].read_uint()
        elem_addr = np.uint64(base) + elems * np.uint64(stride)
    else:
        idx = vs.group(rs2, nregs(idx_lg), UINT[eew])[start:vl].astype(np.uint64)
        elem_addr = np.uint64(base) + idx
    active = None if vm else vs.mask(0, vl)[start:]
    sel = np.arange(count) if active is None else np.flatnonzero(active)
    if len(sel) == 0:
        vs.vstart = 0
        return True
    # Field f of element i at elem_addr[i] + f*esz; laid out element-major in memory order.
    addrs = (
        elem_addr[sel][:, None]
        + np.arange(nf, dtype=np.uint64)[None, :] * np.uint64(esz)
    ).reshape(-1)
    groups = [vs.group(vd + f * n_regs, n_regs, UINT[data_eew]) for f in range(nf)]
    if is_load:
        try:
            raw = _read(mem, addrs, esz)
        except Exception:
            if not ff:
                raise
            raw, k = _read_ff(mem, addrs, esz, nf, len(sel))
            if k < len(sel):
                vs.vl = start + int(sel[k])  # the first failing element's index
                sel = sel[:k]
        vals = raw.view(UINT[data_eew]).reshape(len(sel), nf)
        for f in range(nf):
            groups[f][start + sel] = vals[:, f]
        vs.mem_to_vec_bytes += len(raw)
    else:
        vals = np.stack([groups[f][start + sel] for f in range(nf)], axis=1)
        data = np.ascontiguousarray(vals).view(np.uint8).reshape(-1)
        _write(mem, addrs, esz, data)
        vs.vec_to_mem_bytes += len(data)
    vs.elements += len(sel)
    vs.vstart = 0
    return True


def _read_ff(mem, addrs, esz, nf, n):
    """Fault-only-first: read element by element up to the first failure; ``(bytes, elements read)``."""
    chunks = []
    for k in range(n):
        try:
            chunk = [bytes(mem.read(int(addrs[k * nf + f]), esz)) for f in range(nf)]
        except Exception:
            if k == 0:
                raise
            break
        chunks.append(b"".join(chunk))
    return np.frombuffer(b"".join(chunks), dtype=np.uint8), len(chunks)


def whole(vs, rf, mem, is_load, nreg, eew, vd, rs1):
    """vl<nreg>re<eew>.v / vs<nreg>r.v: ``nreg`` whole registers, regardless of vtype and vl."""
    if nreg not in (1, 2, 4, 8) or vd % nreg:
        return False
    if not is_load and eew != 8:
        return False
    vlenb = vs.vlenb
    total = nreg * vlenb
    esz = eew // 8
    start = vs.vstart * esz  # vstart counts EEW elements
    if start >= total:
        vs.vstart = 0
        return True
    addr = (rf[rs1].read_uint() + start) & M64
    span = vs.vreg_bytes(vd, nreg)
    if is_load:
        span[start:] = np.frombuffer(
            bytes(mem.read(addr, total - start)), dtype=np.uint8
        )
        vs.mem_to_vec_bytes += total - start
    else:
        mem.write(addr, span[start:].tobytes())
        vs.vec_to_mem_bytes += total - start
    vs.elements += (total - start) // esz
    vs.vstart = 0
    return True


def mask_access(vs, rf, mem, is_load, vd, rs1):
    """vlm.v / vsm.v: ceil(vl / 8) bytes of a mask register (EEW 8, vstart in bytes)."""
    if vs.vill:
        return False
    evl = (vs.vl + 7) // 8
    start = vs.vstart
    if start < evl:
        addr = (rf[rs1].read_uint() + start) & M64
        row = vs.vrf[vd]
        if is_load:
            row[start:evl] = np.frombuffer(
                bytes(mem.read(addr, evl - start)), dtype=np.uint8
            )
            vs.mem_to_vec_bytes += evl - start
        else:
            mem.write(addr, row[start:evl].tobytes())
            vs.vec_to_mem_bytes += evl - start
        vs.elements += evl - start
    vs.vstart = 0
    return True


def decode(instr):
    """``(handler args)`` for a vector load/store word, or None if it is not one RVV 1.0 defines."""
    is_load = instr & 0x7F == 0x07
    width = (instr >> 12) & 0x7
    eew = _EEW.get(width)
    if eew is None:
        return None
    if (instr >> 28) & 1:  # mew: reserved in RVV 1.0 (Zvvm's tile loads use it)
        return None
    nf = (instr >> 29) + 1
    mop = (instr >> 26) & 0x3
    vm = (instr >> 25) & 1
    lumop = (instr >> 20) & 0x1F
    vd = (instr >> 7) & 0x1F
    rs1 = (instr >> 15) & 0x1F
    rs2 = lumop
    kind = "l" if is_load else "s"
    if mop == 0:
        if lumop == 0x00:
            seg = f"seg{nf}" if nf > 1 else ""
            return (
                access,
                (is_load, "unit", eew, nf, vd, rs1, 0, vm),
                f"v{kind}{seg}e{eew}.v",
            )
        if lumop == 0x08:
            if not vm:
                return None
            name = f"vl{nf}re{eew}.v" if is_load else f"vs{nf}r.v"
            return (whole, (is_load, nf, eew, vd, rs1), name)
        if lumop == 0x0B:
            if not vm or eew != 8 or nf != 1:
                return None
            return (mask_access, (is_load, vd, rs1), "vlm.v" if is_load else "vsm.v")
        if lumop == 0x10 and is_load:
            seg = f"seg{nf}" if nf > 1 else ""
            return (
                access,
                (True, "unit", eew, nf, vd, rs1, 0, vm, True),
                f"vl{seg}e{eew}ff.v",
            )
        return None
    if mop == 2:
        seg = f"seg{nf}" if nf > 1 else ""
        return (
            access,
            (is_load, "strided", eew, nf, vd, rs1, rs2, vm),
            f"v{kind}s{seg}e{eew}.v",
        )
    order = "o" if mop == 3 else "u"
    seg = f"seg{nf}" if nf > 1 else ""
    return (
        access,
        (is_load, "indexed", eew, nf, vd, rs1, rs2, vm),
        f"v{kind}{order}x{seg}ei{eew}.v",
    )
