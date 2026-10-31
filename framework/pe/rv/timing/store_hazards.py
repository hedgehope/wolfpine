"""Stores in flight: the memory read-after-write hazard between an asynchronous store and a later load.

The functional model writes memory in the tick a store issues, so a load that
follows always reads the right bytes. The *timing* model, though, lets several
store paths run on after issue -- an AME ``tstore``/``astore`` holds the tile
store path for ``T`` cycles, an RVV (or IME/VME tile) store holds the vector
LSU -- and a load on a different path (scalar, FP, AME tile load, a vector
load) could otherwise issue before the store's last byte has been written.
:class:`StoresInFlight` closes that gap: every asynchronous store registers the
bytes it writes and the cycle its write completes, and every load asks for the
cycle it may issue.

**Policy.**

* **Byte-exact overlap.** A footprint is ``rows`` runs of ``row_bytes`` at a
  signed byte ``stride`` (a contiguous access is one run). Two footprints
  conflict only if some byte is in both: a bounding-interval test rejects most
  pairs, and a strided pair is resolved exactly by solving for the rows that
  can meet (O(rows) at worst). Byte granularity rather than a cache-line or
  bank granule because the kernels store a tile into one column block of a
  matrix and then load the neighbouring block, whose bounding intervals
  interleave; a coarser granule (or a bounding box) would charge those
  false hazards and move cycles where no hazard exists. Indexed (gather /
  scatter) vector accesses are the one approximation: they use the bounding
  interval of their element addresses, which can only over-stall.
* **Completion** is the cycle the store's last byte leaves its path into the
  memory region: ``issue + path latency + T - 1`` for an AME tile/acc store
  (the store path's ``tile_store_latency``), ``issue + T`` for a vector-LSU
  store (when the LSU frees). The region's own ``latency`` is not added: it is
  the trip to the memory, which the later load pays too, so it cancels. A load
  that overlaps waits (stall reason ``mem_raw``, counter
  ``timing.stall_mem_raw``) until that cycle and then pays its own latency in
  full: no store-to-load forwarding from an in-flight async store.
* **Scalar stores** are posted and occupy the LSU for ``timing.scalar.store``
  cycles (1 in every shipped design), so by default they are not registered:
  their bytes are written before any later instruction on the same hart can
  load them. With ``timing.memory.scalar_stores`` a scalar store whose LSU
  occupancy (plus any MSI bus cycles) outlasts the next cycle is registered
  until the LSU frees, so a *vector* or tile load (a different issue path that
  can issue while the scalar LSU is still busy), another hart's load and a DMA
  read all wait for it. A one-cycle store is never registered (its bytes are
  visible from the next cycle, and registering it would make a same-cycle
  load on another hart depend on hart tick order).
* **WAR and WAW** (``timing.memory.war`` / ``timing.memory.waw``, off by
  default). The functional model applies every access at issue, so a *later*
  access can overtake an *earlier* one that is still draining on another path:
  a store that lands while an earlier load is still reading the bytes (WAR; the
  load would see the new data) or while an earlier store is still writing them
  (WAW; the earlier store's bytes would win). With the options on, a store
  waits (stall reasons ``mem_war`` / ``mem_waw``) until every overlapping
  in-flight read (``reads``: vector / tile / IME / VME loads, which hold their
  path for ``t`` cycles, and DMA source ranges) or write (this table, plus the
  DMA's destination ranges in ``dma_writes``) has completed. Loads never wait
  on a DMA *destination*: a core reading it before the engine shows DONE is
  the documented race, not a hazard the model resolves. A scalar load reads at
  issue so it is not tracked as an in-flight read.
* **DMA** (``timing.memory.dma_coherence``): see :mod:`framework.ip.dma`. A
  DMA chunk read waits for overlapping in-flight stores from any unit, a chunk
  write for overlapping stores and reads (WAW / WAR); the engine registers its
  own source and destination ranges here when ``war`` / ``waw`` are on.
* **Multi-hart**: one tracker per SoC, shared by every hart (all harts see one
  memory, so a store in flight on hart A delays hart B's load of those bytes
  as it would its own). Harts tick in lockstep, so cycles compare directly.
  DMA engines check and register only under ``timing.memory.dma_coherence``
  / ``war`` / ``waw`` (above).
* **Bounded and cheap when empty.** :attr:`until` is the latest completion
  registered; callers test ``until > cycle`` inline before calling
  :meth:`wait`, so with nothing in flight a load pays one attribute compare.
  At most ``capacity`` entries are kept: completed ones are pruned when the
  table fills, and if it is still full the earliest-completing entry is
  dropped (counted in :attr:`dropped`; optimistic, and unreachable with one
  in-order hart, whose store paths are serialised).
"""

MEM_RAW = "mem_raw"
MEM_WAR = "mem_war"
MEM_WAW = "mem_waw"
_EMPTY = (0, 0, 0, 0, 1, 0)


def footprint(base, stride, rows, row_bytes):
    """``(lo, hi, base, stride, rows, row_bytes)``, normalised: ``stride == 0`` means contiguous ``[lo, hi)``.

    Rows that touch or overlap (``|stride| <= row_bytes``) are one contiguous
    run; a negative stride is reversed so ``base`` is the lowest row. An empty
    access is ``(0, 0, ...)``, which overlaps nothing (addresses are
    non-negative).
    """
    if rows <= 0 or row_bytes <= 0:
        return _EMPTY
    if rows <= 1 or -row_bytes <= stride <= row_bytes:
        end = base + (rows - 1) * stride
        lo = min(base, end)
        return lo, max(base, end) + row_bytes, lo, 0, 1, max(base, end) + row_bytes - lo
    if stride < 0:
        base += (rows - 1) * stride
        stride = -stride
    return base, base + (rows - 1) * stride + row_bytes, base, stride, rows, row_bytes


def _hits(base, stride, rows, row_bytes, lo, hi):
    """True if any row ``[base + r*stride, +row_bytes)``, ``0 <= r < rows``, meets ``[lo, hi)``."""
    r_lo = max(0, (lo - row_bytes - base) // stride + 1)
    r_hi = min(rows - 1, (hi - 1 - base) // stride)
    return r_lo <= r_hi


def overlaps(a, b):
    """Whether two :func:`footprint` s share a byte."""
    if a[1] <= b[0] or b[1] <= a[0]:
        return False
    if not a[3]:
        return not b[3] or _hits(*b[2:], a[0], a[1])
    if not b[3]:
        return _hits(*a[2:], b[0], b[1])
    if a[4] > b[4]:
        a, b = b, a
    base, stride, rows, row_bytes = a[2:]
    for r in range(rows):
        lo = base + r * stride
        if _hits(*b[2:], lo, lo + row_bytes):
            return True
    return False


class StoresInFlight:
    """Address footprint -> completion cycle for every asynchronous store still writing."""

    __slots__ = (
        "entries",
        "until",
        "capacity",
        "dropped",
        "war",
        "waw",
        "ordered",
        "reads",
        "dma_writes",
    )

    def __init__(self, capacity=64):
        self.capacity = capacity
        self.war = self.waw = self.ordered = False
        self.reads = self.dma_writes = None
        self.reset()

    def configure(self, war=False, waw=False):
        """Turn on WAR / WAW tracking (``timing.memory.war`` / ``waw``): adds the read and DMA-write tables."""
        self.war, self.waw = bool(war), bool(waw)
        self.ordered = self.war or self.waw
        if self.ordered and self.reads is None:
            self.reads = StoresInFlight(self.capacity)
            self.dma_writes = StoresInFlight(self.capacity)

    def store_wait(self, fp, cycle):
        """``(cycle, reason)`` a store of footprint ``fp`` may issue given WAW / WAR, or ``(0, None)``.

        Call only when :attr:`ordered`.
        """
        at, why = 0, None
        if self.waw:
            for table in (self, self.dma_writes):
                t = table.wait(fp, cycle)
                if t > at:
                    at, why = t, MEM_WAW
        if self.war:
            t = self.reads.wait(fp, cycle)
            if t > at:
                at, why = t, MEM_WAR
        return at, why

    def reset(self):
        if self.reads is not None:
            self.reads.reset()
            self.dma_writes.reset()
        #: ``[(footprint, done), ...]``.
        self.entries = []
        #: The latest ``done`` registered (0: nothing in flight).
        self.until = 0
        self.dropped = 0

    def add(self, fp, done, cycle):
        """Register a store with :func:`footprint` ``fp`` that completes at ``done`` (issued at ``cycle``)."""
        if done <= cycle:
            return
        entries = self.entries
        if self.until <= cycle:
            entries.clear()
        elif len(entries) >= self.capacity:
            entries[:] = [e for e in entries if e[1] > cycle]
            if len(entries) >= self.capacity:
                entries.remove(min(entries, key=lambda e: e[1]))
                self.dropped += 1
        entries.append((fp, done))
        if done > self.until:
            self.until = done

    def add_span(self, base, nbytes, done, cycle):
        """:meth:`add` for one contiguous run."""
        self.add((base, base + nbytes, base, 0, 1, nbytes), done, cycle)

    def wait(self, fp, cycle):
        """The cycle a load of footprint ``fp`` may issue: the latest overlapping completion, or 0."""
        if self.until <= cycle:
            return 0
        at = 0
        for efp, done in self.entries:
            if done > cycle and done > at and overlaps(efp, fp):
                at = done
        return at

    def wait_span(self, base, nbytes, cycle):
        """:meth:`wait` for one contiguous run."""
        if self.until <= cycle:
            return 0
        return self.wait((base, base + nbytes, base, 0, 1, nbytes), cycle)
