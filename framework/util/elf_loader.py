"""Load a RISC-V ELF executable into a Wolfpine memory space.

Each ``PT_LOAD`` segment is written at its physical address (``p_paddr``, as
Spike does), with ``p_memsz - p_filesz`` bytes of zero fill after the file
image, so ``.bss`` is zero whatever the memory held before. A segment that is
not *wholly* inside one mapped range of the target's memory map is refused
before anything is written: a half-loaded image would fail later, at whatever
PC first touched the missing part, which is a far worse place to find out.

Both ELF classes are accepted; the caller checks the class against the core
it is loading for (:attr:`LoadedImage.xlen`).
"""

from dataclasses import dataclass, field

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

#: Zero fill is written in pieces of this size, so a large ``.bss`` never
#: needs one buffer the size of the whole region.
_ZERO_CHUNK = 1 << 20


class ELFLoadError(Exception):
    pass


@dataclass(frozen=True)
class Segment:
    paddr: int
    vaddr: int
    filesz: int
    memsz: int
    flags: int

    @property
    def end(self):
        """One past the last byte the segment occupies in memory."""
        return self.paddr + self.memsz


@dataclass
class LoadedImage:
    path: str
    entry: int
    xlen: int
    segments: list[Segment] = field(default_factory=list)
    #: Every defined symbol in ``.symtab``, name -> value.
    symbols: dict[str, int] = field(default_factory=dict)

    def symbol(self, name):
        try:
            return self.symbols[name]
        except KeyError:
            raise KeyError(f"ELF '{self.path}' defines no symbol '{name}'") from None


def _covering_range(memory_map, low, high):
    """The mapped range holding every byte of ``[low, high]``, or None."""
    addr_range, _target = memory_map.locate(low)
    if addr_range is None or high > addr_range.high:
        return None
    return addr_range


def read_elf(path):
    """Parse ``path`` without loading it: ``(LoadedImage, [(Segment, bytes)])``."""
    with open(path, "rb") as f:
        elf = ELFFile(f)
        if elf["e_machine"] != "EM_RISCV":
            raise ELFLoadError(
                f"'{path}' is not a RISC-V ELF (e_machine={elf['e_machine']})"
            )
        if elf["e_type"] != "ET_EXEC":
            raise ELFLoadError(
                f"'{path}' is not an executable (e_type={elf['e_type']}); link it first"
            )
        image = LoadedImage(path=str(path), entry=elf["e_entry"], xlen=elf.elfclass)
        payloads = []
        for seg in elf.iter_segments():
            if seg["p_type"] != "PT_LOAD" or seg["p_memsz"] == 0:
                continue
            segment = Segment(
                paddr=seg["p_paddr"],
                vaddr=seg["p_vaddr"],
                filesz=seg["p_filesz"],
                memsz=seg["p_memsz"],
                flags=seg["p_flags"],
            )
            image.segments.append(segment)
            payloads.append((segment, seg.data()[: segment.filesz]))
        for section in elf.iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            for sym in section.iter_symbols():
                if sym.name and sym["st_shndx"] != "SHN_UNDEF":
                    image.symbols[sym.name] = sym["st_value"]
    return image, payloads


def load_elf(path, memory):
    """Write ``path``'s loadable segments into ``memory`` and describe the image.

    ``memory`` is a :class:`~framework.memory.memory.MemorySpace` (anything with
    a ``memory_map`` and a ``write``). Returns a :class:`LoadedImage` carrying
    the entry point, the segments written and the symbol table.
    """
    image, payloads = read_elf(path)
    for segment in image.segments:
        if _covering_range(memory.memory_map, segment.paddr, segment.end - 1) is None:
            raise ELFLoadError(
                f"'{path}': segment [{hex(segment.paddr)}, {hex(segment.end)}) is not "
                f"wholly inside one mapped range of the target memory map"
            )
    for segment, data in payloads:
        if data:
            memory.write(segment.paddr, bytes(data))
        addr = segment.paddr + segment.filesz
        while addr < segment.end:
            n = min(_ZERO_CHUNK, segment.end - addr)
            memory.write(addr, bytes(n))
            addr += n
    return image
