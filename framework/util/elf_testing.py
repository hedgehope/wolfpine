"""Write minimal RISC-V ELF executables for tests, with no toolchain.

Just enough ELF for :mod:`framework.util.elf_loader`: a header, one ``PT_LOAD``
per segment and a ``.symtab`` of absolute global symbols.
"""

import struct

EM_RISCV = 243
ET_EXEC = 2
PT_LOAD = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHN_ABS = 0xFFF1


def _strtab(names):
    blob = b"\0"
    offsets = {}
    for name in names:
        offsets[name] = len(blob)
        blob += name.encode() + b"\0"
    return blob, offsets


def write_elf(path, segments, entry, symbols=None, elfclass=32, machine=EM_RISCV):
    """``segments`` is ``[(addr, data, memsz)]``; ``memsz > len(data)`` makes zero fill."""
    symbols = symbols or {}
    is64 = elfclass == 64
    ehdr_fmt = "<16sHHIQQQIHHHHHH" if is64 else "<16sHHIIIIIHHHHHH"
    phdr_size, shdr_size, sym_size = (56, 64, 24) if is64 else (32, 40, 16)
    ehsize = struct.calcsize(ehdr_fmt)

    phoff = ehsize
    offset = phoff + phdr_size * len(segments)
    phdrs, payload = b"", b""
    for addr, data, memsz in segments:
        if is64:
            phdrs += struct.pack(
                "<IIQQQQQQ", PT_LOAD, 7, offset, addr, addr, len(data), memsz, 4
            )
        else:
            phdrs += struct.pack(
                "<IIIIIIII", PT_LOAD, offset, addr, addr, len(data), memsz, 7, 4
            )
        payload += data
        offset += len(data)

    strtab, str_off = _strtab(symbols)
    symtab = bytes(sym_size)  # the null symbol
    for name, value in symbols.items():
        if is64:
            symtab += struct.pack("<IBBHQQ", str_off[name], 0x11, 0, SHN_ABS, value, 0)
        else:
            symtab += struct.pack("<IIIBBH", str_off[name], value, 0, 0x11, 0, SHN_ABS)
    shstrtab, sh_off = _strtab([".symtab", ".strtab", ".shstrtab"])

    symtab_off = offset
    strtab_off = symtab_off + len(symtab)
    shstrtab_off = strtab_off + len(strtab)
    shoff = shstrtab_off + len(shstrtab)
    sections = [
        (0, 0, 0, 0, 0, 0),
        (sh_off[".symtab"], SHT_SYMTAB, symtab_off, len(symtab), 2, sym_size),
        (sh_off[".strtab"], SHT_STRTAB, strtab_off, len(strtab), 0, 0),
        (sh_off[".shstrtab"], SHT_STRTAB, shstrtab_off, len(shstrtab), 0, 0),
    ]
    shdrs = b""
    for name, kind, off, size, link, entsize in sections:
        info = 1 if kind == SHT_SYMTAB else 0  # index of the first global symbol
        fmt = "<IIQQQQIIQQ" if is64 else "<IIIIIIIIII"
        shdrs += struct.pack(fmt, name, kind, 0, 0, off, size, link, info, 1, entsize)

    ident = b"\x7fELF" + bytes([2 if is64 else 1, 1, 1]) + bytes(9)
    ehdr = struct.pack(
        ehdr_fmt,
        ident,
        ET_EXEC,
        machine,
        1,
        entry,
        phoff,
        shoff,
        0,
        ehsize,
        phdr_size,
        len(segments),
        shdr_size,
        4,
        3,
    )
    with open(path, "wb") as f:
        f.write(ehdr + phdrs + payload + symtab + strtab + shstrtab + shdrs)
    return path
