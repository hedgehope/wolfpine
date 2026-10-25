"""``python -m framework.naiad.costmodel.extract <dwarf> <variant> <size> [in,acc] [-o FILE]``.

Regenerates a kernel listing for :mod:`framework.naiad.costmodel.listing`: builds
the dwarf point (the harness's ``build_point``, cached), disassembles ``main``
and every function it calls with ``llvm-objdump``, and writes one
``addr: mnemonic operands`` line per instruction, matrix-unit words rendered
from ``framework/matrix/encodings.yaml`` as ``mx.<name> field=value ...``.

The listings under ``costmodel/listings/`` were produced this way; a dwarf
formula names the address ranges of its loop bodies in its listing, so a kernel
change means re-extracting and re-checking those ranges (the tests replay each
listing's formula against its recorded cycle count).
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

OBJDUMP = os.environ.get("NAIAD_OBJDUMP", "/opt/llvm/20.1.7/bin/llvm-objdump")
ABI = "zero ra sp gp tp t0 t1 t2 s0 s1 a0 a1 a2 a3 a4 a5 a6 a7 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 t3 t4 t5 t6".split()
LISTINGS = Path(__file__).resolve().parent / "listings"


def _decoder():
    from framework.matrix import encodings

    table = encodings.load_table()
    codes = dict(table.format_names)
    return encodings.Decoder(table), codes


def render_matrix(word, decoder, fmt_names):
    """``mx.<name> op=value ...`` for a matrix word, or None."""
    r = decoder.decode(word)
    if r is None:
        return None
    insn, ops = r
    parts = []
    for o in insn.operands:
        v = ops[o.name]
        if o.kind == "gpr":
            v = ABI[v]
        elif o.kind in ("in_fmt", "acc_fmt"):
            v = fmt_names.get(v, v)
        parts.append(f"{o.name}={v}")
    return f"mx.{insn.name} " + " ".join(parts)


def disassemble(elf, functions=("main",)):
    out = subprocess.run(
        [OBJDUMP, "-d", "--mattr=+v,+zvfh,+zvfbfwma,+zfbfmin,+zvfbfmin", str(elf)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    blocks = dict(
        re.findall(r"^[0-9a-f]+ <([^>]+)>:\n(.*?)(?=\n\n|\Z)", out, re.S | re.M)
    )
    decoder, fmt_names = _decoder()
    seen, order, lines = set(), list(functions), []
    while order:
        fn = order.pop(0)
        if fn in seen or fn not in blocks:
            continue
        seen.add(fn)
        lines.append(f"# <{fn}>")
        for line in blocks[fn].splitlines():
            m = re.match(r"\s*([0-9a-f]+):\s+([0-9a-f]{8}|[0-9a-f]{4})\s+(.*)$", line)
            if not m:
                continue
            addr, raw, text = m.groups()
            word = int(raw, 16)
            text = re.sub(r"\s+", " ", text.strip())
            if "<unknown>" in text or not text:
                mx = render_matrix(word, decoder, fmt_names)
                text = mx if mx else f".word 0x{word:08x}"
            for callee in re.findall(r"<([A-Za-z_][\w.]*)>", text):
                if callee not in seen and "+" not in callee:
                    order.append(callee)
            lines.append(f"{addr}: {text}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m framework.naiad.costmodel.extract")
    ap.add_argument("dwarf")
    ap.add_argument("variant")
    ap.add_argument("size", help="an int or a JSON list")
    ap.add_argument("formats", nargs="?", default="fp32,fp32")
    ap.add_argument("--cflags", default="")
    ap.add_argument("--design", default=None)
    ap.add_argument("--build-root", default=None)
    ap.add_argument("-o", "--out")
    args = ap.parse_args(argv)
    from framework.naiad import dwarfs

    size = json.loads(args.size)
    size = tuple(size) if isinstance(size, list) else size
    in_fmt, acc_fmt = args.formats.split(",")
    spec = dwarfs.load_spec(args.dwarf)
    design = Path(args.design or dwarfs.DEFAULT_DESIGN)
    elf = dwarfs.build_point(
        spec,
        args.variant,
        size,
        in_fmt,
        acc_fmt,
        design,
        args.build_root,
        args.cflags.split(),
    )
    head = f"# {args.dwarf}/{args.variant} size={args.size} {in_fmt}/{acc_fmt} cflags={args.cflags!r} design={design.name}\n"
    text = head + disassemble(elf)
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
