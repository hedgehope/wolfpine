"""Find, build and check the dwarf-reformulation kernels in ``kernels/naiad/dwarfs/``.

Each dwarf is a directory with a ``spec.py`` (the contract in the module
docstring of a spec, and in the Naiad plan's "Dwarf-kernel contract") plus one
subdirectory of C per variant. This module is the shared host side: it loads a
spec by name, builds one (variant, size, formats) point with the kernels
Makefile into its own build directory, and runs it through
:func:`framework.naiad.kernel_io.run_kernel` with the spec's encoded inputs.
The experiment harness and the per-dwarf tests both go through
:func:`run_point`, so a kernel is checked exactly the way it is measured.

``dwarfs_root`` (default ``kernels/naiad/dwarfs``) points :func:`dwarf_names` /
:func:`load_spec` at another tree of dwarfs, as the harness's own tests do with
a toy one; a spec's sources are always found next to its ``spec.py``.

Builds are cached by what reaches the compiler: the design's *generated*
files (``naiad.h``, ``linker.ld``, ``crt0.S``, ``naiad.mk``, banners aside),
the variant and the cflags. A sweep that changes only timing therefore reuses
one ELF for every point, as the Naiad design rules promise. A spec may define
an optional ``prebuild(*, build_dir, size, in_fmt, acc_fmt, design)`` hook
(keyword arguments; ``design`` is the design path make builds against) that
writes generated headers into ``build_dir`` before make runs; ``build_dir``
is then added to the include path. It must be deterministic in its arguments,
since its output is not part of the build key.

Only vector-coupled variants are compiled with the design's vector extensions:
on a design with RVV, every other variant (``scalar``, ``ame*``, ...) is built
with ``-march`` minus the vector extensions (:func:`scalar_march`), so the
compiler cannot auto-vectorise a baseline meant to be scalar and an AME
kernel's scalar side stays what it was on a design without V. A spec may name
its vector variants in ``VECTOR_VARIANTS``; by default they are the variants
whose name starts with one of :data:`VECTOR_VARIANT_PREFIXES`.

A spec may also name, in ``VARIANT_FEATURES``, the matrix features a variant
cannot build without (fft's and nbody's ``ame_fused`` need ``ewise``):
:func:`build_point` refuses a design that lacks one, and the harness skips
such points (:func:`missing_features`).
"""

import fcntl
import hashlib
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from framework.matrix.accuracy import compare
from framework.naiad.kernel_io import run_kernel

REPO = Path(__file__).resolve().parents[2]
KERNELS = REPO / "kernels" / "naiad"
DWARFS = KERNELS / "dwarfs"
DEFAULT_DESIGN = REPO / "framework" / "naiad" / "designs" / "naiad-dwarfs.yaml"
NAIAD_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")


#: Variant-name prefixes compiled with the design's vector extensions (unless a spec sets VECTOR_VARIANTS).
VECTOR_VARIANT_PREFIXES = ("rvv", "ime", "vme")


class DwarfError(Exception):
    pass


def uses_vector(spec, variant):
    """True when ``variant`` of ``spec`` may be compiled with the design's vector extensions."""
    explicit = getattr(spec, "VECTOR_VARIANTS", None)
    if explicit is not None:
        return variant in explicit
    return variant.startswith(VECTOR_VARIANT_PREFIXES)


def missing_features(spec, variant, design):
    """The matrix features ``variant`` needs (the spec's optional ``VARIANT_FEATURES``) that ``design`` lacks."""
    need = getattr(spec, "VARIANT_FEATURES", {}).get(variant, ())
    if not need:
        return []
    from framework.naiad.design import load_design

    if not hasattr(design, "cores"):
        design = load_design(design)
    matrix = design.matrix
    have = set(matrix.features) if matrix is not None else set()
    if matrix is not None and getattr(matrix, "acc_mem", False):
        have.add("acc_mem")
    return [f for f in need if f not in have]


def scalar_march(design):
    """``design``'s ``-march`` without its vector extensions, or None when it has none."""
    from dataclasses import replace

    from framework.naiad.design import (
        VECTOR_EXTENSIONS,
        VECTOR_SUBEXTENSIONS,
        load_design,
    )

    if not hasattr(design, "cores"):
        design = load_design(design)
    cores = design.cores
    if not cores.has_vector:
        return None
    vector = set(VECTOR_EXTENSIONS) | set(VECTOR_SUBEXTENSIONS)
    isa = tuple(e for e in cores.isa if e not in vector)
    return replace(cores, isa=isa).march


def _root(dwarfs_root):
    return DWARFS if dwarfs_root is None else Path(dwarfs_root)


def dwarf_names(dwarfs_root=None):
    return sorted(p.parent.name for p in _root(dwarfs_root).glob("*/spec.py"))


def load_spec(name, dwarfs_root=None):
    root = _root(dwarfs_root)
    path = root / name / "spec.py"
    if not path.exists():
        raise DwarfError(
            f"no dwarf '{name}' (no {path}); known: {dwarf_names(dwarfs_root)}"
        )
    tag = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    module = importlib.util.spec_from_file_location(f"naiad_dwarf_{name}_{tag}", path)
    spec = importlib.util.module_from_spec(module)
    module.loader.exec_module(spec)
    return spec


def spec_dir(spec):
    """The directory holding ``spec``'s ``spec.py`` and its variant subdirectories."""
    return Path(spec.__file__).resolve().parent


def design_build_key(design):
    """A hash of what ``design`` contributes to a binary: its generated files, banners aside.

    Two designs that differ only in timing (or anything else the generators
    ignore) share a key, so they share an ELF.
    """
    from framework.naiad.design import load_design
    from framework.naiad.gen import GENERATORS

    if not hasattr(design, "cores"):
        design = load_design(design)
    generators = dict(GENERATORS)
    try:  # the Makefile also emits naiad_gemm.h from the design
        from framework.naiad.emit.gemm import emit_gemm_header

        generators["naiad_gemm.h"] = emit_gemm_header
    except ImportError:
        pass
    h = hashlib.sha1()
    for filename, gen in sorted(generators.items()):
        try:
            text = gen(design)
        except Exception as e:  # the build will report it; keep the key deterministic
            text = f"error: {type(e).__name__}: {e}"
        text = "\n".join(
            line
            for line in text.splitlines()
            if "Generated by framework.naiad" not in line
        )
        h.update(filename.encode() + b"\0" + text.encode() + b"\0")
    return h.hexdigest()


def _sources_key(spec):
    """A hash of every C source and header a dwarf build can read.

    The kernels Makefile tracks neither a dwarf's local headers nor sources one
    variant ``#include``s from another, and a build directory is reused by key,
    so the sources must be in the key or an edit silently reuses a stale ELF.
    """
    h = hashlib.sha1()
    roots = [
        spec_dir(spec),
        KERNELS / "include",
        REPO / "framework" / "naiad" / "platform",
    ]
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.suffix in (".c", ".h", ".S") and path.is_file():
                h.update(str(path.relative_to(REPO)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()


def build_point(
    spec,
    variant,
    size,
    in_fmt,
    acc_fmt,
    design=DEFAULT_DESIGN,
    build_root=None,
    extra_cflags=(),
):
    """Build one variant of ``spec`` for ``size`` and formats; returns the ELF path.

    The build directory is keyed by everything that changes the binary (the
    design's generated files, variant, cflags, the C sources and headers), so repeated points reuse their
    ELF and different points never share objects. The design is copied into
    the build directory the first time, and make always builds against that
    copy, so a later design with the same key never triggers a rebuild. A lock
    file serialises concurrent builds of one key (parallel sweep workers).
    ``extra_cflags`` are appended after the spec's (a sweep's feature switches).
    """
    design = Path(design)
    prog = f"{spec.NAME}_{variant}"
    if variant not in spec.VARIANTS:
        raise DwarfError(
            f"dwarf '{spec.NAME}' has no variant '{variant}'; it has {sorted(spec.VARIANTS)}"
        )
    prog_dir = spec_dir(spec) / spec.VARIANTS[variant]
    missing = missing_features(spec, variant, design)
    if missing:
        raise DwarfError(
            f"{spec.NAME}/{variant} needs matrix feature(s) {missing}, which {design.name} does not list"
        )
    cflags = ["-O2", "-g", *spec.cflags(size, in_fmt, acc_fmt), *extra_cflags]
    march = None if uses_vector(spec, variant) else scalar_march(design)
    key = hashlib.sha1(
        "\0".join(
            [
                design_build_key(design),
                variant,
                *cflags,
                _sources_key(spec),
                *([f"march={march}"] if march else []),
            ]
        ).encode()
    ).hexdigest()[:12]
    build_dir = Path(build_root or KERNELS / "build" / "dwarfs") / f"{prog}-{key}"
    build_dir.mkdir(parents=True, exist_ok=True)
    with open(build_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build_design = build_dir / "design.yaml"
        if not build_design.exists():
            build_design.write_text(design.read_text())
        if hasattr(spec, "prebuild"):
            spec.prebuild(
                build_dir=build_dir,
                size=size,
                in_fmt=in_fmt,
                acc_fmt=acc_fmt,
                design=build_design,
            )
            cflags.append(f"-I{build_dir}")
        cmd = [
            "make",
            "-C",
            str(KERNELS),
            f"DESIGN={build_design}",
            f"PROG={prog}",
            f"PROG_DIR={prog_dir}",
            f"BUILD={build_dir}",
            f"PYTHON={sys.executable}",
            f"NAIAD_CC={NAIAD_CC}",
            f"CFLAGS={' '.join(cflags)}",
        ]
        if march:
            cmd.append(f"NAIAD_MARCH={march}")
        done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise DwarfError(
            f"building {prog} ({' '.join(cflags)}) failed:\n{done.stdout}{done.stderr}"
        )
    return build_dir / f"{prog}.elf"


@dataclass
class PointResult:
    dwarf: str
    variant: str
    size: object
    in_fmt: str
    acc_fmt: str
    exit_code: int | None
    cycles: int
    region_cycles: int | None
    error: dict = field(default_factory=dict)
    passed: bool = False
    counters: dict = field(default_factory=dict)
    console: str = ""


def run_point(
    spec,
    variant,
    size,
    in_fmt,
    acc_fmt,
    design=DEFAULT_DESIGN,
    seed=0,
    build_root=None,
    count_all=False,
    extra_cflags=(),
):
    """Build, run and check one point; ``error`` maps each output to ``compare``'s metrics."""
    elf = build_point(
        spec, variant, size, in_fmt, acc_fmt, design, build_root, extra_cflags
    )
    raw = spec.make_inputs(size, np.random.default_rng(seed))
    out_spec = dict(spec.outputs(size, acc_fmt))
    out_spec.setdefault("region_cycles", (np.uint64, (1,)))
    run = run_kernel(
        design, elf, spec.encode(raw, in_fmt, acc_fmt), out_spec, count_all=count_all
    )
    region = int(run.outputs.pop("region_cycles")[0])
    golden = spec.golden(raw)
    error = {}
    if run.ok:
        got = spec.decode(run.outputs, acc_fmt)
        error = {
            name: compare(np.asarray(got[name], dtype=np.float64), golden[name])
            for name in golden
        }
    tol = spec.tolerance(in_fmt, acc_fmt)
    passed = (
        run.ok and bool(error) and all(e["normwise"] <= tol for e in error.values())
    )
    return PointResult(
        dwarf=spec.NAME,
        variant=variant,
        size=size,
        in_fmt=in_fmt,
        acc_fmt=acc_fmt,
        exit_code=run.exit_code,
        cycles=run.cycles,
        region_cycles=region or None,
        error=error,
        passed=passed,
        counters=run.counters,
        console=run.console,
    )


def multicore_design(harts, out_dir=None, base=DEFAULT_DESIGN):
    """``base`` (naiad-dwarfs) with ``cores.count = harts``, written to ``out_dir`` and returned as a path.

    The multi-hart presets are generated, not hand-kept copies: everything but
    the hart count (and the name, ``<base name>-<harts>c``) is the base's, so
    the ``*_mc`` variants always compare against the one-hart design they came
    from. Each hart's ``program.stack_size`` comes out of the same program
    memory. ``out_dir`` defaults to ``kernels/naiad/build/designs``; the file is
    rewritten only when its text changes, so concurrent callers are safe. The
    sweep harness reaches the same designs with the override ``cores.count``.
    """
    import yaml

    base = Path(base)
    raw = yaml.safe_load(base.read_text())
    raw["cores"]["count"] = int(harts)
    raw["name"] = f"{raw['name']}-{int(harts)}c"
    out_dir = Path(out_dir or KERNELS / "build" / "designs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{raw['name']}.yaml"
    text = (
        f"# Generated by framework.naiad.dwarfs.multicore_design from {base.name}"
        f" with cores.count = {int(harts)}; do not edit.\n"
        + yaml.safe_dump(raw, sort_keys=False)
    )
    if not path.exists() or path.read_text() != text:
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        tmp.replace(path)
    return path
