"""Sweep configs: YAML -> a list of :class:`Point`, each with its own derived design file.

A sweep file holds one or more *studies*. Top-level keys are defaults for every
study; ``studies:`` lists the studies (without it the file is one study).
Study keys:

``name``
    Study name (default: the file's stem). Rows carry it in the ``study`` column.
``dwarfs``
    ``all`` (every dwarf found) or a list of names. A dwarf that is missing or
    whose ``spec.py`` fails to import is skipped with a message.
``dwarfs_root``
    Optional, repo-relative: another dwarfs tree (the harness tests' toy).
``variants``
    Variant names, e.g. ``[ame, scalar]``, or ``all`` (each dwarf's own). Variants a dwarf lacks are skipped.
``sizes``
    ``smallest`` | ``mid`` | ``largest`` | ``all`` (selectors over the spec's
    ``SIZES``), an explicit list, or ``{dwarf: selector-or-list, "*": default}``.
``formats``
    ``all`` (the spec's ``FORMATS``) or a list of ``[in, acc]`` pairs. Non-AME
    variants only run the pairs in the spec's optional ``VARIANT_FORMATS[variant]``
    (default: fp32/fp32 and fp64/fp64, the F/D baselines).
``overrides``
    ``{dotted.key: [values]}``, the cartesian product of the value lists, each
    deep-merged onto the base design. A list element of the design is addressed
    by its ``name`` (``blocks.l1.params.latency``). A scalar is a one-value axis.
    The key ``kernel.cflags`` is not a design key: its values (a string or a list
    of strings) are appended to the kernel's cflags (feature switches).
``only``
    ``{key: [variant globs]}``: an override key (or ``normalise``) applies only to
    the matching variants (``ame*``); the others keep the base value, so they run once, not once
    per value, and their column is blank.
``normalise``
    ``{peak_flops_per_cycle, operand_bytes, mem_bytes_per_cycle, [tiles, accs,
    mode, power_of_two, vlen]}``, each a value or a list (another product axis):
    :func:`framework.matrix.timing.normalise` derives the unit's shape and
    timing per point (with the point's formats) and merges its ``matrix`` (shape,
    tiles, accs; the base's format list and mode are kept) and ``timing``
    (into ``timing.matrix``) sections after ``overrides``. For ``mode: ime`` it
    also sets ``cores.vlen`` (equal storage, or the pinned ``vlen``; ``null`` is
    equal storage), the vector LSU's ``timing.vector.mem_bytes_per_cycle``, and
    ``matrix.lambdas`` to every lambda permissible at the new VLEN (a lambda
    list is VLEN-specific; the base's may be illegal or lack the reference
    geometry's). None of this moves an address.
``base_design``
    Repo-relative base design (default ``framework/naiad/designs/naiad-dwarfs.yaml``).
``variant_designs``
    ``{variant glob: repo-relative design}``: the base design for matching
    variants (first match wins), e.g. ``{ime: framework/naiad/designs/naiad-ime-timed.yaml}``.
    Overrides, normalisation and the address check apply to it as to ``base_design``.
    A matrix variant whose base design has another coupling mode (``ime*`` needs
    ``matrix.mode: ime``, ``ame*`` needs ``ame``, ``vme*`` ``vme``) is skipped with
    a message, never run, so ``variants: all`` stays safe.
``seed``, ``count_all``
    Input RNG seed (default 0); per-class and per-block counters (default true).
``seeds``
    Optional list of seeds: every point runs once per seed (the seed is part of
    the point), and the CSV carries each group's median and max normwise error
    over the seeds (``normwise_median`` / ``normwise_max`` / ``n_seeds``, and
    ``precision.csv``). One seed can mislead badly for a reduction, whose input
    rounding errors may cancel. Overrides ``seed``.
``baseline``
    ``{variant: rvv, formats: [fp32, fp32] | same}``: the speedup denominator.
``references``
    Optional list of further variants (e.g. ``[scalar]``): each row also gets
    ``speedup_vs_<variant>`` (matched like the baseline, same formats rule) and
    each crossover group a ``crossover_vs_<variant>`` on the same ratio axis.
``ratio``
    ``{vector: scalar | rvv | <number>}``: the vector baseline's peak FMAs/cycle model
    (``rvv``: the design's ``timing.vector.datapath_bits`` over the arithmetic SEW).
``ratio_keys``
    Override keys that move the ratio (they are not grouping keys for the crossover).
``figures``
    Which figures this study feeds: ``crossover``, ``bytes``, ``accuracy``, ``redundancy``,
    ``feature_ranking``, ``sparsity``.
``quick``
    A mapping merged over the study under ``--quick`` (default: ``sizes: smallest``).

**Overrides never move addresses**: every derived design is parsed and each
block's base, size and window compared with the base design's
(:class:`AddressMoved` otherwise).
"""

import copy
import fnmatch
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from framework.naiad import dwarfs as dwarfs_mod
from framework.naiad.design import parse_design

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
DEFAULT_BASE = "framework/naiad/designs/naiad-dwarfs.yaml"
CFLAGS_KEY = "kernel.cflags"
NORMALISE_KEY = "normalise"
#: Pairs a non-matrix (F/D) variant runs unless the spec says otherwise.
FD_FORMATS = [("fp32", "fp32"), ("fp64", "fp64")]
DEFAULT_RATIO_KEYS = [
    "timing.matrix.macs_per_cycle",
    "timing.matrix.mma_ii",
    "timing.matrix.format_passes",
    "timing.scalar.issue_width",
    "normalise.peak_flops_per_cycle",
]
FIGURES = (
    "crossover",
    "bytes",
    "accuracy",
    "redundancy",
    "feature_ranking",
    "sparsity",
)
STUDY_KEYS = {
    "name",
    "description",
    "dwarfs",
    "dwarfs_root",
    "variants",
    "sizes",
    "formats",
    "overrides",
    "only",
    "normalise",
    "base_design",
    "variant_designs",
    "seed",
    "seeds",
    "count_all",
    "baseline",
    "references",
    "ratio",
    "ratio_keys",
    "figures",
    "quick",
}


class SweepError(ValueError):
    pass


class AddressMoved(SweepError):
    pass


@dataclass
class Study:
    name: str
    dwarfs: object = "all"
    dwarfs_root: str | None = None
    variants: object = field(default_factory=lambda: ["ame", "scalar"])
    sizes: object = "mid"
    formats: object = field(default_factory=lambda: [["fp32", "fp32"]])
    overrides: dict = field(default_factory=dict)
    only: dict = field(default_factory=dict)
    normalise: dict | None = None
    base_design: str = DEFAULT_BASE
    variant_designs: dict = field(default_factory=dict)
    seed: int = 0
    seeds: list | None = None
    count_all: bool = True
    baseline: dict = field(
        default_factory=lambda: {"variant": "scalar", "formats": ["fp32", "fp32"]}
    )
    references: list = field(default_factory=list)
    ratio: dict = field(default_factory=lambda: {"vector": "scalar"})
    ratio_keys: list = field(default_factory=lambda: list(DEFAULT_RATIO_KEYS))
    figures: list = field(default_factory=list)
    description: str = ""

    @property
    def base_path(self):
        return _repo_path(self.base_design)

    def base_for(self, variant):
        """The base design path for ``variant`` (``variant_designs``, else ``base_design``)."""
        for pat, path in (self.variant_designs or {}).items():
            if fnmatch.fnmatchcase(variant, pat):
                return _repo_path(path)
        return self.base_path

    @property
    def root(self):
        return None if self.dwarfs_root is None else _repo_path(self.dwarfs_root)

    @property
    def seed_list(self):
        """The seeds every point runs with: ``seeds`` if given, else ``[seed]``."""
        return [int(x) for x in self.seeds] if self.seeds else [int(self.seed)]

    def applies(self, key, variant):
        allowed = self.only.get(key)
        return allowed is None or any(
            fnmatch.fnmatchcase(variant, pat) for pat in allowed
        )


@dataclass
class Sweep:
    name: str
    path: Path
    studies: list

    @property
    def out_dir(self):
        return OUT / self.name


@dataclass
class Point:
    study: str
    dwarf: str
    variant: str
    size: object
    in_fmt: str
    acc_fmt: str
    seed: int
    count_all: bool
    #: Override key -> value, for every key of the study; None where it does not apply.
    overrides: dict
    #: Normalise parameter -> value (None when the study has none or it does not apply).
    normalise: dict | None
    #: The normaliser's ``derived`` section, for the record.
    normalised: dict | None
    extra_cflags: tuple
    design_hash: str
    design_path: str
    dwarfs_root: str | None
    point_id: str = ""

    def __post_init__(self):
        if not self.point_id:
            key = json.dumps(
                [
                    self.dwarf,
                    self.variant,
                    size_str(self.size),
                    self.in_fmt,
                    self.acc_fmt,
                    self.design_hash,
                    list(self.extra_cflags),
                    self.seed,
                    self.count_all,
                    self.dwarfs_root,
                ]
            )
            self.point_id = hashlib.sha1(key.encode()).hexdigest()[:16]


def _repo_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO / path


def size_str(size):
    if isinstance(size, (list, tuple)):
        return "x".join(str(s) for s in size)
    return str(size)


def _as_size(value):
    return tuple(value) if isinstance(value, list) else value


def _as_list(value):
    return value if isinstance(value, list) else [value]


# -- loading ---------------------------------------------------------------------


def load_sweep(path, quick=False):
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise SweepError(f"{path}: a sweep file is a mapping")
    name = str(raw.get("name", path.stem))
    defaults = {k: v for k, v in raw.items() if k not in ("studies", "name")}
    studies_raw = raw.get("studies") or [{"name": name}]
    studies = []
    seen = set()
    for i, entry in enumerate(studies_raw):
        merged = {**copy.deepcopy(defaults), **(entry or {})}
        merged.setdefault("name", name if len(studies_raw) == 1 else f"study{i}")
        if quick:
            merged = {**merged, **{"sizes": "smallest"}, **(merged.get("quick") or {})}
        merged.pop("quick", None)
        unknown = set(merged) - STUDY_KEYS
        if unknown:
            raise SweepError(
                f"{path}: study {merged['name']!r}: unknown keys {sorted(unknown)}"
            )
        bad_figures = set(merged.get("figures", [])) - set(FIGURES)
        if bad_figures:
            raise SweepError(
                f"{path}: unknown figures {sorted(bad_figures)}; known: {FIGURES}"
            )
        if merged["name"] in seen:
            raise SweepError(f"{path}: study {merged['name']!r} appears twice")
        seen.add(merged["name"])
        overrides = {
            k: _as_list(v) for k, v in (merged.pop("overrides", None) or {}).items()
        }
        if merged.get("seeds") is not None and (
            not isinstance(merged["seeds"], list) or not merged["seeds"]
        ):
            raise SweepError(
                f"{path}: study {merged['name']!r}: seeds must be a non-empty list"
            )
        studies.append(Study(overrides=overrides, **merged))
    return Sweep(name=name, path=path, studies=studies)


# -- design overrides --------------------------------------------------------------


def _child(node, part, key, create):
    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and item.get("name") == part:
                return item
        raise SweepError(f"override {key!r}: no list element named {part!r}")
    if not isinstance(node, dict):
        raise SweepError(f"override {key!r}: {part!r} is not inside a mapping")
    if part not in node:
        if not create:
            raise SweepError(f"override {key!r}: no {part!r}")
        node[part] = {}
    return node[part]


def deep_merge(dst, src):
    """Merge mapping ``src`` into ``dst`` in place: mappings merge, everything else replaces."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def apply_override(design, key, value):
    """Set dotted ``key`` in the raw design mapping (deep-merging a mapping value)."""
    parts = key.split(".")
    node = design
    for part in parts[:-1]:
        node = _child(node, part, key, create=True)
    last = parts[-1]
    if isinstance(node, list):
        raise SweepError(f"override {key!r}: cannot replace a whole list element")
    if isinstance(value, dict) and isinstance(node.get(last), dict):
        deep_merge(node[last], value)
    else:
        node[last] = copy.deepcopy(value)
    return design


def block_layout(design):
    return {b.name: (b.base, b.size, b.window) for b in design.blocks}


def check_addresses(base, derived, what="derived design"):
    """Raise :class:`AddressMoved` unless every block keeps its base, size and window."""
    want, got = block_layout(base), block_layout(derived)
    if set(want) != set(got):
        raise AddressMoved(
            f"{what}: blocks changed from {sorted(want)} to {sorted(got)}"
        )
    moved = [
        f"{name}: base/size/window {_fmt_layout(want[name])} -> {_fmt_layout(got[name])}"
        for name in want
        if want[name] != got[name]
    ]
    if moved:
        raise AddressMoved(
            f"{what} moves addresses (overrides must not): " + "; ".join(moved)
        )


def _fmt_layout(layout):
    return "/".join("-" if v is None else hex(v) for v in layout)


def design_hash(raw):
    return hashlib.sha1(
        json.dumps(raw, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


class DesignFactory:
    """Derives, checks and writes each design once per (base, overrides, normalise, formats)."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir) / "designs"
        self._base = {}
        self._made = {}

    def base(self, path):
        path = Path(path)
        if path not in self._base:
            raw = yaml.safe_load(path.read_text())
            self._base[path] = (raw, parse_design(copy.deepcopy(raw), path))
        return self._base[path]

    def derive(
        self, base_path, overrides, normalise=None, in_fmt="fp32", acc_fmt="fp32"
    ):
        """``(hash, path, normalised_derived)`` for the base plus ``overrides`` and ``normalise``."""
        raw, base = self.base(base_path)
        raw = copy.deepcopy(raw)
        applied = {}
        for key, value in overrides.items():
            if value is None or key == CFLAGS_KEY:
                continue
            apply_override(raw, key, value)
            applied[key] = value
        derived_info = None
        if normalise:
            derived_info = _apply_normalise(raw, normalise, in_fmt, acc_fmt)
            applied["normalise"] = normalise
        h = design_hash(raw)
        if h not in self._made:
            path = self.out_dir / f"{h}.yaml"
            try:
                design = parse_design(copy.deepcopy(raw), path)
            except Exception as e:
                raise SweepError(
                    f"overrides {applied} give an invalid design: {e}"
                ) from e
            check_addresses(base, design, f"design with overrides {applied}")
            self.out_dir.mkdir(parents=True, exist_ok=True)
            header = (
                f"# Derived by experiments.naiad from {Path(base_path).name}; do not edit.\n"
                + "".join(
                    f"#   {k} = {json.dumps(v, default=str)}\n"
                    for k, v in applied.items()
                )
            )
            text = header + yaml.safe_dump(raw, sort_keys=False)
            if not path.exists() or path.read_text() != text:
                tmp = path.with_suffix(f".tmp{id(self)}")
                tmp.write_text(text)
                tmp.replace(path)
            self._made[h] = path
        return h, self._made[h], derived_info


def _apply_normalise(raw, params, in_fmt, acc_fmt):
    from framework.matrix.timing import normalise

    params = dict(params)
    budget = [
        params.pop(k)
        for k in ("peak_flops_per_cycle", "operand_bytes", "mem_bytes_per_cycle")
    ]
    cfg = normalise(*budget, in_format=in_fmt, acc_format=acc_fmt, **params)
    matrix = {k: v for k, v in cfg.matrix.items() if k not in ("formats", "mode")}
    deep_merge(raw.setdefault("matrix", {}), matrix)
    deep_merge(raw.setdefault("timing", {}).setdefault("matrix", {}), cfg.timing)
    # VLEN and the vector LSU (VME has no lambdas to rewrite).
    if cfg.mode in ("ime", "vme"):
        _merge_ime(raw, cfg.derived)
    return dict(cfg.derived)


def _merge_ime(raw, derived):
    """The parts of an IME normalisation outside ``matrix`` / ``timing.matrix``: VLEN, LSU bandwidth, lambdas."""
    from framework.matrix.ime import permissible_lambdas

    vlen = int(derived["vlen"])
    raw.setdefault("cores", {})["vlen"] = vlen
    if derived.get("vector_mem_bytes_per_cycle") is not None:
        raw.setdefault("timing", {}).setdefault("vector", {})["mem_bytes_per_cycle"] = (
            derived["vector_mem_bytes_per_cycle"]
        )
    given = (raw.get("matrix") or {}).get("lambdas")
    if (
        given
    ):  # a lambda list is per VLEN: the normalised unit supports every permissible one
        raw["matrix"]["lambdas"] = {
            sew: list(permissible_lambdas(vlen, int(sew)))
            for sew in given
            if permissible_lambdas(vlen, int(sew))
        }


# -- expansion ---------------------------------------------------------------------


def select_sizes(spec, selector):
    if isinstance(selector, dict):
        selector = selector.get(spec.NAME, selector.get("*", "mid"))
    sizes = [_as_size(s) for s in spec.SIZES]
    if isinstance(selector, list):
        return [_as_size(s) for s in selector]
    if selector == "all":
        return sizes
    if not sizes:
        return []
    pick = {"smallest": 0, "mid": len(sizes) // 2, "largest": len(sizes) - 1}
    if selector in pick:
        return [sizes[pick[selector]]]
    if isinstance(selector, int):
        return [selector]
    raise SweepError(f"sizes: unknown selector {selector!r}")


def is_matrix_variant(spec, variant):
    return variant_mode(variant) is not None


def variant_mode(variant):
    """The coupling mode a matrix variant needs (``ame`` / ``ime`` / ``vme``), or None."""
    for mode in ("ame", "ime", "vme"):
        if variant.startswith(mode):
            return mode
    return None


def variant_formats(spec, variant, requested):
    """The ``(in, acc)`` pairs ``variant`` runs, of those requested."""
    supported = [tuple(p) for p in spec.FORMATS]
    if variant in getattr(spec, "VARIANT_FORMATS", {}):
        supported = [tuple(p) for p in spec.VARIANT_FORMATS[variant]]
    elif not is_matrix_variant(spec, variant):
        supported = list(FD_FORMATS)
    if requested == "all":
        return supported
    return [tuple(p) for p in requested if tuple(p) in supported]


def discover(study, log=print):
    """``{name: spec}`` for the study's dwarfs, skipping (with a message) missing or broken ones."""
    names = (
        dwarfs_mod.dwarf_names(study.root)
        if study.dwarfs == "all"
        else list(study.dwarfs)
    )
    specs = {}
    for name in names:
        try:
            spec = dwarfs_mod.load_spec(name, study.root)
            for attr in (
                "NAME",
                "VARIANTS",
                "SIZES",
                "FORMATS",
                "cflags",
                "make_inputs",
            ):
                if not hasattr(spec, attr):
                    raise dwarfs_mod.DwarfError(f"spec.py has no {attr}")
        except Exception as e:
            log(
                f"[{study.name}] skip dwarf {name!r}: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
            )
            continue
        specs[name] = spec
    return specs


def expand(sweep, factory=None, log=print):
    """Every :class:`Point` of every study, in order, with designs written. Points dedupe within a study."""
    factory = factory or DesignFactory(sweep.out_dir)
    points = []
    for study in sweep.studies:
        specs = discover(study, log)
        keys = list(study.overrides)
        combos = list(itertools.product(*(study.overrides[k] for k in keys))) or [()]
        norm = study.normalise or {}
        norm_keys = list(norm)
        norm_combos = (
            list(itertools.product(*(_as_list(norm[k]) for k in norm_keys)))
            if norm
            else [None]
        )
        seen = set()
        for name, spec in specs.items():
            variants = (
                list(spec.VARIANTS) if study.variants == "all" else study.variants
            )
            for variant in variants:
                if variant not in spec.VARIANTS:
                    log(
                        f"[{study.name}] skip {name}/{variant}: the dwarf has no such variant"
                    )
                    continue
                base_path = study.base_for(variant)
                want = variant_mode(variant)
                if want is not None:
                    matrix = factory.base(base_path)[1].matrix
                    have = getattr(matrix, "mode", None) if matrix is not None else None
                    if have != want:
                        log(
                            f"[{study.name}] skip {name}/{variant}: needs a matrix.mode {want} design "
                            f"({Path(base_path).name} has {have or 'no matrix unit'}); map one with variant_designs"
                        )
                        continue
                pairs = variant_formats(spec, variant, study.formats)
                for size in select_sizes(spec, study.sizes):
                    for in_fmt, acc_fmt in pairs:
                        for combo in combos:
                            ovr = {
                                k: (v if study.applies(k, variant) else None)
                                for k, v in zip(keys, combo)
                            }
                            for ncombo in norm_combos:
                                nparams = None
                                if ncombo is not None and study.applies(
                                    NORMALISE_KEY, variant
                                ):
                                    nparams = dict(zip(norm_keys, ncombo))
                                try:
                                    h, dpath, derived = factory.derive(
                                        base_path, ovr, nparams, in_fmt, acc_fmt
                                    )
                                except AddressMoved:
                                    raise
                                except Exception as e:
                                    log(
                                        f"[{study.name}] skip {name}/{variant} {ovr}: {e}"
                                    )
                                    continue
                                cflags = ovr.get(CFLAGS_KEY)
                                for seed in study.seed_list:
                                    point = Point(
                                        study=study.name,
                                        dwarf=name,
                                        variant=variant,
                                        size=size,
                                        in_fmt=in_fmt,
                                        acc_fmt=acc_fmt,
                                        seed=seed,
                                        count_all=study.count_all,
                                        overrides=ovr,
                                        normalise=nparams,
                                        normalised=derived,
                                        extra_cflags=tuple(_as_list(cflags))
                                        if cflags
                                        else (),
                                        design_hash=h,
                                        design_path=str(dpath),
                                        dwarfs_root=None
                                        if study.root is None
                                        else str(study.root),
                                    )
                                    if point.point_id in seen:
                                        continue
                                    seen.add(point.point_id)
                                    points.append(point)
    return points
