"""Validation helpers for YAML-described hardware: the error type, integers with size suffixes, key checks.

Shared by the Naiad design parser (:mod:`framework.naiad.design`, which
re-exports all three) and the generic models its sections configure (the RV
core timing in :mod:`framework.pe.rv.timing`, the IP blocks in
:mod:`framework.ip`), so those raise the same ``DesignError`` without
depending on the SoC target.
"""

import re

_SIZE_SUFFIXES = {
    "": 1,
    "k": 1 << 10,
    "kib": 1 << 10,
    "m": 1 << 20,
    "mib": 1 << 20,
    "g": 1 << 30,
    "gib": 1 << 30,
}


class DesignError(ValueError):
    pass


def parse_int(value, what):
    """An int from YAML: a plain int, or a string like ``0x1000`` / ``64KiB``."""
    if isinstance(value, bool):
        raise DesignError(f"{what}: expected an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(0x[0-9a-fA-F_]+|\d[\d_]*)\s*([a-zA-Z]*)\s*", value)
        if m and m.group(2).lower() in _SIZE_SUFFIXES:
            return int(m.group(1), 0) * _SIZE_SUFFIXES[m.group(2).lower()]
    raise DesignError(f"{what}: expected an integer, got {value!r}")


def check_keys(mapping, allowed, what):
    if not isinstance(mapping, dict):
        raise DesignError(f"{what}: expected a mapping, got {type(mapping).__name__}")
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise DesignError(
            f"{what}: unknown key(s) {unknown}; expected some of {sorted(allowed)}"
        )
