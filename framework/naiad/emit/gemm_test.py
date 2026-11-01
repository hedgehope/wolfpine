import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from framework.naiad.emit.gemm import (
    Blocking,
    GemmShape,
    check_blocking,
    choose_blocking,
    emit_gemm_header,
    format_pairs,
    main,
)

REPO = Path(__file__).resolve().parents[3]
KERNELS = REPO / "kernels" / "naiad"
DESIGNS = REPO / "framework" / "naiad" / "designs"
DWARFS_DESIGN = DESIGNS / "naiad-dwarfs.yaml"
NAIAD_CC = "/opt/llvm/20.1.7/bin/clang"
needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)

SHAPE = GemmShape("ame", 16, 16, 16, 8, 4, ("fp32", "bf16", "int8"), ("fp32", "int32"))


# -- blocking -------------------------------------------------------------------


def test_default_blocking_uses_every_accumulator_and_double_buffers_when_tiles_allow():
    assert choose_blocking(8, 4) == Blocking(2, 2, True)
    assert choose_blocking(4, 4) == Blocking(2, 2, False)
    assert choose_blocking(8, 2) == Blocking(1, 2, True)
    assert choose_blocking(3, 1) == Blocking(1, 1, False)
    assert choose_blocking(8, 4, double_buffer=False) == Blocking(2, 2, False)


def test_blocking_that_does_not_fit_is_refused():
    with pytest.raises(ValueError, match="accumulators"):
        check_blocking(Blocking(2, 3, False), 8, 4)
    with pytest.raises(ValueError, match="tiles"):
        check_blocking(Blocking(2, 2, True), 6, 4)
    with pytest.raises(ValueError, match="cannot hold"):
        choose_blocking(1, 4)
    with pytest.raises(ValueError, match="accumulators"):
        emit_gemm_header(SHAPE, block="4x2")


# -- structure ------------------------------------------------------------------


def test_pairs_are_float_with_float_and_int8_with_int32_never_narrowing():
    shape = GemmShape.from_design(DWARFS_DESIGN)
    pairs = format_pairs(shape)
    assert ("fp32", "fp32") in pairs
    assert ("bf16", "fp32") in pairs
    assert ("fp16", "fp32") in pairs
    assert ("fp64", "fp64") in pairs
    assert ("bf16", "bf16") in pairs
    assert ("int8", "int32") in pairs
    assert ("fp32", "int32") not in pairs
    assert ("int8", "fp32") not in pairs
    assert ("fp64", "fp32") not in pairs
    assert ("fp32", "bf16") not in pairs


def test_header_has_every_function_per_pair_and_names_formats_only_through_naiad_h():
    text = emit_gemm_header(SHAPE)
    for i, a in format_pairs(SHAPE):
        p = f"naiad_gemm_{i}_{a}"
        for fn in (
            "_init",
            "_kload0",
            "_kbody",
            "_kloop",
            "_store",
            "_store_kload0",
            "_ukernel",
            "_run",
            "_acc",
            "",
        ):
            assert re.search(rf"\b{p}{fn}\(", text), f"{p}{fn} missing"
        assert (
            f"#if defined(NAIAD_MATRIX_IN_{i.upper()}) && defined(NAIAD_MATRIX_ACC_{a.upper()})"
            in text
        )
    # Encodings come from naiad.h: no raw .insn, opcode or format literal in the emitted code.
    assert ".insn" not in text
    assert "NAIAD_MFMT_FP32" in text
    assert "NAIAD_MFMT_INT32" in text
    assert (
        "#if NAIAD_MATRIX_M != 16 || NAIAD_MATRIX_N != 16 || NAIAD_MATRIX_K != 16"
        in text
    )
    assert "#define NAIAD_GEMM_MB 32" in text
    assert "#define NAIAD_GEMM_DOUBLE_BUFFER 1" in text


def _function(text, name):
    start = text.index(f" {name}(")
    return text[start : text.index("\n}\n", start)]


def test_double_buffered_kbody_alternates_buffers_and_interleaves_loads_with_mma():
    body = _function(
        emit_gemm_header(SHAPE, pairs=["fp32:fp32"]), "naiad_gemm_fp32_fp32_kbody"
    )
    ops = re.findall(r"mx_(tload_[ab]|mma)\((\d+)(?:, (\d+), (\d+))?", body)
    # Steady state: four loads into buffer 1 (tiles 4-7) alternating with the 2x2 mma on buffer 0 (tiles 0-3).
    first = ops[:8]
    assert [o[0] for o in first] == [
        "tload_a",
        "mma",
        "tload_b",
        "mma",
        "tload_a",
        "mma",
        "tload_b",
        "mma",
    ]
    assert {int(o[1]) for o in first if o[0] != "mma"} == {4, 5, 6, 7}
    assert {(int(o[2]), int(o[3])) for o in first if o[0] == "mma"} == {
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    }
    assert (
        body.count("mx_mma(") == 4 * 4
    )  # mma(buf0)+mma(buf1) in the loop, mma(buf1) and mma(buf0) at the tail


def test_single_buffer_and_other_blockings_emit_the_matching_shape():
    text = emit_gemm_header(
        SHAPE, block="1x2", double_buffer=False, pairs=["fp32:fp32"]
    )
    assert "#define NAIAD_GEMM_MB 16" in text
    assert "#define NAIAD_GEMM_NB 32" in text
    assert "#define NAIAD_GEMM_DOUBLE_BUFFER 0" in text
    body = _function(text, "naiad_gemm_fp32_fp32_kbody")
    assert body.count("mx_mma(") == 2
    assert body.count("mx_tload_") == 3


def test_ime_and_vme_are_stubs():
    for mode in ("ime", "vme"):
        with pytest.raises(NotImplementedError, match=mode.upper()):
            emit_gemm_header(SHAPE, mode=mode)
    with pytest.raises(ValueError, match="unknown matrix mode"):
        emit_gemm_header(SHAPE, mode="sme")


def test_unknown_pair_is_refused():
    with pytest.raises(ValueError, match="not implemented"):
        emit_gemm_header(SHAPE, pairs=["fp64:fp64"])


def test_cli_writes_an_error_header_for_a_design_without_a_matrix_unit(tmp_path):
    out = tmp_path / "naiad_gemm.h"
    assert main([str(DESIGNS / "naiad-base-rv64.yaml"), "-o", str(out)]) == 0
    assert "#error" in out.read_text()
    assert "no matrix unit" in out.read_text()
    assert (
        main(
            [
                str(DWARFS_DESIGN),
                "-o",
                str(out),
                "--block",
                "2x1",
                "--pairs",
                "fp32:fp32",
            ]
        )
        == 0
    )
    text = out.read_text()
    assert "#define NAIAD_GEMM_BN 1" in text
    assert "naiad_gemm_int8_int32" not in text


# -- end to end ------------------------------------------------------------------

#: (m, n, k, extra stride padding): full blocks, odd step counts, every kind of edge, k = 0.
SHAPES = [
    (32, 32, 16, 0),
    (64, 32, 48, 0),
    (32, 64, 64, 3),
    (40, 24, 20, 5),
    (1, 1, 1, 0),
    (33, 65, 17, 2),
    (16, 40, 0, 0),
    (96, 32, 80, 0),
]


def _build(tmp_path, flags=""):
    subprocess.run(
        [
            "make",
            "-C",
            str(KERNELS),
            f"BUILD={tmp_path}",
            f"PYTHON={sys.executable}",
            f"NAIAD_CC={NAIAD_CC}",
            "PROG=emittest",
            f"PROG_DIR={KERNELS / 'dwarfs' / 'gemm' / 'emittest'}",
            f"DESIGN={DWARFS_DESIGN}",
            f"GEMM_EMIT_FLAGS={flags}",
        ],
        check=True,
        capture_output=True,
    )
    return tmp_path / "emittest.elf"


def _run(elf, m, n, k, pad, op, accumulate, rng):
    from framework.naiad.kernel_io import run_kernel

    lda, ldb, ldc = k + pad, n + pad, n + pad
    lo, hi = (-4, 5) if op != 1 else (-100, 100)
    A = rng.integers(lo, hi, (m, lda))
    B = rng.integers(lo, hi, (k, ldb))
    C0 = rng.integers(-50, 50, (m, ldc))
    want = (C0[:, :n] if accumulate else 0) + A[:, :k] @ B[:, :n]
    if op == 1:
        names, dt, ct = ("Ai", "Bi", "Ci"), np.int8, np.int32
    else:
        names, dt, ct = ("Af", "Bf", "Cf"), np.float32, np.float32
    Ain, Bin, Cin = A.astype(dt), B.astype(dt), C0.astype(ct)
    if op != 1 and pad:
        # The driver must never read past k / n: poison the stride padding.
        Ain[:, k:] = np.nan
        Bin[:, n:] = np.nan
    run = run_kernel(
        DWARFS_DESIGN,
        elf,
        inputs={
            "cfg": np.array([m, n, k, lda, ldb, ldc, accumulate, op], np.int64),
            names[0]: Ain,
            names[1]: Bin,
            names[2]: Cin,
        },
        outputs={names[2]: (ct, (m, ldc)), "region_cycles": (np.uint64, (1,))},
    )
    assert run.ok, run.console
    got = run.outputs[names[2]]
    np.testing.assert_array_equal(
        got[:, :n].astype(np.int64), want, err_msg=f"{(m, n, k, pad, op, accumulate)}"
    )
    # The stride padding of C is never written.
    np.testing.assert_array_equal(got[:, n:], Cin[:, n:])
    return run


@needs_clang
def test_emitted_driver_matches_numpy_at_every_edge_shape(tmp_path):
    elf = _build(tmp_path)
    rng = np.random.default_rng(1)
    for m, n, k, pad in SHAPES:
        for op in (0, 1):
            for accumulate in (0, 1):
                _run(elf, m, n, k, pad, op, accumulate, rng)


@needs_clang
@pytest.mark.parametrize(
    "flags", ["--single-buffer", "--block 1x2", "--block 2x1 --single-buffer"]
)
def test_other_blockings_are_also_exact(tmp_path, flags):
    elf = _build(tmp_path, flags)
    rng = np.random.default_rng(2)
    for m, n, k, pad in [(32, 64, 48, 0), (40, 24, 20, 5), (33, 65, 17, 2)]:
        for accumulate in (0, 1):
            _run(elf, m, n, k, pad, 0, accumulate, rng)


@needs_clang
def test_micro_kernel_alone_and_its_overlap(tmp_path):
    elf = _build(tmp_path)
    rng = np.random.default_rng(3)
    for k in (16, 32, 48, 256):
        for accumulate in (0, 1):
            _run(elf, 32, 32, k, 0, 2, accumulate, rng)
    # 2x2 fp32 blocking on the timed design is tile-load bound: 4 loads of 16 cycles per K step.
    # Double-buffered, the loads stream back to back under the mma, so the block takes the
    # load stream plus fill and drain: the first load's latency (2 + 6 + 16 - 1), the last
    # step's four mma (4 x 8) and pipeline (5 - 1), and four 16-cycle accumulator stores.
    steps = 256 // 16
    run = _run(elf, 32, 32, 256, 0, 2, 0, rng)
    region = int(run.outputs["region_cycles"][0])
    model = steps * 64 + 23 + 4 * 8 + 4 + 4 * 16
    assert abs(region - model) <= 8, (region, model)


# -- dual issue (timing.scalar.issue_width: 2) ---------------------------------------------------


def _with_width(base, width, tmp_path):
    import yaml

    from framework.naiad.design import load_design

    raw = yaml.safe_load((DESIGNS / base).read_text())
    raw["timing"]["scalar"]["issue_width"] = width
    path = tmp_path / f"{base}.{width}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_design(path)


def _insns(text):
    """Every ``.insn`` line of the asm in ``text``, in order (the vector / matrix ops)."""
    return [ln.strip() for ln in text.splitlines() if '".insn r' in ln]


def test_ame_single_issue_header_is_unchanged_by_the_dual_issue_emission(tmp_path):
    one = emit_gemm_header(_with_width("naiad-dwarfs.yaml", 1, tmp_path))
    assert "_adv(" not in one
    assert one == emit_gemm_header(DWARFS_DESIGN)  # no issue_width key = width 1
    assert one == emit_gemm_header(GemmShape.from_design(DWARFS_DESIGN))


def test_ame_dual_issue_k_loop_walks_persistent_pointers(tmp_path):
    one = emit_gemm_header(_with_width("naiad-dwarfs.yaml", 1, tmp_path))
    two = emit_gemm_header(_with_width("naiad-dwarfs.yaml", 2, tmp_path))
    assert one != two
    assert "mx_tload_a_adv" in two
    assert "mx_tload_b_adv" in two
    # the K loop no longer rebuilds a pointer per tile per step ...
    assert "const naiad_fp32_t *a1" not in two
    assert "const naiad_fp32_t *a1" in one

    # ... and issues the same tile loads and mmas in the same order, only the loads advance in place
    def ops(text):
        names = ("mx_tload_a", "mx_tload_b", "mx_mma(")
        return [
            ln.split("(")[0].strip().replace("_adv", "")
            for ln in text.splitlines()
            if any(n in ln for n in names)
        ]

    assert ops(one) == ops(two)
    assert one.count("mx_mma(") == two.count("mx_mma(")

    # outside kbody (init, store, kload0, staging, the driver) the text is identical
    def outside_kbody(text):
        out, skip = [], False
        for ln in text.splitlines():
            if "_kbody(long steps" in ln:
                skip = True
            if not skip:
                out.append(ln)
            elif ln == "}":
                skip = False
        return out

    assert outside_kbody(one) == outside_kbody(two)


def test_ame_dual_issue_width_is_read_from_the_design(tmp_path):
    shape = GemmShape.from_design(_with_width("naiad-dwarfs.yaml", 2, tmp_path))
    assert shape.issue_width == 2
    assert GemmShape.from_design(DWARFS_DESIGN).issue_width == 1
    assert "_adv(" in emit_gemm_header(shape)


def test_ime_single_issue_header_is_unchanged_by_the_dual_issue_emission(tmp_path):
    one = emit_gemm_header(_with_width("naiad-ime-timed.yaml", 1, tmp_path))
    assert "[bs]" not in one
    assert '"+r"(pa' not in one
    assert one == emit_gemm_header(DESIGNS / "naiad-ime-timed.yaml")


def test_ime_dual_issue_advances_the_pointers_inside_the_asm(tmp_path):
    one = emit_gemm_header(_with_width("naiad-ime-timed.yaml", 1, tmp_path))
    two = emit_gemm_header(_with_width("naiad-ime-timed.yaml", 2, tmp_path))
    assert '"+r"(pa0)' in two
    assert '"+r"(pb0)' in two
    # the vector / matrix ops are the same, in the same order: only scalar advances were added
    assert _insns(one) == _insns(two)
    n_loads = sum(1 for ln in _insns(two) if "VMTL" in ln or "VMTTL" in ln)
    adds = [
        ln
        for ln in two.splitlines()
        if ln.strip().startswith(('"addi %[pa', '"add %[pb'))
    ]
    assert len(adds) == n_loads
    assert n_loads > 0
