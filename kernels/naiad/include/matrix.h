/*
 * matrix.h: .insn wrappers for the Naiad matrix unit (AME, attached mode).
 *
 * Every opcode, funct3 and field position comes from the design's generated
 * naiad.h (NAIAD_MI_*, NAIAD_MFMT_*, NAIAD_MCFG_*), which is generated from
 * framework/matrix/encodings.yaml, so an encoding change needs no edit here.
 * Semantics are documented per instruction in that table.
 *
 * Tile and accumulator indices live in register fields as literals ("x3" is
 * index 3), and formats in funct7, so every index and format argument must be
 * a compile-time constant. The wrappers are always_inline and use "i"
 * constraints, which fold only with optimisation on: build with -O1 or above.
 *
 *   mx_tload_a(t, fmt, base, stride)    tile t <- M x K, row stride in bytes
 *   mx_tload_b(t, fmt, base, stride)    tile t <- K x N
 *   mx_tload_at / mx_tload_bt           the same, from the transposed layout
 *   mx_tstore_a / mx_tstore_b           tile -> memory
 *   mx_aload(d, fmt, base, stride)      accumulator d <- M x N
 *   mx_astore(d, fmt, base, stride)     accumulator d -> memory
 *   mx_azero(d)                         accumulator d <- 0
 *   mx_mma(d, a, b, in, acc)            acc[d] += tile[a] x tile[b]
 *   mx_mma_minplus / mx_mma_maxplus     tropical forms (feature: semiring)
 *   mx_tshift(td, ts, b, cols, fmt, n)  shift a tile view by n rows / columns
 *   mx_ascal(d, fmt, bits) / mx_aadds   acc[d] *= s / += s (feature: epilogue)
 *   mx_mcfg(sel)                        configuration word (NAIAD_MCFG_*)
 */
#ifndef NAIAD_MATRIX_H
#define NAIAD_MATRIX_H

#include "naiad.h"

#ifndef NAIAD_HAS_MATRIX
#error "this design has no matrix unit (add a matrix: section to the design file)"
#endif
#ifndef __OPTIMIZE__
#error "matrix.h needs -O1 or above: tile indices and formats must fold to immediates"
#endif

#define MX_INLINE static inline __attribute__((always_inline))

/* R-type, rd = literal index, rs1/rs2 = x-registers. */
#define MX_R_IRR(insn, f7_, idx_, r1_, r2_)                                                            \
    __asm__ volatile(".insn r %[op], %[f3], %[f7], x%[i], %[a], %[b]"                              \
                     :                                                                             \
                     : [op] "i"(NAIAD_MI_##insn##_OPCODE), [f3] "i"(NAIAD_MI_##insn##_FUNCT3),     \
                       [f7] "i"(f7_), [i] "i"(idx_), [a] "r"(r1_), [b] "r"(r2_)                        \
                     : "memory")

/* R-type, all three register fields literal indices. */
#define MX_R_III(insn, f7_, d_, a_, b_)                                                                \
    __asm__ volatile(".insn r %[op], %[f3], %[f7], x%[rd], x%[ra], x%[rb]"                         \
                     :                                                                             \
                     : [op] "i"(NAIAD_MI_##insn##_OPCODE), [f3] "i"(NAIAD_MI_##insn##_FUNCT3),     \
                       [f7] "i"(f7_), [rd] "i"(d_), [ra] "i"(a_), [rb] "i"(b_))

/* -- tiles ---------------------------------------------------------------- */

MX_INLINE void mx_tload_a(const int t, const int fmt, const void *base, long stride) {
    MX_R_IRR(TLOAD_A, fmt << NAIAD_MI_TLOAD_A_FMT_SHIFT, t, base, stride);
}

MX_INLINE void mx_tload_b(const int t, const int fmt, const void *base, long stride) {
    MX_R_IRR(TLOAD_B, fmt << NAIAD_MI_TLOAD_B_FMT_SHIFT, t, base, stride);
}

/* From memory holding the transpose: K rows of M (for a) or N rows of K (for b). */
MX_INLINE void mx_tload_at(const int t, const int fmt, const void *base, long stride) {
    MX_R_IRR(TLOAD_A, (fmt << NAIAD_MI_TLOAD_A_FMT_SHIFT) | (1 << NAIAD_MI_TLOAD_A_T_SHIFT), t, base, stride);
}

MX_INLINE void mx_tload_bt(const int t, const int fmt, const void *base, long stride) {
    MX_R_IRR(TLOAD_B, (fmt << NAIAD_MI_TLOAD_B_FMT_SHIFT) | (1 << NAIAD_MI_TLOAD_B_T_SHIFT), t, base, stride);
}

MX_INLINE void mx_tstore_a(const int t, const int fmt, void *base, long stride) {
    MX_R_IRR(TSTORE_A, fmt << NAIAD_MI_TSTORE_A_FMT_SHIFT, t, base, stride);
}

MX_INLINE void mx_tstore_b(const int t, const int fmt, void *base, long stride) {
    MX_R_IRR(TSTORE_B, fmt << NAIAD_MI_TSTORE_B_FMT_SHIFT, t, base, stride);
}

/* td = ts's a view (b = 0) or b view (b = 1) shifted by n rows (cols = 0) or columns. */
MX_INLINE void mx_tshift(const int td, const int ts, const int b, const int cols, const int fmt, long n) {
    __asm__ volatile(".insn r %[op], %[f3], %[f7], x%[d], x%[s], %[n]"
                     :
                     : [op] "i"(NAIAD_MI_TSHIFT_OPCODE), [f3] "i"(NAIAD_MI_TSHIFT_FUNCT3),
                       [f7] "i"((fmt << NAIAD_MI_TSHIFT_FMT_SHIFT) | (cols << NAIAD_MI_TSHIFT_COLS_SHIFT) |
                                (b << NAIAD_MI_TSHIFT_B_SHIFT)),
                       [d] "i"(td), [s] "i"(ts), [n] "r"(n));
}

/* -- accumulators ----------------------------------------------------------- */

MX_INLINE void mx_aload(const int d, const int fmt, const void *base, long stride) {
    MX_R_IRR(ALOAD, fmt << NAIAD_MI_ALOAD_FMT_SHIFT, d, base, stride);
}

MX_INLINE void mx_astore(const int d, const int fmt, void *base, long stride) {
    MX_R_IRR(ASTORE, fmt << NAIAD_MI_ASTORE_FMT_SHIFT, d, base, stride);
}

MX_INLINE void mx_azero(const int d) { MX_R_III(AZERO, 0, d, 0, 0); }

/* bits: one acc-format element in the low bits (e.g. an fp32 bit pattern). */
/* R-type, rd = literal index, rs1 = x-register, rs2 reserved (x0). */
#define MX_R_IR0(insn, f7_, idx_, r1_)                                                                \
    __asm__ volatile(".insn r %[op], %[f3], %[f7], x%[i], %[a], x0"                                \
                     :                                                                             \
                     : [op] "i"(NAIAD_MI_##insn##_OPCODE), [f3] "i"(NAIAD_MI_##insn##_FUNCT3),     \
                       [f7] "i"(f7_), [i] "i"(idx_), [a] "r"(r1_))

MX_INLINE void mx_ascal(const int d, const int fmt, unsigned long bits) {
    MX_R_IR0(ASCAL, fmt << NAIAD_MI_ASCAL_FMT_SHIFT, d, bits);
}

MX_INLINE void mx_aadds(const int d, const int fmt, unsigned long bits) {
    MX_R_IR0(AADDS, fmt << NAIAD_MI_AADDS_FMT_SHIFT, d, bits);
}

/* -- compute ----------------------------------------------------------------- */

MX_INLINE void mx_mma(const int d, const int a, const int b, const int in, const int acc) {
    MX_R_III(MMA, (in << NAIAD_MI_MMA_IN_SHIFT) | (acc << NAIAD_MI_MMA_ACC_SHIFT), d, a, b);
}

MX_INLINE void mx_mma_minplus(const int d, const int a, const int b, const int in, const int acc) {
    MX_R_III(MMA_MINPLUS, (in << NAIAD_MI_MMA_MINPLUS_IN_SHIFT) | (acc << NAIAD_MI_MMA_MINPLUS_ACC_SHIFT), d, a, b);
}

MX_INLINE void mx_mma_maxplus(const int d, const int a, const int b, const int in, const int acc) {
    MX_R_III(MMA_MAXPLUS, (in << NAIAD_MI_MMA_MAXPLUS_IN_SHIFT) | (acc << NAIAD_MI_MMA_MAXPLUS_ACC_SHIFT), d, a, b);
}

/* -- configuration ------------------------------------------------------------- */

MX_INLINE unsigned long mx_mcfg(const int sel) {
    unsigned long v;
    __asm__ volatile(".insn i %[op], %[f3], %[rd], x0, %[sel]"
                     : [rd] "=r"(v)
                     : [op] "i"(NAIAD_MI_MCFG_OPCODE), [f3] "i"(NAIAD_MI_MCFG_FUNCT3), [sel] "i"(sel));
    return v;
}

/* Does the unit implement input / accumulator format `fmt` (a NAIAD_MFMT_* code)? */
MX_INLINE int mx_has_in_format(const int fmt) { return (mx_mcfg(NAIAD_MCFG_IN_FORMATS) >> fmt) & 1; }
MX_INLINE int mx_has_acc_format(const int fmt) { return (mx_mcfg(NAIAD_MCFG_ACC_FORMATS) >> fmt) & 1; }

#endif /* NAIAD_MATRIX_H */
