/*
 * ztt.h: .insn wrappers for the RISC-V AME TG's Ztt v0.6 draft (matrix: {mode: ztt}).
 *
 * Every Ztt instruction is custom-1 (0x2b), funct3 000, selected by funct7 and, for the R2 / R1
 * formats, by the rs2 (and rs1) slots. The match words are the design's naiad.h
 * (NAIAD_ZTT_<NAME>_MATCH, from framework/matrix/ztt_encodings.yaml, imported from the draft's
 * INSTRUCTION_ENCODINGS.md); clang 20 has no Ztt mnemonics.
 *
 * M, Acc and datatype-register operands are LITERAL register numbers (ztt_mls_rm(0, p): m0),
 * not compiler-allocated values; X-register operands are ordinary C values.
 *
 *   ztt_enable()                        mstatus.MS = Initial, then ame.acquire (TRY); returns its rd
 *   ztt_acquire(desc) / ztt_release()   ame.acquire rd, rs1 / ame.release
 *   ztt_msettyp(md, desc) / ztt_asettyp(ad, desc), ztt_mgettyp(ms) / ztt_agettyp(ad)
 *   ztt_mls_rm / _cm / _1r (md, ptr), ztt_mls_st / _tst (md, ptr, stride); ztt_mss_* (ms, ...)
 *   ztt_mzero_m(md) / ztt_mzero_acc(acc)
 *   ztt_mma(NAME, acc, ms1, ms2)        NAME: MMULACC_2D, MMULACCNEG_2D, MMULATACC_2D, ...
 *   ztt_r3(NAME, md, ms1, ms2)          any all-M R3 instruction (MADD_EW, MSUB_EW, MMAX_EW, ...)
 *   ztt_r2(NAME, md, ms1)               any all-register R2 instruction (MCONV_EW, MREDUCEADD_ROW,
 *                                       MMOV_M_A (md, acc), MMOV_A_M (acc, ms), MMOV_M_M, ...)
 *   ztt_ewx(NAME, md, x, ms2)           a .ew.x form (MADD_EW_X, ...; the scalar's type is amestype)
 *   ztt_mbcast(md, x, desc)             mbcast.m.x: x as a desc-typed scalar, into every element
 *   ztt_mmove_m_x(E, md, value, pos) / ztt_mmove_x_m(E, ms, pos)   one E-bit element, E = 8..64
 *   ztt_csrr(NAME) / ztt_csrw(NAME, v)  NAME: AMENLEN, AMEUDSZ, AMESTYPE, AMEOWN, AMEFFLAGS, AMEXSAT,
 *                                       AMESTATUS
 *   ZTT_UN()                            amestatus.UN: the last unsupported tuple since it was cleared
 */
#ifndef NAIAD_ZTT_H
#define NAIAD_ZTT_H

#include <stdint.h>

#include "naiad.h"

#ifndef NAIAD_HAS_ZTT
#error "this design has no Ztt unit (matrix: {mode: ztt} in the design file)"
#endif

#define ZTT_INLINE static inline __attribute__((always_inline, unused))
#define ZTT_STR_(x) #x
#define ZTT_STR(x) ZTT_STR_(x)
#define ZTT_M(NAME) NAIAD_ZTT_##NAME##_MATCH
#define ZTT_F7(m) ((((unsigned)(m)) >> 25) & 0x7fu)
#define ZTT_F20(m) ((((unsigned)(m)) >> 20) & 0x1fu)
#define ZTT_F15(m) ((((unsigned)(m)) >> 15) & 0x1fu)

/* A whole word as one .insn (the operand is printed signed, hence the mask). */
#define ZTT_WORD_(w) __asm__ volatile(".insn (%0) & 0xffffffff" ::"i"((unsigned)(w)) : "memory")
#define ZTT_RRR_(m, d, s1, s2)                                                                     \
    ZTT_WORD_((m) | ((unsigned)(d) << 7) | ((unsigned)(s1) << 15) | ((unsigned)(s2) << 20))

#define ztt_r3(NAME, md, ms1, ms2) ZTT_RRR_(ZTT_M(NAME), md, ms1, ms2)
#define ztt_r2(NAME, md, ms1) ZTT_RRR_(ZTT_M(NAME), md, ms1, 0)
#define ztt_mma(NAME, acc, ms1, ms2) ZTT_RRR_(ZTT_M(NAME), acc, ms1, ms2)
#define ztt_mzero_m(md) ZTT_RRR_(ZTT_M(MZERO_2D_M), md, 0, 0)
#define ztt_mzero_acc(acc) ZTT_RRR_(ZTT_M(MZERO_2D_ACC), acc, 0, 0)
#define ztt_release() ZTT_WORD_(ZTT_M(AME_RELEASE))

/* R2 with a literal destination field and an X source in rs1 (rs2 holds the selector). */
#define ZTT_R2_X_(m_, dst_, x_)                                                                         \
    __asm__ volatile(".insn r 0x2b, 0, %[f7], x%[d], %[rs1], x%[f20]"                              \
                     :                                                                             \
                     : [f7] "i"(ZTT_F7(m_)), [f20] "i"(ZTT_F20(m_)), [d] "i"(dst_),                \
                       [rs1] "r"((unsigned long)(x_))                                              \
                     : "memory")
/* R2 with an X destination and a literal source field. */
#define ZTT_R2_RD_(m_, src_)                                                                          \
    ({                                                                                             \
        unsigned long ztt_r_;                                                                      \
        __asm__ volatile(".insn r 0x2b, 0, %[f7], %[rd], x%[s1], x%[f20]"                          \
                         : [rd] "=r"(ztt_r_)                                                       \
                         : [f7] "i"(ZTT_F7(m_)), [f20] "i"(ZTT_F20(m_)), [s1] "i"(src_)            \
                         : "memory");                                                              \
        ztt_r_;                                                                                    \
    })
/* R3 with a literal destination and two X sources. */
#define ZTT_R3_XX_(m_, dst_, x1_, x2_)                                                                   \
    __asm__ volatile(".insn r 0x2b, 0, %[f7], x%[d], %[rs1], %[rs2]"                               \
                     :                                                                             \
                     : [f7] "i"(ZTT_F7(m_)), [d] "i"(dst_), [rs1] "r"((unsigned long)(x1_)),       \
                       [rs2] "r"((unsigned long)(x2_))                                             \
                     : "memory")

#define ztt_msettyp(md, desc) ZTT_R2_X_(ZTT_M(MSETTYP), md, desc)
#define ztt_asettyp(ad, desc) ZTT_R2_X_(ZTT_M(ASETTYP), ad, desc)
#define ztt_mgettyp(ms) ZTT_R2_RD_(ZTT_M(MGETTYP), ms)
#define ztt_agettyp(ad) ZTT_R2_RD_(ZTT_M(AGETTYP), ad)

#define ztt_mls_rm(md, p) ZTT_R2_X_(ZTT_M(MLS_RM), md, p)
#define ztt_mls_cm(md, p) ZTT_R2_X_(ZTT_M(MLS_CM), md, p)
#define ztt_mls_1r(md, p) ZTT_R2_X_(ZTT_M(MLS_1R), md, p)
#define ztt_mss_rm(ms, p) ZTT_R2_X_(ZTT_M(MSS_RM), ms, p)
#define ztt_mss_cm(ms, p) ZTT_R2_X_(ZTT_M(MSS_CM), ms, p)
#define ztt_mss_1r(ms, p) ZTT_R2_X_(ZTT_M(MSS_1R), ms, p)
#define ztt_mls_st(md, p, stride) ZTT_R3_XX_(ZTT_M(MLS_ST), md, p, stride)
#define ztt_mls_tst(md, p, stride) ZTT_R3_XX_(ZTT_M(MLS_TST), md, p, stride)
#define ztt_mss_st(ms, p, stride) ZTT_R3_XX_(ZTT_M(MSS_ST), ms, p, stride)
#define ztt_mss_tst(ms, p, stride) ZTT_R3_XX_(ZTT_M(MSS_TST), ms, p, stride)
/* mbcast.m.x md, rs1 (the scalar), rs2 (its type) */
#define ztt_mbcast(md, x, desc) ZTT_R3_XX_(ZTT_M(MBCAST_M_X), md, x, desc)

/* .ew.x: md, rs1 (the scalar, typed by amestype), ms2 */
#define ztt_ewx(NAME, md, x, ms2)                                                                  \
    __asm__ volatile(".insn r 0x2b, 0, %[f7], x%[d], %[rs1], x%[s2]"                               \
                     :                                                                             \
                     : [f7] "i"(ZTT_F7(ZTT_M(NAME))), [d] "i"(md), [s2] "i"(ms2),                  \
                       [rs1] "r"((unsigned long)(x))                                               \
                     : "memory")

/* mmove<E>.m.x md, rs2 (the value: bits 19:15), rs1 (the position: bits 24:20) */
#define ztt_mmove_m_x(E, md, value, pos)                                                           \
    __asm__ volatile(".insn r 0x2b, 0, %[f7], x%[d], %[v], %[p]"                                   \
                     :                                                                             \
                     : [f7] "i"(ZTT_F7(ZTT_M(MMOVE##E##_M_X))), [d] "i"(md),                       \
                       [v] "r"((unsigned long)(value)), [p] "r"((unsigned long)(pos))              \
                     : "memory")
/* mmove<E>.x.m rd, ms2 (bits 19:15), rs1 (the position: bits 24:20) */
#define ztt_mmove_x_m(E, ms, pos)                                                                  \
    ({                                                                                             \
        unsigned long ztt_r_;                                                                      \
        __asm__ volatile(".insn r 0x2b, 0, %[f7], %[rd], x%[s], %[p]"                              \
                         : [rd] "=r"(ztt_r_)                                                       \
                         : [f7] "i"(ZTT_F7(ZTT_M(MMOVE##E##_X_M))), [s] "i"(ms),                   \
                           [p] "r"((unsigned long)(pos))                                           \
                         : "memory");                                                              \
        ztt_r_;                                                                                    \
    })

#define ztt_csrr(NAME)                                                                             \
    ({                                                                                             \
        unsigned long ztt_v_;                                                                      \
        __asm__ volatile("csrr %0, " ZTT_STR(NAIAD_ZTT_CSR_##NAME) : "=r"(ztt_v_));                \
        ztt_v_;                                                                                    \
    })
#define ztt_csrw(NAME, v)                                                                          \
    __asm__ volatile("csrw " ZTT_STR(NAIAD_ZTT_CSR_##NAME) ", %0" ::"r"((unsigned long)(v)) : "memory")
#define ZTT_UN() (ztt_csrr(AMESTATUS) & 1u)

ZTT_INLINE unsigned long ztt_acquire(unsigned long desc) {
    unsigned long r;
    __asm__ volatile(".insn r 0x2b, 0, %[f7], %[rd], %[rs1], x%[f20]"
                     : [rd] "=r"(r)
                     : [f7] "i"(ZTT_F7(ZTT_M(AME_ACQUIRE))), [f20] "i"(ZTT_F20(ZTT_M(AME_ACQUIRE))),
                       [rs1] "r"(desc)
                     : "memory");
    return r;
}

/* mstatus.MS = Initial (its position is Naiad's: Ztt v0.6 gives none), then a TRY acquire. */
ZTT_INLINE unsigned long ztt_enable(void) {
    __asm__ volatile("csrs mstatus, %0" ::"r"(1ul << NAIAD_ZTT_MS_SHIFT) : "memory");
    return ztt_acquire(0);
}

#endif /* NAIAD_ZTT_H */
