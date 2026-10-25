/*
 * vme.h: .insn wrappers for the Naiad vector-coupled matrix mode (VME: the Zvt v0.4 draft,
 * "Zvt Family of Vector Matrix Extensions", 2026-08-31).
 *
 * Zvt keeps C in accumulator tiles (new state) and takes A and B from vector register groups,
 * one LMUL group per K-row, rows 8/KMAX registers apart. The match words come from the
 * design's naiad.h (NAIAD_VME_*_MATCH, generated from the vme: section of
 * framework/matrix/encodings.yaml). Clang 20 knows no Zvt, so every instruction is a .insn.
 *
 * Vector register operands are LITERAL register numbers (vme_vtfmm(4, 8, 12): tile mt4,
 * A = v8, B = v12), not compiler-allocated values: msetmtype[i] sets vtype itself, and any
 * vset{i}vl{i} (which an RVV intrinsic may emit) writes mtype = 0, so a VME region must not
 * contain RVV intrinsics. Load operands with vme_vle / store with vme_vse (plain vleN.v /
 * vseN.v under the vtype msetmtype installed) instead.
 *
 *   vme_msetmtypei(sew, alt, twiden)   mtype.mtwiden = twiden, vtype = eSEW[alt]; vl = tm = tk = 0
 *   vme_msettn(n) / vme_msettm(n) / vme_msettk(n)   clamp and set tn (= vl) / tm / tk; return it
 *   vme_mtype()                        read the mtype CSR
 *   vme_vtle(eew, base, tss) / vme_vtse(eew, base, tss)   one tile row / column <-> memory
 *   vme_vtmv_vt(vd, tss) / vme_vtmv_tv(tss, vs2)          one tile row / column <-> vector group
 *   vme_vtzero(tile)                   zero the tm x tn sub-tile (TEW = SEW)
 *   vme_vtfmm(tile, vs2, vs1), vme_vtfmm_alt, vme_vtmmu, vme_vtmms
 *                                      C[tile][tm, tn] += A(vs2)[tk, tm]^T . B(vs1)[tk, tn]
 *   vme_vtdiscard()
 *   vme_vle(eew, vd, ptr) / vme_vse(eew, vs, ptr)          vleEEW.v / vseEEW.v on a literal register
 *   vme_vlre(nf, eew, vd, ptr)          vl<nf>re<eew>.v: nf whole registers from vd (vtype-free)
 *   VME_TSS(tile, col, index)          a Tile Subset Specifier (col: 0 row, 1 column)
 */
#ifndef NAIAD_VME_H
#define NAIAD_VME_H

#include <stdint.h>

#include "naiad.h"

#ifndef NAIAD_HAS_VME
#error "this design has no VME unit (matrix: {mode: vme} in the design file)"
#endif
#ifndef __riscv_vector
#error "vme.h needs the vector extension in -march (the design's NAIAD_MARCH has it)"
#endif

#define VME_INLINE static inline __attribute__((always_inline, unused))
#define VME_STR_(x) #x
#define VME_STR(x) VME_STR_(x)

#define VME_SEW_CODE(sew) ((sew) == 8 ? 0u : (sew) == 16 ? 1u : (sew) == 32 ? 2u : 3u)
#define VME_MTW_CODE(tw) ((tw) == 1 ? 1u : (tw) == 2 ? 2u : 3u)
#define VME_EEE(eew) VME_SEW_CODE(eew)
#define VME_F7(m) ((((unsigned)(m)) >> 25) & 0x7fu)
#define VME_RS2F(m) ((((unsigned)(m)) >> 20) & 0x1fu)

#define VME_TSS(tile, col, index)                                                                  \
    (((unsigned long)(tile) << 27) | ((unsigned long)(col) << 24) | (unsigned long)(index))

/* A whole word as one .insn (the operand is printed signed, hence the mask). */
#define VME_WORD_(w) __asm__ volatile(".insn (%0) & 0xffffffff" ::"i"((unsigned)(w)) : "memory")

#define vme_msetmtypei(sew, alt, tw)                                                               \
    VME_WORD_(NAIAD_VME_MSETMTYPEI_MATCH | ((unsigned)(alt) << 28) | (VME_SEW_CODE(sew) << 23) |   \
              (VME_MTW_CODE(tw) << 15))

#define VME_SETT_(m, n)                                                                            \
    ({                                                                                             \
        unsigned long vme_r_;                                                                      \
        __asm__ volatile(".insn r 0x57, 7, %[f7], %[rd], %[rs1], x%[r2]"                           \
                         : [rd] "=r"(vme_r_)                                                       \
                         : [f7] "i"(VME_F7(m)), [r2] "i"(VME_RS2F(m)), [rs1] "r"((unsigned long)(n)) \
                         : "memory");                                                              \
        vme_r_;                                                                                    \
    })
#define vme_msettn(n) VME_SETT_(NAIAD_VME_MSETTN_MATCH, n)
#define vme_msettm(n) VME_SETT_(NAIAD_VME_MSETTM_MATCH, n)
#define vme_msettk(n) VME_SETT_(NAIAD_VME_MSETTK_MATCH, n)

VME_INLINE unsigned long vme_mtype(void) {
    unsigned long v;
    __asm__ volatile("csrr %0, " VME_STR(NAIAD_VME_CSR_MTYPE) : "=r"(v));
    return v;
}

#define VME_TILE_MEM_(m, eew, base, tss)                                                           \
    __asm__ volatile(".insn r %[op], 7, %[f7], x0, %[rs1], %[rs2]"                                 \
                     :                                                                             \
                     : [op] "i"((m) & 0x7f), [f7] "i"(VME_F7(m) | (VME_EEE(eew) << 4)),             \
                       [rs1] "r"(base), [rs2] "r"((unsigned long)(tss))                            \
                     : "memory")
#define vme_vtle(eew, base, tss) VME_TILE_MEM_(NAIAD_VME_VTLE_MATCH, eew, base, tss)
#define vme_vtse(eew, base, tss) VME_TILE_MEM_(NAIAD_VME_VTSE_MATCH, eew, base, tss)

#define vme_vtmv_vt(vd, tss)                                                                       \
    __asm__ volatile(".insn r 0x57, 6, %[f7], x" VME_STR(vd) ", %[rs1], x%[r2]"                    \
                     :                                                                             \
                     : [f7] "i"(VME_F7(NAIAD_VME_VTMV_V_T_MATCH)),                                 \
                       [r2] "i"(VME_RS2F(NAIAD_VME_VTMV_V_T_MATCH)), [rs1] "r"((unsigned long)(tss)) \
                     : "memory")
#define vme_vtmv_tv(tss, vs2)                                                                      \
    __asm__ volatile(".insn r 0x57, 6, %[f7], x0, %[rs1], x" VME_STR(vs2)                          \
                     :                                                                             \
                     : [f7] "i"(VME_F7(NAIAD_VME_VTMV_T_V_MATCH)), [rs1] "r"((unsigned long)(tss)) \
                     : "memory")

#define vme_vtzero(tile) VME_WORD_(NAIAD_VME_VTZERO_MATCH | ((unsigned)(tile) << 8))
#define vme_vtdiscard() VME_WORD_(NAIAD_VME_VTDISCARD_MATCH)

/* rd[4:2] = t[3:1]; the multiply's tile must be a multiple of 2 (of 4 at TEW 32). */
#define VME_MMA_(m, tile, vs2, vs1)                                                                \
    VME_WORD_((m) | (((unsigned)(tile) >> 1) << 9) | ((unsigned)(vs2) << 20) | ((unsigned)(vs1) << 15))
#define vme_vtfmm(tile, vs2, vs1) VME_MMA_(NAIAD_VME_VTFMM_TVV_MATCH, tile, vs2, vs1)
#define vme_vtfmm_alt(tile, vs2, vs1) VME_MMA_(NAIAD_VME_VTFMM_ALT_TVV_MATCH, tile, vs2, vs1)
#define vme_vtmmu(tile, vs2, vs1) VME_MMA_(NAIAD_VME_VTMMU_TVV_MATCH, tile, vs2, vs1)
#define vme_vtmms(tile, vs2, vs1) VME_MMA_(NAIAD_VME_VTMMS_TVV_MATCH, tile, vs2, vs1)

#define vme_vle(eew, vd, ptr)                                                                      \
    __asm__ volatile("vle" VME_STR(eew) ".v v" VME_STR(vd) ", (%0)" ::"r"(ptr) : "memory")
#define vme_vse(eew, vs, ptr)                                                                      \
    __asm__ volatile("vse" VME_STR(eew) ".v v" VME_STR(vs) ", (%0)" ::"r"(ptr) : "memory")
/* vl<nf>re<eew>.v: nf whole registers from vd (nf = 1, 2, 4, 8); independent of vtype and vl, so
   it loads several operand groups (or K steps of them) in one scalar-addressed instruction. */
#define vme_vlre(nf, eew, vd, ptr)                                                                 \
    __asm__ volatile("vl" VME_STR(nf) "re" VME_STR(eew) ".v v" VME_STR(vd) ", (%0)" ::"r"(ptr) : "memory")

#endif /* NAIAD_VME_H */
