// riscv_test.h for running riscv-vector-tests on Naiad (and Spike).
//
// Stands in for riscv-test-env's "p" environment, which Naiad cannot run: p
// boots through mtvec / satp / PMP set-up and an mret into the test, and ends
// with an ecall, but Naiad models no traps or privilege modes. This one stays
// in M-mode from reset and talks HTIF directly through tohost, which Spike and
// Naiad both speak, so the same ELF runs on both:
//
//   pass:        tohost = 1                    (exit code 0)
//   fail check N: tohost = (N << 1) | 1         (exit code N, the TEST_CASE number)
//
// Every tohost store first waits for the previous command to be consumed
// (Spike polls tohost only every few thousand instructions).
#ifndef _ENV_NAIAD_VECTOR_H
#define _ENV_NAIAD_VECTOR_H

#define NAIAD_MSTATUS_FS_INITIAL (1 << 13)
#define NAIAD_MSTATUS_VS_INITIAL (1 << 9)

#define RVTEST_RV64U .macro init; .endm
#define RVTEST_RV32U .macro init; .endm

#define RVTEST_FP_ENABLE                                                \
  li a0, NAIAD_MSTATUS_FS_INITIAL;                                      \
  csrs mstatus, a0;                                                     \
  csrwi fcsr, 0

#define RVTEST_VECTOR_ENABLE                                            \
  li a0, NAIAD_MSTATUS_VS_INITIAL | NAIAD_MSTATUS_FS_INITIAL;           \
  csrs mstatus, a0;                                                     \
  csrwi fcsr, 0;                                                        \
  csrwi vcsr, 0;

#define RVTEST_ZVE32X_ENABLE                                            \
  li a0, NAIAD_MSTATUS_VS_INITIAL;                                      \
  csrs mstatus, a0;                                                     \
  csrwi vcsr, 0;

#define RVTEST_RV64UF .macro init; RVTEST_FP_ENABLE; .endm
#define RVTEST_RV64UV .macro init; RVTEST_VECTOR_ENABLE; .endm
#define RVTEST_RV64UVX .macro init; RVTEST_ZVE32X_ENABLE; .endm
#define RVTEST_RV32UF RVTEST_RV64UF
#define RVTEST_RV32UV RVTEST_RV64UV
#define RVTEST_RV32UVX RVTEST_RV64UVX

#define INIT_XREG                                                       \
  li x1, 0;  li x2, 0;  li x3, 0;  li x4, 0;  li x5, 0;  li x6, 0;      \
  li x7, 0;  li x8, 0;  li x9, 0;  li x10, 0; li x11, 0; li x12, 0;     \
  li x13, 0; li x14, 0; li x15, 0; li x16, 0; li x17, 0; li x18, 0;     \
  li x19, 0; li x20, 0; li x21, 0; li x22, 0; li x23, 0; li x24, 0;     \
  li x25, 0; li x26, 0; li x27, 0; li x28, 0; li x29, 0; li x30, 0;     \
  li x31, 0;

#define TESTNUM gp

/* Send the HTIF command in a0; never returns. Inlined (stage-2 tests can be
 * larger than a jal's reach). */
#define NAIAD_WRITE_TOHOST                                              \
        la t5, tohost;                                                  \
8:      ld t6, 0(t5);                                                   \
        bnez t6, 8b;                                                    \
        sd a0, 0(t5);                                                   \
9:      j 9b

#define RVTEST_CODE_BEGIN                                               \
        .section .text.init;                                            \
        .align  6;                                                      \
        .globl _start;                                                  \
_start:                                                                 \
        j reset_vector;                                                 \
reset_vector:                                                           \
        INIT_XREG;                                                      \
        li TESTNUM, 0;                                                  \
        init;                                                           \
1:

#define RVTEST_CODE_END                                                 \
        unimp

#define RVTEST_PASS                                                     \
        fence;                                                          \
        li TESTNUM, 1;                                                  \
        li a0, 1;                                                       \
        NAIAD_WRITE_TOHOST

#define RVTEST_FAIL                                                     \
        fence;                                                          \
1:      beqz TESTNUM, 1b;                                               \
        sll TESTNUM, TESTNUM, 1;                                        \
        or TESTNUM, TESTNUM, 1;                                         \
        addi a0, TESTNUM, 0;                                            \
        NAIAD_WRITE_TOHOST

#define EXTRA_DATA

#define RVTEST_DATA_BEGIN                                               \
        EXTRA_DATA                                                      \
        .pushsection .tohost,"aw",@progbits;                            \
        .align 6; .global tohost; tohost: .dword 0; .size tohost, 8;    \
        .align 6; .global fromhost; fromhost: .dword 0; .size fromhost, 8;\
        .popsection;                                                    \
        .align 4; .global begin_signature; begin_signature:

#define RVTEST_DATA_END .align 4; .global end_signature; end_signature:

#endif
