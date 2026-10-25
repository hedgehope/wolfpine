# nbody/scalar size=[256, "clustered"] fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: auipc a1, 0x1
10000070: auipc a2, 0x0
10000074: rdcycle a0
10000078: addi a4, a1, -0x22c
1000007c: addi a1, a2, 0x198
10000080: addi a2, a4, 0x8
10000084: addi a3, a1, 0x8
10000088: addi a4, a4, 0x7ff
1000008c: addi a4, a4, 0x409
10000090: flw fa5, -0x8(a2)
10000094: flw fa4, -0x4(a2)
10000098: flw fa3, 0x0(a2)
1000009c: addi a2, a2, 0xc
100000a0: fsw fa5, -0x8(a3)
100000a4: fsw fa4, -0x4(a3)
100000a8: fsw fa3, 0x0(a3)
100000ac: addi a3, a3, 0xc
100000b0: bne a2, a4, 0x10000090 <main+0x24>
100000b4: li a2, 0x0
100000b8: auipc a3, 0x42
100000bc: auipc a6, 0x2
100000c0: addi a4, a1, 0x8
100000c4: fmv.w.x fa5, zero
100000c8: li a5, 0x100
100000cc: addi a3, a3, 0x588
100000d0: addi a6, a6, -0x67c
100000d4: flw fa4, 0x40(a3)
100000d8: lui a7, 0x3f800
100000dc: fmv.w.x fa3, a7
100000e0: auipc a7, 0x42
100000e4: addi a7, a7, -0x6a0
100000e8: j 0x10000108 <main+0x9c>
100000ec: add t0, a7, t0
100000f0: addi a2, a2, 0x1
100000f4: fsw fa2, 0x0(t0)
100000f8: fsw fa1, 0x4(t0)
100000fc: fsw fa0, 0x8(t0)
10000100: addi a6, a6, 0x400
10000104: beq a2, a5, 0x100001a0 <main+0x134>
10000108: slli t0, a2, 0x2
1000010c: slli t1, a2, 0x4
10000110: sub t0, t1, t0
10000114: add t1, a1, t0
10000118: flw ft0, 0x0(t1)
1000011c: flw ft1, 0x4(t1)
10000120: flw ft2, 0x8(t1)
10000124: mv t1, a4
10000128: mv t2, a6
1000012c: li t3, 0x100
10000130: fmv.s fa2, fa5
10000134: fmv.s fa1, fa5
10000138: fmv.s fa0, fa5
1000013c: j 0x10000150 <main+0xe4>
10000140: addi t3, t3, -0x1
10000144: addi t2, t2, 0x4
10000148: addi t1, t1, 0xc
1000014c: beqz t3, 0x100000ec <main+0x80>
10000150: flw ft3, -0x8(t1)
10000154: flw ft4, -0x4(t1)
10000158: flw ft5, 0x0(t1)
1000015c: add t4, a2, t3
10000160: fsub.s ft3, ft3, ft0
10000164: fsub.s ft4, ft4, ft1
10000168: fsub.s ft5, ft5, ft2
1000016c: fmul.s ft6, ft4, ft4
10000170: fmadd.s ft6, ft3, ft3, ft6
10000174: fmadd.s ft6, ft5, ft5, ft6
10000178: fsw ft6, 0x0(t2)
1000017c: beq t4, a5, 0x10000140 <main+0xd4>
10000180: fadd.s ft6, ft6, fa4
10000184: fsqrt.s ft7, ft6
10000188: fmul.s ft6, ft6, ft7
1000018c: fdiv.s ft6, fa3, ft6
10000190: fmadd.s fa2, ft6, ft3, fa2
10000194: fmadd.s fa1, ft6, ft4, fa1
10000198: fmadd.s fa0, ft6, ft5, fa0
1000019c: j 0x10000140 <main+0xd4>
100001a0: rdcycle a1
100001a4: sub a1, a1, a0
100001a8: sd a1, 0x0(a3)
100001ac: li a0, 0x0
100001b0: ret
