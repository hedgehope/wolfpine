# apsp/scalar size=64 fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: li a0, 0x0
10000070: auipc a1, 0x0
10000074: addi a1, a1, 0x190
10000078: addi a2, a1, 0x8
1000007c: rdcycle a3
10000080: li a4, 0x40
10000084: mv a5, a2
10000088: li a6, 0x0
1000008c: slli a7, a0, 0x8
10000090: add a7, a7, a1
10000094: addi a7, a7, 0x108
10000098: slli t0, a0, 0x2
1000009c: mv t1, a2
100000a0: slli t2, a6, 0x8
100000a4: add t2, a1, t2
100000a8: add t2, t2, t0
100000ac: flw fa5, 0x0(t2)
100000b0: mv t2, t1
100000b4: mv t3, a5
100000b8: flw fa4, -0x8(t2)
100000bc: flw fa3, -0x4(t2)
100000c0: flw fa2, 0x0(t2)
100000c4: flw fa1, 0x4(t2)
100000c8: flw fa0, -0x8(t3)
100000cc: flw ft0, -0x4(t3)
100000d0: flw ft1, 0x0(t3)
100000d4: flw ft2, 0x4(t3)
100000d8: addi t3, t3, 0x10
100000dc: fadd.s fa0, fa5, fa0
100000e0: fadd.s ft0, fa5, ft0
100000e4: fadd.s ft1, fa5, ft1
100000e8: fadd.s ft2, fa5, ft2
100000ec: fmin.s fa4, fa4, fa0
100000f0: fmin.s fa3, fa3, ft0
100000f4: fmin.s fa2, fa2, ft1
100000f8: fmin.s fa1, fa1, ft2
100000fc: fsw fa4, -0x8(t2)
10000100: fsw fa3, -0x4(t2)
10000104: fsw fa2, 0x0(t2)
10000108: fsw fa1, 0x4(t2)
1000010c: addi t2, t2, 0x10
10000110: bne t3, a7, 0x100000b8 <main+0x4c>
10000114: addi a6, a6, 0x1
10000118: addi t1, t1, 0x100
1000011c: bne a6, a4, 0x100000a0 <main+0x34>
10000120: addi a0, a0, 0x1
10000124: addi a5, a5, 0x100
10000128: bne a0, a4, 0x10000088 <main+0x1c>
1000012c: rdcycle a0
10000130: sub a0, a0, a3
10000134: auipc a1, 0x4
10000138: sd a0, 0xcc(a1)
1000013c: li a0, 0x0
10000140: ret
