# jacobi/scalar size=66 fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x40
10000070: sd s0, 0x38(sp)
10000074: sd s1, 0x30(sp)
10000078: sd s2, 0x28(sp)
1000007c: sd s3, 0x20(sp)
10000080: sd s4, 0x18(sp)
10000084: sd s5, 0x10(sp)
10000088: sd s6, 0x8(sp)
1000008c: li a0, 0x0
10000090: auipc a2, 0xd
10000094: auipc a3, 0x4
10000098: auipc a4, 0x0
1000009c: lui a1, 0x4
100000a0: addi a2, a2, -0x150
100000a4: addi a3, a3, 0x66c
100000a8: addi a4, a4, 0x228
100000ac: addiw a5, a1, 0x410
100000b0: add a6, a4, a0
100000b4: lw a6, 0x0(a6)
100000b8: add a7, a3, a1
100000bc: addi a0, a0, 0x4
100000c0: sw a6, 0x410(a7)
100000c4: sw a6, 0x0(a3)
100000c8: sw a6, 0x0(a2)
100000cc: addi a2, a2, 0x4
100000d0: addi a3, a3, 0x4
100000d4: bne a0, a5, 0x100000b0 <main+0x44>
100000d8: li a0, 0x0
100000dc: li a4, 0x0
100000e0: lui a7, 0x4
100000e4: li a2, 0x1
100000e8: rdcycle a1
100000ec: auipc t0, 0x4
100000f0: auipc t1, 0xd
100000f4: li a3, 0x108
100000f8: lui t2, 0x3e800
100000fc: li a5, 0x7
10000100: li a6, 0x41
10000104: addiw a7, a7, 0x410
10000108: addi t0, t0, 0x614
1000010c: addi t1, t1, -0xa4
10000110: fmv.w.x fa5, t2
10000114: li t2, 0x8
10000118: j 0x10000128 <main+0xbc>
1000011c: addi a2, a2, 0x1
10000120: addi a0, a0, 0x1
10000124: beq a4, t2, 0x100001d8 <main+0x16c>
10000128: li t3, 0x0
1000012c: mv t4, a4
10000130: andi t6, a0, 0x1
10000134: andi s0, a2, 0x1
10000138: addiw a4, a4, 0x1
1000013c: mv t5, t1
10000140: mul s1, t6, a7
10000144: mul s0, s0, a7
10000148: addi t6, s1, 0x210
1000014c: add s0, t0, s0
10000150: add s1, t0, s1
10000154: addi s0, s0, 0x10c
10000158: addi s1, s1, 0x110
1000015c: li s2, 0x1
10000160: j 0x1000017c <main+0x110>
10000164: addi s2, s2, 0x1
10000168: addi s0, s0, 0x108
1000016c: addi t5, t5, 0x108
10000170: addi s1, s1, 0x108
10000174: addi t3, t3, 0x1
10000178: beq s2, a6, 0x1000011c <main+0xb0>
1000017c: mul s3, t3, a3
10000180: add s3, t6, s3
10000184: add s3, t0, s3
10000188: mv s4, s1
1000018c: mv s5, t5
10000190: mv s6, s0
10000194: j 0x100001ac <main+0x140>
10000198: fsw fa4, 0x0(s6)
1000019c: addi s6, s6, 0x4
100001a0: addi s4, s4, 0x4
100001a4: addi s5, s5, 0x4
100001a8: beq s4, s3, 0x10000164 <main+0xf8>
100001ac: flw fa4, -0x10c(s4)
100001b0: flw fa3, 0x104(s4)
100001b4: flw fa2, -0x8(s4)
100001b8: flw fa1, 0x0(s4)
100001bc: fadd.s fa4, fa4, fa3
100001c0: fadd.s fa4, fa4, fa2
100001c4: fadd.s fa4, fa4, fa1
100001c8: fmul.s fa4, fa4, fa5
100001cc: bne t4, a5, 0x10000198 <main+0x12c>
100001d0: fsw fa4, 0x0(s5)
100001d4: j 0x1000019c <main+0x130>
100001d8: rdcycle a0
100001dc: sub a0, a0, a1
100001e0: auipc a1, 0x11
100001e4: sd a0, 0x1a0(a1)
100001e8: li a0, 0x0
100001ec: ld s0, 0x38(sp)
100001f0: ld s1, 0x30(sp)
100001f4: ld s2, 0x28(sp)
100001f8: ld s3, 0x20(sp)
100001fc: ld s4, 0x18(sp)
10000200: ld s5, 0x10(sp)
10000204: ld s6, 0x8(sp)
10000208: addi sp, sp, 0x40
1000020c: ret
