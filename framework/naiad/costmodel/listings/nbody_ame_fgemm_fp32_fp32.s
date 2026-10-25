# nbody/ame_fgemm size=[256, "clustered"] fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x160
10000070: sd ra, 0x158(sp)
10000074: sd s0, 0x150(sp)
10000078: sd s1, 0x148(sp)
1000007c: sd s2, 0x140(sp)
10000080: sd s3, 0x138(sp)
10000084: sd s4, 0x130(sp)
10000088: sd s5, 0x128(sp)
1000008c: sd s6, 0x120(sp)
10000090: sd s7, 0x118(sp)
10000094: sd s8, 0x110(sp)
10000098: sd s9, 0x108(sp)
1000009c: sd s10, 0x100(sp)
100000a0: sd s11, 0xf8(sp)
100000a4: fsd fs0, 0xf0(sp)
100000a8: fsd fs1, 0xe8(sp)
100000ac: auipc a0, 0x4
100000b0: lui a2, 0x1
100000b4: auipc a3, 0x4e
100000b8: auipc a4, 0xd
100000bc: auipc a5, 0x0
100000c0: auipc a6, 0x9
100000c4: rdcycle a1
100000c8: sd a1, 0x0(sp)
100000cc: lui a7, 0xc0000
100000d0: lui a1, 0x3f800
100000d4: addi s1, a0, 0x45c
100000d8: addi t0, a3, 0x48c
100000dc: addi a0, a4, -0x3b0
100000e0: addi a5, a5, 0x44c
100000e4: addi s3, a6, -0x7b8
100000e8: fmv.w.x fa5, a7
100000ec: add a2, s1, a2
100000f0: addi a3, t0, 0x8
100000f4: addi a4, a0, 0x8
100000f8: sd a5, 0x10(sp)
100000fc: addi a5, a5, 0x8
10000100: addi a6, s3, 0x8
10000104: addi a7, s1, 0x7ff
10000108: addi t0, t0, 0x7ff
1000010c: addi a7, a7, 0x1
10000110: addi t0, t0, 0x409
10000114: flw fa4, -0x8(a3)
10000118: flw fa3, -0x4(a3)
1000011c: flw fa2, 0x0(a3)
10000120: sw a1, 0x8(a5)
10000124: addi a3, a3, 0xc
10000128: fmul.s fa1, fa3, fa3
1000012c: fsw fa4, -0x8(a4)
10000130: fsw fa3, -0x4(a4)
10000134: fsw fa2, 0x0(a4)
10000138: fmul.s fa0, fa4, fa5
1000013c: fmul.s ft0, fa3, fa5
10000140: fmul.s ft1, fa2, fa5
10000144: fsw fa4, -0x800(a7)
10000148: fsw fa3, -0x400(a7)
1000014c: fsw fa2, 0x0(a7)
10000150: sw a1, -0x400(a2)
10000154: fmadd.s fa1, fa4, fa4, fa1
10000158: fsw fa4, -0x8(a6)
1000015c: fsw fa3, -0x4(a6)
10000160: fsw fa2, 0x0(a6)
10000164: sw a1, 0x4(a6)
10000168: addi a7, a7, 0x4
1000016c: addi a4, a4, 0xc
10000170: fmadd.s fa4, fa2, fa2, fa1
10000174: fsw fa0, -0x8(a5)
10000178: fsw ft0, -0x4(a5)
1000017c: fsw ft1, 0x0(a5)
10000180: fsw fa4, 0x4(a5)
10000184: fsw fa4, 0x0(a2)
10000188: addi a2, a2, 0x4
1000018c: addi a5, a5, 0x40
10000190: addi a6, a6, 0x40
10000194: bne a3, t0, 0x10000114 <main+0xa8>
10000198: li s4, 0x0
1000019c: li a1, 0x0
100001a0: addi a5, a0, 0x4
100001a4: auipc a0, 0x4d
100001a8: auipc a2, 0xd
100001ac: addi s7, sp, 0xf0
100001b0: li s8, 0x40
100001b4: li s9, 0x400
100001b8: fmv.w.x fs0, zero
100001bc: li s10, 0x10
100001c0: li s11, 0xf0
100001c4: lui a3, 0x3f800
100001c8: addi s0, a0, 0x79c
100001cc: addi ra, a2, 0x798
100001d0: fmv.w.x fs1, a3
100001d4: sd s0, 0x8(sp)
100001d8: addi s0, s0, 0x4
100001dc: sd s0, 0x18(sp)
100001e0: sd a5, 0x20(sp)
100001e4: mv s2, a1
100001e8: slli a0, s4, 0x6
100001ec: slli a1, s4, 0x8
100001f0: sub a1, a1, a0
100001f4: ld a0, 0x8(sp)
100001f8: add a1, a1, a0
100001fc: addi s6, a1, 0xc4
10000200: addi s5, s2, -0x10
10000204: addi a0, sp, 0x28
10000208: li a2, 0xc0
1000020c: li a1, 0x0
10000210: mv s0, ra
10000214: jal 0x10000468 <memset>
10000218: mv ra, s0
1000021c: li a1, 0x0
10000220: li a7, 0x0
10000224: slli a0, s2, 0x6
10000228: ld a2, 0x10(sp)
1000022c: add a0, a2, a0
10000230: mx.tload.a td=0 base=a0 stride=s8 fmt=fp32 t=0
10000234: mx.azero ad=2
10000238: slli a2, s2, 0x8
1000023c: mv a3, s2
10000240: mv a4, s0
10000244: j 0x10000278 <main+0x20c>
10000248: auipc a6, 0x8
1000024c: addi a6, a6, 0x2c0
10000250: mx.tload.a td=2 base=a6 stride=s8 fmt=fp32 t=0
10000254: slli a6, a5, 0x6
10000258: add a6, s3, a6
1000025c: mx.tload.b td=3 base=a6 stride=s8 fmt=fp32 t=0
10000260: mx.mma ad=2 ta=2 tb=3 in=fp32 acc=fp32
10000264: addi a7, a5, 0x10
10000268: addi a4, a4, 0x40
1000026c: addi a3, a3, -0x10
10000270: addi a1, a1, 0x1
10000274: bgeu a5, s11, 0x1000033c <main+0x2d0>
10000278: li a6, 0x0
1000027c: mv a5, a7
10000280: slli a7, a1, 0x4
10000284: mx.azero ad=0
10000288: slli a0, a5, 0x2
1000028c: add t0, a5, a2
10000290: auipc t1, 0xd
10000294: auipc t2, 0x4f
10000298: add t3, s1, a0
1000029c: slli t0, t0, 0x2
100002a0: addi t1, t1, 0x6b0
100002a4: addi a0, t2, -0x154
100002a8: mx.tload.b td=1 base=t3 stride=s9 fmt=fp32 t=0
100002ac: add t0, t1, t0
100002b0: mx.mma ad=0 ta=0 tb=1 in=fp32 acc=fp32
100002b4: mx.astore as=0 base=t0 stride=s9 fmt=fp32
100002b8: flw fa5, 0x40(a0)
100002bc: flw fa4, 0x50(a0)
100002c0: sub a7, s5, a7
100002c4: auipc t0, 0x8
100002c8: addi t0, t0, 0x244
100002cc: mv t1, a3
100002d0: mv t2, a4
100002d4: j 0x100002ec <main+0x280>
100002d8: addi a6, a6, 0x1
100002dc: addi t2, t2, 0x400
100002e0: addi t1, t1, 0x1
100002e4: addi t0, t0, 0x40
100002e8: beq a6, s10, 0x10000248 <main+0x1dc>
100002ec: add t3, a7, a6
100002f0: mv t4, t0
100002f4: mv t5, t1
100002f8: mv t6, t2
100002fc: j 0x10000314 <main+0x2a8>
10000300: fsw fa3, 0x0(t4)
10000304: addi t6, t6, 0x4
10000308: addi t5, t5, -0x1
1000030c: addi t4, t4, 0x4
10000310: beq t5, t3, 0x100002d8 <main+0x26c>
10000314: fmv.s fa3, fs0
10000318: beqz t5, 0x10000300 <main+0x294>
1000031c: flw fa3, 0x0(t6)
10000320: fmax.s fa3, fa3, fs0
10000324: fadd.s fa3, fa5, fa3
10000328: fsqrt.s fa2, fa3
1000032c: fmul.s fa3, fa2, fa3
10000330: fdiv.s fa3, fs1, fa3
10000334: fmul.s fa3, fa4, fa3
10000338: j 0x10000300 <main+0x294>
1000033c: auipc a1, 0xc
10000340: addi a1, a1, 0x5cc
10000344: mx.astore as=2 base=a1 stride=s8 fmt=fp32
10000348: flw fa5, 0x50(a0)
1000034c: fdiv.s fa5, fs1, fa5
10000350: addi a1, a1, 0x8
10000354: ld a5, 0x20(sp)
10000358: mv a2, a5
1000035c: addi a4, sp, 0x30
10000360: mv a3, a4
10000364: ld s0, 0x18(sp)
10000368: flw fa4, -0x8(a1)
1000036c: flw fa3, -0x4(a1)
10000370: flw fa2, 0x0(a1)
10000374: flw fa1, 0x4(a1)
10000378: flw fa0, -0x4(a2)
1000037c: flw ft0, 0x0(a2)
10000380: flw ft1, 0x4(a2)
10000384: fnmsub.s fa4, fa0, fa1, fa4
10000388: fnmsub.s fa3, ft0, fa1, fa3
1000038c: fnmsub.s fa2, ft1, fa1, fa2
10000390: fmul.s fa4, fa5, fa4
10000394: fmul.s fa3, fa5, fa3
10000398: fmul.s fa2, fa5, fa2
1000039c: fsw fa4, -0x8(a3)
100003a0: fsw fa3, -0x4(a3)
100003a4: fsw fa2, 0x0(a3)
100003a8: addi a3, a3, 0xc
100003ac: addi a1, a1, 0x40
100003b0: addi a2, a2, 0xc
100003b4: bne a3, s7, 0x10000368 <main+0x2fc>
100003b8: mv a1, a4
100003bc: mv a2, s0
100003c0: flw fa5, -0x8(a1)
100003c4: flw fa4, -0x4(a1)
100003c8: flw fa3, 0x0(a1)
100003cc: fsw fa5, -0x4(a2)
100003d0: fsw fa4, 0x0(a2)
100003d4: fsw fa3, 0x4(a2)
100003d8: addi a2, a2, 0xc
100003dc: addi a1, a1, 0xc
100003e0: bne a2, s6, 0x100003c0 <main+0x354>
100003e4: addi a1, s2, 0x10
100003e8: lui a2, 0x4
100003ec: add ra, ra, a2
100003f0: addi a5, a5, 0xc0
100003f4: addi s0, s0, 0xc0
100003f8: addi s4, s4, 0x1
100003fc: bltu s2, s11, 0x100001dc <main+0x170>
10000400: mx.azero ad=0
10000404: mx.azero ad=1
10000408: mx.azero ad=2
1000040c: mx.azero ad=3
10000410: rdcycle a1
10000414: ld a2, 0x0(sp)
10000418: sub a1, a1, a2
1000041c: sd a1, 0x0(a0)
10000420: li a0, 0x0
10000424: ld ra, 0x158(sp)
10000428: ld s0, 0x150(sp)
1000042c: ld s1, 0x148(sp)
10000430: ld s2, 0x140(sp)
10000434: ld s3, 0x138(sp)
10000438: ld s4, 0x130(sp)
1000043c: ld s5, 0x128(sp)
10000440: ld s6, 0x120(sp)
10000444: ld s7, 0x118(sp)
10000448: ld s8, 0x110(sp)
1000044c: ld s9, 0x108(sp)
10000450: ld s10, 0x100(sp)
10000454: ld s11, 0xf8(sp)
10000458: fld fs0, 0xf0(sp)
1000045c: fld fs1, 0xe8(sp)
10000460: addi sp, sp, 0x160
10000464: ret
# <memset>
10000468: beqz a2, 0x10000484 <memset+0x1c>
1000046c: add a2, a0, a2
10000470: mv a3, a0
10000474: addi a4, a3, 0x1
10000478: sb a1, 0x0(a3)
1000047c: mv a3, a4
10000480: bne a4, a2, 0x10000474 <memset+0xc>
10000484: ret
