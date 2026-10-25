# fft/ame size=1024 fp64/fp64 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0xa0
10000070: sd ra, 0x98(sp)
10000074: sd s0, 0x90(sp)
10000078: sd s1, 0x88(sp)
1000007c: sd s2, 0x80(sp)
10000080: sd s3, 0x78(sp)
10000084: sd s4, 0x70(sp)
10000088: sd s5, 0x68(sp)
1000008c: sd s6, 0x60(sp)
10000090: sd s7, 0x58(sp)
10000094: sd s8, 0x50(sp)
10000098: sd s9, 0x48(sp)
1000009c: sd s10, 0x40(sp)
100000a0: sd s11, 0x38(sp)
100000a4: li a0, 0x0
100000a8: sd zero, 0x18(sp)
100000ac: li a1, 0x1
100000b0: sd a1, 0x20(sp)
100000b4: rdcycle a1
100000b8: sd a1, 0x0(sp)
100000bc: auipc a1, 0x0
100000c0: auipc a2, 0x4
100000c4: auipc a3, 0xc
100000c8: li a4, 0x100
100000cc: addi a1, a1, 0x414
100000d0: sd a1, 0x8(sp)
100000d4: addi t2, a2, 0x500
100000d8: addi a3, a3, 0x4fc
100000dc: li t1, 0x1
100000e0: j 0x100000fc <main+0x90>
100000e4: sd zero, 0x20(sp)
100000e8: mul a0, t4, s10
100000ec: add s9, a0, s9
100000f0: sd s9, 0x18(sp)
100000f4: li a0, 0x1
100000f8: ld t1, 0x10(sp)
100000fc: sd zero, 0x28(sp)
10000100: slli a0, a0, 0x2
10000104: ld a1, 0x8(sp)
10000108: add a0, a1, a0
1000010c: lw s10, 0x0(a0)
10000110: slli s2, t1, 0x4
10000114: mulw a1, s10, t1
10000118: li a0, 0x400
1000011c: sd a1, 0x10(sp)
10000120: divw t4, a0, a1
10000124: slli s3, t1, 0x3
10000128: slli s4, t4, 0x3
1000012c: mul s5, s4, t1
10000130: sd s10, 0x30(sp)
10000134: j 0x1000014c <main+0xe0>
10000138: ld a1, 0x28(sp)
1000013c: addi a1, a1, 0x1
10000140: li a0, 0x4
10000144: sd a1, 0x28(sp)
10000148: beq a1, a0, 0x10000370 <main+0x304>
1000014c: auipc a0, 0x1c
10000150: auipc a1, 0x14
10000154: auipc a5, 0x28
10000158: auipc a6, 0x2a
1000015c: auipc a7, 0x2c
10000160: addi t5, a0, 0x474
10000164: addi t6, a1, 0x470
10000168: addi s6, a5, 0x46c
1000016c: addi s7, a6, 0x468
10000170: addi s8, a7, 0x464
10000174: li a0, 0xf
10000178: bge a0, t4, 0x10000288 <main+0x21c>
1000017c: blez t1, 0x10000138 <main+0xcc>
10000180: li s9, 0x0
10000184: ld s1, 0x28(sp)
10000188: slli s1, s1, 0xa
1000018c: li s11, 0x0
10000190: slli a0, s9, 0x5
10000194: mul ra, a0, t4
10000198: addi a0, a0, 0x10
1000019c: mul a0, a0, t4
100001a0: li a5, 0x0
100001a4: add s10, s11, s1
100001a8: add a1, ra, s10
100001ac: add a6, a0, s10
100001b0: slli a1, a1, 0x3
100001b4: slli s0, a6, 0x3
100001b8: add a7, t2, a1
100001bc: add t0, a3, a1
100001c0: add a6, t2, s0
100001c4: add a1, a3, s0
100001c8: li s0, 0x1
100001cc: mx.azero ad=0
100001d0: mx.azero ad=1
100001d4: mx.tload.b td=0 base=a7 stride=s4 fmt=fp64 t=0
100001d8: slli a2, a5, 0xc
100001dc: mul t3, a5, s2
100001e0: mx.tload.b td=1 base=t0 stride=s4 fmt=fp64 t=0
100001e4: add a5, s6, a2
100001e8: mx.tload.a td=3 base=a5 stride=a4 fmt=fp64 t=0
100001ec: add a5, s7, a2
100001f0: mx.tload.a td=4 base=a5 stride=a4 fmt=fp64 t=0
100001f4: add a5, s8, a2
100001f8: addi a2, a2, 0x80
100001fc: add t3, t3, s9
10000200: mx.tload.a td=5 base=a5 stride=a4 fmt=fp64 t=0
10000204: add a5, s6, a2
10000208: mx.mma ad=0 ta=3 tb=0 in=fp64 acc=fp64
1000020c: mx.mma ad=0 ta=5 tb=1 in=fp64 acc=fp64
10000210: mx.mma ad=1 ta=4 tb=0 in=fp64 acc=fp64
10000214: mx.mma ad=1 ta=3 tb=1 in=fp64 acc=fp64
10000218: mx.tload.b td=0 base=a6 stride=s4 fmt=fp64 t=0
1000021c: mx.tload.b td=1 base=a1 stride=s4 fmt=fp64 t=0
10000220: mx.tload.a td=3 base=a5 stride=a4 fmt=fp64 t=0
10000224: add a5, s7, a2
10000228: add a2, s8, a2
1000022c: mul t3, t3, t4
10000230: add t3, t3, s10
10000234: slli t3, t3, 0x3
10000238: mx.tload.a td=4 base=a5 stride=a4 fmt=fp64 t=0
1000023c: add a5, t6, t3
10000240: mx.tload.a td=5 base=a2 stride=a4 fmt=fp64 t=0
10000244: mx.mma ad=0 ta=3 tb=0 in=fp64 acc=fp64
10000248: mx.mma ad=0 ta=5 tb=1 in=fp64 acc=fp64
1000024c: mx.mma ad=1 ta=4 tb=0 in=fp64 acc=fp64
10000250: mx.mma ad=1 ta=3 tb=1 in=fp64 acc=fp64
10000254: mx.astore as=0 base=a5 stride=s5 fmt=fp64
10000258: add t3, t5, t3
1000025c: mx.astore as=1 base=t3 stride=s5 fmt=fp64
10000260: andi a2, s0, 0x1
10000264: li a5, 0x1
10000268: li s0, 0x0
1000026c: bnez a2, 0x100001cc <main+0x160>
10000270: addi s11, s11, 0x10
10000274: blt s11, t4, 0x100001a0 <main+0x134>
10000278: addi s9, s9, 0x1
1000027c: ld s10, 0x30(sp)
10000280: bne s9, t1, 0x1000018c <main+0x120>
10000284: j 0x10000138 <main+0xcc>
10000288: blez t1, 0x10000138 <main+0xcc>
1000028c: li s9, 0x0
10000290: ld a0, 0x28(sp)
10000294: slli a0, a0, 0xa
10000298: li a5, 0x0
1000029c: slli a1, s9, 0x5
100002a0: add s1, s9, a0
100002a4: add a1, a1, a0
100002a8: slli a1, a1, 0x3
100002ac: add a7, t2, a1
100002b0: add a6, a3, a1
100002b4: addi t0, a1, 0x80
100002b8: add a1, t2, t0
100002bc: add t0, a3, t0
100002c0: li s0, 0x1
100002c4: mx.azero ad=0
100002c8: mx.azero ad=1
100002cc: mx.tload.b td=0 base=a7 stride=a4 fmt=fp64 t=1
100002d0: slli a2, a5, 0xc
100002d4: mul a5, a5, s2
100002d8: mx.tload.b td=1 base=a6 stride=a4 fmt=fp64 t=1
100002dc: add t3, s6, a2
100002e0: mx.tload.a td=3 base=t3 stride=a4 fmt=fp64 t=0
100002e4: add t3, s7, a2
100002e8: mx.tload.a td=4 base=t3 stride=a4 fmt=fp64 t=0
100002ec: add t3, s8, a2
100002f0: addi a2, a2, 0x80
100002f4: add a5, s1, a5
100002f8: mx.tload.a td=5 base=t3 stride=a4 fmt=fp64 t=0
100002fc: add t3, s6, a2
10000300: mx.mma ad=0 ta=3 tb=0 in=fp64 acc=fp64
10000304: mx.mma ad=0 ta=5 tb=1 in=fp64 acc=fp64
10000308: mx.mma ad=1 ta=4 tb=0 in=fp64 acc=fp64
1000030c: mx.mma ad=1 ta=3 tb=1 in=fp64 acc=fp64
10000310: mx.tload.b td=0 base=a1 stride=a4 fmt=fp64 t=1
10000314: mx.tload.b td=1 base=t0 stride=a4 fmt=fp64 t=1
10000318: mx.tload.a td=3 base=t3 stride=a4 fmt=fp64 t=0
1000031c: add t3, s7, a2
10000320: add a2, s8, a2
10000324: slli a5, a5, 0x3
10000328: mx.tload.a td=4 base=t3 stride=a4 fmt=fp64 t=0
1000032c: add t3, t6, a5
10000330: mx.tload.a td=5 base=a2 stride=a4 fmt=fp64 t=0
10000334: mx.mma ad=0 ta=3 tb=0 in=fp64 acc=fp64
10000338: mx.mma ad=0 ta=5 tb=1 in=fp64 acc=fp64
1000033c: mx.mma ad=1 ta=4 tb=0 in=fp64 acc=fp64
10000340: mx.mma ad=1 ta=3 tb=1 in=fp64 acc=fp64
10000344: mx.astore as=0 base=t3 stride=s3 fmt=fp64
10000348: add a5, t5, a5
1000034c: mx.astore as=1 base=a5 stride=s3 fmt=fp64
10000350: andi a2, s0, 0x1
10000354: li a5, 0x1
10000358: li s0, 0x0
1000035c: bnez a2, 0x100002c4 <main+0x258>
10000360: addiw a1, s9, 0x10
10000364: addi s9, s9, 0x10
10000368: blt a1, t1, 0x10000298 <main+0x22c>
1000036c: j 0x10000138 <main+0xcc>
10000370: ld a0, 0x20(sp)
10000374: andi a0, a0, 0x1
10000378: beqz a0, 0x1000046c <main+0x400>
1000037c: li s0, 0x0
10000380: slli a1, t4, 0x20
10000384: ld s9, 0x18(sp)
10000388: slli s1, s9, 0x3
1000038c: auipc a0, 0x2
10000390: auipc a2, 0x0
10000394: addi a0, a0, 0x1f4
10000398: addi a2, a2, 0x1f0
1000039c: add a0, a0, s1
100003a0: add s1, a2, s1
100003a4: srli s2, a1, 0x20
100003a8: srli s3, a1, 0x1d
100003ac: j 0x100003bc <main+0x350>
100003b0: addi s0, s0, 0x1
100003b4: li a1, 0x4
100003b8: beq s0, a1, 0x100000e4 <main+0x78>
100003bc: li a2, 0x0
100003c0: li s4, 0x0
100003c4: slli a7, s0, 0xa
100003c8: j 0x100003d8 <main+0x36c>
100003cc: addi s4, s4, 0x1
100003d0: add a2, a2, t4
100003d4: beq s4, s10, 0x100003b0 <main+0x344>
100003d8: blez t1, 0x100003cc <main+0x360>
100003dc: li a1, 0x0
100003e0: slli a5, a2, 0x20
100003e4: srli a5, a5, 0x1d
100003e8: add a6, a0, a5
100003ec: add t0, s1, a5
100003f0: j 0x100003fc <main+0x390>
100003f4: addiw a1, a1, 0x1
100003f8: beq a1, t1, 0x100003cc <main+0x360>
100003fc: blez t4, 0x100003f4 <main+0x388>
10000400: li a5, 0x0
10000404: slli s8, a7, 0x3
10000408: add s5, a3, s8
1000040c: add s6, t2, s8
10000410: add s7, t5, s8
10000414: add s8, t6, s8
10000418: add a7, s2, a7
1000041c: add t3, s8, a5
10000420: fld fa5, 0x0(t3)
10000424: add t3, s7, a5
10000428: fld fa4, 0x0(t3)
1000042c: add t3, a6, a5
10000430: fld fa3, 0x0(t3)
10000434: add t3, t0, a5
10000438: fld fa2, 0x0(t3)
1000043c: add t3, s6, a5
10000440: fneg.d fa1, fa3
10000444: fmul.d fa1, fa4, fa1
10000448: fmadd.d fa1, fa5, fa2, fa1
1000044c: fsd fa1, 0x0(t3)
10000450: add t3, s5, a5
10000454: addi a5, a5, 0x8
10000458: fmul.d fa4, fa4, fa2
1000045c: fmadd.d fa5, fa5, fa3, fa4
10000460: fsd fa5, 0x0(t3)
10000464: bne s3, a5, 0x1000041c <main+0x3b0>
10000468: j 0x100003f4 <main+0x388>
1000046c: mx.azero ad=0
10000470: mx.azero ad=1
10000474: mx.azero ad=2
10000478: mx.azero ad=3
1000047c: rdcycle a0
10000480: ld a1, 0x0(sp)
10000484: sub a0, a0, a1
10000488: auipc a1, 0x4
1000048c: sd a0, 0xf8(a1)
10000490: li a0, 0x0
10000494: ld ra, 0x98(sp)
10000498: ld s0, 0x90(sp)
1000049c: ld s1, 0x88(sp)
100004a0: ld s2, 0x80(sp)
100004a4: ld s3, 0x78(sp)
100004a8: ld s4, 0x70(sp)
100004ac: ld s5, 0x68(sp)
100004b0: ld s6, 0x60(sp)
100004b4: ld s7, 0x58(sp)
100004b8: ld s8, 0x50(sp)
100004bc: ld s9, 0x48(sp)
100004c0: ld s10, 0x40(sp)
100004c4: ld s11, 0x38(sp)
100004c8: addi sp, sp, 0xa0
100004cc: ret
