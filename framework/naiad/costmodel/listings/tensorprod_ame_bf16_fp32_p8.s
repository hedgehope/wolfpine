# tensorprod/ame size=8 bf16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x40
10000070: sd s0, 0x38(sp)
10000074: sd s1, 0x30(sp)
10000078: sd s2, 0x28(sp)
1000007c: sd s3, 0x20(sp)
10000080: sd s4, 0x18(sp)
10000084: sd s5, 0x10(sp)
10000088: sd s6, 0x8(sp)
1000008c: auipc a0, 0x17
10000090: li a2, 0x8
10000094: auipc a1, 0x0
10000098: addi a0, a0, -0x6cc
1000009c: addi a1, a1, 0x52c
100000a0: sd a2, 0x40(a0)
100000a4: sd a2, 0x48(a0)
100000a8: addi a2, a1, 0x200
100000ac: sd zero, 0x0(a1)
100000b0: addi a1, a1, 0x8
100000b4: bne a1, a2, 0x100000ac <main+0x40>
100000b8: auipc a1, 0x0
100000bc: addi a3, a1, 0x508
100000c0: addi a1, a3, 0x200
100000c4: addi a2, a3, 0x400
100000c8: sd zero, 0x0(a1)
100000cc: addi a1, a1, 0x8
100000d0: bne a1, a2, 0x100000c8 <main+0x5c>
100000d4: li a5, 0x0
100000d8: li a4, 0x1
100000dc: li a1, 0x110
100000e0: addi a2, a0, 0x8e
100000e4: mul a5, a5, a1
100000e8: add a7, a3, a5
100000ec: addi a5, a7, 0x8
100000f0: addi a6, a7, 0x2e0
100000f4: addi a7, a7, 0x108
100000f8: mv t0, a2
100000fc: lh t1, -0xe(t0)
10000100: lh t2, -0xc(t0)
10000104: lh t3, -0xa(t0)
10000108: lh t4, -0x8(t0)
1000010c: sh t1, -0x8(a5)
10000110: sh t1, -0xe0(a6)
10000114: sh t2, -0x6(a5)
10000118: sh t2, -0xc0(a6)
1000011c: sh t3, -0x4(a5)
10000120: sh t3, -0xa0(a6)
10000124: sh t4, -0x2(a5)
10000128: sh t4, -0x80(a6)
1000012c: lh t1, -0x6(t0)
10000130: lh t2, -0x4(t0)
10000134: lh t3, -0x2(t0)
10000138: lh t4, 0x0(t0)
1000013c: sh t1, 0x0(a5)
10000140: sh t1, -0x60(a6)
10000144: sh t2, 0x2(a5)
10000148: sh t2, -0x40(a6)
1000014c: sh t3, 0x4(a5)
10000150: sh t3, -0x20(a6)
10000154: sh t4, 0x6(a5)
10000158: sh t4, 0x0(a6)
1000015c: addi a5, a5, 0x20
10000160: addi a6, a6, 0x2
10000164: addi t0, t0, 0x10
10000168: bne a5, a7, 0x100000fc <main+0x90>
1000016c: andi a6, a4, 0x1
10000170: li a5, 0x1
10000174: li a4, 0x0
10000178: bnez a6, 0x100000e4 <main+0x78>
1000017c: auipc a1, 0x1
10000180: lui a2, 0x2
10000184: li a4, 0x13
10000188: addi a1, a1, -0x7bc
1000018c: slli a4, a4, 0xa
10000190: add a2, a1, a2
10000194: add a4, a1, a4
10000198: sd zero, 0x0(a2)
1000019c: addi a2, a2, 0x8
100001a0: bne a2, a4, 0x10000198 <main+0x12c>
100001a4: li a4, 0x0
100001a8: rdcycle a2
100001ac: li s0, 0x20
100001b0: addi s1, a3, 0x200
100001b4: li a6, 0x80
100001b8: addi s2, a1, 0x20
100001bc: addi a5, a1, 0x7ff
100001c0: li a7, -0x100
100001c4: li t0, -0x4
100001c8: auipc t6, 0x5
100001cc: li t1, 0xc
100001d0: li t2, 0x100
100001d4: li t3, 0x10
100001d8: li t4, 0xb
100001dc: li t5, 0xa
100001e0: li s3, 0x1
100001e4: mx.tload.a td=0 base=a3 stride=s0 fmt=bf16 t=0
100001e8: addi a3, a5, 0x21
100001ec: addi t6, t6, 0x3f8
100001f0: mx.tload.b td=5 base=s1 stride=s0 fmt=bf16 t=0
100001f4: mx.tload.b td=1 base=a1 stride=a6 fmt=bf16 t=0
100001f8: mx.tload.b td=2 base=s2 stride=a6 fmt=bf16 t=0
100001fc: slli s0, s3, 0xb
10000200: lui s1, 0x1
10000204: li s2, 0xd
10000208: mv s3, t6
1000020c: li s4, -0x4
10000210: j 0x10000238 <main+0x1cc>
10000214: mx.azero ad=3
10000218: mx.mma ad=3 ta=0 tb=4 in=bf16 acc=fp32
1000021c: addi s5, s3, 0x80
10000220: mx.astore as=2 base=s5 stride=t2 fmt=fp32
10000224: add a3, a3, s0
10000228: addi a7, a7, 0x400
1000022c: add s3, s3, s1
10000230: add a4, a4, s0
10000234: bgeu s4, s2, 0x100002c0 <main+0x254>
10000238: beq s4, t1, 0x10000258 <main+0x1ec>
1000023c: addi s5, a3, -0x7e0
10000240: mx.tload.b td=3 base=s5 stride=a6 fmt=bf16 t=0
10000244: mx.azero ad=0
10000248: mx.mma ad=0 ta=0 tb=1 in=bf16 acc=fp32
1000024c: bne s4, t0, 0x10000258 <main+0x1ec>
10000250: li s5, 0x0
10000254: j 0x10000278 <main+0x20c>
10000258: andi s5, a7, -0x400
1000025c: slli s5, s5, 0x2
10000260: add s5, t6, s5
10000264: addi s5, s5, 0xc0
10000268: mx.astore as=3 base=s5 stride=t2 fmt=fp32
1000026c: addi s6, s4, 0x5
10000270: mv s5, a4
10000274: bltu t3, s6, 0x100002c0 <main+0x254>
10000278: add s5, a1, s5
1000027c: addi s6, s5, 0x60
10000280: mx.tload.b td=4 base=s6 stride=a6 fmt=bf16 t=0
10000284: mx.azero ad=1
10000288: mx.mma ad=1 ta=0 tb=2 in=bf16 acc=fp32
1000028c: mx.astore as=0 base=s3 stride=t2 fmt=fp32
10000290: addi s4, s4, 0x4
10000294: bltu t4, s4, 0x100002a4 <main+0x238>
10000298: addi s5, s5, 0x7ff
1000029c: addi s5, s5, 0x1
100002a0: mx.tload.b td=1 base=s5 stride=a6 fmt=bf16 t=0
100002a4: mx.azero ad=2
100002a8: mx.mma ad=2 ta=0 tb=3 in=bf16 acc=fp32
100002ac: addi s5, s3, 0x40
100002b0: mx.astore as=1 base=s5 stride=t2 fmt=fp32
100002b4: bltu t5, s4, 0x10000214 <main+0x1a8>
100002b8: mx.tload.b td=2 base=a3 stride=a6 fmt=bf16 t=0
100002bc: j 0x10000214 <main+0x1a8>
100002c0: li a3, 0x0
100002c4: li a4, 0x0
100002c8: li a6, 0x20
100002cc: addi t5, a1, 0x200
100002d0: addi a7, a5, 0x1
100002d4: li a5, -0x4
100002d8: li t0, 0xc
100002dc: auipc t4, 0xa
100002e0: li t1, 0x40
100002e4: li t2, 0x10
100002e8: li t3, 0xb
100002ec: mx.tload.a td=1 base=a1 stride=a6 fmt=bf16 t=0
100002f0: addi t4, t4, 0x2e4
100002f4: mx.tload.a td=2 base=t5 stride=a6 fmt=bf16 t=0
100002f8: li t5, 0xa
100002fc: lui t6, 0x1
10000300: li s0, 0xd
10000304: li s1, -0x4
10000308: j 0x10000338 <main+0x2cc>
1000030c: mx.azero ad=3
10000310: mx.mma ad=3 ta=4 tb=5 in=bf16 acc=fp32
10000314: addi s2, s2, 0x200
10000318: slli s2, s2, 0x2
1000031c: add s2, t4, s2
10000320: mx.astore as=2 base=s2 stride=t1 fmt=fp32
10000324: addi a7, a7, 0x7ff
10000328: addi a7, a7, 0x1
1000032c: addi a4, a4, 0x400
10000330: add a3, a3, t6
10000334: bgeu s1, s0, 0x100003cc <main+0x360>
10000338: beq s1, t0, 0x1000035c <main+0x2f0>
1000033c: addi s2, a7, -0x400
10000340: mx.tload.a td=3 base=s2 stride=a6 fmt=bf16 t=0
10000344: mx.azero ad=0
10000348: mx.mma ad=0 ta=1 tb=5 in=bf16 acc=fp32
1000034c: bne s1, a5, 0x1000035c <main+0x2f0>
10000350: li s3, 0x0
10000354: li s2, 0x0
10000358: j 0x10000378 <main+0x30c>
1000035c: add s2, t4, a3
10000360: addi s2, s2, -0x400
10000364: mx.astore as=3 base=s2 stride=t1 fmt=fp32
10000368: addi s4, s1, 0x5
1000036c: mv s3, a3
10000370: mv s2, a4
10000374: bltu t2, s4, 0x100003cc <main+0x360>
10000378: slli s4, s2, 0x1
1000037c: add s3, t4, s3
10000380: add s4, a1, s4
10000384: addi s4, s4, 0x600
10000388: mx.tload.a td=4 base=s4 stride=a6 fmt=bf16 t=0
1000038c: mx.azero ad=1
10000390: mx.mma ad=1 ta=2 tb=5 in=bf16 acc=fp32
10000394: mx.astore as=0 base=s3 stride=t1 fmt=fp32
10000398: addi s1, s1, 0x4
1000039c: bltu t3, s1, 0x100003a4 <main+0x338>
100003a0: mx.tload.a td=1 base=a7 stride=a6 fmt=bf16 t=0
100003a4: mx.azero ad=2
100003a8: mx.mma ad=2 ta=3 tb=5 in=bf16 acc=fp32
100003ac: slli s3, s2, 0x2
100003b0: add s3, t4, s3
100003b4: addi s3, s3, 0x400
100003b8: mx.astore as=1 base=s3 stride=t1 fmt=fp32
100003bc: bltu t5, s1, 0x1000030c <main+0x2a0>
100003c0: addi s3, a7, 0x200
100003c4: mx.tload.a td=2 base=s3 stride=a6 fmt=bf16 t=0
100003c8: j 0x1000030c <main+0x2a0>
100003cc: li a3, 0x0
100003d0: li a4, 0x10
100003d4: addi t5, a1, 0x100
100003d8: addi a5, a1, 0x400
100003dc: li a6, -0x4
100003e0: li a7, 0x180
100003e4: li t0, 0x1c
100003e8: auipc t4, 0xe
100003ec: li t1, 0x40
100003f0: li t2, 0x20
100003f4: li t3, 0x1b
100003f8: mx.tload.b td=1 base=a1 stride=a4 fmt=bf16 t=0
100003fc: addi t4, t4, 0x1d8
10000400: mx.tload.b td=2 base=t5 stride=a4 fmt=bf16 t=0
10000404: li t5, 0x1a
10000408: lui t6, 0x1
1000040c: li s0, 0x1d
10000410: li s1, -0x4
10000414: j 0x1000043c <main+0x3d0>
10000418: mx.azero ad=3
1000041c: mx.mma ad=3 ta=0 tb=4 in=bf16 acc=fp32
10000420: addi s2, s2, 0x7ff
10000424: addi s2, s2, 0x1
10000428: mx.astore as=2 base=s2 stride=t1 fmt=fp32
1000042c: addi a5, a5, 0x400
10000430: add a3, a3, t6
10000434: addi a7, a7, 0x200
10000438: bgeu s1, s0, 0x100004c4 <main+0x458>
1000043c: add s2, t4, a3
10000440: beq s1, t0, 0x10000464 <main+0x3f8>
10000444: addi s3, a5, -0x200
10000448: mx.tload.b td=3 base=s3 stride=a4 fmt=bf16 t=0
1000044c: mx.azero ad=0
10000450: mx.mma ad=0 ta=0 tb=1 in=bf16 acc=fp32
10000454: bne s1, a6, 0x10000464 <main+0x3f8>
10000458: li s3, 0x0
1000045c: li s4, 0x180
10000460: j 0x1000047c <main+0x410>
10000464: addi s3, s2, -0x400
10000468: mx.astore as=3 base=s3 stride=t1 fmt=fp32
1000046c: addi s5, s1, 0x5
10000470: mv s3, a3
10000474: mv s4, a7
10000478: bltu t2, s5, 0x100004c4 <main+0x458>
1000047c: slli s4, s4, 0x1
10000480: add s3, t4, s3
10000484: add s4, a1, s4
10000488: mx.tload.b td=4 base=s4 stride=a4 fmt=bf16 t=0
1000048c: mx.azero ad=1
10000490: mx.mma ad=1 ta=0 tb=2 in=bf16 acc=fp32
10000494: mx.astore as=0 base=s3 stride=t1 fmt=fp32
10000498: addi s1, s1, 0x4
1000049c: bltu t3, s1, 0x100004a4 <main+0x438>
100004a0: mx.tload.b td=1 base=a5 stride=a4 fmt=bf16 t=0
100004a4: mx.azero ad=2
100004a8: mx.mma ad=2 ta=0 tb=3 in=bf16 acc=fp32
100004ac: addi s3, s2, 0x400
100004b0: mx.astore as=1 base=s3 stride=t1 fmt=fp32
100004b4: bltu t5, s1, 0x10000418 <main+0x3ac>
100004b8: addi s3, a5, 0x100
100004bc: mx.tload.b td=2 base=s3 stride=a4 fmt=bf16 t=0
100004c0: j 0x10000418 <main+0x3ac>
100004c4: mx.azero ad=0
100004c8: mx.azero ad=1
100004cc: mx.azero ad=2
100004d0: mx.azero ad=3
100004d4: rdcycle a1
100004d8: sub a1, a1, a2
100004dc: sd a1, 0x0(a0)
100004e0: li a0, 0x0
100004e4: ld s0, 0x38(sp)
100004e8: ld s1, 0x30(sp)
100004ec: ld s2, 0x28(sp)
100004f0: ld s3, 0x20(sp)
100004f4: ld s4, 0x18(sp)
100004f8: ld s5, 0x10(sp)
100004fc: ld s6, 0x8(sp)
10000500: addi sp, sp, 0x40
10000504: ret
