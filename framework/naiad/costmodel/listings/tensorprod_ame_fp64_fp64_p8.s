# tensorprod/ame size=8 fp64/fp64 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x40
10000070: sd s0, 0x38(sp)
10000074: sd s1, 0x30(sp)
10000078: sd s2, 0x28(sp)
1000007c: sd s3, 0x20(sp)
10000080: sd s4, 0x18(sp)
10000084: sd s5, 0x10(sp)
10000088: sd s6, 0x8(sp)
1000008c: auipc a0, 0x37
10000090: li a2, 0x8
10000094: auipc a1, 0x0
10000098: addi a0, a0, -0x2cc
1000009c: addi a1, a1, 0x52c
100000a0: sd a2, 0x40(a0)
100000a4: sd a2, 0x48(a0)
100000a8: addi a2, a1, 0x7ff
100000ac: addi a2, a2, 0x1
100000b0: sd zero, 0x0(a1)
100000b4: addi a1, a1, 0x8
100000b8: bne a1, a2, 0x100000b0 <main+0x44>
100000bc: auipc a1, 0x1
100000c0: addi a1, a1, -0x2fc
100000c4: addi a2, a1, 0x7ff
100000c8: addi a2, a2, 0x1
100000cc: sd zero, 0x0(a1)
100000d0: addi a1, a1, 0x8
100000d4: bne a1, a2, 0x100000cc <main+0x60>
100000d8: li a6, 0x0
100000dc: li a2, 0x1
100000e0: li a1, 0x440
100000e4: auipc a3, 0x0
100000e8: auipc a4, 0x1
100000ec: addi a3, a3, 0x4dc
100000f0: addi a4, a4, -0x328
100000f4: addi a5, a0, 0xb8
100000f8: mul a6, a6, a1
100000fc: add t0, a3, a6
10000100: add a7, a4, a6
10000104: addi a6, t0, 0x20
10000108: addi a7, a7, 0x200
1000010c: addi t0, t0, 0x420
10000110: mv t1, a5
10000114: fld fa5, -0x38(t1)
10000118: fld fa4, -0x30(t1)
1000011c: fld fa3, -0x28(t1)
10000120: fld fa2, -0x20(t1)
10000124: fsd fa5, -0x200(a7)
10000128: fsd fa4, -0x180(a7)
1000012c: fsd fa3, -0x100(a7)
10000130: fsd fa5, -0x20(a6)
10000134: fsd fa4, -0x18(a6)
10000138: fsd fa3, -0x10(a6)
1000013c: fsd fa2, -0x8(a6)
10000140: fsd fa2, -0x80(a7)
10000144: fld fa5, -0x18(t1)
10000148: fld fa4, -0x10(t1)
1000014c: fld fa3, -0x8(t1)
10000150: fld fa2, 0x0(t1)
10000154: fsd fa5, 0x0(a7)
10000158: fsd fa4, 0x80(a7)
1000015c: fsd fa3, 0x100(a7)
10000160: fsd fa5, 0x0(a6)
10000164: fsd fa4, 0x8(a6)
10000168: fsd fa3, 0x10(a6)
1000016c: fsd fa2, 0x18(a6)
10000170: fsd fa2, 0x180(a7)
10000174: addi a6, a6, 0x80
10000178: addi a7, a7, 0x8
1000017c: addi t1, t1, 0x40
10000180: bne a6, t0, 0x10000114 <main+0xa8>
10000184: andi a7, a2, 0x1
10000188: li a6, 0x1
1000018c: li a2, 0x0
10000190: bnez a7, 0x100000f8 <main+0x8c>
10000194: auipc a1, 0x1
10000198: lui a2, 0x8
1000019c: lui a3, 0x13
100001a0: addi a1, a1, 0x42c
100001a4: add a2, a1, a2
100001a8: add a3, a1, a3
100001ac: sd zero, 0x0(a2)
100001b0: addi a2, a2, 0x8
100001b4: bne a2, a3, 0x100001ac <main+0x140>
100001b8: li a3, 0x0
100001bc: rdcycle a2
100001c0: auipc t3, 0x0
100001c4: li t4, 0x80
100001c8: auipc t5, 0x1
100001cc: li a4, 0x200
100001d0: addi t6, a1, 0x80
100001d4: li a5, -0x100
100001d8: li a6, -0x4
100001dc: li a7, 0xc
100001e0: auipc s0, 0x14
100001e4: li t0, 0x10
100001e8: li t1, 0xb
100001ec: lui t2, 0x2
100001f0: addi s1, t3, 0x400
100001f4: addi t5, t5, -0x408
100001f8: addi t3, s0, 0x3e0
100001fc: mx.tload.a td=0 base=s1 stride=t4 fmt=fp64 t=0
10000200: mx.tload.b td=5 base=t5 stride=t4 fmt=fp64 t=0
10000204: mx.tload.b td=1 base=a1 stride=a4 fmt=fp64 t=0
10000208: mx.tload.b td=2 base=t6 stride=a4 fmt=fp64 t=0
1000020c: li t4, 0xa
10000210: addiw t5, t2, 0x80
10000214: li t6, 0xd
10000218: li s0, -0x4
1000021c: j 0x1000023c <main+0x1d0>
10000220: mx.azero ad=3
10000224: mx.mma ad=3 ta=0 tb=4 in=fp64 acc=fp64
10000228: addi s1, s1, 0x100
1000022c: mx.astore as=2 base=s1 stride=a4 fmt=fp64
10000230: add a3, a3, t2
10000234: addi a5, a5, 0x400
10000238: bgeu s0, t6, 0x100002d0 <main+0x264>
1000023c: beq s0, a7, 0x10000260 <main+0x1f4>
10000240: add s1, a1, a3
10000244: addi s1, s1, 0x100
10000248: mx.tload.b td=3 base=s1 stride=a4 fmt=fp64 t=0
1000024c: mx.azero ad=0
10000250: mx.mma ad=0 ta=0 tb=1 in=fp64 acc=fp64
10000254: bne s0, a6, 0x10000260 <main+0x1f4>
10000258: li s1, 0x0
1000025c: j 0x10000280 <main+0x214>
10000260: andi s1, a5, -0x400
10000264: slli s1, s1, 0x3
10000268: add s1, t3, s1
1000026c: addi s1, s1, 0x180
10000270: mx.astore as=3 base=s1 stride=a4 fmt=fp64
10000274: addi s2, s0, 0x5
10000278: mv s1, a3
1000027c: bltu t0, s2, 0x100002d0 <main+0x264>
10000280: add s2, a1, s1
10000284: add s1, t3, s1
10000288: addi s3, s2, 0x180
1000028c: mx.tload.b td=4 base=s3 stride=a4 fmt=fp64 t=0
10000290: mx.azero ad=1
10000294: mx.mma ad=1 ta=0 tb=2 in=fp64 acc=fp64
10000298: mx.astore as=0 base=s1 stride=a4 fmt=fp64
1000029c: addi s0, s0, 0x4
100002a0: bltu t1, s0, 0x100002ac <main+0x240>
100002a4: add s2, s2, t2
100002a8: mx.tload.b td=1 base=s2 stride=a4 fmt=fp64 t=0
100002ac: mx.azero ad=2
100002b0: mx.mma ad=2 ta=0 tb=3 in=fp64 acc=fp64
100002b4: addi s2, s1, 0x80
100002b8: mx.astore as=1 base=s2 stride=a4 fmt=fp64
100002bc: bltu t4, s0, 0x10000220 <main+0x1b4>
100002c0: add s2, a1, a3
100002c4: add s2, s2, t5
100002c8: mx.tload.b td=2 base=s2 stride=a4 fmt=fp64 t=0
100002cc: j 0x10000220 <main+0x1b4>
100002d0: li a3, 0x0
100002d4: li a4, 0x0
100002d8: li a5, 0x80
100002dc: addi t4, a1, 0x7ff
100002e0: li a6, -0x4
100002e4: li a7, 0xc
100002e8: auipc t5, 0x1e
100002ec: li t0, 0x10
100002f0: li t1, 0xb
100002f4: lui t2, 0x2
100002f8: li t3, 0xa
100002fc: mx.tload.a td=1 base=a1 stride=a5 fmt=fp64 t=0
10000300: addi t6, t4, 0x1
10000304: addi t4, t5, 0x2d8
10000308: mx.tload.a td=2 base=t6 stride=a5 fmt=fp64 t=0
1000030c: li t5, 0x5
10000310: slli t5, t5, 0xb
10000314: li t6, 0xd
10000318: lui s0, 0x1
1000031c: j 0x10000344 <main+0x2d8>
10000320: mx.azero ad=3
10000324: mx.mma ad=3 ta=4 tb=5 in=fp64 acc=fp64
10000328: addi s1, s1, 0x200
1000032c: slli s1, s1, 0x3
10000330: add s1, t4, s1
10000334: mx.astore as=2 base=s1 stride=a5 fmt=fp64
10000338: addi a4, a4, 0x400
1000033c: add a3, a3, t2
10000340: bgeu a6, t6, 0x100003e8 <main+0x37c>
10000344: beq a6, a7, 0x10000360 <main+0x2f4>
10000348: add s1, a1, a3
1000034c: add s1, s1, s0
10000350: mx.tload.a td=3 base=s1 stride=a5 fmt=fp64 t=0
10000354: mx.azero ad=0
10000358: mx.mma ad=0 ta=1 tb=5 in=fp64 acc=fp64
1000035c: beqz a4, 0x100003dc <main+0x370>
10000360: add s1, t4, a3
10000364: addi s1, s1, -0x800
10000368: mx.astore as=3 base=s1 stride=a5 fmt=fp64
1000036c: addi s3, a6, 0x5
10000370: mv s2, a3
10000374: mv s1, a4
10000378: bltu t0, s3, 0x100003e8 <main+0x37c>
1000037c: addi s3, s1, 0x300
10000380: slli s3, s3, 0x3
10000384: add s3, a1, s3
10000388: mx.tload.a td=4 base=s3 stride=a5 fmt=fp64 t=0
1000038c: add s3, t4, s2
10000390: mx.azero ad=1
10000394: mx.mma ad=1 ta=2 tb=5 in=fp64 acc=fp64
10000398: mx.astore as=0 base=s3 stride=a5 fmt=fp64
1000039c: addi a6, a6, 0x4
100003a0: bltu t1, a6, 0x100003b0 <main+0x344>
100003a4: add s3, a1, s2
100003a8: add s3, s3, t2
100003ac: mx.tload.a td=1 base=s3 stride=a5 fmt=fp64 t=0
100003b0: mx.azero ad=2
100003b4: mx.mma ad=2 ta=3 tb=5 in=fp64 acc=fp64
100003b8: addi s3, s1, 0x100
100003bc: slli s3, s3, 0x3
100003c0: add s3, t4, s3
100003c4: mx.astore as=1 base=s3 stride=a5 fmt=fp64
100003c8: bltu t3, a6, 0x10000320 <main+0x2b4>
100003cc: add s2, a1, s2
100003d0: add s2, s2, t5
100003d4: mx.tload.a td=2 base=s2 stride=a5 fmt=fp64 t=0
100003d8: j 0x10000320 <main+0x2b4>
100003dc: li s2, 0x0
100003e0: li s1, 0x0
100003e4: j 0x1000037c <main+0x310>
100003e8: li a3, 0x0
100003ec: li a4, 0x40
100003f0: addi t6, a1, 0x400
100003f4: lui a5, 0x1
100003f8: li a6, -0x4
100003fc: li a7, 0x180
10000400: li t0, 0x1c
10000404: auipc t5, 0x26
10000408: li t1, 0x80
1000040c: li t2, 0x20
10000410: li t3, 0x1b
10000414: mx.tload.b td=1 base=a1 stride=a4 fmt=fp64 t=0
10000418: add t4, a1, a5
1000041c: addi t5, t5, 0x1bc
10000420: mx.tload.b td=2 base=t6 stride=a4 fmt=fp64 t=0
10000424: li t6, 0x1a
10000428: lui s0, 0x2
1000042c: li s1, 0x1d
10000430: li s2, -0x4
10000434: j 0x10000458 <main+0x3ec>
10000438: mx.azero ad=3
1000043c: mx.mma ad=3 ta=0 tb=4 in=fp64 acc=fp64
10000440: add s3, s3, a5
10000444: mx.astore as=2 base=s3 stride=t1 fmt=fp64
10000448: add t4, t4, a5
1000044c: add a3, a3, s0
10000450: addi a7, a7, 0x200
10000454: bgeu s2, s1, 0x100004e4 <main+0x478>
10000458: add s3, t5, a3
1000045c: beq s2, t0, 0x10000480 <main+0x414>
10000460: addi s4, t4, -0x800
10000464: mx.tload.b td=3 base=s4 stride=a4 fmt=fp64 t=0
10000468: mx.azero ad=0
1000046c: mx.mma ad=0 ta=0 tb=1 in=fp64 acc=fp64
10000470: bne s2, a6, 0x10000480 <main+0x414>
10000474: li s4, 0x0
10000478: li s5, 0x180
1000047c: j 0x10000498 <main+0x42c>
10000480: addi s4, s3, -0x800
10000484: mx.astore as=3 base=s4 stride=t1 fmt=fp64
10000488: addi s6, s2, 0x5
1000048c: mv s4, a3
10000490: mv s5, a7
10000494: bltu t2, s6, 0x100004e4 <main+0x478>
10000498: slli s5, s5, 0x3
1000049c: add s4, t5, s4
100004a0: add s5, a1, s5
100004a4: mx.tload.b td=4 base=s5 stride=a4 fmt=fp64 t=0
100004a8: mx.azero ad=1
100004ac: mx.mma ad=1 ta=0 tb=2 in=fp64 acc=fp64
100004b0: mx.astore as=0 base=s4 stride=t1 fmt=fp64
100004b4: addi s2, s2, 0x4
100004b8: bltu t3, s2, 0x100004c0 <main+0x454>
100004bc: mx.tload.b td=1 base=t4 stride=a4 fmt=fp64 t=0
100004c0: mx.azero ad=2
100004c4: mx.mma ad=2 ta=0 tb=3 in=fp64 acc=fp64
100004c8: addi s4, s3, 0x7ff
100004cc: addi s4, s4, 0x1
100004d0: mx.astore as=1 base=s4 stride=t1 fmt=fp64
100004d4: bltu t6, s2, 0x10000438 <main+0x3cc>
100004d8: addi s4, t4, 0x400
100004dc: mx.tload.b td=2 base=s4 stride=a4 fmt=fp64 t=0
100004e0: j 0x10000438 <main+0x3cc>
100004e4: mx.azero ad=0
100004e8: mx.azero ad=1
100004ec: mx.azero ad=2
100004f0: mx.azero ad=3
100004f4: rdcycle a1
100004f8: sub a1, a1, a2
100004fc: sd a1, 0x0(a0)
10000500: li a0, 0x0
10000504: ld s0, 0x38(sp)
10000508: ld s1, 0x30(sp)
1000050c: ld s2, 0x28(sp)
10000510: ld s3, 0x20(sp)
10000514: ld s4, 0x18(sp)
10000518: ld s5, 0x10(sp)
1000051c: ld s6, 0x8(sp)
10000520: addi sp, sp, 0x40
10000524: ret
