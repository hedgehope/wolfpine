# tensorprod/ame size=8 fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x30
10000070: sd s0, 0x28(sp)
10000074: sd s1, 0x20(sp)
10000078: sd s2, 0x18(sp)
1000007c: sd s3, 0x10(sp)
10000080: sd s4, 0x8(sp)
10000084: sd s5, 0x0(sp)
10000088: auipc a0, 0x1c
1000008c: li a2, 0x8
10000090: auipc a1, 0x0
10000094: addi a0, a0, -0x6c8
10000098: addi a1, a1, 0x530
1000009c: sd a2, 0x40(a0)
100000a0: sd a2, 0x48(a0)
100000a4: addi a2, a1, 0x400
100000a8: sd zero, 0x0(a1)
100000ac: addi a1, a1, 0x8
100000b0: bne a1, a2, 0x100000a8 <main+0x3c>
100000b4: auipc a1, 0x1
100000b8: addi a1, a1, -0x6f4
100000bc: addi a2, a1, 0x400
100000c0: sd zero, 0x0(a1)
100000c4: addi a1, a1, 0x8
100000c8: bne a1, a2, 0x100000c0 <main+0x54>
100000cc: li a6, 0x0
100000d0: li a2, 0x1
100000d4: li a1, 0x220
100000d8: auipc a3, 0x0
100000dc: auipc a4, 0x1
100000e0: addi a3, a3, 0x4e8
100000e4: addi a4, a4, -0x71c
100000e8: addi a5, a0, 0x9c
100000ec: mul a6, a6, a1
100000f0: add t0, a3, a6
100000f4: add a7, a4, a6
100000f8: addi a6, t0, 0x10
100000fc: addi a7, a7, 0x100
10000100: addi t0, t0, 0x210
10000104: mv t1, a5
10000108: flw fa5, -0x1c(t1)
1000010c: flw fa4, -0x18(t1)
10000110: flw fa3, -0x14(t1)
10000114: flw fa2, -0x10(t1)
10000118: fsw fa5, -0x100(a7)
1000011c: fsw fa4, -0xc0(a7)
10000120: fsw fa3, -0x80(a7)
10000124: fsw fa5, -0x10(a6)
10000128: fsw fa4, -0xc(a6)
1000012c: fsw fa3, -0x8(a6)
10000130: fsw fa2, -0x4(a6)
10000134: fsw fa2, -0x40(a7)
10000138: flw fa5, -0xc(t1)
1000013c: flw fa4, -0x8(t1)
10000140: flw fa3, -0x4(t1)
10000144: flw fa2, 0x0(t1)
10000148: fsw fa5, 0x0(a7)
1000014c: fsw fa4, 0x40(a7)
10000150: fsw fa3, 0x80(a7)
10000154: fsw fa5, 0x0(a6)
10000158: fsw fa4, 0x4(a6)
1000015c: fsw fa3, 0x8(a6)
10000160: fsw fa2, 0xc(a6)
10000164: fsw fa2, 0xc0(a7)
10000168: addi a6, a6, 0x40
1000016c: addi a7, a7, 0x4
10000170: addi t1, t1, 0x20
10000174: bne a6, t0, 0x10000108 <main+0x9c>
10000178: andi a7, a2, 0x1
1000017c: li a6, 0x1
10000180: li a2, 0x0
10000184: bnez a7, 0x100000ec <main+0x80>
10000188: auipc a1, 0x1
1000018c: lui a2, 0x4
10000190: li a3, 0x13
10000194: addi a1, a1, -0x3c8
10000198: slli a3, a3, 0xb
1000019c: add a2, a1, a2
100001a0: add a3, a1, a3
100001a4: sd zero, 0x0(a2)
100001a8: addi a2, a2, 0x8
100001ac: bne a2, a3, 0x100001a4 <main+0x138>
100001b0: li a3, 0x0
100001b4: rdcycle a2
100001b8: auipc t3, 0x0
100001bc: li t4, 0x40
100001c0: auipc t5, 0x1
100001c4: li a4, 0x100
100001c8: addi t6, a1, 0x40
100001cc: li a5, -0x100
100001d0: li a6, -0x4
100001d4: li a7, 0xc
100001d8: auipc s0, 0xa
100001dc: li t0, 0x10
100001e0: li t1, 0xb
100001e4: lui t2, 0x1
100001e8: addi s1, t3, 0x408
100001ec: addi t5, t5, -0x800
100001f0: addi t3, s0, 0x3e8
100001f4: mx.tload.a td=0 base=s1 stride=t4 fmt=fp32 t=0
100001f8: mx.tload.b td=5 base=t5 stride=t4 fmt=fp32 t=0
100001fc: mx.tload.b td=1 base=a1 stride=a4 fmt=fp32 t=0
10000200: mx.tload.b td=2 base=t6 stride=a4 fmt=fp32 t=0
10000204: li t4, 0xa
10000208: addiw t5, t2, 0x40
1000020c: li t6, 0xd
10000210: li s0, -0x4
10000214: j 0x10000234 <main+0x1c8>
10000218: mx.azero ad=3
1000021c: mx.mma ad=3 ta=0 tb=4 in=fp32 acc=fp32
10000220: addi s1, s1, 0x80
10000224: mx.astore as=2 base=s1 stride=a4 fmt=fp32
10000228: add a3, a3, t2
1000022c: addi a5, a5, 0x400
10000230: bgeu s0, t6, 0x100002c8 <main+0x25c>
10000234: beq s0, a7, 0x10000258 <main+0x1ec>
10000238: add s1, a1, a3
1000023c: addi s1, s1, 0x80
10000240: mx.tload.b td=3 base=s1 stride=a4 fmt=fp32 t=0
10000244: mx.azero ad=0
10000248: mx.mma ad=0 ta=0 tb=1 in=fp32 acc=fp32
1000024c: bne s0, a6, 0x10000258 <main+0x1ec>
10000250: li s1, 0x0
10000254: j 0x10000278 <main+0x20c>
10000258: andi s1, a5, -0x400
1000025c: slli s1, s1, 0x2
10000260: add s1, t3, s1
10000264: addi s1, s1, 0xc0
10000268: mx.astore as=3 base=s1 stride=a4 fmt=fp32
1000026c: addi s2, s0, 0x5
10000270: mv s1, a3
10000274: bltu t0, s2, 0x100002c8 <main+0x25c>
10000278: add s2, a1, s1
1000027c: add s1, t3, s1
10000280: addi s3, s2, 0xc0
10000284: mx.tload.b td=4 base=s3 stride=a4 fmt=fp32 t=0
10000288: mx.azero ad=1
1000028c: mx.mma ad=1 ta=0 tb=2 in=fp32 acc=fp32
10000290: mx.astore as=0 base=s1 stride=a4 fmt=fp32
10000294: addi s0, s0, 0x4
10000298: bltu t1, s0, 0x100002a4 <main+0x238>
1000029c: add s2, s2, t2
100002a0: mx.tload.b td=1 base=s2 stride=a4 fmt=fp32 t=0
100002a4: mx.azero ad=2
100002a8: mx.mma ad=2 ta=0 tb=3 in=fp32 acc=fp32
100002ac: addi s2, s1, 0x40
100002b0: mx.astore as=1 base=s2 stride=a4 fmt=fp32
100002b4: bltu t4, s0, 0x10000218 <main+0x1ac>
100002b8: add s2, a1, a3
100002bc: add s2, s2, t5
100002c0: mx.tload.b td=2 base=s2 stride=a4 fmt=fp32 t=0
100002c4: j 0x10000218 <main+0x1ac>
100002c8: li a3, 0x0
100002cc: li a4, 0x0
100002d0: li a5, 0x40
100002d4: addi t5, a1, 0x400
100002d8: li a6, -0x4
100002dc: li a7, 0xc
100002e0: auipc t4, 0xf
100002e4: li t0, 0x10
100002e8: li t1, 0xb
100002ec: lui t2, 0x1
100002f0: li t3, 0xa
100002f4: mx.tload.a td=1 base=a1 stride=a5 fmt=fp32 t=0
100002f8: addi t4, t4, 0x2e0
100002fc: mx.tload.a td=2 base=t5 stride=a5 fmt=fp32 t=0
10000300: li t5, 0x5
10000304: slli t5, t5, 0xa
10000308: li t6, 0xd
1000030c: li s0, -0x4
10000310: j 0x10000338 <main+0x2cc>
10000314: mx.azero ad=3
10000318: mx.mma ad=3 ta=4 tb=5 in=fp32 acc=fp32
1000031c: addi s1, s1, 0x200
10000320: slli s1, s1, 0x2
10000324: add s1, t4, s1
10000328: mx.astore as=2 base=s1 stride=a5 fmt=fp32
1000032c: addi a4, a4, 0x400
10000330: add a3, a3, t2
10000334: bgeu s0, t6, 0x100003e0 <main+0x374>
10000338: beq s0, a7, 0x10000364 <main+0x2f8>
1000033c: add s1, a1, a3
10000340: addi s1, s1, 0x7ff
10000344: addi s1, s1, 0x1
10000348: mx.tload.a td=3 base=s1 stride=a5 fmt=fp32 t=0
1000034c: mx.azero ad=0
10000350: mx.mma ad=0 ta=1 tb=5 in=fp32 acc=fp32
10000354: bne s0, a6, 0x10000364 <main+0x2f8>
10000358: li s2, 0x0
1000035c: li s1, 0x0
10000360: j 0x10000380 <main+0x314>
10000364: add s1, t4, a3
10000368: addi s1, s1, -0x400
1000036c: mx.astore as=3 base=s1 stride=a5 fmt=fp32
10000370: addi s3, s0, 0x5
10000374: mv s2, a3
10000378: mv s1, a4
1000037c: bltu t0, s3, 0x100003e0 <main+0x374>
10000380: addi s3, s1, 0x300
10000384: slli s3, s3, 0x2
10000388: add s3, a1, s3
1000038c: mx.tload.a td=4 base=s3 stride=a5 fmt=fp32 t=0
10000390: add s3, t4, s2
10000394: mx.azero ad=1
10000398: mx.mma ad=1 ta=2 tb=5 in=fp32 acc=fp32
1000039c: mx.astore as=0 base=s3 stride=a5 fmt=fp32
100003a0: addi s0, s0, 0x4
100003a4: bltu t1, s0, 0x100003b4 <main+0x348>
100003a8: add s3, a1, s2
100003ac: add s3, s3, t2
100003b0: mx.tload.a td=1 base=s3 stride=a5 fmt=fp32 t=0
100003b4: mx.azero ad=2
100003b8: mx.mma ad=2 ta=3 tb=5 in=fp32 acc=fp32
100003bc: slli s3, s1, 0x2
100003c0: add s3, t4, s3
100003c4: addi s3, s3, 0x400
100003c8: mx.astore as=1 base=s3 stride=a5 fmt=fp32
100003cc: bltu t3, s0, 0x10000314 <main+0x2a8>
100003d0: add s2, a1, s2
100003d4: add s2, s2, t5
100003d8: mx.tload.a td=2 base=s2 stride=a5 fmt=fp32 t=0
100003dc: j 0x10000314 <main+0x2a8>
100003e0: li a3, 0x0
100003e4: li a4, 0x20
100003e8: addi t5, a1, 0x200
100003ec: addi t3, a1, 0x7ff
100003f0: li a5, -0x4
100003f4: li a6, 0x180
100003f8: li a7, 0x1c
100003fc: auipc t4, 0x13
10000400: li t0, 0x40
10000404: li t1, 0x1b
10000408: li t2, 0x1a
1000040c: li t6, 0x1
10000410: mx.tload.b td=1 base=a1 stride=a4 fmt=fp32 t=0
10000414: addi t3, t3, 0x1
10000418: addi t4, t4, 0x1c4
1000041c: mx.tload.b td=2 base=t5 stride=a4 fmt=fp32 t=0
10000420: slli t5, t6, 0xb
10000424: lui t6, 0x1
10000428: li s0, 0x1d
1000042c: li s1, -0x4
10000430: j 0x10000454 <main+0x3e8>
10000434: mx.azero ad=3
10000438: mx.mma ad=3 ta=0 tb=4 in=fp32 acc=fp32
1000043c: add s2, s2, t5
10000440: mx.astore as=2 base=s2 stride=t0 fmt=fp32
10000444: add t3, t3, t5
10000448: add a3, a3, t6
1000044c: addi a6, a6, 0x200
10000450: bgeu s1, s0, 0x100004dc <main+0x470>
10000454: add s2, t4, a3
10000458: beq s1, a7, 0x1000047c <main+0x410>
1000045c: addi s3, t3, -0x400
10000460: mx.tload.b td=3 base=s3 stride=a4 fmt=fp32 t=0
10000464: mx.azero ad=0
10000468: mx.mma ad=0 ta=0 tb=1 in=fp32 acc=fp32
1000046c: bne s1, a5, 0x1000047c <main+0x410>
10000470: li s3, 0x0
10000474: li s4, 0x180
10000478: j 0x10000494 <main+0x428>
1000047c: addi s3, s2, -0x400
10000480: mx.astore as=3 base=s3 stride=t0 fmt=fp32
10000484: addi s5, s1, 0x5
10000488: mv s3, a3
1000048c: mv s4, a6
10000490: bltu a4, s5, 0x100004dc <main+0x470>
10000494: slli s4, s4, 0x2
10000498: add s3, t4, s3
1000049c: add s4, a1, s4
100004a0: mx.tload.b td=4 base=s4 stride=a4 fmt=fp32 t=0
100004a4: mx.azero ad=1
100004a8: mx.mma ad=1 ta=0 tb=2 in=fp32 acc=fp32
100004ac: mx.astore as=0 base=s3 stride=t0 fmt=fp32
100004b0: addi s1, s1, 0x4
100004b4: bltu t1, s1, 0x100004bc <main+0x450>
100004b8: mx.tload.b td=1 base=t3 stride=a4 fmt=fp32 t=0
100004bc: mx.azero ad=2
100004c0: mx.mma ad=2 ta=0 tb=3 in=fp32 acc=fp32
100004c4: addi s3, s2, 0x400
100004c8: mx.astore as=1 base=s3 stride=t0 fmt=fp32
100004cc: bltu t2, s1, 0x10000434 <main+0x3c8>
100004d0: addi s3, t3, 0x200
100004d4: mx.tload.b td=2 base=s3 stride=a4 fmt=fp32 t=0
100004d8: j 0x10000434 <main+0x3c8>
100004dc: mx.azero ad=0
100004e0: mx.azero ad=1
100004e4: mx.azero ad=2
100004e8: mx.azero ad=3
100004ec: rdcycle a1
100004f0: sub a1, a1, a2
100004f4: sd a1, 0x0(a0)
100004f8: li a0, 0x0
100004fc: ld s0, 0x28(sp)
10000500: ld s1, 0x20(sp)
10000504: ld s2, 0x18(sp)
10000508: ld s3, 0x10(sp)
1000050c: ld s4, 0x8(sp)
10000510: ld s5, 0x0(sp)
10000514: addi sp, sp, 0x30
10000518: ret
