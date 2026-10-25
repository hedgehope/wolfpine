# tensorprod/ame size=16 fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x30
10000070: sd s0, 0x28(sp)
10000074: sd s1, 0x20(sp)
10000078: sd s2, 0x18(sp)
1000007c: sd s3, 0x10(sp)
10000080: sd s4, 0x8(sp)
10000084: auipc a0, 0xaa
10000088: li a2, 0x10
1000008c: li a3, 0x8
10000090: auipc a1, 0x0
10000094: addi a0, a0, -0x6c4
10000098: addi a1, a1, 0x530
1000009c: sd a2, 0x40(a0)
100000a0: sd a3, 0x48(a0)
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
100000cc: li a1, 0x0
100000d0: addi a2, a0, 0x80
100000d4: auipc a3, 0x1
100000d8: auipc a4, 0x0
100000dc: addi a3, a3, -0x714
100000e0: addi a4, a4, 0x4e8
100000e4: li a5, 0x10
100000e8: mv a6, a3
100000ec: slli a7, a1, 0x2
100000f0: add a7, a7, a3
100000f4: addi a7, a7, 0x400
100000f8: mv t0, a4
100000fc: mv t1, a2
10000100: mv t2, a6
10000104: flw fa5, 0x0(t1)
10000108: addi t1, t1, 0x4
1000010c: fsw fa5, 0x0(t0)
10000110: fsw fa5, 0x0(t2)
10000114: addi t2, t2, 0x40
10000118: addi t0, t0, 0x4
1000011c: bne t2, a7, 0x10000104 <main+0x98>
10000120: addi a1, a1, 0x1
10000124: addi a6, a6, 0x4
10000128: addi a2, a2, 0x40
1000012c: addi a4, a4, 0x40
10000130: bne a1, a5, 0x100000ec <main+0x80>
10000134: auipc a1, 0x1
10000138: lui a2, 0x20
1000013c: lui a3, 0x45
10000140: addi a1, a1, -0x374
10000144: addiw a3, a3, -0x800
10000148: add a2, a1, a2
1000014c: add a3, a1, a3
10000150: sd zero, 0x0(a2)
10000154: addi a2, a2, 0x8
10000158: bne a2, a3, 0x10000150 <main+0xe4>
1000015c: li a3, 0x0
10000160: rdcycle a2
10000164: auipc t2, 0x0
10000168: li t3, 0x40
1000016c: auipc t4, 0x1
10000170: li a4, 0x400
10000174: addi t5, a1, 0x40
10000178: li a5, -0x10
1000017c: li a6, -0x4
10000180: li a7, 0x7f0
10000184: lui t0, 0xfffff
10000188: auipc t6, 0x45
1000018c: li t1, 0x80
10000190: addi s0, t2, 0x45c
10000194: addi t4, t4, -0x7ac
10000198: addi t2, t6, 0x438
1000019c: mx.tload.a td=0 base=s0 stride=t3 fmt=fp32 t=0
100001a0: mx.tload.b td=5 base=t4 stride=t3 fmt=fp32 t=0
100001a4: mx.tload.b td=1 base=a1 stride=a4 fmt=fp32 t=0
100001a8: mx.tload.b td=2 base=t5 stride=a4 fmt=fp32 t=0
100001ac: lui t3, 0xf
100001b0: li t4, 0x7b
100001b4: li t5, 0x7a
100001b8: li t6, 0x7d
100001bc: j 0x100001dc <main+0x170>
100001c0: mx.azero ad=3
100001c4: mx.mma ad=3 ta=0 tb=4 in=fp32 acc=fp32
100001c8: addi s0, s0, 0x80
100001cc: mx.astore as=2 base=s0 stride=a4 fmt=fp32
100001d0: addi a3, a3, 0x400
100001d4: addi a5, a5, 0x40
100001d8: bgeu a6, t6, 0x100002d4 <main+0x268>
100001dc: and s0, a3, t3
100001e0: beq a5, a7, 0x10000210 <main+0x1a4>
100001e4: slli s1, s0, 0x2
100001e8: addi s2, a5, 0x10
100001ec: add s3, a1, s1
100001f0: andi s1, s2, 0xc0
100001f4: slli s4, s1, 0x2
100001f8: add s3, s3, s4
100001fc: addi s3, s3, 0x80
10000200: mx.tload.b td=3 base=s3 stride=a4 fmt=fp32 t=0
10000204: mx.azero ad=0
10000208: mx.mma ad=0 ta=0 tb=1 in=fp32 acc=fp32
1000020c: beqz s2, 0x10000240 <main+0x1d4>
10000210: addi s1, a3, -0x100
10000214: andi s2, a5, 0xf0
10000218: and s1, s1, t0
1000021c: slli s2, s2, 0x2
10000220: slli s1, s1, 0x2
10000224: add s2, t2, s2
10000228: add s1, s2, s1
1000022c: mx.astore as=3 base=s1 stride=a4 fmt=fp32
10000230: addi s1, a6, 0x5
10000234: bltu t1, s1, 0x100002d4 <main+0x268>
10000238: addi s1, a5, 0x10
1000023c: andi s1, s1, 0xc0
10000240: slli s0, s0, 0x2
10000244: slli s1, s1, 0x2
10000248: add s2, a1, s0
1000024c: add s0, t2, s0
10000250: add s2, s2, s1
10000254: add s0, s0, s1
10000258: addi s1, s2, 0xc0
1000025c: mx.tload.b td=4 base=s1 stride=a4 fmt=fp32 t=0
10000260: mx.azero ad=1
10000264: mx.mma ad=1 ta=0 tb=2 in=fp32 acc=fp32
10000268: mx.astore as=0 base=s0 stride=a4 fmt=fp32
1000026c: addi a6, a6, 0x4
10000270: bltu t4, a6, 0x10000298 <main+0x22c>
10000274: addi s1, a3, 0x400
10000278: addi s2, a5, 0x50
1000027c: and s1, s1, t3
10000280: andi s2, s2, 0xc0
10000284: slli s1, s1, 0x2
10000288: add s1, a1, s1
1000028c: slli s2, s2, 0x2
10000290: add s1, s1, s2
10000294: mx.tload.b td=1 base=s1 stride=a4 fmt=fp32 t=0
10000298: mx.azero ad=2
1000029c: mx.mma ad=2 ta=0 tb=3 in=fp32 acc=fp32
100002a0: addi s1, s0, 0x40
100002a4: mx.astore as=1 base=s1 stride=a4 fmt=fp32
100002a8: bltu t5, a6, 0x100001c0 <main+0x154>
100002ac: addi s1, a3, 0x500
100002b0: addi s2, a5, 0x60
100002b4: and s1, s1, t3
100002b8: andi s2, s2, 0xd0
100002bc: slli s1, s1, 0x2
100002c0: add s1, a1, s1
100002c4: slli s2, s2, 0x2
100002c8: add s1, s1, s2
100002cc: mx.tload.b td=2 base=s1 stride=a4 fmt=fp32 t=0
100002d0: j 0x100001c0 <main+0x154>
100002d4: li a4, 0x0
100002d8: li a5, 0x0
100002dc: li a6, 0x40
100002e0: addi a3, a1, 0x400
100002e4: li a7, -0x4
100002e8: li t0, 0x7c
100002ec: auipc t5, 0x69
100002f0: li t1, 0x80
100002f4: li t2, 0x7b
100002f8: lui t3, 0x1
100002fc: li t4, 0x7a
10000300: mx.tload.a td=1 base=a1 stride=a6 fmt=fp32 t=0
10000304: addi t5, t5, 0x2d4
10000308: mx.tload.a td=2 base=a3 stride=a6 fmt=fp32 t=0
1000030c: li t6, 0x5
10000310: slli t6, t6, 0xa
10000314: li s0, 0x7d
10000318: li s1, -0x4
1000031c: j 0x10000344 <main+0x2d8>
10000320: mx.azero ad=3
10000324: mx.mma ad=3 ta=4 tb=5 in=fp32 acc=fp32
10000328: addi s2, s2, 0x200
1000032c: slli s2, s2, 0x2
10000330: add s2, t5, s2
10000334: mx.astore as=2 base=s2 stride=a6 fmt=fp32
10000338: addi a5, a5, 0x400
1000033c: add a4, a4, t3
10000340: bgeu s1, s0, 0x100003ec <main+0x380>
10000344: beq s1, t0, 0x10000370 <main+0x304>
10000348: add s2, a1, a4
1000034c: addi s2, s2, 0x7ff
10000350: addi s2, s2, 0x1
10000354: mx.tload.a td=3 base=s2 stride=a6 fmt=fp32 t=0
10000358: mx.azero ad=0
1000035c: mx.mma ad=0 ta=1 tb=5 in=fp32 acc=fp32
10000360: bne s1, a7, 0x10000370 <main+0x304>
10000364: li s3, 0x0
10000368: li s2, 0x0
1000036c: j 0x1000038c <main+0x320>
10000370: add s2, t5, a4
10000374: addi s2, s2, -0x400
10000378: mx.astore as=3 base=s2 stride=a6 fmt=fp32
1000037c: addi s4, s1, 0x5
10000380: mv s3, a4
10000384: mv s2, a5
10000388: bltu t1, s4, 0x100003ec <main+0x380>
1000038c: addi s4, s2, 0x300
10000390: slli s4, s4, 0x2
10000394: add s4, a1, s4
10000398: mx.tload.a td=4 base=s4 stride=a6 fmt=fp32 t=0
1000039c: add s4, t5, s3
100003a0: mx.azero ad=1
100003a4: mx.mma ad=1 ta=2 tb=5 in=fp32 acc=fp32
100003a8: mx.astore as=0 base=s4 stride=a6 fmt=fp32
100003ac: addi s1, s1, 0x4
100003b0: bltu t2, s1, 0x100003c0 <main+0x354>
100003b4: add s4, a1, s3
100003b8: add s4, s4, t3
100003bc: mx.tload.a td=1 base=s4 stride=a6 fmt=fp32 t=0
100003c0: mx.azero ad=2
100003c4: mx.mma ad=2 ta=3 tb=5 in=fp32 acc=fp32
100003c8: slli s4, s2, 0x2
100003cc: add s4, t5, s4
100003d0: addi s4, s4, 0x400
100003d4: mx.astore as=1 base=s4 stride=a6 fmt=fp32
100003d8: bltu t4, s1, 0x10000320 <main+0x2b4>
100003dc: add s3, a1, s3
100003e0: add s3, s3, t6
100003e4: mx.tload.a td=2 base=s3 stride=a6 fmt=fp32 t=0
100003e8: j 0x10000320 <main+0x2b4>
100003ec: li a4, 0x0
100003f0: li a5, 0x0
100003f4: li a6, 0x40
100003f8: li a7, -0x4
100003fc: li t0, 0x7c
10000400: auipc t5, 0x89
10000404: li t1, 0x80
10000408: li t2, 0x7b
1000040c: lui t3, 0x1
10000410: li t4, 0x7a
10000414: mx.tload.b td=1 base=a1 stride=a6 fmt=fp32 t=0
10000418: addi t5, t5, 0x1c0
1000041c: mx.tload.b td=2 base=a3 stride=a6 fmt=fp32 t=0
10000420: li a3, 0x5
10000424: slli a3, a3, 0xa
10000428: li t6, 0x7d
1000042c: li s0, -0x4
10000430: j 0x10000458 <main+0x3ec>
10000434: mx.azero ad=3
10000438: mx.mma ad=3 ta=0 tb=4 in=fp32 acc=fp32
1000043c: addi s1, s1, 0x200
10000440: slli s1, s1, 0x2
10000444: add s1, t5, s1
10000448: mx.astore as=2 base=s1 stride=a6 fmt=fp32
1000044c: addi a5, a5, 0x400
10000450: add a4, a4, t3
10000454: bgeu s0, t6, 0x10000500 <main+0x494>
10000458: beq s0, t0, 0x10000484 <main+0x418>
1000045c: add s1, a1, a4
10000460: addi s1, s1, 0x7ff
10000464: addi s1, s1, 0x1
10000468: mx.tload.b td=3 base=s1 stride=a6 fmt=fp32 t=0
1000046c: mx.azero ad=0
10000470: mx.mma ad=0 ta=0 tb=1 in=fp32 acc=fp32
10000474: bne s0, a7, 0x10000484 <main+0x418>
10000478: li s2, 0x0
1000047c: li s1, 0x0
10000480: j 0x100004a0 <main+0x434>
10000484: add s1, t5, a4
10000488: addi s1, s1, -0x400
1000048c: mx.astore as=3 base=s1 stride=a6 fmt=fp32
10000490: addi s3, s0, 0x5
10000494: mv s2, a4
10000498: mv s1, a5
1000049c: bltu t1, s3, 0x10000500 <main+0x494>
100004a0: addi s3, s1, 0x300
100004a4: slli s3, s3, 0x2
100004a8: add s3, a1, s3
100004ac: mx.tload.b td=4 base=s3 stride=a6 fmt=fp32 t=0
100004b0: add s3, t5, s2
100004b4: mx.azero ad=1
100004b8: mx.mma ad=1 ta=0 tb=2 in=fp32 acc=fp32
100004bc: mx.astore as=0 base=s3 stride=a6 fmt=fp32
100004c0: addi s0, s0, 0x4
100004c4: bltu t2, s0, 0x100004d4 <main+0x468>
100004c8: add s3, a1, s2
100004cc: add s3, s3, t3
100004d0: mx.tload.b td=1 base=s3 stride=a6 fmt=fp32 t=0
100004d4: mx.azero ad=2
100004d8: mx.mma ad=2 ta=0 tb=3 in=fp32 acc=fp32
100004dc: slli s3, s1, 0x2
100004e0: add s3, t5, s3
100004e4: addi s3, s3, 0x400
100004e8: mx.astore as=1 base=s3 stride=a6 fmt=fp32
100004ec: bltu t4, s0, 0x10000434 <main+0x3c8>
100004f0: add s2, a1, s2
100004f4: add s2, s2, a3
100004f8: mx.tload.b td=2 base=s2 stride=a6 fmt=fp32 t=0
100004fc: j 0x10000434 <main+0x3c8>
10000500: mx.azero ad=0
10000504: mx.azero ad=1
10000508: mx.azero ad=2
1000050c: mx.azero ad=3
10000510: rdcycle a1
10000514: sub a1, a1, a2
10000518: sd a1, 0x0(a0)
1000051c: li a0, 0x0
10000520: ld s0, 0x28(sp)
10000524: ld s1, 0x20(sp)
10000528: ld s2, 0x18(sp)
1000052c: ld s3, 0x10(sp)
10000530: ld s4, 0x8(sp)
10000534: addi sp, sp, 0x30
10000538: ret
