# tensorprod/ame size=12 fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x30
10000070: sd s0, 0x28(sp)
10000074: sd s1, 0x20(sp)
10000078: sd s2, 0x18(sp)
1000007c: sd s3, 0x10(sp)
10000080: sd s4, 0x8(sp)
10000084: sd s5, 0x0(sp)
10000088: auipc a0, 0x53
1000008c: li a2, 0xc
10000090: li a3, 0x8
10000094: auipc a1, 0x0
10000098: addi a0, a0, -0x588
1000009c: addi a1, a1, 0x66c
100000a0: sd a2, 0x40(a0)
100000a4: sd a3, 0x48(a0)
100000a8: addi a2, a1, 0x400
100000ac: sd zero, 0x0(a1)
100000b0: addi a1, a1, 0x8
100000b4: bne a1, a2, 0x100000ac <main+0x40>
100000b8: auipc a1, 0x1
100000bc: addi a1, a1, -0x5b8
100000c0: addi a2, a1, 0x400
100000c4: sd zero, 0x0(a1)
100000c8: addi a1, a1, 0x8
100000cc: bne a1, a2, 0x100000c4 <main+0x58>
100000d0: auipc a2, 0x0
100000d4: addi a1, a0, 0xac
100000d8: auipc a3, 0x1
100000dc: addi a4, a2, 0x630
100000e0: addi a2, a3, -0x458
100000e4: addi a3, a4, 0x18
100000e8: addi a4, a4, 0x318
100000ec: flw fa5, -0x2c(a1)
100000f0: flw fa4, -0x28(a1)
100000f4: flw fa3, -0x24(a1)
100000f8: flw fa2, -0x20(a1)
100000fc: fsw fa5, -0x180(a2)
10000100: fsw fa4, -0x140(a2)
10000104: fsw fa3, -0x100(a2)
10000108: fsw fa5, -0x18(a3)
1000010c: fsw fa4, -0x14(a3)
10000110: fsw fa3, -0x10(a3)
10000114: fsw fa2, -0xc(a3)
10000118: fsw fa2, -0xc0(a2)
1000011c: flw fa5, -0x1c(a1)
10000120: flw fa4, -0x18(a1)
10000124: flw fa3, -0x14(a1)
10000128: flw fa2, -0x10(a1)
1000012c: fsw fa5, -0x80(a2)
10000130: fsw fa4, -0x40(a2)
10000134: fsw fa3, 0x0(a2)
10000138: fsw fa5, -0x8(a3)
1000013c: fsw fa4, -0x4(a3)
10000140: fsw fa3, 0x0(a3)
10000144: fsw fa2, 0x4(a3)
10000148: fsw fa2, 0x40(a2)
1000014c: flw fa5, -0xc(a1)
10000150: flw fa4, -0x8(a1)
10000154: flw fa3, -0x4(a1)
10000158: flw fa2, 0x0(a1)
1000015c: fsw fa5, 0x80(a2)
10000160: fsw fa4, 0xc0(a2)
10000164: fsw fa3, 0x100(a2)
10000168: fsw fa5, 0x8(a3)
1000016c: fsw fa4, 0xc(a3)
10000170: fsw fa3, 0x10(a3)
10000174: fsw fa2, 0x14(a3)
10000178: fsw fa2, 0x140(a2)
1000017c: addi a3, a3, 0x40
10000180: addi a1, a1, 0x30
10000184: addi a2, a2, 0x4
10000188: bne a3, a4, 0x100000ec <main+0x80>
1000018c: auipc a1, 0x1
10000190: li a2, 0x1b
10000194: lui a3, 0x1e
10000198: addi a1, a1, -0x28c
1000019c: slli a2, a2, 0xb
100001a0: addiw a3, a3, -0x400
100001a4: add a2, a1, a2
100001a8: add a3, a1, a3
100001ac: sd zero, 0x0(a2)
100001b0: addi a2, a2, 0x8
100001b4: bne a2, a3, 0x100001ac <main+0x140>
100001b8: li t4, 0x0
100001bc: li a5, 0x1b
100001c0: rdcycle a2
100001c4: auipc a6, 0x0
100001c8: li a7, 0x40
100001cc: auipc t0, 0x1
100001d0: li a3, 0x240
100001d4: addi t1, a1, 0x40
100001d8: li a4, 0x39
100001dc: auipc t2, 0x1f
100001e0: slli a5, a5, 0x8
100001e4: addi t3, a6, 0x53c
100001e8: addi t0, t0, -0x6cc
100001ec: addi a6, t2, -0x6dc
100001f0: mx.tload.a td=0 base=t3 stride=a7 fmt=fp32 t=0
100001f4: mx.tload.b td=5 base=t0 stride=a7 fmt=fp32 t=0
100001f8: mx.tload.b td=1 base=a1 stride=a3 fmt=fp32 t=0
100001fc: mx.tload.b td=2 base=t1 stride=a3 fmt=fp32 t=0
10000200: li a7, 0x48
10000204: li t0, 0x43
10000208: li t1, 0x42
1000020c: li t2, 0x45
10000210: j 0x10000254 <main+0x1e8>
10000214: mx.azero ad=3
10000218: mx.mma ad=3 ta=0 tb=4 in=fp32 acc=fp32
1000021c: andi t4, t5, 0xff
10000220: mul t4, t4, a4
10000224: srli t4, t4, 0x9
10000228: mul t6, t4, a5
1000022c: slli s0, t4, 0x3
10000230: subw t4, t5, t4
10000234: add t6, a6, t6
10000238: subw t4, t4, s0
1000023c: andi t4, t4, 0xff
10000240: slli t4, t4, 0x6
10000244: add t4, t6, t4
10000248: mx.astore as=2 base=t4 stride=a3 fmt=fp32
1000024c: addi t4, t3, 0x4
10000250: bgeu t3, t2, 0x1000040c <main+0x3a0>
10000254: mv t3, t4
10000258: li t4, 0x48
1000025c: beq t3, a7, 0x100002a4 <main+0x238>
10000260: ori t4, t3, 0x2
10000264: andi t5, t4, 0xff
10000268: mul t5, t5, a4
1000026c: srli t5, t5, 0x9
10000270: mul t6, t5, a5
10000274: slli s0, t5, 0x3
10000278: subw t4, t4, t5
1000027c: add t6, a1, t6
10000280: subw t4, t4, s0
10000284: andi t4, t4, 0xff
10000288: slli t4, t4, 0x6
1000028c: add t4, t6, t4
10000290: mx.tload.b td=3 base=t4 stride=a3 fmt=fp32 t=0
10000294: mx.azero ad=0
10000298: mx.mma ad=0 ta=0 tb=1 in=fp32 acc=fp32
1000029c: mv t4, t3
100002a0: beqz t3, 0x100003fc <main+0x390>
100002a4: addi t5, t4, -0x1
100002a8: andi t6, t5, 0xff
100002ac: mul t6, t6, a4
100002b0: srli t6, t6, 0x9
100002b4: mul s0, t6, a5
100002b8: slli s1, t6, 0x3
100002bc: subw t5, t5, t6
100002c0: add s0, a6, s0
100002c4: subw t5, t5, s1
100002c8: andi t5, t5, 0xff
100002cc: slli t5, t5, 0x6
100002d0: add t5, s0, t5
100002d4: mx.astore as=3 base=t5 stride=a3 fmt=fp32
100002d8: addi t6, t3, 0x1
100002dc: bltu a7, t6, 0x1000040c <main+0x3a0>
100002e0: ori t5, t4, 0x2
100002e4: ori s0, t4, 0x3
100002e8: andi s1, t4, 0xff
100002ec: andi s2, s0, 0xff
100002f0: mul s1, s1, a4
100002f4: mul s2, s2, a4
100002f8: srli s1, s1, 0x9
100002fc: srli s2, s2, 0x9
10000300: slli s3, s1, 0x3
10000304: subw s4, t4, s1
10000308: subw s3, s4, s3
1000030c: slli s4, s2, 0x3
10000310: subw s0, s0, s2
10000314: subw s0, s0, s4
10000318: mul s1, s1, a5
1000031c: mul s2, s2, a5
10000320: add s1, a6, s1
10000324: add s2, a1, s2
10000328: andi s3, s3, 0xff
1000032c: andi s0, s0, 0xff
10000330: slli s3, s3, 0x6
10000334: slli s0, s0, 0x6
10000338: add s1, s1, s3
1000033c: add s0, s2, s0
10000340: mx.tload.b td=4 base=s0 stride=a3 fmt=fp32 t=0
10000344: mx.azero ad=1
10000348: mx.mma ad=1 ta=0 tb=2 in=fp32 acc=fp32
1000034c: mx.astore as=0 base=s1 stride=a3 fmt=fp32
10000350: bltu t0, t3, 0x10000388 <main+0x31c>
10000354: addi s0, t4, 0x4
10000358: andi s1, s0, 0xff
1000035c: mul s1, s1, a4
10000360: srli s1, s1, 0x9
10000364: mul s2, s1, a5
10000368: slli s3, s1, 0x3
1000036c: subw s0, s0, s1
10000370: add s2, a1, s2
10000374: subw s0, s0, s3
10000378: andi s0, s0, 0xff
1000037c: slli s0, s0, 0x6
10000380: add s0, s2, s0
10000384: mx.tload.b td=1 base=s0 stride=a3 fmt=fp32 t=0
10000388: mx.azero ad=2
1000038c: mx.mma ad=2 ta=0 tb=3 in=fp32 acc=fp32
10000390: andi s0, t6, 0xff
10000394: mul s0, s0, a4
10000398: srli s0, s0, 0x9
1000039c: mul s1, s0, a5
100003a0: slli s2, s0, 0x3
100003a4: subw t6, t6, s0
100003a8: add s1, a6, s1
100003ac: subw t6, t6, s2
100003b0: andi t6, t6, 0xff
100003b4: slli t6, t6, 0x6
100003b8: add t6, s1, t6
100003bc: mx.astore as=1 base=t6 stride=a3 fmt=fp32
100003c0: bltu t1, t3, 0x10000214 <main+0x1a8>
100003c4: addi t4, t4, 0x5
100003c8: andi t6, t4, 0xff
100003cc: mul t6, t6, a4
100003d0: srli t6, t6, 0x9
100003d4: mul s0, t6, a5
100003d8: slli s1, t6, 0x3
100003dc: subw t4, t4, t6
100003e0: add s0, a1, s0
100003e4: subw t4, t4, s1
100003e8: andi t4, t4, 0xff
100003ec: slli t4, t4, 0x6
100003f0: add t4, s0, t4
100003f4: mx.tload.b td=2 base=t4 stride=a3 fmt=fp32 t=0
100003f8: j 0x10000214 <main+0x1a8>
100003fc: li t4, 0x0
10000400: li t6, 0x1
10000404: li t5, 0x2
10000408: j 0x100002e4 <main+0x278>
1000040c: li a4, 0x0
10000410: li a5, 0x0
10000414: li a3, 0x3
10000418: li a6, 0x30
1000041c: addi t5, a1, 0x300
10000420: li a7, -0x4
10000424: auipc t4, 0x2e
10000428: li t0, 0x44
1000042c: li t1, 0x40
10000430: li t2, 0x48
10000434: li t3, 0x43
10000438: slli a3, a3, 0xa
1000043c: mx.tload.a td=1 base=a1 stride=a6 fmt=fp32 t=0
10000440: addi t4, t4, 0x2dc
10000444: mx.tload.a td=2 base=t5 stride=a6 fmt=fp32 t=0
10000448: li t5, 0x42
1000044c: lui t6, 0x1
10000450: li s0, 0x45
10000454: mv s1, t4
10000458: li s2, -0x4
1000045c: j 0x10000484 <main+0x418>
10000460: mx.azero ad=3
10000464: mx.mma ad=3 ta=4 tb=5 in=fp32 acc=fp32
10000468: addi s3, s1, 0x7ff
1000046c: addi s3, s3, 0x1
10000470: mx.astore as=2 base=s3 stride=t1 fmt=fp32
10000474: add s1, s1, t6
10000478: add a5, a5, a3
1000047c: add a4, a4, t6
10000480: bgeu s2, s0, 0x10000514 <main+0x4a8>
10000484: beq s2, t0, 0x100004ac <main+0x440>
10000488: add s3, a1, a5
1000048c: addi s3, s3, 0x600
10000490: mx.tload.a td=3 base=s3 stride=a6 fmt=fp32 t=0
10000494: mx.azero ad=0
10000498: mx.mma ad=0 ta=1 tb=5 in=fp32 acc=fp32
1000049c: bne s2, a7, 0x100004ac <main+0x440>
100004a0: li s3, 0x0
100004a4: li s4, 0x0
100004a8: j 0x100004c4 <main+0x458>
100004ac: addi s3, s1, -0x400
100004b0: mx.astore as=3 base=s3 stride=t1 fmt=fp32
100004b4: addi s5, s2, 0x5
100004b8: mv s3, a4
100004bc: mv s4, a5
100004c0: bltu t2, s5, 0x10000514 <main+0x4a8>
100004c4: add s4, a1, s4
100004c8: add s5, t4, s3
100004cc: addi s3, s4, 0x7ff
100004d0: addi s4, s3, 0x101
100004d4: mx.tload.a td=4 base=s4 stride=a6 fmt=fp32 t=0
100004d8: mx.azero ad=1
100004dc: mx.mma ad=1 ta=2 tb=5 in=fp32 acc=fp32
100004e0: mx.astore as=0 base=s5 stride=t1 fmt=fp32
100004e4: addi s2, s2, 0x4
100004e8: bltu t3, s2, 0x100004f4 <main+0x488>
100004ec: addi s4, s3, 0x401
100004f0: mx.tload.a td=1 base=s4 stride=a6 fmt=fp32 t=0
100004f4: mx.azero ad=2
100004f8: mx.mma ad=2 ta=3 tb=5 in=fp32 acc=fp32
100004fc: addi s4, s1, 0x400
10000500: mx.astore as=1 base=s4 stride=t1 fmt=fp32
10000504: bltu t5, s2, 0x10000460 <main+0x3f4>
10000508: addi s3, s3, 0x701
1000050c: mx.tload.a td=2 base=s3 stride=a6 fmt=fp32 t=0
10000510: j 0x10000460 <main+0x3f4>
10000514: li a4, 0x0
10000518: li a6, 0x0
1000051c: li a5, 0x30
10000520: addi t4, a1, 0x240
10000524: auipc t3, 0x40
10000528: li a7, -0x4
1000052c: li t0, 0x5c
10000530: li t1, 0x40
10000534: li t2, 0x60
10000538: mx.tload.b td=1 base=a1 stride=a5 fmt=fp32 t=0
1000053c: addi t3, t3, 0x1dc
10000540: mx.tload.b td=2 base=t4 stride=a5 fmt=fp32 t=0
10000544: addi t6, t3, 0x300
10000548: li t4, 0x5b
1000054c: li t5, 0x5a
10000550: li s0, 0x5d
10000554: li s1, -0x4
10000558: j 0x10000584 <main+0x518>
1000055c: mx.azero ad=3
10000560: mx.mma ad=3 ta=0 tb=4 in=fp32 acc=fp32
10000564: addi s2, s2, 0x600
10000568: addi t6, t6, 0x7ff
1000056c: mx.astore as=2 base=s2 stride=t1 fmt=fp32
10000570: addi t6, t6, 0x401
10000574: add a6, a3, a6
10000578: addi a6, a6, -0x300
1000057c: add a4, a4, a3
10000580: bgeu s1, s0, 0x10000614 <main+0x5a8>
10000584: beq s1, t0, 0x100005ac <main+0x540>
10000588: add s2, a1, a6
1000058c: addi s2, s2, 0x480
10000590: mx.tload.b td=3 base=s2 stride=a5 fmt=fp32 t=0
10000594: mx.azero ad=0
10000598: mx.mma ad=0 ta=0 tb=1 in=fp32 acc=fp32
1000059c: bne s1, a7, 0x100005ac <main+0x540>
100005a0: li s2, 0x0
100005a4: li s3, 0x0
100005a8: j 0x100005c4 <main+0x558>
100005ac: addi s2, t6, -0x600
100005b0: mx.astore as=3 base=s2 stride=t1 fmt=fp32
100005b4: addi s4, s1, 0x5
100005b8: mv s2, a4
100005bc: mv s3, a6
100005c0: bltu t2, s4, 0x10000614 <main+0x5a8>
100005c4: add s3, a1, s3
100005c8: add s2, t3, s2
100005cc: addi s4, s3, 0x6c0
100005d0: mx.tload.b td=4 base=s4 stride=a5 fmt=fp32 t=0
100005d4: mx.azero ad=1
100005d8: mx.mma ad=1 ta=0 tb=2 in=fp32 acc=fp32
100005dc: mx.astore as=0 base=s2 stride=t1 fmt=fp32
100005e0: addi s1, s1, 0x4
100005e4: bltu t4, s1, 0x100005f4 <main+0x588>
100005e8: addi s4, s3, 0x7ff
100005ec: addi s4, s4, 0x101
100005f0: mx.tload.b td=1 base=s4 stride=a5 fmt=fp32 t=0
100005f4: mx.azero ad=2
100005f8: mx.mma ad=2 ta=0 tb=3 in=fp32 acc=fp32
100005fc: mx.astore as=1 base=t6 stride=t1 fmt=fp32
10000600: bltu t5, s1, 0x1000055c <main+0x4f0>
10000604: addi s3, s3, 0x7ff
10000608: addi s3, s3, 0x341
1000060c: mx.tload.b td=2 base=s3 stride=a5 fmt=fp32 t=0
10000610: j 0x1000055c <main+0x4f0>
10000614: mx.azero ad=0
10000618: mx.azero ad=1
1000061c: mx.azero ad=2
10000620: mx.azero ad=3
10000624: rdcycle a1
10000628: sub a1, a1, a2
1000062c: sd a1, 0x0(a0)
10000630: li a0, 0x0
10000634: ld s0, 0x28(sp)
10000638: ld s1, 0x20(sp)
1000063c: ld s2, 0x18(sp)
10000640: ld s3, 0x10(sp)
10000644: ld s4, 0x8(sp)
10000648: ld s5, 0x0(sp)
1000064c: addi sp, sp, 0x30
10000650: ret
