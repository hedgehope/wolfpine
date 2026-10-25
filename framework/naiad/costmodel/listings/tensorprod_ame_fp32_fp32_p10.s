# tensorprod/ame size=10 fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x40
10000070: sd s0, 0x38(sp)
10000074: sd s1, 0x30(sp)
10000078: sd s2, 0x28(sp)
1000007c: sd s3, 0x20(sp)
10000080: sd s4, 0x18(sp)
10000084: sd s5, 0x10(sp)
10000088: sd s6, 0x8(sp)
1000008c: auipc a0, 0x36
10000090: li a2, 0xa
10000094: li a3, 0x8
10000098: auipc a1, 0x0
1000009c: addi a0, a0, 0x5f4
100000a0: addi a1, a1, 0x6e8
100000a4: sd a2, 0x40(a0)
100000a8: sd a3, 0x48(a0)
100000ac: addi a2, a1, 0x400
100000b0: sd zero, 0x0(a1)
100000b4: addi a1, a1, 0x8
100000b8: bne a1, a2, 0x100000b0 <main+0x44>
100000bc: auipc a1, 0x1
100000c0: addi a1, a1, -0x53c
100000c4: addi a2, a1, 0x400
100000c8: sd zero, 0x0(a1)
100000cc: addi a1, a1, 0x8
100000d0: bne a1, a2, 0x100000c8 <main+0x5c>
100000d4: auipc a2, 0x0
100000d8: addi a1, a0, 0xa4
100000dc: auipc a3, 0x1
100000e0: addi a4, a2, 0x6ac
100000e4: addi a2, a3, -0x41c
100000e8: addi a3, a4, 0x14
100000ec: addi a4, a4, 0x294
100000f0: flw fa5, -0x24(a1)
100000f4: flw fa4, -0x20(a1)
100000f8: flw fa3, -0x1c(a1)
100000fc: flw fa2, -0x18(a1)
10000100: flw fa1, -0x14(a1)
10000104: flw fa0, -0x10(a1)
10000108: flw ft0, -0xc(a1)
1000010c: flw ft1, -0x8(a1)
10000110: fsw fa5, -0x140(a2)
10000114: fsw fa4, -0x100(a2)
10000118: fsw fa3, -0xc0(a2)
1000011c: fsw fa5, -0x14(a3)
10000120: fsw fa4, -0x10(a3)
10000124: fsw fa3, -0xc(a3)
10000128: fsw fa2, -0x8(a3)
1000012c: fsw fa2, -0x80(a2)
10000130: flw fa5, -0x4(a1)
10000134: flw fa4, 0x0(a1)
10000138: fsw fa1, -0x40(a2)
1000013c: fsw fa0, 0x0(a2)
10000140: fsw ft0, 0x40(a2)
10000144: fsw fa1, -0x4(a3)
10000148: fsw fa0, 0x0(a3)
1000014c: fsw ft0, 0x4(a3)
10000150: fsw ft1, 0x8(a3)
10000154: fsw ft1, 0x80(a2)
10000158: fsw fa5, 0xc0(a2)
1000015c: fsw fa5, 0xc(a3)
10000160: fsw fa4, 0x10(a3)
10000164: fsw fa4, 0x100(a2)
10000168: addi a3, a3, 0x40
1000016c: addi a1, a1, 0x28
10000170: addi a2, a2, 0x4
10000174: bne a3, a4, 0x100000f0 <main+0x84>
10000178: auipc a1, 0x1
1000017c: lui a2, 0x8
10000180: lui a3, 0x12
10000184: addi a1, a1, -0x1f8
10000188: addiw a2, a2, -0x300
1000018c: addiw a3, a3, -0x500
10000190: add a2, a1, a2
10000194: add a3, a1, a3
10000198: sd zero, 0x0(a2)
1000019c: addi a2, a2, 0x8
100001a0: bne a2, a3, 0x10000198 <main+0x12c>
100001a4: li t6, 0x0
100001a8: lui a3, 0x1
100001ac: rdcycle a2
100001b0: auipc a7, 0x0
100001b4: li t1, 0x40
100001b8: auipc t0, 0x1
100001bc: li a4, 0x190
100001c0: addi t2, a1, 0x40
100001c4: li a5, 0x25
100001c8: auipc t3, 0x13
100001cc: li a6, 0x1c0
100001d0: addiw a3, a3, -0x60
100001d4: addi t4, a7, 0x5d0
100001d8: addi t5, t0, -0x638
100001dc: addi a7, t3, -0x748
100001e0: mx.tload.a td=0 base=t4 stride=t1 fmt=fp32 t=0
100001e4: addi t0, a3, 0x1e0
100001e8: mx.tload.b td=5 base=t5 stride=t1 fmt=fp32 t=0
100001ec: mx.tload.b td=1 base=a1 stride=a4 fmt=fp32 t=0
100001f0: mx.tload.b td=2 base=t2 stride=a4 fmt=fp32 t=0
100001f4: li t1, 0x38
100001f8: li t2, 0x33
100001fc: li t3, 0x32
10000200: li t4, 0x35
10000204: j 0x1000025c <main+0x1f0>
10000208: mx.azero ad=3
1000020c: mx.mma ad=3 ta=0 tb=4 in=fp32 acc=fp32
10000210: andi s0, t6, 0xff
10000214: mul s0, s0, a5
10000218: srli s0, s0, 0x8
1000021c: subw s1, t6, s0
10000220: slli s1, s1, 0x38
10000224: srli s1, s1, 0x39
10000228: add s0, s1, s0
1000022c: srli s0, s0, 0x2
10000230: mul s1, s0, t0
10000234: slli s2, s0, 0x3
10000238: add t6, t6, s0
1000023c: add s1, a7, s1
10000240: subw t6, t6, s2
10000244: andi t6, t6, 0xff
10000248: slli t6, t6, 0x6
1000024c: add t6, s1, t6
10000250: mx.astore as=2 base=t6 stride=a6 fmt=fp32
10000254: addi t6, t5, 0x4
10000258: bgeu t5, t4, 0x100004a0 <main+0x434>
1000025c: mv t5, t6
10000260: li s0, 0x38
10000264: beq t6, t1, 0x100002c0 <main+0x254>
10000268: ori t6, t5, 0x2
1000026c: andi s0, t6, 0xff
10000270: mul s0, s0, a5
10000274: srli s0, s0, 0x8
10000278: subw s1, t6, s0
1000027c: slli s1, s1, 0x38
10000280: srli s1, s1, 0x39
10000284: add s0, s1, s0
10000288: srli s0, s0, 0x2
1000028c: mul s1, s0, a3
10000290: slli s2, s0, 0x3
10000294: add t6, t6, s0
10000298: add s1, a1, s1
1000029c: subw t6, t6, s2
100002a0: andi t6, t6, 0xff
100002a4: slli t6, t6, 0x6
100002a8: add t6, s1, t6
100002ac: mx.tload.b td=3 base=t6 stride=a4 fmt=fp32 t=0
100002b0: mx.azero ad=0
100002b4: mx.mma ad=0 ta=0 tb=1 in=fp32 acc=fp32
100002b8: mv s0, t5
100002bc: beqz t5, 0x10000490 <main+0x424>
100002c0: addi t6, s0, -0x1
100002c4: andi s1, t6, 0xff
100002c8: mul s1, s1, a5
100002cc: srli s1, s1, 0x8
100002d0: subw s2, t6, s1
100002d4: slli s2, s2, 0x38
100002d8: srli s2, s2, 0x39
100002dc: add s1, s2, s1
100002e0: srli s1, s1, 0x2
100002e4: mul s2, s1, t0
100002e8: slli s3, s1, 0x3
100002ec: add t6, t6, s1
100002f0: add s2, a7, s2
100002f4: subw t6, t6, s3
100002f8: andi t6, t6, 0xff
100002fc: slli t6, t6, 0x6
10000300: add t6, s2, t6
10000304: mx.astore as=3 base=t6 stride=a6 fmt=fp32
10000308: addi s1, t5, 0x1
1000030c: bltu t1, s1, 0x100004a0 <main+0x434>
10000310: ori t6, s0, 0x2
10000314: ori s2, s0, 0x3
10000318: andi s3, s0, 0xff
1000031c: andi s4, s2, 0xff
10000320: mul s3, s3, a5
10000324: mul s4, s4, a5
10000328: srli s3, s3, 0x8
1000032c: srli s4, s4, 0x8
10000330: subw s5, s0, s3
10000334: subw s6, s2, s4
10000338: slli s5, s5, 0x38
1000033c: slli s6, s6, 0x38
10000340: srli s5, s5, 0x39
10000344: srli s6, s6, 0x39
10000348: add s3, s5, s3
1000034c: add s4, s6, s4
10000350: srli s3, s3, 0x2
10000354: srli s4, s4, 0x2
10000358: slli s5, s3, 0x3
1000035c: add s6, s0, s3
10000360: subw s5, s6, s5
10000364: slli s6, s4, 0x3
10000368: add s2, s2, s4
1000036c: subw s2, s2, s6
10000370: mul s3, s3, t0
10000374: mul s4, s4, a3
10000378: add s3, a7, s3
1000037c: add s4, a1, s4
10000380: andi s5, s5, 0xff
10000384: andi s2, s2, 0xff
10000388: slli s5, s5, 0x6
1000038c: slli s2, s2, 0x6
10000390: add s3, s3, s5
10000394: add s2, s4, s2
10000398: mx.tload.b td=4 base=s2 stride=a4 fmt=fp32 t=0
1000039c: mx.azero ad=1
100003a0: mx.mma ad=1 ta=0 tb=2 in=fp32 acc=fp32
100003a4: mx.astore as=0 base=s3 stride=a6 fmt=fp32
100003a8: bltu t2, t5, 0x100003f4 <main+0x388>
100003ac: addi s2, s0, 0x4
100003b0: andi s3, s2, 0xff
100003b4: mul s3, s3, a5
100003b8: srli s3, s3, 0x8
100003bc: subw s4, s2, s3
100003c0: slli s4, s4, 0x38
100003c4: srli s4, s4, 0x39
100003c8: add s3, s4, s3
100003cc: srli s3, s3, 0x2
100003d0: mul s4, s3, a3
100003d4: slli s5, s3, 0x3
100003d8: add s2, s2, s3
100003dc: add s4, a1, s4
100003e0: subw s2, s2, s5
100003e4: andi s2, s2, 0xff
100003e8: slli s2, s2, 0x6
100003ec: add s2, s4, s2
100003f0: mx.tload.b td=1 base=s2 stride=a4 fmt=fp32 t=0
100003f4: mx.azero ad=2
100003f8: mx.mma ad=2 ta=0 tb=3 in=fp32 acc=fp32
100003fc: andi s2, s1, 0xff
10000400: mul s2, s2, a5
10000404: srli s2, s2, 0x8
10000408: subw s3, s1, s2
1000040c: slli s3, s3, 0x38
10000410: srli s3, s3, 0x39
10000414: add s2, s3, s2
10000418: srli s2, s2, 0x2
1000041c: mul s3, s2, t0
10000420: slli s4, s2, 0x3
10000424: add s1, s1, s2
10000428: add s3, a7, s3
1000042c: subw s1, s1, s4
10000430: andi s1, s1, 0xff
10000434: slli s1, s1, 0x6
10000438: add s1, s3, s1
1000043c: mx.astore as=1 base=s1 stride=a6 fmt=fp32
10000440: bltu t3, t5, 0x10000208 <main+0x19c>
10000444: addi s0, s0, 0x5
10000448: andi s1, s0, 0xff
1000044c: mul s1, s1, a5
10000450: srli s1, s1, 0x8
10000454: subw s2, s0, s1
10000458: slli s2, s2, 0x38
1000045c: srli s2, s2, 0x39
10000460: add s1, s2, s1
10000464: srli s1, s1, 0x2
10000468: mul s2, s1, a3
1000046c: slli s3, s1, 0x3
10000470: add s0, s0, s1
10000474: add s2, a1, s2
10000478: subw s0, s0, s3
1000047c: andi s0, s0, 0xff
10000480: slli s0, s0, 0x6
10000484: add s0, s2, s0
10000488: mx.tload.b td=2 base=s0 stride=a4 fmt=fp32 t=0
1000048c: j 0x10000208 <main+0x19c>
10000490: li s0, 0x0
10000494: li s1, 0x1
10000498: li t6, 0x2
1000049c: j 0x10000314 <main+0x2a8>
100004a0: li a4, 0x0
100004a4: li a5, 0x0
100004a8: li a6, 0x28
100004ac: addi t4, a1, 0x280
100004b0: addi t2, a1, 0x7ff
100004b4: auipc t5, 0x1d
100004b8: li a7, 0x31
100004bc: li t0, 0x40
100004c0: li t1, 0x2d
100004c4: mx.tload.a td=1 base=a1 stride=a6 fmt=fp32 t=0
100004c8: addi t3, t2, 0x481
100004cc: addi t2, t5, -0x234
100004d0: mx.tload.a td=2 base=t4 stride=a6 fmt=fp32 t=0
100004d4: li t4, 0x32
100004d8: li t5, 0x2c
100004dc: lui t6, 0x1
100004e0: li s0, 0x30
100004e4: mv s1, t2
100004e8: j 0x10000514 <main+0x4a8>
100004ec: mx.azero ad=3
100004f0: mx.mma ad=3 ta=4 tb=5 in=fp32 acc=fp32
100004f4: addi s2, s1, 0x7ff
100004f8: addi a5, a5, 0x4
100004fc: addi s2, s2, 0x1
10000500: mx.astore as=2 base=s2 stride=t0 fmt=fp32
10000504: addi t3, t3, 0x7ff
10000508: addi t3, t3, 0x201
1000050c: add s1, s1, t6
10000510: add a4, a4, t6
10000514: addi s2, a5, 0x2
10000518: bltu a7, s2, 0x10000524 <main+0x4b8>
1000051c: addi s2, t3, -0x780
10000520: mx.tload.a td=3 base=s2 stride=a6 fmt=fp32 t=0
10000524: mx.azero ad=0
10000528: mx.mma ad=0 ta=1 tb=5 in=fp32 acc=fp32
1000052c: addi s2, a5, 0x3
10000530: beqz a5, 0x10000548 <main+0x4dc>
10000534: addi s3, s1, -0x400
10000538: mx.astore as=3 base=s3 stride=t0 fmt=fp32
1000053c: mv s3, a4
10000540: bgeu a7, s2, 0x1000054c <main+0x4e0>
10000544: j 0x10000554 <main+0x4e8>
10000548: li s3, 0x0
1000054c: addi s4, t3, -0x500
10000550: mx.tload.a td=4 base=s4 stride=a6 fmt=fp32 t=0
10000554: mx.azero ad=1
10000558: mx.mma ad=1 ta=2 tb=5 in=fp32 acc=fp32
1000055c: add s3, t2, s3
10000560: mx.astore as=0 base=s3 stride=t0 fmt=fp32
10000564: bltu t1, a5, 0x10000574 <main+0x508>
10000568: addi s3, t3, -0x280
1000056c: mx.tload.a td=1 base=s3 stride=a6 fmt=fp32 t=0
10000570: j 0x10000578 <main+0x50c>
10000574: beq a5, s0, 0x10000580 <main+0x514>
10000578: mx.azero ad=2
1000057c: mx.mma ad=2 ta=3 tb=5 in=fp32 acc=fp32
10000580: addi s3, s1, 0x400
10000584: mx.astore as=1 base=s3 stride=t0 fmt=fp32
10000588: bltu t4, s2, 0x10000598 <main+0x52c>
1000058c: bltu t5, a5, 0x100004ec <main+0x480>
10000590: mx.tload.a td=2 base=t3 stride=a6 fmt=fp32 t=0
10000594: j 0x100004ec <main+0x480>
10000598: li a4, 0x0
1000059c: li a5, 0x0
100005a0: li a6, 0x28
100005a4: addi t4, a1, 0x190
100005a8: li a7, -0x4
100005ac: li t0, 0x4c
100005b0: auipc t3, 0x29
100005b4: li t1, 0x40
100005b8: li t2, 0x50
100005bc: mx.tload.b td=1 base=a1 stride=a6 fmt=fp32 t=0
100005c0: addi t3, t3, 0x4d0
100005c4: mx.tload.b td=2 base=t4 stride=a6 fmt=fp32 t=0
100005c8: li t4, 0x4b
100005cc: li t5, 0x4a
100005d0: li t6, 0x4d
100005d4: li s0, -0x4
100005d8: j 0x100005fc <main+0x590>
100005dc: mx.azero ad=3
100005e0: mx.mma ad=3 ta=0 tb=4 in=fp32 acc=fp32
100005e4: addi s1, s1, 0x500
100005e8: mx.astore as=2 base=s1 stride=t1 fmt=fp32
100005ec: addi a5, a5, 0x640
100005f0: add a4, a3, a4
100005f4: addi a4, a4, -0x5a0
100005f8: bgeu s0, t6, 0x1000068c <main+0x620>
100005fc: beq s0, t0, 0x10000624 <main+0x5b8>
10000600: add s1, a1, a5
10000604: addi s1, s1, 0x320
10000608: mx.tload.b td=3 base=s1 stride=a6 fmt=fp32 t=0
1000060c: mx.azero ad=0
10000610: mx.mma ad=0 ta=0 tb=1 in=fp32 acc=fp32
10000614: bne s0, a7, 0x10000624 <main+0x5b8>
10000618: li s1, 0x0
1000061c: li s2, 0x0
10000620: j 0x10000640 <main+0x5d4>
10000624: add s1, t3, a4
10000628: addi s1, s1, -0x280
1000062c: mx.astore as=3 base=s1 stride=t1 fmt=fp32
10000630: addi s3, s0, 0x5
10000634: mv s1, a4
10000638: mv s2, a5
1000063c: bltu t2, s3, 0x1000068c <main+0x620>
10000640: add s2, a1, s2
10000644: add s1, t3, s1
10000648: addi s3, s2, 0x4b0
1000064c: mx.tload.b td=4 base=s3 stride=a6 fmt=fp32 t=0
10000650: mx.azero ad=1
10000654: mx.mma ad=1 ta=0 tb=2 in=fp32 acc=fp32
10000658: mx.astore as=0 base=s1 stride=t1 fmt=fp32
1000065c: addi s0, s0, 0x4
10000660: bltu t4, s0, 0x1000066c <main+0x600>
10000664: addi s3, s2, 0x640
10000668: mx.tload.b td=1 base=s3 stride=a6 fmt=fp32 t=0
1000066c: mx.azero ad=2
10000670: mx.mma ad=2 ta=0 tb=3 in=fp32 acc=fp32
10000674: addi s3, s1, 0x280
10000678: mx.astore as=1 base=s3 stride=t1 fmt=fp32
1000067c: bltu t5, s0, 0x100005dc <main+0x570>
10000680: addi s2, s2, 0x7d0
10000684: mx.tload.b td=2 base=s2 stride=a6 fmt=fp32 t=0
10000688: j 0x100005dc <main+0x570>
1000068c: mx.azero ad=0
10000690: mx.azero ad=1
10000694: mx.azero ad=2
10000698: mx.azero ad=3
1000069c: rdcycle a1
100006a0: sub a1, a1, a2
100006a4: sd a1, 0x0(a0)
100006a8: li a0, 0x0
100006ac: ld s0, 0x38(sp)
100006b0: ld s1, 0x30(sp)
100006b4: ld s2, 0x28(sp)
100006b8: ld s3, 0x20(sp)
100006bc: ld s4, 0x18(sp)
100006c0: ld s5, 0x10(sp)
100006c4: ld s6, 0x8(sp)
100006c8: addi sp, sp, 0x40
100006cc: ret
