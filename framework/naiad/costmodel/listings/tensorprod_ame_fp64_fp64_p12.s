# tensorprod/ame size=12 fp64/fp64 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x50
10000070: sd s0, 0x48(sp)
10000074: sd s1, 0x40(sp)
10000078: sd s2, 0x38(sp)
1000007c: sd s3, 0x30(sp)
10000080: sd s4, 0x28(sp)
10000084: sd s5, 0x20(sp)
10000088: sd s6, 0x18(sp)
1000008c: sd s7, 0x10(sp)
10000090: sd s8, 0x8(sp)
10000094: auipc a0, 0xa5
10000098: li a2, 0xc
1000009c: li a3, 0x8
100000a0: auipc a1, 0x0
100000a4: addi a0, a0, -0x154
100000a8: addi a1, a1, 0x6a0
100000ac: sd a2, 0x40(a0)
100000b0: sd a3, 0x48(a0)
100000b4: addi a2, a1, 0x7ff
100000b8: addi a2, a2, 0x1
100000bc: sd zero, 0x0(a1)
100000c0: addi a1, a1, 0x8
100000c4: bne a1, a2, 0x100000bc <main+0x50>
100000c8: auipc a1, 0x1
100000cc: addi a1, a1, -0x188
100000d0: addi a2, a1, 0x7ff
100000d4: addi a2, a2, 0x1
100000d8: sd zero, 0x0(a1)
100000dc: addi a1, a1, 0x8
100000e0: bne a1, a2, 0x100000d8 <main+0x6c>
100000e4: auipc a2, 0x0
100000e8: addi a1, a0, 0xd8
100000ec: auipc a3, 0x1
100000f0: addi a4, a2, 0x65c
100000f4: addi a2, a3, 0x154
100000f8: addi a3, a4, 0x30
100000fc: addi a4, a4, 0x630
10000100: fld fa5, -0x58(a1)
10000104: fld fa4, -0x50(a1)
10000108: fld fa3, -0x48(a1)
1000010c: fld fa2, -0x40(a1)
10000110: fsd fa5, -0x300(a2)
10000114: fsd fa4, -0x280(a2)
10000118: fsd fa3, -0x200(a2)
1000011c: fsd fa5, -0x30(a3)
10000120: fsd fa4, -0x28(a3)
10000124: fsd fa3, -0x20(a3)
10000128: fsd fa2, -0x18(a3)
1000012c: fsd fa2, -0x180(a2)
10000130: fld fa5, -0x38(a1)
10000134: fld fa4, -0x30(a1)
10000138: fld fa3, -0x28(a1)
1000013c: fld fa2, -0x20(a1)
10000140: fsd fa5, -0x100(a2)
10000144: fsd fa4, -0x80(a2)
10000148: fsd fa3, 0x0(a2)
1000014c: fsd fa5, -0x10(a3)
10000150: fsd fa4, -0x8(a3)
10000154: fsd fa3, 0x0(a3)
10000158: fsd fa2, 0x8(a3)
1000015c: fsd fa2, 0x80(a2)
10000160: fld fa5, -0x18(a1)
10000164: fld fa4, -0x10(a1)
10000168: fld fa3, -0x8(a1)
1000016c: fld fa2, 0x0(a1)
10000170: fsd fa5, 0x100(a2)
10000174: fsd fa4, 0x180(a2)
10000178: fsd fa3, 0x200(a2)
1000017c: fsd fa5, 0x10(a3)
10000180: fsd fa4, 0x18(a3)
10000184: fsd fa3, 0x20(a3)
10000188: fsd fa2, 0x28(a3)
1000018c: fsd fa2, 0x280(a2)
10000190: addi a3, a3, 0x80
10000194: addi a1, a1, 0x60
10000198: addi a2, a2, 0x8
1000019c: bne a3, a4, 0x10000100 <main+0x94>
100001a0: auipc a1, 0x1
100001a4: lui a2, 0x1b
100001a8: lui a3, 0x3c
100001ac: addi a1, a1, 0x5a0
100001b0: addiw a3, a3, -0x800
100001b4: add a2, a1, a2
100001b8: add a3, a1, a3
100001bc: sd zero, 0x0(a2)
100001c0: addi a2, a2, 0x8
100001c4: bne a2, a3, 0x100001bc <main+0x150>
100001c8: li t4, 0x0
100001cc: li a5, 0x1b
100001d0: rdcycle a2
100001d4: auipc a6, 0x0
100001d8: li a7, 0x80
100001dc: auipc t0, 0x1
100001e0: li a3, 0x480
100001e4: addi t1, a1, 0x80
100001e8: li a4, 0x39
100001ec: auipc t2, 0x3d
100001f0: slli a5, a5, 0x9
100001f4: addi t3, a6, 0x56c
100001f8: addi t0, t0, -0x29c
100001fc: addi a6, t2, -0x2ac
10000200: mx.tload.a td=0 base=t3 stride=a7 fmt=fp64 t=0
10000204: mx.tload.b td=5 base=t0 stride=a7 fmt=fp64 t=0
10000208: mx.tload.b td=1 base=a1 stride=a3 fmt=fp64 t=0
1000020c: mx.tload.b td=2 base=t1 stride=a3 fmt=fp64 t=0
10000210: li a7, 0x48
10000214: li t0, 0x43
10000218: li t1, 0x42
1000021c: li t2, 0x45
10000220: j 0x10000264 <main+0x1f8>
10000224: mx.azero ad=3
10000228: mx.mma ad=3 ta=0 tb=4 in=fp64 acc=fp64
1000022c: andi t4, t5, 0xff
10000230: mul t4, t4, a4
10000234: srli t4, t4, 0x9
10000238: mul t6, t4, a5
1000023c: slli s0, t4, 0x3
10000240: subw t4, t5, t4
10000244: add t6, a6, t6
10000248: subw t4, t4, s0
1000024c: andi t4, t4, 0xff
10000250: slli t4, t4, 0x7
10000254: add t4, t6, t4
10000258: mx.astore as=2 base=t4 stride=a3 fmt=fp64
1000025c: addi t4, t3, 0x4
10000260: bgeu t3, t2, 0x1000041c <main+0x3b0>
10000264: mv t3, t4
10000268: li t4, 0x48
1000026c: beq t3, a7, 0x100002b4 <main+0x248>
10000270: ori t4, t3, 0x2
10000274: andi t5, t4, 0xff
10000278: mul t5, t5, a4
1000027c: srli t5, t5, 0x9
10000280: mul t6, t5, a5
10000284: slli s0, t5, 0x3
10000288: subw t4, t4, t5
1000028c: add t6, a1, t6
10000290: subw t4, t4, s0
10000294: andi t4, t4, 0xff
10000298: slli t4, t4, 0x7
1000029c: add t4, t6, t4
100002a0: mx.tload.b td=3 base=t4 stride=a3 fmt=fp64 t=0
100002a4: mx.azero ad=0
100002a8: mx.mma ad=0 ta=0 tb=1 in=fp64 acc=fp64
100002ac: mv t4, t3
100002b0: beqz t3, 0x1000040c <main+0x3a0>
100002b4: addi t5, t4, -0x1
100002b8: andi t6, t5, 0xff
100002bc: mul t6, t6, a4
100002c0: srli t6, t6, 0x9
100002c4: mul s0, t6, a5
100002c8: slli s1, t6, 0x3
100002cc: subw t5, t5, t6
100002d0: add s0, a6, s0
100002d4: subw t5, t5, s1
100002d8: andi t5, t5, 0xff
100002dc: slli t5, t5, 0x7
100002e0: add t5, s0, t5
100002e4: mx.astore as=3 base=t5 stride=a3 fmt=fp64
100002e8: addi t6, t3, 0x1
100002ec: bltu a7, t6, 0x1000041c <main+0x3b0>
100002f0: ori t5, t4, 0x2
100002f4: ori s0, t4, 0x3
100002f8: andi s1, t4, 0xff
100002fc: andi s2, s0, 0xff
10000300: mul s1, s1, a4
10000304: mul s2, s2, a4
10000308: srli s1, s1, 0x9
1000030c: srli s2, s2, 0x9
10000310: slli s3, s1, 0x3
10000314: subw s4, t4, s1
10000318: subw s3, s4, s3
1000031c: slli s4, s2, 0x3
10000320: subw s0, s0, s2
10000324: subw s0, s0, s4
10000328: mul s1, s1, a5
1000032c: mul s2, s2, a5
10000330: add s1, a6, s1
10000334: add s2, a1, s2
10000338: andi s3, s3, 0xff
1000033c: andi s0, s0, 0xff
10000340: slli s3, s3, 0x7
10000344: slli s0, s0, 0x7
10000348: add s1, s1, s3
1000034c: add s0, s2, s0
10000350: mx.tload.b td=4 base=s0 stride=a3 fmt=fp64 t=0
10000354: mx.azero ad=1
10000358: mx.mma ad=1 ta=0 tb=2 in=fp64 acc=fp64
1000035c: mx.astore as=0 base=s1 stride=a3 fmt=fp64
10000360: bltu t0, t3, 0x10000398 <main+0x32c>
10000364: addi s0, t4, 0x4
10000368: andi s1, s0, 0xff
1000036c: mul s1, s1, a4
10000370: srli s1, s1, 0x9
10000374: mul s2, s1, a5
10000378: slli s3, s1, 0x3
1000037c: subw s0, s0, s1
10000380: add s2, a1, s2
10000384: subw s0, s0, s3
10000388: andi s0, s0, 0xff
1000038c: slli s0, s0, 0x7
10000390: add s0, s2, s0
10000394: mx.tload.b td=1 base=s0 stride=a3 fmt=fp64 t=0
10000398: mx.azero ad=2
1000039c: mx.mma ad=2 ta=0 tb=3 in=fp64 acc=fp64
100003a0: andi s0, t6, 0xff
100003a4: mul s0, s0, a4
100003a8: srli s0, s0, 0x9
100003ac: mul s1, s0, a5
100003b0: slli s2, s0, 0x3
100003b4: subw t6, t6, s0
100003b8: add s1, a6, s1
100003bc: subw t6, t6, s2
100003c0: andi t6, t6, 0xff
100003c4: slli t6, t6, 0x7
100003c8: add t6, s1, t6
100003cc: mx.astore as=1 base=t6 stride=a3 fmt=fp64
100003d0: bltu t1, t3, 0x10000224 <main+0x1b8>
100003d4: addi t4, t4, 0x5
100003d8: andi t6, t4, 0xff
100003dc: mul t6, t6, a4
100003e0: srli t6, t6, 0x9
100003e4: mul s0, t6, a5
100003e8: slli s1, t6, 0x3
100003ec: subw t4, t4, t6
100003f0: add s0, a1, s0
100003f4: subw t4, t4, s1
100003f8: andi t4, t4, 0xff
100003fc: slli t4, t4, 0x7
10000400: add t4, s0, t4
10000404: mx.tload.b td=2 base=t4 stride=a3 fmt=fp64 t=0
10000408: j 0x10000224 <main+0x1b8>
1000040c: li t4, 0x0
10000410: li t6, 0x1
10000414: li t5, 0x2
10000418: j 0x100002f4 <main+0x288>
1000041c: li a4, 0x0
10000420: li a5, 0x0
10000424: li a3, 0x3
10000428: li a6, 0x60
1000042c: addi s1, a1, 0x600
10000430: li a7, -0x4
10000434: li t0, 0x44
10000438: auipc t5, 0x5c
1000043c: li t1, 0x80
10000440: li t2, 0x48
10000444: li t6, 0x9
10000448: li t3, 0x43
1000044c: li t4, 0x42
10000450: li s0, 0xf
10000454: slli a3, a3, 0xb
10000458: mx.tload.a td=1 base=a1 stride=a6 fmt=fp64 t=0
1000045c: addi t5, t5, 0x308
10000460: slli t6, t6, 0x9
10000464: slli s0, s0, 0x9
10000468: mx.tload.a td=2 base=s1 stride=a6 fmt=fp64 t=0
1000046c: lui s1, 0x1
10000470: lui s2, 0x2
10000474: li s3, 0x45
10000478: li s4, -0x4
1000047c: j 0x1000049c <main+0x430>
10000480: mx.azero ad=3
10000484: mx.mma ad=3 ta=4 tb=5 in=fp64 acc=fp64
10000488: add s5, s5, s1
1000048c: mx.astore as=2 base=s5 stride=t1 fmt=fp64
10000490: add a5, a5, s2
10000494: add a4, a4, a3
10000498: bgeu s4, s3, 0x10000534 <main+0x4c8>
1000049c: add s5, t5, a5
100004a0: beq s4, t0, 0x100004cc <main+0x460>
100004a4: add s6, a1, a4
100004a8: addi s6, s6, 0x7ff
100004ac: addi s6, s6, 0x401
100004b0: mx.tload.a td=3 base=s6 stride=a6 fmt=fp64 t=0
100004b4: mx.azero ad=0
100004b8: mx.mma ad=0 ta=1 tb=5 in=fp64 acc=fp64
100004bc: bne s4, a7, 0x100004cc <main+0x460>
100004c0: li s7, 0x0
100004c4: li s6, 0x0
100004c8: j 0x100004e4 <main+0x478>
100004cc: addi s6, s5, -0x800
100004d0: mx.astore as=3 base=s6 stride=t1 fmt=fp64
100004d4: addi s8, s4, 0x5
100004d8: mv s7, a5
100004dc: mv s6, a4
100004e0: bltu t2, s8, 0x10000534 <main+0x4c8>
100004e4: add s6, a1, s6
100004e8: add s7, t5, s7
100004ec: add s8, s6, t6
100004f0: mx.tload.a td=4 base=s8 stride=a6 fmt=fp64 t=0
100004f4: mx.azero ad=1
100004f8: mx.mma ad=1 ta=2 tb=5 in=fp64 acc=fp64
100004fc: mx.astore as=0 base=s7 stride=t1 fmt=fp64
10000500: addi s4, s4, 0x4
10000504: bltu t3, s4, 0x10000510 <main+0x4a4>
10000508: add s7, s6, a3
1000050c: mx.tload.a td=1 base=s7 stride=a6 fmt=fp64 t=0
10000510: mx.azero ad=2
10000514: mx.mma ad=2 ta=3 tb=5 in=fp64 acc=fp64
10000518: addi s7, s5, 0x7ff
1000051c: addi s7, s7, 0x1
10000520: mx.astore as=1 base=s7 stride=t1 fmt=fp64
10000524: bltu t4, s4, 0x10000480 <main+0x414>
10000528: add s6, s6, s0
1000052c: mx.tload.a td=2 base=s6 stride=a6 fmt=fp64 t=0
10000530: j 0x10000480 <main+0x414>
10000534: li a4, 0x0
10000538: li a6, 0x0
1000053c: li a5, 0x60
10000540: addi s0, a1, 0x480
10000544: li a7, -0x4
10000548: auipc t4, 0x80
1000054c: li t0, 0x5c
10000550: li t1, 0x80
10000554: li t2, 0x5b
10000558: li t5, 0x9
1000055c: li t3, 0x5a
10000560: lui t6, 0x1
10000564: mx.tload.b td=1 base=a1 stride=a5 fmt=fp64 t=0
10000568: addi t4, t4, 0x1f8
1000056c: slli t5, t5, 0x9
10000570: addiw t6, t6, 0x680
10000574: mx.tload.b td=2 base=s0 stride=a5 fmt=fp64 t=0
10000578: li s0, 0x3
1000057c: slli s0, s0, 0xb
10000580: li s1, 0x5d
10000584: mv s2, t4
10000588: li s3, -0x4
1000058c: j 0x100005b8 <main+0x54c>
10000590: mx.azero ad=3
10000594: mx.mma ad=3 ta=0 tb=4 in=fp64 acc=fp64
10000598: addi s4, s5, 0x7ff
1000059c: addi s4, s4, 0x401
100005a0: mx.astore as=2 base=s4 stride=t1 fmt=fp64
100005a4: add s2, s2, s0
100005a8: add a6, a3, a6
100005ac: addi a6, a6, -0x600
100005b0: add a4, a4, a3
100005b4: bgeu s3, s1, 0x1000064c <main+0x5e0>
100005b8: beq s3, t0, 0x100005e4 <main+0x578>
100005bc: add s4, a1, a6
100005c0: addi s4, s4, 0x7ff
100005c4: addi s4, s4, 0x101
100005c8: mx.tload.b td=3 base=s4 stride=a5 fmt=fp64 t=0
100005cc: mx.azero ad=0
100005d0: mx.mma ad=0 ta=0 tb=1 in=fp64 acc=fp64
100005d4: bne s3, a7, 0x100005e4 <main+0x578>
100005d8: li s5, 0x0
100005dc: li s4, 0x0
100005e0: j 0x100005fc <main+0x590>
100005e4: addi s4, s2, -0x600
100005e8: mx.astore as=3 base=s4 stride=t1 fmt=fp64
100005ec: addi s6, s3, 0x5
100005f0: mv s5, a4
100005f4: mv s4, a6
100005f8: bltu a5, s6, 0x1000064c <main+0x5e0>
100005fc: add s4, a1, s4
10000600: add s5, t4, s5
10000604: addi s6, s4, 0x7ff
10000608: addi s6, s6, 0x581
1000060c: mx.tload.b td=4 base=s6 stride=a5 fmt=fp64 t=0
10000610: mx.azero ad=1
10000614: mx.mma ad=1 ta=0 tb=2 in=fp64 acc=fp64
10000618: mx.astore as=0 base=s5 stride=t1 fmt=fp64
1000061c: addi s3, s3, 0x4
10000620: bltu t2, s3, 0x1000062c <main+0x5c0>
10000624: add s6, s4, t5
10000628: mx.tload.b td=1 base=s6 stride=a5 fmt=fp64 t=0
1000062c: mx.azero ad=2
10000630: mx.mma ad=2 ta=0 tb=3 in=fp64 acc=fp64
10000634: addi s6, s2, 0x600
10000638: mx.astore as=1 base=s6 stride=t1 fmt=fp64
1000063c: bltu t3, s3, 0x10000590 <main+0x524>
10000640: add s4, s4, t6
10000644: mx.tload.b td=2 base=s4 stride=a5 fmt=fp64 t=0
10000648: j 0x10000590 <main+0x524>
1000064c: mx.azero ad=0
10000650: mx.azero ad=1
10000654: mx.azero ad=2
10000658: mx.azero ad=3
1000065c: rdcycle a1
10000660: sub a1, a1, a2
10000664: sd a1, 0x0(a0)
10000668: li a0, 0x0
1000066c: ld s0, 0x48(sp)
10000670: ld s1, 0x40(sp)
10000674: ld s2, 0x38(sp)
10000678: ld s3, 0x30(sp)
1000067c: ld s4, 0x28(sp)
10000680: ld s5, 0x20(sp)
10000684: ld s6, 0x18(sp)
10000688: ld s7, 0x10(sp)
1000068c: ld s8, 0x8(sp)
10000690: addi sp, sp, 0x50
10000694: ret
