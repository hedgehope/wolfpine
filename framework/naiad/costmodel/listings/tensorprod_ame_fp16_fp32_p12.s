# tensorprod/ame size=12 fp16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x40
10000070: sd s0, 0x38(sp)
10000074: sd s1, 0x30(sp)
10000078: sd s2, 0x28(sp)
1000007c: sd s3, 0x20(sp)
10000080: sd s4, 0x18(sp)
10000084: sd s5, 0x10(sp)
10000088: sd s6, 0x8(sp)
1000008c: auipc a0, 0x44
10000090: li a2, 0xc
10000094: li a3, 0x8
10000098: auipc a1, 0x0
1000009c: addi a0, a0, -0x7cc
100000a0: addi a1, a1, 0x628
100000a4: sd a2, 0x40(a0)
100000a8: sd a3, 0x48(a0)
100000ac: addi a2, a1, 0x200
100000b0: sd zero, 0x0(a1)
100000b4: addi a1, a1, 0x8
100000b8: bne a1, a2, 0x100000b0 <main+0x44>
100000bc: auipc a1, 0x0
100000c0: addi t0, a1, 0x604
100000c4: addi a1, t0, 0x200
100000c8: addi a2, t0, 0x400
100000cc: sd zero, 0x0(a1)
100000d0: addi a1, a1, 0x8
100000d4: bne a1, a2, 0x100000cc <main+0x60>
100000d8: addi a1, t0, 0xc
100000dc: addi a2, a0, 0x96
100000e0: addi a3, t0, 0x360
100000e4: addi a4, t0, 0x18c
100000e8: lh a5, -0x16(a2)
100000ec: lh a6, -0x14(a2)
100000f0: lh a7, -0x12(a2)
100000f4: lh t1, -0x10(a2)
100000f8: sh a5, -0xc(a1)
100000fc: sh a5, -0x160(a3)
10000100: sh a6, -0xa(a1)
10000104: sh a6, -0x140(a3)
10000108: sh a7, -0x8(a1)
1000010c: sh a7, -0x120(a3)
10000110: sh t1, -0x6(a1)
10000114: sh t1, -0x100(a3)
10000118: lh a5, -0xe(a2)
1000011c: lh a6, -0xc(a2)
10000120: lh a7, -0xa(a2)
10000124: lh t1, -0x8(a2)
10000128: sh a5, -0x4(a1)
1000012c: sh a5, -0xe0(a3)
10000130: sh a6, -0x2(a1)
10000134: sh a6, -0xc0(a3)
10000138: sh a7, 0x0(a1)
1000013c: sh a7, -0xa0(a3)
10000140: sh t1, 0x2(a1)
10000144: sh t1, -0x80(a3)
10000148: lh a5, -0x6(a2)
1000014c: lh a6, -0x4(a2)
10000150: lh a7, -0x2(a2)
10000154: lh t1, 0x0(a2)
10000158: sh a5, 0x4(a1)
1000015c: sh a5, -0x60(a3)
10000160: sh a6, 0x6(a1)
10000164: sh a6, -0x40(a3)
10000168: sh a7, 0x8(a1)
1000016c: sh a7, -0x20(a3)
10000170: sh t1, 0xa(a1)
10000174: sh t1, 0x0(a3)
10000178: addi a1, a1, 0x20
1000017c: addi a2, a2, 0x18
10000180: addi a3, a3, 0x2
10000184: bne a1, a4, 0x100000e8 <main+0x7c>
10000188: auipc a1, 0x1
1000018c: li a2, 0x1b
10000190: lui a3, 0xf
10000194: addi a1, a1, -0x6c8
10000198: slli a2, a2, 0xa
1000019c: addiw a3, a3, -0x200
100001a0: add a2, a1, a2
100001a4: add a3, a1, a3
100001a8: sd zero, 0x0(a2)
100001ac: addi a2, a2, 0x8
100001b0: bne a2, a3, 0x100001a8 <main+0x13c>
100001b4: li t6, 0x0
100001b8: li a3, 0x1b
100001bc: rdcycle a2
100001c0: li t1, 0x20
100001c4: addi t2, t0, 0x200
100001c8: li a4, 0x120
100001cc: addi t3, a1, 0x20
100001d0: li a5, 0x39
100001d4: auipc t4, 0xf
100001d8: li a6, 0x240
100001dc: slli a7, a3, 0x8
100001e0: slli a3, a3, 0x7
100001e4: mx.tload.a td=0 base=t0 stride=t1 fmt=fp16 t=0
100001e8: addi t0, t4, 0x6ec
100001ec: mx.tload.b td=5 base=t2 stride=t1 fmt=fp16 t=0
100001f0: mx.tload.b td=1 base=a1 stride=a4 fmt=fp16 t=0
100001f4: mx.tload.b td=2 base=t3 stride=a4 fmt=fp16 t=0
100001f8: li t1, 0x48
100001fc: li t2, 0x43
10000200: li t3, 0x42
10000204: li t4, 0x45
10000208: j 0x1000024c <main+0x1e0>
1000020c: mx.azero ad=3
10000210: mx.mma ad=3 ta=0 tb=4 in=fp16 acc=fp32
10000214: andi t6, s0, 0xff
10000218: mul t6, t6, a5
1000021c: srli t6, t6, 0x9
10000220: mul s1, t6, a7
10000224: slli s2, t6, 0x3
10000228: subw t6, s0, t6
1000022c: add s1, t0, s1
10000230: subw t6, t6, s2
10000234: andi t6, t6, 0xff
10000238: slli t6, t6, 0x6
1000023c: add t6, s1, t6
10000240: mx.astore as=2 base=t6 stride=a6 fmt=fp32
10000244: addi t6, t5, 0x4
10000248: bgeu t5, t4, 0x10000404 <main+0x398>
1000024c: mv t5, t6
10000250: li t6, 0x48
10000254: beq t5, t1, 0x1000029c <main+0x230>
10000258: ori t6, t5, 0x2
1000025c: andi s0, t6, 0xff
10000260: mul s0, s0, a5
10000264: srli s0, s0, 0x9
10000268: mul s1, s0, a3
1000026c: slli s2, s0, 0x3
10000270: subw t6, t6, s0
10000274: add s1, a1, s1
10000278: subw t6, t6, s2
1000027c: andi t6, t6, 0xff
10000280: slli t6, t6, 0x5
10000284: add t6, s1, t6
10000288: mx.tload.b td=3 base=t6 stride=a4 fmt=fp16 t=0
1000028c: mx.azero ad=0
10000290: mx.mma ad=0 ta=0 tb=1 in=fp16 acc=fp32
10000294: mv t6, t5
10000298: beqz t5, 0x100003f4 <main+0x388>
1000029c: addi s0, t6, -0x1
100002a0: andi s1, s0, 0xff
100002a4: mul s1, s1, a5
100002a8: srli s1, s1, 0x9
100002ac: mul s2, s1, a7
100002b0: slli s3, s1, 0x3
100002b4: subw s0, s0, s1
100002b8: add s2, t0, s2
100002bc: subw s0, s0, s3
100002c0: andi s0, s0, 0xff
100002c4: slli s0, s0, 0x6
100002c8: add s0, s2, s0
100002cc: mx.astore as=3 base=s0 stride=a6 fmt=fp32
100002d0: addi s1, t5, 0x1
100002d4: bltu t1, s1, 0x10000404 <main+0x398>
100002d8: ori s0, t6, 0x2
100002dc: ori s2, t6, 0x3
100002e0: andi s3, t6, 0xff
100002e4: andi s4, s2, 0xff
100002e8: mul s3, s3, a5
100002ec: mul s4, s4, a5
100002f0: srli s3, s3, 0x9
100002f4: srli s4, s4, 0x9
100002f8: slli s5, s3, 0x3
100002fc: subw s6, t6, s3
10000300: subw s5, s6, s5
10000304: slli s6, s4, 0x3
10000308: subw s2, s2, s4
1000030c: subw s2, s2, s6
10000310: mul s3, s3, a7
10000314: mul s4, s4, a3
10000318: add s3, t0, s3
1000031c: add s4, a1, s4
10000320: andi s5, s5, 0xff
10000324: andi s2, s2, 0xff
10000328: slli s5, s5, 0x6
1000032c: slli s2, s2, 0x5
10000330: add s3, s3, s5
10000334: add s2, s4, s2
10000338: mx.tload.b td=4 base=s2 stride=a4 fmt=fp16 t=0
1000033c: mx.azero ad=1
10000340: mx.mma ad=1 ta=0 tb=2 in=fp16 acc=fp32
10000344: mx.astore as=0 base=s3 stride=a6 fmt=fp32
10000348: bltu t2, t5, 0x10000380 <main+0x314>
1000034c: addi s2, t6, 0x4
10000350: andi s3, s2, 0xff
10000354: mul s3, s3, a5
10000358: srli s3, s3, 0x9
1000035c: mul s4, s3, a3
10000360: slli s5, s3, 0x3
10000364: subw s2, s2, s3
10000368: add s4, a1, s4
1000036c: subw s2, s2, s5
10000370: andi s2, s2, 0xff
10000374: slli s2, s2, 0x5
10000378: add s2, s4, s2
1000037c: mx.tload.b td=1 base=s2 stride=a4 fmt=fp16 t=0
10000380: mx.azero ad=2
10000384: mx.mma ad=2 ta=0 tb=3 in=fp16 acc=fp32
10000388: andi s2, s1, 0xff
1000038c: mul s2, s2, a5
10000390: srli s2, s2, 0x9
10000394: mul s3, s2, a7
10000398: slli s4, s2, 0x3
1000039c: subw s1, s1, s2
100003a0: add s3, t0, s3
100003a4: subw s1, s1, s4
100003a8: andi s1, s1, 0xff
100003ac: slli s1, s1, 0x6
100003b0: add s1, s3, s1
100003b4: mx.astore as=1 base=s1 stride=a6 fmt=fp32
100003b8: bltu t3, t5, 0x1000020c <main+0x1a0>
100003bc: addi t6, t6, 0x5
100003c0: andi s1, t6, 0xff
100003c4: mul s1, s1, a5
100003c8: srli s1, s1, 0x9
100003cc: mul s2, s1, a3
100003d0: slli s3, s1, 0x3
100003d4: subw t6, t6, s1
100003d8: add s2, a1, s2
100003dc: subw t6, t6, s3
100003e0: andi t6, t6, 0xff
100003e4: slli t6, t6, 0x5
100003e8: add t6, s2, t6
100003ec: mx.tload.b td=2 base=t6 stride=a4 fmt=fp16 t=0
100003f0: j 0x1000020c <main+0x1a0>
100003f4: li t6, 0x0
100003f8: li s1, 0x1
100003fc: li s0, 0x2
10000400: j 0x100002dc <main+0x270>
10000404: li a4, 0x0
10000408: li a5, 0x0
1000040c: li a6, 0x18
10000410: addi t5, a1, 0x180
10000414: li a7, -0x4
10000418: auipc t4, 0x1f
1000041c: li t0, 0x44
10000420: li t1, 0x40
10000424: li t2, 0x48
10000428: li t3, 0x43
1000042c: mx.tload.a td=1 base=a1 stride=a6 fmt=fp16 t=0
10000430: addi t4, t4, 0xa8
10000434: mx.tload.a td=2 base=t5 stride=a6 fmt=fp16 t=0
10000438: li t5, 0x42
1000043c: lui t6, 0x1
10000440: li s0, 0x45
10000444: mv s1, t4
10000448: li s2, -0x4
1000044c: j 0x10000474 <main+0x408>
10000450: mx.azero ad=3
10000454: mx.mma ad=3 ta=4 tb=5 in=fp16 acc=fp32
10000458: addi s3, s1, 0x7ff
1000045c: addi s3, s3, 0x1
10000460: mx.astore as=2 base=s3 stride=t1 fmt=fp32
10000464: add s1, s1, t6
10000468: addi a5, a5, 0x600
1000046c: add a4, a4, t6
10000470: bgeu s2, s0, 0x10000500 <main+0x494>
10000474: beq s2, t0, 0x1000049c <main+0x430>
10000478: add s3, a1, a5
1000047c: addi s3, s3, 0x300
10000480: mx.tload.a td=3 base=s3 stride=a6 fmt=fp16 t=0
10000484: mx.azero ad=0
10000488: mx.mma ad=0 ta=1 tb=5 in=fp16 acc=fp32
1000048c: bne s2, a7, 0x1000049c <main+0x430>
10000490: li s4, 0x0
10000494: li s3, 0x0
10000498: j 0x100004b4 <main+0x448>
1000049c: addi s3, s1, -0x400
100004a0: mx.astore as=3 base=s3 stride=t1 fmt=fp32
100004a4: addi s5, s2, 0x5
100004a8: mv s4, a4
100004ac: mv s3, a5
100004b0: bltu t2, s5, 0x10000500 <main+0x494>
100004b4: add s3, a1, s3
100004b8: add s4, t4, s4
100004bc: addi s5, s3, 0x480
100004c0: mx.tload.a td=4 base=s5 stride=a6 fmt=fp16 t=0
100004c4: mx.azero ad=1
100004c8: mx.mma ad=1 ta=2 tb=5 in=fp16 acc=fp32
100004cc: mx.astore as=0 base=s4 stride=t1 fmt=fp32
100004d0: addi s2, s2, 0x4
100004d4: bltu t3, s2, 0x100004e0 <main+0x474>
100004d8: addi s4, s3, 0x600
100004dc: mx.tload.a td=1 base=s4 stride=a6 fmt=fp16 t=0
100004e0: mx.azero ad=2
100004e4: mx.mma ad=2 ta=3 tb=5 in=fp16 acc=fp32
100004e8: addi s4, s1, 0x400
100004ec: mx.astore as=1 base=s4 stride=t1 fmt=fp32
100004f0: bltu t5, s2, 0x10000450 <main+0x3e4>
100004f4: addi s3, s3, 0x780
100004f8: mx.tload.a td=2 base=s3 stride=a6 fmt=fp16 t=0
100004fc: j 0x10000450 <main+0x3e4>
10000500: li a4, 0x0
10000504: li a5, 0x0
10000508: li a6, 0x18
1000050c: addi t4, a1, 0x120
10000510: auipc t3, 0x31
10000514: li a7, -0x4
10000518: li t0, 0x5c
1000051c: li t1, 0x40
10000520: li t2, 0x60
10000524: mx.tload.b td=1 base=a1 stride=a6 fmt=fp16 t=0
10000528: addi t3, t3, -0x50
1000052c: mx.tload.b td=2 base=t4 stride=a6 fmt=fp16 t=0
10000530: addi t6, t3, 0x300
10000534: li t4, 0x5b
10000538: li t5, 0x5a
1000053c: li s0, 0x5d
10000540: li s1, -0x4
10000544: j 0x10000570 <main+0x504>
10000548: mx.azero ad=3
1000054c: mx.mma ad=3 ta=0 tb=4 in=fp16 acc=fp32
10000550: addi s2, s2, 0x600
10000554: addi t6, t6, 0x7ff
10000558: mx.astore as=2 base=s2 stride=t1 fmt=fp32
1000055c: addi t6, t6, 0x401
10000560: addi a5, a5, 0x480
10000564: add a4, a3, a4
10000568: addi a4, a4, -0x180
1000056c: bgeu s1, s0, 0x100005f8 <main+0x58c>
10000570: beq s1, t0, 0x10000598 <main+0x52c>
10000574: add s2, a1, a5
10000578: addi s2, s2, 0x240
1000057c: mx.tload.b td=3 base=s2 stride=a6 fmt=fp16 t=0
10000580: mx.azero ad=0
10000584: mx.mma ad=0 ta=0 tb=1 in=fp16 acc=fp32
10000588: bne s1, a7, 0x10000598 <main+0x52c>
1000058c: li s2, 0x0
10000590: li s3, 0x0
10000594: j 0x100005b0 <main+0x544>
10000598: addi s2, t6, -0x600
1000059c: mx.astore as=3 base=s2 stride=t1 fmt=fp32
100005a0: addi s4, s1, 0x5
100005a4: mv s2, a4
100005a8: mv s3, a5
100005ac: bltu t2, s4, 0x100005f8 <main+0x58c>
100005b0: add s3, a1, s3
100005b4: add s2, t3, s2
100005b8: addi s4, s3, 0x360
100005bc: mx.tload.b td=4 base=s4 stride=a6 fmt=fp16 t=0
100005c0: mx.azero ad=1
100005c4: mx.mma ad=1 ta=0 tb=2 in=fp16 acc=fp32
100005c8: mx.astore as=0 base=s2 stride=t1 fmt=fp32
100005cc: addi s1, s1, 0x4
100005d0: bltu t4, s1, 0x100005dc <main+0x570>
100005d4: addi s4, s3, 0x480
100005d8: mx.tload.b td=1 base=s4 stride=a6 fmt=fp16 t=0
100005dc: mx.azero ad=2
100005e0: mx.mma ad=2 ta=0 tb=3 in=fp16 acc=fp32
100005e4: mx.astore as=1 base=t6 stride=t1 fmt=fp32
100005e8: bltu t5, s1, 0x10000548 <main+0x4dc>
100005ec: addi s3, s3, 0x5a0
100005f0: mx.tload.b td=2 base=s3 stride=a6 fmt=fp16 t=0
100005f4: j 0x10000548 <main+0x4dc>
100005f8: mx.azero ad=0
100005fc: mx.azero ad=1
10000600: mx.azero ad=2
10000604: mx.azero ad=3
10000608: rdcycle a1
1000060c: sub a1, a1, a2
10000610: sd a1, 0x0(a0)
10000614: li a0, 0x0
10000618: ld s0, 0x38(sp)
1000061c: ld s1, 0x30(sp)
10000620: ld s2, 0x28(sp)
10000624: ld s3, 0x20(sp)
10000628: ld s4, 0x18(sp)
1000062c: ld s5, 0x10(sp)
10000630: ld s6, 0x8(sp)
10000634: addi sp, sp, 0x40
10000638: ret
