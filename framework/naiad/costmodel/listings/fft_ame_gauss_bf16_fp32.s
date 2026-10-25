# fft/ame_gauss size=1024 bf16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0xc0
10000070: sd ra, 0xb8(sp)
10000074: sd s0, 0xb0(sp)
10000078: sd s1, 0xa8(sp)
1000007c: sd s2, 0xa0(sp)
10000080: sd s3, 0x98(sp)
10000084: sd s4, 0x90(sp)
10000088: sd s5, 0x88(sp)
1000008c: sd s6, 0x80(sp)
10000090: sd s7, 0x78(sp)
10000094: sd s8, 0x70(sp)
10000098: sd s9, 0x68(sp)
1000009c: sd s10, 0x60(sp)
100000a0: sd s11, 0x58(sp)
100000a4: lui a1, 0x8
100000a8: auipc a0, 0x0
100000ac: auipc a2, 0x8
100000b0: auipc a3, 0x6
100000b4: rdcycle a4
100000b8: sd a4, 0x0(sp)
100000bc: lui a4, 0x7f800
100000c0: addi a1, a1, -0x1
100000c4: addi a0, a0, 0x6e0
100000c8: addi a2, a2, 0x714
100000cc: addi a3, a3, 0x710
100000d0: addiw a4, a4, 0x1
100000d4: lui a5, 0x2
100000d8: add a5, a0, a5
100000dc: j 0x10000108 <main+0x9c>
100000e0: add a7, a6, a1
100000e4: slli a6, a6, 0x2f
100000e8: srli a6, a6, 0x3f
100000ec: add a6, a7, a6
100000f0: srliw a6, a6, 0x10
100000f4: sh a6, 0x0(a0)
100000f8: addi a0, a0, 0x2
100000fc: addi a2, a2, 0x2
10000100: addi a3, a3, 0x2
10000104: beq a0, a5, 0x10000140 <main+0xd4>
10000108: lh a6, 0x0(a3)
1000010c: lh a7, 0x0(a2)
10000110: slli a6, a6, 0x10
10000114: slli a7, a7, 0x10
10000118: fmv.w.x fa5, a6
1000011c: fmv.w.x fa4, a7
10000120: fadd.s fa5, fa5, fa4
10000124: fmv.x.w a6, fa5
10000128: slli a7, a6, 0x21
1000012c: srli a7, a7, 0x21
10000130: bltu a7, a4, 0x100000e0 <main+0x74>
10000134: srliw a6, a6, 0x10
10000138: ori a6, a6, 0x40
1000013c: j 0x100000f4 <main+0x88>
10000140: li a0, 0x0
10000144: sd zero, 0x18(sp)
10000148: li a2, 0x1
1000014c: sd a2, 0x20(sp)
10000150: auipc a2, 0x0
10000154: auipc a3, 0x6
10000158: li a5, 0x40
1000015c: lui t1, 0x7f800
10000160: addi a2, a2, 0x5d0
10000164: sd a2, 0x8(sp)
10000168: addi a6, a3, 0x66c
1000016c: addiw t1, t1, 0x1
10000170: li ra, 0x1
10000174: j 0x10000194 <main+0x128>
10000178: sd zero, 0x20(sp)
1000017c: ld a0, 0x50(sp)
10000180: mul a0, t5, a0
10000184: add s11, a0, s11
10000188: sd s11, 0x18(sp)
1000018c: li a0, 0x1
10000190: ld ra, 0x10(sp)
10000194: sd zero, 0x28(sp)
10000198: slli a0, a0, 0x2
1000019c: ld a2, 0x8(sp)
100001a0: add a0, a2, a0
100001a4: lw a0, 0x0(a0)
100001a8: slli s4, ra, 0x4
100001ac: slli s5, ra, 0x2
100001b0: sd a0, 0x50(sp)
100001b4: mulw a2, a0, ra
100001b8: li a0, 0x400
100001bc: sd a2, 0x10(sp)
100001c0: divw t5, a0, a2
100001c4: mul s6, t5, ra
100001c8: slli s6, s6, 0x2
100001cc: slli s7, t5, 0x1
100001d0: sd ra, 0x48(sp)
100001d4: j 0x100001f0 <main+0x184>
100001d8: ld a2, 0x28(sp)
100001dc: addi a2, a2, 0x1
100001e0: li a0, 0x4
100001e4: ld ra, 0x48(sp)
100001e8: sd a2, 0x28(sp)
100001ec: beq a2, a0, 0x10000498 <main+0x42c>
100001f0: auipc a0, 0x2
100001f4: auipc a2, 0x10
100001f8: auipc a3, 0xc
100001fc: auipc a4, 0x16
10000200: auipc a7, 0x16
10000204: auipc t0, 0x17
10000208: addi s8, a0, 0x598
1000020c: addi a0, a2, 0x5cc
10000210: sd a0, 0x40(sp)
10000214: addi a0, a3, 0x5c8
10000218: sd a0, 0x38(sp)
1000021c: addi t2, a4, -0x1fc
10000220: addi s9, a7, 0x600
10000224: addi s10, t0, 0x5fc
10000228: li a0, 0xf
1000022c: bge a0, t5, 0x10000380 <main+0x314>
10000230: blez ra, 0x100001d8 <main+0x16c>
10000234: li s11, 0x0
10000238: ld ra, 0x28(sp)
1000023c: slli ra, ra, 0xa
10000240: sd s8, 0x30(sp)
10000244: li a2, 0x0
10000248: slli a0, s11, 0x5
1000024c: mul a3, a0, t5
10000250: addi a0, a0, 0x10
10000254: mul a7, a0, t5
10000258: li s1, 0x0
1000025c: add a0, a2, ra
10000260: auipc a4, 0x8
10000264: auipc t0, 0x0
10000268: add t3, a3, a0
1000026c: addi s2, a4, 0x560
10000270: addi s0, t0, 0x524
10000274: add a4, a7, a0
10000278: slli t3, t3, 0x1
1000027c: slli s3, a4, 0x1
10000280: add t0, a6, t3
10000284: add t6, s2, t3
10000288: add a4, s0, t3
1000028c: add t4, a6, s3
10000290: add s2, s2, s3
10000294: add t3, s0, s3
10000298: li s0, 0x1
1000029c: mx.azero ad=0
100002a0: mx.azero ad=1
100002a4: mx.azero ad=2
100002a8: mx.tload.b td=0 base=t0 stride=s7 fmt=bf16 t=0
100002ac: slli s3, s1, 0xa
100002b0: mul s1, s1, s4
100002b4: mx.tload.b td=1 base=t6 stride=s7 fmt=bf16 t=0
100002b8: add s8, t2, s3
100002bc: mx.tload.b td=2 base=a4 stride=s7 fmt=bf16 t=0
100002c0: mx.tload.a td=3 base=s8 stride=a5 fmt=bf16 t=0
100002c4: add s8, s9, s3
100002c8: mx.tload.a td=4 base=s8 stride=a5 fmt=bf16 t=0
100002cc: add s8, s10, s3
100002d0: addi s3, s3, 0x20
100002d4: mx.tload.a td=5 base=s8 stride=a5 fmt=bf16 t=0
100002d8: add s8, t2, s3
100002dc: mx.mma ad=0 ta=3 tb=0 in=bf16 acc=fp32
100002e0: mx.mma ad=1 ta=4 tb=1 in=bf16 acc=fp32
100002e4: mx.mma ad=2 ta=5 tb=2 in=bf16 acc=fp32
100002e8: mx.tload.b td=0 base=t4 stride=s7 fmt=bf16 t=0
100002ec: mx.tload.b td=1 base=s2 stride=s7 fmt=bf16 t=0
100002f0: mx.tload.b td=2 base=t3 stride=s7 fmt=bf16 t=0
100002f4: mx.tload.a td=3 base=s8 stride=a5 fmt=bf16 t=0
100002f8: add s8, s9, s3
100002fc: mx.tload.a td=4 base=s8 stride=a5 fmt=bf16 t=0
10000300: auipc s8, 0xc
10000304: add s3, s10, s3
10000308: mx.tload.a td=5 base=s3 stride=a5 fmt=bf16 t=0
1000030c: auipc s3, 0x10
10000310: add s1, s1, s11
10000314: addi s8, s8, 0x4c0
10000318: mul s1, s1, t5
1000031c: add s1, s1, a0
10000320: slli s1, s1, 0x2
10000324: add s8, s8, s1
10000328: mx.mma ad=0 ta=3 tb=0 in=bf16 acc=fp32
1000032c: mx.mma ad=1 ta=4 tb=1 in=bf16 acc=fp32
10000330: mx.mma ad=2 ta=5 tb=2 in=bf16 acc=fp32
10000334: mx.astore as=0 base=s8 stride=s6 fmt=fp32
10000338: auipc s8, 0x2
1000033c: addi s3, s3, 0x4b4
10000340: addi s8, s8, 0x450
10000344: add s3, s3, s1
10000348: mx.astore as=1 base=s3 stride=s6 fmt=fp32
1000034c: add s1, s8, s1
10000350: mx.astore as=2 base=s1 stride=s6 fmt=fp32
10000354: andi s3, s0, 0x1
10000358: li s1, 0x1
1000035c: li s0, 0x0
10000360: bnez s3, 0x1000029c <main+0x230>
10000364: addi a2, a2, 0x10
10000368: blt a2, t5, 0x10000258 <main+0x1ec>
1000036c: addi s11, s11, 0x1
10000370: ld a0, 0x48(sp)
10000374: ld s8, 0x30(sp)
10000378: bne s11, a0, 0x10000244 <main+0x1d8>
1000037c: j 0x100001d8 <main+0x16c>
10000380: blez ra, 0x100001d8 <main+0x16c>
10000384: li a2, 0x0
10000388: ld a0, 0x28(sp)
1000038c: slli a0, a0, 0xa
10000390: li s1, 0x0
10000394: slli a4, a2, 0x5
10000398: add a3, a2, a0
1000039c: auipc a7, 0x8
100003a0: auipc t0, 0x0
100003a4: add a4, a4, a0
100003a8: addi t4, a7, 0x424
100003ac: addi t6, t0, 0x3e8
100003b0: slli t3, a4, 0x1
100003b4: add a4, a6, t3
100003b8: add a7, t4, t3
100003bc: add t0, t6, t3
100003c0: addi s0, t3, 0x20
100003c4: add t3, a6, s0
100003c8: add t4, t4, s0
100003cc: add t6, t6, s0
100003d0: li s0, 0x1
100003d4: ld s11, 0x40(sp)
100003d8: ld ra, 0x38(sp)
100003dc: mx.azero ad=0
100003e0: mx.azero ad=1
100003e4: mx.azero ad=2
100003e8: mx.tload.b td=0 base=a4 stride=a5 fmt=bf16 t=1
100003ec: slli s2, s1, 0xa
100003f0: mul s1, s1, s4
100003f4: mx.tload.b td=1 base=a7 stride=a5 fmt=bf16 t=1
100003f8: add s3, t2, s2
100003fc: mx.tload.b td=2 base=t0 stride=a5 fmt=bf16 t=1
10000400: mx.tload.a td=3 base=s3 stride=a5 fmt=bf16 t=0
10000404: add s3, s9, s2
10000408: mx.tload.a td=4 base=s3 stride=a5 fmt=bf16 t=0
1000040c: add s3, s10, s2
10000410: addi s2, s2, 0x20
10000414: add s1, a3, s1
10000418: mx.tload.a td=5 base=s3 stride=a5 fmt=bf16 t=0
1000041c: add s3, t2, s2
10000420: mx.mma ad=0 ta=3 tb=0 in=bf16 acc=fp32
10000424: mx.mma ad=1 ta=4 tb=1 in=bf16 acc=fp32
10000428: mx.mma ad=2 ta=5 tb=2 in=bf16 acc=fp32
1000042c: mx.tload.b td=0 base=t3 stride=a5 fmt=bf16 t=1
10000430: mx.tload.b td=1 base=t4 stride=a5 fmt=bf16 t=1
10000434: mx.tload.b td=2 base=t6 stride=a5 fmt=bf16 t=1
10000438: mx.tload.a td=3 base=s3 stride=a5 fmt=bf16 t=0
1000043c: add s3, s9, s2
10000440: add s2, s10, s2
10000444: slli s1, s1, 0x2
10000448: mx.tload.a td=4 base=s3 stride=a5 fmt=bf16 t=0
1000044c: add s3, ra, s1
10000450: mx.tload.a td=5 base=s2 stride=a5 fmt=bf16 t=0
10000454: add s2, s11, s1
10000458: mx.mma ad=0 ta=3 tb=0 in=bf16 acc=fp32
1000045c: mx.mma ad=1 ta=4 tb=1 in=bf16 acc=fp32
10000460: mx.mma ad=2 ta=5 tb=2 in=bf16 acc=fp32
10000464: mx.astore as=0 base=s3 stride=s5 fmt=fp32
10000468: mx.astore as=1 base=s2 stride=s5 fmt=fp32
1000046c: add s1, s8, s1
10000470: mx.astore as=2 base=s1 stride=s5 fmt=fp32
10000474: andi s2, s0, 0x1
10000478: li s1, 0x1
1000047c: li s0, 0x0
10000480: bnez s2, 0x100003dc <main+0x370>
10000484: addiw a3, a2, 0x10
10000488: addi a2, a2, 0x10
1000048c: ld a4, 0x48(sp)
10000490: blt a3, a4, 0x10000390 <main+0x324>
10000494: j 0x100001d8 <main+0x16c>
10000498: ld a0, 0x20(sp)
1000049c: andi a0, a0, 0x1
100004a0: beqz a0, 0x1000067c <main+0x610>
100004a4: li t6, 0x0
100004a8: slli a0, t5, 0x20
100004ac: ld s11, 0x18(sp)
100004b0: slli s1, s11, 0x2
100004b4: auipc a2, 0xb
100004b8: auipc a3, 0xa
100004bc: addi s0, a2, 0x30c
100004c0: addi a2, a3, 0x308
100004c4: add s0, s0, s1
100004c8: add s1, a2, s1
100004cc: srli s2, a0, 0x20
100004d0: srli s3, a0, 0x1e
100004d4: j 0x100004e4 <main+0x478>
100004d8: addi t6, t6, 0x1
100004dc: li a0, 0x4
100004e0: beq t6, a0, 0x10000178 <main+0x10c>
100004e4: li s4, 0x0
100004e8: li s5, 0x0
100004ec: slli s6, t6, 0xa
100004f0: j 0x10000504 <main+0x498>
100004f4: addi s5, s5, 0x1
100004f8: add s4, s4, t5
100004fc: ld a0, 0x50(sp)
10000500: beq s5, a0, 0x100004d8 <main+0x46c>
10000504: blez ra, 0x100004f4 <main+0x488>
10000508: li s7, 0x0
1000050c: slli a0, s4, 0x20
10000510: srli a0, a0, 0x1e
10000514: add s8, s0, a0
10000518: add s9, s1, a0
1000051c: j 0x10000528 <main+0x4bc>
10000520: addiw s7, s7, 0x1
10000524: beq s7, ra, 0x100004f4 <main+0x488>
10000528: blez t5, 0x10000520 <main+0x4b4>
1000052c: li a0, 0x0
10000530: slli t3, s6, 0x1
10000534: auipc a7, 0x0
10000538: slli a4, s6, 0x2
1000053c: auipc a2, 0x10
10000540: auipc a3, 0xc
10000544: auipc t0, 0x2
10000548: addi a2, a2, 0x284
1000054c: addi a3, a3, 0x280
10000550: addi t0, t0, 0x244
10000554: add a2, a2, a4
10000558: add a3, a3, a4
1000055c: add a4, t0, a4
10000560: auipc t0, 0x8
10000564: addi a7, a7, 0x254
10000568: addi t0, t0, 0x260
1000056c: add a7, a7, t3
10000570: add t0, t0, t3
10000574: add t3, a6, t3
10000578: add s6, s2, s6
1000057c: j 0x100005a0 <main+0x534>
10000580: srliw t2, s10, 0x10
10000584: ori t4, t2, 0x40
10000588: sh t4, 0x0(a7)
1000058c: addi a0, a0, 0x4
10000590: addi a7, a7, 0x2
10000594: addi t0, t0, 0x2
10000598: addi t3, t3, 0x2
1000059c: beq s3, a0, 0x10000520 <main+0x4b4>
100005a0: add t2, a4, a0
100005a4: add t4, a3, a0
100005a8: flw fa5, 0x0(t2)
100005ac: add t2, a2, a0
100005b0: flw fa2, 0x0(t4)
100005b4: add t4, s9, a0
100005b8: flw fa1, 0x0(t2)
100005bc: add t2, s8, a0
100005c0: flw fa4, 0x0(t4)
100005c4: flw fa3, 0x0(t2)
100005c8: fsub.s fa5, fa5, fa2
100005cc: fsub.s fa2, fa2, fa1
100005d0: fsub.s fa1, fa5, fa1
100005d4: fneg.s fa5, fa3
100005d8: fmul.s fa5, fa1, fa5
100005dc: fmadd.s fa5, fa2, fa4, fa5
100005e0: fmv.x.w t4, fa5
100005e4: slli t2, t4, 0x21
100005e8: srli t2, t2, 0x21
100005ec: bltu t2, t1, 0x100005fc <main+0x590>
100005f0: srliw t2, t4, 0x10
100005f4: ori t4, t2, 0x40
100005f8: j 0x10000610 <main+0x5a4>
100005fc: add t2, t4, a1
10000600: slli t4, t4, 0x2f
10000604: srli t4, t4, 0x3f
10000608: add t2, t2, t4
1000060c: srliw t4, t2, 0x10
10000610: fmul.s fa4, fa1, fa4
10000614: fmadd.s fa4, fa2, fa3, fa4
10000618: fmv.x.w s10, fa4
1000061c: slli t2, s10, 0x21
10000620: srli t2, t2, 0x21
10000624: sh t4, 0x0(t3)
10000628: bltu t2, t1, 0x10000638 <main+0x5cc>
1000062c: srliw t2, s10, 0x10
10000630: ori t4, t2, 0x40
10000634: j 0x1000064c <main+0x5e0>
10000638: add t2, s10, a1
1000063c: slli s10, s10, 0x2f
10000640: srli t4, s10, 0x3f
10000644: add t2, t2, t4
10000648: srliw t4, t2, 0x10
1000064c: fadd.s fa5, fa4, fa5
10000650: fmv.x.w s10, fa5
10000654: slli t2, s10, 0x21
10000658: srli t2, t2, 0x21
1000065c: sh t4, 0x0(t0)
10000660: bgeu t2, t1, 0x10000580 <main+0x514>
10000664: add t2, s10, a1
10000668: slli s10, s10, 0x2f
1000066c: srli t4, s10, 0x3f
10000670: add t2, t2, t4
10000674: srliw t4, t2, 0x10
10000678: j 0x10000588 <main+0x51c>
1000067c: lui a0, 0x4
10000680: add a0, s8, a0
10000684: ld a1, 0x40(sp)
10000688: ld a2, 0x38(sp)
1000068c: flw fa5, 0x0(a2)
10000690: flw fa4, 0x0(s8)
10000694: flw fa3, 0x0(a1)
10000698: addi s8, s8, 0x4
1000069c: fsub.s fa4, fa4, fa5
100006a0: fsub.s fa5, fa5, fa3
100006a4: fsw fa5, 0x0(a2)
100006a8: fsub.s fa5, fa4, fa3
100006ac: fsw fa5, 0x0(a1)
100006b0: addi a1, a1, 0x4
100006b4: addi a2, a2, 0x4
100006b8: bne s8, a0, 0x1000068c <main+0x620>
100006bc: mx.azero ad=0
100006c0: mx.azero ad=1
100006c4: mx.azero ad=2
100006c8: mx.azero ad=3
100006cc: rdcycle a0
100006d0: ld a1, 0x0(sp)
100006d4: sub a0, a0, a1
100006d8: auipc a1, 0x14
100006dc: sd a0, 0xe8(a1)
100006e0: li a0, 0x0
100006e4: ld ra, 0xb8(sp)
100006e8: ld s0, 0xb0(sp)
100006ec: ld s1, 0xa8(sp)
100006f0: ld s2, 0xa0(sp)
100006f4: ld s3, 0x98(sp)
100006f8: ld s4, 0x90(sp)
100006fc: ld s5, 0x88(sp)
10000700: ld s6, 0x80(sp)
10000704: ld s7, 0x78(sp)
10000708: ld s8, 0x70(sp)
1000070c: ld s9, 0x68(sp)
10000710: ld s10, 0x60(sp)
10000714: ld s11, 0x58(sp)
10000718: addi sp, sp, 0xc0
1000071c: ret
