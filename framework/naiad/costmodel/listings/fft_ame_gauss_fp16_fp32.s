# fft/ame_gauss size=1024 fp16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x100
10000070: sd ra, 0xf8(sp)
10000074: sd s0, 0xf0(sp)
10000078: sd s1, 0xe8(sp)
1000007c: sd s2, 0xe0(sp)
10000080: sd s3, 0xd8(sp)
10000084: sd s4, 0xd0(sp)
10000088: sd s5, 0xc8(sp)
1000008c: sd s6, 0xc0(sp)
10000090: sd s7, 0xb8(sp)
10000094: sd s8, 0xb0(sp)
10000098: sd s9, 0xa8(sp)
1000009c: sd s10, 0xa0(sp)
100000a0: sd s11, 0x98(sp)
100000a4: lui a1, 0x8001
100000a8: li a0, 0x1f
100000ac: auipc t0, 0x1
100000b0: auipc t1, 0x9
100000b4: auipc t2, 0x7
100000b8: rdcycle a2
100000bc: sd a2, 0x0(sp)
100000c0: lui a3, 0x80000
100000c4: lui a4, 0x7f800
100000c8: li a5, 0xff
100000cc: lui t3, 0x8
100000d0: lui t4, 0x2
100000d4: lui a6, 0x477ff
100000d8: li a7, 0x70
100000dc: lui a2, 0x4b800
100000e0: lui t5, 0x33800
100000e4: addi a1, a1, -0x1
100000e8: fmv.w.x fa5, a2
100000ec: slli a2, a0, 0xa
100000f0: addi t0, t0, -0x6a4
100000f4: addi t1, t1, -0x670
100000f8: addi t2, t2, -0x674
100000fc: addi t3, t3, -0x200
10000100: fmv.w.x fa4, t5
10000104: add t4, t0, t4
10000108: lui t5, 0x38000
1000010c: j 0x1000013c <main+0xd0>
10000110: add s1, s0, a1
10000114: slli s0, s0, 0x32
10000118: srli s0, s0, 0x3f
1000011c: add s0, s1, s0
10000120: srliw s0, s0, 0xd
10000124: or t6, s0, t6
10000128: sh t6, 0x0(t0)
1000012c: addi t0, t0, 0x2
10000130: addi t1, t1, 0x2
10000134: addi t2, t2, 0x2
10000138: beq t0, t4, 0x1000024c <main+0x1e0>
1000013c: lhu s1, 0x0(t2)
10000140: slli s0, s1, 0x30
10000144: slli t6, s1, 0x31
10000148: srai s0, s0, 0x30
1000014c: srli s2, t6, 0x3b
10000150: and t6, s0, a3
10000154: beq s2, a0, 0x10000174 <main+0x108>
10000158: andi s1, s1, 0x3ff
1000015c: bnez s2, 0x10000184 <main+0x118>
10000160: fcvt.s.wu fa3, s1
10000164: fmul.s fa3, fa3, fa4
10000168: bgez s0, 0x1000019c <main+0x130>
1000016c: fneg.s fa3, fa3
10000170: j 0x1000019c <main+0x130>
10000174: slli s1, s1, 0xd
10000178: or s1, s1, a4
1000017c: or t6, t6, s1
10000180: j 0x10000198 <main+0x12c>
10000184: slli s2, s2, 0x17
10000188: slli s1, s1, 0xd
1000018c: or s0, s2, s1
10000190: add t6, t6, t5
10000194: add t6, s0, t6
10000198: fmv.w.x fa3, t6
1000019c: lhu s1, 0x0(t1)
100001a0: slli s0, s1, 0x30
100001a4: slli t6, s1, 0x31
100001a8: srai s0, s0, 0x30
100001ac: srli s2, t6, 0x3b
100001b0: and t6, s0, a3
100001b4: beq s2, a0, 0x100001d4 <main+0x168>
100001b8: andi s1, s1, 0x3ff
100001bc: bnez s2, 0x100001e4 <main+0x178>
100001c0: fcvt.s.wu fa2, s1
100001c4: fmul.s fa2, fa2, fa4
100001c8: bgez s0, 0x100001fc <main+0x190>
100001cc: fneg.s fa2, fa2
100001d0: j 0x100001fc <main+0x190>
100001d4: slli s1, s1, 0xd
100001d8: or s1, s1, a4
100001dc: or t6, t6, s1
100001e0: j 0x100001f8 <main+0x18c>
100001e4: slli s2, s2, 0x17
100001e8: slli s1, s1, 0xd
100001ec: or s0, s2, s1
100001f0: add t6, t6, t5
100001f4: add t6, s0, t6
100001f8: fmv.w.x fa2, t6
100001fc: fadd.s fa3, fa3, fa2
10000200: fmv.x.w s0, fa3
10000204: fabs.s fa3, fa3
10000208: srliw t6, s0, 0x1f
1000020c: fmv.x.w s1, fa3
10000210: srliw s2, s1, 0x17
10000214: slli t6, t6, 0xf
10000218: bltu s2, a5, 0x1000022c <main+0x1c0>
1000021c: mv s0, a2
10000220: beq s1, a4, 0x10000124 <main+0xb8>
10000224: mv s0, t3
10000228: j 0x10000124 <main+0xb8>
1000022c: bltu s1, a6, 0x10000238 <main+0x1cc>
10000230: or t6, t6, a2
10000234: j 0x10000128 <main+0xbc>
10000238: bltu a7, s2, 0x10000110 <main+0xa4>
1000023c: fmul.s fa3, fa3, fa5
10000240: fcvt.l.s s0, fa3, rne
10000244: or t6, t6, s0
10000248: j 0x10000128 <main+0xbc>
1000024c: li a0, 0x0
10000250: sd zero, 0x20(sp)
10000254: li a3, 0x1
10000258: sd a3, 0x18(sp)
1000025c: auipc a3, 0x0
10000260: auipc a4, 0x6
10000264: li a6, 0x40
10000268: li t0, 0xff
1000026c: lui a7, 0x8
10000270: li s2, 0x1f
10000274: lui t3, 0x7f800
10000278: lui t4, 0x477ff
1000027c: li t5, 0x70
10000280: lui a5, 0x4b800
10000284: addi a3, a3, 0x744
10000288: sd a3, 0x8(sp)
1000028c: addi a3, a4, 0x7e0
10000290: sd a3, 0x88(sp)
10000294: addi a3, a7, -0x200
10000298: sd a3, 0x90(sp)
1000029c: slli s2, s2, 0xa
100002a0: fmv.w.x fa5, a5
100002a4: li t2, 0x1
100002a8: j 0x100002cc <main+0x260>
100002ac: sd zero, 0x18(sp)
100002b0: ld a0, 0x60(sp)
100002b4: mul a0, t6, a0
100002b8: ld a3, 0x20(sp)
100002bc: add a3, a0, a3
100002c0: sd a3, 0x20(sp)
100002c4: li a0, 0x1
100002c8: ld t2, 0x10(sp)
100002cc: sd zero, 0x28(sp)
100002d0: slli a0, a0, 0x2
100002d4: ld a3, 0x8(sp)
100002d8: add a0, a3, a0
100002dc: lw a0, 0x0(a0)
100002e0: slli ra, t2, 0x4
100002e4: slli a3, t2, 0x2
100002e8: sd a3, 0x30(sp)
100002ec: sd a0, 0x60(sp)
100002f0: mulw a3, a0, t2
100002f4: li a0, 0x400
100002f8: sd a3, 0x10(sp)
100002fc: divw t6, a0, a3
10000300: mul a4, t6, t2
10000304: slli a4, a4, 0x2
10000308: slli s6, t6, 0x1
1000030c: sd t2, 0x80(sp)
10000310: j 0x10000328 <main+0x2bc>
10000314: ld a3, 0x28(sp)
10000318: addi a3, a3, 0x1
1000031c: li a0, 0x4
10000320: sd a3, 0x28(sp)
10000324: beq a3, a0, 0x1000061c <main+0x5b0>
10000328: auipc a0, 0x2
1000032c: auipc a3, 0x10
10000330: auipc a5, 0xc
10000334: auipc a7, 0x16
10000338: auipc t1, 0x16
1000033c: auipc s0, 0x17
10000340: addi a0, a0, 0x6e0
10000344: sd a0, 0x50(sp)
10000348: addi a0, a3, 0x714
1000034c: sd a0, 0x48(sp)
10000350: addi a0, a5, 0x710
10000354: sd a0, 0x40(sp)
10000358: addi s1, a7, -0xb4
1000035c: addi a0, t1, 0x748
10000360: addi s7, s0, 0x744
10000364: li a3, 0xf
10000368: bge a3, t6, 0x100004d0 <main+0x464>
1000036c: blez t2, 0x10000314 <main+0x2a8>
10000370: li a7, 0x0
10000374: ld a3, 0x28(sp)
10000378: slli a3, a3, 0xa
1000037c: sd a3, 0x68(sp)
10000380: li s4, 0x0
10000384: slli a5, a7, 0x5
10000388: mul a3, a5, t6
1000038c: sd a3, 0x78(sp)
10000390: addi a5, a5, 0x10
10000394: mul a3, a5, t6
10000398: sd a3, 0x70(sp)
1000039c: li t2, 0x0
100003a0: ld s0, 0x68(sp)
100003a4: add s0, s4, s0
100003a8: auipc a5, 0x8
100003ac: auipc s3, 0x0
100003b0: ld s5, 0x78(sp)
100003b4: add s5, s5, s0
100003b8: addi s10, a5, 0x698
100003bc: addi a5, s3, 0x65c
100003c0: ld s3, 0x70(sp)
100003c4: add s3, s3, s0
100003c8: slli s5, s5, 0x1
100003cc: slli s3, s3, 0x1
100003d0: ld s9, 0x88(sp)
100003d4: add s8, s9, s5
100003d8: add s11, s10, s5
100003dc: add s5, a5, s5
100003e0: add s9, s9, s3
100003e4: add s10, s10, s3
100003e8: add a5, a5, s3
100003ec: li s3, 0x1
100003f0: mx.azero ad=0
100003f4: mx.azero ad=1
100003f8: mx.azero ad=2
100003fc: mx.tload.b td=0 base=s8 stride=s6 fmt=fp16 t=0
10000400: slli t1, t2, 0xa
10000404: mul t2, t2, ra
10000408: mx.tload.b td=1 base=s11 stride=s6 fmt=fp16 t=0
1000040c: add a3, s1, t1
10000410: mx.tload.b td=2 base=s5 stride=s6 fmt=fp16 t=0
10000414: mx.tload.a td=3 base=a3 stride=a6 fmt=fp16 t=0
10000418: add a3, a0, t1
1000041c: mx.tload.a td=4 base=a3 stride=a6 fmt=fp16 t=0
10000420: add a3, s7, t1
10000424: addi t1, t1, 0x20
10000428: mx.tload.a td=5 base=a3 stride=a6 fmt=fp16 t=0
1000042c: add a3, s1, t1
10000430: mx.mma ad=0 ta=3 tb=0 in=fp16 acc=fp32
10000434: mx.mma ad=1 ta=4 tb=1 in=fp16 acc=fp32
10000438: mx.mma ad=2 ta=5 tb=2 in=fp16 acc=fp32
1000043c: mx.tload.b td=0 base=s9 stride=s6 fmt=fp16 t=0
10000440: mx.tload.b td=1 base=s10 stride=s6 fmt=fp16 t=0
10000444: mx.tload.b td=2 base=a5 stride=s6 fmt=fp16 t=0
10000448: mx.tload.a td=3 base=a3 stride=a6 fmt=fp16 t=0
1000044c: add a3, a0, t1
10000450: mx.tload.a td=4 base=a3 stride=a6 fmt=fp16 t=0
10000454: auipc a3, 0xc
10000458: add t1, s7, t1
1000045c: mx.tload.a td=5 base=t1 stride=a6 fmt=fp16 t=0
10000460: auipc t1, 0x10
10000464: add t2, t2, a7
10000468: addi a3, a3, 0x5ec
1000046c: mul t2, t2, t6
10000470: add t2, t2, s0
10000474: slli t2, t2, 0x2
10000478: add a3, a3, t2
1000047c: mx.mma ad=0 ta=3 tb=0 in=fp16 acc=fp32
10000480: mx.mma ad=1 ta=4 tb=1 in=fp16 acc=fp32
10000484: mx.mma ad=2 ta=5 tb=2 in=fp16 acc=fp32
10000488: mx.astore as=0 base=a3 stride=a4 fmt=fp32
1000048c: auipc a3, 0x2
10000490: addi t1, t1, 0x5e0
10000494: addi a3, a3, 0x57c
10000498: add t1, t1, t2
1000049c: mx.astore as=1 base=t1 stride=a4 fmt=fp32
100004a0: add a3, a3, t2
100004a4: mx.astore as=2 base=a3 stride=a4 fmt=fp32
100004a8: andi a3, s3, 0x1
100004ac: li t2, 0x1
100004b0: li s3, 0x0
100004b4: bnez a3, 0x100003f0 <main+0x384>
100004b8: addi s4, s4, 0x10
100004bc: blt s4, t6, 0x1000039c <main+0x330>
100004c0: addi a7, a7, 0x1
100004c4: ld t2, 0x80(sp)
100004c8: bne a7, t2, 0x10000380 <main+0x314>
100004cc: j 0x10000314 <main+0x2a8>
100004d0: blez t2, 0x10000314 <main+0x2a8>
100004d4: li a7, 0x0
100004d8: ld a3, 0x28(sp)
100004dc: slli a3, a3, 0xa
100004e0: sd a3, 0x38(sp)
100004e4: li t1, 0x0
100004e8: slli a3, a7, 0x5
100004ec: ld a5, 0x38(sp)
100004f0: sd a7, 0x58(sp)
100004f4: add a7, a7, a5
100004f8: sd a7, 0x78(sp)
100004fc: auipc t2, 0x8
10000500: auipc s3, 0x0
10000504: add a3, a3, a5
10000508: addi s9, t2, 0x544
1000050c: addi s10, s3, 0x508
10000510: slli a3, a3, 0x1
10000514: ld s8, 0x88(sp)
10000518: add a5, s8, a3
1000051c: sd a5, 0x70(sp)
10000520: add a5, s9, a3
10000524: sd a5, 0x68(sp)
10000528: add s5, s10, a3
1000052c: addi a3, a3, 0x20
10000530: add s8, s8, a3
10000534: add s9, s9, a3
10000538: add s10, s10, a3
1000053c: li s11, 0x1
10000540: ld a5, 0x50(sp)
10000544: ld s0, 0x48(sp)
10000548: ld a7, 0x40(sp)
1000054c: ld s3, 0x30(sp)
10000550: mx.azero ad=0
10000554: mx.azero ad=1
10000558: mx.azero ad=2
1000055c: ld a3, 0x70(sp)
10000560: mx.tload.b td=0 base=a3 stride=a6 fmt=fp16 t=1
10000564: slli a3, t1, 0xa
10000568: mul s4, t1, ra
1000056c: ld t1, 0x68(sp)
10000570: mx.tload.b td=1 base=t1 stride=a6 fmt=fp16 t=1
10000574: add t2, s1, a3
10000578: mx.tload.b td=2 base=s5 stride=a6 fmt=fp16 t=1
1000057c: mx.tload.a td=3 base=t2 stride=a6 fmt=fp16 t=0
10000580: add t2, a0, a3
10000584: mx.tload.a td=4 base=t2 stride=a6 fmt=fp16 t=0
10000588: add t2, s7, a3
1000058c: addi a3, a3, 0x20
10000590: ld t1, 0x78(sp)
10000594: add t1, t1, s4
10000598: mx.tload.a td=5 base=t2 stride=a6 fmt=fp16 t=0
1000059c: add t2, s1, a3
100005a0: mx.mma ad=0 ta=3 tb=0 in=fp16 acc=fp32
100005a4: mx.mma ad=1 ta=4 tb=1 in=fp16 acc=fp32
100005a8: mx.mma ad=2 ta=5 tb=2 in=fp16 acc=fp32
100005ac: mx.tload.b td=0 base=s8 stride=a6 fmt=fp16 t=1
100005b0: mx.tload.b td=1 base=s9 stride=a6 fmt=fp16 t=1
100005b4: mx.tload.b td=2 base=s10 stride=a6 fmt=fp16 t=1
100005b8: mx.tload.a td=3 base=t2 stride=a6 fmt=fp16 t=0
100005bc: add t2, a0, a3
100005c0: add a3, s7, a3
100005c4: slli t1, t1, 0x2
100005c8: mx.tload.a td=4 base=t2 stride=a6 fmt=fp16 t=0
100005cc: add t2, a7, t1
100005d0: mx.tload.a td=5 base=a3 stride=a6 fmt=fp16 t=0
100005d4: add a3, s0, t1
100005d8: mx.mma ad=0 ta=3 tb=0 in=fp16 acc=fp32
100005dc: mx.mma ad=1 ta=4 tb=1 in=fp16 acc=fp32
100005e0: mx.mma ad=2 ta=5 tb=2 in=fp16 acc=fp32
100005e4: mx.astore as=0 base=t2 stride=s3 fmt=fp32
100005e8: mx.astore as=1 base=a3 stride=s3 fmt=fp32
100005ec: add t1, a5, t1
100005f0: mx.astore as=2 base=t1 stride=s3 fmt=fp32
100005f4: andi a3, s11, 0x1
100005f8: li t1, 0x1
100005fc: li s11, 0x0
10000600: bnez a3, 0x10000550 <main+0x4e4>
10000604: ld a7, 0x58(sp)
10000608: addiw a3, a7, 0x10
1000060c: addi a7, a7, 0x10
10000610: ld t2, 0x80(sp)
10000614: blt a3, t2, 0x100004e4 <main+0x478>
10000618: j 0x10000314 <main+0x2a8>
1000061c: ld a0, 0x18(sp)
10000620: andi a0, a0, 0x1
10000624: beqz a0, 0x100008f8 <main+0x88c>
10000628: li a7, 0x0
1000062c: slli a0, t6, 0x20
10000630: ld a3, 0x20(sp)
10000634: slli a3, a3, 0x2
10000638: auipc a4, 0xb
1000063c: auipc a5, 0xa
10000640: addi a4, a4, 0x408
10000644: addi a5, a5, 0x404
10000648: add a4, a4, a3
1000064c: sd a4, 0x68(sp)
10000650: add a3, a5, a3
10000654: sd a3, 0x58(sp)
10000658: srli s3, a0, 0x20
1000065c: srli s6, a0, 0x1e
10000660: j 0x10000674 <main+0x608>
10000664: ld a7, 0x50(sp)
10000668: addi a7, a7, 0x1
1000066c: li a0, 0x4
10000670: beq a7, a0, 0x100002ac <main+0x240>
10000674: li a3, 0x0
10000678: li a4, 0x0
1000067c: sd a7, 0x50(sp)
10000680: slli s9, a7, 0xa
10000684: j 0x100006a0 <main+0x634>
10000688: ld a4, 0x70(sp)
1000068c: addi a4, a4, 0x1
10000690: ld a3, 0x78(sp)
10000694: add a3, a3, t6
10000698: ld a0, 0x60(sp)
1000069c: beq a4, a0, 0x10000664 <main+0x5f8>
100006a0: sd a4, 0x70(sp)
100006a4: sd a3, 0x78(sp)
100006a8: blez t2, 0x10000688 <main+0x61c>
100006ac: li s10, 0x0
100006b0: slli a0, a3, 0x20
100006b4: srli a0, a0, 0x1e
100006b8: ld s11, 0x68(sp)
100006bc: add s11, s11, a0
100006c0: ld ra, 0x58(sp)
100006c4: add ra, ra, a0
100006c8: j 0x100006d8 <main+0x66c>
100006cc: ld t2, 0x80(sp)
100006d0: addiw s10, s10, 0x1
100006d4: beq s10, t2, 0x10000688 <main+0x61c>
100006d8: blez t6, 0x100006d0 <main+0x664>
100006dc: li a5, 0x0
100006e0: slli s0, s9, 0x1
100006e4: auipc a3, 0x0
100006e8: slli s4, s9, 0x2
100006ec: auipc a0, 0x10
100006f0: auipc a4, 0xc
100006f4: auipc a7, 0x2
100006f8: addi a0, a0, 0x354
100006fc: addi a4, a4, 0x350
10000700: addi t1, a7, 0x314
10000704: add a7, a0, s4
10000708: add a0, a4, s4
1000070c: add s4, t1, s4
10000710: auipc a4, 0x8
10000714: addi a3, a3, 0x324
10000718: addi a4, a4, 0x330
1000071c: add a3, a3, s0
10000720: add a4, a4, s0
10000724: ld t1, 0x88(sp)
10000728: add s0, t1, s0
1000072c: add s9, s3, s9
10000730: j 0x10000750 <main+0x6e4>
10000734: or t1, t1, a2
10000738: sh t1, 0x0(a3)
1000073c: addi a5, a5, 0x4
10000740: addi a3, a3, 0x2
10000744: addi a4, a4, 0x2
10000748: addi s0, s0, 0x2
1000074c: beq s6, a5, 0x100006cc <main+0x660>
10000750: add t1, s4, a5
10000754: add t2, a0, a5
10000758: add s1, a7, a5
1000075c: add s5, ra, a5
10000760: flw fa4, 0x0(t1)
10000764: add t1, s11, a5
10000768: flw fa2, 0x0(t2)
1000076c: flw fa0, 0x0(s1)
10000770: flw fa3, 0x0(t1)
10000774: flw fa1, 0x0(s5)
10000778: fsub.s fa4, fa4, fa2
1000077c: fsub.s fa2, fa2, fa0
10000780: fneg.s ft0, fa3
10000784: fsub.s fa0, fa4, fa0
10000788: fmul.s fa4, fa0, ft0
1000078c: fmadd.s fa4, fa2, fa1, fa4
10000790: fmv.x.w s5, fa4
10000794: fabs.s ft0, fa4
10000798: srliw t1, s5, 0x1f
1000079c: fmv.x.w s8, ft0
100007a0: srliw s7, s8, 0x17
100007a4: slli t1, t1, 0xf
100007a8: bltu s7, t0, 0x100007c0 <main+0x754>
100007ac: mv s5, s2
100007b0: beq s8, t3, 0x100007b8 <main+0x74c>
100007b4: ld s5, 0x90(sp)
100007b8: or t1, s5, t1
100007bc: j 0x100007f8 <main+0x78c>
100007c0: bltu s8, t4, 0x100007cc <main+0x760>
100007c4: or t1, t1, a2
100007c8: j 0x100007f8 <main+0x78c>
100007cc: bltu t5, s7, 0x100007e0 <main+0x774>
100007d0: fmul.s ft0, ft0, fa5
100007d4: fcvt.l.s t2, ft0, rne
100007d8: or t1, t1, t2
100007dc: j 0x100007f8 <main+0x78c>
100007e0: add t2, s5, a1
100007e4: slli s5, s5, 0x32
100007e8: srli s1, s5, 0x3f
100007ec: add t2, t2, s1
100007f0: srliw t2, t2, 0xd
100007f4: or t1, t2, t1
100007f8: fmul.s fa1, fa0, fa1
100007fc: sh t1, 0x0(s0)
10000800: fmadd.s fa3, fa2, fa3, fa1
10000804: fmv.x.w s5, fa3
10000808: fabs.s fa2, fa3
1000080c: srliw t1, s5, 0x1f
10000810: fmv.x.w s8, fa2
10000814: srliw s7, s8, 0x17
10000818: slli t1, t1, 0xf
1000081c: bltu s7, t0, 0x10000834 <main+0x7c8>
10000820: beq s8, t3, 0x10000840 <main+0x7d4>
10000824: lui t2, 0x8
10000828: addi s5, t2, -0x200
1000082c: or t1, s5, t1
10000830: j 0x1000087c <main+0x810>
10000834: bltu s8, t4, 0x10000850 <main+0x7e4>
10000838: or t1, t1, a2
1000083c: j 0x1000087c <main+0x810>
10000840: li s5, 0x1f
10000844: slli s5, s5, 0xa
10000848: or t1, s5, t1
1000084c: j 0x1000087c <main+0x810>
10000850: bltu t5, s7, 0x10000864 <main+0x7f8>
10000854: fmul.s fa2, fa2, fa5
10000858: fcvt.l.s t2, fa2, rne
1000085c: or t1, t1, t2
10000860: j 0x1000087c <main+0x810>
10000864: add t2, s5, a1
10000868: slli s5, s5, 0x32
1000086c: srli s1, s5, 0x3f
10000870: add t2, t2, s1
10000874: srliw t2, t2, 0xd
10000878: or t1, t2, t1
1000087c: sh t1, 0x0(a4)
10000880: fadd.s fa4, fa3, fa4
10000884: fmv.x.w s5, fa4
10000888: fabs.s fa4, fa4
1000088c: srliw t1, s5, 0x1f
10000890: fmv.x.w s8, fa4
10000894: srliw s7, s8, 0x17
10000898: slli t1, t1, 0xf
1000089c: bltu s7, t0, 0x100008b4 <main+0x848>
100008a0: beq s8, t3, 0x100008cc <main+0x860>
100008a4: lui t2, 0x8
100008a8: addi s5, t2, -0x200
100008ac: or t1, s5, t1
100008b0: j 0x10000738 <main+0x6cc>
100008b4: bgeu s8, t4, 0x10000734 <main+0x6c8>
100008b8: bltu t5, s7, 0x100008dc <main+0x870>
100008bc: fmul.s fa4, fa4, fa5
100008c0: fcvt.l.s t2, fa4, rne
100008c4: or t1, t1, t2
100008c8: j 0x10000738 <main+0x6cc>
100008cc: li s5, 0x1f
100008d0: slli s5, s5, 0xa
100008d4: or t1, s5, t1
100008d8: j 0x10000738 <main+0x6cc>
100008dc: add t2, s5, a1
100008e0: slli s5, s5, 0x32
100008e4: srli s1, s5, 0x3f
100008e8: add t2, t2, s1
100008ec: srliw t2, t2, 0xd
100008f0: or t1, t2, t1
100008f4: j 0x10000738 <main+0x6cc>
100008f8: lui a0, 0x4
100008fc: ld a1, 0x50(sp)
10000900: add a0, a1, a0
10000904: ld a2, 0x48(sp)
10000908: ld a3, 0x40(sp)
1000090c: flw fa5, 0x0(a3)
10000910: flw fa4, 0x0(a1)
10000914: flw fa3, 0x0(a2)
10000918: addi a1, a1, 0x4
1000091c: fsub.s fa4, fa4, fa5
10000920: fsub.s fa5, fa5, fa3
10000924: fsw fa5, 0x0(a3)
10000928: fsub.s fa5, fa4, fa3
1000092c: fsw fa5, 0x0(a2)
10000930: addi a2, a2, 0x4
10000934: addi a3, a3, 0x4
10000938: bne a1, a0, 0x1000090c <main+0x8a0>
1000093c: mx.azero ad=0
10000940: mx.azero ad=1
10000944: mx.azero ad=2
10000948: mx.azero ad=3
1000094c: rdcycle a0
10000950: ld a1, 0x0(sp)
10000954: sub a0, a0, a1
10000958: auipc a1, 0x14
1000095c: sd a0, 0xe8(a1)
10000960: li a0, 0x0
10000964: ld ra, 0xf8(sp)
10000968: ld s0, 0xf0(sp)
1000096c: ld s1, 0xe8(sp)
10000970: ld s2, 0xe0(sp)
10000974: ld s3, 0xd8(sp)
10000978: ld s4, 0xd0(sp)
1000097c: ld s5, 0xc8(sp)
10000980: ld s6, 0xc0(sp)
10000984: ld s7, 0xb8(sp)
10000988: ld s8, 0xb0(sp)
1000098c: ld s9, 0xa8(sp)
10000990: ld s10, 0xa0(sp)
10000994: ld s11, 0x98(sp)
10000998: addi sp, sp, 0x100
1000099c: ret
