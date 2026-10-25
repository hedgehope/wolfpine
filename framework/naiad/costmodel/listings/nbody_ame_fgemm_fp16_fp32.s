# nbody/ame_fgemm size=[256, "clustered"] fp16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x1a0
10000070: sd ra, 0x198(sp)
10000074: sd s0, 0x190(sp)
10000078: sd s1, 0x188(sp)
1000007c: sd s2, 0x180(sp)
10000080: sd s3, 0x178(sp)
10000084: sd s4, 0x170(sp)
10000088: sd s5, 0x168(sp)
1000008c: sd s6, 0x160(sp)
10000090: sd s7, 0x158(sp)
10000094: sd s8, 0x150(sp)
10000098: sd s9, 0x148(sp)
1000009c: sd s10, 0x140(sp)
100000a0: sd s11, 0x138(sp)
100000a4: fsd fs0, 0x130(sp)
100000a8: fsd fs1, 0x128(sp)
100000ac: fsd fs2, 0x120(sp)
100000b0: lui s2, 0x8001
100000b4: li a0, 0x1f
100000b8: li t0, 0xf
100000bc: auipc a7, 0x48
100000c0: auipc t1, 0x7
100000c4: auipc t2, 0x3
100000c8: auipc t3, 0x1
100000cc: auipc t4, 0x4
100000d0: rdcycle a1
100000d4: sd a1, 0x0(sp)
100000d8: lui a1, 0x80000
100000dc: lui a2, 0x7f800
100000e0: lui t5, 0xc0000
100000e4: li a3, 0xff
100000e8: lui a4, 0x8
100000ec: lui a5, 0x477ff
100000f0: li a6, 0x70
100000f4: lui t6, 0x4b800
100000f8: lui s0, 0x33800
100000fc: addi s2, s2, -0x1
10000100: slli s3, a0, 0xa
10000104: slli t0, t0, 0xa
10000108: addi s1, a7, 0x644
1000010c: addi a7, t1, -0x7f8
10000110: addi s4, t2, -0x7fc
10000114: addi s5, t3, -0x800
10000118: addi s6, t4, 0x7fc
1000011c: fmv.w.x fa5, t5
10000120: addi t1, a4, -0x200
10000124: fmv.w.x fa4, t6
10000128: fmv.w.x fa3, s0
1000012c: sd s1, 0x58(sp)
10000130: addi t2, s1, 0x84
10000134: addi t3, a7, 0x8
10000138: sd s4, 0x50(sp)
1000013c: addi t4, s4, 0x400
10000140: sd s5, 0x10(sp)
10000144: addi t5, s5, 0x4
10000148: sd s6, 0x48(sp)
1000014c: addi t6, s6, 0x4
10000150: lui s7, 0x38000
10000154: j 0x100001b0 <main+0x144>
10000158: add s6, s5, s2
1000015c: slli s5, s5, 0x32
10000160: srli s5, s5, 0x3f
10000164: add s5, s6, s5
10000168: srliw s5, s5, 0xd
1000016c: or s1, s5, s1
10000170: sh s1, 0x2(t5)
10000174: sh t0, 0x4(t5)
10000178: sh t0, 0x200(t4)
1000017c: sh s1, 0x400(t4)
10000180: sh s8, -0x4(t6)
10000184: sh s0, -0x2(t6)
10000188: sh s4, 0x0(t6)
1000018c: sh t0, 0x2(t6)
10000190: addi t2, t2, 0x6
10000194: addi t3, t3, 0xc
10000198: addi t4, t4, 0x2
1000019c: addi t5, t5, 0x20
100001a0: ld s0, 0x58(sp)
100001a4: addi s0, s0, 0x684
100001a8: addi t6, t6, 0x20
100001ac: beq t2, s0, 0x10000490 <main+0x424>
100001b0: lhu s1, -0x4(t2)
100001b4: slli s0, s1, 0x30
100001b8: slli s4, s1, 0x31
100001bc: srai s8, s0, 0x30
100001c0: srli s4, s4, 0x3b
100001c4: and s0, s8, a1
100001c8: beq s4, a0, 0x100001e8 <main+0x17c>
100001cc: andi s1, s1, 0x3ff
100001d0: bnez s4, 0x100001f8 <main+0x18c>
100001d4: fcvt.s.wu fa2, s1
100001d8: fmul.s fa2, fa2, fa3
100001dc: bgez s8, 0x10000210 <main+0x1a4>
100001e0: fneg.s fa2, fa2
100001e4: j 0x10000210 <main+0x1a4>
100001e8: slli s1, s1, 0xd
100001ec: or s1, s1, a2
100001f0: or s0, s0, s1
100001f4: j 0x1000020c <main+0x1a0>
100001f8: slli s4, s4, 0x17
100001fc: slli s1, s1, 0xd
10000200: or s1, s4, s1
10000204: add s0, s0, s7
10000208: add s0, s1, s0
1000020c: fmv.w.x fa2, s0
10000210: lhu s4, -0x2(t2)
10000214: slli s0, s4, 0x30
10000218: slli s1, s4, 0x31
1000021c: srai s0, s0, 0x30
10000220: srli s5, s1, 0x3b
10000224: and s1, s0, a1
10000228: beq s5, a0, 0x10000248 <main+0x1dc>
1000022c: andi s4, s4, 0x3ff
10000230: bnez s5, 0x10000258 <main+0x1ec>
10000234: fcvt.s.wu fa1, s4
10000238: fmul.s fa1, fa1, fa3
1000023c: bgez s0, 0x10000270 <main+0x204>
10000240: fneg.s fa1, fa1
10000244: j 0x10000270 <main+0x204>
10000248: slli s4, s4, 0xd
1000024c: or s4, s4, a2
10000250: or s1, s1, s4
10000254: j 0x1000026c <main+0x200>
10000258: slli s5, s5, 0x17
1000025c: slli s4, s4, 0xd
10000260: or s4, s5, s4
10000264: add s1, s1, s7
10000268: add s1, s4, s1
1000026c: fmv.w.x fa1, s1
10000270: lhu s5, 0x0(t2)
10000274: slli s1, s5, 0x30
10000278: slli s6, s5, 0x31
1000027c: srai s4, s1, 0x30
10000280: srli s6, s6, 0x3b
10000284: and s1, s4, a1
10000288: beq s6, a0, 0x100002a8 <main+0x23c>
1000028c: andi s5, s5, 0x3ff
10000290: bnez s6, 0x100002b8 <main+0x24c>
10000294: fcvt.s.wu fa0, s5
10000298: fmul.s fa0, fa0, fa3
1000029c: bgez s4, 0x100002d0 <main+0x264>
100002a0: fneg.s fa0, fa0
100002a4: j 0x100002d0 <main+0x264>
100002a8: slli s5, s5, 0xd
100002ac: or s5, s5, a2
100002b0: or s1, s1, s5
100002b4: j 0x100002cc <main+0x260>
100002b8: slli s6, s6, 0x17
100002bc: slli s5, s5, 0xd
100002c0: or s5, s6, s5
100002c4: add s1, s1, s7
100002c8: add s1, s5, s1
100002cc: fmv.w.x fa0, s1
100002d0: fsw fa2, -0x8(t3)
100002d4: fsw fa1, -0x4(t3)
100002d8: fsw fa0, 0x0(t3)
100002dc: fmul.s ft0, fa2, fa5
100002e0: fmv.x.w s5, ft0
100002e4: fabs.s ft0, ft0
100002e8: srliw s1, s5, 0x1f
100002ec: fmv.x.w s9, ft0
100002f0: srliw s6, s9, 0x17
100002f4: slli s1, s1, 0xf
100002f8: bltu s6, a3, 0x10000308 <main+0x29c>
100002fc: beq s9, a2, 0x10000314 <main+0x2a8>
10000300: mv s5, t1
10000304: j 0x10000344 <main+0x2d8>
10000308: bltu s9, a5, 0x1000031c <main+0x2b0>
1000030c: or s1, s1, s3
10000310: j 0x10000348 <main+0x2dc>
10000314: slli s5, a0, 0xa
10000318: j 0x10000344 <main+0x2d8>
1000031c: bltu a6, s6, 0x10000330 <main+0x2c4>
10000320: fmul.s ft0, ft0, fa4
10000324: fcvt.l.s s5, ft0, rne
10000328: or s1, s1, s5
1000032c: j 0x10000348 <main+0x2dc>
10000330: add s6, s5, s2
10000334: slli s5, s5, 0x32
10000338: srli s5, s5, 0x3f
1000033c: add s5, s6, s5
10000340: srliw s5, s5, 0xd
10000344: or s1, s5, s1
10000348: sh s1, -0x4(t5)
1000034c: fmul.s ft0, fa1, fa5
10000350: fmv.x.w s5, ft0
10000354: fabs.s ft0, ft0
10000358: srliw s1, s5, 0x1f
1000035c: fmv.x.w s9, ft0
10000360: srliw s6, s9, 0x17
10000364: slli s1, s1, 0xf
10000368: bltu s6, a3, 0x10000378 <main+0x30c>
1000036c: beq s9, a2, 0x10000384 <main+0x318>
10000370: addi s5, a4, -0x200
10000374: j 0x100003b4 <main+0x348>
10000378: bltu s9, a5, 0x1000038c <main+0x320>
1000037c: or s1, s1, s3
10000380: j 0x100003b8 <main+0x34c>
10000384: slli s5, a0, 0xa
10000388: j 0x100003b4 <main+0x348>
1000038c: bltu a6, s6, 0x100003a0 <main+0x334>
10000390: fmul.s ft0, ft0, fa4
10000394: fcvt.l.s s5, ft0, rne
10000398: or s1, s1, s5
1000039c: j 0x100003b8 <main+0x34c>
100003a0: add s6, s5, s2
100003a4: slli s5, s5, 0x32
100003a8: srli s5, s5, 0x3f
100003ac: add s5, s6, s5
100003b0: srliw s5, s5, 0xd
100003b4: or s1, s5, s1
100003b8: sh s1, -0x2(t5)
100003bc: fmul.s ft0, fa0, fa5
100003c0: fmv.x.w s5, ft0
100003c4: fabs.s ft0, ft0
100003c8: srliw s1, s5, 0x1f
100003cc: fmv.x.w s9, ft0
100003d0: srliw s6, s9, 0x17
100003d4: slli s1, s1, 0xf
100003d8: bltu s6, a3, 0x100003e8 <main+0x37c>
100003dc: beq s9, a2, 0x100003f4 <main+0x388>
100003e0: addi s5, a4, -0x200
100003e4: j 0x10000424 <main+0x3b8>
100003e8: bltu s9, a5, 0x100003fc <main+0x390>
100003ec: or s1, s1, s3
100003f0: j 0x10000428 <main+0x3bc>
100003f4: slli s5, a0, 0xa
100003f8: j 0x10000424 <main+0x3b8>
100003fc: bltu a6, s6, 0x10000410 <main+0x3a4>
10000400: fmul.s ft0, ft0, fa4
10000404: fcvt.l.s s5, ft0, rne
10000408: or s1, s1, s5
1000040c: j 0x10000428 <main+0x3bc>
10000410: add s6, s5, s2
10000414: slli s5, s5, 0x32
10000418: srli s5, s5, 0x3f
1000041c: add s5, s6, s5
10000420: srliw s5, s5, 0xd
10000424: or s1, s5, s1
10000428: fmul.s fa1, fa1, fa1
1000042c: sh s1, 0x0(t5)
10000430: sh s8, -0x400(t4)
10000434: sh s0, -0x200(t4)
10000438: sh s4, 0x0(t4)
1000043c: fmadd.s fa2, fa2, fa2, fa1
10000440: fmadd.s fa2, fa0, fa0, fa2
10000444: fmv.x.w s5, fa2
10000448: fabs.s fa2, fa2
1000044c: srliw s1, s5, 0x1f
10000450: fmv.x.w s6, fa2
10000454: srliw s9, s6, 0x17
10000458: slli s1, s1, 0xf
1000045c: bltu s9, a3, 0x10000470 <main+0x404>
10000460: mv s5, s3
10000464: beq s6, a2, 0x1000016c <main+0x100>
10000468: addi s5, a4, -0x200
1000046c: j 0x1000016c <main+0x100>
10000470: bltu s6, a5, 0x1000047c <main+0x410>
10000474: or s1, s1, s3
10000478: j 0x10000170 <main+0x104>
1000047c: bltu a6, s9, 0x10000158 <main+0xec>
10000480: fmul.s fa2, fa2, fa4
10000484: fcvt.l.s s5, fa2, rne
10000488: or s1, s1, s5
1000048c: j 0x10000170 <main+0x104>
10000490: li a6, 0x0
10000494: li a0, 0x0
10000498: addi a7, a7, 0x4
1000049c: auipc a1, 0x47
100004a0: auipc a2, 0x7
100004a4: li s9, 0x20
100004a8: fmv.w.x fs0, zero
100004ac: lui a3, 0x4b800
100004b0: li s8, 0x10
100004b4: lui a4, 0x3f800
100004b8: li s7, 0xff
100004bc: lui a5, 0x8
100004c0: li s0, 0x1f
100004c4: lui s4, 0x7f800
100004c8: lui s6, 0x477ff
100004cc: addi a1, a1, 0x664
100004d0: addi s11, a2, 0x660
100004d4: fmv.w.x fs1, a3
100004d8: fmv.w.x fs2, a4
100004dc: addi s10, a5, -0x200
100004e0: slli s0, s0, 0xa
100004e4: sd a1, 0x8(sp)
100004e8: addi a2, a1, 0x4
100004ec: li s5, 0x70
100004f0: sd a2, 0x20(sp)
100004f4: sd a7, 0x30(sp)
100004f8: mv s1, a0
100004fc: sd a0, 0x40(sp)
10000500: slli a0, a6, 0x6
10000504: sd a6, 0x38(sp)
10000508: slli a1, a6, 0x8
1000050c: sub a1, a1, a0
10000510: ld a0, 0x8(sp)
10000514: add a1, a1, a0
10000518: addi a0, a1, 0xc4
1000051c: sd a0, 0x18(sp)
10000520: addi s1, s1, -0x10
10000524: addi a0, sp, 0x60
10000528: li a2, 0xc0
1000052c: li a1, 0x0
10000530: jal 0x1000082c <memset>
10000534: li a0, 0x0
10000538: li a6, 0x0
1000053c: ld a3, 0x40(sp)
10000540: slli a1, a3, 0x5
10000544: ld a2, 0x10(sp)
10000548: add a1, a2, a1
1000054c: mx.tload.a td=0 base=a1 stride=s9 fmt=fp16 t=0
10000550: mx.azero ad=2
10000554: slli a1, a3, 0x8
10000558: mv a2, a3
1000055c: sd s11, 0x28(sp)
10000560: mv a3, s11
10000564: j 0x100005a8 <main+0x53c>
10000568: auipc a5, 0x7
1000056c: slli a6, a4, 0x5
10000570: addi a5, a5, -0xa0
10000574: ld a7, 0x48(sp)
10000578: add a6, a7, a6
1000057c: li a7, 0x20
10000580: mx.tload.a td=2 base=a5 stride=a7 fmt=fp16 t=0
10000584: li s9, 0x20
10000588: mx.tload.b td=3 base=a6 stride=a7 fmt=fp16 t=0
1000058c: mx.mma ad=2 ta=2 tb=3 in=fp16 acc=fp32
10000590: addi a6, a4, 0x10
10000594: addi a3, a3, 0x40
10000598: addi a2, a2, -0x10
1000059c: addi a0, a0, 0x1
100005a0: li a7, 0xf0
100005a4: bgeu a4, a7, 0x100006dc <main+0x670>
100005a8: li a5, 0x0
100005ac: mv a4, a6
100005b0: slli a6, a0, 0x4
100005b4: mx.azero ad=0
100005b8: slli a7, a4, 0x1
100005bc: add t0, a4, a1
100005c0: auipc t1, 0x7
100005c4: ld t2, 0x50(sp)
100005c8: add a7, t2, a7
100005cc: slli t0, t0, 0x2
100005d0: addi t1, t1, 0x540
100005d4: li t2, 0x200
100005d8: mx.tload.b td=1 base=a7 stride=t2 fmt=fp16 t=0
100005dc: add t0, t1, t0
100005e0: mx.mma ad=0 ta=0 tb=1 in=fp16 acc=fp32
100005e4: li a7, 0x400
100005e8: mx.astore as=0 base=t0 stride=a7 fmt=fp32
100005ec: ld a7, 0x58(sp)
100005f0: flw fa5, 0x40(a7)
100005f4: flw fa4, 0x50(a7)
100005f8: sub a6, s1, a6
100005fc: auipc a7, 0x7
10000600: addi a7, a7, -0x134
10000604: mv t0, a2
10000608: mv t1, a3
1000060c: j 0x10000624 <main+0x5b8>
10000610: addi a5, a5, 0x1
10000614: addi t1, t1, 0x400
10000618: addi t0, t0, 0x1
1000061c: addi a7, a7, 0x20
10000620: beq a5, s8, 0x10000568 <main+0x4fc>
10000624: add t2, a6, a5
10000628: mv t3, a7
1000062c: mv t4, t0
10000630: mv t5, t1
10000634: j 0x10000664 <main+0x5f8>
10000638: add s9, ra, s2
1000063c: slli ra, ra, 0x32
10000640: srli s11, ra, 0x3f
10000644: add s9, s9, s11
10000648: srliw s9, s9, 0xd
1000064c: or t6, s9, t6
10000650: sh t6, 0x0(t3)
10000654: addi t5, t5, 0x4
10000658: addi t4, t4, -0x1
1000065c: addi t3, t3, 0x2
10000660: beq t4, t2, 0x10000610 <main+0x5a4>
10000664: beqz t4, 0x100006b0 <main+0x644>
10000668: flw fa3, 0x0(t5)
1000066c: fmax.s fa3, fa3, fs0
10000670: fadd.s fa3, fa5, fa3
10000674: fsqrt.s fa2, fa3
10000678: fmul.s fa3, fa2, fa3
1000067c: fdiv.s fa3, fs2, fa3
10000680: fmul.s fa3, fa4, fa3
10000684: fmv.x.w ra, fa3
10000688: fabs.s fa3, fa3
1000068c: srliw t6, ra, 0x1f
10000690: fmv.x.w s11, fa3
10000694: srliw s9, s11, 0x17
10000698: slli t6, t6, 0xf
1000069c: bltu s9, s7, 0x100006c8 <main+0x65c>
100006a0: mv s9, s0
100006a4: beq s11, s4, 0x1000064c <main+0x5e0>
100006a8: mv s9, s10
100006ac: j 0x1000064c <main+0x5e0>
100006b0: li t6, 0x0
100006b4: fmv.s fa3, fs0
100006b8: fmul.s fa3, fa3, fs1
100006bc: fcvt.l.s s9, fa3, rne
100006c0: or t6, t6, s9
100006c4: j 0x10000650 <main+0x5e4>
100006c8: bltu s11, s6, 0x100006d4 <main+0x668>
100006cc: or t6, t6, s3
100006d0: j 0x10000650 <main+0x5e4>
100006d4: bltu s5, s9, 0x10000638 <main+0x5cc>
100006d8: j 0x100006b8 <main+0x64c>
100006dc: addi a0, a5, 0x200
100006e0: li a1, 0x40
100006e4: mx.astore as=2 base=a0 stride=a1 fmt=fp32
100006e8: ld a0, 0x58(sp)
100006ec: flw fa5, 0x50(a0)
100006f0: fdiv.s fa5, fs2, fa5
100006f4: addi a0, a5, 0x20c
100006f8: ld a7, 0x30(sp)
100006fc: mv a1, a7
10000700: addi a3, sp, 0x68
10000704: mv a2, a3
10000708: addi a4, sp, 0x128
1000070c: flw fa4, -0xc(a0)
10000710: flw fa3, -0x8(a0)
10000714: flw fa2, -0x4(a0)
10000718: flw fa1, 0x0(a0)
1000071c: flw fa0, -0x4(a1)
10000720: flw ft0, 0x0(a1)
10000724: flw ft1, 0x4(a1)
10000728: fnmsub.s fa4, fa0, fa1, fa4
1000072c: fnmsub.s fa3, ft0, fa1, fa3
10000730: fnmsub.s fa2, ft1, fa1, fa2
10000734: fmul.s fa4, fa5, fa4
10000738: fmul.s fa3, fa5, fa3
1000073c: fmul.s fa2, fa5, fa2
10000740: fsw fa4, -0x8(a2)
10000744: fsw fa3, -0x4(a2)
10000748: fsw fa2, 0x0(a2)
1000074c: addi a2, a2, 0xc
10000750: addi a0, a0, 0x40
10000754: addi a1, a1, 0xc
10000758: bne a2, a4, 0x1000070c <main+0x6a0>
1000075c: mv a0, a3
10000760: ld a2, 0x20(sp)
10000764: mv a1, a2
10000768: ld a6, 0x38(sp)
1000076c: ld a3, 0x18(sp)
10000770: flw fa5, -0x8(a0)
10000774: flw fa4, -0x4(a0)
10000778: flw fa3, 0x0(a0)
1000077c: fsw fa5, -0x4(a1)
10000780: fsw fa4, 0x0(a1)
10000784: fsw fa3, 0x4(a1)
10000788: addi a1, a1, 0xc
1000078c: addi a0, a0, 0xc
10000790: bne a1, a3, 0x10000770 <main+0x704>
10000794: ld a3, 0x40(sp)
10000798: addi a0, a3, 0x10
1000079c: lui a1, 0x4
100007a0: ld s11, 0x28(sp)
100007a4: add s11, s11, a1
100007a8: addi a7, a7, 0xc0
100007ac: addi a2, a2, 0xc0
100007b0: addi a6, a6, 0x1
100007b4: li a1, 0xf0
100007b8: bltu a3, a1, 0x100004f0 <main+0x484>
100007bc: mx.azero ad=0
100007c0: mx.azero ad=1
100007c4: mx.azero ad=2
100007c8: mx.azero ad=3
100007cc: rdcycle a0
100007d0: ld a1, 0x0(sp)
100007d4: sub a0, a0, a1
100007d8: ld a1, 0x58(sp)
100007dc: sd a0, 0x0(a1)
100007e0: li a0, 0x0
100007e4: ld ra, 0x198(sp)
100007e8: ld s0, 0x190(sp)
100007ec: ld s1, 0x188(sp)
100007f0: ld s2, 0x180(sp)
100007f4: ld s3, 0x178(sp)
100007f8: ld s4, 0x170(sp)
100007fc: ld s5, 0x168(sp)
10000800: ld s6, 0x160(sp)
10000804: ld s7, 0x158(sp)
10000808: ld s8, 0x150(sp)
1000080c: ld s9, 0x148(sp)
10000810: ld s10, 0x140(sp)
10000814: ld s11, 0x138(sp)
10000818: fld fs0, 0x130(sp)
1000081c: fld fs1, 0x128(sp)
10000820: fld fs2, 0x120(sp)
10000824: addi sp, sp, 0x1a0
10000828: ret
# <memset>
1000082c: beqz a2, 0x10000848 <memset+0x1c>
10000830: add a2, a0, a2
10000834: mv a3, a0
10000838: addi a4, a3, 0x1
1000083c: sb a1, 0x0(a3)
10000840: mv a3, a4
10000844: bne a4, a2, 0x10000838 <memset+0xc>
10000848: ret
