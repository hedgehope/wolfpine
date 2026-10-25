# fft/ame_gauss size=1024 fp64/fp64 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0xb0
10000070: sd ra, 0xa8(sp)
10000074: sd s0, 0xa0(sp)
10000078: sd s1, 0x98(sp)
1000007c: sd s2, 0x90(sp)
10000080: sd s3, 0x88(sp)
10000084: sd s4, 0x80(sp)
10000088: sd s5, 0x78(sp)
1000008c: sd s6, 0x70(sp)
10000090: sd s7, 0x68(sp)
10000094: sd s8, 0x60(sp)
10000098: sd s9, 0x58(sp)
1000009c: sd s10, 0x50(sp)
100000a0: sd s11, 0x48(sp)
100000a4: auipc a0, 0x0
100000a8: auipc a1, 0x18
100000ac: auipc a2, 0x10
100000b0: rdcycle a3
100000b4: sd a3, 0x0(sp)
100000b8: addi a0, a0, 0x5e4
100000bc: addi a1, a1, 0x618
100000c0: addi a2, a2, 0x614
100000c4: lui a3, 0x8
100000c8: add a3, a0, a3
100000cc: fld fa5, 0x0(a2)
100000d0: fld fa4, 0x0(a1)
100000d4: fadd.d fa5, fa5, fa4
100000d8: fsd fa5, 0x0(a0)
100000dc: addi a0, a0, 0x8
100000e0: addi a1, a1, 0x8
100000e4: addi a2, a2, 0x8
100000e8: bne a0, a3, 0x100000cc <main+0x60>
100000ec: li a0, 0x0
100000f0: sd zero, 0x20(sp)
100000f4: li a1, 0x1
100000f8: sd a1, 0x28(sp)
100000fc: auipc a1, 0x0
10000100: auipc a2, 0x10
10000104: li a4, 0x100
10000108: addi a1, a1, 0x50c
1000010c: sd a1, 0x8(sp)
10000110: addi t1, a2, 0x5c0
10000114: li t0, 0x1
10000118: j 0x10000134 <main+0xc8>
1000011c: sd zero, 0x28(sp)
10000120: mul a0, t3, s9
10000124: add s8, a0, s8
10000128: sd s8, 0x20(sp)
1000012c: li a0, 0x1
10000130: ld t0, 0x18(sp)
10000134: sd zero, 0x30(sp)
10000138: slli a0, a0, 0x2
1000013c: ld a1, 0x8(sp)
10000140: add a0, a1, a0
10000144: lw a0, 0x0(a0)
10000148: slli s2, t0, 0x4
1000014c: sd a0, 0x10(sp)
10000150: mulw a1, a0, t0
10000154: li a0, 0x400
10000158: sd a1, 0x18(sp)
1000015c: divw t3, a0, a1
10000160: slli s3, t0, 0x3
10000164: slli s4, t3, 0x3
10000168: mul s5, s4, t0
1000016c: j 0x10000184 <main+0x118>
10000170: ld a1, 0x30(sp)
10000174: addi a1, a1, 0x1
10000178: li a0, 0x4
1000017c: sd a1, 0x30(sp)
10000180: beq a1, a0, 0x1000041c <main+0x3b0>
10000184: auipc a0, 0x8
10000188: auipc a1, 0x2c
1000018c: auipc a2, 0x24
10000190: auipc a3, 0x38
10000194: auipc a5, 0x3a
10000198: auipc a6, 0x3e
1000019c: addi s10, a0, 0x504
100001a0: addi s11, a1, 0x538
100001a4: addi s0, a2, 0x534
100001a8: addi t6, a3, 0x570
100001ac: addi s7, a5, 0x56c
100001b0: addi s8, a6, 0x568
100001b4: li a0, 0xf
100001b8: bge a0, t3, 0x10000310 <main+0x2a4>
100001bc: blez t0, 0x10000170 <main+0x104>
100001c0: li s9, 0x0
100001c4: ld s1, 0x30(sp)
100001c8: slli s1, s1, 0xa
100001cc: sd s10, 0x40(sp)
100001d0: sd s11, 0x38(sp)
100001d4: li s11, 0x0
100001d8: slli a0, s9, 0x5
100001dc: mul ra, a0, t3
100001e0: addi a0, a0, 0x10
100001e4: mul a0, a0, t3
100001e8: li t2, 0x0
100001ec: add s10, s11, s1
100001f0: auipc a1, 0x18
100001f4: auipc a2, 0x0
100001f8: add a3, ra, s10
100001fc: addi t5, a1, 0x4d0
10000200: addi a2, a2, 0x494
10000204: add a1, a0, s10
10000208: slli a3, a3, 0x3
1000020c: slli a5, a1, 0x3
10000210: add a7, t1, a3
10000214: add t4, t5, a3
10000218: add a1, a2, a3
1000021c: add a6, t1, a5
10000220: add a3, t5, a5
10000224: add a5, a2, a5
10000228: li a2, 0x1
1000022c: mx.azero ad=0
10000230: mx.azero ad=1
10000234: mx.azero ad=2
10000238: mx.tload.b td=0 base=a7 stride=s4 fmt=fp64 t=0
1000023c: slli t5, t2, 0xc
10000240: mul t2, t2, s2
10000244: mx.tload.b td=1 base=t4 stride=s4 fmt=fp64 t=0
10000248: add s6, t6, t5
1000024c: mx.tload.b td=2 base=a1 stride=s4 fmt=fp64 t=0
10000250: mx.tload.a td=3 base=s6 stride=a4 fmt=fp64 t=0
10000254: add s6, s7, t5
10000258: mx.tload.a td=4 base=s6 stride=a4 fmt=fp64 t=0
1000025c: add s6, s8, t5
10000260: addi t5, t5, 0x80
10000264: mx.tload.a td=5 base=s6 stride=a4 fmt=fp64 t=0
10000268: add s6, t6, t5
1000026c: mx.mma ad=0 ta=3 tb=0 in=fp64 acc=fp64
10000270: mx.mma ad=1 ta=4 tb=1 in=fp64 acc=fp64
10000274: mx.mma ad=2 ta=5 tb=2 in=fp64 acc=fp64
10000278: mx.tload.b td=0 base=a6 stride=s4 fmt=fp64 t=0
1000027c: mx.tload.b td=1 base=a3 stride=s4 fmt=fp64 t=0
10000280: mx.tload.b td=2 base=a5 stride=s4 fmt=fp64 t=0
10000284: mx.tload.a td=3 base=s6 stride=a4 fmt=fp64 t=0
10000288: add s6, s7, t5
1000028c: mx.tload.a td=4 base=s6 stride=a4 fmt=fp64 t=0
10000290: auipc s6, 0x24
10000294: add t5, s8, t5
10000298: mx.tload.a td=5 base=t5 stride=a4 fmt=fp64 t=0
1000029c: auipc t5, 0x2c
100002a0: add t2, t2, s9
100002a4: addi s6, s6, 0x430
100002a8: mul t2, t2, t3
100002ac: add t2, t2, s10
100002b0: slli t2, t2, 0x3
100002b4: add s6, s6, t2
100002b8: mx.mma ad=0 ta=3 tb=0 in=fp64 acc=fp64
100002bc: mx.mma ad=1 ta=4 tb=1 in=fp64 acc=fp64
100002c0: mx.mma ad=2 ta=5 tb=2 in=fp64 acc=fp64
100002c4: mx.astore as=0 base=s6 stride=s5 fmt=fp64
100002c8: auipc s6, 0x8
100002cc: addi t5, t5, 0x424
100002d0: addi s6, s6, 0x3c0
100002d4: add t5, t5, t2
100002d8: mx.astore as=1 base=t5 stride=s5 fmt=fp64
100002dc: add t2, s6, t2
100002e0: mx.astore as=2 base=t2 stride=s5 fmt=fp64
100002e4: andi t5, a2, 0x1
100002e8: li t2, 0x1
100002ec: li a2, 0x0
100002f0: bnez t5, 0x1000022c <main+0x1c0>
100002f4: addi s11, s11, 0x10
100002f8: blt s11, t3, 0x100001e8 <main+0x17c>
100002fc: addi s9, s9, 0x1
10000300: ld s10, 0x40(sp)
10000304: ld s11, 0x38(sp)
10000308: bne s9, t0, 0x100001d4 <main+0x168>
1000030c: j 0x10000170 <main+0x104>
10000310: blez t0, 0x10000170 <main+0x104>
10000314: li s1, 0x0
10000318: ld a0, 0x30(sp)
1000031c: slli a0, a0, 0xa
10000320: li t2, 0x0
10000324: slli a2, s1, 0x5
10000328: add a1, s1, a0
1000032c: auipc a3, 0x18
10000330: auipc a5, 0x0
10000334: add a2, a2, a0
10000338: addi a7, a3, 0x394
1000033c: addi t4, a5, 0x358
10000340: slli a6, a2, 0x3
10000344: add a2, t1, a6
10000348: add a3, a7, a6
1000034c: add a5, t4, a6
10000350: addi t5, a6, 0x80
10000354: add a6, t1, t5
10000358: add a7, a7, t5
1000035c: add t4, t4, t5
10000360: li s9, 0x1
10000364: mx.azero ad=0
10000368: mx.azero ad=1
1000036c: mx.azero ad=2
10000370: mx.tload.b td=0 base=a2 stride=a4 fmt=fp64 t=1
10000374: slli t5, t2, 0xc
10000378: mul t2, t2, s2
1000037c: mx.tload.b td=1 base=a3 stride=a4 fmt=fp64 t=1
10000380: add s6, t6, t5
10000384: mx.tload.b td=2 base=a5 stride=a4 fmt=fp64 t=1
10000388: mx.tload.a td=3 base=s6 stride=a4 fmt=fp64 t=0
1000038c: add s6, s7, t5
10000390: mx.tload.a td=4 base=s6 stride=a4 fmt=fp64 t=0
10000394: add s6, s8, t5
10000398: addi t5, t5, 0x80
1000039c: add t2, a1, t2
100003a0: mx.tload.a td=5 base=s6 stride=a4 fmt=fp64 t=0
100003a4: add s6, t6, t5
100003a8: mx.mma ad=0 ta=3 tb=0 in=fp64 acc=fp64
100003ac: mx.mma ad=1 ta=4 tb=1 in=fp64 acc=fp64
100003b0: mx.mma ad=2 ta=5 tb=2 in=fp64 acc=fp64
100003b4: mx.tload.b td=0 base=a6 stride=a4 fmt=fp64 t=1
100003b8: mx.tload.b td=1 base=a7 stride=a4 fmt=fp64 t=1
100003bc: mx.tload.b td=2 base=t4 stride=a4 fmt=fp64 t=1
100003c0: mx.tload.a td=3 base=s6 stride=a4 fmt=fp64 t=0
100003c4: add s6, s7, t5
100003c8: add t5, s8, t5
100003cc: slli t2, t2, 0x3
100003d0: mx.tload.a td=4 base=s6 stride=a4 fmt=fp64 t=0
100003d4: add s6, s0, t2
100003d8: mx.tload.a td=5 base=t5 stride=a4 fmt=fp64 t=0
100003dc: add t5, s11, t2
100003e0: mx.mma ad=0 ta=3 tb=0 in=fp64 acc=fp64
100003e4: mx.mma ad=1 ta=4 tb=1 in=fp64 acc=fp64
100003e8: mx.mma ad=2 ta=5 tb=2 in=fp64 acc=fp64
100003ec: mx.astore as=0 base=s6 stride=s3 fmt=fp64
100003f0: mx.astore as=1 base=t5 stride=s3 fmt=fp64
100003f4: add t2, s10, t2
100003f8: mx.astore as=2 base=t2 stride=s3 fmt=fp64
100003fc: andi t5, s9, 0x1
10000400: li t2, 0x1
10000404: li s9, 0x0
10000408: bnez t5, 0x10000364 <main+0x2f8>
1000040c: addiw a1, s1, 0x10
10000410: addi s1, s1, 0x10
10000414: blt a1, t0, 0x10000320 <main+0x2b4>
10000418: j 0x10000170 <main+0x104>
1000041c: ld a0, 0x28(sp)
10000420: andi a0, a0, 0x1
10000424: beqz a0, 0x1000056c <main+0x500>
10000428: li t4, 0x0
1000042c: slli s0, t3, 0x20
10000430: ld s8, 0x20(sp)
10000434: slli t5, s8, 0x3
10000438: auipc a0, 0x22
1000043c: auipc a1, 0x20
10000440: addi a0, a0, 0x288
10000444: addi a1, a1, 0x284
10000448: add a0, a0, t5
1000044c: add t5, a1, t5
10000450: srli t6, s0, 0x20
10000454: srli s0, s0, 0x1d
10000458: ld s9, 0x10(sp)
1000045c: j 0x1000046c <main+0x400>
10000460: addi t4, t4, 0x1
10000464: li a1, 0x4
10000468: beq t4, a1, 0x1000011c <main+0xb0>
1000046c: li a2, 0x0
10000470: li s1, 0x0
10000474: slli s2, t4, 0xa
10000478: j 0x10000488 <main+0x41c>
1000047c: addi s1, s1, 0x1
10000480: add a2, a2, t3
10000484: beq s1, s9, 0x10000460 <main+0x3f4>
10000488: blez t0, 0x1000047c <main+0x410>
1000048c: li a1, 0x0
10000490: slli a3, a2, 0x20
10000494: srli a5, a3, 0x1d
10000498: add a3, a0, a5
1000049c: add a5, t5, a5
100004a0: j 0x100004ac <main+0x440>
100004a4: addiw a1, a1, 0x1
100004a8: beq a1, t0, 0x1000047c <main+0x410>
100004ac: blez t3, 0x100004a4 <main+0x438>
100004b0: li a6, 0x0
100004b4: slli s6, s2, 0x3
100004b8: auipc a7, 0x0
100004bc: auipc t2, 0x18
100004c0: auipc s3, 0x2c
100004c4: auipc s4, 0x24
100004c8: auipc s5, 0x8
100004cc: addi a7, a7, 0x1d0
100004d0: addi t2, t2, 0x204
100004d4: addi s3, s3, 0x200
100004d8: addi s4, s4, 0x1fc
100004dc: addi s5, s5, 0x1c0
100004e0: add a7, a7, s6
100004e4: add t2, t2, s6
100004e8: add s3, s3, s6
100004ec: add s4, s4, s6
100004f0: add s5, s5, s6
100004f4: add s6, t1, s6
100004f8: add s2, t6, s2
100004fc: add s7, s5, a6
10000500: fld fa5, 0x0(s7)
10000504: add s7, s4, a6
10000508: fld fa4, 0x0(s7)
1000050c: add s7, s3, a6
10000510: fld fa3, 0x0(s7)
10000514: add s7, a5, a6
10000518: fld fa2, 0x0(s7)
1000051c: add s7, a3, a6
10000520: fld fa1, 0x0(s7)
10000524: add s7, s6, a6
10000528: fsub.d fa5, fa5, fa4
1000052c: fsub.d fa4, fa4, fa3
10000530: fsub.d fa5, fa5, fa3
10000534: fneg.d fa3, fa1
10000538: fmul.d fa3, fa5, fa3
1000053c: fmadd.d fa3, fa4, fa2, fa3
10000540: fsd fa3, 0x0(s7)
10000544: add s7, t2, a6
10000548: fmul.d fa5, fa5, fa2
1000054c: fmadd.d fa5, fa4, fa1, fa5
10000550: fsd fa5, 0x0(s7)
10000554: add s7, a7, a6
10000558: fadd.d fa5, fa5, fa3
1000055c: addi a6, a6, 0x8
10000560: fsd fa5, 0x0(s7)
10000564: bne s0, a6, 0x100004fc <main+0x490>
10000568: j 0x100004a4 <main+0x438>
1000056c: lui a0, 0x8
10000570: add a0, s10, a0
10000574: fld fa5, 0x0(s0)
10000578: fld fa4, 0x0(s10)
1000057c: fld fa3, 0x0(s11)
10000580: addi s10, s10, 0x8
10000584: fsub.d fa4, fa4, fa5
10000588: fsub.d fa5, fa5, fa3
1000058c: fsd fa5, 0x0(s0)
10000590: fsub.d fa5, fa4, fa3
10000594: fsd fa5, 0x0(s11)
10000598: addi s11, s11, 0x8
1000059c: addi s0, s0, 0x8
100005a0: bne s10, a0, 0x10000574 <main+0x508>
100005a4: mx.azero ad=0
100005a8: mx.azero ad=1
100005ac: mx.azero ad=2
100005b0: mx.azero ad=3
100005b4: rdcycle a0
100005b8: ld a1, 0x0(sp)
100005bc: sub a0, a0, a1
100005c0: auipc a1, 0x34
100005c4: sd a0, 0x100(a1)
100005c8: li a0, 0x0
100005cc: ld ra, 0xa8(sp)
100005d0: ld s0, 0xa0(sp)
100005d4: ld s1, 0x98(sp)
100005d8: ld s2, 0x90(sp)
100005dc: ld s3, 0x88(sp)
100005e0: ld s4, 0x80(sp)
100005e4: ld s5, 0x78(sp)
100005e8: ld s6, 0x70(sp)
100005ec: ld s7, 0x68(sp)
100005f0: ld s8, 0x60(sp)
100005f4: ld s9, 0x58(sp)
100005f8: ld s10, 0x50(sp)
100005fc: ld s11, 0x48(sp)
10000600: addi sp, sp, 0xb0
10000604: ret
