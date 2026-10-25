# fft/ame size=1024 fp16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0xe0
10000070: sd ra, 0xd8(sp)
10000074: sd s0, 0xd0(sp)
10000078: sd s1, 0xc8(sp)
1000007c: sd s2, 0xc0(sp)
10000080: sd s3, 0xb8(sp)
10000084: sd s4, 0xb0(sp)
10000088: sd s5, 0xa8(sp)
1000008c: sd s6, 0xa0(sp)
10000090: sd s7, 0x98(sp)
10000094: sd s8, 0x90(sp)
10000098: sd s9, 0x88(sp)
1000009c: sd s10, 0x80(sp)
100000a0: sd s11, 0x78(sp)
100000a4: li a0, 0x0
100000a8: sd zero, 0x28(sp)
100000ac: lui t4, 0x8001
100000b0: li t5, 0x1f
100000b4: li a1, 0x1
100000b8: sd a1, 0x20(sp)
100000bc: rdcycle a1
100000c0: sd a1, 0x8(sp)
100000c4: auipc a1, 0x0
100000c8: auipc a3, 0x2
100000cc: auipc a4, 0x4
100000d0: li a5, 0x40
100000d4: li a7, 0xff
100000d8: lui a2, 0x8
100000dc: lui t1, 0x7f800
100000e0: lui t2, 0x477ff
100000e4: li t3, 0x70
100000e8: lui a6, 0x4b800
100000ec: addi t4, t4, -0x1
100000f0: slli t5, t5, 0xa
100000f4: addi a1, a1, 0x5ac
100000f8: sd a1, 0x10(sp)
100000fc: addi a1, a3, 0x678
10000100: sd a1, 0x70(sp)
10000104: addi a1, a4, 0x674
10000108: sd a1, 0x68(sp)
1000010c: addi s2, a2, -0x200
10000110: fmv.w.x fa5, a6
10000114: li s3, 0x1
10000118: j 0x1000013c <main+0xd0>
1000011c: sd zero, 0x20(sp)
10000120: ld a0, 0x40(sp)
10000124: mul a0, a3, a0
10000128: ld a1, 0x28(sp)
1000012c: add a1, a0, a1
10000130: sd a1, 0x28(sp)
10000134: li a0, 0x1
10000138: ld s3, 0x18(sp)
1000013c: sd zero, 0x30(sp)
10000140: slli a0, a0, 0x2
10000144: ld a1, 0x10(sp)
10000148: add a0, a1, a0
1000014c: lw a0, 0x0(a0)
10000150: slli s11, s3, 0x4
10000154: slli a1, s3, 0x2
10000158: sd a1, 0x38(sp)
1000015c: sd a0, 0x40(sp)
10000160: mulw a1, a0, s3
10000164: li a0, 0x400
10000168: sd a1, 0x18(sp)
1000016c: divw a3, a0, a1
10000170: mul a1, a3, s3
10000174: slli a1, a1, 0x2
10000178: slli s6, a3, 0x1
1000017c: sd s3, 0x60(sp)
10000180: j 0x10000198 <main+0x12c>
10000184: ld a2, 0x30(sp)
10000188: addi a2, a2, 0x1
1000018c: li a0, 0x4
10000190: sd a2, 0x30(sp)
10000194: beq a2, a0, 0x100003f4 <main+0x388>
10000198: auipc a0, 0xa
1000019c: auipc a4, 0x6
100001a0: auipc t6, 0x10
100001a4: auipc s0, 0x10
100001a8: auipc s1, 0x11
100001ac: addi a6, a0, 0x5a8
100001b0: addi s10, a4, 0x5a4
100001b4: addi t0, t6, -0x260
100001b8: addi a0, s0, 0x59c
100001bc: addi s9, s1, -0x268
100001c0: li a2, 0xf
100001c4: bge a2, a3, 0x100002f4 <main+0x288>
100001c8: blez s3, 0x10000184 <main+0x118>
100001cc: li s1, 0x0
100001d0: ld a2, 0x30(sp)
100001d4: slli a2, a2, 0xa
100001d8: sd a2, 0x48(sp)
100001dc: li s7, 0x0
100001e0: slli a4, s1, 0x5
100001e4: mul a2, a4, a3
100001e8: sd a2, 0x58(sp)
100001ec: addi a4, a4, 0x10
100001f0: mul a2, a4, a3
100001f4: sd a2, 0x50(sp)
100001f8: li a2, 0x0
100001fc: ld s5, 0x48(sp)
10000200: add s5, s7, s5
10000204: ld a4, 0x58(sp)
10000208: add a4, a4, s5
1000020c: ld s3, 0x50(sp)
10000210: add s3, s3, s5
10000214: slli s4, a4, 0x1
10000218: slli s3, s3, 0x1
1000021c: ld ra, 0x70(sp)
10000220: add a4, ra, s4
10000224: ld t6, 0x68(sp)
10000228: add s4, t6, s4
1000022c: add ra, ra, s3
10000230: add s3, t6, s3
10000234: li s8, 0x1
10000238: mx.azero ad=0
1000023c: mx.azero ad=1
10000240: mx.tload.b td=0 base=a4 stride=s6 fmt=fp16 t=0
10000244: slli s0, a2, 0xa
10000248: mul a2, a2, s11
1000024c: mx.tload.b td=1 base=s4 stride=s6 fmt=fp16 t=0
10000250: add t6, t0, s0
10000254: mx.tload.a td=3 base=t6 stride=a5 fmt=fp16 t=0
10000258: add t6, a0, s0
1000025c: mx.tload.a td=4 base=t6 stride=a5 fmt=fp16 t=0
10000260: add t6, s9, s0
10000264: addi s0, s0, 0x20
10000268: add a2, a2, s1
1000026c: mx.tload.a td=5 base=t6 stride=a5 fmt=fp16 t=0
10000270: add t6, t0, s0
10000274: mx.mma ad=0 ta=3 tb=0 in=fp16 acc=fp32
10000278: mx.mma ad=0 ta=5 tb=1 in=fp16 acc=fp32
1000027c: mx.mma ad=1 ta=4 tb=0 in=fp16 acc=fp32
10000280: mx.mma ad=1 ta=3 tb=1 in=fp16 acc=fp32
10000284: mx.tload.b td=0 base=ra stride=s6 fmt=fp16 t=0
10000288: mx.tload.b td=1 base=s3 stride=s6 fmt=fp16 t=0
1000028c: mx.tload.a td=3 base=t6 stride=a5 fmt=fp16 t=0
10000290: add t6, a0, s0
10000294: add s0, s9, s0
10000298: mul a2, a2, a3
1000029c: add a2, a2, s5
100002a0: slli a2, a2, 0x2
100002a4: mx.tload.a td=4 base=t6 stride=a5 fmt=fp16 t=0
100002a8: add t6, s10, a2
100002ac: mx.tload.a td=5 base=s0 stride=a5 fmt=fp16 t=0
100002b0: mx.mma ad=0 ta=3 tb=0 in=fp16 acc=fp32
100002b4: mx.mma ad=0 ta=5 tb=1 in=fp16 acc=fp32
100002b8: mx.mma ad=1 ta=4 tb=0 in=fp16 acc=fp32
100002bc: mx.mma ad=1 ta=3 tb=1 in=fp16 acc=fp32
100002c0: mx.astore as=0 base=t6 stride=a1 fmt=fp32
100002c4: add a2, a6, a2
100002c8: mx.astore as=1 base=a2 stride=a1 fmt=fp32
100002cc: andi t6, s8, 0x1
100002d0: li a2, 0x1
100002d4: li s8, 0x0
100002d8: bnez t6, 0x10000238 <main+0x1cc>
100002dc: addi s7, s7, 0x10
100002e0: blt s7, a3, 0x100001f8 <main+0x18c>
100002e4: addi s1, s1, 0x1
100002e8: ld s3, 0x60(sp)
100002ec: bne s1, s3, 0x100001dc <main+0x170>
100002f0: j 0x10000184 <main+0x118>
100002f4: blez s3, 0x10000184 <main+0x118>
100002f8: li s1, 0x0
100002fc: ld s0, 0x30(sp)
10000300: slli s0, s0, 0xa
10000304: sd s0, 0x58(sp)
10000308: li ra, 0x0
1000030c: slli a2, s1, 0x5
10000310: ld a4, 0x58(sp)
10000314: add s3, s1, a4
10000318: add a2, a2, a4
1000031c: slli a2, a2, 0x1
10000320: ld s5, 0x70(sp)
10000324: add a4, s5, a2
10000328: ld t6, 0x68(sp)
1000032c: add s4, t6, a2
10000330: addi s7, a2, 0x20
10000334: add s5, s5, s7
10000338: add s7, t6, s7
1000033c: li s8, 0x1
10000340: ld s0, 0x38(sp)
10000344: mx.azero ad=0
10000348: mx.azero ad=1
1000034c: mx.tload.b td=0 base=a4 stride=a5 fmt=fp16 t=1
10000350: slli a2, ra, 0xa
10000354: mul t6, ra, s11
10000358: mx.tload.b td=1 base=s4 stride=a5 fmt=fp16 t=1
1000035c: add ra, t0, a2
10000360: mx.tload.a td=3 base=ra stride=a5 fmt=fp16 t=0
10000364: add ra, a0, a2
10000368: mx.tload.a td=4 base=ra stride=a5 fmt=fp16 t=0
1000036c: add ra, s9, a2
10000370: addi a2, a2, 0x20
10000374: add t6, s3, t6
10000378: mx.tload.a td=5 base=ra stride=a5 fmt=fp16 t=0
1000037c: add ra, t0, a2
10000380: mx.mma ad=0 ta=3 tb=0 in=fp16 acc=fp32
10000384: mx.mma ad=0 ta=5 tb=1 in=fp16 acc=fp32
10000388: mx.mma ad=1 ta=4 tb=0 in=fp16 acc=fp32
1000038c: mx.mma ad=1 ta=3 tb=1 in=fp16 acc=fp32
10000390: mx.tload.b td=0 base=s5 stride=a5 fmt=fp16 t=1
10000394: mx.tload.b td=1 base=s7 stride=a5 fmt=fp16 t=1
10000398: mx.tload.a td=3 base=ra stride=a5 fmt=fp16 t=0
1000039c: add ra, a0, a2
100003a0: add a2, s9, a2
100003a4: slli t6, t6, 0x2
100003a8: mx.tload.a td=4 base=ra stride=a5 fmt=fp16 t=0
100003ac: add ra, s10, t6
100003b0: mx.tload.a td=5 base=a2 stride=a5 fmt=fp16 t=0
100003b4: mx.mma ad=0 ta=3 tb=0 in=fp16 acc=fp32
100003b8: mx.mma ad=0 ta=5 tb=1 in=fp16 acc=fp32
100003bc: mx.mma ad=1 ta=4 tb=0 in=fp16 acc=fp32
100003c0: mx.mma ad=1 ta=3 tb=1 in=fp16 acc=fp32
100003c4: mx.astore as=0 base=ra stride=s0 fmt=fp32
100003c8: add t6, a6, t6
100003cc: mx.astore as=1 base=t6 stride=s0 fmt=fp32
100003d0: andi a2, s8, 0x1
100003d4: li ra, 0x1
100003d8: li s8, 0x0
100003dc: bnez a2, 0x10000344 <main+0x2d8>
100003e0: addiw a2, s1, 0x10
100003e4: addi s1, s1, 0x10
100003e8: ld s3, 0x60(sp)
100003ec: blt a2, s3, 0x10000308 <main+0x29c>
100003f0: j 0x10000184 <main+0x118>
100003f4: ld a0, 0x20(sp)
100003f8: andi a0, a0, 0x1
100003fc: beqz a0, 0x1000060c <main+0x5a0>
10000400: li t0, 0x0
10000404: slli a0, a3, 0x20
10000408: ld a1, 0x28(sp)
1000040c: slli a1, a1, 0x2
10000410: auipc a2, 0x1
10000414: auipc a4, 0x0
10000418: addi a2, a2, 0x2f0
1000041c: addi a4, a4, 0x2ec
10000420: add a2, a2, a1
10000424: sd a2, 0x48(sp)
10000428: add a1, a4, a1
1000042c: sd a1, 0x38(sp)
10000430: srli s6, a0, 0x20
10000434: srli s9, a0, 0x1e
10000438: j 0x1000044c <main+0x3e0>
1000043c: ld t0, 0x30(sp)
10000440: addi t0, t0, 0x1
10000444: li a0, 0x4
10000448: beq t0, a0, 0x1000011c <main+0xb0>
1000044c: li a1, 0x0
10000450: li a2, 0x0
10000454: sd t0, 0x30(sp)
10000458: slli ra, t0, 0xa
1000045c: j 0x10000478 <main+0x40c>
10000460: ld a2, 0x50(sp)
10000464: addi a2, a2, 0x1
10000468: ld a1, 0x58(sp)
1000046c: add a1, a1, a3
10000470: ld a0, 0x40(sp)
10000474: beq a2, a0, 0x1000043c <main+0x3d0>
10000478: sd a2, 0x50(sp)
1000047c: sd a1, 0x58(sp)
10000480: blez s3, 0x10000460 <main+0x3f4>
10000484: li s5, 0x0
10000488: slli a0, a1, 0x20
1000048c: srli a0, a0, 0x1e
10000490: ld a1, 0x48(sp)
10000494: add a1, a1, a0
10000498: ld t6, 0x38(sp)
1000049c: add t6, t6, a0
100004a0: j 0x100004b0 <main+0x444>
100004a4: ld s3, 0x60(sp)
100004a8: addiw s5, s5, 0x1
100004ac: beq s5, s3, 0x10000460 <main+0x3f4>
100004b0: blez a3, 0x100004a8 <main+0x43c>
100004b4: li a0, 0x0
100004b8: slli s0, ra, 0x1
100004bc: slli s8, ra, 0x2
100004c0: ld s1, 0x68(sp)
100004c4: add s1, s1, s0
100004c8: ld a2, 0x70(sp)
100004cc: add s0, a2, s0
100004d0: add s7, a6, s8
100004d4: add s8, s10, s8
100004d8: add ra, s6, ra
100004dc: j 0x100004f8 <main+0x48c>
100004e0: or a4, s3, a4
100004e4: sh a4, 0x0(s1)
100004e8: addi a0, a0, 0x4
100004ec: addi s1, s1, 0x2
100004f0: addi s0, s0, 0x2
100004f4: beq s9, a0, 0x100004a4 <main+0x438>
100004f8: add a2, s8, a0
100004fc: add a4, s7, a0
10000500: add t0, t6, a0
10000504: add s3, a1, a0
10000508: flw fa4, 0x0(a2)
1000050c: flw fa3, 0x0(s3)
10000510: flw fa2, 0x0(a4)
10000514: flw fa1, 0x0(t0)
10000518: fneg.s fa0, fa3
1000051c: fmul.s fa0, fa2, fa0
10000520: fmadd.s fa0, fa4, fa1, fa0
10000524: fmv.x.w s3, fa0
10000528: fabs.s fa0, fa0
1000052c: srliw a4, s3, 0x1f
10000530: fmv.x.w s11, fa0
10000534: srliw s4, s11, 0x17
10000538: slli a4, a4, 0xf
1000053c: bltu s4, a7, 0x1000054c <main+0x4e0>
10000540: beq s11, t1, 0x10000558 <main+0x4ec>
10000544: or a4, s2, a4
10000548: j 0x10000594 <main+0x528>
1000054c: bltu s11, t2, 0x10000568 <main+0x4fc>
10000550: or a4, a4, t5
10000554: j 0x10000594 <main+0x528>
10000558: li s3, 0x1f
1000055c: slli s3, s3, 0xa
10000560: or a4, s3, a4
10000564: j 0x10000594 <main+0x528>
10000568: bltu t3, s4, 0x1000057c <main+0x510>
1000056c: fmul.s fa0, fa0, fa5
10000570: fcvt.l.s a2, fa0, rne
10000574: or a4, a4, a2
10000578: j 0x10000594 <main+0x528>
1000057c: add a2, s3, t4
10000580: slli s3, s3, 0x32
10000584: srli t0, s3, 0x3f
10000588: add a2, a2, t0
1000058c: srliw a2, a2, 0xd
10000590: or a4, a2, a4
10000594: fmul.s fa2, fa2, fa1
10000598: sh a4, 0x0(s0)
1000059c: fmadd.s fa4, fa4, fa3, fa2
100005a0: fmv.x.w s3, fa4
100005a4: fabs.s fa4, fa4
100005a8: srliw a4, s3, 0x1f
100005ac: fmv.x.w s4, fa4
100005b0: srliw s11, s4, 0x17
100005b4: slli a4, a4, 0xf
100005b8: bltu s11, a7, 0x100005d0 <main+0x564>
100005bc: mv s3, t5
100005c0: beq s4, t1, 0x100004e0 <main+0x474>
100005c4: lui a2, 0x8
100005c8: addi s3, a2, -0x200
100005cc: j 0x100004e0 <main+0x474>
100005d0: bltu s4, t2, 0x100005dc <main+0x570>
100005d4: or a4, a4, t5
100005d8: j 0x100004e4 <main+0x478>
100005dc: bltu t3, s11, 0x100005f0 <main+0x584>
100005e0: fmul.s fa4, fa4, fa5
100005e4: fcvt.l.s a2, fa4, rne
100005e8: or a4, a4, a2
100005ec: j 0x100004e4 <main+0x478>
100005f0: add a2, s3, t4
100005f4: slli s3, s3, 0x32
100005f8: srli t0, s3, 0x3f
100005fc: add a2, a2, t0
10000600: srliw a2, a2, 0xd
10000604: or a4, a2, a4
10000608: j 0x100004e4 <main+0x478>
1000060c: mx.azero ad=0
10000610: mx.azero ad=1
10000614: mx.azero ad=2
10000618: mx.azero ad=3
1000061c: rdcycle a0
10000620: ld a1, 0x8(sp)
10000624: sub a0, a0, a1
10000628: auipc a1, 0x2
1000062c: sd a0, 0xd8(a1)
10000630: li a0, 0x0
10000634: ld ra, 0xd8(sp)
10000638: ld s0, 0xd0(sp)
1000063c: ld s1, 0xc8(sp)
10000640: ld s2, 0xc0(sp)
10000644: ld s3, 0xb8(sp)
10000648: ld s4, 0xb0(sp)
1000064c: ld s5, 0xa8(sp)
10000650: ld s6, 0xa0(sp)
10000654: ld s7, 0x98(sp)
10000658: ld s8, 0x90(sp)
1000065c: ld s9, 0x88(sp)
10000660: ld s10, 0x80(sp)
10000664: ld s11, 0x78(sp)
10000668: addi sp, sp, 0xe0
1000066c: ret
