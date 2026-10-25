# fft/ame size=1024 bf16/fp32 cflags='' design=naiad-dwarfs.yaml
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
100000a4: li a0, 0x0
100000a8: sd zero, 0x18(sp)
100000ac: lui a6, 0x8
100000b0: li a1, 0x1
100000b4: sd a1, 0x20(sp)
100000b8: rdcycle a1
100000bc: sd a1, 0x0(sp)
100000c0: auipc a1, 0x0
100000c4: auipc a2, 0x2
100000c8: auipc a3, 0x4
100000cc: li a4, 0x40
100000d0: lui t2, 0x7f800
100000d4: addi a6, a6, -0x1
100000d8: addi a1, a1, 0x4b0
100000dc: sd a1, 0x8(sp)
100000e0: addi a1, a2, 0x57c
100000e4: sd a1, 0x40(sp)
100000e8: addi s2, a3, 0x578
100000ec: addiw t2, t2, 0x1
100000f0: li a5, 0x1
100000f4: j 0x10000114 <main+0xa8>
100000f8: sd zero, 0x20(sp)
100000fc: ld a0, 0x30(sp)
10000100: mul a0, t6, a0
10000104: add ra, a0, ra
10000108: sd ra, 0x18(sp)
1000010c: li a0, 0x1
10000110: ld a5, 0x10(sp)
10000114: sd zero, 0x28(sp)
10000118: slli a0, a0, 0x2
1000011c: ld a1, 0x8(sp)
10000120: add a0, a1, a0
10000124: lw a0, 0x0(a0)
10000128: slli s4, a5, 0x4
1000012c: slli s5, a5, 0x2
10000130: sd a0, 0x30(sp)
10000134: mulw a1, a0, a5
10000138: li a0, 0x400
1000013c: sd a1, 0x10(sp)
10000140: divw t6, a0, a1
10000144: mul s6, t6, a5
10000148: slli s6, s6, 0x2
1000014c: slli s7, t6, 0x1
10000150: sd a5, 0x38(sp)
10000154: j 0x1000016c <main+0x100>
10000158: ld a1, 0x28(sp)
1000015c: addi a1, a1, 0x1
10000160: li a0, 0x4
10000164: sd a1, 0x28(sp)
10000168: beq a1, a0, 0x1000039c <main+0x330>
1000016c: auipc a0, 0xa
10000170: auipc a1, 0x6
10000174: auipc a2, 0x10
10000178: auipc a3, 0x10
1000017c: auipc a7, 0x11
10000180: addi s0, a0, 0x4d4
10000184: addi s1, a1, 0x4d0
10000188: addi s8, a2, -0x334
1000018c: addi s9, a3, 0x4c8
10000190: addi s10, a7, -0x33c
10000194: li a0, 0xf
10000198: bge a0, t6, 0x100002ac <main+0x240>
1000019c: blez a5, 0x10000158 <main+0xec>
100001a0: li s11, 0x0
100001a4: ld ra, 0x28(sp)
100001a8: slli ra, ra, 0xa
100001ac: li a0, 0x0
100001b0: slli a1, s11, 0x5
100001b4: mul a2, a1, t6
100001b8: addi a1, a1, 0x10
100001bc: mul a7, a1, t6
100001c0: li t4, 0x0
100001c4: add a1, a0, ra
100001c8: add a3, a2, a1
100001cc: add t0, a7, a1
100001d0: slli a3, a3, 0x1
100001d4: slli s3, t0, 0x1
100001d8: ld t5, 0x40(sp)
100001dc: add t0, t5, a3
100001e0: add t1, s2, a3
100001e4: add t5, t5, s3
100001e8: add a3, s2, s3
100001ec: li s3, 0x1
100001f0: mx.azero ad=0
100001f4: mx.azero ad=1
100001f8: mx.tload.b td=0 base=t0 stride=s7 fmt=bf16 t=0
100001fc: slli t3, t4, 0xa
10000200: mul a5, t4, s4
10000204: mx.tload.b td=1 base=t1 stride=s7 fmt=bf16 t=0
10000208: add t4, s8, t3
1000020c: mx.tload.a td=3 base=t4 stride=a4 fmt=bf16 t=0
10000210: add t4, s9, t3
10000214: mx.tload.a td=4 base=t4 stride=a4 fmt=bf16 t=0
10000218: add t4, s10, t3
1000021c: addi t3, t3, 0x20
10000220: add a5, a5, s11
10000224: mx.tload.a td=5 base=t4 stride=a4 fmt=bf16 t=0
10000228: add t4, s8, t3
1000022c: mx.mma ad=0 ta=3 tb=0 in=bf16 acc=fp32
10000230: mx.mma ad=0 ta=5 tb=1 in=bf16 acc=fp32
10000234: mx.mma ad=1 ta=4 tb=0 in=bf16 acc=fp32
10000238: mx.mma ad=1 ta=3 tb=1 in=bf16 acc=fp32
1000023c: mx.tload.b td=0 base=t5 stride=s7 fmt=bf16 t=0
10000240: mx.tload.b td=1 base=a3 stride=s7 fmt=bf16 t=0
10000244: mx.tload.a td=3 base=t4 stride=a4 fmt=bf16 t=0
10000248: add t4, s9, t3
1000024c: add t3, s10, t3
10000250: mul a5, a5, t6
10000254: add a5, a5, a1
10000258: slli a5, a5, 0x2
1000025c: mx.tload.a td=4 base=t4 stride=a4 fmt=bf16 t=0
10000260: add t4, s1, a5
10000264: mx.tload.a td=5 base=t3 stride=a4 fmt=bf16 t=0
10000268: mx.mma ad=0 ta=3 tb=0 in=bf16 acc=fp32
1000026c: mx.mma ad=0 ta=5 tb=1 in=bf16 acc=fp32
10000270: mx.mma ad=1 ta=4 tb=0 in=bf16 acc=fp32
10000274: mx.mma ad=1 ta=3 tb=1 in=bf16 acc=fp32
10000278: mx.astore as=0 base=t4 stride=s6 fmt=fp32
1000027c: add a5, s0, a5
10000280: mx.astore as=1 base=a5 stride=s6 fmt=fp32
10000284: andi a5, s3, 0x1
10000288: li t4, 0x1
1000028c: li s3, 0x0
10000290: bnez a5, 0x100001f0 <main+0x184>
10000294: addi a0, a0, 0x10
10000298: blt a0, t6, 0x100001c0 <main+0x154>
1000029c: addi s11, s11, 0x1
100002a0: ld a5, 0x38(sp)
100002a4: bne s11, a5, 0x100001ac <main+0x140>
100002a8: j 0x10000158 <main+0xec>
100002ac: blez a5, 0x10000158 <main+0xec>
100002b0: li a2, 0x0
100002b4: ld a1, 0x28(sp)
100002b8: slli a1, a1, 0xa
100002bc: li t4, 0x0
100002c0: slli a3, a2, 0x5
100002c4: add a0, a2, a1
100002c8: add a3, a3, a1
100002cc: slli a3, a3, 0x1
100002d0: ld a5, 0x40(sp)
100002d4: add a7, a5, a3
100002d8: add t0, s2, a3
100002dc: addi t1, a3, 0x20
100002e0: add a3, a5, t1
100002e4: add t1, s2, t1
100002e8: li t5, 0x1
100002ec: mx.azero ad=0
100002f0: mx.azero ad=1
100002f4: mx.tload.b td=0 base=a7 stride=a4 fmt=bf16 t=1
100002f8: slli a5, t4, 0xa
100002fc: mul t3, t4, s4
10000300: mx.tload.b td=1 base=t0 stride=a4 fmt=bf16 t=1
10000304: add t4, s8, a5
10000308: mx.tload.a td=3 base=t4 stride=a4 fmt=bf16 t=0
1000030c: add t4, s9, a5
10000310: mx.tload.a td=4 base=t4 stride=a4 fmt=bf16 t=0
10000314: add t4, s10, a5
10000318: addi a5, a5, 0x20
1000031c: add t3, a0, t3
10000320: mx.tload.a td=5 base=t4 stride=a4 fmt=bf16 t=0
10000324: add t4, s8, a5
10000328: mx.mma ad=0 ta=3 tb=0 in=bf16 acc=fp32
1000032c: mx.mma ad=0 ta=5 tb=1 in=bf16 acc=fp32
10000330: mx.mma ad=1 ta=4 tb=0 in=bf16 acc=fp32
10000334: mx.mma ad=1 ta=3 tb=1 in=bf16 acc=fp32
10000338: mx.tload.b td=0 base=a3 stride=a4 fmt=bf16 t=1
1000033c: mx.tload.b td=1 base=t1 stride=a4 fmt=bf16 t=1
10000340: mx.tload.a td=3 base=t4 stride=a4 fmt=bf16 t=0
10000344: add t4, s9, a5
10000348: add a5, s10, a5
1000034c: slli t3, t3, 0x2
10000350: mx.tload.a td=4 base=t4 stride=a4 fmt=bf16 t=0
10000354: add t4, s1, t3
10000358: mx.tload.a td=5 base=a5 stride=a4 fmt=bf16 t=0
1000035c: mx.mma ad=0 ta=3 tb=0 in=bf16 acc=fp32
10000360: mx.mma ad=0 ta=5 tb=1 in=bf16 acc=fp32
10000364: mx.mma ad=1 ta=4 tb=0 in=bf16 acc=fp32
10000368: mx.mma ad=1 ta=3 tb=1 in=bf16 acc=fp32
1000036c: mx.astore as=0 base=t4 stride=s5 fmt=fp32
10000370: add t3, s0, t3
10000374: mx.astore as=1 base=t3 stride=s5 fmt=fp32
10000378: andi a5, t5, 0x1
1000037c: li t4, 0x1
10000380: li t5, 0x0
10000384: bnez a5, 0x100002ec <main+0x280>
10000388: addiw a0, a2, 0x10
1000038c: addi a2, a2, 0x10
10000390: ld a5, 0x38(sp)
10000394: blt a0, a5, 0x100002bc <main+0x250>
10000398: j 0x10000158 <main+0xec>
1000039c: ld a0, 0x20(sp)
100003a0: andi a0, a0, 0x1
100003a4: beqz a0, 0x10000508 <main+0x49c>
100003a8: li a0, 0x0
100003ac: slli a1, t6, 0x20
100003b0: ld ra, 0x18(sp)
100003b4: slli s4, ra, 0x2
100003b8: auipc a2, 0x1
100003bc: auipc a3, 0x0
100003c0: addi s3, a2, 0x248
100003c4: addi a2, a3, 0x244
100003c8: add s3, s3, s4
100003cc: add s4, a2, s4
100003d0: srli s5, a1, 0x20
100003d4: srli s6, a1, 0x1e
100003d8: j 0x100003e8 <main+0x37c>
100003dc: addi a0, a0, 0x1
100003e0: li a1, 0x4
100003e4: beq a0, a1, 0x100000f8 <main+0x8c>
100003e8: li s7, 0x0
100003ec: li s8, 0x0
100003f0: slli s9, a0, 0xa
100003f4: j 0x10000408 <main+0x39c>
100003f8: addi s8, s8, 0x1
100003fc: add s7, s7, t6
10000400: ld a1, 0x30(sp)
10000404: beq s8, a1, 0x100003dc <main+0x370>
10000408: blez a5, 0x100003f8 <main+0x38c>
1000040c: li a1, 0x0
10000410: slli a2, s7, 0x20
10000414: srli a7, a2, 0x1e
10000418: add a2, s3, a7
1000041c: add a7, s4, a7
10000420: j 0x10000430 <main+0x3c4>
10000424: ld a5, 0x38(sp)
10000428: addiw a1, a1, 0x1
1000042c: beq a1, a5, 0x100003f8 <main+0x38c>
10000430: blez t6, 0x10000428 <main+0x3bc>
10000434: li a3, 0x0
10000438: slli t1, s9, 0x1
1000043c: slli s10, s9, 0x2
10000440: add t0, s2, t1
10000444: ld a5, 0x40(sp)
10000448: add t1, a5, t1
1000044c: add t5, s0, s10
10000450: add s10, s1, s10
10000454: add s9, s5, s9
10000458: j 0x10000478 <main+0x40c>
1000045c: srliw a5, s11, 0x10
10000460: ori t4, a5, 0x40
10000464: sh t4, 0x0(t0)
10000468: addi a3, a3, 0x4
1000046c: addi t0, t0, 0x2
10000470: addi t1, t1, 0x2
10000474: beq s6, a3, 0x10000424 <main+0x3b8>
10000478: add a5, s10, a3
1000047c: add t3, t5, a3
10000480: flw fa5, 0x0(a5)
10000484: add a5, a2, a3
10000488: flw fa4, 0x0(t3)
1000048c: flw fa3, 0x0(a5)
10000490: add a5, a7, a3
10000494: flw fa2, 0x0(a5)
10000498: fneg.s fa1, fa3
1000049c: fmul.s fa1, fa4, fa1
100004a0: fmadd.s fa1, fa5, fa2, fa1
100004a4: fmv.x.w t4, fa1
100004a8: slli a5, t4, 0x21
100004ac: srli a5, a5, 0x21
100004b0: bltu a5, t2, 0x100004c0 <main+0x454>
100004b4: srliw a5, t4, 0x10
100004b8: ori t4, a5, 0x40
100004bc: j 0x100004d4 <main+0x468>
100004c0: add a5, t4, a6
100004c4: slli t4, t4, 0x2f
100004c8: srli t3, t4, 0x3f
100004cc: add a5, a5, t3
100004d0: srliw t4, a5, 0x10
100004d4: fmul.s fa4, fa4, fa2
100004d8: fmadd.s fa5, fa5, fa3, fa4
100004dc: fmv.x.w s11, fa5
100004e0: slli a5, s11, 0x21
100004e4: srli a5, a5, 0x21
100004e8: sh t4, 0x0(t1)
100004ec: bgeu a5, t2, 0x1000045c <main+0x3f0>
100004f0: add a5, s11, a6
100004f4: slli s11, s11, 0x2f
100004f8: srli t3, s11, 0x3f
100004fc: add a5, a5, t3
10000500: srliw t4, a5, 0x10
10000504: j 0x10000464 <main+0x3f8>
10000508: mx.azero ad=0
1000050c: mx.azero ad=1
10000510: mx.azero ad=2
10000514: mx.azero ad=3
10000518: rdcycle a0
1000051c: ld a1, 0x0(sp)
10000520: sub a0, a0, a1
10000524: auipc a1, 0x2
10000528: sd a0, 0xdc(a1)
1000052c: li a0, 0x0
10000530: ld ra, 0xa8(sp)
10000534: ld s0, 0xa0(sp)
10000538: ld s1, 0x98(sp)
1000053c: ld s2, 0x90(sp)
10000540: ld s3, 0x88(sp)
10000544: ld s4, 0x80(sp)
10000548: ld s5, 0x78(sp)
1000054c: ld s6, 0x70(sp)
10000550: ld s7, 0x68(sp)
10000554: ld s8, 0x60(sp)
10000558: ld s9, 0x58(sp)
1000055c: ld s10, 0x50(sp)
10000560: ld s11, 0x48(sp)
10000564: addi sp, sp, 0xb0
10000568: ret
