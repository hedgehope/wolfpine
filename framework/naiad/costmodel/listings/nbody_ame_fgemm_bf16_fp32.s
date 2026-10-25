# nbody/ame_fgemm size=[256, "clustered"] bf16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x180
10000070: sd ra, 0x178(sp)
10000074: sd s0, 0x170(sp)
10000078: sd s1, 0x168(sp)
1000007c: sd s2, 0x160(sp)
10000080: sd s3, 0x158(sp)
10000084: sd s4, 0x150(sp)
10000088: sd s5, 0x148(sp)
1000008c: sd s6, 0x140(sp)
10000090: sd s7, 0x138(sp)
10000094: sd s8, 0x130(sp)
10000098: sd s9, 0x128(sp)
1000009c: sd s10, 0x120(sp)
100000a0: sd s11, 0x118(sp)
100000a4: fsd fs0, 0x110(sp)
100000a8: fsd fs1, 0x108(sp)
100000ac: lui s2, 0x8
100000b0: lui a0, 0x4
100000b4: auipc a2, 0x48
100000b8: auipc a3, 0x6
100000bc: auipc a4, 0x2
100000c0: auipc a5, 0x0
100000c4: auipc a6, 0x4
100000c8: rdcycle a1
100000cc: sd a1, 0x8(sp)
100000d0: lui a7, 0xc0000
100000d4: lui t0, 0x7f800
100000d8: addi s2, s2, -0x1
100000dc: addi a1, a0, -0x80
100000e0: addi s1, a2, 0x38c
100000e4: addi a0, a3, 0x550
100000e8: addi s3, a4, 0x54c
100000ec: addi t1, a5, 0x548
100000f0: addi s5, a6, 0x544
100000f4: fmv.w.x fa5, a7
100000f8: addiw a2, t0, 0x1
100000fc: addi a3, s1, 0x84
10000100: addi a4, a0, 0x8
10000104: addi a5, s3, 0x400
10000108: sd t1, 0x18(sp)
1000010c: addi a6, t1, 0x4
10000110: addi a7, s5, 0x4
10000114: addi t0, s1, 0x684
10000118: j 0x10000168 <main+0xfc>
1000011c: add t5, t4, s2
10000120: slli t4, t4, 0x2f
10000124: srli t4, t4, 0x3f
10000128: add t4, t5, t4
1000012c: srliw t4, t4, 0x10
10000130: sh t4, 0x2(a6)
10000134: sh a1, 0x4(a6)
10000138: sh a1, 0x200(a5)
1000013c: sh t4, 0x400(a5)
10000140: sh t1, -0x4(a7)
10000144: sh t2, -0x2(a7)
10000148: sh t3, 0x0(a7)
1000014c: sh a1, 0x2(a7)
10000150: addi a3, a3, 0x6
10000154: addi a4, a4, 0xc
10000158: addi a5, a5, 0x2
1000015c: addi a6, a6, 0x20
10000160: addi a7, a7, 0x20
10000164: beq a3, t0, 0x10000274 <main+0x208>
10000168: lhu t1, -0x4(a3)
1000016c: lhu t2, -0x2(a3)
10000170: lhu t3, 0x0(a3)
10000174: slli s0, t1, 0x10
10000178: slli t5, t2, 0x10
1000017c: slli t4, t3, 0x10
10000180: fmv.w.x fa4, s0
10000184: fmul.s fa3, fa4, fa5
10000188: fmv.x.w t6, fa3
1000018c: slli s4, t6, 0x21
10000190: srli s4, s4, 0x21
10000194: sw s0, -0x8(a4)
10000198: sw t5, -0x4(a4)
1000019c: sw t4, 0x0(a4)
100001a0: bltu s4, a2, 0x100001b0 <main+0x144>
100001a4: srliw t6, t6, 0x10
100001a8: ori t6, t6, 0x40
100001ac: j 0x100001c4 <main+0x158>
100001b0: add s0, t6, s2
100001b4: slli t6, t6, 0x2f
100001b8: srli t6, t6, 0x3f
100001bc: add t6, s0, t6
100001c0: srliw t6, t6, 0x10
100001c4: fmv.w.x fa3, t5
100001c8: fmul.s fa2, fa3, fa5
100001cc: fmv.x.w t5, fa2
100001d0: slli s0, t5, 0x21
100001d4: srli s0, s0, 0x21
100001d8: sh t6, -0x4(a6)
100001dc: bltu s0, a2, 0x100001ec <main+0x180>
100001e0: srliw t5, t5, 0x10
100001e4: ori t5, t5, 0x40
100001e8: j 0x10000200 <main+0x194>
100001ec: add t6, t5, s2
100001f0: slli t5, t5, 0x2f
100001f4: srli t5, t5, 0x3f
100001f8: add t5, t6, t5
100001fc: srliw t5, t5, 0x10
10000200: fmv.w.x fa2, t4
10000204: fmul.s fa1, fa2, fa5
10000208: fmv.x.w t4, fa1
1000020c: slli t6, t4, 0x21
10000210: srli t6, t6, 0x21
10000214: sh t5, -0x2(a6)
10000218: bltu t6, a2, 0x10000228 <main+0x1bc>
1000021c: srliw t4, t4, 0x10
10000220: ori t4, t4, 0x40
10000224: j 0x1000023c <main+0x1d0>
10000228: add t5, t4, s2
1000022c: slli t4, t4, 0x2f
10000230: srli t4, t4, 0x3f
10000234: add t4, t5, t4
10000238: srliw t4, t4, 0x10
1000023c: fmul.s fa3, fa3, fa3
10000240: sh t4, 0x0(a6)
10000244: sh t1, -0x400(a5)
10000248: fmadd.s fa4, fa4, fa4, fa3
1000024c: fmadd.s fa4, fa2, fa2, fa4
10000250: sh t2, -0x200(a5)
10000254: fmv.x.w t4, fa4
10000258: slli t5, t4, 0x21
1000025c: srli t5, t5, 0x21
10000260: sh t3, 0x0(a5)
10000264: bltu t5, a2, 0x1000011c <main+0xb0>
10000268: srliw t4, t4, 0x10
1000026c: ori t4, t4, 0x40
10000270: j 0x10000130 <main+0xc4>
10000274: li a4, 0x0
10000278: li s0, 0x0
1000027c: addi a5, a0, 0x4
10000280: auipc a0, 0x47
10000284: auipc a2, 0x7
10000288: li s10, 0x20
1000028c: li s4, 0x10
10000290: li s7, 0xf0
10000294: lui a3, 0x3f800
10000298: fmv.w.x fs0, zero
1000029c: lui s8, 0x7f800
100002a0: addi a0, a0, 0x5c0
100002a4: addi s11, a2, 0x5bc
100002a8: fmv.w.x fs1, a3
100002ac: sd a0, 0x10(sp)
100002b0: addi a2, a0, 0x4
100002b4: addiw s8, s8, 0x1
100002b8: sd a2, 0x28(sp)
100002bc: sd a5, 0x38(sp)
100002c0: slli a0, a4, 0x6
100002c4: sd a4, 0x40(sp)
100002c8: slli a1, a4, 0x8
100002cc: sub a1, a1, a0
100002d0: ld a0, 0x10(sp)
100002d4: add a1, a1, a0
100002d8: addi s6, a1, 0xc4
100002dc: addi s9, s0, -0x10
100002e0: addi a0, sp, 0x48
100002e4: li a2, 0xc0
100002e8: li a1, 0x0
100002ec: jal 0x10000584 <memset>
100002f0: li a0, 0x0
100002f4: li a6, 0x0
100002f8: slli a1, s0, 0x5
100002fc: ld a2, 0x18(sp)
10000300: add a1, a2, a1
10000304: mx.tload.a td=0 base=a1 stride=s10 fmt=bf16 t=0
10000308: mx.azero ad=2
1000030c: slli a1, s0, 0x8
10000310: sd s0, 0x20(sp)
10000314: mv a2, s0
10000318: li s0, 0x400
1000031c: sd s11, 0x30(sp)
10000320: mv a3, s11
10000324: li s11, 0x200
10000328: j 0x1000035c <main+0x2f0>
1000032c: auipc a5, 0x7
10000330: slli a6, a4, 0x5
10000334: addi a5, a5, -0x124
10000338: add a6, s5, a6
1000033c: mx.tload.a td=2 base=a5 stride=s10 fmt=bf16 t=0
10000340: mx.tload.b td=3 base=a6 stride=s10 fmt=bf16 t=0
10000344: mx.mma ad=2 ta=2 tb=3 in=bf16 acc=fp32
10000348: addi a6, a4, 0x10
1000034c: addi a3, a3, 0x40
10000350: addi a2, a2, -0x10
10000354: addi a0, a0, 0x1
10000358: bgeu a4, s7, 0x10000448 <main+0x3dc>
1000035c: li a5, 0x0
10000360: mv a4, a6
10000364: slli a6, a0, 0x4
10000368: mx.azero ad=0
1000036c: slli a7, a4, 0x1
10000370: add t0, a4, a1
10000374: auipc t1, 0x7
10000378: add a7, s3, a7
1000037c: slli t0, t0, 0x2
10000380: addi t1, t1, 0x4cc
10000384: mx.tload.b td=1 base=a7 stride=s11 fmt=bf16 t=0
10000388: add t0, t1, t0
1000038c: mx.mma ad=0 ta=0 tb=1 in=bf16 acc=fp32
10000390: mx.astore as=0 base=t0 stride=s0 fmt=fp32
10000394: flw fa5, 0x40(s1)
10000398: flw fa4, 0x50(s1)
1000039c: sub a6, s9, a6
100003a0: auipc a7, 0x7
100003a4: addi a7, a7, -0x198
100003a8: mv t0, a2
100003ac: mv t1, a3
100003b0: j 0x100003c8 <main+0x35c>
100003b4: addi a5, a5, 0x1
100003b8: addi t1, t1, 0x400
100003bc: addi t0, t0, 0x1
100003c0: addi a7, a7, 0x20
100003c4: beq a5, s4, 0x1000032c <main+0x2c0>
100003c8: add t2, a6, a5
100003cc: mv t3, a7
100003d0: mv t4, t0
100003d4: mv t5, t1
100003d8: j 0x100003f8 <main+0x38c>
100003dc: srliw t6, t6, 0x10
100003e0: ori t6, t6, 0x40
100003e4: sh t6, 0x0(t3)
100003e8: addi t5, t5, 0x4
100003ec: addi t4, t4, -0x1
100003f0: addi t3, t3, 0x2
100003f4: beq t4, t2, 0x100003b4 <main+0x348>
100003f8: beqz t4, 0x1000042c <main+0x3c0>
100003fc: flw fa3, 0x0(t5)
10000400: fmax.s fa3, fa3, fs0
10000404: fadd.s fa3, fa5, fa3
10000408: fsqrt.s fa2, fa3
1000040c: fmul.s fa3, fa2, fa3
10000410: fdiv.s fa3, fs1, fa3
10000414: fmul.s fa3, fa4, fa3
10000418: fmv.x.w t6, fa3
1000041c: slli ra, t6, 0x21
10000420: srli ra, ra, 0x21
10000424: bgeu ra, s8, 0x100003dc <main+0x370>
10000428: j 0x10000430 <main+0x3c4>
1000042c: li t6, 0x0
10000430: add ra, t6, s2
10000434: slli t6, t6, 0x2f
10000438: srli t6, t6, 0x3f
1000043c: add t6, ra, t6
10000440: srliw t6, t6, 0x10
10000444: j 0x100003e4 <main+0x378>
10000448: addi a0, a5, 0x200
1000044c: li a1, 0x40
10000450: mx.astore as=2 base=a0 stride=a1 fmt=fp32
10000454: flw fa5, 0x50(s1)
10000458: fdiv.s fa5, fs1, fa5
1000045c: addi a0, a5, 0x20c
10000460: ld a5, 0x38(sp)
10000464: mv a1, a5
10000468: addi a3, sp, 0x50
1000046c: mv a2, a3
10000470: addi a4, sp, 0x110
10000474: flw fa4, -0xc(a0)
10000478: flw fa3, -0x8(a0)
1000047c: flw fa2, -0x4(a0)
10000480: flw fa1, 0x0(a0)
10000484: flw fa0, -0x4(a1)
10000488: flw ft0, 0x0(a1)
1000048c: flw ft1, 0x4(a1)
10000490: fnmsub.s fa4, fa0, fa1, fa4
10000494: fnmsub.s fa3, ft0, fa1, fa3
10000498: fnmsub.s fa2, ft1, fa1, fa2
1000049c: fmul.s fa4, fa5, fa4
100004a0: fmul.s fa3, fa5, fa3
100004a4: fmul.s fa2, fa5, fa2
100004a8: fsw fa4, -0x8(a2)
100004ac: fsw fa3, -0x4(a2)
100004b0: fsw fa2, 0x0(a2)
100004b4: addi a2, a2, 0xc
100004b8: addi a0, a0, 0x40
100004bc: addi a1, a1, 0xc
100004c0: bne a2, a4, 0x10000474 <main+0x408>
100004c4: mv a0, a3
100004c8: ld a2, 0x28(sp)
100004cc: mv a1, a2
100004d0: ld a4, 0x40(sp)
100004d4: flw fa5, -0x8(a0)
100004d8: flw fa4, -0x4(a0)
100004dc: flw fa3, 0x0(a0)
100004e0: fsw fa5, -0x4(a1)
100004e4: fsw fa4, 0x0(a1)
100004e8: fsw fa3, 0x4(a1)
100004ec: addi a1, a1, 0xc
100004f0: addi a0, a0, 0xc
100004f4: bne a1, s6, 0x100004d4 <main+0x468>
100004f8: ld a3, 0x20(sp)
100004fc: addi s0, a3, 0x10
10000500: lui a0, 0x4
10000504: ld s11, 0x30(sp)
10000508: add s11, s11, a0
1000050c: addi a5, a5, 0xc0
10000510: addi a2, a2, 0xc0
10000514: addi a4, a4, 0x1
10000518: bltu a3, s7, 0x100002b8 <main+0x24c>
1000051c: mx.azero ad=0
10000520: mx.azero ad=1
10000524: mx.azero ad=2
10000528: mx.azero ad=3
1000052c: rdcycle a0
10000530: ld a1, 0x8(sp)
10000534: sub a0, a0, a1
10000538: sd a0, 0x0(s1)
1000053c: li a0, 0x0
10000540: ld ra, 0x178(sp)
10000544: ld s0, 0x170(sp)
10000548: ld s1, 0x168(sp)
1000054c: ld s2, 0x160(sp)
10000550: ld s3, 0x158(sp)
10000554: ld s4, 0x150(sp)
10000558: ld s5, 0x148(sp)
1000055c: ld s6, 0x140(sp)
10000560: ld s7, 0x138(sp)
10000564: ld s8, 0x130(sp)
10000568: ld s9, 0x128(sp)
1000056c: ld s10, 0x120(sp)
10000570: ld s11, 0x118(sp)
10000574: fld fs0, 0x110(sp)
10000578: fld fs1, 0x108(sp)
1000057c: addi sp, sp, 0x180
10000580: ret
# <memset>
10000584: beqz a2, 0x100005a0 <memset+0x1c>
10000588: add a2, a0, a2
1000058c: mv a3, a0
10000590: addi a4, a3, 0x1
10000594: sb a1, 0x0(a3)
10000598: mv a3, a4
1000059c: bne a4, a2, 0x10000590 <memset+0xc>
100005a0: ret
