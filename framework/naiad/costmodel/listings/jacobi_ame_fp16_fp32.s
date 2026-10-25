# jacobi/ame size=66 fp16/fp32 cflags='' design=naiad-dwarfs.yaml
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
100000a8: li a1, 0x0
100000ac: auipc a2, 0x0
100000b0: li a3, 0xd
100000b4: addi a2, a2, 0x554
100000b8: slli a3, a3, 0xa
100000bc: li a4, 0x1e
100000c0: li a5, 0x10
100000c4: mv a6, a2
100000c8: li a7, 0x0
100000cc: addi t3, a1, -0xf
100000d0: snez t0, a1
100000d4: slli t1, a1, 0x5
100000d8: addi t2, a1, -0x1
100000dc: addi t0, t0, -0x1
100000e0: add t4, a2, t1
100000e4: and t0, t0, a3
100000e8: sh t0, 0x0(t4)
100000ec: addi t0, a0, -0x4
100000f0: snez t1, t2
100000f4: addi t1, t1, -0x1
100000f8: and t1, t1, a3
100000fc: sh t1, 0x200(t4)
10000100: snez t1, t3
10000104: addi t1, t1, -0x1
10000108: and t1, t1, a3
1000010c: sh t1, 0x400(t4)
10000110: addi t1, a0, -0x2
10000114: sh zero, 0x600(t4)
10000118: addi t2, t4, 0x7ff
1000011c: sh zero, 0x1(t2)
10000120: seqz t2, a1
10000124: seqz t3, t3
10000128: xor t4, t1, a7
1000012c: xor t5, t0, a7
10000130: xor t6, a0, a7
10000134: seqz t5, t5
10000138: seqz t6, t6
1000013c: or t5, t5, t6
10000140: add t6, a6, a7
10000144: snez t4, t4
10000148: addi t4, t4, -0x1
1000014c: and t4, t4, a3
10000150: sh t4, 0x2(t6)
10000154: addi t4, a7, -0x1c
10000158: seqz t4, t4
1000015c: negw t5, t5
10000160: and t5, t5, a3
10000164: sh t5, 0x202(t6)
10000168: and t5, t2, t4
1000016c: negw t5, t5
10000170: and t5, t5, a3
10000174: sh t5, 0x602(t6)
10000178: addi t5, a7, -0x1a
1000017c: sh zero, 0x402(t6)
10000180: seqz t5, t5
10000184: addi t6, t6, 0x7ff
10000188: and t5, t2, t5
1000018c: and t4, t3, t4
10000190: or t4, t5, t4
10000194: negw t4, t4
10000198: and t4, t4, a3
1000019c: addi a7, a7, 0x2
100001a0: sh t4, 0x3(t6)
100001a4: bne a7, a4, 0x10000128 <main+0xbc>
100001a8: addi a1, a1, 0x1
100001ac: addi a0, a0, 0x2
100001b0: addi a6, a6, 0x20
100001b4: bne a1, a5, 0x100000c8 <main+0x5c>
100001b8: li a0, 0x0
100001bc: auipc a2, 0x10
100001c0: auipc a3, 0x3
100001c4: auipc a4, 0x1
100001c8: lui a1, 0x2
100001cc: lui a5, 0x77800
100001d0: addi a2, a2, -0x7bc
100001d4: addi a3, a3, 0x80
100001d8: addi a4, a4, -0x1c4
100001dc: fmv.w.x fa5, a5
100001e0: addiw a5, a1, 0x208
100001e4: add a6, a4, a0
100001e8: lhu a6, 0x0(a6)
100001ec: add a7, a3, a1
100001f0: addi a0, a0, 0x2
100001f4: sh a6, 0x208(a7)
100001f8: sh a6, 0x0(a3)
100001fc: srli a7, a6, 0xf
10000200: slli a6, a6, 0x31
10000204: slli a7, a7, 0x1f
10000208: srli a6, a6, 0x24
1000020c: fmv.w.x fa4, a6
10000210: fmul.s fa4, fa4, fa5
10000214: fmv.x.w a6, fa4
10000218: or a6, a7, a6
1000021c: sw a6, 0x0(a2)
10000220: addi a2, a2, 0x4
10000224: addi a3, a3, 0x2
10000228: bne a0, a5, 0x100001e4 <main+0x178>
1000022c: li a0, 0x0
10000230: li s11, 0x0
10000234: lui a1, 0x2
10000238: li a2, 0x1
1000023c: rdcycle a3
10000240: sd a3, 0x8(sp)
10000244: auipc t1, 0x3
10000248: lui t3, 0x4
1000024c: auipc t6, 0x0
10000250: auipc s0, 0xf
10000254: li a4, 0x84
10000258: lui a5, 0x1
1000025c: li a6, 0xe
10000260: li a7, 0x108
10000264: li t0, 0x4
10000268: li s3, 0x1f
1000026c: lui s4, 0x477ff
10000270: li t2, 0x41
10000274: li t4, 0x70
10000278: lui s1, 0x3f000
1000027c: lui s10, 0x8001
10000280: addiw t5, a1, 0x208
10000284: addi s5, t1, -0x4
10000288: addiw a1, t3, 0x410
1000028c: sd a1, 0x40(sp)
10000290: addi a1, t6, 0x3b4
10000294: addi a3, s0, 0x7b0
10000298: addiw s2, a5, -0x73c
1000029c: slli s3, s3, 0xa
100002a0: addiw s4, s4, -0x1
100002a4: fmv.w.x fa5, s1
100002a8: addi t1, a1, 0x200
100002ac: sd t1, 0x38(sp)
100002b0: addi t1, a1, 0x7ff
100002b4: addi t3, a1, 0x400
100002b8: sd t3, 0x30(sp)
100002bc: addi a1, a1, 0x600
100002c0: sd a1, 0x28(sp)
100002c4: sd a3, 0x10(sp)
100002c8: addi a1, a3, 0xcc
100002cc: sd a1, 0x20(sp)
100002d0: addi t1, t1, 0x1
100002d4: sd t1, 0x18(sp)
100002d8: addi s10, s10, -0x1
100002dc: j 0x100002f4 <main+0x288>
100002e0: addi a0, a0, 0x1
100002e4: addi a2, a2, 0x1
100002e8: mv s11, a1
100002ec: li a3, 0x8
100002f0: beq a1, a3, 0x10000504 <main+0x498>
100002f4: li t1, 0x0
100002f8: andi a1, a2, 0x1
100002fc: andi t3, a0, 0x1
10000300: slli t6, s11, 0x3f
10000304: li a3, 0x20
10000308: ld s0, 0x38(sp)
1000030c: mx.tload.b td=0 base=s0 stride=a3 fmt=fp16 t=0
10000310: mul a1, a1, t5
10000314: mul t3, t3, t5
10000318: ld s0, 0x18(sp)
1000031c: mx.tload.a td=1 base=s0 stride=a3 fmt=fp16 t=0
10000320: addi ra, a1, 0x106
10000324: add a1, s5, a1
10000328: add s0, s5, t3
1000032c: ld t3, 0x30(sp)
10000330: mx.tload.b td=2 base=t3 stride=a3 fmt=fp16 t=0
10000334: addi t3, a1, 0x86
10000338: addi s7, s0, 0xe6
1000033c: ld a1, 0x28(sp)
10000340: mx.tload.b td=3 base=a1 stride=a3 fmt=fp16 t=0
10000344: ld a1, 0x40(sp)
10000348: mulhu t6, t6, a1
1000034c: add t6, s5, t6
10000350: addiw a1, s11, 0x1
10000354: ld s6, 0x20(sp)
10000358: j 0x10000378 <main+0x30c>
1000035c: mx.astore as=0 base=s1 stride=a7 fmt=fp32
10000360: addi t1, t1, 0x1
10000364: addi a3, s7, 0x7ff
10000368: addi s7, a3, 0x41
1000036c: addiw a3, a5, 0x80
10000370: add s6, s6, a3
10000374: beq t1, t0, 0x10000460 <main+0x3f4>
10000378: li s1, 0x0
1000037c: slli s0, t1, 0x4
10000380: addi s0, s0, 0x1
10000384: mul s0, s0, a4
10000388: add s0, t6, s0
1000038c: addi s8, s0, 0x2
10000390: addi s0, s0, -0x1e
10000394: mx.tload.a td=4 base=s0 stride=a4 fmt=fp16 t=0
10000398: mx.tload.a td=5 base=s8 stride=a4 fmt=fp16 t=0
1000039c: li s9, 0x1
100003a0: mv s8, s6
100003a4: mv s0, s7
100003a8: addi a3, s0, -0x40
100003ac: mx.tload.a td=6 base=a3 stride=a4 fmt=fp16 t=0
100003b0: addi a3, s0, -0xe4
100003b4: mx.tload.b td=7 base=a3 stride=s2 fmt=fp16 t=0
100003b8: andi a3, s9, 0x1
100003bc: bnez a3, 0x100003c4 <main+0x358>
100003c0: mx.astore as=2 base=s8 stride=a7 fmt=fp32
100003c4: mx.azero ad=0
100003c8: mx.mma ad=0 ta=5 tb=0 in=fp16 acc=fp32
100003cc: mx.mma ad=0 ta=4 tb=2 in=fp16 acc=fp32
100003d0: mx.mma ad=0 ta=0 tb=5 in=fp16 acc=fp32
100003d4: mx.tshift td=7 ts=7 amount=a6 fmt=fp16 cols=0 b=1
100003d8: mx.mma ad=0 ta=1 tb=7 in=fp16 acc=fp32
100003dc: mx.mma ad=0 ta=6 tb=3 in=fp16 acc=fp32
100003e0: andi a3, s1, 0x1
100003e4: addi s1, s8, 0x40
100003e8: bnez a3, 0x1000035c <main+0x2f0>
100003ec: li s9, 0x0
100003f0: addi a3, s0, -0x20
100003f4: mx.tload.a td=4 base=a3 stride=a4 fmt=fp16 t=0
100003f8: addi a3, s0, -0xc4
100003fc: mx.tload.b td=7 base=a3 stride=s2 fmt=fp16 t=0
10000400: mx.astore as=0 base=s1 stride=a7 fmt=fp32
10000404: addi a3, s0, -0xa4
10000408: mx.azero ad=1
1000040c: mx.mma ad=1 ta=6 tb=0 in=fp16 acc=fp32
10000410: mx.mma ad=1 ta=5 tb=2 in=fp16 acc=fp32
10000414: mx.mma ad=1 ta=0 tb=6 in=fp16 acc=fp32
10000418: mx.tshift td=7 ts=7 amount=a6 fmt=fp16 cols=0 b=1
1000041c: mx.mma ad=1 ta=1 tb=7 in=fp16 acc=fp32
10000420: mx.mma ad=1 ta=4 tb=3 in=fp16 acc=fp32
10000424: mx.tload.a td=5 base=s0 stride=a4 fmt=fp16 t=0
10000428: mx.tload.b td=7 base=a3 stride=s2 fmt=fp16 t=0
1000042c: addi a3, s8, 0x80
10000430: mx.astore as=1 base=a3 stride=a7 fmt=fp32
10000434: mx.azero ad=2
10000438: mx.mma ad=2 ta=4 tb=0 in=fp16 acc=fp32
1000043c: mx.mma ad=2 ta=6 tb=2 in=fp16 acc=fp32
10000440: mx.mma ad=2 ta=0 tb=4 in=fp16 acc=fp32
10000444: mx.tshift td=7 ts=7 amount=a6 fmt=fp16 cols=0 b=1
10000448: mx.mma ad=2 ta=1 tb=7 in=fp16 acc=fp32
1000044c: mx.mma ad=2 ta=5 tb=3 in=fp16 acc=fp32
10000450: addi s0, s0, 0x60
10000454: addi s8, s8, 0xc0
10000458: li s1, 0x1
1000045c: j 0x100003a8 <main+0x33c>
10000460: li a3, 0x7
10000464: beq s11, a3, 0x10000504 <main+0x498>
10000468: li t1, 0x0
1000046c: ld a3, 0x10(sp)
10000470: addi t6, a3, 0x10c
10000474: li s0, 0x1
10000478: j 0x10000490 <main+0x424>
1000047c: addi s0, s0, 0x1
10000480: addi t3, t3, 0x84
10000484: addi t6, t6, 0x108
10000488: addi t1, t1, 0x1
1000048c: beq s0, t2, 0x100002e0 <main+0x274>
10000490: mul a3, t1, a4
10000494: add a3, ra, a3
10000498: add s1, s5, a3
1000049c: mv s6, t6
100004a0: mv s7, t3
100004a4: j 0x100004d8 <main+0x46c>
100004a8: slli a3, s8, 0x32
100004ac: srli a3, a3, 0x3f
100004b0: add s9, s8, s10
100004b4: add a3, s9, a3
100004b8: srliw s9, a3, 0xd
100004bc: srliw a3, s8, 0x1f
100004c0: slli a3, a3, 0xf
100004c4: or a3, s9, a3
100004c8: sh a3, 0x0(s7)
100004cc: addi s7, s7, 0x2
100004d0: addi s6, s6, 0x4
100004d4: beq s7, s1, 0x1000047c <main+0x410>
100004d8: flw fa3, 0x0(s6)
100004dc: fabs.s fa4, fa3
100004e0: fmv.x.w s11, fa4
100004e4: fmv.x.w s8, fa3
100004e8: mv s9, s3
100004ec: bltu s4, s11, 0x100004bc <main+0x450>
100004f0: srliw a3, s11, 0x17
100004f4: bltu t4, a3, 0x100004a8 <main+0x43c>
100004f8: fadd.s fa4, fa4, fa5
100004fc: fmv.x.w s9, fa4
10000500: j 0x100004bc <main+0x450>
10000504: mx.azero ad=0
10000508: mx.azero ad=1
1000050c: mx.azero ad=2
10000510: mx.azero ad=3
10000514: rdcycle a0
10000518: ld a1, 0x8(sp)
1000051c: sub a0, a0, a1
10000520: auipc a1, 0x14
10000524: sd a0, -0x6e0(a1)
10000528: li a0, 0x0
1000052c: ld ra, 0xa8(sp)
10000530: ld s0, 0xa0(sp)
10000534: ld s1, 0x98(sp)
10000538: ld s2, 0x90(sp)
1000053c: ld s3, 0x88(sp)
10000540: ld s4, 0x80(sp)
10000544: ld s5, 0x78(sp)
10000548: ld s6, 0x70(sp)
1000054c: ld s7, 0x68(sp)
10000550: ld s8, 0x60(sp)
10000554: ld s9, 0x58(sp)
10000558: ld s10, 0x50(sp)
1000055c: ld s11, 0x48(sp)
10000560: addi sp, sp, 0xb0
10000564: ret
