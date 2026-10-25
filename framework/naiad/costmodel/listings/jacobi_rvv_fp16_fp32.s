# jacobi/rvv size=66 fp16/fp32 cflags='' design=naiad-dwarfs.yaml
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
10000094: csrr a1, vlenb
10000098: lui a0, 0x2
1000009c: addiw a0, a0, 0x208
100000a0: li a3, 0x11
100000a4: addi a2, a0, -0x20c
100000a8: slli a3, a3, 0x8
100000ac: srli a7, a1, 0x3
100000b0: srli a4, a1, 0x1
100000b4: lui t2, 0x77800
100000b8: auipc a5, 0x2
100000bc: auipc a6, 0x7
100000c0: lui t1, 0x8
100000c4: mul a2, a7, a2
100000c8: addi a5, a5, 0x688
100000cc: and a2, a2, a3
100000d0: addi a6, a6, -0x53c
100000d4: sub a3, a2, a4
100000d8: addi t1, t1, -0x1
100000dc: fmv.w.x fa5, t2
100000e0: vsetvli t2, zero, e32, m2, ta, ma
100000e4: divu a3, a3, a4
100000e8: auipc a4, 0x0
100000ec: addi a4, a4, 0x418
100000f0: slli t0, a3, 0x3
100000f4: lui a3, 0x80000
100000f8: addi t0, t0, 0x8
100000fc: mul t0, t0, a7
10000100: slli a7, a1, 0x1
10000104: add t0, a4, t0
10000108: vl1re16.v v8, (a4)
1000010c: add t2, a5, a0
10000110: add a4, a4, a1
10000114: vsext.vf2 v10, v8
10000118: vsetvli zero, zero, e16, m1, ta, ma
1000011c: vs1r.v v8, (a5)
10000120: add a5, a5, a1
10000124: vs1r.v v8, (t2)
10000128: vand.vx v9, v8, t1
1000012c: vsetvli zero, zero, e32, m2, ta, ma
10000130: vand.vx v10, v10, a3
10000134: vzext.vf2 v12, v9
10000138: vsll.vi v12, v12, 0xd
1000013c: vfmul.vf v12, v12, fa5
10000140: vor.vv v10, v10, v12
10000144: vs2r.v v10, (a6)
10000148: add a6, a6, a7
1000014c: bne a4, t0, 0x10000108 <main+0x9c>
10000150: lui a1, 0x1
10000154: addiw a1, a1, 0x104
10000158: beq a2, a1, 0x100001e4 <main+0x178>
1000015c: auipc a3, 0x7
10000160: slli a1, a2, 0x2
10000164: slli a4, a2, 0x1
10000168: addi a5, a3, -0x5dc
1000016c: auipc a2, 0x2
10000170: auipc a3, 0x0
10000174: lui a6, 0x77800
10000178: addi a2, a2, 0x5d4
1000017c: addi a3, a3, 0x390
10000180: add a2, a2, a4
10000184: add a3, a3, a4
10000188: lui a4, 0x4
1000018c: fmv.w.x fa5, a6
10000190: addiw a6, a4, 0x410
10000194: add a1, a5, a1
10000198: lui a4, 0x2
1000019c: add a5, a5, a6
100001a0: lhu a6, 0x0(a3)
100001a4: addi a3, a3, 0x2
100001a8: slli a7, a6, 0x31
100001ac: srli t0, a6, 0xf
100001b0: srli a7, a7, 0x24
100001b4: slli t0, t0, 0x1f
100001b8: sh a6, 0x0(a2)
100001bc: fmv.w.x fa4, a7
100001c0: fmul.s fa4, fa4, fa5
100001c4: fmv.x.w a7, fa4
100001c8: or a7, t0, a7
100001cc: add t0, a2, a4
100001d0: addi a2, a2, 0x2
100001d4: sw a7, 0x0(a1)
100001d8: addi a1, a1, 0x4
100001dc: sh a6, 0x208(t0)
100001e0: bne a1, a5, 0x100001a0 <main+0x134>
100001e4: lui a7, 0x3e800
100001e8: auipc a4, 0x2
100001ec: auipc t0, 0x7
100001f0: li a1, 0x0
100001f4: li a2, 0x0
100001f8: li a3, 0x1
100001fc: addi a4, a4, 0x558
10000200: li a5, 0x7
10000204: li a6, 0x41
10000208: addi t0, t0, -0x66c
1000020c: fmv.w.x fa5, a7
10000210: li t1, 0x3f
10000214: li t2, 0x8
10000218: rdcycle a7
1000021c: j 0x10000230 <main+0x1c4>
10000220: addi a3, a3, 0x1
10000224: addi a1, a1, 0x1
10000228: addiw a2, a2, 0x1
1000022c: beq a2, t2, 0x10000404 <main+0x398>
10000230: andi t3, a1, 0x1
10000234: mul t3, t3, a0
10000238: bne a2, a5, 0x1000030c <main+0x2a0>
1000023c: add t3, a4, t3
10000240: li t4, 0x1
10000244: addi t3, t3, 0x106
10000248: slli t6, t4, 0x2
1000024c: sub t5, a6, t4
10000250: slli s1, t4, 0x1
10000254: add s0, t0, t6
10000258: vsetvli t6, t5, e32, m4, ta, ma
1000025c: addi s0, s0, 0x210
10000260: add s1, t3, s1
10000264: li s2, -0x1
10000268: addi s3, s1, -0x106
1000026c: addi s2, s2, 0x2
10000270: vle16.v v12, (s3)
10000274: addi s3, s1, 0x2
10000278: vle16.v v14, (s3)
1000027c: addi s3, s1, -0x82
10000280: vle16.v v16, (s3)
10000284: addi s3, s1, 0x86
10000288: vle16.v v18, (s3)
1000028c: addi s3, s1, -0x84
10000290: vsetvli zero, t5, e16, m2, ta, ma
10000294: vfwadd.vv v8, v12, v14
10000298: vle16.v v20, (s3)
1000029c: addi s3, s1, -0x80
100002a0: vle16.v v22, (s1)
100002a4: vsetvli zero, t5, e16, m2, ta, ma
100002a8: vfwadd.vv v12, v16, v18
100002ac: vle16.v v24, (s3)
100002b0: addi s3, s1, 0x4
100002b4: addi s1, s1, 0x108
100002b8: vle16.v v16, (s3)
100002bc: vsetvli zero, t5, e16, m2, ta, ma
100002c0: vfwadd.wv v8, v8, v20
100002c4: vsetvli zero, t5, e16, m2, ta, ma
100002c8: addi s3, s0, -0x108
100002cc: vfwadd.wv v12, v12, v22
100002d0: vsetvli zero, t5, e16, m2, ta, ma
100002d4: vfwadd.wv v8, v8, v24
100002d8: vsetvli zero, t5, e16, m2, ta, ma
100002dc: vfwadd.wv v12, v12, v16
100002e0: vsetvli zero, t5, e32, m4, ta, ma
100002e4: vfmul.vf v8, v8, fa5
100002e8: vsetvli zero, t5, e32, m4, ta, ma
100002ec: vfmul.vf v12, v12, fa5
100002f0: vse32.v v8, (s3)
100002f4: vse32.v v12, (s0)
100002f8: addi s0, s0, 0x210
100002fc: bltu s2, t1, 0x10000268 <main+0x1fc>
10000300: add t4, t6, t4
10000304: bltu t4, a6, 0x10000248 <main+0x1dc>
10000308: j 0x10000220 <main+0x1b4>
1000030c: andi t4, a3, 0x1
10000310: add t3, a4, t3
10000314: li t6, 0x1
10000318: mul t4, t4, a0
1000031c: add t5, a4, t4
10000320: addi t4, t5, 0x84
10000324: addi t5, t5, 0x108
10000328: sub s0, a6, t6
1000032c: slli s1, t6, 0x1
10000330: li s3, -0x1
10000334: vsetvli s2, s0, e32, m4, ta, ma
10000338: mv s4, t5
1000033c: mv s5, t3
10000340: mv s6, t4
10000344: add s7, s5, s1
10000348: vsetvli zero, zero, e16, m2, ta, ma
1000034c: addi s8, s7, 0x108
10000350: addi s5, s5, 0x108
10000354: addi s3, s3, 0x2
10000358: vle16.v v12, (s7)
1000035c: vle16.v v14, (s8)
10000360: addi s8, s7, 0x84
10000364: vle16.v v16, (s8)
10000368: addi s8, s7, 0x18c
1000036c: vle16.v v18, (s8)
10000370: addi s8, s7, 0x82
10000374: vsetvli zero, s0, e16, m2, ta, ma
10000378: vfwadd.vv v8, v12, v14
1000037c: vle16.v v20, (s8)
10000380: addi s8, s7, 0x106
10000384: vle16.v v22, (s8)
10000388: addi s8, s7, 0x86
1000038c: addi s7, s7, 0x10a
10000390: vsetvli zero, s0, e16, m2, ta, ma
10000394: vfwadd.vv v12, v16, v18
10000398: vle16.v v24, (s8)
1000039c: vle16.v v16, (s7)
100003a0: vsetvli zero, s0, e16, m2, ta, ma
100003a4: vfwadd.wv v8, v8, v20
100003a8: vsetvli zero, s0, e16, m2, ta, ma
100003ac: add s7, s6, s1
100003b0: vfwadd.wv v12, v12, v22
100003b4: vsetvli zero, s0, e16, m2, ta, ma
100003b8: addi s6, s6, 0x108
100003bc: vfwadd.wv v8, v8, v24
100003c0: vsetvli zero, s0, e16, m2, ta, ma
100003c4: vfwadd.wv v12, v12, v16
100003c8: vsetvli zero, s0, e32, m4, ta, ma
100003cc: vfmul.vf v8, v8, fa5
100003d0: vsetvli zero, s0, e32, m4, ta, ma
100003d4: vfmul.vf v12, v12, fa5
100003d8: vsetvli zero, zero, e16, m2, ta, ma
100003dc: vfncvt.f.f.w v16, v8
100003e0: vfncvt.f.f.w v8, v12
100003e4: vse16.v v16, (s7)
100003e8: add s7, s4, s1
100003ec: addi s4, s4, 0x108
100003f0: vse16.v v8, (s7)
100003f4: bltu s3, t1, 0x10000344 <main+0x2d8>
100003f8: add t6, s2, t6
100003fc: bltu t6, a6, 0x10000328 <main+0x2bc>
10000400: j 0x10000220 <main+0x1b4>
10000404: auipc a1, 0xb
10000408: rdcycle a2
1000040c: ld s0, 0x48(sp)
10000410: sub a2, a2, a7
10000414: ld s1, 0x40(sp)
10000418: li a0, 0x0
1000041c: ld s2, 0x38(sp)
10000420: ld s3, 0x30(sp)
10000424: ld s4, 0x28(sp)
10000428: ld s5, 0x20(sp)
1000042c: ld s6, 0x18(sp)
10000430: ld s7, 0x10(sp)
10000434: ld s8, 0x8(sp)
10000438: sd a2, -0x444(a1)
1000043c: addi sp, sp, 0x50
10000440: ret
