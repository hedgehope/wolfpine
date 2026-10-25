# jacobi/rvv size=66 bf16/fp32 cflags='' design=naiad-dwarfs.yaml
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
100000ac: srli a4, a1, 0x3
100000b0: srli a5, a1, 0x1
100000b4: auipc a6, 0x2
100000b8: auipc a7, 0x7
100000bc: addi a6, a6, 0x64c
100000c0: addi a7, a7, -0x578
100000c4: mul a2, a4, a2
100000c8: vsetvli t0, zero, e32, m2, ta, ma
100000cc: and a2, a2, a3
100000d0: sub a3, a2, a5
100000d4: divu a3, a3, a5
100000d8: slli a3, a3, 0x3
100000dc: addi a5, a3, 0x8
100000e0: auipc a3, 0x0
100000e4: addi a3, a3, 0x3e0
100000e8: mul a5, a5, a4
100000ec: slli a4, a1, 0x1
100000f0: add a5, a3, a5
100000f4: vl1re16.v v8, (a3)
100000f8: add t0, a6, a0
100000fc: add a3, a3, a1
10000100: vzext.vf2 v10, v8
10000104: vs1r.v v8, (a6)
10000108: add a6, a6, a1
1000010c: vs1r.v v8, (t0)
10000110: vsll.vi v10, v10, 0x10
10000114: vs2r.v v10, (a7)
10000118: add a7, a7, a4
1000011c: bne a3, a5, 0x100000f4 <main+0x88>
10000120: lui a1, 0x1
10000124: addiw a1, a1, 0x104
10000128: beq a2, a1, 0x10000190 <main+0x124>
1000012c: auipc a3, 0x7
10000130: slli a1, a2, 0x2
10000134: slli a4, a2, 0x1
10000138: addi a5, a3, -0x5ec
1000013c: auipc a2, 0x2
10000140: auipc a3, 0x0
10000144: lui a6, 0x4
10000148: addi a2, a2, 0x5c4
1000014c: addi a3, a3, 0x380
10000150: addiw a6, a6, 0x410
10000154: add a1, a5, a1
10000158: add a2, a2, a4
1000015c: add a3, a3, a4
10000160: lui a4, 0x2
10000164: add a5, a5, a6
10000168: lh a6, 0x0(a3)
1000016c: add a7, a2, a4
10000170: slli t0, a6, 0x10
10000174: sh a6, 0x0(a2)
10000178: addi a2, a2, 0x2
1000017c: addi a3, a3, 0x2
10000180: sw t0, 0x0(a1)
10000184: addi a1, a1, 0x4
10000188: sh a6, 0x208(a7)
1000018c: bne a1, a5, 0x10000168 <main+0xfc>
10000190: lui a7, 0x3e800
10000194: auipc t3, 0x0
10000198: auipc a4, 0x2
1000019c: auipc t0, 0x7
100001a0: li a1, 0x0
100001a4: li a2, 0x0
100001a8: li a3, 0x1
100001ac: addi a4, a4, 0x568
100001b0: li a5, 0x7
100001b4: li a6, 0x41
100001b8: addi t0, t0, -0x65c
100001bc: fmv.w.x fa5, a7
100001c0: li t1, 0x3f
100001c4: li t2, 0x8
100001c8: flh fa4, 0x27c(t3)
100001cc: rdcycle a7
100001d0: j 0x100001e4 <main+0x178>
100001d4: addi a3, a3, 0x1
100001d8: addi a1, a1, 0x1
100001dc: addiw a2, a2, 0x1
100001e0: beq a2, t2, 0x100003d0 <main+0x364>
100001e4: andi t3, a1, 0x1
100001e8: mul t3, t3, a0
100001ec: bne a2, a5, 0x100002cc <main+0x260>
100001f0: add t3, a4, t3
100001f4: li t4, 0x1
100001f8: addi t3, t3, 0x106
100001fc: slli t6, t4, 0x2
10000200: sub t5, a6, t4
10000204: slli s1, t4, 0x1
10000208: add s0, t0, t6
1000020c: vsetvli t6, t5, e32, m4, ta, ma
10000210: addi s0, s0, 0x210
10000214: add s1, t3, s1
10000218: li s2, -0x1
1000021c: addi s3, s1, -0x106
10000220: addi s2, s2, 0x2
10000224: vle16.v v12, (s3)
10000228: addi s3, s1, 0x2
1000022c: vle16.v v18, (s3)
10000230: addi s3, s1, -0x82
10000234: vle16.v v20, (s3)
10000238: addi s3, s1, 0x86
1000023c: vle16.v v22, (s3)
10000240: addi s3, s1, -0x84
10000244: vsetvli zero, t5, e16, m2, ta, ma
10000248: vfwcvtbf16.f.f.v v8, v12
1000024c: vle16.v v16, (s3)
10000250: addi s3, s1, -0x80
10000254: vle16.v v24, (s1)
10000258: vsetvli zero, t5, e16, m2, ta, ma
1000025c: vfwcvtbf16.f.f.v v12, v20
10000260: vfwmaccbf16.vf v8, fa4, v18
10000264: vsetvli zero, t5, e16, m2, ta, ma
10000268: vle16.v v18, (s3)
1000026c: addi s3, s1, 0x4
10000270: addi s1, s1, 0x108
10000274: vfwmaccbf16.vf v12, fa4, v22
10000278: vle16.v v20, (s3)
1000027c: vsetvli zero, t5, e16, m2, ta, ma
10000280: vfwmaccbf16.vf v8, fa4, v16
10000284: vsetvli zero, t5, e16, m2, ta, ma
10000288: addi s3, s0, -0x108
1000028c: vfwmaccbf16.vf v12, fa4, v24
10000290: vsetvli zero, t5, e16, m2, ta, ma
10000294: vfwmaccbf16.vf v8, fa4, v18
10000298: vsetvli zero, t5, e16, m2, ta, ma
1000029c: vfwmaccbf16.vf v12, fa4, v20
100002a0: vsetvli zero, t5, e32, m4, ta, ma
100002a4: vfmul.vf v8, v8, fa5
100002a8: vsetvli zero, t5, e32, m4, ta, ma
100002ac: vfmul.vf v12, v12, fa5
100002b0: vse32.v v8, (s3)
100002b4: vse32.v v12, (s0)
100002b8: addi s0, s0, 0x210
100002bc: bltu s2, t1, 0x1000021c <main+0x1b0>
100002c0: add t4, t6, t4
100002c4: bltu t4, a6, 0x100001fc <main+0x190>
100002c8: j 0x100001d4 <main+0x168>
100002cc: andi t4, a3, 0x1
100002d0: add t3, a4, t3
100002d4: li t6, 0x1
100002d8: mul t4, t4, a0
100002dc: add t5, a4, t4
100002e0: addi t4, t5, 0x84
100002e4: addi t5, t5, 0x108
100002e8: sub s0, a6, t6
100002ec: slli s1, t6, 0x1
100002f0: li s3, -0x1
100002f4: vsetvli s2, s0, e32, m4, ta, ma
100002f8: mv s4, t5
100002fc: mv s5, t3
10000300: mv s6, t4
10000304: add s7, s5, s1
10000308: vsetvli zero, zero, e16, m2, ta, ma
1000030c: addi s8, s7, 0x108
10000310: addi s5, s5, 0x108
10000314: addi s3, s3, 0x2
10000318: vle16.v v12, (s7)
1000031c: vle16.v v18, (s8)
10000320: addi s8, s7, 0x84
10000324: vle16.v v20, (s8)
10000328: addi s8, s7, 0x18c
1000032c: vle16.v v22, (s8)
10000330: addi s8, s7, 0x82
10000334: vsetvli zero, s0, e16, m2, ta, ma
10000338: vfwcvtbf16.f.f.v v8, v12
1000033c: vle16.v v16, (s8)
10000340: addi s8, s7, 0x106
10000344: vle16.v v24, (s8)
10000348: vsetvli zero, s0, e16, m2, ta, ma
1000034c: vfwcvtbf16.f.f.v v12, v20
10000350: addi s8, s7, 0x86
10000354: addi s7, s7, 0x10a
10000358: vfwmaccbf16.vf v8, fa4, v18
1000035c: vsetvli zero, s0, e16, m2, ta, ma
10000360: vle16.v v18, (s8)
10000364: vfwmaccbf16.vf v12, fa4, v22
10000368: vle16.v v20, (s7)
1000036c: vsetvli zero, s0, e16, m2, ta, ma
10000370: vfwmaccbf16.vf v8, fa4, v16
10000374: vsetvli zero, s0, e16, m2, ta, ma
10000378: add s7, s6, s1
1000037c: addi s6, s6, 0x108
10000380: vfwmaccbf16.vf v12, fa4, v24
10000384: vsetvli zero, s0, e16, m2, ta, ma
10000388: vfwmaccbf16.vf v8, fa4, v18
1000038c: vsetvli zero, s0, e16, m2, ta, ma
10000390: vfwmaccbf16.vf v12, fa4, v20
10000394: vsetvli zero, s0, e32, m4, ta, ma
10000398: vfmul.vf v8, v8, fa5
1000039c: vsetvli zero, s0, e32, m4, ta, ma
100003a0: vfmul.vf v12, v12, fa5
100003a4: vsetvli zero, zero, e16, m2, ta, ma
100003a8: vfncvtbf16.f.f.w v16, v8
100003ac: vfncvtbf16.f.f.w v8, v12
100003b0: vse16.v v16, (s7)
100003b4: add s7, s4, s1
100003b8: addi s4, s4, 0x108
100003bc: vse16.v v8, (s7)
100003c0: bltu s3, t1, 0x10000304 <main+0x298>
100003c4: add t6, s2, t6
100003c8: bltu t6, a6, 0x100002e8 <main+0x27c>
100003cc: j 0x100001d4 <main+0x168>
100003d0: auipc a1, 0xb
100003d4: rdcycle a2
100003d8: ld s0, 0x48(sp)
100003dc: sub a2, a2, a7
100003e0: ld s1, 0x40(sp)
100003e4: li a0, 0x0
100003e8: ld s2, 0x38(sp)
100003ec: ld s3, 0x30(sp)
100003f0: ld s4, 0x28(sp)
100003f4: ld s5, 0x20(sp)
100003f8: ld s6, 0x18(sp)
100003fc: ld s7, 0x10(sp)
10000400: ld s8, 0x8(sp)
10000404: sd a2, -0x450(a1)
10000408: addi sp, sp, 0x50
1000040c: ret
