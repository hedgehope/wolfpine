# scan/rvv size=16384 fp16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x20
10000070: sd s0, 0x18(sp)
10000074: sd s1, 0x10(sp)
10000078: sd s2, 0x8(sp)
1000007c: sd s3, 0x0(sp)
10000080: vsetvli a3, zero, e32, m2, ta, ma
10000084: auipc a0, 0x1
10000088: vid.v v10
1000008c: auipc a1, 0x9
10000090: auipc a4, 0x0
10000094: addi a0, a0, 0x3c
10000098: addi a2, a1, 0x34
1000009c: addi a6, a4, 0x530
100000a0: li a4, 0x2
100000a4: li a5, 0x0
100000a8: rdcycle a1
100000ac: bltu a3, a4, 0x100002ac <main+0x240>
100000b0: li t0, 0x1
100000b4: mv a7, a6
100000b8: vsetvli t1, zero, e32, m2, ta, ma
100000bc: vmsltu.vx v8, v10, t0
100000c0: mv t0, a5
100000c4: addi a5, a5, 0x1
100000c8: sll t0, a4, t0
100000cc: vmnot.m v8, v8
100000d0: vsm.v v8, (a7)
100000d4: addi a7, a7, 0x100
100000d8: blt t0, a3, 0x100000bc <main+0x50>
100000dc: li a7, 0x10
100000e0: bne a3, a7, 0x100002a4 <main+0x238>
100000e4: addi a3, a0, 0x20
100000e8: vsetivli zero, 0x10, e16, m1, ta, ma
100000ec: vle16.v v8, (a0)
100000f0: addi a4, a0, 0x40
100000f4: fmv.w.x fa5, zero
100000f8: vle16.v v9, (a3)
100000fc: addi a3, a0, 0x60
10000100: vle16.v v10, (a4)
10000104: vfwcvt.f.f.v v18, v8
10000108: vle16.v v8, (a3)
1000010c: auipc a3, 0x0
10000110: addi a3, a3, 0x4b4
10000114: vfwcvt.f.f.v v14, v9
10000118: addi a4, a3, 0x100
1000011c: addi a5, a3, 0x200
10000120: vfwcvt.f.f.v v16, v10
10000124: vfwcvt.f.f.v v12, v8
10000128: vlm.v v8, (a3)
1000012c: addi a3, a3, 0x300
10000130: vlm.v v9, (a4)
10000134: lui a4, 0x4
10000138: vlm.v v10, (a5)
1000013c: addiw a4, a4, -0x7f
10000140: vlm.v v11, (a3)
10000144: li a3, 0x40
10000148: vsetivli zero, 0x10, e32, m2, ta, mu
1000014c: vslideup.vi v20, v18, 0x1
10000150: addi a5, a3, -0x40
10000154: mv a6, a5
10000158: vmv1r.v v0, v8
1000015c: bgeu a5, a4, 0x10000164 <main+0xf8>
10000160: mv a6, a3
10000164: vslideup.vi v22, v14, 0x1
10000168: slli a6, a6, 0x1
1000016c: add a6, a0, a6
10000170: addi a3, a3, 0x40
10000174: addi a7, a6, 0x20
10000178: vfadd.vv v18, v18, v20, v0.t
1000017c: vslideup.vi v24, v16, 0x1
10000180: vslideup.vi v26, v12, 0x1
10000184: vfadd.vv v14, v14, v22, v0.t
10000188: vfadd.vv v16, v16, v24, v0.t
1000018c: vfadd.vv v12, v12, v26, v0.t
10000190: vslideup.vi v20, v18, 0x2
10000194: vmv1r.v v0, v9
10000198: vslideup.vi v22, v14, 0x2
1000019c: vfadd.vv v18, v18, v20, v0.t
100001a0: vslideup.vi v24, v16, 0x2
100001a4: vslideup.vi v26, v12, 0x2
100001a8: vfadd.vv v14, v14, v22, v0.t
100001ac: vfadd.vv v16, v16, v24, v0.t
100001b0: vfadd.vv v12, v12, v26, v0.t
100001b4: vslideup.vi v20, v18, 0x4
100001b8: vmv1r.v v0, v10
100001bc: vslideup.vi v22, v14, 0x4
100001c0: vfadd.vv v18, v18, v20, v0.t
100001c4: vslideup.vi v24, v16, 0x4
100001c8: vslideup.vi v26, v12, 0x4
100001cc: vfadd.vv v14, v14, v22, v0.t
100001d0: vfadd.vv v16, v16, v24, v0.t
100001d4: vfadd.vv v12, v12, v26, v0.t
100001d8: vslideup.vi v20, v18, 0x8
100001dc: vmv1r.v v0, v11
100001e0: vslideup.vi v22, v14, 0x8
100001e4: vfadd.vv v18, v18, v20, v0.t
100001e8: vfadd.vv v14, v14, v22, v0.t
100001ec: vslideup.vi v26, v12, 0x8
100001f0: vslidedown.vi v20, v18, 0xf
100001f4: vslideup.vi v24, v16, 0x8
100001f8: vfadd.vv v12, v12, v26, v0.t
100001fc: vfmv.f.s fa4, v20
10000200: vfadd.vv v16, v16, v24, v0.t
10000204: fadd.s fa4, fa5, fa4
10000208: vslidedown.vi v22, v14, 0xf
1000020c: vfmv.f.s fa3, v22
10000210: vslidedown.vi v20, v12, 0xf
10000214: vle16.v v22, (a6)
10000218: vslidedown.vi v24, v16, 0xf
1000021c: vle16.v v21, (a7)
10000220: addi a7, a6, 0x40
10000224: addi a6, a6, 0x60
10000228: vfadd.vf v18, v18, fa5
1000022c: fadd.s fa5, fa4, fa3
10000230: vle16.v v23, (a7)
10000234: vfadd.vf v14, v14, fa4
10000238: vfmv.f.s fa4, v24
1000023c: vfadd.vf v16, v16, fa5
10000240: vsetvli zero, zero, e16, m1, ta, ma
10000244: vse32.v v18, (a2)
10000248: fadd.s fa5, fa5, fa4
1000024c: vfwcvt.f.f.v v24, v22
10000250: vle16.v v22, (a6)
10000254: addi a6, a2, 0x40
10000258: vfwcvt.f.f.v v26, v21
1000025c: vfwcvt.f.f.v v28, v23
10000260: vse32.v v14, (a6)
10000264: addi a6, a2, 0x80
10000268: vfwcvt.f.f.v v30, v22
1000026c: vsetvli zero, zero, e32, m2, ta, ma
10000270: vfmv.f.s fa4, v20
10000274: vfadd.vf v20, v12, fa5
10000278: vse32.v v16, (a6)
1000027c: addi a6, a2, 0xc0
10000280: addi a2, a2, 0x100
10000284: fadd.s fa5, fa5, fa4
10000288: vmv2r.v v18, v24
1000028c: vmv2r.v v14, v26
10000290: vse32.v v20, (a6)
10000294: vmv2r.v v16, v28
10000298: vmv2r.v v12, v30
1000029c: bltu a5, a4, 0x10000148 <main+0xdc>
100002a0: j 0x10000488 <main+0x41c>
100002a4: lui a7, 0x1
100002a8: bltu a7, a3, 0x100004b4 <main+0x448>
100002ac: slli a7, a3, 0x2
100002b0: slli t1, a3, 0x1
100002b4: add t3, a0, a7
100002b8: add t2, t1, a3
100002bc: slli t5, t2, 0x1
100002c0: vsetvli t6, zero, e16, m1, ta, ma
100002c4: add t4, a0, t1
100002c8: add t5, a0, t5
100002cc: vle16.v v8, (t3)
100002d0: li t0, 0x0
100002d4: slli t3, a3, 0x3
100002d8: fmv.w.x fa5, zero
100002dc: vle16.v v9, (t4)
100002e0: addi t4, a3, -0x1
100002e4: addi t6, a6, 0x100
100002e8: li s0, 0x1
100002ec: vle16.v v18, (t5)
100002f0: lui t5, 0x4
100002f4: vfwcvt.f.f.v v14, v8
100002f8: addiw t5, t5, 0x1
100002fc: vle16.v v8, (a0)
10000300: mv s1, a7
10000304: vfwcvt.f.f.v v16, v9
10000308: vfwcvt.f.f.v v12, v18
1000030c: vfwcvt.f.f.v v18, v8
10000310: vlm.v v8, (a6)
10000314: j 0x100003c4 <main+0x358>
10000318: vslidedown.vx v24, v18, t4
1000031c: slli s1, t0, 0x2
10000320: add s1, a2, s1
10000324: add s2, t0, a3
10000328: slli s2, s2, 0x2
1000032c: add s2, a2, s2
10000330: vfmv.f.s fa4, v24
10000334: vslidedown.vx v26, v16, t4
10000338: fadd.s fa4, fa5, fa4
1000033c: vfmv.f.s fa3, v26
10000340: vslidedown.vx v28, v14, t4
10000344: vfadd.vf v18, v18, fa5
10000348: fadd.s fa5, fa4, fa3
1000034c: vfmv.f.s fa3, v28
10000350: vfadd.vf v16, v16, fa4
10000354: fadd.s fa4, fa5, fa3
10000358: vslidedown.vx v24, v12, t4
1000035c: vfadd.vf v14, v14, fa5
10000360: vfadd.vf v26, v12, fa4
10000364: vsetvli zero, zero, e16, m1, ta, ma
10000368: vse32.v v18, (s1)
1000036c: add s1, t0, t1
10000370: slli s1, s1, 0x2
10000374: add t0, t0, t2
10000378: add s1, a2, s1
1000037c: slli t0, t0, 0x2
10000380: vfwcvt.f.f.v v28, v20
10000384: vse32.v v16, (s2)
10000388: add s2, a2, t0
1000038c: vfwcvt.f.f.v v30, v22
10000390: mv t0, a6
10000394: vfwcvt.f.f.v v18, v9
10000398: vfwcvt.f.f.v v12, v21
1000039c: vsetvli zero, zero, e32, m2, ta, ma
100003a0: vse32.v v14, (s1)
100003a4: add s1, a6, a7
100003a8: vfmv.f.s fa5, v24
100003ac: vmv2r.v v14, v30
100003b0: vse32.v v26, (s2)
100003b4: fadd.s fa5, fa4, fa5
100003b8: vmv2r.v v16, v18
100003bc: vmv2r.v v18, v28
100003c0: bgeu s1, t5, 0x10000480 <main+0x414>
100003c4: vsetvli a6, zero, e32, m2, ta, mu
100003c8: mv s2, t0
100003cc: mv a6, s1
100003d0: add s1, t0, t3
100003d4: vmv1r.v v0, v8
100003d8: bgeu s1, t5, 0x100003e0 <main+0x374>
100003dc: mv s2, a6
100003e0: vslideup.vi v20, v18, 0x1
100003e4: slli s1, s2, 0x1
100003e8: add s1, a0, s1
100003ec: add s3, s2, a3
100003f0: slli s3, s3, 0x1
100003f4: add s3, a0, s3
100003f8: vslideup.vi v26, v16, 0x1
100003fc: vle16.v v9, (s3)
10000400: vslideup.vi v24, v14, 0x1
10000404: vslideup.vi v22, v12, 0x1
10000408: vfadd.vv v18, v18, v20, v0.t
1000040c: vle16.v v20, (s1)
10000410: add s1, s2, t1
10000414: add s2, s2, t2
10000418: slli s1, s1, 0x1
1000041c: slli s2, s2, 0x1
10000420: add s1, a0, s1
10000424: vfadd.vv v16, v16, v26, v0.t
10000428: add s2, a0, s2
1000042c: vle16.v v21, (s2)
10000430: vfadd.vv v14, v14, v24, v0.t
10000434: vfadd.vv v12, v12, v22, v0.t
10000438: vle16.v v22, (s1)
1000043c: bltu a5, a4, 0x10000318 <main+0x2ac>
10000440: li s1, 0x1
10000444: mv s2, t6
10000448: sll s3, s0, s1
1000044c: vlm.v v0, (s2)
10000450: addi s2, s2, 0x100
10000454: addi s1, s1, 0x1
10000458: vslideup.vx v24, v18, s3
1000045c: vslideup.vx v26, v16, s3
10000460: vslideup.vx v28, v14, s3
10000464: vslideup.vx v30, v12, s3
10000468: vfadd.vv v18, v18, v24, v0.t
1000046c: vfadd.vv v16, v16, v26, v0.t
10000470: vfadd.vv v14, v14, v28, v0.t
10000474: vfadd.vv v12, v12, v30, v0.t
10000478: bne a5, s1, 0x10000448 <main+0x3dc>
1000047c: j 0x10000318 <main+0x2ac>
10000480: srli a3, a6, 0xe
10000484: beqz a3, 0x100004bc <main+0x450>
10000488: auipc a2, 0x19
1000048c: rdcycle a3
10000490: ld s0, 0x18(sp)
10000494: sub a3, a3, a1
10000498: ld s1, 0x10(sp)
1000049c: li a0, 0x0
100004a0: ld s2, 0x8(sp)
100004a4: ld s3, 0x0(sp)
100004a8: sd a3, -0x3c8(a2)
100004ac: addi sp, sp, 0x20
100004b0: ret
100004b4: li a6, 0x0
100004b8: fmv.w.x fa5, zero
100004bc: lui a3, 0x4
100004c0: li a4, 0x2
100004c4: j 0x100004f8 <main+0x48c>
100004c8: addi a7, a5, -0x1
100004cc: vsetvli zero, zero, e32, m2, ta, ma
100004d0: vslidedown.vx v12, v8, a7
100004d4: slli a7, a6, 0x2
100004d8: add a6, a5, a6
100004dc: add a7, a2, a7
100004e0: srli a5, a6, 0xe
100004e4: vfmv.f.s fa4, v12
100004e8: vfadd.vf v8, v8, fa5
100004ec: fadd.s fa5, fa5, fa4
100004f0: vse32.v v8, (a7)
100004f4: bnez a5, 0x10000488 <main+0x41c>
100004f8: slli a7, a6, 0x1
100004fc: sub a5, a3, a6
10000500: add a7, a0, a7
10000504: vsetvli a5, a5, e16, m1, ta, ma
10000508: vle16.v v12, (a7)
1000050c: vfwcvt.f.f.v v8, v12
10000510: bltu a5, a4, 0x100004c8 <main+0x45c>
10000514: li a7, 0x1
10000518: vsetvli zero, zero, e32, m2, ta, mu
1000051c: vmsltu.vx v12, v10, a7
10000520: vmnot.m v0, v12
10000524: vslideup.vx v12, v8, a7
10000528: slli a7, a7, 0x1
1000052c: vfadd.vv v8, v8, v12, v0.t
10000530: bltu a7, a5, 0x10000518 <main+0x4ac>
10000534: j 0x100004c8 <main+0x45c>
