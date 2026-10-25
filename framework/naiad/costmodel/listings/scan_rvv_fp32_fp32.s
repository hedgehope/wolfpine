# scan/rvv size=16384 fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x20
10000070: sd s0, 0x18(sp)
10000074: sd s1, 0x10(sp)
10000078: sd s2, 0x8(sp)
1000007c: sd s3, 0x0(sp)
10000080: vsetvli a3, zero, e32, m2, ta, ma
10000084: auipc a0, 0x1
10000088: vid.v v10
1000008c: auipc a1, 0x11
10000090: auipc a4, 0x0
10000094: addi a0, a0, -0x4
10000098: addi a2, a1, -0xc
1000009c: addi a6, a4, 0x4f0
100000a0: li a4, 0x2
100000a4: li a5, 0x0
100000a8: rdcycle a1
100000ac: bltu a3, a4, 0x10000278 <main+0x20c>
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
100000e0: bne a3, a7, 0x1000045c <main+0x3f0>
100000e4: addi a3, a0, 0x40
100000e8: vsetivli zero, 0x10, e32, m2, ta, ma
100000ec: auipc a4, 0x0
100000f0: vle32.v v18, (a0)
100000f4: fmv.w.x fa5, zero
100000f8: vle32.v v16, (a3)
100000fc: addi a3, a0, 0x80
10000100: vle32.v v14, (a3)
10000104: addi a3, a0, 0xc0
10000108: vle32.v v12, (a3)
1000010c: addi a3, a4, 0x494
10000110: addi a4, a3, 0x100
10000114: addi a5, a3, 0x200
10000118: vlm.v v8, (a3)
1000011c: addi a3, a3, 0x300
10000120: vlm.v v9, (a4)
10000124: lui a4, 0x4
10000128: vlm.v v10, (a5)
1000012c: addiw a4, a4, -0x7f
10000130: vlm.v v11, (a3)
10000134: li a3, 0x40
10000138: vsetivli zero, 0x10, e32, m2, ta, mu
1000013c: vslideup.vi v20, v18, 0x1
10000140: addi a5, a3, -0x40
10000144: mv a6, a5
10000148: vmv1r.v v0, v8
1000014c: bgeu a5, a4, 0x10000154 <main+0xe8>
10000150: mv a6, a3
10000154: vslideup.vi v22, v16, 0x1
10000158: slli a6, a6, 0x2
1000015c: add a6, a0, a6
10000160: addi a3, a3, 0x40
10000164: addi a7, a6, 0x40
10000168: vfadd.vv v18, v18, v20, v0.t
1000016c: vslideup.vi v24, v14, 0x1
10000170: vslideup.vi v26, v12, 0x1
10000174: vfadd.vv v16, v16, v22, v0.t
10000178: vfadd.vv v14, v14, v24, v0.t
1000017c: vfadd.vv v12, v12, v26, v0.t
10000180: vslideup.vi v20, v18, 0x2
10000184: vmv1r.v v0, v9
10000188: vslideup.vi v22, v16, 0x2
1000018c: vfadd.vv v18, v18, v20, v0.t
10000190: vslideup.vi v24, v14, 0x2
10000194: vslideup.vi v26, v12, 0x2
10000198: vfadd.vv v16, v16, v22, v0.t
1000019c: vfadd.vv v14, v14, v24, v0.t
100001a0: vfadd.vv v12, v12, v26, v0.t
100001a4: vslideup.vi v20, v18, 0x4
100001a8: vmv1r.v v0, v10
100001ac: vslideup.vi v22, v16, 0x4
100001b0: vfadd.vv v18, v18, v20, v0.t
100001b4: vslideup.vi v24, v14, 0x4
100001b8: vslideup.vi v26, v12, 0x4
100001bc: vfadd.vv v16, v16, v22, v0.t
100001c0: vfadd.vv v14, v14, v24, v0.t
100001c4: vfadd.vv v12, v12, v26, v0.t
100001c8: vslideup.vi v20, v18, 0x8
100001cc: vmv1r.v v0, v11
100001d0: vslideup.vi v22, v16, 0x8
100001d4: vfadd.vv v18, v18, v20, v0.t
100001d8: vfadd.vv v16, v16, v22, v0.t
100001dc: vslideup.vi v24, v14, 0x8
100001e0: vslidedown.vi v20, v18, 0xf
100001e4: vfadd.vv v14, v14, v24, v0.t
100001e8: vle32.v v24, (a6)
100001ec: vfmv.f.s fa4, v20
100001f0: vslidedown.vi v22, v16, 0xf
100001f4: fadd.s fa4, fa5, fa4
100001f8: vslideup.vi v26, v12, 0x8
100001fc: vfmv.f.s fa3, v22
10000200: vfadd.vv v12, v12, v26, v0.t
10000204: vle32.v v22, (a7)
10000208: addi a7, a6, 0x80
1000020c: addi a6, a6, 0xc0
10000210: vslidedown.vi v20, v14, 0xf
10000214: vfadd.vf v18, v18, fa5
10000218: fadd.s fa5, fa4, fa3
1000021c: vfadd.vf v16, v16, fa4
10000220: vfmv.f.s fa4, v20
10000224: vfadd.vf v14, v14, fa5
10000228: vse32.v v18, (a2)
1000022c: vle32.v v20, (a7)
10000230: addi a7, a2, 0x40
10000234: fadd.s fa5, fa5, fa4
10000238: vslidedown.vi v18, v12, 0xf
1000023c: vse32.v v16, (a7)
10000240: addi a7, a2, 0x80
10000244: vfmv.f.s fa4, v18
10000248: vfadd.vf v26, v12, fa5
1000024c: vle32.v v12, (a6)
10000250: addi a6, a2, 0xc0
10000254: vse32.v v14, (a7)
10000258: addi a2, a2, 0x100
1000025c: fadd.s fa5, fa5, fa4
10000260: vmv.v.v v18, v24
10000264: vmv.v.v v16, v22
10000268: vse32.v v26, (a6)
1000026c: vmv.v.v v14, v20
10000270: bltu a5, a4, 0x10000138 <main+0xcc>
10000274: j 0x10000430 <main+0x3c4>
10000278: vsetvli a7, zero, e32, m2, ta, ma
1000027c: slli a7, a3, 0x1
10000280: vle32.v v18, (a0)
10000284: add t0, a7, a3
10000288: lui t5, 0x4
1000028c: slli t1, t0, 0x2
10000290: li t2, 0x0
10000294: add t3, a0, t1
10000298: slli t1, a3, 0x3
1000029c: add t4, a0, t1
100002a0: fmv.w.x fa5, zero
100002a4: addiw t5, t5, 0x1
100002a8: addi t6, a6, 0x100
100002ac: vle32.v v12, (t3)
100002b0: slli t3, a3, 0x2
100002b4: li s0, 0x1
100002b8: mv s1, t3
100002bc: vle32.v v14, (t4)
100002c0: add t4, a0, t3
100002c4: vle32.v v16, (t4)
100002c8: addi t4, a3, -0x1
100002cc: vlm.v v8, (a6)
100002d0: j 0x1000035c <main+0x2f0>
100002d4: vslidedown.vx v28, v24, t4
100002d8: slli s1, t2, 0x2
100002dc: add s1, a2, s1
100002e0: add s2, t2, a3
100002e4: slli s2, s2, 0x2
100002e8: add s2, a2, s2
100002ec: vfmv.f.s fa4, v28
100002f0: vslidedown.vx v30, v26, t4
100002f4: vfadd.vf v24, v24, fa5
100002f8: fadd.s fa5, fa5, fa4
100002fc: vfmv.f.s fa4, v30
10000300: vslidedown.vx v28, v22, t4
10000304: vse32.v v24, (s1)
10000308: add s1, t2, a7
1000030c: vfadd.vf v26, v26, fa5
10000310: fadd.s fa5, fa5, fa4
10000314: slli s1, s1, 0x2
10000318: add t2, t2, t0
1000031c: add s1, a2, s1
10000320: slli t2, t2, 0x2
10000324: vfmv.f.s fa4, v28
10000328: vslidedown.vx v28, v20, t4
1000032c: fadd.s fa4, fa5, fa4
10000330: vse32.v v26, (s2)
10000334: vfadd.vf v22, v22, fa5
10000338: add s2, a2, t2
1000033c: mv t2, a6
10000340: vfmv.f.s fa5, v28
10000344: vfadd.vf v20, v20, fa4
10000348: fadd.s fa5, fa4, fa5
1000034c: vse32.v v22, (s1)
10000350: add s1, a6, t3
10000354: vse32.v v20, (s2)
10000358: bgeu s1, t5, 0x10000428 <main+0x3bc>
1000035c: vsetvli a6, zero, e32, m2, ta, mu
10000360: vslideup.vi v20, v18, 0x1
10000364: mv a6, s1
10000368: add s1, t2, t1
1000036c: mv s2, t2
10000370: vmv1r.v v0, v8
10000374: bgeu s1, t5, 0x1000037c <main+0x310>
10000378: mv s2, a6
1000037c: vslideup.vi v22, v16, 0x1
10000380: slli s1, s2, 0x2
10000384: add s1, a0, s1
10000388: add s3, s2, a3
1000038c: slli s3, s3, 0x2
10000390: add s3, a0, s3
10000394: vfadd.vv v18, v18, v20, v0.t
10000398: vslideup.vi v24, v14, 0x1
1000039c: vslideup.vi v20, v12, 0x1
100003a0: vfadd.vv v16, v16, v22, v0.t
100003a4: vfadd.vv v14, v14, v24, v0.t
100003a8: vfadd.vv v12, v12, v20, v0.t
100003ac: vmv.v.v v24, v18
100003b0: vle32.v v18, (s1)
100003b4: add s1, s2, a7
100003b8: add s2, s2, t0
100003bc: slli s1, s1, 0x2
100003c0: slli s2, s2, 0x2
100003c4: add s1, a0, s1
100003c8: add s2, a0, s2
100003cc: vmv.v.v v26, v16
100003d0: vle32.v v16, (s3)
100003d4: vmv.v.v v22, v14
100003d8: vle32.v v14, (s1)
100003dc: vmv.v.v v20, v12
100003e0: vle32.v v12, (s2)
100003e4: bltu a5, a4, 0x100002d4 <main+0x268>
100003e8: li s1, 0x1
100003ec: mv s2, t6
100003f0: sll s3, s0, s1
100003f4: vlm.v v0, (s2)
100003f8: addi s2, s2, 0x100
100003fc: addi s1, s1, 0x1
10000400: vslideup.vx v28, v24, s3
10000404: vslideup.vx v30, v26, s3
10000408: vslideup.vx v6, v22, s3
1000040c: vslideup.vx v4, v20, s3
10000410: vfadd.vv v24, v24, v28, v0.t
10000414: vfadd.vv v26, v26, v30, v0.t
10000418: vfadd.vv v22, v22, v6, v0.t
1000041c: vfadd.vv v20, v20, v4, v0.t
10000420: bne a5, s1, 0x100003f0 <main+0x384>
10000424: j 0x100002d4 <main+0x268>
10000428: srli a3, a6, 0xe
1000042c: beqz a3, 0x1000046c <main+0x400>
10000430: auipc a2, 0x21
10000434: rdcycle a3
10000438: ld s0, 0x18(sp)
1000043c: sub a3, a3, a1
10000440: ld s1, 0x10(sp)
10000444: li a0, 0x0
10000448: ld s2, 0x8(sp)
1000044c: ld s3, 0x0(sp)
10000450: sd a3, -0x3b0(a2)
10000454: addi sp, sp, 0x20
10000458: ret
1000045c: lui a7, 0x1
10000460: bgeu a7, a3, 0x1000027c <main+0x210>
10000464: li a6, 0x0
10000468: fmv.w.x fa5, zero
1000046c: lui a3, 0x4
10000470: li a4, 0x2
10000474: j 0x100004a4 <main+0x438>
10000478: addi t0, a5, -0x1
1000047c: vsetvli zero, zero, e32, m2, ta, ma
10000480: add a6, a5, a6
10000484: add a7, a2, a7
10000488: srli a5, a6, 0xe
1000048c: vslidedown.vx v12, v8, t0
10000490: vfmv.f.s fa4, v12
10000494: vfadd.vf v8, v8, fa5
10000498: fadd.s fa5, fa5, fa4
1000049c: vse32.v v8, (a7)
100004a0: bnez a5, 0x10000430 <main+0x3c4>
100004a4: slli a7, a6, 0x2
100004a8: sub a5, a3, a6
100004ac: add t0, a0, a7
100004b0: vsetvli a5, a5, e32, m2, ta, ma
100004b4: vle32.v v8, (t0)
100004b8: bltu a5, a4, 0x10000478 <main+0x40c>
100004bc: li t0, 0x1
100004c0: vsetvli zero, zero, e32, m2, ta, mu
100004c4: vmsltu.vx v12, v10, t0
100004c8: vmnot.m v0, v12
100004cc: vslideup.vx v12, v8, t0
100004d0: slli t0, t0, 0x1
100004d4: vfadd.vv v8, v8, v12, v0.t
100004d8: bltu t0, a5, 0x100004c0 <main+0x454>
100004dc: j 0x10000478 <main+0x40c>
