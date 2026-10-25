# scan/rvv size=16384 fp64/fp64 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x20
10000070: sd s0, 0x18(sp)
10000074: sd s1, 0x10(sp)
10000078: sd s2, 0x8(sp)
1000007c: sd s3, 0x0(sp)
10000080: vsetvli a2, zero, e64, m2, ta, ma
10000084: vid.v v10
10000088: auipc a0, 0x0
1000008c: li a3, 0x2
10000090: addi a5, a0, 0x4b8
10000094: li a4, 0x0
10000098: rdcycle a1
1000009c: bltu a2, a3, 0x10000248 <main+0x1dc>
100000a0: li a6, 0x1
100000a4: mv a0, a5
100000a8: vsetvli a7, zero, e64, m2, ta, ma
100000ac: vmsltu.vx v8, v10, a6
100000b0: mv a6, a4
100000b4: addi a4, a4, 0x1
100000b8: sll a6, a3, a6
100000bc: vmnot.m v8, v8
100000c0: vsm.v v8, (a0)
100000c4: addi a0, a0, 0x100
100000c8: blt a6, a2, 0x100000ac <main+0x40>
100000cc: li a0, 0x8
100000d0: bne a2, a0, 0x1000024c <main+0x1e0>
100000d4: auipc a0, 0x1
100000d8: vsetivli zero, 0x8, e64, m2, ta, ma
100000dc: addi a0, a0, -0x94
100000e0: auipc a3, 0x0
100000e4: addi a2, a0, 0x40
100000e8: fmv.d.x fa5, zero
100000ec: vle64.v v18, (a0)
100000f0: vle64.v v16, (a2)
100000f4: addi a2, a0, 0x80
100000f8: vle64.v v14, (a2)
100000fc: addi a2, a0, 0xc0
10000100: vle64.v v12, (a2)
10000104: addi a2, a3, 0x460
10000108: addi a3, a2, 0x100
1000010c: addi a4, a2, 0x200
10000110: vlm.v v8, (a2)
10000114: li a2, 0x20
10000118: vlm.v v9, (a3)
1000011c: auipc a3, 0x21
10000120: vlm.v v10, (a4)
10000124: lui a4, 0x4
10000128: addi a3, a3, -0xdc
1000012c: addiw a4, a4, -0x3f
10000130: vsetivli zero, 0x8, e64, m2, ta, mu
10000134: vslideup.vi v20, v18, 0x1
10000138: addi a5, a2, -0x20
1000013c: mv a6, a5
10000140: vmv1r.v v0, v8
10000144: bgeu a5, a4, 0x1000014c <main+0xe0>
10000148: mv a6, a2
1000014c: vslideup.vi v22, v16, 0x1
10000150: slli a6, a6, 0x3
10000154: add a6, a0, a6
10000158: addi a2, a2, 0x20
1000015c: addi a7, a6, 0x40
10000160: vfadd.vv v18, v18, v20, v0.t
10000164: vslideup.vi v24, v14, 0x1
10000168: vslideup.vi v26, v12, 0x1
1000016c: vfadd.vv v16, v16, v22, v0.t
10000170: vfadd.vv v14, v14, v24, v0.t
10000174: vfadd.vv v12, v12, v26, v0.t
10000178: vslideup.vi v20, v18, 0x2
1000017c: vmv1r.v v0, v9
10000180: vslideup.vi v22, v16, 0x2
10000184: vfadd.vv v18, v18, v20, v0.t
10000188: vslideup.vi v24, v14, 0x2
1000018c: vslideup.vi v26, v12, 0x2
10000190: vfadd.vv v16, v16, v22, v0.t
10000194: vfadd.vv v14, v14, v24, v0.t
10000198: vfadd.vv v12, v12, v26, v0.t
1000019c: vslideup.vi v20, v18, 0x4
100001a0: vmv1r.v v0, v10
100001a4: vslideup.vi v22, v16, 0x4
100001a8: vfadd.vv v18, v18, v20, v0.t
100001ac: vfadd.vv v16, v16, v22, v0.t
100001b0: vslidedown.vi v20, v18, 0x7
100001b4: vslideup.vi v24, v14, 0x4
100001b8: vfmv.f.s fa4, v20
100001bc: vfadd.vv v14, v14, v24, v0.t
100001c0: fadd.d fa4, fa5, fa4
100001c4: vslidedown.vi v22, v16, 0x7
100001c8: vslideup.vi v26, v12, 0x4
100001cc: vfmv.f.s fa3, v22
100001d0: vfadd.vv v12, v12, v26, v0.t
100001d4: vle64.v v22, (a7)
100001d8: addi a7, a6, 0x80
100001dc: vslidedown.vi v24, v14, 0x7
100001e0: vfadd.vf v18, v18, fa5
100001e4: fadd.d fa5, fa4, fa3
100001e8: vfmv.f.s fa3, v24
100001ec: vslidedown.vi v20, v12, 0x7
100001f0: vle64.v v24, (a7)
100001f4: addi a7, a3, 0x40
100001f8: vse64.v v18, (a3)
100001fc: vfadd.vf v16, v16, fa4
10000200: fadd.d fa4, fa5, fa3
10000204: vle64.v v18, (a6)
10000208: addi a6, a6, 0xc0
1000020c: vfadd.vf v14, v14, fa5
10000210: vfmv.f.s fa5, v20
10000214: vfadd.vf v20, v12, fa4
10000218: vse64.v v16, (a7)
1000021c: addi a7, a3, 0x80
10000220: vle64.v v12, (a6)
10000224: addi a6, a3, 0xc0
10000228: fadd.d fa5, fa4, fa5
1000022c: addi a3, a3, 0x100
10000230: vmv.v.v v16, v22
10000234: vse64.v v14, (a7)
10000238: vmv.v.v v14, v24
1000023c: vse64.v v20, (a6)
10000240: bltu a5, a4, 0x10000130 <main+0xc4>
10000244: j 0x10000488 <main+0x41c>
10000248: vsetvli a0, zero, e64, m2, ta, ma
1000024c: slli a7, a2, 0x1
10000250: auipc a0, 0x1
10000254: add t0, a7, a2
10000258: addi a0, a0, -0x210
1000025c: slli t2, a2, 0x4
10000260: slli a6, t0, 0x3
10000264: add t2, a0, t2
10000268: add a6, a0, a6
1000026c: slli t3, a2, 0x2
10000270: vle64.v v18, (a0)
10000274: lui t5, 0x4
10000278: li t1, 0x0
1000027c: addi t4, a2, -0x1
10000280: fmv.d.x fa5, zero
10000284: addiw t5, t5, 0x1
10000288: addi t6, a5, 0x100
1000028c: vle64.v v14, (t2)
10000290: slli t2, a2, 0x3
10000294: li s0, 0x1
10000298: mv s1, t3
1000029c: vle64.v v12, (a6)
100002a0: add a6, a0, t2
100002a4: vle64.v v16, (a6)
100002a8: auipc a6, 0x21
100002ac: vlm.v v8, (a5)
100002b0: addi a5, a6, -0x268
100002b4: j 0x10000340 <main+0x2d4>
100002b8: vslidedown.vx v28, v24, t4
100002bc: slli s1, t1, 0x3
100002c0: add s1, a5, s1
100002c4: add s2, t1, a2
100002c8: slli s2, s2, 0x3
100002cc: add s2, a5, s2
100002d0: vfmv.f.s fa4, v28
100002d4: vslidedown.vx v30, v26, t4
100002d8: fadd.d fa4, fa5, fa4
100002dc: vfmv.f.s fa3, v30
100002e0: vslidedown.vx v6, v22, t4
100002e4: vfadd.vf v24, v24, fa5
100002e8: fadd.d fa5, fa4, fa3
100002ec: vfmv.f.s fa3, v6
100002f0: vslidedown.vx v28, v20, t4
100002f4: vse64.v v24, (s1)
100002f8: add s1, t1, a7
100002fc: vfadd.vf v26, v26, fa4
10000300: fadd.d fa4, fa5, fa3
10000304: slli s1, s1, 0x3
10000308: add t1, t1, t0
1000030c: add s1, a5, s1
10000310: slli t1, t1, 0x3
10000314: vfadd.vf v22, v22, fa5
10000318: vfmv.f.s fa5, v28
1000031c: vfadd.vf v20, v20, fa4
10000320: vse64.v v26, (s2)
10000324: add s2, a5, t1
10000328: mv t1, a6
1000032c: fadd.d fa5, fa4, fa5
10000330: vse64.v v22, (s1)
10000334: add s1, a6, t3
10000338: vse64.v v20, (s2)
1000033c: bgeu s1, t5, 0x1000040c <main+0x3a0>
10000340: vsetvli a6, zero, e64, m2, ta, mu
10000344: vslideup.vi v20, v18, 0x1
10000348: mv a6, s1
1000034c: add s1, t1, t2
10000350: mv s2, t1
10000354: vmv1r.v v0, v8
10000358: bgeu s1, t5, 0x10000360 <main+0x2f4>
1000035c: mv s2, a6
10000360: vslideup.vi v22, v16, 0x1
10000364: slli s1, s2, 0x3
10000368: add s1, a0, s1
1000036c: add s3, s2, a2
10000370: slli s3, s3, 0x3
10000374: add s3, a0, s3
10000378: vfadd.vv v18, v18, v20, v0.t
1000037c: vslideup.vi v24, v14, 0x1
10000380: vslideup.vi v20, v12, 0x1
10000384: vfadd.vv v16, v16, v22, v0.t
10000388: vfadd.vv v14, v14, v24, v0.t
1000038c: vfadd.vv v12, v12, v20, v0.t
10000390: vmv.v.v v24, v18
10000394: vle64.v v18, (s1)
10000398: add s1, s2, a7
1000039c: add s2, s2, t0
100003a0: slli s1, s1, 0x3
100003a4: slli s2, s2, 0x3
100003a8: add s1, a0, s1
100003ac: add s2, a0, s2
100003b0: vmv.v.v v26, v16
100003b4: vle64.v v16, (s3)
100003b8: vmv.v.v v22, v14
100003bc: vle64.v v14, (s1)
100003c0: vmv.v.v v20, v12
100003c4: vle64.v v12, (s2)
100003c8: bltu a4, a3, 0x100002b8 <main+0x24c>
100003cc: li s1, 0x1
100003d0: mv s2, t6
100003d4: sll s3, s0, s1
100003d8: vlm.v v0, (s2)
100003dc: addi s2, s2, 0x100
100003e0: addi s1, s1, 0x1
100003e4: vslideup.vx v28, v24, s3
100003e8: vslideup.vx v30, v26, s3
100003ec: vslideup.vx v6, v22, s3
100003f0: vslideup.vx v4, v20, s3
100003f4: vfadd.vv v24, v24, v28, v0.t
100003f8: vfadd.vv v26, v26, v30, v0.t
100003fc: vfadd.vv v22, v22, v6, v0.t
10000400: vfadd.vv v20, v20, v4, v0.t
10000404: bne a4, s1, 0x100003d4 <main+0x368>
10000408: j 0x100002b8 <main+0x24c>
1000040c: srli a2, a6, 0xe
10000410: bnez a2, 0x10000488 <main+0x41c>
10000414: lui a2, 0x4
10000418: li a3, 0x2
1000041c: j 0x1000044c <main+0x3e0>
10000420: addi t0, a4, -0x1
10000424: vsetvli zero, zero, e64, m2, ta, ma
10000428: add a6, a4, a6
1000042c: add a7, a5, a7
10000430: srli a4, a6, 0xe
10000434: vslidedown.vx v12, v8, t0
10000438: vfmv.f.s fa4, v12
1000043c: vfadd.vf v8, v8, fa5
10000440: fadd.d fa5, fa5, fa4
10000444: vse64.v v8, (a7)
10000448: bnez a4, 0x10000488 <main+0x41c>
1000044c: slli a7, a6, 0x3
10000450: sub a4, a2, a6
10000454: add t0, a0, a7
10000458: vsetvli a4, a4, e64, m2, ta, ma
1000045c: vle64.v v8, (t0)
10000460: bltu a4, a3, 0x10000420 <main+0x3b4>
10000464: li t0, 0x1
10000468: vsetvli zero, zero, e64, m2, ta, mu
1000046c: vmsltu.vx v12, v10, t0
10000470: vmnot.m v0, v12
10000474: vslideup.vx v12, v8, t0
10000478: slli t0, t0, 0x1
1000047c: vfadd.vv v8, v8, v12, v0.t
10000480: bltu t0, a4, 0x10000468 <main+0x3fc>
10000484: j 0x10000420 <main+0x3b4>
10000488: auipc a2, 0x41
1000048c: rdcycle a3
10000490: ld s0, 0x18(sp)
10000494: sub a3, a3, a1
10000498: ld s1, 0x10(sp)
1000049c: li a0, 0x0
100004a0: ld s2, 0x8(sp)
100004a4: ld s3, 0x0(sp)
100004a8: sd a3, -0x448(a2)
100004ac: addi sp, sp, 0x20
100004b0: ret
