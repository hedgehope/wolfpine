# nbody/rvv size=[256, "clustered"] fp64/fp64 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x60
10000070: sd s0, 0x58(sp)
10000074: sd s1, 0x50(sp)
10000078: sd s2, 0x48(sp)
1000007c: sd s3, 0x40(sp)
10000080: sd s4, 0x38(sp)
10000084: sd s5, 0x30(sp)
10000088: sd s6, 0x28(sp)
1000008c: sd s7, 0x20(sp)
10000090: sd s8, 0x18(sp)
10000094: sd s9, 0x10(sp)
10000098: sd s10, 0x8(sp)
1000009c: sd s11, 0x0(sp)
100000a0: auipc a1, 0x2
100000a4: auipc a0, 0x0
100000a8: addi a6, a1, -0x4a0
100000ac: auipc a1, 0x1
100000b0: auipc a3, 0x1
100000b4: li a4, 0x0
100000b8: li a5, 0x100
100000bc: addi a0, a0, 0x324
100000c0: li a7, 0x18
100000c4: addi a2, a1, -0x4e4
100000c8: addi a3, a3, 0x318
100000cc: rdcycle a1
100000d0: slli t0, a4, 0x3
100000d4: slli t1, a4, 0x5
100000d8: sub t1, t1, t0
100000dc: sub t2, a5, a4
100000e0: add t1, a6, t1
100000e4: vsetvli t2, t2, e64, m2, ta, ma
100000e8: add t3, a0, t0
100000ec: addi t4, t1, 0x8
100000f0: vlse64.v v8, (t1), a7
100000f4: addi t1, t1, 0x10
100000f8: add a4, t2, a4
100000fc: vse64.v v8, (t3)
10000100: add t3, a2, t0
10000104: vlse64.v v8, (t4), a7
10000108: add t0, a3, t0
1000010c: vse64.v v8, (t3)
10000110: vlse64.v v8, (t1), a7
10000114: vse64.v v8, (t0)
10000118: bltu a4, a5, 0x100000d0 <main+0x64>
1000011c: auipc a6, 0x0
10000120: auipc s1, 0x0
10000124: auipc t0, 0x3
10000128: addi a7, a6, 0x24c
1000012c: auipc a6, 0x85
10000130: addi t5, a2, 0x7ff
10000134: auipc t6, 0x83
10000138: li a4, 0x0
1000013c: li a5, 0x100
10000140: addi t0, t0, 0x2dc
10000144: addi t1, a2, 0x8
10000148: addi t2, a3, 0x8
1000014c: addi t3, a0, 0x8
10000150: addi a6, a6, -0x52c
10000154: lui t4, 0x1
10000158: addi t5, t5, 0x9
1000015c: addi t6, t6, 0x2cc
10000160: li s0, 0x18
10000164: fld fa5, 0x240(s1)
10000168: slli s2, a4, 0x3
1000016c: sub s3, a5, a4
10000170: add s7, t0, s2
10000174: vsetvli s1, s3, e64, m2, ta, ma
10000178: vmv.v.i v8, 0x0
1000017c: addi s7, s7, 0x7ff
10000180: vlse64.v v10, (a7), zero
10000184: add s4, a0, s2
10000188: add s5, a2, s2
1000018c: add s6, a3, s2
10000190: addi s7, s7, 0x1
10000194: mv s8, t3
10000198: vmv.v.i v12, 0x0
1000019c: mv s9, t2
100001a0: mv s10, t1
100001a4: vmv.v.i v14, 0x0
100001a8: vle64.v v20, (s4)
100001ac: fld fa4, -0x8(s8)
100001b0: fld fa3, 0x0(s8)
100001b4: addi s11, s7, -0x800
100001b8: fld fa1, -0x8(s9)
100001bc: addi s8, s8, 0x10
100001c0: fld fa0, 0x0(s9)
100001c4: addi s9, s9, 0x10
100001c8: vfrsub.vf v18, v20, fa4
100001cc: fld fa4, -0x8(s10)
100001d0: fld fa2, 0x0(s10)
100001d4: addi s10, s10, 0x10
100001d8: vfrsub.vf v16, v20, fa3
100001dc: vsetvli zero, s3, e64, m2, ta, ma
100001e0: vle64.v v22, (s5)
100001e4: vfrsub.vf v24, v22, fa4
100001e8: vfrsub.vf v20, v22, fa2
100001ec: vsetvli zero, s3, e64, m2, ta, ma
100001f0: vle64.v v28, (s6)
100001f4: fld fa4, 0x40(a6)
100001f8: vfmul.vv v30, v18, v18
100001fc: vfmul.vv v6, v16, v16
10000200: vfrsub.vf v26, v28, fa1
10000204: vfmacc.vv v30, v24, v24
10000208: vfrsub.vf v22, v28, fa0
1000020c: vfmacc.vv v6, v20, v20
10000210: vfmacc.vv v30, v26, v26
10000214: vfmacc.vv v6, v22, v22
10000218: vsetvli zero, s3, e64, m2, ta, ma
1000021c: vfadd.vf v4, v30, fa4
10000220: vse64.v v30, (s11)
10000224: vfadd.vf v2, v6, fa4
10000228: vse64.v v6, (s7)
1000022c: add s7, s7, t4
10000230: vfmul.vf v28, v4, fa5
10000234: vfrsqrt7.v v4, v4
10000238: vfrsqrt7.v v0, v2
1000023c: vfmul.vf v30, v2, fa5
10000240: vsetvli zero, s3, e64, m2, ta, ma
10000244: vfmul.vv v6, v4, v4
10000248: vsetvli zero, s3, e64, m2, ta, ma
1000024c: vfmul.vv v2, v0, v0
10000250: vfnmsub.vv v6, v28, v10
10000254: vfnmsub.vv v2, v30, v10
10000258: vfmul.vv v6, v4, v6
1000025c: vfmul.vv v4, v0, v2
10000260: vsetvli zero, s3, e64, m2, ta, ma
10000264: vfmul.vv v2, v6, v6
10000268: vsetvli zero, s3, e64, m2, ta, ma
1000026c: vfmul.vv v0, v4, v4
10000270: vfnmsub.vv v2, v28, v10
10000274: vfnmsub.vv v0, v30, v10
10000278: vfmul.vv v6, v6, v2
1000027c: vfmul.vv v4, v4, v0
10000280: vsetvli zero, s3, e64, m2, ta, ma
10000284: vfmul.vv v2, v6, v6
10000288: vsetvli zero, s3, e64, m2, ta, ma
1000028c: vfmul.vv v0, v4, v4
10000290: vfnmsub.vv v2, v28, v10
10000294: vfnmsub.vv v0, v30, v10
10000298: vfmul.vv v6, v6, v2
1000029c: vfmul.vv v4, v4, v0
100002a0: vsetvli zero, s3, e64, m2, ta, ma
100002a4: vfmul.vv v2, v6, v6
100002a8: vsetvli zero, s3, e64, m2, ta, ma
100002ac: vfmul.vv v0, v4, v4
100002b0: vfnmsub.vv v2, v28, v10
100002b4: vfnmsub.vv v0, v30, v10
100002b8: vfmul.vv v28, v6, v2
100002bc: vfmul.vv v30, v4, v0
100002c0: vfmul.vv v6, v28, v28
100002c4: vfmul.vv v4, v30, v30
100002c8: vfmul.vv v28, v6, v28
100002cc: vfmul.vv v30, v4, v30
100002d0: vfmacc.vv v8, v18, v28
100002d4: vfmacc.vv v12, v24, v28
100002d8: vfmadd.vv v28, v26, v14
100002dc: vfmacc.vv v8, v16, v30
100002e0: vmv.v.v v14, v28
100002e4: vfmacc.vv v12, v20, v30
100002e8: vfmacc.vv v14, v22, v30
100002ec: bne s10, t5, 0x100001a8 <main+0x13c>
100002f0: slli s3, a4, 0x5
100002f4: sub s2, t6, s2
100002f8: add s2, s2, s3
100002fc: add a4, s1, a4
10000300: addi s3, s2, 0x8
10000304: vsse64.v v8, (s2), s0
10000308: addi s2, s2, 0x10
1000030c: vsse64.v v12, (s3), s0
10000310: vsse64.v v14, (s2), s0
10000314: bltu a4, a5, 0x10000168 <main+0xfc>
10000318: rdcycle a2
1000031c: ld s0, 0x58(sp)
10000320: ld s1, 0x50(sp)
10000324: sub a2, a2, a1
10000328: ld s2, 0x48(sp)
1000032c: li a0, 0x0
10000330: ld s3, 0x40(sp)
10000334: ld s4, 0x38(sp)
10000338: ld s5, 0x30(sp)
1000033c: ld s6, 0x28(sp)
10000340: ld s7, 0x20(sp)
10000344: ld s8, 0x18(sp)
10000348: ld s9, 0x10(sp)
1000034c: ld s10, 0x8(sp)
10000350: ld s11, 0x0(sp)
10000354: sd a2, 0x0(a6)
10000358: addi sp, sp, 0x60
1000035c: ret
