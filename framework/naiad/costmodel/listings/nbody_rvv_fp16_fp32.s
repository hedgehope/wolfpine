# nbody/rvv size=[256, "clustered"] fp16/fp32 cflags='' design=naiad-dwarfs.yaml
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
1000009c: auipc a0, 0x0
100000a0: auipc a1, 0x42
100000a4: auipc a2, 0x0
100000a8: auipc a4, 0x1
100000ac: li a5, 0x0
100000b0: li a6, 0x100
100000b4: addi a0, a0, 0x2ec
100000b8: addi a1, a1, -0x4e0
100000bc: li a7, 0x6
100000c0: addi a3, a2, 0x6e4
100000c4: addi a4, a4, -0x520
100000c8: rdcycle a2
100000cc: slli t0, a5, 0x1
100000d0: slli t1, a5, 0x3
100000d4: sub t0, t1, t0
100000d8: sub t1, a6, a5
100000dc: add t0, a1, t0
100000e0: vsetvli t1, t1, e16, m1, ta, ma
100000e4: addi t2, t0, 0x80
100000e8: addi t4, t0, 0x82
100000ec: addi t0, t0, 0x84
100000f0: vlse16.v v8, (t2), a7
100000f4: slli t2, a5, 0x2
100000f8: add t3, a0, t2
100000fc: add a5, t1, a5
10000100: vfwcvt.f.f.v v10, v8
10000104: vse32.v v10, (t3)
10000108: add t3, a3, t2
1000010c: vlse16.v v8, (t4), a7
10000110: add t2, a4, t2
10000114: vfwcvt.f.f.v v10, v8
10000118: vse32.v v10, (t3)
1000011c: vlse16.v v8, (t0), a7
10000120: vfwcvt.f.f.v v10, v8
10000124: vse32.v v10, (t2)
10000128: bltu a5, a6, 0x100000cc <main+0x60>
1000012c: lui t4, 0x3f000
10000130: auipc t0, 0x1
10000134: auipc t5, 0x41
10000138: li a5, 0x0
1000013c: li a6, 0x100
10000140: lui a7, 0x3fc00
10000144: addi t0, t0, -0x170
10000148: addi t1, a3, 0x4
1000014c: addi t2, a4, 0x4
10000150: addi t3, a0, 0x4
10000154: fmv.w.x fa5, t4
10000158: addi t4, a3, 0x404
1000015c: addi t5, t5, -0x174
10000160: li t6, 0xc
10000164: sub s1, a6, a5
10000168: slli s2, a5, 0x2
1000016c: vsetvli s0, s1, e32, m2, ta, ma
10000170: vmv.v.i v8, 0x0
10000174: add s3, a0, s2
10000178: add s4, a3, s2
1000017c: add s5, a4, s2
10000180: add s7, t0, s2
10000184: mv s6, t3
10000188: mv s8, t2
1000018c: vmv.v.x v12, a7
10000190: mv s9, t1
10000194: vmv.v.i v10, 0x0
10000198: vmv.v.i v14, 0x0
1000019c: vle32.v v20, (s3)
100001a0: flw fa4, -0x4(s6)
100001a4: flw fa3, 0x0(s6)
100001a8: addi s10, s7, 0x400
100001ac: flw fa1, -0x4(s8)
100001b0: addi s6, s6, 0x8
100001b4: flw fa0, 0x0(s8)
100001b8: addi s8, s8, 0x8
100001bc: vfrsub.vf v18, v20, fa4
100001c0: flw fa4, -0x4(s9)
100001c4: flw fa2, 0x0(s9)
100001c8: addi s9, s9, 0x8
100001cc: vfrsub.vf v16, v20, fa3
100001d0: vsetvli zero, s1, e32, m2, ta, ma
100001d4: vle32.v v22, (s4)
100001d8: vfrsub.vf v24, v22, fa4
100001dc: vfrsub.vf v20, v22, fa2
100001e0: vsetvli zero, s1, e32, m2, ta, ma
100001e4: vle32.v v28, (s5)
100001e8: flw fa4, 0x40(a1)
100001ec: vfmul.vv v30, v18, v18
100001f0: vfmul.vv v6, v16, v16
100001f4: vfrsub.vf v26, v28, fa1
100001f8: vfmacc.vv v30, v24, v24
100001fc: vfrsub.vf v22, v28, fa0
10000200: vfmacc.vv v6, v20, v20
10000204: vfmacc.vv v30, v26, v26
10000208: vfmacc.vv v6, v22, v22
1000020c: vsetvli zero, s1, e32, m2, ta, ma
10000210: vfadd.vf v28, v30, fa4
10000214: vse32.v v30, (s7)
10000218: addi s7, s7, 0x7ff
1000021c: addi s7, s7, 0x1
10000220: vfadd.vf v4, v6, fa4
10000224: vse32.v v6, (s10)
10000228: vfmul.vf v2, v28, fa5
1000022c: vfrsqrt7.v v28, v28
10000230: vfrsqrt7.v v0, v4
10000234: vfmul.vf v30, v4, fa5
10000238: vsetvli zero, s1, e32, m2, ta, ma
1000023c: vfmul.vv v6, v28, v28
10000240: vsetvli zero, s1, e32, m2, ta, ma
10000244: vfmul.vv v4, v0, v0
10000248: vfnmsub.vv v6, v2, v12
1000024c: vfnmsub.vv v4, v30, v12
10000250: vfmul.vv v28, v28, v6
10000254: vfmul.vv v6, v0, v4
10000258: vsetvli zero, s1, e32, m2, ta, ma
1000025c: vfmul.vv v4, v28, v28
10000260: vsetvli zero, s1, e32, m2, ta, ma
10000264: vfmul.vv v0, v6, v6
10000268: vfnmsub.vv v4, v2, v12
1000026c: vfnmsub.vv v0, v30, v12
10000270: vfmul.vv v28, v28, v4
10000274: vfmul.vv v30, v6, v0
10000278: vfmul.vv v6, v28, v28
1000027c: vfmul.vv v4, v30, v30
10000280: vfmul.vv v28, v6, v28
10000284: vfmul.vv v30, v4, v30
10000288: vfmacc.vv v8, v18, v28
1000028c: vfmacc.vv v10, v24, v28
10000290: vfmadd.vv v28, v26, v14
10000294: vfmacc.vv v8, v16, v30
10000298: vmv.v.v v14, v28
1000029c: vfmacc.vv v10, v20, v30
100002a0: vfmacc.vv v14, v22, v30
100002a4: bne s9, t4, 0x1000019c <main+0x130>
100002a8: slli s1, a5, 0x4
100002ac: sub s2, t5, s2
100002b0: add s1, s2, s1
100002b4: add a5, s0, a5
100002b8: addi s2, s1, 0x4
100002bc: vsse32.v v8, (s1), t6
100002c0: addi s1, s1, 0x8
100002c4: vsse32.v v10, (s2), t6
100002c8: vsse32.v v14, (s1), t6
100002cc: bltu a5, a6, 0x10000164 <main+0xf8>
100002d0: rdcycle a3
100002d4: ld s0, 0x58(sp)
100002d8: ld s1, 0x50(sp)
100002dc: sub a3, a3, a2
100002e0: ld s2, 0x48(sp)
100002e4: li a0, 0x0
100002e8: ld s3, 0x40(sp)
100002ec: ld s4, 0x38(sp)
100002f0: ld s5, 0x30(sp)
100002f4: ld s6, 0x28(sp)
100002f8: ld s7, 0x20(sp)
100002fc: ld s8, 0x18(sp)
10000300: ld s9, 0x10(sp)
10000304: ld s10, 0x8(sp)
10000308: sd a3, 0x0(a1)
1000030c: addi sp, sp, 0x60
10000310: ret
