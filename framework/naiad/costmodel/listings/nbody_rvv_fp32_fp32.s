# nbody/rvv size=[256, "clustered"] fp32/fp32 cflags='' design=naiad-dwarfs.yaml
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
1000009c: auipc a1, 0x1
100000a0: auipc a0, 0x0
100000a4: addi a6, a1, -0xdc
100000a8: auipc a1, 0x0
100000ac: auipc a3, 0x1
100000b0: li a4, 0x0
100000b4: li a5, 0x100
100000b8: addi a0, a0, 0x2e8
100000bc: li a7, 0xc
100000c0: addi a2, a1, 0x6e0
100000c4: addi a3, a3, -0x524
100000c8: rdcycle a1
100000cc: slli t0, a4, 0x2
100000d0: slli t1, a4, 0x4
100000d4: sub t1, t1, t0
100000d8: sub t2, a5, a4
100000dc: add t1, a6, t1
100000e0: vsetvli t2, t2, e32, m2, ta, ma
100000e4: add t3, a0, t0
100000e8: addi t4, t1, 0x4
100000ec: vlse32.v v8, (t1), a7
100000f0: addi t1, t1, 0x8
100000f4: add a4, t2, a4
100000f8: vse32.v v8, (t3)
100000fc: add t3, a2, t0
10000100: vlse32.v v8, (t4), a7
10000104: add t0, a3, t0
10000108: vse32.v v8, (t3)
1000010c: vlse32.v v8, (t1), a7
10000110: vse32.v v8, (t0)
10000114: bltu a4, a5, 0x100000cc <main+0x60>
10000118: lui t4, 0x3f000
1000011c: auipc a7, 0x2
10000120: auipc t0, 0x42
10000124: auipc t5, 0x42
10000128: li a4, 0x0
1000012c: li a5, 0x100
10000130: lui a6, 0x3fc00
10000134: addi a7, a7, -0x55c
10000138: addi t1, a2, 0x4
1000013c: addi t2, a3, 0x4
10000140: addi t3, a0, 0x4
10000144: addi t0, t0, 0x6a0
10000148: fmv.w.x fa5, t4
1000014c: addi t4, a2, 0x404
10000150: addi t5, t5, -0x564
10000154: li t6, 0xc
10000158: sub s1, a5, a4
1000015c: slli s2, a4, 0x2
10000160: vsetvli s0, s1, e32, m2, ta, ma
10000164: vmv.v.i v8, 0x0
10000168: add s3, a0, s2
1000016c: add s4, a2, s2
10000170: add s5, a3, s2
10000174: add s7, a7, s2
10000178: mv s6, t3
1000017c: mv s8, t2
10000180: vmv.v.x v12, a6
10000184: mv s9, t1
10000188: vmv.v.i v10, 0x0
1000018c: vmv.v.i v14, 0x0
10000190: vle32.v v20, (s3)
10000194: flw fa4, -0x4(s6)
10000198: flw fa3, 0x0(s6)
1000019c: addi s10, s7, 0x400
100001a0: flw fa1, -0x4(s8)
100001a4: addi s6, s6, 0x8
100001a8: flw fa0, 0x0(s8)
100001ac: addi s8, s8, 0x8
100001b0: vfrsub.vf v18, v20, fa4
100001b4: flw fa4, -0x4(s9)
100001b8: flw fa2, 0x0(s9)
100001bc: addi s9, s9, 0x8
100001c0: vfrsub.vf v16, v20, fa3
100001c4: vsetvli zero, s1, e32, m2, ta, ma
100001c8: vle32.v v22, (s4)
100001cc: vfrsub.vf v24, v22, fa4
100001d0: vfrsub.vf v20, v22, fa2
100001d4: vsetvli zero, s1, e32, m2, ta, ma
100001d8: vle32.v v28, (s5)
100001dc: flw fa4, 0x40(t0)
100001e0: vfmul.vv v30, v18, v18
100001e4: vfmul.vv v6, v16, v16
100001e8: vfrsub.vf v26, v28, fa1
100001ec: vfmacc.vv v30, v24, v24
100001f0: vfrsub.vf v22, v28, fa0
100001f4: vfmacc.vv v6, v20, v20
100001f8: vfmacc.vv v30, v26, v26
100001fc: vfmacc.vv v6, v22, v22
10000200: vsetvli zero, s1, e32, m2, ta, ma
10000204: vfadd.vf v28, v30, fa4
10000208: vse32.v v30, (s7)
1000020c: addi s7, s7, 0x7ff
10000210: addi s7, s7, 0x1
10000214: vfadd.vf v4, v6, fa4
10000218: vse32.v v6, (s10)
1000021c: vfmul.vf v2, v28, fa5
10000220: vfrsqrt7.v v28, v28
10000224: vfrsqrt7.v v0, v4
10000228: vfmul.vf v30, v4, fa5
1000022c: vsetvli zero, s1, e32, m2, ta, ma
10000230: vfmul.vv v6, v28, v28
10000234: vsetvli zero, s1, e32, m2, ta, ma
10000238: vfmul.vv v4, v0, v0
1000023c: vfnmsub.vv v6, v2, v12
10000240: vfnmsub.vv v4, v30, v12
10000244: vfmul.vv v28, v28, v6
10000248: vfmul.vv v6, v0, v4
1000024c: vsetvli zero, s1, e32, m2, ta, ma
10000250: vfmul.vv v4, v28, v28
10000254: vsetvli zero, s1, e32, m2, ta, ma
10000258: vfmul.vv v0, v6, v6
1000025c: vfnmsub.vv v4, v2, v12
10000260: vfnmsub.vv v0, v30, v12
10000264: vfmul.vv v28, v28, v4
10000268: vfmul.vv v30, v6, v0
1000026c: vfmul.vv v6, v28, v28
10000270: vfmul.vv v4, v30, v30
10000274: vfmul.vv v28, v6, v28
10000278: vfmul.vv v30, v4, v30
1000027c: vfmacc.vv v8, v18, v28
10000280: vfmacc.vv v10, v24, v28
10000284: vfmadd.vv v28, v26, v14
10000288: vfmacc.vv v8, v16, v30
1000028c: vmv.v.v v14, v28
10000290: vfmacc.vv v10, v20, v30
10000294: vfmacc.vv v14, v22, v30
10000298: bne s9, t4, 0x10000190 <main+0x124>
1000029c: slli s1, a4, 0x4
100002a0: sub s2, t5, s2
100002a4: add s1, s2, s1
100002a8: add a4, s0, a4
100002ac: addi s2, s1, 0x4
100002b0: vsse32.v v8, (s1), t6
100002b4: addi s1, s1, 0x8
100002b8: vsse32.v v10, (s2), t6
100002bc: vsse32.v v14, (s1), t6
100002c0: bltu a4, a5, 0x10000158 <main+0xec>
100002c4: rdcycle a2
100002c8: ld s0, 0x58(sp)
100002cc: ld s1, 0x50(sp)
100002d0: sub a2, a2, a1
100002d4: ld s2, 0x48(sp)
100002d8: li a0, 0x0
100002dc: ld s3, 0x40(sp)
100002e0: ld s4, 0x38(sp)
100002e4: ld s5, 0x30(sp)
100002e8: ld s6, 0x28(sp)
100002ec: ld s7, 0x20(sp)
100002f0: ld s8, 0x18(sp)
100002f4: ld s9, 0x10(sp)
100002f8: ld s10, 0x8(sp)
100002fc: sd a2, 0x0(t0)
10000300: addi sp, sp, 0x60
10000304: ret
