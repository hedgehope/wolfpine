# jacobi/rvv size=66 fp64/fp64 cflags='' design=naiad-dwarfs.yaml
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
10000094: sd s9, 0x0(sp)
10000098: csrr a3, vlenb
1000009c: li a0, 0x11
100000a0: slli a0, a0, 0x8
100000a4: auipc a5, 0x9
100000a8: auipc a6, 0x1a
100000ac: lui t0, 0x9
100000b0: srli a1, a3, 0x3
100000b4: srli a4, a3, 0x2
100000b8: slli a2, a1, 0x1
100000bc: slli a3, a3, 0x1
100000c0: neg a2, a2
100000c4: and a0, a2, a0
100000c8: sub a2, a0, a4
100000cc: divu a2, a2, a4
100000d0: auipc a4, 0x0
100000d4: slli a2, a2, 0x4
100000d8: addi a2, a2, 0x10
100000dc: mul a7, a2, a1
100000e0: addi a1, a4, 0x3b0
100000e4: addi a2, a5, -0x3e4
100000e8: add a4, a1, a7
100000ec: addi a5, a6, -0x3a8
100000f0: addiw a6, t0, -0x7e0
100000f4: vl2re64.v v8, (a1)
100000f8: add a7, a2, a6
100000fc: add a1, a1, a3
10000100: vs2r.v v8, (a2)
10000104: add a2, a2, a3
10000108: vs2r.v v8, (a5)
1000010c: add a5, a5, a3
10000110: vs2r.v v8, (a7)
10000114: bne a1, a4, 0x100000f4 <main+0x88>
10000118: lui a1, 0x1
1000011c: addiw a1, a1, 0x104
10000120: beq a0, a1, 0x1000017c <main+0x110>
10000124: slli a3, a0, 0x3
10000128: auipc a0, 0x1a
1000012c: auipc a1, 0x9
10000130: addi a4, a0, -0x428
10000134: addi a0, a1, -0x46c
10000138: auipc a1, 0x0
1000013c: addi a2, a1, 0x348
10000140: lui a1, 0x9
10000144: addiw a5, a1, -0x7e0
10000148: add a0, a0, a3
1000014c: add a2, a2, a3
10000150: add a3, a4, a3
10000154: add a4, a4, a5
10000158: ld a5, 0x0(a2)
1000015c: add a6, a0, a1
10000160: sd a5, 0x0(a0)
10000164: addi a0, a0, 0x8
10000168: addi a2, a2, 0x8
1000016c: sd a5, 0x0(a3)
10000170: addi a3, a3, 0x8
10000174: sd a5, -0x7e0(a6)
10000178: bne a3, a4, 0x10000158 <main+0xec>
1000017c: auipc a7, 0x1a
10000180: auipc t4, 0x0
10000184: lui a2, 0x9
10000188: auipc a4, 0x9
1000018c: addi t0, a7, -0x47c
10000190: li a0, 0x0
10000194: li a1, 0x0
10000198: addiw a2, a2, -0x7e0
1000019c: li a3, 0x1
100001a0: addi a4, a4, -0x4c8
100001a4: li a5, 0x7
100001a8: li a6, 0x41
100001ac: addi a7, t0, 0x420
100001b0: addi t1, t0, 0x210
100001b4: li t2, 0x3f
100001b8: li t3, 0x8
100001bc: fld fa5, 0x260(t4)
100001c0: rdcycle t0
100001c4: j 0x100001d8 <main+0x16c>
100001c8: addi a3, a3, 0x1
100001cc: addi a0, a0, 0x1
100001d0: addiw a1, a1, 0x1
100001d4: beq a1, t3, 0x1000039c <main+0x330>
100001d8: andi t4, a0, 0x1
100001dc: mul t4, t4, a2
100001e0: add t4, a4, t4
100001e4: bne a1, a5, 0x100002b8 <main+0x24c>
100001e8: li t5, 0x1
100001ec: sub t6, a6, t5
100001f0: slli s0, t5, 0x3
100001f4: li s2, -0x1
100001f8: vsetvli s1, t6, e64, m4, ta, ma
100001fc: mv s3, t4
10000200: mv s4, t1
10000204: mv s5, a7
10000208: add s6, s3, s0
1000020c: addi s3, s3, 0x420
10000210: addi s7, s6, 0x420
10000214: addi s2, s2, 0x2
10000218: vle64.v v16, (s6)
1000021c: vle64.v v28, (s7)
10000220: addi s7, s6, 0x210
10000224: vle64.v v20, (s7)
10000228: addi s7, s6, 0x630
1000022c: vle64.v v24, (s7)
10000230: addi s7, s6, 0x208
10000234: vsetvli zero, t6, e64, m4, ta, ma
10000238: vfadd.vv v28, v16, v28
1000023c: vle64.v v8, (s7)
10000240: addi s7, s6, 0x418
10000244: vle64.v v12, (s7)
10000248: addi s7, s6, 0x218
1000024c: addi s6, s6, 0x428
10000250: vsetvli zero, t6, e64, m4, ta, ma
10000254: vfadd.vv v24, v20, v24
10000258: vle64.v v16, (s7)
1000025c: vle64.v v20, (s6)
10000260: vsetvli zero, t6, e64, m4, ta, ma
10000264: vfadd.vv v8, v28, v8
10000268: vsetvli zero, t6, e64, m4, ta, ma
1000026c: add s6, s4, s0
10000270: addi s4, s4, 0x420
10000274: vfadd.vv v12, v24, v12
10000278: vsetvli zero, t6, e64, m4, ta, ma
1000027c: vfadd.vv v8, v8, v16
10000280: vsetvli zero, t6, e64, m4, ta, ma
10000284: vfadd.vv v12, v12, v20
10000288: vsetvli zero, t6, e64, m4, ta, ma
1000028c: vfmul.vf v8, v8, fa5
10000290: vsetvli zero, t6, e64, m4, ta, ma
10000294: vse64.v v8, (s6)
10000298: add s6, s5, s0
1000029c: vfmul.vf v8, v12, fa5
100002a0: addi s5, s5, 0x420
100002a4: vse64.v v8, (s6)
100002a8: bltu s2, t2, 0x10000208 <main+0x19c>
100002ac: add t5, s1, t5
100002b0: bltu t5, a6, 0x100001ec <main+0x180>
100002b4: j 0x100001c8 <main+0x15c>
100002b8: andi t5, a3, 0x1
100002bc: li s0, 0x1
100002c0: mul t5, t5, a2
100002c4: add t6, a4, t5
100002c8: addi t5, t6, 0x210
100002cc: addi t6, t6, 0x420
100002d0: sub s1, a6, s0
100002d4: slli s2, s0, 0x3
100002d8: li s4, -0x1
100002dc: vsetvli s3, s1, e64, m4, ta, ma
100002e0: mv s5, t6
100002e4: mv s6, t4
100002e8: mv s7, t5
100002ec: add s8, s6, s2
100002f0: addi s6, s6, 0x420
100002f4: addi s9, s8, 0x420
100002f8: addi s4, s4, 0x2
100002fc: vle64.v v16, (s8)
10000300: vle64.v v28, (s9)
10000304: addi s9, s8, 0x210
10000308: vle64.v v20, (s9)
1000030c: addi s9, s8, 0x630
10000310: vle64.v v24, (s9)
10000314: addi s9, s8, 0x208
10000318: vsetvli zero, s1, e64, m4, ta, ma
1000031c: vfadd.vv v28, v16, v28
10000320: vle64.v v8, (s9)
10000324: addi s9, s8, 0x418
10000328: vle64.v v12, (s9)
1000032c: addi s9, s8, 0x218
10000330: addi s8, s8, 0x428
10000334: vsetvli zero, s1, e64, m4, ta, ma
10000338: vfadd.vv v24, v20, v24
1000033c: vle64.v v16, (s9)
10000340: vle64.v v20, (s8)
10000344: vsetvli zero, s1, e64, m4, ta, ma
10000348: vfadd.vv v8, v28, v8
1000034c: vsetvli zero, s1, e64, m4, ta, ma
10000350: add s8, s7, s2
10000354: addi s7, s7, 0x420
10000358: vfadd.vv v12, v24, v12
1000035c: vsetvli zero, s1, e64, m4, ta, ma
10000360: vfadd.vv v8, v8, v16
10000364: vsetvli zero, s1, e64, m4, ta, ma
10000368: vfadd.vv v12, v12, v20
1000036c: vsetvli zero, s1, e64, m4, ta, ma
10000370: vfmul.vf v8, v8, fa5
10000374: vsetvli zero, s1, e64, m4, ta, ma
10000378: vse64.v v8, (s8)
1000037c: add s8, s5, s2
10000380: vfmul.vf v8, v12, fa5
10000384: addi s5, s5, 0x420
10000388: vse64.v v8, (s8)
1000038c: bltu s4, t2, 0x100002ec <main+0x280>
10000390: add s0, s3, s0
10000394: bltu s0, a6, 0x100002d0 <main+0x264>
10000398: j 0x100001c8 <main+0x15c>
1000039c: auipc a1, 0x22
100003a0: rdcycle a2
100003a4: ld s0, 0x48(sp)
100003a8: sub a2, a2, t0
100003ac: ld s1, 0x40(sp)
100003b0: li a0, 0x0
100003b4: ld s2, 0x38(sp)
100003b8: ld s3, 0x30(sp)
100003bc: ld s4, 0x28(sp)
100003c0: ld s5, 0x20(sp)
100003c4: ld s6, 0x18(sp)
100003c8: ld s7, 0x10(sp)
100003cc: ld s8, 0x8(sp)
100003d0: ld s9, 0x0(sp)
100003d4: sd a2, 0x1a4(a1)
100003d8: addi sp, sp, 0x50
100003dc: ret
