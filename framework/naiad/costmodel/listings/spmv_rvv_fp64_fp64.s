# spmv/rvv size=[256, 0.25] fp64/fp64 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x30
10000070: sd s0, 0x28(sp)
10000074: sd s1, 0x20(sp)
10000078: sd s2, 0x18(sp)
1000007c: sd s3, 0x10(sp)
10000080: sd s4, 0x8(sp)
10000084: auipc a1, 0x30
10000088: auipc a4, 0x20
1000008c: addi a1, a1, 0x57c
10000090: auipc a3, 0x0
10000094: addi a5, a4, 0x4f8
10000098: auipc a4, 0x28
1000009c: li a0, 0x0
100000a0: vsetivli zero, 0x10, e64, m4, ta, ma
100000a4: addi a2, a1, 0x40
100000a8: addi a3, a3, 0x4f0
100000ac: addi a6, a4, 0x568
100000b0: vmv.v.i v8, 0x0
100000b4: li a7, 0xc
100000b8: li t0, 0x10
100000bc: rdcycle a4
100000c0: j 0x100000c8 <main+0x5c>
100000c4: beq a0, t0, 0x100004a8 <main+0x43c>
100000c8: mv t3, a0
100000cc: slli t1, a0, 0x2
100000d0: addi a0, a0, 0x1
100000d4: add t1, a2, t1
100000d8: slli t2, a0, 0x2
100000dc: li s1, 0x0
100000e0: add t2, a2, t2
100000e4: slli t3, t3, 0x8
100000e8: lw t1, 0x0(t1)
100000ec: lw t2, 0x0(t2)
100000f0: not t4, t1
100000f4: add t4, t4, t2
100000f8: slli t5, t1, 0x2
100000fc: slli t4, t4, 0x20
10000100: add t6, a1, t5
10000104: srli t4, t4, 0x1e
10000108: slli s0, t1, 0xb
1000010c: add t6, t6, t4
10000110: add t5, a1, t5
10000114: add s0, a3, s0
10000118: addi t4, t5, 0xc0
1000011c: addi t5, s0, 0x100
10000120: addi t6, t6, 0xc4
10000124: j 0x10000164 <main+0xf8>
10000128: slli s1, s0, 0x4
1000012c: vsetivli zero, 0x10, e64, m4, ta, ma
10000130: add s1, s1, t3
10000134: addi t5, t5, 0x200
10000138: slli s1, s1, 0x3
1000013c: add s1, a6, s1
10000140: addi s2, s1, 0x80
10000144: vse64.v v20, (s1)
10000148: vse64.v v16, (s2)
1000014c: addi s2, s1, 0x100
10000150: vse64.v v12, (s2)
10000154: addi s2, s1, 0x180
10000158: addi s1, s0, 0x4
1000015c: vse64.v v28, (s2)
10000160: bgeu s0, a7, 0x100000c4 <main+0x58>
10000164: vsetivli zero, 0x1, e8, m1, ta, ma
10000168: mv s0, s1
1000016c: vmv4r.v v28, v8
10000170: mv s1, t5
10000174: mv s2, t4
10000178: vmv4r.v v12, v8
1000017c: vmv4r.v v16, v8
10000180: vmv4r.v v20, v8
10000184: bge t1, t2, 0x10000128 <main+0xbc>
10000188: lw s3, 0x0(s2)
1000018c: addi s2, s2, 0x4
10000190: slli s3, s3, 0xb
10000194: vsetivli zero, 0x10, e64, m4, ta, ma
10000198: add s3, a5, s3
1000019c: fld fa5, -0x100(s1)
100001a0: fld fa4, -0x80(s1)
100001a4: addi s4, s3, 0x80
100001a8: fld fa3, 0x0(s1)
100001ac: vle64.v v24, (s3)
100001b0: fld fa2, 0x80(s1)
100001b4: vsetivli zero, 0x10, e64, m4, ta, ma
100001b8: vfmacc.vf v20, fa5, v24
100001bc: fld fa5, -0xf8(s1)
100001c0: vfmacc.vf v16, fa4, v24
100001c4: fld fa4, -0x78(s1)
100001c8: vfmacc.vf v12, fa3, v24
100001cc: fld fa3, 0x8(s1)
100001d0: vfmadd.vf v24, fa2, v28
100001d4: fld fa2, 0x88(s1)
100001d8: vle64.v v28, (s4)
100001dc: vsetivli zero, 0x10, e64, m4, ta, ma
100001e0: addi s4, s3, 0x100
100001e4: vfmacc.vf v24, fa2, v28
100001e8: fld fa2, 0x90(s1)
100001ec: vfmacc.vf v20, fa5, v28
100001f0: fld fa5, -0xf0(s1)
100001f4: vfmacc.vf v16, fa4, v28
100001f8: fld fa4, -0x70(s1)
100001fc: vfmacc.vf v12, fa3, v28
10000200: vsetivli zero, 0x10, e64, m4, ta, ma
10000204: vle64.v v28, (s4)
10000208: fld fa3, 0x10(s1)
1000020c: vsetivli zero, 0x10, e64, m4, ta, ma
10000210: addi s4, s3, 0x180
10000214: vfmacc.vf v24, fa2, v28
10000218: fld fa2, 0x98(s1)
1000021c: vfmacc.vf v20, fa5, v28
10000220: fld fa5, -0xe8(s1)
10000224: vfmacc.vf v16, fa4, v28
10000228: fld fa4, -0x68(s1)
1000022c: vfmacc.vf v12, fa3, v28
10000230: vsetivli zero, 0x10, e64, m4, ta, ma
10000234: vle64.v v28, (s4)
10000238: fld fa3, 0x18(s1)
1000023c: vsetivli zero, 0x10, e64, m4, ta, ma
10000240: addi s4, s3, 0x200
10000244: vfmacc.vf v24, fa2, v28
10000248: fld fa2, 0xa0(s1)
1000024c: vfmacc.vf v20, fa5, v28
10000250: fld fa5, -0xe0(s1)
10000254: vfmacc.vf v16, fa4, v28
10000258: fld fa4, -0x60(s1)
1000025c: vfmacc.vf v12, fa3, v28
10000260: vsetivli zero, 0x10, e64, m4, ta, ma
10000264: vle64.v v28, (s4)
10000268: fld fa3, 0x20(s1)
1000026c: vsetivli zero, 0x10, e64, m4, ta, ma
10000270: addi s4, s3, 0x280
10000274: vfmacc.vf v24, fa2, v28
10000278: fld fa2, 0xa8(s1)
1000027c: vfmacc.vf v20, fa5, v28
10000280: fld fa5, -0xd8(s1)
10000284: vfmacc.vf v16, fa4, v28
10000288: fld fa4, -0x58(s1)
1000028c: vfmacc.vf v12, fa3, v28
10000290: vsetivli zero, 0x10, e64, m4, ta, ma
10000294: vle64.v v28, (s4)
10000298: fld fa3, 0x28(s1)
1000029c: vsetivli zero, 0x10, e64, m4, ta, ma
100002a0: addi s4, s3, 0x300
100002a4: vfmacc.vf v24, fa2, v28
100002a8: fld fa2, 0xb0(s1)
100002ac: vfmacc.vf v20, fa5, v28
100002b0: fld fa5, -0xd0(s1)
100002b4: vfmacc.vf v16, fa4, v28
100002b8: fld fa4, -0x50(s1)
100002bc: vfmacc.vf v12, fa3, v28
100002c0: vsetivli zero, 0x10, e64, m4, ta, ma
100002c4: vle64.v v28, (s4)
100002c8: fld fa3, 0x30(s1)
100002cc: vsetivli zero, 0x10, e64, m4, ta, ma
100002d0: addi s4, s3, 0x380
100002d4: vfmacc.vf v24, fa2, v28
100002d8: fld fa2, 0xb8(s1)
100002dc: vfmacc.vf v20, fa5, v28
100002e0: fld fa5, -0xc8(s1)
100002e4: vfmacc.vf v16, fa4, v28
100002e8: fld fa4, -0x48(s1)
100002ec: vfmacc.vf v12, fa3, v28
100002f0: vsetivli zero, 0x10, e64, m4, ta, ma
100002f4: vle64.v v28, (s4)
100002f8: fld fa3, 0x38(s1)
100002fc: vsetivli zero, 0x10, e64, m4, ta, ma
10000300: addi s4, s3, 0x400
10000304: vfmacc.vf v24, fa2, v28
10000308: fld fa2, 0xc0(s1)
1000030c: vfmacc.vf v20, fa5, v28
10000310: fld fa5, -0xc0(s1)
10000314: vfmacc.vf v16, fa4, v28
10000318: fld fa4, -0x40(s1)
1000031c: vfmacc.vf v12, fa3, v28
10000320: vsetivli zero, 0x10, e64, m4, ta, ma
10000324: vle64.v v28, (s4)
10000328: fld fa3, 0x40(s1)
1000032c: vsetivli zero, 0x10, e64, m4, ta, ma
10000330: addi s4, s3, 0x480
10000334: vfmacc.vf v24, fa2, v28
10000338: fld fa2, 0xc8(s1)
1000033c: vfmacc.vf v20, fa5, v28
10000340: fld fa5, -0xb8(s1)
10000344: vfmacc.vf v16, fa4, v28
10000348: fld fa4, -0x38(s1)
1000034c: vfmacc.vf v12, fa3, v28
10000350: vsetivli zero, 0x10, e64, m4, ta, ma
10000354: vle64.v v28, (s4)
10000358: fld fa3, 0x48(s1)
1000035c: vsetivli zero, 0x10, e64, m4, ta, ma
10000360: addi s4, s3, 0x500
10000364: vfmacc.vf v24, fa2, v28
10000368: fld fa2, 0xd0(s1)
1000036c: vfmacc.vf v20, fa5, v28
10000370: fld fa5, -0xb0(s1)
10000374: vfmacc.vf v16, fa4, v28
10000378: fld fa4, -0x30(s1)
1000037c: vfmacc.vf v12, fa3, v28
10000380: vsetivli zero, 0x10, e64, m4, ta, ma
10000384: vle64.v v28, (s4)
10000388: fld fa3, 0x50(s1)
1000038c: vsetivli zero, 0x10, e64, m4, ta, ma
10000390: addi s4, s3, 0x580
10000394: vfmacc.vf v24, fa2, v28
10000398: fld fa2, 0xd8(s1)
1000039c: vfmacc.vf v20, fa5, v28
100003a0: fld fa5, -0xa8(s1)
100003a4: vfmacc.vf v16, fa4, v28
100003a8: fld fa4, -0x28(s1)
100003ac: vfmacc.vf v12, fa3, v28
100003b0: vsetivli zero, 0x10, e64, m4, ta, ma
100003b4: vle64.v v28, (s4)
100003b8: fld fa3, 0x58(s1)
100003bc: vsetivli zero, 0x10, e64, m4, ta, ma
100003c0: addi s4, s3, 0x600
100003c4: vfmacc.vf v24, fa2, v28
100003c8: fld fa2, 0xe0(s1)
100003cc: vfmacc.vf v20, fa5, v28
100003d0: fld fa5, -0xa0(s1)
100003d4: vfmacc.vf v16, fa4, v28
100003d8: fld fa4, -0x20(s1)
100003dc: vfmacc.vf v12, fa3, v28
100003e0: vsetivli zero, 0x10, e64, m4, ta, ma
100003e4: vle64.v v28, (s4)
100003e8: fld fa3, 0x60(s1)
100003ec: vsetivli zero, 0x10, e64, m4, ta, ma
100003f0: addi s4, s3, 0x680
100003f4: vfmacc.vf v24, fa2, v28
100003f8: fld fa2, 0xe8(s1)
100003fc: vfmacc.vf v20, fa5, v28
10000400: fld fa5, -0x98(s1)
10000404: vfmacc.vf v16, fa4, v28
10000408: fld fa4, -0x18(s1)
1000040c: vfmacc.vf v12, fa3, v28
10000410: vsetivli zero, 0x10, e64, m4, ta, ma
10000414: vle64.v v28, (s4)
10000418: fld fa3, 0x68(s1)
1000041c: vsetivli zero, 0x10, e64, m4, ta, ma
10000420: addi s4, s3, 0x700
10000424: addi s3, s3, 0x780
10000428: vfmacc.vf v24, fa2, v28
1000042c: fld fa2, 0x70(s1)
10000430: vfmacc.vf v20, fa5, v28
10000434: fld fa5, -0x90(s1)
10000438: vfmacc.vf v16, fa4, v28
1000043c: fld fa4, -0x10(s1)
10000440: vfmacc.vf v12, fa3, v28
10000444: fld fa3, 0xf0(s1)
10000448: vsetivli zero, 0x10, e64, m4, ta, ma
1000044c: vle64.v v28, (s4)
10000450: vsetivli zero, 0x10, e64, m4, ta, ma
10000454: vfmacc.vf v24, fa3, v28
10000458: fld fa3, 0x78(s1)
1000045c: vfmacc.vf v12, fa2, v28
10000460: fld fa2, 0xf8(s1)
10000464: vfmacc.vf v16, fa4, v28
10000468: fld fa4, -0x8(s1)
1000046c: vfmacc.vf v20, fa5, v28
10000470: vsetivli zero, 0x10, e64, m4, ta, ma
10000474: vle64.v v28, (s3)
10000478: fld fa5, -0x88(s1)
1000047c: vsetivli zero, 0x10, e64, m4, ta, ma
10000480: addi s1, s1, 0x7ff
10000484: addi s1, s1, 0x1
10000488: vfmacc.vf v24, fa2, v28
1000048c: vfmacc.vf v20, fa5, v28
10000490: vfmacc.vf v16, fa4, v28
10000494: vfmacc.vf v12, fa3, v28
10000498: vsetivli zero, 0x1, e8, m1, ta, ma
1000049c: vmv4r.v v28, v24
100004a0: bne s2, t6, 0x10000188 <main+0x11c>
100004a4: j 0x10000128 <main+0xbc>
100004a8: rdcycle a2
100004ac: ld s0, 0x28(sp)
100004b0: ld s1, 0x20(sp)
100004b4: sub a2, a2, a4
100004b8: ld s2, 0x18(sp)
100004bc: li a0, 0x0
100004c0: ld s3, 0x10(sp)
100004c4: ld s4, 0x8(sp)
100004c8: sd a2, 0x0(a1)
100004cc: addi sp, sp, 0x30
100004d0: ret
