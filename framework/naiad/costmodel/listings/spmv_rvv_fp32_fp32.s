# spmv/rvv size=[256, 0.25] fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x30
10000070: sd s0, 0x28(sp)
10000074: sd s1, 0x20(sp)
10000078: sd s2, 0x18(sp)
1000007c: sd s3, 0x10(sp)
10000080: sd s4, 0x8(sp)
10000084: auipc a1, 0x18
10000088: auipc a4, 0x10
1000008c: addi a1, a1, 0x7bc
10000090: auipc a3, 0x0
10000094: addi a5, a4, 0x778
10000098: auipc a4, 0x14
1000009c: li a0, 0x0
100000a0: vsetivli zero, 0x10, e32, m2, ta, ma
100000a4: addi a2, a1, 0x40
100000a8: addi a3, a3, 0x770
100000ac: vmv.v.i v8, 0x0
100000b0: addi a6, a4, 0x7a8
100000b4: li a7, 0x10
100000b8: rdcycle a4
100000bc: j 0x100000c4 <main+0x58>
100000c0: beq a0, a7, 0x1000072c <main+0x6c0>
100000c4: mv t2, a0
100000c8: slli t1, a0, 0x2
100000cc: addi a0, a0, 0x1
100000d0: add t1, a2, t1
100000d4: slli t3, a0, 0x2
100000d8: li t0, 0x0
100000dc: add t3, a2, t3
100000e0: slli t2, t2, 0x8
100000e4: lw t1, 0x0(t1)
100000e8: lw t3, 0x0(t3)
100000ec: not t4, t1
100000f0: slli t5, t1, 0x2
100000f4: add s1, t4, t3
100000f8: add s0, a1, t5
100000fc: slli s1, s1, 0x20
10000100: slli t6, t1, 0xa
10000104: addi t4, s0, 0xc0
10000108: srli s1, s1, 0x1e
1000010c: add s0, a1, t5
10000110: add t6, a3, t6
10000114: add s0, s0, s1
10000118: addi t5, t6, 0x100
1000011c: addi t6, s0, 0xc4
10000120: li s0, 0x1
10000124: j 0x10000184 <main+0x118>
10000128: or s1, t0, t2
1000012c: vsetivli zero, 0x10, e32, m2, ta, ma
10000130: slli s1, s1, 0x2
10000134: li t0, 0x80
10000138: add s1, a6, s1
1000013c: addi s2, s1, 0x40
10000140: vse32.v v22, (s1)
10000144: vse32.v v20, (s2)
10000148: addi s2, s1, 0x80
1000014c: vse32.v v18, (s2)
10000150: addi s2, s1, 0xc0
10000154: vse32.v v16, (s2)
10000158: addi s2, s1, 0x100
1000015c: vse32.v v14, (s2)
10000160: addi s2, s1, 0x140
10000164: vse32.v v12, (s2)
10000168: addi s2, s1, 0x180
1000016c: addi s1, s1, 0x1c0
10000170: vse32.v v10, (s2)
10000174: andi s2, s0, 0x1
10000178: li s0, 0x0
1000017c: vse32.v v24, (s1)
10000180: beqz s2, 0x100000c0 <main+0x54>
10000184: vsetivli zero, 0x1, e8, m1, ta, ma
10000188: vmv2r.v v24, v8
1000018c: vmv2r.v v10, v8
10000190: vmv2r.v v12, v8
10000194: vmv2r.v v14, v8
10000198: vmv2r.v v16, v8
1000019c: vmv2r.v v18, v8
100001a0: vmv2r.v v20, v8
100001a4: vmv2r.v v22, v8
100001a8: bge t1, t3, 0x10000128 <main+0xbc>
100001ac: slli s2, t0, 0x2
100001b0: vmv2r.v v22, v8
100001b4: mv s1, t4
100001b8: add s2, t5, s2
100001bc: vmv2r.v v20, v8
100001c0: vmv2r.v v18, v8
100001c4: vmv2r.v v16, v8
100001c8: vmv2r.v v14, v8
100001cc: vmv2r.v v12, v8
100001d0: vmv2r.v v10, v8
100001d4: vmv2r.v v26, v8
100001d8: lw s3, 0x0(s1)
100001dc: addi s1, s1, 0x4
100001e0: slli s3, s3, 0xa
100001e4: flw fa5, -0x100(s2)
100001e8: add s3, a5, s3
100001ec: flw fa4, -0xc0(s2)
100001f0: flw fa3, -0x80(s2)
100001f4: addi s4, s3, 0x40
100001f8: flw fa2, -0x40(s2)
100001fc: flw fa1, 0x0(s2)
10000200: vsetivli zero, 0x10, e32, m2, ta, ma
10000204: vle32.v v24, (s3)
10000208: flw fa0, 0x40(s2)
1000020c: flw ft0, 0x80(s2)
10000210: flw ft1, 0xc0(s2)
10000214: vsetivli zero, 0x10, e32, m2, ta, ma
10000218: vfmacc.vf v22, fa5, v24
1000021c: flw fa5, -0xfc(s2)
10000220: vle32.v v28, (s4)
10000224: addi s4, s3, 0x80
10000228: vfmacc.vf v20, fa4, v24
1000022c: flw fa4, -0xbc(s2)
10000230: vfmacc.vf v18, fa3, v24
10000234: flw fa3, -0x7c(s2)
10000238: vfmacc.vf v16, fa2, v24
1000023c: flw fa2, -0x3c(s2)
10000240: vfmacc.vf v14, fa1, v24
10000244: flw fa1, 0x4(s2)
10000248: vfmacc.vf v12, fa0, v24
1000024c: flw fa0, 0x44(s2)
10000250: vfmacc.vf v10, ft0, v24
10000254: flw ft0, 0x84(s2)
10000258: vfmadd.vf v24, ft1, v26
1000025c: flw ft1, 0xc4(s2)
10000260: vsetivli zero, 0x10, e32, m2, ta, ma
10000264: vfmacc.vf v22, fa5, v28
10000268: flw fa5, -0xf8(s2)
1000026c: vfmacc.vf v20, fa4, v28
10000270: flw fa4, -0xb8(s2)
10000274: vfmacc.vf v18, fa3, v28
10000278: flw fa3, -0x78(s2)
1000027c: vfmacc.vf v16, fa2, v28
10000280: flw fa2, -0x38(s2)
10000284: vfmacc.vf v14, fa1, v28
10000288: vsetivli zero, 0x10, e32, m2, ta, ma
1000028c: flw fa1, 0x8(s2)
10000290: vfmacc.vf v12, fa0, v28
10000294: vsetivli zero, 0x10, e32, m2, ta, ma
10000298: vle32.v v26, (s4)
1000029c: flw fa0, 0x48(s2)
100002a0: addi s4, s3, 0xc0
100002a4: vfmacc.vf v24, ft1, v28
100002a8: flw ft1, 0xc8(s2)
100002ac: vfmacc.vf v10, ft0, v28
100002b0: vsetivli zero, 0x10, e32, m2, ta, ma
100002b4: flw ft0, 0x88(s2)
100002b8: vfmacc.vf v22, fa5, v26
100002bc: flw fa5, -0xf4(s2)
100002c0: vfmacc.vf v20, fa4, v26
100002c4: flw fa4, -0xb4(s2)
100002c8: vfmacc.vf v18, fa3, v26
100002cc: flw fa3, -0x74(s2)
100002d0: vfmacc.vf v16, fa2, v26
100002d4: flw fa2, -0x34(s2)
100002d8: vfmacc.vf v14, fa1, v26
100002dc: vsetivli zero, 0x10, e32, m2, ta, ma
100002e0: flw fa1, 0xc(s2)
100002e4: vfmacc.vf v12, fa0, v26
100002e8: vsetivli zero, 0x10, e32, m2, ta, ma
100002ec: vle32.v v28, (s4)
100002f0: flw fa0, 0x4c(s2)
100002f4: addi s4, s3, 0x100
100002f8: vfmacc.vf v24, ft1, v26
100002fc: flw ft1, 0xcc(s2)
10000300: vfmacc.vf v10, ft0, v26
10000304: vsetivli zero, 0x10, e32, m2, ta, ma
10000308: flw ft0, 0x8c(s2)
1000030c: vfmacc.vf v22, fa5, v28
10000310: flw fa5, -0xf0(s2)
10000314: vfmacc.vf v20, fa4, v28
10000318: flw fa4, -0xb0(s2)
1000031c: vfmacc.vf v18, fa3, v28
10000320: flw fa3, -0x70(s2)
10000324: vfmacc.vf v16, fa2, v28
10000328: flw fa2, -0x30(s2)
1000032c: vfmacc.vf v14, fa1, v28
10000330: vsetivli zero, 0x10, e32, m2, ta, ma
10000334: flw fa1, 0x10(s2)
10000338: vfmacc.vf v12, fa0, v28
1000033c: vsetivli zero, 0x10, e32, m2, ta, ma
10000340: vle32.v v26, (s4)
10000344: flw fa0, 0x50(s2)
10000348: addi s4, s3, 0x140
1000034c: vfmacc.vf v24, ft1, v28
10000350: flw ft1, 0xd0(s2)
10000354: vfmacc.vf v10, ft0, v28
10000358: vsetivli zero, 0x10, e32, m2, ta, ma
1000035c: flw ft0, 0x90(s2)
10000360: vfmacc.vf v22, fa5, v26
10000364: flw fa5, -0xec(s2)
10000368: vfmacc.vf v20, fa4, v26
1000036c: flw fa4, -0xac(s2)
10000370: vfmacc.vf v18, fa3, v26
10000374: flw fa3, -0x6c(s2)
10000378: vfmacc.vf v16, fa2, v26
1000037c: flw fa2, -0x2c(s2)
10000380: vfmacc.vf v14, fa1, v26
10000384: vsetivli zero, 0x10, e32, m2, ta, ma
10000388: flw fa1, 0x14(s2)
1000038c: vfmacc.vf v12, fa0, v26
10000390: vsetivli zero, 0x10, e32, m2, ta, ma
10000394: vle32.v v28, (s4)
10000398: flw fa0, 0x54(s2)
1000039c: addi s4, s3, 0x180
100003a0: vfmacc.vf v24, ft1, v26
100003a4: flw ft1, 0xd4(s2)
100003a8: vfmacc.vf v10, ft0, v26
100003ac: vsetivli zero, 0x10, e32, m2, ta, ma
100003b0: flw ft0, 0x94(s2)
100003b4: vfmacc.vf v22, fa5, v28
100003b8: flw fa5, -0xe8(s2)
100003bc: vfmacc.vf v20, fa4, v28
100003c0: flw fa4, -0xa8(s2)
100003c4: vfmacc.vf v18, fa3, v28
100003c8: flw fa3, -0x68(s2)
100003cc: vfmacc.vf v16, fa2, v28
100003d0: flw fa2, -0x28(s2)
100003d4: vfmacc.vf v14, fa1, v28
100003d8: vsetivli zero, 0x10, e32, m2, ta, ma
100003dc: flw fa1, 0x18(s2)
100003e0: vfmacc.vf v12, fa0, v28
100003e4: vsetivli zero, 0x10, e32, m2, ta, ma
100003e8: vle32.v v26, (s4)
100003ec: flw fa0, 0x58(s2)
100003f0: addi s4, s3, 0x1c0
100003f4: vfmacc.vf v24, ft1, v28
100003f8: flw ft1, 0xd8(s2)
100003fc: vfmacc.vf v10, ft0, v28
10000400: vsetivli zero, 0x10, e32, m2, ta, ma
10000404: flw ft0, 0x98(s2)
10000408: vfmacc.vf v22, fa5, v26
1000040c: flw fa5, -0xe4(s2)
10000410: vfmacc.vf v20, fa4, v26
10000414: flw fa4, -0xa4(s2)
10000418: vfmacc.vf v18, fa3, v26
1000041c: flw fa3, -0x64(s2)
10000420: vfmacc.vf v16, fa2, v26
10000424: flw fa2, -0x24(s2)
10000428: vfmacc.vf v14, fa1, v26
1000042c: vsetivli zero, 0x10, e32, m2, ta, ma
10000430: flw fa1, 0x1c(s2)
10000434: vfmacc.vf v12, fa0, v26
10000438: vsetivli zero, 0x10, e32, m2, ta, ma
1000043c: vle32.v v28, (s4)
10000440: flw fa0, 0x5c(s2)
10000444: addi s4, s3, 0x200
10000448: vfmacc.vf v24, ft1, v26
1000044c: flw ft1, 0xdc(s2)
10000450: vfmacc.vf v10, ft0, v26
10000454: vsetivli zero, 0x10, e32, m2, ta, ma
10000458: flw ft0, 0x9c(s2)
1000045c: vfmacc.vf v22, fa5, v28
10000460: flw fa5, -0xe0(s2)
10000464: vfmacc.vf v20, fa4, v28
10000468: flw fa4, -0xa0(s2)
1000046c: vfmacc.vf v18, fa3, v28
10000470: flw fa3, -0x60(s2)
10000474: vfmacc.vf v16, fa2, v28
10000478: flw fa2, -0x20(s2)
1000047c: vfmacc.vf v14, fa1, v28
10000480: vsetivli zero, 0x10, e32, m2, ta, ma
10000484: flw fa1, 0x20(s2)
10000488: vfmacc.vf v12, fa0, v28
1000048c: vsetivli zero, 0x10, e32, m2, ta, ma
10000490: vle32.v v26, (s4)
10000494: flw fa0, 0x60(s2)
10000498: addi s4, s3, 0x240
1000049c: vfmacc.vf v24, ft1, v28
100004a0: flw ft1, 0xe0(s2)
100004a4: vfmacc.vf v10, ft0, v28
100004a8: vsetivli zero, 0x10, e32, m2, ta, ma
100004ac: flw ft0, 0xa0(s2)
100004b0: vfmacc.vf v22, fa5, v26
100004b4: flw fa5, -0xdc(s2)
100004b8: vfmacc.vf v20, fa4, v26
100004bc: flw fa4, -0x9c(s2)
100004c0: vfmacc.vf v18, fa3, v26
100004c4: flw fa3, -0x5c(s2)
100004c8: vfmacc.vf v16, fa2, v26
100004cc: flw fa2, -0x1c(s2)
100004d0: vfmacc.vf v14, fa1, v26
100004d4: vsetivli zero, 0x10, e32, m2, ta, ma
100004d8: flw fa1, 0x24(s2)
100004dc: vfmacc.vf v12, fa0, v26
100004e0: vsetivli zero, 0x10, e32, m2, ta, ma
100004e4: vle32.v v28, (s4)
100004e8: flw fa0, 0x64(s2)
100004ec: addi s4, s3, 0x280
100004f0: vfmacc.vf v24, ft1, v26
100004f4: flw ft1, 0xe4(s2)
100004f8: vfmacc.vf v10, ft0, v26
100004fc: vsetivli zero, 0x10, e32, m2, ta, ma
10000500: flw ft0, 0xa4(s2)
10000504: vfmacc.vf v22, fa5, v28
10000508: flw fa5, -0xd8(s2)
1000050c: vfmacc.vf v20, fa4, v28
10000510: flw fa4, -0x98(s2)
10000514: vfmacc.vf v18, fa3, v28
10000518: flw fa3, -0x58(s2)
1000051c: vfmacc.vf v16, fa2, v28
10000520: flw fa2, -0x18(s2)
10000524: vfmacc.vf v14, fa1, v28
10000528: vsetivli zero, 0x10, e32, m2, ta, ma
1000052c: flw fa1, 0x28(s2)
10000530: vfmacc.vf v12, fa0, v28
10000534: vsetivli zero, 0x10, e32, m2, ta, ma
10000538: vle32.v v26, (s4)
1000053c: flw fa0, 0x68(s2)
10000540: addi s4, s3, 0x2c0
10000544: vfmacc.vf v24, ft1, v28
10000548: flw ft1, 0xe8(s2)
1000054c: vfmacc.vf v10, ft0, v28
10000550: vsetivli zero, 0x10, e32, m2, ta, ma
10000554: flw ft0, 0xa8(s2)
10000558: vfmacc.vf v22, fa5, v26
1000055c: flw fa5, -0xd4(s2)
10000560: vfmacc.vf v20, fa4, v26
10000564: flw fa4, -0x94(s2)
10000568: vfmacc.vf v18, fa3, v26
1000056c: flw fa3, -0x54(s2)
10000570: vfmacc.vf v16, fa2, v26
10000574: flw fa2, -0x14(s2)
10000578: vfmacc.vf v14, fa1, v26
1000057c: vsetivli zero, 0x10, e32, m2, ta, ma
10000580: flw fa1, 0x2c(s2)
10000584: vfmacc.vf v12, fa0, v26
10000588: vsetivli zero, 0x10, e32, m2, ta, ma
1000058c: vle32.v v28, (s4)
10000590: flw fa0, 0x6c(s2)
10000594: addi s4, s3, 0x300
10000598: vfmacc.vf v24, ft1, v26
1000059c: flw ft1, 0xec(s2)
100005a0: vfmacc.vf v10, ft0, v26
100005a4: vsetivli zero, 0x10, e32, m2, ta, ma
100005a8: flw ft0, 0xac(s2)
100005ac: vfmacc.vf v22, fa5, v28
100005b0: flw fa5, -0xd0(s2)
100005b4: vfmacc.vf v20, fa4, v28
100005b8: flw fa4, -0x90(s2)
100005bc: vfmacc.vf v18, fa3, v28
100005c0: flw fa3, -0x50(s2)
100005c4: vfmacc.vf v16, fa2, v28
100005c8: flw fa2, -0x10(s2)
100005cc: vfmacc.vf v14, fa1, v28
100005d0: vsetivli zero, 0x10, e32, m2, ta, ma
100005d4: flw fa1, 0x30(s2)
100005d8: vfmacc.vf v12, fa0, v28
100005dc: vsetivli zero, 0x10, e32, m2, ta, ma
100005e0: vle32.v v30, (s4)
100005e4: flw fa0, 0x70(s2)
100005e8: addi s4, s3, 0x340
100005ec: vfmacc.vf v24, ft1, v28
100005f0: flw ft1, 0xf0(s2)
100005f4: vfmacc.vf v10, ft0, v28
100005f8: vsetivli zero, 0x10, e32, m2, ta, ma
100005fc: flw ft0, 0xb0(s2)
10000600: vfmacc.vf v22, fa5, v30
10000604: flw fa5, -0xcc(s2)
10000608: vfmacc.vf v20, fa4, v30
1000060c: flw fa4, -0x8c(s2)
10000610: vfmacc.vf v18, fa3, v30
10000614: flw fa3, -0x4c(s2)
10000618: vfmacc.vf v16, fa2, v30
1000061c: flw fa2, -0xc(s2)
10000620: vfmacc.vf v14, fa1, v30
10000624: vsetivli zero, 0x10, e32, m2, ta, ma
10000628: flw fa1, 0x34(s2)
1000062c: vfmacc.vf v12, fa0, v30
10000630: vsetivli zero, 0x10, e32, m2, ta, ma
10000634: vle32.v v26, (s4)
10000638: flw fa0, 0x74(s2)
1000063c: addi s4, s3, 0x380
10000640: addi s3, s3, 0x3c0
10000644: vfmacc.vf v24, ft1, v30
10000648: flw ft1, 0xf4(s2)
1000064c: vfmacc.vf v10, ft0, v30
10000650: vsetivli zero, 0x10, e32, m2, ta, ma
10000654: flw ft0, 0xb4(s2)
10000658: vfmacc.vf v22, fa5, v26
1000065c: flw fa5, -0xc8(s2)
10000660: vfmacc.vf v20, fa4, v26
10000664: flw fa4, -0x88(s2)
10000668: vfmacc.vf v18, fa3, v26
1000066c: flw fa3, -0x48(s2)
10000670: vfmacc.vf v16, fa2, v26
10000674: flw fa2, -0x8(s2)
10000678: vfmacc.vf v14, fa1, v26
1000067c: vsetivli zero, 0x10, e32, m2, ta, ma
10000680: flw fa1, 0x38(s2)
10000684: vfmacc.vf v12, fa0, v26
10000688: vsetivli zero, 0x10, e32, m2, ta, ma
1000068c: vle32.v v28, (s4)
10000690: flw fa0, 0x78(s2)
10000694: vfmacc.vf v24, ft1, v26
10000698: flw ft1, 0xb8(s2)
1000069c: vfmacc.vf v10, ft0, v26
100006a0: flw ft0, 0xf8(s2)
100006a4: vsetivli zero, 0x10, e32, m2, ta, ma
100006a8: vfmacc.vf v24, ft0, v28
100006ac: flw ft0, 0xbc(s2)
100006b0: vfmacc.vf v10, ft1, v28
100006b4: flw ft1, 0xfc(s2)
100006b8: vfmacc.vf v12, fa0, v28
100006bc: flw fa0, 0x3c(s2)
100006c0: vfmacc.vf v14, fa1, v28
100006c4: flw fa1, -0x4(s2)
100006c8: vfmacc.vf v16, fa2, v28
100006cc: vsetivli zero, 0x10, e32, m2, ta, ma
100006d0: flw fa2, -0x44(s2)
100006d4: vfmacc.vf v18, fa3, v28
100006d8: vsetivli zero, 0x10, e32, m2, ta, ma
100006dc: vle32.v v26, (s3)
100006e0: flw fa3, -0xc4(s2)
100006e4: vfmacc.vf v22, fa5, v28
100006e8: flw fa5, -0x84(s2)
100006ec: vfmacc.vf v20, fa4, v28
100006f0: flw fa4, 0x7c(s2)
100006f4: vsetivli zero, 0x10, e32, m2, ta, ma
100006f8: addi s2, s2, 0x400
100006fc: vfmacc.vf v24, ft1, v26
10000700: vfmacc.vf v22, fa3, v26
10000704: vfmacc.vf v20, fa5, v26
10000708: vfmacc.vf v18, fa2, v26
1000070c: vfmacc.vf v16, fa1, v26
10000710: vfmacc.vf v14, fa0, v26
10000714: vfmacc.vf v12, fa4, v26
10000718: vfmacc.vf v10, ft0, v26
1000071c: vsetivli zero, 0x1, e8, m1, ta, ma
10000720: vmv2r.v v26, v24
10000724: bne s1, t6, 0x100001d8 <main+0x16c>
10000728: j 0x10000128 <main+0xbc>
1000072c: rdcycle a2
10000730: ld s0, 0x28(sp)
10000734: ld s1, 0x20(sp)
10000738: sub a2, a2, a4
1000073c: ld s2, 0x18(sp)
10000740: li a0, 0x0
10000744: ld s3, 0x10(sp)
10000748: ld s4, 0x8(sp)
1000074c: sd a2, 0x0(a1)
10000750: addi sp, sp, 0x30
10000754: ret
