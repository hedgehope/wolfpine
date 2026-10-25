# spmv/rvv size=[256, 0.25] fp16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x30
10000070: sd s0, 0x28(sp)
10000074: sd s1, 0x20(sp)
10000078: sd s2, 0x18(sp)
1000007c: sd s3, 0x10(sp)
10000080: sd s4, 0x8(sp)
10000084: auipc a1, 0xe
10000088: auipc a4, 0x8
1000008c: addi a1, a1, 0x7fc
10000090: auipc a3, 0x0
10000094: addi a5, a4, 0x7b8
10000098: auipc a4, 0xa
1000009c: li a0, 0x0
100000a0: vsetivli zero, 0x10, e32, m2, ta, ma
100000a4: addi a2, a1, 0x40
100000a8: addi a3, a3, 0x7b0
100000ac: vmv.v.i v8, 0x0
100000b0: addi a6, a4, 0x7e8
100000b4: li a7, 0x10
100000b8: rdcycle a4
100000bc: j 0x100000c4 <main+0x58>
100000c0: beq a0, a7, 0x10000794 <main+0x728>
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
10000100: slli t6, t1, 0x9
10000104: addi t4, s0, 0xc0
10000108: srli s1, s1, 0x1e
1000010c: add s0, a1, t5
10000110: add t6, a3, t6
10000114: add s0, s0, s1
10000118: addi t5, t6, 0x80
1000011c: addi t6, s0, 0xc4
10000120: li s0, 0x1
10000124: j 0x10000184 <main+0x118>
10000128: or s1, t0, t2
1000012c: vsetivli zero, 0x10, e32, m2, ta, ma
10000130: slli s1, s1, 0x2
10000134: li t0, 0x80
10000138: add s1, a6, s1
1000013c: addi s2, s1, 0x40
10000140: vse32.v v24, (s1)
10000144: vse32.v v22, (s2)
10000148: addi s2, s1, 0x80
1000014c: vse32.v v20, (s2)
10000150: addi s2, s1, 0xc0
10000154: vse32.v v18, (s2)
10000158: addi s2, s1, 0x100
1000015c: vse32.v v16, (s2)
10000160: addi s2, s1, 0x140
10000164: vse32.v v14, (s2)
10000168: addi s2, s1, 0x180
1000016c: addi s1, s1, 0x1c0
10000170: vse32.v v12, (s2)
10000174: andi s2, s0, 0x1
10000178: li s0, 0x0
1000017c: vse32.v v10, (s1)
10000180: beqz s2, 0x100000c0 <main+0x54>
10000184: vsetivli zero, 0x1, e8, m1, ta, ma
10000188: vmv2r.v v10, v8
1000018c: vmv2r.v v12, v8
10000190: vmv2r.v v14, v8
10000194: vmv2r.v v16, v8
10000198: vmv2r.v v18, v8
1000019c: vmv2r.v v20, v8
100001a0: vmv2r.v v22, v8
100001a4: vmv2r.v v24, v8
100001a8: bge t1, t3, 0x10000128 <main+0xbc>
100001ac: slli s2, t0, 0x1
100001b0: vmv2r.v v24, v8
100001b4: mv s1, t4
100001b8: add s2, t5, s2
100001bc: vmv2r.v v22, v8
100001c0: vmv2r.v v20, v8
100001c4: vmv2r.v v18, v8
100001c8: vmv2r.v v16, v8
100001cc: vmv2r.v v14, v8
100001d0: vmv2r.v v12, v8
100001d4: vmv2r.v v10, v8
100001d8: lw s3, 0x0(s1)
100001dc: addi s1, s1, 0x4
100001e0: slli s3, s3, 0x9
100001e4: flh fa5, -0x80(s2)
100001e8: flh fa4, -0x60(s2)
100001ec: add s3, a5, s3
100001f0: flh fa3, -0x40(s2)
100001f4: addi s4, s3, 0x20
100001f8: flh fa2, -0x20(s2)
100001fc: vsetivli zero, 0x10, e16, m1, ta, ma
10000200: vle16.v v26, (s3)
10000204: flh fa1, 0x0(s2)
10000208: flh fa0, 0x20(s2)
1000020c: flh ft0, 0x40(s2)
10000210: flh ft1, 0x60(s2)
10000214: vsetivli zero, 0x10, e16, m1, ta, ma
10000218: vfwmacc.vf v24, fa5, v26
1000021c: flh fa5, -0x7e(s2)
10000220: vle16.v v27, (s4)
10000224: vfwmacc.vf v22, fa4, v26
10000228: flh fa4, -0x5e(s2)
1000022c: addi s4, s3, 0x40
10000230: vfwmacc.vf v20, fa3, v26
10000234: flh fa3, -0x3e(s2)
10000238: vfwmacc.vf v18, fa2, v26
1000023c: flh fa2, -0x1e(s2)
10000240: vfwmacc.vf v10, ft1, v26
10000244: flh ft1, 0x62(s2)
10000248: vfwmacc.vf v12, ft0, v26
1000024c: flh ft0, 0x42(s2)
10000250: vfwmacc.vf v14, fa0, v26
10000254: flh fa0, 0x22(s2)
10000258: vfwmacc.vf v16, fa1, v26
1000025c: vsetivli zero, 0x10, e16, m1, ta, ma
10000260: flh fa1, 0x2(s2)
10000264: vfwmacc.vf v24, fa5, v27
10000268: vsetivli zero, 0x10, e16, m1, ta, ma
1000026c: vfwmacc.vf v22, fa4, v27
10000270: vsetivli zero, 0x10, e16, m1, ta, ma
10000274: vfwmacc.vf v20, fa3, v27
10000278: vsetivli zero, 0x10, e16, m1, ta, ma
1000027c: vfwmacc.vf v18, fa2, v27
10000280: vsetivli zero, 0x10, e16, m1, ta, ma
10000284: vfwmacc.vf v10, ft1, v27
10000288: flh fa5, -0x7c(s2)
1000028c: flh fa4, -0x5c(s2)
10000290: vle16.v v26, (s4)
10000294: flh fa3, -0x3c(s2)
10000298: vfwmacc.vf v12, ft0, v27
1000029c: flh fa2, -0x1c(s2)
100002a0: addi s4, s3, 0x60
100002a4: vfwmacc.vf v14, fa0, v27
100002a8: flh fa0, 0x24(s2)
100002ac: flh ft0, 0x44(s2)
100002b0: flh ft1, 0x64(s2)
100002b4: vfwmacc.vf v16, fa1, v27
100002b8: vsetivli zero, 0x10, e16, m1, ta, ma
100002bc: flh fa1, 0x4(s2)
100002c0: vfwmacc.vf v24, fa5, v26
100002c4: vsetivli zero, 0x10, e16, m1, ta, ma
100002c8: vfwmacc.vf v22, fa4, v26
100002cc: vsetivli zero, 0x10, e16, m1, ta, ma
100002d0: vfwmacc.vf v20, fa3, v26
100002d4: vsetivli zero, 0x10, e16, m1, ta, ma
100002d8: vfwmacc.vf v18, fa2, v26
100002dc: vsetivli zero, 0x10, e16, m1, ta, ma
100002e0: vfwmacc.vf v10, ft1, v26
100002e4: flh fa5, -0x7a(s2)
100002e8: flh fa4, -0x5a(s2)
100002ec: vle16.v v27, (s4)
100002f0: flh fa3, -0x3a(s2)
100002f4: vfwmacc.vf v12, ft0, v26
100002f8: flh fa2, -0x1a(s2)
100002fc: addi s4, s3, 0x80
10000300: vfwmacc.vf v14, fa0, v26
10000304: flh fa0, 0x26(s2)
10000308: flh ft0, 0x46(s2)
1000030c: flh ft1, 0x66(s2)
10000310: vfwmacc.vf v16, fa1, v26
10000314: vsetivli zero, 0x10, e16, m1, ta, ma
10000318: flh fa1, 0x6(s2)
1000031c: vfwmacc.vf v24, fa5, v27
10000320: vsetivli zero, 0x10, e16, m1, ta, ma
10000324: vfwmacc.vf v22, fa4, v27
10000328: vsetivli zero, 0x10, e16, m1, ta, ma
1000032c: vfwmacc.vf v20, fa3, v27
10000330: vsetivli zero, 0x10, e16, m1, ta, ma
10000334: vfwmacc.vf v18, fa2, v27
10000338: vsetivli zero, 0x10, e16, m1, ta, ma
1000033c: vfwmacc.vf v10, ft1, v27
10000340: flh fa5, -0x78(s2)
10000344: flh fa4, -0x58(s2)
10000348: vle16.v v26, (s4)
1000034c: flh fa3, -0x38(s2)
10000350: vfwmacc.vf v12, ft0, v27
10000354: flh fa2, -0x18(s2)
10000358: addi s4, s3, 0xa0
1000035c: vfwmacc.vf v14, fa0, v27
10000360: flh fa0, 0x28(s2)
10000364: flh ft0, 0x48(s2)
10000368: flh ft1, 0x68(s2)
1000036c: vfwmacc.vf v16, fa1, v27
10000370: vsetivli zero, 0x10, e16, m1, ta, ma
10000374: flh fa1, 0x8(s2)
10000378: vfwmacc.vf v24, fa5, v26
1000037c: vsetivli zero, 0x10, e16, m1, ta, ma
10000380: vfwmacc.vf v22, fa4, v26
10000384: vsetivli zero, 0x10, e16, m1, ta, ma
10000388: vfwmacc.vf v20, fa3, v26
1000038c: vsetivli zero, 0x10, e16, m1, ta, ma
10000390: vfwmacc.vf v18, fa2, v26
10000394: vsetivli zero, 0x10, e16, m1, ta, ma
10000398: vfwmacc.vf v10, ft1, v26
1000039c: flh fa5, -0x76(s2)
100003a0: flh fa4, -0x56(s2)
100003a4: vle16.v v27, (s4)
100003a8: flh fa3, -0x36(s2)
100003ac: vfwmacc.vf v12, ft0, v26
100003b0: flh fa2, -0x16(s2)
100003b4: addi s4, s3, 0xc0
100003b8: vfwmacc.vf v14, fa0, v26
100003bc: flh fa0, 0x2a(s2)
100003c0: flh ft0, 0x4a(s2)
100003c4: flh ft1, 0x6a(s2)
100003c8: vfwmacc.vf v16, fa1, v26
100003cc: vsetivli zero, 0x10, e16, m1, ta, ma
100003d0: flh fa1, 0xa(s2)
100003d4: vfwmacc.vf v24, fa5, v27
100003d8: vsetivli zero, 0x10, e16, m1, ta, ma
100003dc: vfwmacc.vf v22, fa4, v27
100003e0: vsetivli zero, 0x10, e16, m1, ta, ma
100003e4: vfwmacc.vf v20, fa3, v27
100003e8: vsetivli zero, 0x10, e16, m1, ta, ma
100003ec: vfwmacc.vf v18, fa2, v27
100003f0: vsetivli zero, 0x10, e16, m1, ta, ma
100003f4: vfwmacc.vf v10, ft1, v27
100003f8: flh fa5, -0x74(s2)
100003fc: flh fa4, -0x54(s2)
10000400: vle16.v v26, (s4)
10000404: flh fa3, -0x34(s2)
10000408: vfwmacc.vf v12, ft0, v27
1000040c: flh fa2, -0x14(s2)
10000410: addi s4, s3, 0xe0
10000414: vfwmacc.vf v14, fa0, v27
10000418: flh fa0, 0x2c(s2)
1000041c: flh ft0, 0x4c(s2)
10000420: flh ft1, 0x6c(s2)
10000424: vfwmacc.vf v16, fa1, v27
10000428: vsetivli zero, 0x10, e16, m1, ta, ma
1000042c: flh fa1, 0xc(s2)
10000430: vfwmacc.vf v24, fa5, v26
10000434: vsetivli zero, 0x10, e16, m1, ta, ma
10000438: vfwmacc.vf v22, fa4, v26
1000043c: vsetivli zero, 0x10, e16, m1, ta, ma
10000440: vfwmacc.vf v20, fa3, v26
10000444: vsetivli zero, 0x10, e16, m1, ta, ma
10000448: vfwmacc.vf v18, fa2, v26
1000044c: vsetivli zero, 0x10, e16, m1, ta, ma
10000450: vfwmacc.vf v10, ft1, v26
10000454: flh fa5, -0x72(s2)
10000458: flh fa4, -0x52(s2)
1000045c: vle16.v v27, (s4)
10000460: flh fa3, -0x32(s2)
10000464: vfwmacc.vf v12, ft0, v26
10000468: flh fa2, -0x12(s2)
1000046c: addi s4, s3, 0x100
10000470: vfwmacc.vf v14, fa0, v26
10000474: flh fa0, 0x2e(s2)
10000478: flh ft0, 0x4e(s2)
1000047c: flh ft1, 0x6e(s2)
10000480: vfwmacc.vf v16, fa1, v26
10000484: vsetivli zero, 0x10, e16, m1, ta, ma
10000488: flh fa1, 0xe(s2)
1000048c: vfwmacc.vf v24, fa5, v27
10000490: vsetivli zero, 0x10, e16, m1, ta, ma
10000494: vfwmacc.vf v22, fa4, v27
10000498: vsetivli zero, 0x10, e16, m1, ta, ma
1000049c: vfwmacc.vf v20, fa3, v27
100004a0: vsetivli zero, 0x10, e16, m1, ta, ma
100004a4: vfwmacc.vf v18, fa2, v27
100004a8: vsetivli zero, 0x10, e16, m1, ta, ma
100004ac: vfwmacc.vf v10, ft1, v27
100004b0: flh fa5, -0x70(s2)
100004b4: flh fa4, -0x50(s2)
100004b8: vle16.v v26, (s4)
100004bc: flh fa3, -0x30(s2)
100004c0: vfwmacc.vf v12, ft0, v27
100004c4: flh fa2, -0x10(s2)
100004c8: addi s4, s3, 0x120
100004cc: vfwmacc.vf v14, fa0, v27
100004d0: flh fa0, 0x30(s2)
100004d4: flh ft0, 0x50(s2)
100004d8: flh ft1, 0x70(s2)
100004dc: vfwmacc.vf v16, fa1, v27
100004e0: vsetivli zero, 0x10, e16, m1, ta, ma
100004e4: flh fa1, 0x10(s2)
100004e8: vfwmacc.vf v24, fa5, v26
100004ec: vsetivli zero, 0x10, e16, m1, ta, ma
100004f0: vfwmacc.vf v22, fa4, v26
100004f4: vsetivli zero, 0x10, e16, m1, ta, ma
100004f8: vfwmacc.vf v20, fa3, v26
100004fc: vsetivli zero, 0x10, e16, m1, ta, ma
10000500: vfwmacc.vf v18, fa2, v26
10000504: vsetivli zero, 0x10, e16, m1, ta, ma
10000508: vfwmacc.vf v10, ft1, v26
1000050c: flh fa5, -0x6e(s2)
10000510: flh fa4, -0x4e(s2)
10000514: vle16.v v27, (s4)
10000518: flh fa3, -0x2e(s2)
1000051c: vfwmacc.vf v12, ft0, v26
10000520: flh fa2, -0xe(s2)
10000524: addi s4, s3, 0x140
10000528: vfwmacc.vf v14, fa0, v26
1000052c: flh fa0, 0x32(s2)
10000530: flh ft0, 0x52(s2)
10000534: flh ft1, 0x72(s2)
10000538: vfwmacc.vf v16, fa1, v26
1000053c: vsetivli zero, 0x10, e16, m1, ta, ma
10000540: flh fa1, 0x12(s2)
10000544: vfwmacc.vf v24, fa5, v27
10000548: vsetivli zero, 0x10, e16, m1, ta, ma
1000054c: vfwmacc.vf v22, fa4, v27
10000550: vsetivli zero, 0x10, e16, m1, ta, ma
10000554: vfwmacc.vf v20, fa3, v27
10000558: vsetivli zero, 0x10, e16, m1, ta, ma
1000055c: vfwmacc.vf v18, fa2, v27
10000560: vsetivli zero, 0x10, e16, m1, ta, ma
10000564: vfwmacc.vf v10, ft1, v27
10000568: flh fa5, -0x6c(s2)
1000056c: flh fa4, -0x4c(s2)
10000570: vle16.v v26, (s4)
10000574: flh fa3, -0x2c(s2)
10000578: vfwmacc.vf v12, ft0, v27
1000057c: flh fa2, -0xc(s2)
10000580: addi s4, s3, 0x160
10000584: vfwmacc.vf v14, fa0, v27
10000588: flh fa0, 0x34(s2)
1000058c: flh ft0, 0x54(s2)
10000590: flh ft1, 0x74(s2)
10000594: vfwmacc.vf v16, fa1, v27
10000598: vsetivli zero, 0x10, e16, m1, ta, ma
1000059c: flh fa1, 0x14(s2)
100005a0: vfwmacc.vf v24, fa5, v26
100005a4: vsetivli zero, 0x10, e16, m1, ta, ma
100005a8: vfwmacc.vf v22, fa4, v26
100005ac: vsetivli zero, 0x10, e16, m1, ta, ma
100005b0: vfwmacc.vf v20, fa3, v26
100005b4: vsetivli zero, 0x10, e16, m1, ta, ma
100005b8: vfwmacc.vf v18, fa2, v26
100005bc: vsetivli zero, 0x10, e16, m1, ta, ma
100005c0: vfwmacc.vf v10, ft1, v26
100005c4: flh fa5, -0x6a(s2)
100005c8: flh fa4, -0x4a(s2)
100005cc: vle16.v v27, (s4)
100005d0: flh fa3, -0x2a(s2)
100005d4: vfwmacc.vf v12, ft0, v26
100005d8: flh fa2, -0xa(s2)
100005dc: addi s4, s3, 0x180
100005e0: vfwmacc.vf v14, fa0, v26
100005e4: flh fa0, 0x36(s2)
100005e8: flh ft0, 0x56(s2)
100005ec: flh ft1, 0x76(s2)
100005f0: vfwmacc.vf v16, fa1, v26
100005f4: vsetivli zero, 0x10, e16, m1, ta, ma
100005f8: flh fa1, 0x16(s2)
100005fc: vfwmacc.vf v24, fa5, v27
10000600: vsetivli zero, 0x10, e16, m1, ta, ma
10000604: vfwmacc.vf v22, fa4, v27
10000608: vsetivli zero, 0x10, e16, m1, ta, ma
1000060c: vfwmacc.vf v20, fa3, v27
10000610: vsetivli zero, 0x10, e16, m1, ta, ma
10000614: vfwmacc.vf v18, fa2, v27
10000618: vsetivli zero, 0x10, e16, m1, ta, ma
1000061c: vfwmacc.vf v10, ft1, v27
10000620: flh fa5, -0x68(s2)
10000624: flh fa4, -0x48(s2)
10000628: vle16.v v28, (s4)
1000062c: flh fa3, -0x28(s2)
10000630: vfwmacc.vf v12, ft0, v27
10000634: flh fa2, -0x8(s2)
10000638: addi s4, s3, 0x1a0
1000063c: vfwmacc.vf v14, fa0, v27
10000640: flh fa0, 0x38(s2)
10000644: flh ft0, 0x58(s2)
10000648: flh ft1, 0x78(s2)
1000064c: vfwmacc.vf v16, fa1, v27
10000650: vsetivli zero, 0x10, e16, m1, ta, ma
10000654: flh fa1, 0x18(s2)
10000658: vfwmacc.vf v24, fa5, v28
1000065c: vsetivli zero, 0x10, e16, m1, ta, ma
10000660: vfwmacc.vf v22, fa4, v28
10000664: vsetivli zero, 0x10, e16, m1, ta, ma
10000668: vfwmacc.vf v20, fa3, v28
1000066c: vsetivli zero, 0x10, e16, m1, ta, ma
10000670: vfwmacc.vf v18, fa2, v28
10000674: vsetivli zero, 0x10, e16, m1, ta, ma
10000678: vfwmacc.vf v10, ft1, v28
1000067c: flh fa4, -0x66(s2)
10000680: flh fa3, -0x46(s2)
10000684: vle16.v v26, (s4)
10000688: flh fa2, -0x26(s2)
1000068c: vfwmacc.vf v12, ft0, v28
10000690: flh fa5, 0x1a(s2)
10000694: addi s4, s3, 0x1c0
10000698: vfwmacc.vf v14, fa0, v28
1000069c: flh fa0, 0x3a(s2)
100006a0: flh ft0, 0x5a(s2)
100006a4: addi s3, s3, 0x1e0
100006a8: flh ft1, 0x7a(s2)
100006ac: vfwmacc.vf v16, fa1, v28
100006b0: vsetivli zero, 0x10, e16, m1, ta, ma
100006b4: flh fa1, -0x6(s2)
100006b8: vfwmacc.vf v24, fa4, v26
100006bc: vsetivli zero, 0x10, e16, m1, ta, ma
100006c0: vfwmacc.vf v22, fa3, v26
100006c4: vsetivli zero, 0x10, e16, m1, ta, ma
100006c8: vfwmacc.vf v20, fa2, v26
100006cc: vsetivli zero, 0x10, e16, m1, ta, ma
100006d0: vfwmacc.vf v18, fa1, v26
100006d4: vsetivli zero, 0x10, e16, m1, ta, ma
100006d8: vfwmacc.vf v10, ft1, v26
100006dc: flh fa4, 0x1c(s2)
100006e0: flh fa3, 0x3c(s2)
100006e4: vle16.v v27, (s4)
100006e8: flh fa2, 0x5c(s2)
100006ec: vfwmacc.vf v12, ft0, v26
100006f0: flh fa1, 0x7c(s2)
100006f4: vfwmacc.vf v14, fa0, v26
100006f8: flh fa0, -0x64(s2)
100006fc: flh ft0, -0x44(s2)
10000700: vfwmacc.vf v16, fa5, v26
10000704: flh fa5, -0x24(s2)
10000708: flh ft1, -0x4(s2)
1000070c: vsetivli zero, 0x10, e16, m1, ta, ma
10000710: vfwmacc.vf v10, fa1, v27
10000714: vsetivli zero, 0x10, e16, m1, ta, ma
10000718: vfwmacc.vf v12, fa2, v27
1000071c: vsetivli zero, 0x10, e16, m1, ta, ma
10000720: vfwmacc.vf v14, fa3, v27
10000724: vsetivli zero, 0x10, e16, m1, ta, ma
10000728: vfwmacc.vf v16, fa4, v27
1000072c: vsetivli zero, 0x10, e16, m1, ta, ma
10000730: vfwmacc.vf v24, fa0, v27
10000734: flh fa4, -0x62(s2)
10000738: vle16.v v26, (s3)
1000073c: flh fa3, -0x42(s2)
10000740: vfwmacc.vf v22, ft0, v27
10000744: flh fa2, -0x22(s2)
10000748: flh fa1, -0x2(s2)
1000074c: vfwmacc.vf v20, fa5, v27
10000750: flh fa5, 0x1e(s2)
10000754: flh fa0, 0x3e(s2)
10000758: vfwmacc.vf v18, ft1, v27
1000075c: flh ft0, 0x5e(s2)
10000760: flh ft1, 0x7e(s2)
10000764: vsetivli zero, 0x10, e16, m1, ta, ma
10000768: vfwmacc.vf v24, fa4, v26
1000076c: addi s2, s2, 0x200
10000770: vfwmacc.vf v22, fa3, v26
10000774: vfwmacc.vf v20, fa2, v26
10000778: vfwmacc.vf v18, fa1, v26
1000077c: vfwmacc.vf v16, fa5, v26
10000780: vfwmacc.vf v14, fa0, v26
10000784: vfwmacc.vf v12, ft0, v26
10000788: vfwmacc.vf v10, ft1, v26
1000078c: bne s1, t6, 0x100001d8 <main+0x16c>
10000790: j 0x10000128 <main+0xbc>
10000794: rdcycle a2
10000798: ld s0, 0x28(sp)
1000079c: ld s1, 0x20(sp)
100007a0: sub a2, a2, a4
100007a4: ld s2, 0x18(sp)
100007a8: li a0, 0x0
100007ac: ld s3, 0x10(sp)
100007b0: ld s4, 0x8(sp)
100007b4: sd a2, 0x0(a1)
100007b8: addi sp, sp, 0x30
100007bc: ret
