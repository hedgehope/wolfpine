# spmv/rvv size=[256, 0.25] bf16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x30
10000070: sd s0, 0x28(sp)
10000074: sd s1, 0x20(sp)
10000078: sd s2, 0x18(sp)
1000007c: sd s3, 0x10(sp)
10000080: sd s4, 0x8(sp)
10000084: auipc a1, 0xe
10000088: auipc a4, 0x8
1000008c: addi a1, a1, 0x73c
10000090: auipc a3, 0x0
10000094: addi a5, a4, 0x6f8
10000098: auipc a4, 0xa
1000009c: li a0, 0x0
100000a0: vsetivli zero, 0x10, e32, m2, ta, ma
100000a4: addi a2, a1, 0x40
100000a8: addi a3, a3, 0x6f0
100000ac: vmv.v.i v8, 0x0
100000b0: addi a6, a4, 0x728
100000b4: li a7, 0x10
100000b8: rdcycle a4
100000bc: j 0x100000c4 <main+0x58>
100000c0: beq a0, a7, 0x100006b4 <main+0x648>
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
100001fc: flh fa1, 0x0(s2)
10000200: flh fa0, 0x20(s2)
10000204: flh ft0, 0x40(s2)
10000208: flh ft1, 0x60(s2)
1000020c: vsetivli zero, 0x10, e16, m1, ta, ma
10000210: vle16.v v26, (s3)
10000214: vsetivli zero, 0x10, e16, m1, ta, ma
10000218: vfwmaccbf16.vf v24, fa5, v26
1000021c: flh fa5, -0x7e(s2)
10000220: vle16.v v27, (s4)
10000224: addi s4, s3, 0x40
10000228: vfwmaccbf16.vf v22, fa4, v26
1000022c: flh fa4, -0x5e(s2)
10000230: vfwmaccbf16.vf v20, fa3, v26
10000234: flh fa3, -0x3e(s2)
10000238: vfwmaccbf16.vf v18, fa2, v26
1000023c: flh fa2, -0x1e(s2)
10000240: vfwmaccbf16.vf v16, fa1, v26
10000244: flh fa1, 0x2(s2)
10000248: vfwmaccbf16.vf v14, fa0, v26
1000024c: flh fa0, 0x22(s2)
10000250: vfwmaccbf16.vf v12, ft0, v26
10000254: flh ft0, 0x42(s2)
10000258: vfwmaccbf16.vf v10, ft1, v26
1000025c: flh ft1, 0x62(s2)
10000260: vsetivli zero, 0x10, e16, m1, ta, ma
10000264: vfwmaccbf16.vf v24, fa5, v27
10000268: flh fa5, -0x7c(s2)
1000026c: vle16.v v26, (s4)
10000270: addi s4, s3, 0x60
10000274: vfwmaccbf16.vf v22, fa4, v27
10000278: flh fa4, -0x5c(s2)
1000027c: vfwmaccbf16.vf v20, fa3, v27
10000280: flh fa3, -0x3c(s2)
10000284: vfwmaccbf16.vf v18, fa2, v27
10000288: flh fa2, -0x1c(s2)
1000028c: vfwmaccbf16.vf v16, fa1, v27
10000290: flh fa1, 0x4(s2)
10000294: vfwmaccbf16.vf v14, fa0, v27
10000298: flh fa0, 0x24(s2)
1000029c: vfwmaccbf16.vf v12, ft0, v27
100002a0: flh ft0, 0x44(s2)
100002a4: vfwmaccbf16.vf v10, ft1, v27
100002a8: flh ft1, 0x64(s2)
100002ac: vsetivli zero, 0x10, e16, m1, ta, ma
100002b0: vfwmaccbf16.vf v24, fa5, v26
100002b4: flh fa5, -0x7a(s2)
100002b8: vle16.v v27, (s4)
100002bc: addi s4, s3, 0x80
100002c0: vfwmaccbf16.vf v22, fa4, v26
100002c4: flh fa4, -0x5a(s2)
100002c8: vfwmaccbf16.vf v20, fa3, v26
100002cc: flh fa3, -0x3a(s2)
100002d0: vfwmaccbf16.vf v18, fa2, v26
100002d4: flh fa2, -0x1a(s2)
100002d8: vfwmaccbf16.vf v16, fa1, v26
100002dc: flh fa1, 0x6(s2)
100002e0: vfwmaccbf16.vf v14, fa0, v26
100002e4: flh fa0, 0x26(s2)
100002e8: vfwmaccbf16.vf v12, ft0, v26
100002ec: flh ft0, 0x46(s2)
100002f0: vfwmaccbf16.vf v10, ft1, v26
100002f4: flh ft1, 0x66(s2)
100002f8: vsetivli zero, 0x10, e16, m1, ta, ma
100002fc: vfwmaccbf16.vf v24, fa5, v27
10000300: flh fa5, -0x78(s2)
10000304: vle16.v v26, (s4)
10000308: addi s4, s3, 0xa0
1000030c: vfwmaccbf16.vf v22, fa4, v27
10000310: flh fa4, -0x58(s2)
10000314: vfwmaccbf16.vf v20, fa3, v27
10000318: flh fa3, -0x38(s2)
1000031c: vfwmaccbf16.vf v18, fa2, v27
10000320: flh fa2, -0x18(s2)
10000324: vfwmaccbf16.vf v16, fa1, v27
10000328: flh fa1, 0x8(s2)
1000032c: vfwmaccbf16.vf v14, fa0, v27
10000330: flh fa0, 0x28(s2)
10000334: vfwmaccbf16.vf v12, ft0, v27
10000338: flh ft0, 0x48(s2)
1000033c: vfwmaccbf16.vf v10, ft1, v27
10000340: flh ft1, 0x68(s2)
10000344: vsetivli zero, 0x10, e16, m1, ta, ma
10000348: vfwmaccbf16.vf v24, fa5, v26
1000034c: flh fa5, -0x76(s2)
10000350: vle16.v v27, (s4)
10000354: addi s4, s3, 0xc0
10000358: vfwmaccbf16.vf v22, fa4, v26
1000035c: flh fa4, -0x56(s2)
10000360: vfwmaccbf16.vf v20, fa3, v26
10000364: flh fa3, -0x36(s2)
10000368: vfwmaccbf16.vf v18, fa2, v26
1000036c: flh fa2, -0x16(s2)
10000370: vfwmaccbf16.vf v16, fa1, v26
10000374: flh fa1, 0xa(s2)
10000378: vfwmaccbf16.vf v14, fa0, v26
1000037c: flh fa0, 0x2a(s2)
10000380: vfwmaccbf16.vf v12, ft0, v26
10000384: flh ft0, 0x4a(s2)
10000388: vfwmaccbf16.vf v10, ft1, v26
1000038c: flh ft1, 0x6a(s2)
10000390: vsetivli zero, 0x10, e16, m1, ta, ma
10000394: vfwmaccbf16.vf v24, fa5, v27
10000398: flh fa5, -0x74(s2)
1000039c: vle16.v v26, (s4)
100003a0: addi s4, s3, 0xe0
100003a4: vfwmaccbf16.vf v22, fa4, v27
100003a8: flh fa4, -0x54(s2)
100003ac: vfwmaccbf16.vf v20, fa3, v27
100003b0: flh fa3, -0x34(s2)
100003b4: vfwmaccbf16.vf v18, fa2, v27
100003b8: flh fa2, -0x14(s2)
100003bc: vfwmaccbf16.vf v16, fa1, v27
100003c0: flh fa1, 0xc(s2)
100003c4: vfwmaccbf16.vf v14, fa0, v27
100003c8: flh fa0, 0x2c(s2)
100003cc: vfwmaccbf16.vf v12, ft0, v27
100003d0: flh ft0, 0x4c(s2)
100003d4: vfwmaccbf16.vf v10, ft1, v27
100003d8: flh ft1, 0x6c(s2)
100003dc: vsetivli zero, 0x10, e16, m1, ta, ma
100003e0: vfwmaccbf16.vf v24, fa5, v26
100003e4: flh fa5, -0x72(s2)
100003e8: vle16.v v27, (s4)
100003ec: addi s4, s3, 0x100
100003f0: vfwmaccbf16.vf v22, fa4, v26
100003f4: flh fa4, -0x52(s2)
100003f8: vfwmaccbf16.vf v20, fa3, v26
100003fc: flh fa3, -0x32(s2)
10000400: vfwmaccbf16.vf v18, fa2, v26
10000404: flh fa2, -0x12(s2)
10000408: vfwmaccbf16.vf v16, fa1, v26
1000040c: flh fa1, 0xe(s2)
10000410: vfwmaccbf16.vf v14, fa0, v26
10000414: flh fa0, 0x2e(s2)
10000418: vfwmaccbf16.vf v12, ft0, v26
1000041c: flh ft0, 0x4e(s2)
10000420: vfwmaccbf16.vf v10, ft1, v26
10000424: flh ft1, 0x6e(s2)
10000428: vsetivli zero, 0x10, e16, m1, ta, ma
1000042c: vfwmaccbf16.vf v24, fa5, v27
10000430: flh fa5, -0x70(s2)
10000434: vle16.v v26, (s4)
10000438: addi s4, s3, 0x120
1000043c: vfwmaccbf16.vf v22, fa4, v27
10000440: flh fa4, -0x50(s2)
10000444: vfwmaccbf16.vf v20, fa3, v27
10000448: flh fa3, -0x30(s2)
1000044c: vfwmaccbf16.vf v18, fa2, v27
10000450: flh fa2, -0x10(s2)
10000454: vfwmaccbf16.vf v16, fa1, v27
10000458: flh fa1, 0x10(s2)
1000045c: vfwmaccbf16.vf v14, fa0, v27
10000460: flh fa0, 0x30(s2)
10000464: vfwmaccbf16.vf v12, ft0, v27
10000468: flh ft0, 0x50(s2)
1000046c: vfwmaccbf16.vf v10, ft1, v27
10000470: flh ft1, 0x70(s2)
10000474: vsetivli zero, 0x10, e16, m1, ta, ma
10000478: vfwmaccbf16.vf v24, fa5, v26
1000047c: flh fa5, -0x6e(s2)
10000480: vle16.v v27, (s4)
10000484: addi s4, s3, 0x140
10000488: vfwmaccbf16.vf v22, fa4, v26
1000048c: flh fa4, -0x4e(s2)
10000490: vfwmaccbf16.vf v20, fa3, v26
10000494: flh fa3, -0x2e(s2)
10000498: vfwmaccbf16.vf v18, fa2, v26
1000049c: flh fa2, -0xe(s2)
100004a0: vfwmaccbf16.vf v16, fa1, v26
100004a4: flh fa1, 0x12(s2)
100004a8: vfwmaccbf16.vf v14, fa0, v26
100004ac: flh fa0, 0x32(s2)
100004b0: vfwmaccbf16.vf v12, ft0, v26
100004b4: flh ft0, 0x52(s2)
100004b8: vfwmaccbf16.vf v10, ft1, v26
100004bc: flh ft1, 0x72(s2)
100004c0: vsetivli zero, 0x10, e16, m1, ta, ma
100004c4: vfwmaccbf16.vf v24, fa5, v27
100004c8: flh fa5, -0x6c(s2)
100004cc: vle16.v v26, (s4)
100004d0: addi s4, s3, 0x160
100004d4: vfwmaccbf16.vf v22, fa4, v27
100004d8: flh fa4, -0x4c(s2)
100004dc: vfwmaccbf16.vf v20, fa3, v27
100004e0: flh fa3, -0x2c(s2)
100004e4: vfwmaccbf16.vf v18, fa2, v27
100004e8: flh fa2, -0xc(s2)
100004ec: vfwmaccbf16.vf v16, fa1, v27
100004f0: flh fa1, 0x14(s2)
100004f4: vfwmaccbf16.vf v14, fa0, v27
100004f8: flh fa0, 0x34(s2)
100004fc: vfwmaccbf16.vf v12, ft0, v27
10000500: flh ft0, 0x54(s2)
10000504: vfwmaccbf16.vf v10, ft1, v27
10000508: flh ft1, 0x74(s2)
1000050c: vsetivli zero, 0x10, e16, m1, ta, ma
10000510: vfwmaccbf16.vf v24, fa5, v26
10000514: flh fa5, -0x6a(s2)
10000518: vle16.v v27, (s4)
1000051c: addi s4, s3, 0x180
10000520: vfwmaccbf16.vf v22, fa4, v26
10000524: flh fa4, -0x4a(s2)
10000528: vfwmaccbf16.vf v20, fa3, v26
1000052c: flh fa3, -0x2a(s2)
10000530: vfwmaccbf16.vf v18, fa2, v26
10000534: flh fa2, -0xa(s2)
10000538: vfwmaccbf16.vf v16, fa1, v26
1000053c: flh fa1, 0x16(s2)
10000540: vfwmaccbf16.vf v14, fa0, v26
10000544: flh fa0, 0x36(s2)
10000548: vfwmaccbf16.vf v12, ft0, v26
1000054c: flh ft0, 0x56(s2)
10000550: vfwmaccbf16.vf v10, ft1, v26
10000554: flh ft1, 0x76(s2)
10000558: vsetivli zero, 0x10, e16, m1, ta, ma
1000055c: vfwmaccbf16.vf v24, fa5, v27
10000560: flh fa5, -0x68(s2)
10000564: vle16.v v26, (s4)
10000568: addi s4, s3, 0x1a0
1000056c: vfwmaccbf16.vf v22, fa4, v27
10000570: flh fa4, -0x48(s2)
10000574: vfwmaccbf16.vf v20, fa3, v27
10000578: flh fa3, -0x28(s2)
1000057c: vfwmaccbf16.vf v18, fa2, v27
10000580: flh fa2, -0x8(s2)
10000584: vfwmaccbf16.vf v16, fa1, v27
10000588: flh fa1, 0x18(s2)
1000058c: vfwmaccbf16.vf v14, fa0, v27
10000590: flh fa0, 0x38(s2)
10000594: vfwmaccbf16.vf v12, ft0, v27
10000598: flh ft0, 0x58(s2)
1000059c: vfwmaccbf16.vf v10, ft1, v27
100005a0: flh ft1, 0x78(s2)
100005a4: vsetivli zero, 0x10, e16, m1, ta, ma
100005a8: vfwmaccbf16.vf v24, fa5, v26
100005ac: flh fa5, -0x66(s2)
100005b0: vle16.v v27, (s4)
100005b4: addi s4, s3, 0x1c0
100005b8: vfwmaccbf16.vf v22, fa4, v26
100005bc: flh fa4, -0x46(s2)
100005c0: addi s3, s3, 0x1e0
100005c4: vfwmaccbf16.vf v20, fa3, v26
100005c8: flh fa3, -0x26(s2)
100005cc: vfwmaccbf16.vf v18, fa2, v26
100005d0: flh fa2, -0x6(s2)
100005d4: vfwmaccbf16.vf v16, fa1, v26
100005d8: flh fa1, 0x1a(s2)
100005dc: vfwmaccbf16.vf v14, fa0, v26
100005e0: flh fa0, 0x3a(s2)
100005e4: vfwmaccbf16.vf v12, ft0, v26
100005e8: flh ft0, 0x5a(s2)
100005ec: vfwmaccbf16.vf v10, ft1, v26
100005f0: flh ft1, 0x7a(s2)
100005f4: vsetivli zero, 0x10, e16, m1, ta, ma
100005f8: vfwmaccbf16.vf v24, fa5, v27
100005fc: flh fa5, 0x1c(s2)
10000600: vle16.v v26, (s4)
10000604: vfwmaccbf16.vf v22, fa4, v27
10000608: flh fa4, 0x3c(s2)
1000060c: vfwmaccbf16.vf v20, fa3, v27
10000610: flh fa3, 0x5c(s2)
10000614: vfwmaccbf16.vf v18, fa2, v27
10000618: flh fa2, 0x7c(s2)
1000061c: vfwmaccbf16.vf v10, ft1, v27
10000620: flh ft1, -0x4(s2)
10000624: vfwmaccbf16.vf v12, ft0, v27
10000628: flh ft0, -0x44(s2)
1000062c: vfwmaccbf16.vf v14, fa0, v27
10000630: flh fa0, -0x64(s2)
10000634: vfwmaccbf16.vf v16, fa1, v27
10000638: flh fa1, -0x24(s2)
1000063c: vsetivli zero, 0x10, e16, m1, ta, ma
10000640: vfwmaccbf16.vf v10, fa2, v26
10000644: flh fa2, -0x2(s2)
10000648: vle16.v v27, (s3)
1000064c: vfwmaccbf16.vf v12, fa3, v26
10000650: flh fa3, -0x22(s2)
10000654: vfwmaccbf16.vf v14, fa4, v26
10000658: flh fa4, -0x42(s2)
1000065c: vfwmaccbf16.vf v16, fa5, v26
10000660: flh fa5, -0x62(s2)
10000664: vfwmaccbf16.vf v24, fa0, v26
10000668: flh fa0, 0x3e(s2)
1000066c: vfwmaccbf16.vf v22, ft0, v26
10000670: flh ft0, 0x5e(s2)
10000674: vfwmaccbf16.vf v20, fa1, v26
10000678: flh fa1, 0x1e(s2)
1000067c: vfwmaccbf16.vf v18, ft1, v26
10000680: flh ft1, 0x7e(s2)
10000684: vsetivli zero, 0x10, e16, m1, ta, ma
10000688: vfwmaccbf16.vf v24, fa5, v27
1000068c: addi s2, s2, 0x200
10000690: vfwmaccbf16.vf v22, fa4, v27
10000694: vfwmaccbf16.vf v20, fa3, v27
10000698: vfwmaccbf16.vf v18, fa2, v27
1000069c: vfwmaccbf16.vf v16, fa1, v27
100006a0: vfwmaccbf16.vf v14, fa0, v27
100006a4: vfwmaccbf16.vf v12, ft0, v27
100006a8: vfwmaccbf16.vf v10, ft1, v27
100006ac: bne s1, t6, 0x100001d8 <main+0x16c>
100006b0: j 0x10000128 <main+0xbc>
100006b4: rdcycle a2
100006b8: ld s0, 0x28(sp)
100006bc: ld s1, 0x20(sp)
100006c0: sub a2, a2, a4
100006c4: ld s2, 0x18(sp)
100006c8: li a0, 0x0
100006cc: ld s3, 0x10(sp)
100006d0: ld s4, 0x8(sp)
100006d4: sd a2, 0x0(a1)
100006d8: addi sp, sp, 0x30
100006dc: ret
