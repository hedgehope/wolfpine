# nbody/ame size=[256, "clustered"] fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x150
10000070: sd ra, 0x148(sp)
10000074: sd s0, 0x140(sp)
10000078: sd s1, 0x138(sp)
1000007c: sd s2, 0x130(sp)
10000080: sd s3, 0x128(sp)
10000084: sd s4, 0x120(sp)
10000088: sd s5, 0x118(sp)
1000008c: sd s6, 0x110(sp)
10000090: sd s7, 0x108(sp)
10000094: sd s8, 0x100(sp)
10000098: sd s9, 0xf8(sp)
1000009c: sd s10, 0xf0(sp)
100000a0: sd s11, 0xe8(sp)
100000a4: fsd fs0, 0xe0(sp)
100000a8: fsd fs1, 0xd8(sp)
100000ac: auipc a1, 0x4
100000b0: lui a2, 0x1
100000b4: auipc a3, 0x4a
100000b8: auipc a4, 0x8
100000bc: auipc a5, 0x0
100000c0: rdcycle a0
100000c4: sd a0, 0x0(sp)
100000c8: lui a6, 0xc0000
100000cc: lui a0, 0x3f800
100000d0: addi s1, a1, 0x39c
100000d4: addi a7, a3, -0x434
100000d8: addi s2, a4, 0x390
100000dc: addi a4, a5, 0x38c
100000e0: fmv.w.x fa5, a6
100000e4: add a1, s1, a2
100000e8: addi a2, a7, 0x8
100000ec: addi a3, s2, 0x8
100000f0: sd a4, 0x10(sp)
100000f4: addi a4, a4, 0x8
100000f8: addi a5, s1, 0x7ff
100000fc: addi a6, a7, 0x7ff
10000100: addi a5, a5, 0x1
10000104: addi a6, a6, 0x409
10000108: flw fa4, -0x8(a2)
1000010c: flw fa3, -0x4(a2)
10000110: flw fa2, 0x0(a2)
10000114: sw a0, 0x8(a4)
10000118: addi a2, a2, 0xc
1000011c: fmul.s fa1, fa3, fa3
10000120: fsw fa4, -0x8(a3)
10000124: fsw fa3, -0x4(a3)
10000128: fsw fa2, 0x0(a3)
1000012c: fmul.s fa0, fa4, fa5
10000130: fsw fa3, -0x400(a5)
10000134: fmul.s fa3, fa3, fa5
10000138: fsw fa4, -0x800(a5)
1000013c: fmadd.s fa4, fa4, fa4, fa1
10000140: fmul.s fa1, fa2, fa5
10000144: fsw fa2, 0x0(a5)
10000148: sw a0, -0x400(a1)
1000014c: addi a5, a5, 0x4
10000150: addi a3, a3, 0xc
10000154: fmadd.s fa4, fa2, fa2, fa4
10000158: fsw fa0, -0x8(a4)
1000015c: fsw fa3, -0x4(a4)
10000160: fsw fa1, 0x0(a4)
10000164: fsw fa4, 0x4(a4)
10000168: fsw fa4, 0x0(a1)
1000016c: addi a1, a1, 0x4
10000170: addi a4, a4, 0x40
10000174: bne a2, a6, 0x10000108 <main+0x9c>
10000178: li a0, 0x0
1000017c: auipc a1, 0x49
10000180: auipc a2, 0x9
10000184: addi s5, sp, 0xe0
10000188: addi s6, sp, 0x18
1000018c: addi a3, s2, 0x8
10000190: sd a3, 0x8(sp)
10000194: li s9, 0x400
10000198: auipc a3, 0x4a
1000019c: fmv.w.x fs0, zero
100001a0: li s11, 0x10
100001a4: li s0, 0xf0
100001a8: addi s7, a1, -0xf4
100001ac: addi s4, a2, -0x100
100001b0: addi s10, a3, 0x6e8
100001b4: lui a1, 0x3f800
100001b8: fmv.w.x fs1, a1
100001bc: mv s3, s4
100001c0: mv s8, a0
100001c4: addi a0, sp, 0x18
100001c8: li a2, 0xc0
100001cc: li a1, 0x0
100001d0: jal 0x100003d0 <memset>
100001d4: li a6, 0x0
100001d8: slli a0, s8, 0x6
100001dc: ld a1, 0x10(sp)
100001e0: add a0, a1, a0
100001e4: li a1, 0x40
100001e8: mx.tload.a td=0 base=a0 stride=a1 fmt=fp32 t=0
100001ec: slli a0, s8, 0x8
100001f0: ld a1, 0x8(sp)
100001f4: mv a2, s8
100001f8: mv a3, s3
100001fc: j 0x10000214 <main+0x1a8>
10000200: addi a6, a4, 0x10
10000204: addi a3, a3, 0x40
10000208: addi a2, a2, -0x10
1000020c: addi a1, a1, 0xc0
10000210: bgeu a4, s0, 0x10000328 <main+0x2bc>
10000214: li a5, 0x0
10000218: mv a4, a6
1000021c: mx.azero ad=0
10000220: slli a6, a6, 0x2
10000224: add a7, a4, a0
10000228: add a6, s1, a6
1000022c: slli a7, a7, 0x2
10000230: mx.tload.b td=1 base=a6 stride=s9 fmt=fp32 t=0
10000234: add a7, s4, a7
10000238: mx.mma ad=0 ta=0 tb=1 in=fp32 acc=fp32
1000023c: mx.astore as=0 base=a7 stride=s9 fmt=fp32
10000240: flw fa5, 0x40(s10)
10000244: mv a6, a2
10000248: mv a7, a3
1000024c: j 0x10000294 <main+0x228>
10000250: slli t0, a5, 0x2
10000254: slli t1, a5, 0x4
10000258: sub t0, t1, t0
1000025c: add t0, s6, t0
10000260: flw fa1, 0x0(t0)
10000264: flw fa0, 0x4(t0)
10000268: flw ft0, 0x8(t0)
1000026c: fadd.s fa2, fa2, fa1
10000270: fadd.s fa3, fa3, fa0
10000274: addi a5, a5, 0x1
10000278: addi a7, a7, 0x400
1000027c: fadd.s fa4, fa4, ft0
10000280: fsw fa2, 0x0(t0)
10000284: fsw fa3, 0x4(t0)
10000288: fsw fa4, 0x8(t0)
1000028c: addi a6, a6, 0x1
10000290: beq a5, s11, 0x10000200 <main+0x194>
10000294: or t0, a5, s8
10000298: slli t1, t0, 0x2
1000029c: slli t0, t0, 0x4
100002a0: sub t0, t0, t1
100002a4: add t0, s2, t0
100002a8: flw fa1, 0x0(t0)
100002ac: flw fa0, 0x4(t0)
100002b0: flw ft0, 0x8(t0)
100002b4: li t0, 0x10
100002b8: mv t1, a1
100002bc: mv t2, a7
100002c0: fmv.s fa4, fs0
100002c4: fmv.s fa3, fs0
100002c8: fmv.s fa2, fs0
100002cc: j 0x100002e0 <main+0x274>
100002d0: addi t2, t2, 0x4
100002d4: addi t0, t0, -0x1
100002d8: addi t1, t1, 0xc
100002dc: beqz t0, 0x10000250 <main+0x1e4>
100002e0: add t3, a6, t0
100002e4: beq t3, s11, 0x100002d0 <main+0x264>
100002e8: flw ft1, 0x0(t2)
100002ec: fmax.s ft1, ft1, fs0
100002f0: fadd.s ft1, fa5, ft1
100002f4: fsqrt.s ft2, ft1
100002f8: flw ft3, -0x8(t1)
100002fc: fmul.s ft1, ft2, ft1
10000300: flw ft2, -0x4(t1)
10000304: flw ft4, 0x0(t1)
10000308: fsub.s ft3, ft3, fa1
1000030c: fdiv.s ft1, fs1, ft1
10000310: fmadd.s fa2, ft1, ft3, fa2
10000314: fsub.s ft2, ft2, fa0
10000318: fmadd.s fa3, ft1, ft2, fa3
1000031c: fsub.s ft2, ft4, ft0
10000320: fmadd.s fa4, ft1, ft2, fa4
10000324: j 0x100002d0 <main+0x264>
10000328: mv a0, s7
1000032c: addi a1, sp, 0x20
10000330: flw fa5, -0x8(a1)
10000334: flw fa4, -0x4(a1)
10000338: flw fa3, 0x0(a1)
1000033c: addi a1, a1, 0xc
10000340: fsw fa5, -0x8(a0)
10000344: fsw fa4, -0x4(a0)
10000348: fsw fa3, 0x0(a0)
1000034c: addi a0, a0, 0xc
10000350: bne a1, s5, 0x10000330 <main+0x2c4>
10000354: addi a0, s8, 0x10
10000358: lui a1, 0x4
1000035c: add s3, s3, a1
10000360: addi s7, s7, 0xc0
10000364: bltu s8, s0, 0x100001c0 <main+0x154>
10000368: mx.azero ad=0
1000036c: mx.azero ad=1
10000370: mx.azero ad=2
10000374: mx.azero ad=3
10000378: rdcycle a0
1000037c: ld a1, 0x0(sp)
10000380: sub a0, a0, a1
10000384: sd a0, 0x0(s10)
10000388: li a0, 0x0
1000038c: ld ra, 0x148(sp)
10000390: ld s0, 0x140(sp)
10000394: ld s1, 0x138(sp)
10000398: ld s2, 0x130(sp)
1000039c: ld s3, 0x128(sp)
100003a0: ld s4, 0x120(sp)
100003a4: ld s5, 0x118(sp)
100003a8: ld s6, 0x110(sp)
100003ac: ld s7, 0x108(sp)
100003b0: ld s8, 0x100(sp)
100003b4: ld s9, 0xf8(sp)
100003b8: ld s10, 0xf0(sp)
100003bc: ld s11, 0xe8(sp)
100003c0: fld fs0, 0xe0(sp)
100003c4: fld fs1, 0xd8(sp)
100003c8: addi sp, sp, 0x150
100003cc: ret
# <memset>
100003d0: beqz a2, 0x100003ec <memset+0x1c>
100003d4: add a2, a0, a2
100003d8: mv a3, a0
100003dc: addi a4, a3, 0x1
100003e0: sb a1, 0x0(a3)
100003e4: mv a3, a4
100003e8: bne a4, a2, 0x100003dc <memset+0xc>
100003ec: ret
