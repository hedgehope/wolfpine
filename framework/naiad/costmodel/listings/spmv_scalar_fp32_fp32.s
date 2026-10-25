# spmv/scalar size=[256, 0.25] fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x30
10000070: sd s0, 0x28(sp)
10000074: sd s1, 0x20(sp)
10000078: sd s2, 0x18(sp)
1000007c: sd s3, 0x10(sp)
10000080: sd s4, 0x8(sp)
10000084: li a1, 0x0
10000088: rdcycle a0
1000008c: auipc a2, 0x18
10000090: auipc a3, 0x0
10000094: fmv.w.x fa5, zero
10000098: auipc a5, 0x10
1000009c: addi a2, a2, 0x2b4
100000a0: addi a3, a3, 0x270
100000a4: lw t0, 0x40(a2)
100000a8: addi a4, a2, 0xc0
100000ac: addi a5, a5, 0x268
100000b0: auipc a6, 0x14
100000b4: addi a6, a6, 0x290
100000b8: li a7, 0x10
100000bc: j 0x100000c4 <main+0x58>
100000c0: beq a1, a7, 0x1000022c <main+0x1c0>
100000c4: li t1, 0x0
100000c8: mv t2, t0
100000cc: slli t3, a1, 0x8
100000d0: addi a1, a1, 0x1
100000d4: slli t4, t0, 0x2
100000d8: slli t0, a1, 0x2
100000dc: add t0, a2, t0
100000e0: lw t0, 0x40(t0)
100000e4: add t4, a4, t4
100000e8: slli t5, t2, 0xa
100000ec: add t5, a3, t5
100000f0: addi t5, t5, 0x20
100000f4: slli t6, t0, 0x2
100000f8: add t6, a4, t6
100000fc: j 0x1000010c <main+0xa0>
10000100: addi t1, t1, 0x1
10000104: addi t5, t5, 0x40
10000108: beq t1, a7, 0x100000c0 <main+0x54>
1000010c: li s0, 0x0
10000110: slli s1, t1, 0x4
10000114: add s1, s1, t3
10000118: j 0x10000134 <main+0xc8>
1000011c: add s2, s0, s1
10000120: slli s2, s2, 0x2
10000124: add s2, a6, s2
10000128: addi s0, s0, 0x1
1000012c: fsw fa4, 0x0(s2)
10000130: beq s0, a7, 0x10000100 <main+0x94>
10000134: fmv.s fa4, fa5
10000138: bge t2, t0, 0x1000011c <main+0xb0>
1000013c: mv s2, t5
10000140: mv s3, t4
10000144: fmv.s fa4, fa5
10000148: lw s4, 0x0(s3)
1000014c: slli s4, s4, 0x8
10000150: add s4, s4, s0
10000154: slli s4, s4, 0x2
10000158: add s4, a5, s4
1000015c: flw fa3, 0x0(s4)
10000160: flw fa2, -0x20(s2)
10000164: flw fa1, -0x1c(s2)
10000168: flw fa0, -0x18(s2)
1000016c: flw ft0, -0x14(s2)
10000170: flw ft1, 0x40(s4)
10000174: flw ft2, 0x80(s4)
10000178: fmadd.s fa4, fa2, fa3, fa4
1000017c: flw fa3, 0xc0(s4)
10000180: fmadd.s fa4, fa1, ft1, fa4
10000184: fmadd.s fa4, fa0, ft2, fa4
10000188: flw fa2, -0x10(s2)
1000018c: flw fa1, -0xc(s2)
10000190: flw fa0, -0x8(s2)
10000194: flw ft1, -0x4(s2)
10000198: flw ft2, 0x100(s4)
1000019c: fmadd.s fa4, ft0, fa3, fa4
100001a0: flw fa3, 0x140(s4)
100001a4: flw ft0, 0x180(s4)
100001a8: fmadd.s fa4, fa2, ft2, fa4
100001ac: flw fa2, 0x1c0(s4)
100001b0: fmadd.s fa4, fa1, fa3, fa4
100001b4: fmadd.s fa4, fa0, ft0, fa4
100001b8: flw fa3, 0x0(s2)
100001bc: flw fa1, 0x4(s2)
100001c0: flw fa0, 0x8(s2)
100001c4: flw ft0, 0xc(s2)
100001c8: flw ft2, 0x200(s4)
100001cc: fmadd.s fa4, ft1, fa2, fa4
100001d0: flw fa2, 0x240(s4)
100001d4: flw ft1, 0x280(s4)
100001d8: fmadd.s fa4, fa3, ft2, fa4
100001dc: flw fa3, 0x2c0(s4)
100001e0: fmadd.s fa4, fa1, fa2, fa4
100001e4: fmadd.s fa4, fa0, ft1, fa4
100001e8: flw fa2, 0x10(s2)
100001ec: flw fa1, 0x14(s2)
100001f0: flw fa0, 0x18(s2)
100001f4: flw ft1, 0x1c(s2)
100001f8: flw ft2, 0x300(s4)
100001fc: fmadd.s fa4, ft0, fa3, fa4
10000200: flw fa3, 0x340(s4)
10000204: flw ft0, 0x380(s4)
10000208: fmadd.s fa4, fa2, ft2, fa4
1000020c: flw fa2, 0x3c0(s4)
10000210: fmadd.s fa4, fa1, fa3, fa4
10000214: fmadd.s fa4, fa0, ft0, fa4
10000218: addi s3, s3, 0x4
1000021c: fmadd.s fa4, ft1, fa2, fa4
10000220: addi s2, s2, 0x400
10000224: bne s3, t6, 0x10000148 <main+0xdc>
10000228: j 0x1000011c <main+0xb0>
1000022c: rdcycle a1
10000230: sub a1, a1, a0
10000234: sd a1, 0x0(a2)
10000238: li a0, 0x0
1000023c: ld s0, 0x28(sp)
10000240: ld s1, 0x20(sp)
10000244: ld s2, 0x18(sp)
10000248: ld s3, 0x10(sp)
1000024c: ld s4, 0x8(sp)
10000250: addi sp, sp, 0x30
10000254: ret
