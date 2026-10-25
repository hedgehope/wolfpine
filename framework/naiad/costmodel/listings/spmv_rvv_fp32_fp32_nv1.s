# spmv/rvv size=[256,0.25,1] fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x20
10000070: sd s0, 0x18(sp)
10000074: sd s1, 0x10(sp)
10000078: sd s2, 0x8(sp)
1000007c: auipc a1, 0x10
10000080: auipc a2, 0x0
10000084: addi a1, a1, 0x684
10000088: auipc a3, 0x10
1000008c: li a0, 0x0
10000090: vsetivli zero, 0x10, e32, m2, ta, ma
10000094: vmv.v.i v8, 0x0
10000098: addi a2, a2, 0x240
1000009c: addi a4, a1, 0x40
100000a0: addi a5, a3, 0x238
100000a4: li a6, 0xc
100000a8: li a7, 0x10
100000ac: rdcycle a3
100000b0: vmv.s.x v10, zero
100000b4: j 0x100000bc <main+0x50>
100000b8: beq a0, a7, 0x100001ec <main+0x180>
100000bc: mv t1, a0
100000c0: slli t0, a0, 0x2
100000c4: addi a0, a0, 0x1
100000c8: add t0, a4, t0
100000cc: slli t2, a0, 0x2
100000d0: li s0, 0x0
100000d4: add t2, a4, t2
100000d8: slli t1, t1, 0x4
100000dc: lw t0, 0x0(t0)
100000e0: lw t2, 0x0(t2)
100000e4: not t3, t0
100000e8: slli t4, t0, 0x2
100000ec: add t6, t3, t2
100000f0: add t5, a1, t4
100000f4: slli t6, t6, 0x20
100000f8: addi t3, t5, 0xc0
100000fc: srli t5, t6, 0x1e
10000100: add t4, a1, t4
10000104: add t5, t4, t5
10000108: slli t4, t0, 0xa
1000010c: add t4, a2, t4
10000110: addi t5, t5, 0xc4
10000114: j 0x10000168 <main+0xfc>
10000118: vsetivli zero, 0x10, e32, m2, ta, ma
1000011c: vfredusum.vs v11, v18, v10
10000120: add s0, t6, t1
10000124: addi t4, t4, 0x100
10000128: slli s0, s0, 0x2
1000012c: add s0, a1, s0
10000130: addi s1, s0, 0x1c0
10000134: addi s2, s0, 0x1c4
10000138: vfredusum.vs v14, v14, v10
1000013c: vfredusum.vs v16, v16, v10
10000140: vfredusum.vs v12, v12, v10
10000144: vsetivli zero, 0x1, e32, m1, ta, ma
10000148: vse32.v v11, (s1)
1000014c: addi s1, s0, 0x1c8
10000150: vse32.v v16, (s2)
10000154: vse32.v v14, (s1)
10000158: addi s1, s0, 0x1cc
1000015c: addi s0, t6, 0x4
10000160: vse32.v v12, (s1)
10000164: bgeu t6, a6, 0x100000b8 <main+0x4c>
10000168: vsetivli zero, 0x1, e8, m1, ta, ma
1000016c: mv t6, s0
10000170: vmv2r.v v12, v8
10000174: mv s0, t4
10000178: mv s1, t3
1000017c: vmv2r.v v14, v8
10000180: vmv2r.v v16, v8
10000184: vmv2r.v v18, v8
10000188: bge t0, t2, 0x10000118 <main+0xac>
1000018c: lw s2, 0x0(s1)
10000190: addi s1, s1, 0x4
10000194: slli s2, s2, 0x6
10000198: vsetivli zero, 0x10, e32, m2, ta, ma
1000019c: add s2, a5, s2
100001a0: vle32.v v22, (s0)
100001a4: vle32.v v20, (s2)
100001a8: addi s2, s0, 0x40
100001ac: vle32.v v24, (s2)
100001b0: addi s2, s0, 0x80
100001b4: vle32.v v26, (s2)
100001b8: addi s2, s0, 0xc0
100001bc: addi s0, s0, 0x400
100001c0: vle32.v v28, (s2)
100001c4: vsetivli zero, 0x10, e32, m2, ta, ma
100001c8: vfmacc.vv v18, v20, v22
100001cc: vsetivli zero, 0x10, e32, m2, ta, ma
100001d0: vfmacc.vv v16, v20, v24
100001d4: vsetivli zero, 0x10, e32, m2, ta, ma
100001d8: vfmacc.vv v14, v20, v26
100001dc: vsetivli zero, 0x10, e32, m2, ta, ma
100001e0: vfmacc.vv v12, v20, v28
100001e4: bne s1, t5, 0x1000018c <main+0x120>
100001e8: j 0x10000118 <main+0xac>
100001ec: rdcycle a2
100001f0: ld s0, 0x18(sp)
100001f4: ld s1, 0x10(sp)
100001f8: sub a2, a2, a3
100001fc: ld s2, 0x8(sp)
10000200: li a0, 0x0
10000204: sd a2, 0x0(a1)
10000208: addi sp, sp, 0x20
1000020c: ret
