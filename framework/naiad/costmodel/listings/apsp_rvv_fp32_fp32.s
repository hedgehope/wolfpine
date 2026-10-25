# apsp/rvv size=64 fp32/fp32 cflags='' design=naiad-dwarfs.yaml
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
10000094: auipc a2, 0x0
10000098: li a0, 0x0
1000009c: li a1, 0x500
100000a0: addi a3, a2, 0x22c
100000a4: li a4, 0x40
100000a8: li a5, 0x3b
100000ac: li a6, 0x3c
100000b0: rdcycle a2
100000b4: j 0x100000c4 <main+0x58>
100000b8: addi a1, a1, 0x4
100000bc: addi a0, a0, 0x1
100000c0: beq a0, a4, 0x100001f4 <main+0x188>
100000c4: slli t0, a0, 0x8
100000c8: slli t1, a0, 0x2
100000cc: li a7, 0x0
100000d0: add t0, a3, t0
100000d4: add t1, a3, t1
100000d8: addi t2, a0, 0x40
100000dc: j 0x100000e8 <main+0x7c>
100000e0: add a7, t6, a7
100000e4: bgeu a7, a4, 0x100000b8 <main+0x4c>
100000e8: slli s0, a7, 0x2
100000ec: sub t5, a4, a7
100000f0: add s1, t0, s0
100000f4: vsetvli t6, t5, e32, m4, ta, ma
100000f8: sub s2, a0, a7
100000fc: sub s3, t2, a7
10000100: li t3, 0x0
10000104: li t4, 0x0
10000108: vle32.v v8, (s1)
1000010c: add s1, a3, s0
10000110: vsetvli zero, t5, e32, m4, ta, ma
10000114: slli s2, s2, 0x2
10000118: slli s3, s3, 0x2
1000011c: mv s4, a3
10000120: flw fa5, 0x0(t1)
10000124: flw fa4, 0x100(t1)
10000128: vle32.v v12, (s1)
1000012c: addi s1, s1, 0x100
10000130: vle32.v v16, (s1)
10000134: ori s1, a7, 0x80
10000138: vsetvli zero, t5, e32, m4, ta, ma
1000013c: j 0x1000017c <main+0x110>
10000140: vsetvli zero, t5, e32, m4, ta, ma
10000144: vfadd.vf v28, v8, fa3
10000148: addi t4, t4, 0x4
1000014c: addi t3, t3, 0x100
10000150: addi s4, s4, 0x400
10000154: vfadd.vf v4, v8, fa2
10000158: vsetvli zero, t5, e32, m4, ta, ma
1000015c: vfmin.vv v20, v20, v28
10000160: vsetvli zero, t5, e32, m4, ta, ma
10000164: vfmin.vv v24, v24, v4
10000168: vsetvli zero, t5, e32, m4, ta, ma
1000016c: vse32.v v20, (s5)
10000170: addi s5, t4, -0x4
10000174: vse32.v v24, (s6)
10000178: bgeu s5, a6, 0x100000e0 <main+0x74>
1000017c: or s5, t3, s1
10000180: vfadd.vf v28, v8, fa5
10000184: slli s5, s5, 0x2
10000188: add s5, a3, s5
1000018c: add s7, s5, s2
10000190: add s8, s5, s3
10000194: addi s6, s5, 0x100
10000198: vle32.v v20, (s5)
1000019c: flw fa3, 0x0(s7)
100001a0: flw fa2, 0x0(s8)
100001a4: add s7, s4, s0
100001a8: addi s8, s7, 0x100
100001ac: vfadd.vf v4, v8, fa4
100001b0: vle32.v v24, (s6)
100001b4: vsetvli zero, t5, e32, m4, ta, ma
100001b8: vfmin.vv v12, v12, v28
100001bc: vsetvli zero, t5, e32, m4, ta, ma
100001c0: vfmin.vv v16, v16, v4
100001c4: vsetvli zero, t5, e32, m4, ta, ma
100001c8: vse32.v v12, (s7)
100001cc: vse32.v v16, (s8)
100001d0: bltu a5, t4, 0x10000140 <main+0xd4>
100001d4: addi s8, s7, 0x400
100001d8: addi s7, s7, 0x500
100001dc: vle32.v v12, (s8)
100001e0: add s8, s4, a1
100001e4: flw fa5, -0x100(s8)
100001e8: flw fa4, 0x0(s8)
100001ec: vle32.v v16, (s7)
100001f0: j 0x10000140 <main+0xd4>
100001f4: auipc a1, 0x4
100001f8: rdcycle a3
100001fc: ld s0, 0x48(sp)
10000200: sub a3, a3, a2
10000204: ld s1, 0x40(sp)
10000208: li a0, 0x0
1000020c: ld s2, 0x38(sp)
10000210: ld s3, 0x30(sp)
10000214: ld s4, 0x28(sp)
10000218: ld s5, 0x20(sp)
1000021c: ld s6, 0x18(sp)
10000220: ld s7, 0x10(sp)
10000224: ld s8, 0x8(sp)
10000228: sd a3, 0xcc(a1)
1000022c: addi sp, sp, 0x50
10000230: ret
