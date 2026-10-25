# apsp/rvv size=64 fp64/fp64 cflags='' design=naiad-dwarfs.yaml
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
1000009c: sd s11, 0x0(sp)
100000a0: li a1, 0x1
100000a4: li a3, 0x5
100000a8: auipc a2, 0x0
100000ac: li a0, 0x0
100000b0: slli a1, a1, 0xb
100000b4: slli a3, a3, 0x9
100000b8: addi a4, a2, 0x258
100000bc: li a5, 0x40
100000c0: li a6, 0x3b
100000c4: li a7, 0x3c
100000c8: rdcycle a2
100000cc: j 0x100000dc <main+0x70>
100000d0: addi a3, a3, 0x8
100000d4: addi a0, a0, 0x1
100000d8: beq a0, a5, 0x10000218 <main+0x1ac>
100000dc: slli t1, a0, 0x9
100000e0: slli t2, a0, 0x3
100000e4: li t0, 0x0
100000e8: add t1, a4, t1
100000ec: add t2, a4, t2
100000f0: addi t3, a0, 0x40
100000f4: j 0x10000100 <main+0x94>
100000f8: add t0, s0, t0
100000fc: bgeu t0, a5, 0x100000d0 <main+0x64>
10000100: slli s1, t0, 0x3
10000104: sub t6, a5, t0
10000108: add s2, t1, s1
1000010c: vsetvli s0, t6, e64, m4, ta, ma
10000110: sub s5, a0, t0
10000114: sub s6, t3, t0
10000118: add s3, s1, a1
1000011c: li t4, 0x0
10000120: vle64.v v8, (s2)
10000124: add s2, a4, s1
10000128: vsetvli zero, t6, e64, m4, ta, ma
1000012c: li t5, 0x0
10000130: addi s4, s3, 0x200
10000134: slli s5, s5, 0x3
10000138: slli s6, s6, 0x3
1000013c: mv s7, a4
10000140: fld fa5, 0x0(t2)
10000144: fld fa4, 0x200(t2)
10000148: vle64.v v12, (s2)
1000014c: addi s2, s2, 0x200
10000150: vle64.v v16, (s2)
10000154: ori s2, t0, 0x80
10000158: vsetvli zero, t6, e64, m4, ta, ma
1000015c: j 0x100001a0 <main+0x134>
10000160: vsetvli zero, t6, e64, m4, ta, ma
10000164: vfadd.vf v28, v8, fa3
10000168: addi t5, t5, 0x4
1000016c: addi s7, s7, 0x7ff
10000170: addi t4, t4, 0x100
10000174: addi s7, s7, 0x1
10000178: vfadd.vf v4, v8, fa2
1000017c: vsetvli zero, t6, e64, m4, ta, ma
10000180: vfmin.vv v20, v20, v28
10000184: vsetvli zero, t6, e64, m4, ta, ma
10000188: vfmin.vv v24, v24, v4
1000018c: vsetvli zero, t6, e64, m4, ta, ma
10000190: vse64.v v20, (s8)
10000194: addi s8, t5, -0x4
10000198: vse64.v v24, (s9)
1000019c: bgeu s8, a7, 0x100000f8 <main+0x8c>
100001a0: or s8, t4, s2
100001a4: vfadd.vf v28, v8, fa5
100001a8: slli s8, s8, 0x3
100001ac: add s8, a4, s8
100001b0: add s10, s8, s5
100001b4: add s11, s8, s6
100001b8: addi s9, s8, 0x200
100001bc: vle64.v v20, (s8)
100001c0: fld fa3, 0x0(s10)
100001c4: fld fa2, 0x0(s11)
100001c8: add s10, s7, s1
100001cc: vfadd.vf v4, v8, fa4
100001d0: vle64.v v24, (s9)
100001d4: vsetvli zero, t6, e64, m4, ta, ma
100001d8: vfmin.vv v12, v12, v28
100001dc: vsetvli zero, t6, e64, m4, ta, ma
100001e0: vfmin.vv v16, v16, v4
100001e4: vsetvli zero, t6, e64, m4, ta, ma
100001e8: vse64.v v12, (s10)
100001ec: addi s10, s10, 0x200
100001f0: vse64.v v16, (s10)
100001f4: bltu a6, t5, 0x10000160 <main+0xf4>
100001f8: add s10, s7, s3
100001fc: add s11, s7, a3
10000200: vle64.v v12, (s10)
10000204: add s10, s7, s4
10000208: fld fa5, -0x200(s11)
1000020c: fld fa4, 0x0(s11)
10000210: vle64.v v16, (s10)
10000214: j 0x10000160 <main+0xf4>
10000218: auipc a1, 0x8
1000021c: rdcycle a3
10000220: ld s0, 0x58(sp)
10000224: sub a3, a3, a2
10000228: ld s1, 0x50(sp)
1000022c: li a0, 0x0
10000230: ld s2, 0x48(sp)
10000234: ld s3, 0x40(sp)
10000238: ld s4, 0x38(sp)
1000023c: ld s5, 0x30(sp)
10000240: ld s6, 0x28(sp)
10000244: ld s7, 0x20(sp)
10000248: ld s8, 0x18(sp)
1000024c: ld s9, 0x10(sp)
10000250: ld s10, 0x8(sp)
10000254: ld s11, 0x0(sp)
10000258: sd a3, 0xe8(a1)
1000025c: addi sp, sp, 0x60
10000260: ret
