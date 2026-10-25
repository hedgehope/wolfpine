# fft/scalar size=1024 fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x70
10000070: sd ra, 0x68(sp)
10000074: sd s0, 0x60(sp)
10000078: sd s1, 0x58(sp)
1000007c: sd s2, 0x50(sp)
10000080: sd s3, 0x48(sp)
10000084: sd s4, 0x40(sp)
10000088: sd s5, 0x38(sp)
1000008c: sd s6, 0x30(sp)
10000090: sd s7, 0x28(sp)
10000094: sd s8, 0x20(sp)
10000098: sd s9, 0x18(sp)
1000009c: sd s10, 0x10(sp)
100000a0: sd s11, 0x8(sp)
100000a4: li a0, 0x0
100000a8: auipc a5, 0x8
100000ac: auipc a6, 0xc
100000b0: rdcycle a1
100000b4: sd a1, 0x0(sp)
100000b8: auipc a7, 0x0
100000bc: auipc t0, 0x4
100000c0: li a2, 0x400
100000c4: auipc t1, 0x10
100000c8: auipc t2, 0x11
100000cc: li a3, 0x3ff
100000d0: li a4, 0x401
100000d4: addi a5, a5, 0x298
100000d8: addi a6, a6, 0x294
100000dc: addi a7, a7, 0x288
100000e0: addi t0, t0, 0x284
100000e4: addi t1, t1, 0x27c
100000e8: addi t2, t2, -0x588
100000ec: lui t3, 0x1
100000f0: li t4, 0x4
100000f4: mv t5, a6
100000f8: mv t6, a5
100000fc: j 0x10000110 <main+0xa4>
10000100: addi a0, a0, 0x1
10000104: add t6, t6, t3
10000108: add t5, t5, t3
1000010c: beq a0, t4, 0x10000260 <main+0x1f4>
10000110: li s0, 0x0
10000114: li s1, 0x0
10000118: slli s5, a0, 0xc
1000011c: add s2, a7, s5
10000120: add s3, t0, s5
10000124: add s4, a5, s5
10000128: add s5, a6, s5
1000012c: j 0x10000140 <main+0xd4>
10000130: li s6, 0x200
10000134: addi s0, s0, 0x1
10000138: or s1, s1, s6
1000013c: beq s0, a2, 0x10000188 <main+0x11c>
10000140: slli s6, s0, 0x2
10000144: slli s7, s1, 0x2
10000148: add s8, s2, s6
1000014c: flw fa5, 0x0(s8)
10000150: add s8, s4, s7
10000154: add s6, s3, s6
10000158: flw fa4, 0x0(s6)
1000015c: fsw fa5, 0x0(s8)
10000160: andi s6, s1, 0x200
10000164: add s7, s5, s7
10000168: fsw fa4, 0x0(s7)
1000016c: beqz s6, 0x10000130 <main+0xc4>
10000170: li s6, 0x200
10000174: xor s1, s1, s6
10000178: srli s6, s6, 0x1
1000017c: and s7, s1, s6
10000180: bnez s7, 0x10000174 <main+0x108>
10000184: j 0x10000134 <main+0xc8>
10000188: li s0, 0x200
1000018c: li s1, 0x2
10000190: j 0x100001a0 <main+0x134>
10000194: slliw s1, s1, 0x1
10000198: srli s0, s0, 0x1
1000019c: bge s1, a4, 0x10000100 <main+0x94>
100001a0: sraiw a1, s1, 0x1
100001a4: blez a1, 0x10000194 <main+0x128>
100001a8: li s3, 0x0
100001ac: slli s6, s1, 0x20
100001b0: slli s2, a1, 0x2
100001b4: srli s5, s6, 0x20
100001b8: srli s6, s6, 0x1e
100001bc: mv s7, t5
100001c0: mv s8, t6
100001c4: j 0x100001d8 <main+0x16c>
100001c8: addi s3, s3, 0x1
100001cc: addi s8, s8, 0x4
100001d0: addi s7, s7, 0x4
100001d4: beq s3, a1, 0x10000194 <main+0x128>
100001d8: bltu a3, s3, 0x100001c8 <main+0x15c>
100001dc: mul s9, s3, s0
100001e0: slli s9, s9, 0x2
100001e4: add s10, t1, s9
100001e8: add s9, t2, s9
100001ec: flw fa5, 0x0(s9)
100001f0: flw fa4, 0x0(s10)
100001f4: fneg.s fa3, fa5
100001f8: mv s9, s7
100001fc: mv s10, s8
10000200: mv s11, s3
10000204: flw fa2, 0x0(s10)
10000208: add ra, s10, s2
1000020c: add s4, s9, s2
10000210: flw fa1, 0x0(s4)
10000214: add s11, s11, s5
10000218: flw fa0, 0x0(ra)
1000021c: flw ft0, 0x0(s9)
10000220: fmul.s ft1, fa1, fa3
10000224: fmul.s fa1, fa4, fa1
10000228: fmadd.s ft1, fa0, fa4, ft1
1000022c: fmadd.s fa1, fa0, fa5, fa1
10000230: fadd.s fa0, fa2, ft1
10000234: fsub.s fa2, fa2, ft1
10000238: fadd.s ft1, ft0, fa1
1000023c: fsub.s fa1, ft0, fa1
10000240: fsw fa0, 0x0(s10)
10000244: fsw fa1, 0x0(s4)
10000248: fsw ft1, 0x0(s9)
1000024c: fsw fa2, 0x0(ra)
10000250: add s10, s10, s6
10000254: add s9, s9, s6
10000258: bltu s11, a2, 0x10000204 <main+0x198>
1000025c: j 0x100001c8 <main+0x15c>
10000260: rdcycle a0
10000264: ld a1, 0x0(sp)
10000268: sub a0, a0, a1
1000026c: auipc a1, 0x11
10000270: sd a0, 0xd4(a1)
10000274: li a0, 0x0
10000278: ld ra, 0x68(sp)
1000027c: ld s0, 0x60(sp)
10000280: ld s1, 0x58(sp)
10000284: ld s2, 0x50(sp)
10000288: ld s3, 0x48(sp)
1000028c: ld s4, 0x40(sp)
10000290: ld s5, 0x38(sp)
10000294: ld s6, 0x30(sp)
10000298: ld s7, 0x28(sp)
1000029c: ld s8, 0x20(sp)
100002a0: ld s9, 0x18(sp)
100002a4: ld s10, 0x10(sp)
100002a8: ld s11, 0x8(sp)
100002ac: addi sp, sp, 0x70
100002b0: ret
