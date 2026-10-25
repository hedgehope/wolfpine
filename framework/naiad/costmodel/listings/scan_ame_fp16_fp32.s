# scan/ame size=16384 fp16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: addi sp, sp, -0x20
10000070: sd s0, 0x18(sp)
10000074: sd s1, 0x10(sp)
10000078: sd s2, 0x8(sp)
1000007c: li a0, 0x0
10000080: auipc a1, 0x1
10000084: auipc a4, 0x0
10000088: auipc a5, 0x0
1000008c: lui a6, 0x8001
10000090: li a7, 0x1f
10000094: lui t0, 0x477ff
10000098: lui a2, 0x3f800
1000009c: li a3, 0x10
100000a0: lui t1, 0x3f000
100000a4: addi a1, a1, 0x280
100000a8: addi a4, a4, 0x67c
100000ac: addi a5, a5, 0x278
100000b0: addi a6, a6, -0x1
100000b4: slli a7, a7, 0xa
100000b8: addiw t0, t0, -0x1
100000bc: fmv.w.x fa5, t1
100000c0: addi t1, a1, 0x200
100000c4: li t2, 0x70
100000c8: j 0x100000e0 <main+0x74>
100000cc: addi a0, a0, 0x1
100000d0: addi a4, a4, 0x40
100000d4: addi a5, a5, 0x40
100000d8: addi t1, t1, 0x20
100000dc: beq a0, a3, 0x1000019c <main+0x130>
100000e0: li t3, 0x0
100000e4: addi t4, a0, -0xf
100000e8: seqz t4, t4
100000ec: fcvt.s.w fa4, t4
100000f0: mv t4, t1
100000f4: mv t5, a5
100000f8: mv t6, a4
100000fc: j 0x10000134 <main+0xc8>
10000100: slli s1, s0, 0x32
10000104: srli s1, s1, 0x3f
10000108: add s0, s0, a6
1000010c: add s0, s0, s1
10000110: srliw s1, s0, 0xd
10000114: sh s1, 0x0(t4)
10000118: sw a2, 0x0(t5)
1000011c: fsw fa4, 0x0(t6)
10000120: addi t3, t3, 0x1
10000124: addi t6, t6, 0x4
10000128: addi t5, t5, 0x4
1000012c: addi t4, t4, 0x2
10000130: beq t3, a3, 0x100000cc <main+0x60>
10000134: sltu s0, t3, a0
10000138: xori s1, s0, 0x1
1000013c: fcvt.s.w fa3, s1
10000140: fmv.x.w s2, fa3
10000144: mv s1, a7
10000148: bltu t0, s2, 0x10000174 <main+0x108>
1000014c: srliw s1, s2, 0x17
10000150: bltu t2, s1, 0x10000160 <main+0xf4>
10000154: fadd.s fa3, fa3, fa5
10000158: fmv.x.w s1, fa3
1000015c: j 0x10000174 <main+0x108>
10000160: slli s1, s2, 0x32
10000164: srli s1, s1, 0x3f
10000168: add s2, s2, a6
1000016c: add s1, s2, s1
10000170: srliw s1, s1, 0xd
10000174: fcvt.s.w fa3, s0
10000178: fmv.x.w s0, fa3
1000017c: sh s1, -0x200(t4)
10000180: mv s1, a7
10000184: bltu t0, s0, 0x10000114 <main+0xa8>
10000188: srliw s1, s0, 0x17
1000018c: bltu t2, s1, 0x10000100 <main+0x94>
10000190: fadd.s fa3, fa3, fa5
10000194: fmv.x.w s1, fa3
10000198: j 0x10000114 <main+0xa8>
1000019c: li a2, 0x0
100001a0: rdcycle a0
100001a4: li a3, 0x20
100001a8: addi a5, a1, 0x200
100001ac: auipc a6, 0x0
100001b0: li a4, 0x40
100001b4: auipc a7, 0x0
100001b8: mx.tload.b td=0 base=a1 stride=a3 fmt=fp16 t=0
100001bc: auipc a1, 0x9
100001c0: addi a6, a6, 0x154
100001c4: addi a7, a7, 0x54c
100001c8: addi a1, a1, 0x544
100001cc: mx.tload.a td=1 base=a5 stride=a3 fmt=fp16 t=0
100001d0: mx.tload.b td=2 base=a6 stride=a4 fmt=fp32 t=0
100001d4: mx.tload.b td=3 base=a7 stride=a4 fmt=fp32 t=0
100001d8: auipc a5, 0x1
100001dc: addi a5, a5, 0x528
100001e0: auipc a6, 0x1
100001e4: addi a6, a6, -0x6e0
100001e8: li a7, 0x0
100001ec: j 0x10000204 <main+0x198>
100001f0: mx.astore as=0 base=a1 stride=a4 fmt=fp32
100001f4: addi a7, a7, 0x1
100001f8: addi a1, a1, 0x400
100001fc: addi a5, a5, 0x200
10000200: beq a7, a4, 0x10000244 <main+0x1d8>
10000204: andi t0, a7, 0x1
10000208: mx.tload.a td=4 base=a5 stride=a3 fmt=fp16 t=0
1000020c: slli t0, t0, 0xa
10000210: mx.azero ad=0
10000214: mx.azero ad=1
10000218: mx.mma ad=1 ta=1 tb=4 in=fp16 acc=fp32
1000021c: add t0, a6, t0
10000220: mx.mma ad=0 ta=4 tb=0 in=fp16 acc=fp32
10000224: mx.astore as=1 base=t0 stride=a4 fmt=fp32
10000228: mx.tload.a td=5 base=t0 stride=a4 fmt=fp32 t=0
1000022c: mx.mma ad=0 ta=5 tb=2 in=fp32 acc=fp32
10000230: beqz a7, 0x100001f0 <main+0x184>
10000234: addi t0, a1, -0x40
10000238: mx.tload.a td=6 base=t0 stride=a2 fmt=fp32 t=0
1000023c: mx.mma ad=0 ta=6 tb=3 in=fp32 acc=fp32
10000240: j 0x100001f0 <main+0x184>
10000244: mx.azero ad=0
10000248: mx.azero ad=1
1000024c: mx.azero ad=2
10000250: mx.azero ad=3
10000254: rdcycle a1
10000258: sub a1, a1, a0
1000025c: auipc a0, 0x19
10000260: sd a1, 0x4a4(a0)
10000264: li a0, 0x0
10000268: ld s0, 0x18(sp)
1000026c: ld s1, 0x10(sp)
10000270: ld s2, 0x8(sp)
10000274: addi sp, sp, 0x20
10000278: ret
