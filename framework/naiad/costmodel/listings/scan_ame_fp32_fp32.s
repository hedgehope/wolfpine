# scan/ame size=16384 fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: li a0, 0x0
10000070: auipc a2, 0x1
10000074: auipc a3, 0x1
10000078: auipc a4, 0x0
1000007c: auipc a5, 0x0
10000080: lui a1, 0x3f800
10000084: addi a2, a2, -0x1f0
10000088: addi a3, a3, -0x5f4
1000008c: addi a4, a4, 0x608
10000090: addi a5, a5, 0x204
10000094: li a6, 0x10
10000098: li a7, 0x0
1000009c: addi t0, a0, -0xf
100000a0: seqz t0, t0
100000a4: fcvt.s.w fa5, t0
100000a8: mv t0, a5
100000ac: mv t1, a4
100000b0: mv t2, a3
100000b4: mv t3, a2
100000b8: sltu t4, a7, a0
100000bc: sw a1, 0x0(t2)
100000c0: fsw fa5, 0x0(t3)
100000c4: addi a7, a7, 0x1
100000c8: addi t3, t3, 0x4
100000cc: addi t2, t2, 0x4
100000d0: addi t5, t4, -0x1
100000d4: negw t4, t4
100000d8: and t5, t5, a1
100000dc: and t4, t4, a1
100000e0: sw t5, 0x0(t0)
100000e4: sw t4, 0x0(t1)
100000e8: addi t1, t1, 0x4
100000ec: addi t0, t0, 0x4
100000f0: bne a7, a6, 0x100000b8 <main+0x4c>
100000f4: addi a0, a0, 0x1
100000f8: addi a2, a2, 0x40
100000fc: addi a3, a3, 0x40
10000100: addi a4, a4, 0x40
10000104: addi a5, a5, 0x40
10000108: bne a0, a6, 0x10000098 <main+0x2c>
1000010c: li a1, 0x0
10000110: rdcycle a0
10000114: auipc a3, 0x0
10000118: li a2, 0x40
1000011c: auipc a4, 0x0
10000120: auipc a5, 0x1
10000124: auipc a6, 0x1
10000128: addi a3, a3, 0x16c
1000012c: mx.tload.b td=0 base=a3 stride=a2 fmt=fp32 t=0
10000130: auipc a3, 0x12
10000134: addi a4, a4, 0x564
10000138: addi a5, a5, -0x6a0
1000013c: addi a6, a6, -0x2a4
10000140: addi a3, a3, -0x6b0
10000144: mx.tload.a td=1 base=a4 stride=a2 fmt=fp32 t=0
10000148: mx.tload.b td=2 base=a5 stride=a2 fmt=fp32 t=0
1000014c: mx.tload.b td=3 base=a6 stride=a2 fmt=fp32 t=0
10000150: auipc a4, 0x2
10000154: addi a4, a4, -0x6d0
10000158: auipc a5, 0x1
1000015c: addi a5, a5, 0x128
10000160: li a6, 0x0
10000164: j 0x1000017c <main+0x110>
10000168: mx.astore as=0 base=a3 stride=a2 fmt=fp32
1000016c: addi a6, a6, 0x1
10000170: addi a3, a3, 0x400
10000174: addi a4, a4, 0x400
10000178: beq a6, a2, 0x100001bc <main+0x150>
1000017c: andi a7, a6, 0x1
10000180: mx.tload.a td=4 base=a4 stride=a2 fmt=fp32 t=0
10000184: slli a7, a7, 0xa
10000188: mx.azero ad=0
1000018c: mx.azero ad=1
10000190: mx.mma ad=1 ta=1 tb=4 in=fp32 acc=fp32
10000194: add a7, a5, a7
10000198: mx.mma ad=0 ta=4 tb=0 in=fp32 acc=fp32
1000019c: mx.astore as=1 base=a7 stride=a2 fmt=fp32
100001a0: mx.tload.a td=5 base=a7 stride=a2 fmt=fp32 t=0
100001a4: mx.mma ad=0 ta=5 tb=2 in=fp32 acc=fp32
100001a8: beqz a6, 0x10000168 <main+0xfc>
100001ac: addi a7, a3, -0x40
100001b0: mx.tload.a td=6 base=a7 stride=a1 fmt=fp32 t=0
100001b4: mx.mma ad=0 ta=6 tb=3 in=fp32 acc=fp32
100001b8: j 0x10000168 <main+0xfc>
100001bc: mx.azero ad=0
100001c0: mx.azero ad=1
100001c4: mx.azero ad=2
100001c8: mx.azero ad=3
100001cc: rdcycle a1
100001d0: sub a1, a1, a0
100001d4: auipc a0, 0x22
100001d8: sd a1, -0x754(a0)
100001dc: li a0, 0x0
100001e0: ret
