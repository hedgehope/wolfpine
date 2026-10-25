# scan/ame size=16384 bf16/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: li a1, 0x0
10000070: auipc a0, 0x1
10000074: auipc a3, 0x0
10000078: auipc a4, 0x0
1000007c: lui a5, 0x4
10000080: lui a2, 0x3f800
10000084: addi a0, a0, 0x210
10000088: addi a3, a3, 0x60c
1000008c: addi a4, a4, 0x208
10000090: addi a5, a5, -0x80
10000094: addi a6, a0, 0x200
10000098: li a7, 0x10
1000009c: li t0, 0x0
100000a0: addi t1, a1, -0xf
100000a4: seqz t1, t1
100000a8: fcvt.s.w fa5, t1
100000ac: mv t1, a6
100000b0: mv t2, a4
100000b4: mv t3, a3
100000b8: sltu t4, t0, a1
100000bc: sw a2, 0x0(t2)
100000c0: fsw fa5, 0x0(t3)
100000c4: addi t0, t0, 0x1
100000c8: addi t3, t3, 0x4
100000cc: addi t2, t2, 0x4
100000d0: addi t5, t4, -0x1
100000d4: negw t4, t4
100000d8: and t5, t5, a5
100000dc: and t4, t4, a5
100000e0: sh t5, -0x200(t1)
100000e4: sh t4, 0x0(t1)
100000e8: addi t1, t1, 0x2
100000ec: bne t0, a7, 0x100000b8 <main+0x4c>
100000f0: addi a1, a1, 0x1
100000f4: addi a3, a3, 0x40
100000f8: addi a4, a4, 0x40
100000fc: addi a6, a6, 0x20
10000100: bne a1, a7, 0x1000009c <main+0x30>
10000104: li a2, 0x0
10000108: rdcycle a1
1000010c: li a3, 0x20
10000110: addi a5, a0, 0x200
10000114: auipc a6, 0x0
10000118: li a4, 0x40
1000011c: auipc a7, 0x0
10000120: mx.tload.b td=0 base=a0 stride=a3 fmt=bf16 t=0
10000124: auipc a0, 0x9
10000128: addi a6, a6, 0x16c
1000012c: addi a7, a7, 0x564
10000130: addi a0, a0, 0x55c
10000134: mx.tload.a td=1 base=a5 stride=a3 fmt=bf16 t=0
10000138: mx.tload.b td=2 base=a6 stride=a4 fmt=fp32 t=0
1000013c: mx.tload.b td=3 base=a7 stride=a4 fmt=fp32 t=0
10000140: auipc a5, 0x1
10000144: addi a5, a5, 0x540
10000148: auipc a6, 0x1
1000014c: addi a6, a6, -0x6c8
10000150: li a7, 0x0
10000154: j 0x1000016c <main+0x100>
10000158: mx.astore as=0 base=a0 stride=a4 fmt=fp32
1000015c: addi a7, a7, 0x1
10000160: addi a0, a0, 0x400
10000164: addi a5, a5, 0x200
10000168: beq a7, a4, 0x100001ac <main+0x140>
1000016c: andi t0, a7, 0x1
10000170: mx.tload.a td=4 base=a5 stride=a3 fmt=bf16 t=0
10000174: slli t0, t0, 0xa
10000178: mx.azero ad=0
1000017c: mx.azero ad=1
10000180: mx.mma ad=1 ta=1 tb=4 in=bf16 acc=fp32
10000184: add t0, a6, t0
10000188: mx.mma ad=0 ta=4 tb=0 in=bf16 acc=fp32
1000018c: mx.astore as=1 base=t0 stride=a4 fmt=fp32
10000190: mx.tload.a td=5 base=t0 stride=a4 fmt=fp32 t=0
10000194: mx.mma ad=0 ta=5 tb=2 in=fp32 acc=fp32
10000198: beqz a7, 0x10000158 <main+0xec>
1000019c: addi t0, a0, -0x40
100001a0: mx.tload.a td=6 base=t0 stride=a2 fmt=fp32 t=0
100001a4: mx.mma ad=0 ta=6 tb=3 in=fp32 acc=fp32
100001a8: j 0x10000158 <main+0xec>
100001ac: mx.azero ad=0
100001b0: mx.azero ad=1
100001b4: mx.azero ad=2
100001b8: mx.azero ad=3
100001bc: rdcycle a0
100001c0: sub a0, a0, a1
100001c4: auipc a1, 0x19
100001c8: sd a0, 0x4bc(a1)
100001cc: li a0, 0x0
100001d0: ret
