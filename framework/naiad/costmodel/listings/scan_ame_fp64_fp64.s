# scan/ame size=16384 fp64/fp64 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: li a0, 0x0
10000070: auipc a1, 0x2
10000074: auipc a2, 0x1
10000078: auipc a3, 0x1
1000007c: auipc a4, 0x0
10000080: li a5, 0x3ff
10000084: addi a1, a1, -0x5f0
10000088: addi a2, a2, 0x20c
1000008c: addi a3, a3, -0x5f8
10000090: addi a4, a4, 0x204
10000094: slli a5, a5, 0x34
10000098: li a6, 0x10
1000009c: li a7, 0x0
100000a0: addi t0, a0, -0xf
100000a4: seqz t0, t0
100000a8: fcvt.d.w fa5, t0
100000ac: mv t0, a4
100000b0: mv t1, a3
100000b4: mv t2, a2
100000b8: mv t3, a1
100000bc: sltu t4, a7, a0
100000c0: sd a5, 0x0(t2)
100000c4: fsd fa5, 0x0(t3)
100000c8: addi a7, a7, 0x1
100000cc: addi t3, t3, 0x8
100000d0: addi t2, t2, 0x8
100000d4: addi t5, t4, -0x1
100000d8: neg t4, t4
100000dc: and t5, t5, a5
100000e0: and t4, t4, a5
100000e4: sd t5, 0x0(t0)
100000e8: sd t4, 0x0(t1)
100000ec: addi t1, t1, 0x8
100000f0: addi t0, t0, 0x8
100000f4: bne a7, a6, 0x100000bc <main+0x50>
100000f8: addi a0, a0, 0x1
100000fc: addi a1, a1, 0x80
10000100: addi a2, a2, 0x80
10000104: addi a3, a3, 0x80
10000108: addi a4, a4, 0x80
1000010c: bne a0, a6, 0x1000009c <main+0x30>
10000110: li a1, 0x0
10000114: rdcycle a0
10000118: auipc a3, 0x0
1000011c: li a2, 0x80
10000120: auipc a4, 0x1
10000124: auipc a5, 0x1
10000128: auipc a6, 0x2
1000012c: auipc a7, 0x23
10000130: auipc t0, 0x3
10000134: addi a3, a3, 0x168
10000138: mx.tload.b td=0 base=a3 stride=a2 fmt=fp64 t=0
1000013c: auipc t1, 0x2
10000140: addi t2, a4, -0x6a0
10000144: addi a5, a5, 0x15c
10000148: addi a6, a6, -0x6a8
1000014c: addi a3, a7, 0x154
10000150: addi a4, t0, 0x150
10000154: mx.tload.a td=1 base=t2 stride=a2 fmt=fp64 t=0
10000158: mx.tload.b td=2 base=a5 stride=a2 fmt=fp64 t=0
1000015c: mx.tload.b td=3 base=a6 stride=a2 fmt=fp64 t=0
10000160: addi a5, t1, 0x144
10000164: li a6, 0x1
10000168: slli a6, a6, 0xb
1000016c: li a7, 0x40
10000170: li t0, 0x0
10000174: j 0x1000018c <main+0x120>
10000178: mx.astore as=0 base=a3 stride=a2 fmt=fp64
1000017c: addi t0, t0, 0x1
10000180: add a3, a3, a6
10000184: add a4, a4, a6
10000188: beq t0, a7, 0x100001cc <main+0x160>
1000018c: andi t1, t0, 0x1
10000190: mx.tload.a td=4 base=a4 stride=a2 fmt=fp64 t=0
10000194: slli t1, t1, 0xb
10000198: mx.azero ad=0
1000019c: mx.azero ad=1
100001a0: mx.mma ad=1 ta=1 tb=4 in=fp64 acc=fp64
100001a4: add t1, a5, t1
100001a8: mx.mma ad=0 ta=4 tb=0 in=fp64 acc=fp64
100001ac: mx.astore as=1 base=t1 stride=a2 fmt=fp64
100001b0: mx.tload.a td=5 base=t1 stride=a2 fmt=fp64 t=0
100001b4: mx.mma ad=0 ta=5 tb=2 in=fp64 acc=fp64
100001b8: beqz t0, 0x10000178 <main+0x10c>
100001bc: addi t1, a3, -0x80
100001c0: mx.tload.a td=6 base=t1 stride=a1 fmt=fp64 t=0
100001c4: mx.mma ad=0 ta=6 tb=3 in=fp64 acc=fp64
100001c8: j 0x10000178 <main+0x10c>
100001cc: mx.azero ad=0
100001d0: mx.azero ad=1
100001d4: mx.azero ad=2
100001d8: mx.azero ad=3
100001dc: rdcycle a1
100001e0: sub a1, a1, a0
100001e4: auipc a0, 0x43
100001e8: sd a1, 0x9c(a0)
100001ec: li a0, 0x0
100001f0: ret
