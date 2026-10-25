# scan/scalar size=16384 fp32/fp32 cflags='' design=naiad-dwarfs.yaml
# <main>
1000006c: fmv.w.x fa5, zero
10000070: auipc a1, 0x10
10000074: auipc a2, 0x0
10000078: rdcycle a0
1000007c: addi a1, a1, 0xd0
10000080: addi a2, a2, 0xcc
10000084: lui a3, 0x10
10000088: add a3, a1, a3
1000008c: flw fa4, 0x0(a2)
10000090: fadd.s fa5, fa5, fa4
10000094: fsw fa5, 0x0(a1)
10000098: addi a1, a1, 0x4
1000009c: addi a2, a2, 0x4
100000a0: bne a1, a3, 0x1000008c <main+0x20>
100000a4: rdcycle a1
100000a8: sub a1, a1, a0
100000ac: auipc a0, 0x20
100000b0: sd a1, 0x94(a0)
100000b4: li a0, 0x0
100000b8: ret
