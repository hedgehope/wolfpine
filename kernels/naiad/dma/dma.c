/*
 * dma: move tiles between DRAM and the scratchpad with the DMA engine.
 *
 * A 1D copy of a buffer DRAM -> scratchpad, a 2D copy of a 16x16 tile out of a
 * 64x64 DRAM matrix into a packed scratchpad tile, a pass over the tile in the
 * scratchpad, and a 2D copy back into a second DRAM matrix. Every transfer is
 * polled to DONE and checked against what the CPU computes from the source. It
 * also shows the race: a read of the destination straight after START, before
 * DONE, still sees the old bytes. Build against naiad-dma.yaml.
 */
#include "naiad_platform.h"

#if !defined(NAIAD_HAS_DMA) || !defined(NAIAD_SCRATCHPAD_BASE)
#error "dma needs a design with a 'dma' block and a block named 'scratchpad' (naiad-dma.yaml)"
#endif

#define BUF_WORDS 1024 /* 4 KiB 1D transfer */
#define MAT_N 64       /* 64x64 uint32 matrices in DRAM */
#define TILE 16        /* 16x16 tile */
#define TILE_ROW0 8
#define TILE_COL0 16

static uint32_t buf[BUF_WORDS];
static uint32_t mat[MAT_N][MAT_N] __attribute__((aligned(64)));
static uint32_t out[MAT_N][MAT_N] __attribute__((aligned(64)));

static volatile uint64_t *dma_reg(uintptr_t offset) {
    return (volatile uint64_t *)(NAIAD_DMA_ADDR + offset);
}

static void dma_start(uintptr_t src, uintptr_t dst, uint64_t len, uint64_t rows, uint64_t src_stride,
                      uint64_t dst_stride) {
    *dma_reg(NAIAD_DMA_SRC) = src;
    *dma_reg(NAIAD_DMA_DST) = dst;
    *dma_reg(NAIAD_DMA_LEN) = len;
    *dma_reg(NAIAD_DMA_ROWS) = rows;
    *dma_reg(NAIAD_DMA_SRC_STRIDE) = src_stride;
    *dma_reg(NAIAD_DMA_DST_STRIDE) = dst_stride;
    *dma_reg(NAIAD_DMA_CTRL) = NAIAD_DMA_CTRL_START | (rows ? NAIAD_DMA_CTRL_2D : 0);
}

/* Poll until the channel is idle; returns STATUS (DONE or ERROR) and clears it. */
static uint64_t dma_wait(unsigned *polls) {
    uint64_t status;
    unsigned n = 0;
    do {
        status = *dma_reg(NAIAD_DMA_STATUS);
        n++;
    } while (status & NAIAD_DMA_STATUS_BUSY);
    *dma_reg(NAIAD_DMA_STATUS) = status;
    if (polls) {
        *polls = n;
    }
    return status;
}

static uint32_t checksum(const volatile uint32_t *p, unsigned n) {
    uint32_t sum = 0;
    for (unsigned i = 0; i < n; i++) {
        sum = (sum << 5 | sum >> 27) ^ p[i];
    }
    return sum;
}

static int check(const char *what, unsigned long long got, unsigned long long want) {
    naiad_printf("  %s: 0x%llx (expect 0x%llx) %s\n", what, got, want, got == want ? "ok" : "MISMATCH");
    return got == want;
}

int main(void) {
    volatile uint32_t *sp = (volatile uint32_t *)NAIAD_SCRATCHPAD_BASE;
    volatile uint32_t *sp_tile = sp + BUF_WORDS;
    int ok = 1;
    unsigned polls;

    naiad_printf("dma on %s: %d bytes/cycle, setup %d cycles\n", NAIAD_DESIGN, (int)NAIAD_DMA_BYTES_PER_CYCLE,
                 (int)NAIAD_DMA_SETUP_LATENCY);

    for (unsigned i = 0; i < BUF_WORDS; i++) {
        buf[i] = i * 2654435761u;
        sp[i] = 0xAAAAAAAAu;
    }
    for (unsigned r = 0; r < MAT_N; r++) {
        for (unsigned c = 0; c < MAT_N; c++) {
            mat[r][c] = r << 16 | c;
        }
    }

    /* 1D: DRAM -> scratchpad. The first read races the transfer and sees the fill. */
    uint64_t start = naiad_cycles();
    dma_start((uintptr_t)buf, (uintptr_t)sp, sizeof buf, 0, 0, 0);
    uint32_t early = sp[BUF_WORDS - 1];
    uint64_t status = dma_wait(&polls);
    uint64_t elapsed = naiad_cycles() - start;
    ok &= check("1D status", status, NAIAD_DMA_STATUS_DONE);
    ok &= check("race: last word before DONE", early, 0xAAAAAAAAu);
    ok &= check("1D checksum", checksum(sp, BUF_WORDS), checksum(buf, BUF_WORDS));
    naiad_printf("  1D %u bytes: %llu cycles, %u polls\n", (unsigned)sizeof buf, (unsigned long long)elapsed, polls);

    /* 2D: one 16x16 tile of mat -> a packed scratchpad tile. */
    dma_start((uintptr_t)&mat[TILE_ROW0][TILE_COL0], (uintptr_t)sp_tile, TILE * 4, TILE, MAT_N * 4, TILE * 4);
    ok &= check("2D in status", dma_wait(0), NAIAD_DMA_STATUS_DONE);
    unsigned bad = 0;
    for (unsigned r = 0; r < TILE; r++) {
        for (unsigned c = 0; c < TILE; c++) {
            bad += sp_tile[r * TILE + c] != mat[TILE_ROW0 + r][TILE_COL0 + c];
        }
    }
    ok &= check("2D tile mismatches", bad, 0);

    /* Work on the tile in the scratchpad, then 2D it back into out at the same place. */
    for (unsigned i = 0; i < TILE * TILE; i++) {
        sp_tile[i] = ~sp_tile[i];
    }
    dma_start((uintptr_t)sp_tile, (uintptr_t)&out[TILE_ROW0][TILE_COL0], TILE * 4, TILE, TILE * 4, MAT_N * 4);
    ok &= check("2D out status", dma_wait(0), NAIAD_DMA_STATUS_DONE);
    bad = 0;
    for (unsigned r = 0; r < MAT_N; r++) {
        for (unsigned c = 0; c < MAT_N; c++) {
            int in_tile = r - TILE_ROW0 < TILE && c - TILE_COL0 < TILE;
            bad += out[r][c] != (in_tile ? ~mat[r][c] : 0u);
        }
    }
    ok &= check("2D round trip mismatches", bad, 0);

    /* A misaligned 2D transfer is refused with ERR_ALIGN and moves nothing. */
    dma_start((uintptr_t)&mat[0][1], (uintptr_t)sp_tile, 4, 2, MAT_N * 4, 4);
    status = dma_wait(0);
    ok &= check("misaligned 2D", status,
                NAIAD_DMA_STATUS_ERROR | (NAIAD_DMA_ERR_ALIGN << NAIAD_DMA_STATUS_ERR_SHIFT));

    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
