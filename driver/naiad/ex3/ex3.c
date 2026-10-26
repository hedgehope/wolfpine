/*
 * ex3: the DMA engine. Design: naiad-dma (RV64IM, a 1-channel DMA, a 256 KiB
 * scratchpad).
 *
 * A channel is a set of 64-bit registers (NAIAD_DMA_*, offsets from naiad.h):
 * write SRC, DST, LEN (bytes per row) and, for a 2D transfer, ROWS and the two
 * row strides, then CTRL = START (| 2D). STATUS reads BUSY until the transfer
 * finishes, then DONE (or ERROR with a code); writing STATUS back clears it.
 * The engine moves NAIAD_DMA_BYTES_PER_CYCLE bytes a cycle after
 * NAIAD_DMA_SETUP_LATENCY cycles, in parallel with the core.
 *
 * Two transfers, each polled to DONE and checked word by word:
 *   1D: a 2 KiB buffer, DRAM -> scratchpad;
 *   2D: an 8 x 8 block out of a 32 x 32 DRAM matrix -> a packed scratchpad tile.
 */
#include "naiad_platform.h"

#define WORDS 512
#define N 32
#define T 8

static uint32_t src[WORDS];
static uint32_t mat[N][N] __attribute__((aligned(64)));

static volatile uint64_t *dma(uintptr_t offset) { return (volatile uint64_t *)(NAIAD_DMA_ADDR + offset); }

static void dma_start(const void *from, volatile void *to, uint64_t len, uint64_t rows, uint64_t src_stride,
                      uint64_t dst_stride) {
    *dma(NAIAD_DMA_SRC) = (uintptr_t)from;
    *dma(NAIAD_DMA_DST) = (uintptr_t)to;
    *dma(NAIAD_DMA_LEN) = len;
    *dma(NAIAD_DMA_ROWS) = rows;
    *dma(NAIAD_DMA_SRC_STRIDE) = src_stride;
    *dma(NAIAD_DMA_DST_STRIDE) = dst_stride;
    *dma(NAIAD_DMA_CTRL) = NAIAD_DMA_CTRL_START | (rows ? NAIAD_DMA_CTRL_2D : 0);
}

/* Poll STATUS until the channel is idle; clear and return it. */
static uint64_t dma_wait(unsigned long *polls) {
    uint64_t status;
    *polls = 0;
    do {
        status = *dma(NAIAD_DMA_STATUS);
        ++*polls;
    } while (status & NAIAD_DMA_STATUS_BUSY);
    *dma(NAIAD_DMA_STATUS) = status;
    return status;
}

static int check(const char *what, long got, long want) {
    naiad_printf("  %s: %ld (expect %ld) %s\n", what, got, want, got == want ? "ok" : "MISMATCH");
    return got == want;
}

int main(void) {
    int ok = 1;
    unsigned long polls;
    volatile uint32_t *sp = (volatile uint32_t *)NAIAD_SCRATCHPAD_BASE;
    volatile uint32_t *tile = sp + WORDS;
    naiad_printf("ex3: DMA on %s, %lu bytes/cycle after %lu setup cycles\n", NAIAD_DESIGN,
                 (unsigned long)NAIAD_DMA_BYTES_PER_CYCLE, (unsigned long)NAIAD_DMA_SETUP_LATENCY);

    for (unsigned i = 0; i < WORDS; i++) {
        src[i] = i * 0x9E3779B9u;
    }
    for (unsigned r = 0; r < N; r++) {
        for (unsigned c = 0; c < N; c++) {
            mat[r][c] = r << 8 | c;
        }
    }

    /* 1D. */
    uint64_t start = naiad_cycles();
    dma_start(src, sp, sizeof src, 0, 0, 0);
    uint64_t status = dma_wait(&polls);
    uint64_t took = naiad_cycles() - start;
    ok &= check("1D status DONE", status == NAIAD_DMA_STATUS_DONE, 1);
    unsigned bad = 0;
    for (unsigned i = 0; i < WORDS; i++) {
        bad += sp[i] != src[i];
    }
    ok &= check("1D mismatched words", bad, 0);
    naiad_printf("  1D %lu bytes: %lu cycles, %lu polls\n", (unsigned long)sizeof src, (unsigned long)took, polls);
    ok &= check("1D took at least LEN / bytes_per_cycle", took >= sizeof src / NAIAD_DMA_BYTES_PER_CYCLE, 1);

    /* 2D: rows of T words, the source row stride is a matrix row, the destination is packed. */
    dma_start(&mat[4][8], tile, T * 4, T, N * 4, T * 4);
    status = dma_wait(&polls);
    ok &= check("2D status DONE", status == NAIAD_DMA_STATUS_DONE, 1);
    bad = 0;
    for (unsigned r = 0; r < T; r++) {
        for (unsigned c = 0; c < T; c++) {
            bad += tile[r * T + c] != mat[4 + r][8 + c];
        }
    }
    ok &= check("2D mismatched words", bad, 0);

    naiad_puts(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
