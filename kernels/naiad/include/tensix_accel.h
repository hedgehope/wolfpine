/*
 * tensix_accel.h: drive a Tensix tile on the Naiad bus (a `type: tensix` block,
 * framework/naiad/ip/tensix.py) from a Naiad program: load an image, launch,
 * wait, and move operands between Naiad memory and the tile's L1 with the DMA
 * engine.
 *
 * Needs a design with a block named `tensix` (type tensix) and a block named
 * `dma` (type dma) -- naiad-tensix.yaml. Header-only; every function is static.
 *
 * An image (tensix_image_t) is generated from a captured tt-metal launch by
 * framework.naiad.tensix_image (e.g. tensix_gemm_fp32.h). Its life cycle,
 * which is exactly what a tt-metal host does over the wire:
 *
 *   tensix_boot(&img)            hold the cores in reset, DMA the firmware and
 *                                the init handshake into L1, release BRISC, and
 *                                wait for the firmware to answer go = DONE
 *   tensix_load_launch(&img)     DMA the kernels, CB config, runtime args and
 *                                launch message into L1
 *   tensix_set_args(&img, a, n)  overwrite the runtime args (every processor's)
 *   ... DMA operands in (tensix_dma_tilize / tensix_dma_to_l1), tensix_dma_wait_all()
 *   tensix_launch()              ring the doorbell with RUN_MSG_GO
 *   tensix_wait()                poll STATUS until the firmware reports DONE
 *   ... DMA results out (tensix_dma_untilize / tensix_dma_from_l1), tensix_dma_wait_all()
 *
 * Relaunching the same image is tensix_load_launch (or just tensix_set_args,
 * when only the args change) + tensix_launch + tensix_wait: with slow dispatch
 * the firmware always reads launch slot 0.
 *
 * DMA transfers are spread over the engine's channels round robin; a channel is
 * reused only after its previous transfer is DONE, so up to NAIAD_DMA_CHANNELS
 * transfers are in flight. Every function that polls returns 0 on success and a
 * negative code on a DMA error; tensix_check() turns that into an exit.
 */
#ifndef TENSIX_ACCEL_H
#define TENSIX_ACCEL_H

#include <stddef.h>
#include <stdint.h>

#include "naiad_platform.h"

#if !defined(NAIAD_TENSIX_PRESENT) || !defined(NAIAD_HAS_DMA)
#error "tensix_accel.h needs a design with a 'tensix' block and a 'dma' block (naiad-tensix.yaml)"
#endif

/* One image segment: `len` bytes of the image blob at `off`, for L1 address `dst`. */
typedef struct {
    uint32_t dst;
    uint32_t off;
    uint32_t len;
} tensix_segment_t;

typedef struct {
    const uint8_t *blob;
    const tensix_segment_t *boot;
    uint32_t n_boot;
    const tensix_segment_t *launch;
    uint32_t n_launch;
    const uint32_t *rta; /* runtime-arg base of every enabled processor (deduplicated) */
    uint32_t n_rta;
    uint32_t go_addr;
} tensix_image_t;

#define TENSIX_L1(addr) ((uintptr_t)NAIAD_TENSIX_BASE + NAIAD_TENSIX_L1_OFFSET + (uintptr_t)(addr))
#define TENSIX_ERR_DMA (-1)

static inline volatile uint64_t *tensix_reg(uintptr_t off) {
    return (volatile uint64_t *)((uintptr_t)NAIAD_TENSIX_BASE + NAIAD_TENSIX_CTRL_OFFSET + off);
}

static inline uint64_t tensix_status(void) { return *tensix_reg(NAIAD_TENSIX_REG_STATUS); }

/* -- DMA ------------------------------------------------------------------ */

static unsigned tensix_dma_next;
static uint8_t tensix_dma_used[NAIAD_DMA_CHANNELS];

static inline volatile uint64_t *tensix_dma_reg(unsigned ch, uintptr_t off) {
    return (volatile uint64_t *)(NAIAD_DMA_ADDR + ch * NAIAD_DMA_CH_STRIDE + off);
}

/* Wait for channel `ch` to go idle; 0, or TENSIX_ERR_DMA if its transfer failed. */
static inline int tensix_dma_wait(unsigned ch) {
    uint64_t status;
    if (!tensix_dma_used[ch]) {
        return 0;
    }
    do {
        status = *tensix_dma_reg(ch, NAIAD_DMA_STATUS);
    } while (status & NAIAD_DMA_STATUS_BUSY);
    *tensix_dma_reg(ch, NAIAD_DMA_STATUS) = status; /* W1C */
    tensix_dma_used[ch] = 0;
    return (status & NAIAD_DMA_STATUS_ERROR) ? TENSIX_ERR_DMA : 0;
}

static inline int tensix_dma_wait_all(void) {
    int rc = 0;
    for (unsigned ch = 0; ch < NAIAD_DMA_CHANNELS; ch++) {
        if (tensix_dma_wait(ch)) {
            rc = TENSIX_ERR_DMA;
        }
    }
    return rc;
}

/* Start a transfer on the next channel: 1D when rows == 0, else `rows` rows of `len` bytes. */
static inline int tensix_dma_start(uintptr_t src, uintptr_t dst, uint64_t len, uint64_t rows,
                                   uint64_t src_stride, uint64_t dst_stride) {
    unsigned ch = tensix_dma_next;
    tensix_dma_next = (ch + 1) % NAIAD_DMA_CHANNELS;
    int rc = tensix_dma_wait(ch);
    *tensix_dma_reg(ch, NAIAD_DMA_SRC) = src;
    *tensix_dma_reg(ch, NAIAD_DMA_DST) = dst;
    *tensix_dma_reg(ch, NAIAD_DMA_LEN) = len;
    *tensix_dma_reg(ch, NAIAD_DMA_ROWS) = rows;
    *tensix_dma_reg(ch, NAIAD_DMA_SRC_STRIDE) = src_stride;
    *tensix_dma_reg(ch, NAIAD_DMA_DST_STRIDE) = dst_stride;
    *tensix_dma_reg(ch, NAIAD_DMA_CTRL) = NAIAD_DMA_CTRL_START | (rows ? NAIAD_DMA_CTRL_2D : 0);
    tensix_dma_used[ch] = 1;
    return rc;
}

static inline int tensix_dma_to_l1(const void *src, uint32_t l1_addr, uint64_t len) {
    return tensix_dma_start((uintptr_t)src, TENSIX_L1(l1_addr), len, 0, 0, 0);
}

static inline int tensix_dma_from_l1(uint32_t l1_addr, void *dst, uint64_t len) {
    return tensix_dma_start(TENSIX_L1(l1_addr), (uintptr_t)dst, len, 0, 0, 0);
}

/*
 * Tile layout. A 32x32 tile is four 16x16 faces (f0 rows 0-15 cols 0-15, f1
 * rows 0-15 cols 16-31, f2 rows 16-31 cols 0-15, f3), each row-major, stored
 * f0..f3 back to back: the layout the Tensix unpackers read and the packer
 * writes. Moving one face between a row-major matrix and L1 is one 2D DMA
 * transfer (16 rows of 16 elements), so the DMA does the (un)tilize: four
 * transfers per tile.
 *
 * A face stream programs LEN, ROWS and both strides once per channel (they
 * are the same for every face, and START latches them), then issues each face
 * as SRC, DST and CTRL only, so the core spends a few instructions per face
 * and the channels, not the issue slot, set the pace.
 */
static inline int tensix_dma_faces(uintptr_t mat, uint32_t ld, uint32_t esize, uint32_t l1_addr, uint32_t rt,
                                   uint32_t ct, int to_l1) {
    const uint64_t row = 16 * esize, pitch = (uint64_t)ld * esize, face = 256 * esize;
    const uintptr_t face_off[4] = {0, row, 16 * pitch, 16 * pitch + row};
    int rc = 0;
    for (unsigned ch = 0; ch < NAIAD_DMA_CHANNELS; ch++) {
        rc |= tensix_dma_wait(ch);
        *tensix_dma_reg(ch, NAIAD_DMA_LEN) = row;
        *tensix_dma_reg(ch, NAIAD_DMA_ROWS) = 16;
        *tensix_dma_reg(ch, NAIAD_DMA_SRC_STRIDE) = to_l1 ? pitch : row;
        *tensix_dma_reg(ch, NAIAD_DMA_DST_STRIDE) = to_l1 ? row : pitch;
        tensix_dma_used[ch] = 1;
    }
    uintptr_t l1 = TENSIX_L1(l1_addr);
    uint64_t status = 0;
    unsigned ch = 0;
    for (uint32_t tr = 0; tr < rt; tr++) {
        uintptr_t m = mat + (uintptr_t)tr * 32 * pitch;
        for (uint32_t tc = 0; tc < ct; tc++, m += 32 * esize) {
            for (unsigned f = 0; f < 4; f++, l1 += face) {
                volatile uint64_t *r = tensix_dma_reg(ch, 0);
                uint64_t s;
                do { /* START also clears the channel's DONE/ERROR, so no W1C is needed */
                    s = r[NAIAD_DMA_STATUS / 8];
                } while (s & NAIAD_DMA_STATUS_BUSY);
                status |= s;
                r[NAIAD_DMA_SRC / 8] = to_l1 ? m + face_off[f] : l1;
                r[NAIAD_DMA_DST / 8] = to_l1 ? l1 : m + face_off[f];
                r[NAIAD_DMA_CTRL / 8] = NAIAD_DMA_CTRL_START | NAIAD_DMA_CTRL_2D;
                ch = (ch + 1) % NAIAD_DMA_CHANNELS;
            }
        }
    }
    tensix_dma_next = ch;
    return rc | ((status & NAIAD_DMA_STATUS_ERROR) ? TENSIX_ERR_DMA : 0);
}

/* The rt x ct tiles of the row-major matrix at `src` (`ld` elements a row, `esize` bytes each)
 * into L1 at `l1_addr`, in row-major tile order. */
static inline int tensix_dma_tilize(const void *src, uint32_t ld, uint32_t esize, uint32_t l1_addr, uint32_t rt,
                                    uint32_t ct) {
    return tensix_dma_faces((uintptr_t)src, ld, esize, l1_addr, rt, ct, 1);
}

/* The inverse: rt x ct tiles at `l1_addr` into the row-major matrix at `dst`. */
static inline int tensix_dma_untilize(uint32_t l1_addr, void *dst, uint32_t ld, uint32_t esize, uint32_t rt,
                                      uint32_t ct) {
    return tensix_dma_faces((uintptr_t)dst, ld, esize, l1_addr, rt, ct, 0);
}

/* -- image, launch, wait --------------------------------------------------- */

static inline int tensix_load_segments(const tensix_image_t *img, const tensix_segment_t *segs, uint32_t n) {
    int rc = 0;
    for (uint32_t i = 0; i < n; i++) {
        rc |= tensix_dma_to_l1(img->blob + segs[i].off, segs[i].dst, segs[i].len);
    }
    return rc | tensix_dma_wait_all();
}

static inline uint32_t tensix_go_signal(void) {
    return (uint32_t)(tensix_status() >> NAIAD_TENSIX_STATUS_SIGNAL_SHIFT) & 0xFF;
}

/* Bring the tile up from `img`'s boot segments; returns once the firmware is idle (go = DONE). */
static inline int tensix_boot(const tensix_image_t *img) {
    *tensix_reg(NAIAD_TENSIX_REG_CTRL) = NAIAD_TENSIX_CTRL_RESET_ASSERT;
    int rc = tensix_load_segments(img, img->boot, img->n_boot);
    if (rc) {
        return rc;
    }
    *tensix_reg(NAIAD_TENSIX_REG_CTRL) = NAIAD_TENSIX_CTRL_RESET_RELEASE;
    while (tensix_go_signal() != NAIAD_TENSIX_RUN_MSG_DONE) {
    }
    return 0;
}

static inline int tensix_load_launch(const tensix_image_t *img) {
    return tensix_load_segments(img, img->launch, img->n_launch);
}

/* Write `n` 32-bit runtime args at every processor's runtime-arg base. */
static inline void tensix_set_args(const tensix_image_t *img, const uint32_t *args, uint32_t n) {
    for (uint32_t r = 0; r < img->n_rta; r++) {
        volatile uint32_t *p = (volatile uint32_t *)TENSIX_L1(img->rta[r]);
        for (uint32_t i = 0; i < n; i++) {
            p[i] = args[i];
        }
    }
}

static inline void tensix_launch(void) { *tensix_reg(NAIAD_TENSIX_REG_DOORBELL) = NAIAD_TENSIX_RUN_MSG_GO; }

/* Poll until the launch finishes; returns the number of STATUS polls. */
static inline uint64_t tensix_wait(void) {
    uint64_t polls = 0;
    while (!(tensix_status() & NAIAD_TENSIX_STATUS_DONE)) {
        polls++;
    }
    return polls;
}

static inline void tensix_check(int rc, const char *what) {
    if (rc) {
        naiad_printf("tensix: %s failed (%d)\n", what, rc);
        naiad_exit(3);
    }
}

#endif /* TENSIX_ACCEL_H */
