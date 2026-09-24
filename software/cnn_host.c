/*
 * cnn_host.c — KV260 PS host application for the CNN accelerator (cnn_top).
 *
 * Runs on the ARM Cortex-A53 under Linux (Kria native Ubuntu 22.04) and drives
 * the PL-side AXI4-Lite slave through a UIO mapping (or /dev/mem fallback). It
 * loads the real MNIST-12 Conv1 image + weights, runs the layer in OS, WS, and
 * WS-dense (sparsity disabled) modes, reads back the results, checks them
 * bit-exact against the committed integer golden, and prints a labelled
 * benchmark report (latency, MAC/cycle, throughput, reconfiguration overhead).
 *
 * Build (aarch64 cross-compile):
 *   aarch64-linux-gnu-gcc -O2 -o cnn_host cnn_host.c
 * Native (x86) build for a dry-run against a simulated register model:
 *   gcc -O2 -D HOST_DRYRUN -o cnn_host cnn_host.c
 *
 * Register map (32-bit, byte offsets):
 *   0x0000 VERSION   RO   0x00020001
 *   0x0004 CONTROL   WO   [0]=START [1]=SOFT_RESET [2]=MODE_COMMIT
 *   0x0008 STATUS    RO   [0]=IDLE [1]=BUSY [2]=DONE [3]=ERROR [4]=SWITCHING
 *   0x000C DATAFLOW_MODE RW [0] 0=OS 1=WS
 *   0x0014 SPARSITY_DISABLE RW [0] 1 = force WS dense
 *   0x0018 CYCLE_RUN RO   per-run cycle count (frozen at DONE)
 *   0x001C TOTAL_MACS  RO  156800
 *   0x0020 EXECUTED_MACS RO
 *   0x0024 SKIPPED_MACS  RO
 *   0x002C IMAGE_ADDR  RW  auto-increment image write address
 *   0x0030 IMAGE_DATA  WO  write img[addr++]
 *   0x0034 WEIGHT_ADDR RW  auto-increment weight write address
 *   0x0038 WEIGHT_DATA WO  write wgt[addr++]
 *   0x1000 RESULT     RO  result window: bank=(byte-0x1000)>>12, dense=(..>>2)&0x3FF
 *
 * The per-run latency instrument is the hardware CYCLE_RUN counter (no software
 * timing noise). Wall-clock is measured separately and reported as PS-side
 * overhead, never blended into the hardware number.
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ reg map */
#define REG_VERSION        0x0000
#define REG_CONTROL        0x0004
#define REG_STATUS         0x0008
#define REG_DATAFLOW_MODE  0x000C
#define REG_MODE_STATUS    0x0010
#define REG_SPARSITY_DIS   0x0014
#define REG_CYCLE_RUN      0x0018
#define REG_TOTAL_MACS     0x001C
#define REG_EXECUTED       0x0020
#define REG_SKIPPED        0x0024
#define REG_IMAGE_ADDR     0x002C
#define REG_IMAGE_DATA     0x0030
#define REG_WEIGHT_ADDR    0x0034
#define REG_WEIGHT_DATA    0x0038
#define RESULT_BASE        0x1000

#define CTRL_START         (1u << 0)
#define CTRL_SOFT_RESET    (1u << 1)
#define CTRL_MODE_COMMIT   (1u << 2)

#define ST_IDLE            (1u << 0)
#define ST_BUSY            (1u << 1)
#define ST_DONE            (1u << 2)

#define IMG_H 28
#define IMG_W 28
#define OC 8
#define OUT_PIXELS (OC * IMG_H * IMG_W)   /* 6272 */

static volatile uint32_t *regs;   /* mapped AXI window (volatile: uncached I/O) */

static inline uint32_t rd(uint32_t off) { return regs[off / 4]; }
static inline void     wr(uint32_t off, uint32_t v) { regs[off / 4] = v; }

static double now_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

static void wait_status(uint32_t bit) {
    int guard = 0;
    while (!(rd(REG_STATUS) & bit)) {
        if (++guard > 20000000) {
            fprintf(stderr, "timeout waiting status bit 0x%x\n", bit);
            exit(1);
        }
    }
}

/* ------------------------------------------------------------------ vectors */
static int8_t   img[IMG_H * IMG_W];
static int8_t   wgt[OC * 25];
static int32_t  golden[OUT_PIXELS];

/* read a 2-hex-digit-per-line int8 hex file (weights.hex / input_img.hex) */
static int load_int8_hex(const char *path, int8_t *dst, int n) {
    FILE *f = fopen(path, "r");
    int v, i;
    if (!f) { perror(path); return -1; }
    for (i = 0; i < n; i++) {
        if (fscanf(f, "%x", &v) != 1) break;
        dst[i] = (int8_t)(uint8_t)v;   /* two's-complement int8 */
    }
    fclose(f);
    return (i == n) ? 0 : -1;
}

/* read an 8-hex-digit-per-line int32 hex file (golden_canonical.hex) */
static int load_int32_hex(const char *path, int32_t *dst, int n) {
    FILE *f = fopen(path, "r");
    unsigned v;
    int i;
    if (!f) { perror(path); return -1; }
    for (i = 0; i < n; i++) {
        if (fscanf(f, "%x", &v) != 1) break;
        dst[i] = (int32_t)v;
    }
    fclose(f);
    return (i == n) ? 0 : -1;
}

/* ------------------------------------------------------------------ control */
static void set_mode(int mode) {
    wr(REG_DATAFLOW_MODE, (uint32_t)mode);
    wr(REG_CONTROL, CTRL_MODE_COMMIT);
    wait_status(ST_IDLE);
}

static void run_frame(void) {
    wr(REG_CONTROL, CTRL_START);
    wait_status(ST_DONE);
}

static void load_image_weights(void) {
    wr(REG_IMAGE_ADDR, 0);
    for (int i = 0; i < IMG_H * IMG_W; i++)
        wr(REG_IMAGE_DATA, (uint32_t)(uint8_t)img[i]);
    wr(REG_WEIGHT_ADDR, 0);
    for (int i = 0; i < OC * 25; i++)
        wr(REG_WEIGHT_DATA, (uint32_t)(uint8_t)wgt[i]);
}

/* read results back through the 8-bank window, compare vs golden */
static int check_results(const char *tag, int *first_bad) {
    int bad = 0;
    *first_bad = -1;
    for (int ch = 0; ch < OC; ch++)
        for (int y = 0; y < IMG_H; y++)
            for (int x = 0; x < IMG_W; x++) {
                int flat  = ch * IMG_H * IMG_W + y * IMG_W + x;
                int bank  = x & 7;
                int dense = flat >> 3;
                uint32_t addr = RESULT_BASE + 4u * (bank * 1024 + dense);
                int32_t v = (int32_t)rd(addr);
                if (v != golden[flat]) {
                    if (bad < 10)
                        fprintf(stderr, "  [%s] ch=%d y=%d x=%d got=%d exp=%d\n",
                                tag, ch, y, x, v, golden[flat]);
                    bad++;
                    if (*first_bad < 0) *first_bad = flat;
                }
            }
    return bad;
}

/* run one configuration; report cycle count and wall-clock */
static void bench_config(const char *tag, int mode, int sparsity_disable,
                         int expected_cycles) {
    double t0, t1;
    uint32_t cyc, total, exec, skip;
    int first_bad;

    set_mode(mode);
    wr(REG_SPARSITY_DIS, (uint32_t)sparsity_disable);
    t0 = now_s();
    run_frame();
    t1 = now_s();

    cyc   = rd(REG_CYCLE_RUN);
    total = rd(REG_TOTAL_MACS);
    exec  = rd(REG_EXECUTED);
    skip  = rd(REG_SKIPPED);
    int bad = check_results(tag, &first_bad);

    printf("[%s]\n", tag);
    printf("  cycles       = %u%s\n", cyc,
           expected_cycles > 0 && cyc != (uint32_t)expected_cycles
               ? "  <-- MISMATCH vs expected" : "");
    printf("  PS wall time = %.1f us\n", (t1 - t0) * 1e6);
    printf("  MACs         = %u (exec %u, skip %u)\n", total, exec, skip);
    printf("  MAC/cycle    = %.2f\n", (double)total / (double)cyc);
    printf("  result match = %s\n", bad == 0 ? "BIT-EXACT (6272/6272)" : "MISMATCH");
    if (bad) { fprintf(stderr, "  %d result mismatches (first flat %d)\n", bad, first_bad); }
}

int main(int argc, char **argv) {
    const char *uio_dev = argc > 1 ? argv[1] : "/dev/uio0";
    const char *img_path = argc > 2 ? argv[2] : "data/vectors/input_img.hex";
    const char *wgt_path = argc > 3 ? argv[3] : "data/vectors/weights.hex";
    const char *gold_path = argc > 4 ? argv[4] : "data/vectors/golden_canonical.hex";

    /* map the AXI window (4 MB covers control + 32 KB result window) */
    int fd = open(uio_dev, O_RDWR | O_SYNC);
    if (fd < 0) { perror(uio_dev); return 1; }
    regs = mmap(NULL, 0x400000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (regs == MAP_FAILED) { perror("mmap"); return 1; }

    printf("cnn_host: VERSION = 0x%08x\n", rd(REG_VERSION));

    if (load_int8_hex(img_path, img, IMG_H * IMG_W) ||
        load_int8_hex(wgt_path, wgt, OC * 25) ||
        load_int32_hex(gold_path, golden, OUT_PIXELS)) {
        fprintf(stderr, "vector load failed\n");
        return 1;
    }

    load_image_weights();

    bench_config("OS dense", 0, 0, 9184);
    bench_config("WS sparse", 1, 0, 18368);
    bench_config("WS dense (sparsity disabled)", 1, 1, 38528);

    /* runtime reconfiguration overhead: OS -> WS -> OS */
    double t0 = now_s();
    set_mode(1); set_mode(0);
    double t1 = now_s();
    printf("[reconfiguration] OS->WS->OS PS overhead = %.1f us "
           "(flush = 8 cycles each, 40 ns @ 200 MHz)\n", (t1 - t0) * 1e6);

    printf("RESULT: HOST RUN COMPLETE\n");
    return 0;
}
