# V2 Frozen Architecture Spec — generalized INT8 8x8 output-stationary accelerator (KV260)

**Status:** FROZEN 2026-09-24 (post-recon).
**Target:** Kria KV260, `xck26-sfvc784-2LV-c`, Vivado 2023.1.

## Workloads

VALID padding, stride 1, all pools 2x2/2 MAX on INT8 after ReLU.

- **LeNet-5:** 32x32x1 → conv5x5 1→6, ReLU, pool → conv5x5 6→16, ReLU, pool → conv5x5 16→120, ReLU → FC 120→84, ReLU → FC 84→10.
- **CIFAR-10:** 32x32x3 → conv5x5 3→32, ReLU, pool → conv5x5 32→32, ReLU, pool → conv5x5 32→64, ReLU → FC 64→10.

**Out of scope:** weight-stationary/reconfigurable dataflow, sparsity skipping, padding, stride>1, batching.

## Numeric contract (bit-exact to the frozen float64 reference)

- INT8 [-128,127] symmetric, zero-point 0; per-tensor activation scale, per-output-channel weight scale; accumulator INT32 (max |acc| = 800\*128\*128 = 13,107,200 < 2^24).
- `q_bias = rint_float64(b/(Sa*Sw))` → int32, computed offline (assert it fits int32).
- **Reference requant:** `rint_float64((acc + q_bias) * M_float64)`, clip [-128,127]; ReLU on INT8 after requant for all layers except the final one.
- **Hardware requant:**
  ```
  v = acc + q_bias
  p = v * m                                   (exact integer, m has B bits, unsigned)
  q = (p + 2^(s-1) - 1 + ((p >> s) & 1)) >>> s    (round half to even)
  relu; clip
  ```
- **B and per-channel (m, s):** chosen by the exhaustive equivalence check (model step) as the smallest B in {32, 40, 48} with 0 mismatches vs the float64 reference over every reachable v (|v| ≤ K\*16384 + |q_bias|) for every channel of both nets. If none passes: fallback = integer contract becomes the V2 golden, accuracy re-measured and reported (record in DECISIONS.md).
- **Final layer:** no requant. Hardware writes raw INT32 `v = acc + q_bias` per class to `LOGIT[0..15]` (AXI-Lite readable). The PS applies the reference's own float32 dequantization and argmax. Bit-exact comparison is on the INT32 logits.
- Max-pool after requant is exact (requant/ReLU/clip are monotonic).

## Datapath

- **PE:** one DSP48E2; `acc <= first ? a*w : acc + a*w` (no clear cycle).
- **Array:** 8x8 outer-product output-stationary, operands broadcast (registered before fanout 8), no skew. Rows = 8 consecutive output x positions in one output row; columns = 8 output channels. Each cycle consumes one k = (ic, ky, kx), kx innermost; K = IC\*KH\*KW.
- **Drain:** on `last`, the 64 accumulators copy to shadow registers; the shadow drains 1 column (1 output channel x 8 pixels) per cycle over 8 cycles, overlapped with the next tile. Back-pressure stall if the drain is busy (minimum K = 25, so it should never stall).
- **Requant:** 8 lanes, 3–4 pipeline stages, one output channel's (q_bias, m, s) per drain cycle.
- **Pool:** fused 2x2 max. The dy=0 tile is stored (8x8 bytes); on dy=1: vertical max, then horizontal pair max, writing 4 bytes per column via byte enables. Bypassable.

## Tiling / loop nest

```
for oc_tile in ceil(OC/8):
  for oy (or oy pair when pooling):
    for ox_tile (ox0 = 8*ox_tile):
      for dy in {0,1} if pooling:
        stream k
```

- T = ceil(OC/8)\*OH\*ceil(OW/8).
- FC = 1x1 conv on a 1x1 map (conv3 outputs are 1x1, so no flatten reorder).
- OC tail: weights zero-padded offline, writes masked.
- x tail: writes masked (masked rows may read garbage).

## Memory (all BRAM, all weights resident, no DRAM traffic during inference)

- **ACT0/ACT1 ping-pong:** each 8 banks x 8-bit x 4096 deep. bank = x mod 8; word = (ic\*IH + y)\*ceil(IW/8) + floor(x/8).
- **Conflict-free read:** bank b supplies row r = (b - kx) mod 8 at address rowbase + ox0/8 + (b < kx ? 1 : 0); a byte rotator by kx aligns banks to rows.
- **PS view of ACT:** one 64-bit word + byte enables (custom TDP: 64-bit port A, 8 independent 8-bit ports B).
- **WGT:** 64-bit word = 8 output channels at one k, 16K deep, packed per layer as oc_tile → k (LeNet 7,813 words; CIFAR 10,028).
- **QPARAM:** one entry per output channel {q_bias int32, m (B bits), s (6 bits)}, 256 entries, width set by B.

## Controller

- Counter-only address generation.
- Token pipeline (valid, first, last, tile/oc ids, lane masks, pool phase) through BRAM read (2 cycles) → rotator → operand regs → array → drain → requant → pool → write.
- Descriptor sequencer: up to 8 layers, swaps ACT buffers per layer, snapshots per-layer cycles.
- Config checker raises error and refuses start on: pool with odd OH/OW, K < 8, buffer overflow, unsupported shape.

## CSR (AXI-Lite, 32-bit)

| Register | Contents |
|---|---|
| CTRL | start, soft_reset |
| STATUS | busy, done, error |
| N_LAYERS | number of layers |
| DESC[0..7] | IC, OC, IH, IW, KH, KW, OH, OW, WGT_BASE, QP_BASE, flags: relu_en, pool_en, out_raw, in_sel |
| TOTAL_CYC | 64-bit |
| MAC_ACTIVE | 64-bit |
| STALL | 64-bit |
| LAYER_CYC[0..7] | per-layer cycles |
| LOGIT[0..15] | int32 |

## Cycle model (V2)

cycles(layer) = T\*K + C_pipe, where C_pipe is a fixed per-layer fill/flush constant set by the final RTL pipeline depth.

Compute-only T\*K:

- LeNet: 2,800 / 6,000 / 6,000 / 1,320 / 168 = 16,288
- CIFAR: 33,600 / 64,000 / 6,400 / 128 = 104,128

RTL simulation must match the model exactly per layer.

## Integration

- Plain Verilog wrapper top.
- Block design: Zynq MPSoC (KV260 preset), SmartConnect, AXI-Lite to CSR, 4 AXI BRAM controllers (ACT0, ACT1, WGT, QPARAM), proc_sys_reset.
- 200 MHz target from pl_clk0 (fallbacks 150 and 100). Single clock domain, no external pins.
- Board: .bit + .hwh overlay under PYNQ.

## Estimated resources (pre-synthesis estimates)

DSP ≈ 64 + requant lanes (≤ ~8%), BRAM36 ≈ 49 (≈34%), LUT < 20%.

## Fallbacks (not redesigns)

Clock 200→150→100 MHz; extra requant pipeline stages; pool bypass with PS pooling; n_layers=1 with the PS driving each layer.
