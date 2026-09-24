# V2 Memory and Descriptor Formats (host <-> RTL contract)

**Status:** Step 2.2. Implements `ARCH_SPEC.md` "Memory", "Tiling", "Controller", "CSR".
**Reference implementation:** `v2/model/gos_pack.py` (tests: `v2/model/tests/test_gos_pack.py`).
Any change here must be made in `gos_pack.py` and its tests first.

Conventions: bit `[n]` is bit n of an unsigned word, bit 0 = LSB. int8 values are stored as
two's-complement bytes; int32 as two's-complement 32-bit. "Word" means one BRAM word of the named
memory. `ceil(a/8)` is computed on the host only.

## 1. ACT: activation buffers ACT0 / ACT1

Each buffer is 8 banks x 8 bit x 4096 words (a custom TDP: 64-bit port A for the PS, 8 independent
8-bit ports B for the core).

### Address formula

A map with C channels, height H and width W, stored from word 0:

```
WPR   = ceil(W/8)                       words per map row
bank  = x mod 8          (= x[2:0])
word  = (c*H + y)*WPR + (x >> 3)
depth = C*H*WPR                         words occupied (must be <= 4096)
```

### PS view (port A)

64-bit word `w`, byte lane `b` (bits `8b+7 : 8b`) = bank `b` at word `w`:

```
 63     56 55     48 47     40 39     32 31     24 23     16 15      8 7       0
+---------+---------+---------+---------+---------+---------+---------+---------+
| bank 7  | bank 6  | bank 5  | bank 4  | bank 3  | bank 2  | bank 1  | bank 0  |
| x=8j+7  | x=8j+6  | x=8j+5  | x=8j+4  | x=8j+3  | x=8j+2  | x=8j+1  | x=8j+0  |
+---------+---------+---------+---------+---------+---------+---------+---------+
  word (c*H + y)*WPR + j holds pixels x = 8j..8j+7 of row y of channel c
```

### Buffer usage per layer

- The input image is written to **ACT0** from word 0.
- Layer `i` reads `ACT[in_sel]` from word 0 and writes `ACT[!in_sel]` from word 0, with
  `in_sel = i mod 2` (0, 1, 0, ...). The output map of layer i is the input map of layer i+1
  (`gos_pack.make_descriptors` asserts IC/IH/IW of layer i+1 = OC/OUT_H/OUT_W of layer i).
- A pooled output is stored as a map of height OH/2 and width OW/2 in the same layout. The pool
  writes 4 bytes per column (banks 0-3 or 4-7 of one word) using byte enables.
- The final layer (`out_raw`) writes no ACT: its raw int32 values go to LOGIT (section 4).

### 1x1 maps (FC input)

With H = W = 1: WPR = 1, bank = 0, word = (c*1 + 0)*1 + 0 = **c**. Channel c of a 1x1 map is bank 0,
word c, which is the FC input layout. Since conv3 outputs are 1x1, no flatten reorder is needed
(ARCH_SPEC "Tiling"). Banks 1-7 of those words are unwritten.

### Unwritten bytes

- The model (`pack_act`) initializes the whole buffer to 0. Bytes not covered by a map (x tail
  of each row when W is not a multiple of 8, banks 1-7 of 1x1 maps, and words >= depth) are 0 in
  model images.
- **The RTL must not rely on unwritten bytes.** All core writes use per-bank write enables, and
  a buffer keeps whatever an earlier layer or inference left there. Values the core reads from
  unwritten bytes may only feed masked rows (x tail) or OC-tail lanes, which are never written.
- Note (from the conflict-free read, ARCH_SPEC "Memory"): the rotator reads bank b at
  `rowbase + ox0/8 + (b < kx)`. For masked rows of the last x tile this can address the word
  after the row, and in the last row of the last channel the word **IN_END + 1**. `gos_pack`
  computes the maximum as 128/168/80 (LeNet conv1/conv3/conv5) and 384/896/160 (CIFAR
  conv1/conv2/conv3), all < 4096, and the data is discarded. If IN_END = 4095 the 12-bit address
  wraps to 0 (still masked data). The checker's `IN_END <= 4095` bound is therefore sufficient.

### Worked example

LeNet conv3 input (C=6, H=14, W=14, WPR=2), element (c=2, y=3, x=10):
bank = 10 mod 8 = 2, word = (2*14 + 3)*2 + (10 >> 3) = 63. The PS reads it as bits 23:16 of
64-bit word 63 of ACT1 (conv3 has in_sel = 1). `act_addr(2,3,10,14,14) == (2, 63)`.
1x1 example: `act_addr(119,0,0,1,1) == (0, 119)` (LeNet fc1 input channel 119).

## 2. WGT: weights (64-bit x 16384)

For layer L, with `K = IC*KH*KW` and `k = (ic*KH + ky)*KW + kx` (kx innermost; FC: KH=KW=1, K=IC):

```
word = WGT_BASE[L] + oc_tile*K + k          oc_tile = 0 .. ceil(OC/8)-1
byte lane j (bits 8j+7:8j) = int8 weight of output channel oc_tile*8 + j  (0 if >= OC)
```

```
 63     56 55     48 47     40 39     32 31     24 23     16 15      8 7       0
+---------+---------+---------+---------+---------+---------+---------+---------+
| oc=8t+7 | oc=8t+6 | oc=8t+5 | oc=8t+4 | oc=8t+3 | oc=8t+2 | oc=8t+1 | oc=8t+0 |  at one k
+---------+---------+---------+---------+---------+---------+---------+---------+
```

Layers are packed consecutively from WGT_BASE[0] = 0: `WGT_BASE[L+1] = WGT_BASE[L] + ceil(OC/8)*K`.
Totals (asserted <= 16384): LeNet-5 7,813 words, CIFAR-10 10,028 words.

Worked example: LeNet conv3 (WGT_BASE = 25, K = 150), weight (oc=11, ic=4, ky=2, kx=3):
oc_tile = 1, lane j = 3, k = (4*5 + 2)*5 + 3 = 113, word = 25 + 1*150 + 113 = 288.
The model gives word 288 = `0xd52de0e6306fb081`, whose lane 3 is `0x30` = 48 = q_w[11,4,2,3].
OC tail example: LeNet fc2 (OC = 10) tile 1 uses lanes 0-1 (oc 8, 9); lanes 2-7 are 0
(word WGT_BASE+84+0 = `0x0000000000003918`: oc 9 = 0x39 = 57, oc 8 = 0x18 = 24).

## 3. QPARAM: requant parameters (2 x 64-bit words per channel)

One channel entry per output channel of every layer. Channel index `i = QP_BASE[L] + oc`. Layers
are packed consecutively from QP_BASE[0] = 0; at most 256 channels (512 words).

```
word 2i   : bits 63:32 = m[31:0] (unsigned, B = 32)   bits 31:0 = q_bias[31:0] (two's complement)
word 2i+1 : bits 63:6  = 0                              bits 5:0  = s[5:0] (unsigned)
```

Hardware stores the even and odd words in **two 64-bit BRAMs read in parallel** at address i:

| view | depth | contents |
|---|---|---|
| combined (PS / `pack_qparam`) | 512 x 64 | word 2i = {m, q_bias}, word 2i+1 = {0, s} |
| bank QP_E (`qparam_banks()[0]`) | 256 x 64 | QP_E[i] = combined word 2i |
| bank QP_O (`qparam_banks()[1]`) | 256 x 64 | QP_O[i] = combined word 2i+1 |

Host asserts: m < 2^32, s < 64 (B = 32 per DECISIONS D1 outcome).

**Final layer (APPROVED):** it also gets QPARAM entries, with its q_bias and m = 0, s = 0. The
core adds q_bias to form `v = acc + q_bias` and writes v to LOGIT; m and s are unused (out_raw).
Channel counts: LeNet-5 236 (472 words), CIFAR-10 138 (276 words).

Worked example: LeNet conv3 oc 5, i = QP_BASE 6 + 5 = 11. Word 22 = `0x9b2f1e5bffffff48`
(m = 0x9b2f1e5b = 2,603,556,443, q_bias = 0xffffff48 = -184), word 23 = `0x29` (s = 41).
Final layer: LeNet fc2 oc 0, i = 226. Word 452 = `0x00000000fffffeee` (m = 0, q_bias = -274),
word 453 = `0x0` (s = 0).

## 4. LOGIT[0..15]

The final layer (out_raw = 1) writes the raw int32 `v = acc + q_bias` of class oc to `LOGIT[oc]`
(two's complement, AXI-Lite readable). Requires OC <= 16 (asserted on the host and checked by the
RTL config checker). LOGIT[OC..15] are not written. The PS applies the reference float32
dequantization and argmax (DECISIONS D2).

## 5. Layer DESCRIPTOR (16 x 32-bit words per layer)

The host precomputes every derived field so the RTL needs no multipliers, not even at config
time. All fields are unsigned. Up to 8 layers (N_LAYERS <= 8). CSR byte offset of DESC[n][w]:
0x100 + 0x40*n + 4*w (see "CSR register map").

| word | bits 31:16 | bits 15:0 | notes |
|---|---|---|---|
| w0 | OC | IC | |
| w1 | IW | IH | |
| w2 | 0 | KW[15:8], KH[7:0] | 8-bit fields; bits 31:16 = 0 |
| w3 | OW | OH | conv output, before pooling |
| w4 | WGT_BASE[31:0] | | |
| w5 | QP_BASE[31:0] | | |
| w6 | 0 | flags | b0 relu_en, b1 pool_en, b2 out_raw (final layer only), b3 in_sel; bits 31:4 = 0 |
| w7 | K[31:0] | | K = IC*KH*KW |
| w8 | IN_PLANE | IN_WPR | IN_WPR = ceil(IW/8), IN_PLANE = IH*IN_WPR |
| w9 | OW_TILES | OC_TILES | ceil(OC/8), ceil(OW/8) |
| w10 | OUT_H | OUT_W | pool ? OH/2 : OH ; pool ? OW/2 : OW |
| w11 | OUT_PLANE | OUT_WPR | ceil(OUT_W/8), OUT_H*OUT_WPR |
| w12 | WGT_END[31:0] | | WGT_BASE + OC_TILES*K - 1 |
| w13 | IN_END[31:0] | | IC*IN_PLANE - 1 |
| w14 | OUT_END[31:0] | | OC*OUT_PLANE - 1 |
| w15 | QP_END[31:0] | | QP_BASE + OC - 1 |

Host rules (`gos_pack.encode_descriptor` / `make_descriptors`): each field must fit its width
(16-bit fields < 65536, KH/KW < 256), and derived fields are recomputed from the raw fields and
must match. `decode_descriptor(words)` returns all named fields; `descriptors_json(net)` emits
the named fields (raw + derived) plus the 16 words.

### RTL config checker (`gos_pack.check_descriptor`)

The checker raises `error` and refuses `start` if any layer fails. The model uses only field
extraction (shifts/masks, i.e. wiring) and comparisons or single-bit tests. It has **no
multiplication, division or modulo**; a test enforces this on the source (AST). Rules:

| rule | reject when |
|---|---|
| pool parity | pool_en and OH[0] = 1; pool_en and OW[0] = 1 |
| minimum K | K < 8 |
| WGT bound | WGT_END > 16383 |
| ACT bounds | IN_END > 4095; OUT_END > 4095 |
| QPARAM bound | QP_END > 255 |
| nonzero | any of IC, OC, IH, IW, KH, KW, OH, OW, K, IN_WPR, IN_PLANE, OC_TILES, OW_TILES, OUT_W, OUT_H, OUT_WPR, OUT_PLANE = 0 |
| shape | OH > IH; OW > IW |
| logits | out_raw and OC > 16 |

**Rule ids and ERR_CODE (Step 4, `gos_cfg_check.sv`; Python: `gos_pack.check_descriptor_rules`,
`check_descriptor_code`, `job_err_code`).** Priority = ascending id.

| id | rule | id | rule |
|---|---|---|---|
| 1 | pool_en and OH odd | 15 | OW == 0 |
| 2 | pool_en and OW odd | 16 | K == 0 (never reported first: rule 3 fires) |
| 3 | K < 8 | 17 | IN_WPR == 0 |
| 4 | WGT_END > 16383 | 18 | IN_PLANE == 0 |
| 5 | IN_END > 4095 | 19 | OC_TILES == 0 |
| 6 | OUT_END > 4095 | 20 | OW_TILES == 0 |
| 7 | QP_END > 255 | 21 | OUT_W == 0 |
| 8 | IC == 0 | 22 | OUT_H == 0 |
| 9 | OC == 0 | 23 | OUT_WPR == 0 |
| 10 | IH == 0 | 24 | OUT_PLANE == 0 |
| 11 | IW == 0 | 25 | OH > IH |
| 12 | KH == 0 | 26 | OW > IW |
| 13 | KW == 0 | 27 | out_raw and OC > 16 |
| 14 | OH == 0 | 32 | N_LAYERS not in 1..8 (job level, reported with layer 0) |

On `start`, every descriptor l < N_LAYERS is checked in parallel (S_CHK1: rule vectors
registered; S_CHK2: priority encode). If any rule fails the job is refused: STATUS.error = 1,
STATUS.busy = 0, and `ERR_CODE = {16'b0, rule_id[7:0], 5'b0, layer[2:0]}` for the first failing
layer (lowest l) and its lowest failing rule id. Descriptor slots l >= N_LAYERS are ignored.

**What the RTL checker cannot verify without multipliers or adders** (the host is responsible;
`gos_pack` asserts all of these before emitting words):

- K = IC*KH*KW; IN_WPR = ceil(IW/8); IN_PLANE = IH*IN_WPR; OC_TILES = ceil(OC/8);
  OW_TILES = ceil(OW/8); OUT_W/OUT_H = OW/2, OH/2 (or OW, OH); OUT_WPR = ceil(OUT_W/8);
  OUT_PLANE = OUT_H*OUT_WPR.
- WGT_END, IN_END, OUT_END, QP_END being consistent with their bases and sizes, and so that the
  bound checks on the END fields reflect the real footprint.
- OH = IH-KH+1, OW = IW-KW+1 (VALID, stride 1) and KH <= IH, KW <= IW.
- Layer chaining (IC/IH/IW of layer i+1 = OC/OUT_H/OUT_W of layer i; in_sel alternation), and
  that the layers do not overlap in WGT or QPARAM.
- out_raw only on the last layer; relu_en = 0 with out_raw.
- Reserved bits (w2[31:16], w6[31:4]) are 0. The checker ignores them.

## 6. Per-net tables

The tables below were produced by `gos_pack.layer_table()` / `descriptor_table()` and pasted
unchanged. **Do not edit by hand.** Regenerate with:

```
cd v2/model && ../../.venv/bin/python gos_pack.py tables
```

(`gos_pack.py json [net]` prints the descriptors as JSON; `gos_pack.py usage` prints `memory_usage`.)
Depths are in ACT words. "unpooled out depth" is the no-pool fallback (PS pooling), which must also
fit.

### lenet5

| layer | IC | OC | IH | IW | OH | OW | K | WGT_BASE | WGT words | QP_BASE | in ACT depth | out ACT depth | unpooled out depth | in_sel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conv1 | 1 | 6 | 32 | 32 | 28 | 28 | 25 | 0 | 25 | 0 | 128 | 168 | 672 | 0 |
| conv3 | 6 | 16 | 14 | 14 | 10 | 10 | 150 | 25 | 300 | 6 | 168 | 80 | 320 | 1 |
| conv5 | 16 | 120 | 5 | 5 | 1 | 1 | 400 | 325 | 6000 | 22 | 80 | 120 | - | 0 |
| fc1 | 120 | 84 | 1 | 1 | 1 | 1 | 120 | 6325 | 1320 | 142 | 120 | 84 | - | 1 |
| fc2 | 84 | 10 | 1 | 1 | 1 | 1 | 84 | 7645 | 168 | 226 | 84 | LOGIT[0..9] | - | 0 |

Totals (lenet5): WGT 7813 / 16384 words; QPARAM 236 / 256 channels = 472 / 512 words; max ACT depth 672 / 4096 (pooled path 168); logits 10 / 16.

Descriptor words (hex, w0..w15):

| layer | w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w9 | w10 | w11 | w12 | w13 | w14 | w15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conv1 | 00060001 | 00200020 | 00000505 | 001C001C | 00000000 | 00000000 | 00000003 | 00000019 | 00800004 | 00040001 | 000E000E | 001C0002 | 00000018 | 0000007F | 000000A7 | 00000005 |
| conv3 | 00100006 | 000E000E | 00000505 | 000A000A | 00000019 | 00000006 | 0000000B | 00000096 | 001C0002 | 00020002 | 00050005 | 00050001 | 00000144 | 000000A7 | 0000004F | 00000015 |
| conv5 | 00780010 | 00050005 | 00000505 | 00010001 | 00000145 | 00000016 | 00000001 | 00000190 | 00050001 | 0001000F | 00010001 | 00010001 | 000018B4 | 0000004F | 00000077 | 0000008D |
| fc1 | 00540078 | 00010001 | 00000101 | 00010001 | 000018B5 | 0000008E | 00000009 | 00000078 | 00010001 | 0001000B | 00010001 | 00010001 | 00001DDC | 00000077 | 00000053 | 000000E1 |
| fc2 | 000A0054 | 00010001 | 00000101 | 00010001 | 00001DDD | 000000E2 | 00000004 | 00000054 | 00010001 | 00010002 | 00010001 | 00010001 | 00001E84 | 00000053 | 00000009 | 000000EB |

### cifar10

| layer | IC | OC | IH | IW | OH | OW | K | WGT_BASE | WGT words | QP_BASE | in ACT depth | out ACT depth | unpooled out depth | in_sel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conv1 | 3 | 32 | 32 | 32 | 28 | 28 | 75 | 0 | 300 | 0 | 384 | 896 | 3584 | 0 |
| conv2 | 32 | 32 | 14 | 14 | 10 | 10 | 800 | 300 | 3200 | 32 | 896 | 160 | 640 | 1 |
| conv3 | 32 | 64 | 5 | 5 | 1 | 1 | 800 | 3500 | 6400 | 64 | 160 | 64 | - | 0 |
| fc | 64 | 10 | 1 | 1 | 1 | 1 | 64 | 9900 | 128 | 128 | 64 | LOGIT[0..9] | - | 1 |

Totals (cifar10): WGT 10028 / 16384 words; QPARAM 138 / 256 channels = 276 / 512 words; max ACT depth 3584 / 4096 (pooled path 896); logits 10 / 16.

Descriptor words (hex, w0..w15):

| layer | w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w9 | w10 | w11 | w12 | w13 | w14 | w15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conv1 | 00200003 | 00200020 | 00000505 | 001C001C | 00000000 | 00000000 | 00000003 | 0000004B | 00800004 | 00040004 | 000E000E | 001C0002 | 0000012B | 0000017F | 0000037F | 0000001F |
| conv2 | 00200020 | 000E000E | 00000505 | 000A000A | 0000012C | 00000020 | 0000000B | 00000320 | 001C0002 | 00020004 | 00050005 | 00050001 | 00000DAB | 0000037F | 0000009F | 0000003F |
| conv3 | 00400020 | 00050005 | 00000505 | 00010001 | 00000DAC | 00000040 | 00000001 | 00000320 | 00050001 | 00010008 | 00010001 | 00010001 | 000026AB | 0000009F | 0000003F | 0000007F |
| fc | 000A0040 | 00010001 | 00000101 | 00010001 | 000026AC | 00000080 | 0000000C | 00000040 | 00010001 | 00010002 | 00010001 | 00010001 | 0000272B | 0000003F | 00000009 | 00000089 |

Descriptor worked example (LeNet conv1): w0 = `00060001` = OC 6 | IC 1; w6 = `00000003` = relu_en
+ pool_en, in_sel 0; w8 = `00800004` = IN_PLANE 128 | IN_WPR 4; w10 = `000E000E` = OUT_H 14 |
OUT_W 14; w14 = `000000A7` = OUT_END 167 = 6*14*2 - 1.

## 7. Hex files ($readmemh)

`gos_pack.write_hex(path, values, width_bits)` (also `write_hex64/32/8`) writes one word per line,
lowercase hex, MSB first, fixed width (16 digits for 64-bit, 8 for 32-bit, 2 for bytes), no
address markers, with the first line at address 0. Values must be unsigned, so int8 data is
passed as `.view(np.uint8)`. `read_hex` reads them back.

## CSR register map (AXI4-Lite, `gos_csr`)

Implemented by `v2/rtl/gos_csr.sv` (TB `v2/tb/tb_gos_csr.sv`). 32-bit registers, 12-bit byte
address (4 KB window). Word index = `addr[11:2]`; `addr[1:0]` are ignored (word-aligned access).

| offset | name | access | contents |
|---|---|---|---|
| 0x000 | CTRL | W | bit0 start, bit1 soft_reset: writing 1 (with `wstrb[0]`) emits a one-cycle pulse to the core; reads 0 |
| 0x004 | STATUS | RO | `{29'b0, error, done, busy}` |
| 0x008 | N_LAYERS | RW | bits [3:0] (`wstrb[0]`); bits 31:4 read 0 |
| 0x00C | ERR_CODE | RO | `{16'b0, rule_id[7:0], 5'b0, layer[2:0]}` (driven by the core) |
| 0x010 / 0x014 | TOTAL_CYC lo / hi | RO | 64-bit cycle counter |
| 0x018 / 0x01C | MAC_ACTIVE lo / hi | RO | 64-bit |
| 0x020 / 0x024 | STALL lo / hi | RO | 64-bit (0 by construction, D10) |
| 0x028 | PS_BUSY_VIOLATION | RO | `{24'b0, flags[7:0]}` sticky core error flags: bit0 ACT0 PS access while busy, bit1 ACT1 PS access while busy (D11-1), bit4 array overrun (`gos_array.err_overrun`, D10); other bits 0. Cleared by rst / soft_reset |
| 0x040 + 4*l | LAYER_CYC[l] | RO | l = 0..7 |
| 0x080 + 4*i | LOGIT[i] | RO | int32, i = 0..15 (§4) |
| 0x0F8 | VERSION | RO | `0x474F5302` ("GOS", 2). The empty shell returns `0x474F5300` at this offset |
| 0x0FC | BUILD_ID | RO | synthesis parameter `BUILD_ID` (8-hex-digit short git commit) |
| 0x100 + 0x40*l + 4*w | DESC[l][w] | RW | l = 0..7, w = 0..15 (0x100..0x2FC); word layout §5; byte strobes honoured |

All other offsets are unmapped.

Access rules:

- **Responses:** OKAY (`2'b00`) or SLVERR (`2'b10`). Unmapped read: SLVERR, rdata 0. Unmapped
  write: SLVERR, no effect. Write to an RO register: SLVERR, no effect. CTRL writes are always OKAY.
- **Busy lock:** a write to N_LAYERS or DESC while `busy = 1` returns SLVERR and has no effect (the
  core reads these registers live during a run). CTRL (start, soft_reset) is writable while busy.
- **Byte strobes:** honoured per byte on DESC; N_LAYERS and CTRL use only `wstrb[0]`. A write with
  `wstrb = 0` to a writable register is OKAY and changes nothing.
- **64-bit counters:** reading a lo word latches the live hi word of the same counter into a shadow
  register in the same cycle; reading the hi word returns the shadow (0 after reset). Read lo then
  hi for a tear-free 64-bit value; a hi read without a preceding lo read returns the last latched value.
- **Reset:** all CSR state (N_LAYERS, DESC, shadows, AXI state) is cleared by the PL reset (`rst`);
  soft_reset does not clear the CSR, it is only forwarded to the core.
- **AXI behaviour:** AW and W are independent (either may arrive first, or together); one
  outstanding write and one outstanding read at a time (a write and a read may be in flight
  concurrently); `bvalid`/`rvalid` are held until `bready`/`rready`. `awprot`/`arprot` are ignored.
  Latency: the write commits the cycle after both AW and W have been accepted (B follows);
  R is returned one cycle after the AR handshake. Core-side inputs are sampled when the read is served.

## PL top ports and address map (KV260 block design)

Built by `v2/vivado/bd_shell.tcl` (driver `v2/vivado/build_shell.sh`); the block design is
design-independent and reused unchanged for the real design (Step 6: `top=gos_top_wrapper`).
Board part `xilinx.com:kv260_som:part0:1.4` with `board_connections` to the KV260 carrier 1.3
(`som240_1_connector`); part `xck26-sfvc784-2LV-c`. No PL pins.

### Block design

- `zynq_ultra_ps_e` with the KV260 board preset; only **M_AXI_HPM0_FPD** is used (the preset also
  enables HPM1_FPD and pl_clk1; both are disabled). HPM0 data width: preset default (128 bit).
- **pl_clk0 requested 200 MHz.** The Vivado-configured actual frequency (`PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ`)
  is recorded per build in `impl_*.csv` / `summary.json`. With the carrier preset, PL0 from IOPLL
  cannot reach 200 MHz, so the script selects the source PLL (IOPLL or RPLL) whose actual frequency
  is closest to the request (build fails if > 1 % off). **On the board**, PYNQ programs only the
  PL0 divisors from the `.hwh` onto whatever source PLL the boot firmware set up, so the real
  pl_clk0 must be read back (`pynq.ps.Clocks.fclk0_mhz`; `v2/board/test_shell.py` prints and checks it).
- Single clock domain: pl_clk0 clocks HPM0, SmartConnect, the 4 BRAM controllers and the PL top.
  `proc_sys_reset` (ext_reset_in = pl_resetn0) → `peripheral_aresetn` → SmartConnect, controllers,
  PL top `rstn`.
- SmartConnect 1 SI → 5 MI: the PL top AXI4-Lite slave `S_AXI_CSR` and 4 **AXI BRAM Controllers**
  (`axi_bram_ctrl` 4.1: AXI4, 64-bit data, single port, no ECC, **READ_LATENCY = 1** — also the IP
  default; this matches the memories' `PS_RD_LAT = 1`, DECISIONS D11-2).
- Deliverables: `<name>.bit` and `<name>.hwh` with identical basenames (PYNQ overlay).

### Address map (PS view, HPM0_FPD)

| window | BD cell | base | size | PL top interface |
|---|---|---|---|---|
| CSR (AXI4-Lite) | `gos_top_0/S_AXI_CSR` | 0xA000_0000 | 4 KB | `s_axi_csr_*` |
| QPARAM | `qparam_ctrl` | 0xA001_0000 | 4 KB (512 x 64 bit) | `bram_qparam_*` |
| ACT0 | `act0_ctrl` | 0xA002_0000 | 32 KB (4096 x 64 bit) | `bram_act0_*` |
| ACT1 | `act1_ctrl` | 0xA004_0000 | 32 KB (4096 x 64 bit) | `bram_act1_*` |
| WGT | `wgt_ctrl` | 0xA008_0000 | 128 KB (16384 x 64 bit) | `bram_wgt_*` |

Memory word `w` (64 bit, layouts in §1–§3) is at byte offset `8*w` of its window. **Endianness:**
little-endian byte lanes; byte lane `k` (bits `8k+7:8k`) is at byte offset `8*w + k`. The host
accesses each 64-bit word as two 32-bit MMIO words: bits 31:0 at `8*w`, bits 63:32 at `8*w + 4`
(a 32-bit write reaches the memory as `we = 8'h0F` or `8'hF0`). Wider/burst accesses also work
(the controllers accept AXI4 bursts). QPARAM odd words read back as `{58'b0, s}` (§3).

### PL top ports (`v2/rtl/gos_top_ports.vh`)

One header holds the port list (with `X_INTERFACE_INFO` / `X_INTERFACE_PARAMETER` attributes) of
every PL top (`gos_shell_top` now, `gos_top_wrapper` in Step 6). The top is **plain Verilog** (BD
module reference) with `parameter [31:0] BUILD_ID`, which `bd_shell.tcl` sets from the short git
commit. The header body is guarded: a top writes `` `define GOS_TOP_PORTS `` before `module` and
`` `undef GOS_TOP_PORTS `` after the port list (without the guard, Vivado parses the bare header on
its own and reports CRITICAL WARNING HDL 9-1206).

| port | dir | width | notes |
|---|---|---|---|
| `clk` | in | 1 | pl_clk0; interface `clk` (ASSOCIATED_BUSIF S_AXI_CSR, ASSOCIATED_RESET rstn); FREQ_HZ propagated by the BD |
| `rstn` | in | 1 | active low (proc_sys_reset); internal sync active-high `rst` = registered `!rstn` |
| `s_axi_csr_aw{addr,valid,ready}` | in/in/out | 12/1/1 | AXI4-Lite `S_AXI_CSR`, 32-bit data, 12-bit byte address |
| `s_axi_csr_w{data,strb,valid,ready}` | in/in/in/out | 32/4/1/1 | |
| `s_axi_csr_b{resp,valid,ready}` | out/out/in | 2/1/1 | |
| `s_axi_csr_ar{addr,valid,ready}` | in/in/out | 12/1/1 | |
| `s_axi_csr_r{data,resp,valid,ready}` | out/out/out/in | 32/2/1/1 | no `awprot`/`arprot` ports |
| `bram_<m>_addr` | in | ACT0/ACT1 15, WGT 17, QPARAM 12 | **byte** address from the controller (`[2:0]` = 0); word = `addr >> 3` (ACT `[14:3]`, WGT `[16:3]`, QPARAM `[11:3]`) |
| `bram_<m>_clk`, `bram_<m>_rst` | in | 1, 1 | from the controller; **unused** (same clock as `clk`) |
| `bram_<m>_en` | in | 1 | → `ps_en` |
| `bram_<m>_we` | in | 8 | → `ps_we = \|we`, `ps_be = we` (D11-9) |
| `bram_<m>_din` | in | 64 | write data → `ps_wdata` |
| `bram_<m>_dout` | out | 64 | `ps_rdata`, valid 1 cycle after `en` (READ_LATENCY 1) |

`<m>` ∈ {`act0`, `act1`, `wgt`, `qparam`}; interfaces `BRAM_ACT0`, `BRAM_ACT1`, `BRAM_WGT`,
`BRAM_QPARAM` (`xilinx.com:interface:bram:1.0`, slave, MASTER_TYPE BRAM_CTRL, MEM_WIDTH 64,
READ_WRITE_MODE READ_WRITE, READ_LATENCY 1). The address widths equal the controllers' `bram_addr_a`
widths for the window sizes above (log2 of the window in bytes).

### Empty shell (`gos_shell_top`, Step 4)

Contains the real `gos_act_buf` x2, `gos_wgt_mem`, `gos_qparam_mem` (PS_RD_LAT = 1, accel_busy = 0,
accelerator ports tied off) and `gos_shell_scratch` (AXI4-Lite), **shell only**:

| offset | name | access | contents |
|---|---|---|---|
| 0x0E0..0x0EC | SCRATCH0..3 | RW | reset 0, byte strobes honoured (unmapped in the real CSR map) |
| 0x0F8 | VERSION | RO | `0x474F5300` (the real `gos_csr` returns `0x474F5302` here) |
| 0x0FC | BUILD_ID | RO | parameter `BUILD_ID` (8-hex-digit short git commit) |

All other shell offsets read 0 and ignore writes, always with an **OKAY** response (unlike the real
CSR, which answers SLVERR; so `test_shell.py --skip-scratch` is used with the real design).
