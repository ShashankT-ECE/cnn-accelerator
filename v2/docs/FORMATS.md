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
time. All fields are unsigned. Up to 8 layers (N_LAYERS <= 8). **CSR byte offsets of DESC[n][w]
are deferred** and will be fixed with the CSR RTL.

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
