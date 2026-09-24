# vectors — $readmemh test vectors for the gos_ RTL

Produced only by `v2/scripts/gen_vectors.py` from the `v2/model` golden (Python, `.venv`).
`generated/` is gitignored; `MANIFEST.json` is committed and pins every generated file by SHA256.

```
cd <repo> && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v2/scripts/gen_vectors.py --require-clean
```

Options: `--out DIR` (default `v2/vectors/generated`; the directory is wiped on regeneration, and
only if it is empty or carries the `.gen_vectors_output` marker), `--manifest PATH` (default
`v2/vectors/MANIFEST.json` for the default `--out`, else `DIR/MANIFEST.json`), `--quick` (reduced
sizes, for tests), `--require-clean` (refuse to run on a dirty git tree). A dirty run prints a
warning and records `git_dirty: true`; vectors of record are generated from a clean, committed tree.

**Status of the layouts.** Memory images (ACT/WGT/QPARAM), descriptors and LOGIT follow the
host <-> RTL contract in `v2/docs/FORMATS.md`. The **unit-vector layouts (section 3) and the
address-stream record layouts (section 4.3) are PROVISIONAL** (approved decision 5): they will
change when the RTL module interfaces are fixed. The generator's `*_LAYOUT` tables are the single
source for the layouts below; `MANIFEST.json` → `layouts` repeats them.

## 1. Common file format

- One word per line, lowercase hex, MSB first, fixed width (`width_bits/4` digits), no `@`
  address markers, no comments; line n is address n. Written by `gos_pack.write_hex`.
- Widths: 8 (2 digits), 32 (8), 64 (16), and composite records 160 (40), 192 (48), 256 (64).
- Signed values (int8, int32) are two's complement in their field width.
- Composite records: one record per line, fields at the documented bit positions; every bit not
  listed is 0 (tested).
- Determinism: fixed seeds (`SEEDS` in the script, recorded in the manifest), fixed image indices
  test[0..N-1], no timestamps; two runs give identical file hashes (tested).

## 2. Directory layout

```
generated/
  unit/pe/        pe_cycles.hex  pe_results.hex
  unit/array/     array_a.hex  array_w.hex  array_ctl.hex  array_k.hex  array_acc.hex
  unit/rotator/   rot_banks.hex  rot_kx.hex  rot_rows.hex
  unit/requant/   lenet5_requant.hex  cifar10_requant.hex
  unit/pool/      pool_cols.hex
  <net>/          wgt.hex  qparam.hex  qparam_e.hex  qparam_o.hex  desc.hex  n_layers.hex
  <net>/layers/L<i>_<name>/
                  desc.hex  expect_tk.hex  issue.hex  drain.hex  [act_out_mask.hex]
                  img<n>/act_in.hex   img<n>/act_out.hex | img<n>/logit16.hex
  <net>/net/      pred.hex  label.hex
                  img<n>/act0.hex  img<n>/logit10.hex  img<n>/logit16.hex
```

`<net>` ∈ {lenet5, cifar10}; `L<i>_<name>` is layer i (0-based) with its NET_CONFIGS name
(lenet5: L0_conv1 L1_conv3 L2_conv5 L3_fc1 L4_fc2; cifar10: L0_conv1 L1_conv2 L2_conv3 L3_fc).
Full mode: layer vectors for test[0..2] (`img0..img2`), network vectors for test[0..9]
(`img0..img9`). `--quick`: 1 and 2 images.

## 3. Unit vectors (PROVISIONAL layouts)

### 3.1 PE — `unit/pe/`

`acc <= first ? a*w : acc + a*w`. Sequences are back-to-back (no idle cycles): every line is one
cycle. Contents: all 36 length-1 corner products of {-128,-127,-1,0,1,127}², four K=800 extreme
sequences (acc reaches ±13,107,200 / -13,004,800), one K=25 extreme, then seeded random sequences
(lengths 1, 2-32, 33-200, 201-800; values int8 with 15 % corner values).

`pe_cycles.hex` (64 bit, one line per cycle):

| bits | field |
|---|---|
| 31:0 | expected acc after this cycle (int32) |
| 39:32 | w (int8) |
| 47:40 | a (int8) |
| 48 | first |
| 49 | last |

`pe_results.hex` (32 bit): expected acc at the `last` cycle of each sequence, in order.

### 3.2 Array — `unit/array/` (one file per field)

Tiles back-to-back; one line per cycle (k) in `array_a/w/ctl`. Row r = activation r (output x),
column j = weight j (output channel). Tiles: for each K in {1, 25, 800}: an extremes pattern
(a = [-128,127,-128,127,-128,-1,0,1], w = [-128,-128,127,127,-1,1,0,-128] every k), all
(-128,-128) (acc max 13,107,200 at K=800), all (127,-128) (min), and a random tile; then
seeded random tiles (K 1-64, some 65-800).

| file | width | line | content |
|---|---|---|---|
| array_a.hex | 64 | per cycle | byte r (bits 8r+7:8r) = a[r], int8 |
| array_w.hex | 64 | per cycle | byte j (bits 8j+7:8j) = w[j], int8 |
| array_ctl.hex | 8 | per cycle | bit 0 first, bit 1 last |
| array_k.hex | 32 | per tile | K of the tile |
| array_acc.hex | 32 | tile*64 + r*8 + j | expected acc[r][j] (int32) at `last` |

### 3.3 Rotator — `unit/rotator/`

Rule (ARCH_SPEC Memory): row r = bank (r + kx) mod 8, i.e. bank b → row (b - kx) mod 8. For every
kx 0..7: 5 fixed patterns (incl. `0706050403020100`) + seeded random 64-bit bank words.

| file | width | content |
|---|---|---|
| rot_banks.hex | 64 | byte b = bank b |
| rot_kx.hex | 8 | kx (0..7) |
| rot_rows.hex | 64 | expected, byte r = row r |

### 3.4 Requant lane — `unit/requant/<net>_requant.hex` (160 bit)

Per net: 100,000 seeded rows, channel uniform over **all** requantized channels of the net
(QPARAM channel index), v uniform over the reachable range [-K·16384 + q_bias, K·16384 + q_bias]
on even rows and over the exact region |v·M| ≤ 130 (requant_check.regions) on odd rows; then
**every near-tie** v of every channel: exact |v·M − (n + ½)| < 1e-9 with Fraction(float64 M),
searched over the whole exact region (these include the Step 2.1 values lenet5 conv5 ch82
v = ±55930, fc1 ch54 v = ±31044, cifar10 (r2) conv3 ch57 v = ±36475, asserted present). (q_bias, m, s)
come from `gos_pack.load_params` (hw_requant.npz via NET_CONFIGS). Expected q = requant_check
hardware arithmetic (`p = v·m; q = (p + 2^(s-1) − 1 + ((p >> s) & 1)) >>> s; clip`), asserted equal
to the legacy float64 `requantize(v, M)` and to the Python-int formula for every row.

| bits | field |
|---|---|
| 31:0 | acc (int32) = v − q_bias (the lane's array input) |
| 63:32 | q_bias (int32) |
| 95:64 | m (uint32, B = 32) |
| 127:96 | v = acc + q_bias (int32) |
| 133:128 | s (uint6) |
| 134 | near_tie (row is from the near-tie search) |
| 135 | exact_region (\|v·M\| ≤ 130) |
| 143:136 | q: expected int8 after clip, **pre-ReLU** |
| 151:144 | q_relu: expected int8 after clip and ReLU (max(q, 0)) |
| 159:152 | QPARAM channel index i (= QP_BASE[layer] + oc; the `<net>/qparam*.hex` entry) |

Both q and q_relu are given because ARCH_SPEC does not say in which stage ReLU happens
(the tile model applies it after the requant lanes; relu_en is per layer).

### 3.5 Pool — `unit/pool/pool_cols.hex` (192 bit)

One line per drain column j of a case (8 lines per case, j = 0..7). A case is a dy=0 tile and a
dy=1 tile (8x8 int8, [row r][column j]). Expected: vertical max (dy0, dy1) then horizontal pair
max of rows (2i, 2i+1), i = 0..3. Byte enable as the drain stream: pooled byte i goes to bank
4·half + i and is enabled iff 2i < rem (rem = OW − ox0; even under pooling: 2, 4, 6, ≥ 8).
Cases cycle half ∈ {0,1}, signed_data ∈ {0,1}, rem ∈ {2,4,6,8, random even 10..30}, with
all-min / all-127 / dy0-dominant cases.

| bits | field |
|---|---|
| 63:0 | dy0 column j: byte r = dy0[r][j] (int8) |
| 127:64 | dy1 column j: byte r = dy1[r][j] (int8) |
| 159:128 | expected h: byte i = max(dy0[2i][j], dy0[2i+1][j], dy1[2i][j], dy1[2i+1][j]) (int8) |
| 167:160 | be: 8-bit bank byte enable (tail-masked, half-shifted) |
| 171:168 | j |
| 172 | half (= ox_tile & 1) |
| 173 | signed_data: 0 = values 0..127 (post-ReLU), 1 = full int8 (signed compare) |
| 183:176 | rem |

h is given for all 4 bytes, also for masked ones (they are not written). The bank placement of the
64-bit write data (tile model: data[b] = h[b & 3]) is not fixed by the spec; only enabled bytes
are defined.

## 4. Net and layer vectors (FORMATS.md contract)

### 4.1 Net-level memory images — `<net>/`

| file | width | content |
|---|---|---|
| wgt.hex | 64 | full WGT image from word 0, all layers (`gos_pack.pack_wgt`); LeNet 7,813 / CIFAR 10,028 words |
| qparam.hex | 64 | QPARAM combined PS view, 2 words per channel (`pack_qparam`): 472 / 276 words |
| qparam_e.hex | 64 | bank QP_E[i] = combined word 2i = {m[63:32], q_bias[31:0]} |
| qparam_o.hex | 64 | bank QP_O[i] = combined word 2i+1 = {0, s[5:0]} |
| desc.hex | 32 | all descriptors: line 16·layer + w = DESC[layer] word w |
| n_layers.hex | 32 | N_LAYERS |

Files hold only the used words (load from address 0; the rest of each memory is don't-care).
The layer vectors use these **full-net** WGT/QPARAM images (no per-layer slices), so every
descriptor's WGT_BASE / QP_BASE is valid as is.

### 4.2 Layer vectors — `<net>/layers/L<i>_<name>/`

| file | width | content |
|---|---|---|
| desc.hex | 32 | this layer's 16 descriptor words (identical to lines 16i..16i+15 of `<net>/desc.hex`) |
| expect_tk.hex | 32 | 3 lines: T, K, T·K (gos_cycle_model, compute only; C_PIPE excluded) |
| issue.hex | 256 | issue stream, one record per core cycle, T·K lines (4.3) |
| drain.hex | 160 | drain / write-event stream, 8 records per tile, 8·T lines (4.3) |
| act_out_mask.hex | 8 | non-final layers: per output word 0..OUT_END, byte-enable mask of the bytes the layer writes (bank b = bit b) |
| img<n>/act_in.hex | 64 | input map in ACT[in_sel] PS words 0..IN_END (golden input of this layer for test[n]) |
| img<n>/act_out.hex | 64 | non-final: expected ACT[!in_sel] PS words 0..OUT_END; bytes outside act_out_mask are 0 here and are **don't-care** in the RTL buffer (compare under the mask) |
| img<n>/logit16.hex | 32 | final layer: expected LOGIT[0..15] int32; entries ≥ OC are 0 and are not written by the hardware (don't-care) |

The stream files are data independent (once per layer, approved decision 3). Generation
cross-check: for each layer and image the tile model (`gos_tile_model.run_layer_mem`) is run on
memories built from the **read-back hex files**, with seeded garbage in every other word, the
whole output buffer and LOGIT; it must reproduce act_out under the mask (and leave unmasked bytes,
words > OUT_END and the input buffer untouched) or LOGIT[0..OC-1] (LOGIT[OC..15] untouched).
The stream files are round-tripped and T·K / T / 8T are asserted against the cycle model.

### 4.3 Address-stream records (PROVISIONAL)

`issue.hex` (256 bit), one per cycle, from `gos_addr_stream` (field names as its ISSUE_DTYPE):

| bits | field |
|---|---|
| 16b+15:16b (b = 0..7) | rd_addr[b]: full counter value `rowbase + ox0/8 + (b < kx)`; the ACT port uses bits 11:0 |
| 143:128 | wgt_addr |
| 151:144 | row_mask (bit r: ox0 + r < OW) |
| 155:152 | rot (= kx) |
| 156 | first (k == 0) |
| 157 | last (k == K−1) |
| 158 | dy |
| 175:160 | k |
| 191:176 | ic |
| 199:192 | ky |
| 207:200 | kx |
| 215:208 | ox_tile |
| 223:216 | oy |
| 231:224 | oc_tile |
| 255:240 | tile |

`drain.hex` (160 bit), 8 per tile in drain order j = 0..7 (every drain step, including those that
write nothing; DRAIN_DTYPE):

| bits | field |
|---|---|
| 15:0 | word (ACT write address in ACT[buf]) |
| 23:16 | be (bank byte enables; 0 = no ACT write) |
| 31:24 | row_mask |
| 47:32 | qp_idx (QPARAM channel read by the requant lanes; port uses bits 7:0) |
| 63:48 | ch (output channel = oc_tile·8 + j) |
| 67:64 | j |
| 71:68 | kind: 0 ACT write, 1 pool store (dy = 0, no write), 2 LOGIT |
| 72 | we |
| 73 | ch_valid (ch < OC) |
| 74 | buf (= !in_sel) |
| 75 | dy |
| 87:80 | logit_idx (kind 2) |
| 95:88 | ox_tile |
| 103:96 | out_row |
| 111:104 | oy |
| 119:112 | oc_tile |
| 143:128 | tile |

### 4.4 Network vectors — `<net>/net/`

For test[0..9] (quick: 0..1). Use with `<net>/wgt.hex`, `qparam*.hex`, `desc.hex`, `n_layers.hex`.

| file | width | content |
|---|---|---|
| img<n>/act0.hex | 64 | initial ACT0 image: layer-0 input, PS words 0..IN_END |
| img<n>/logit10.hex | 32 | expected LOGIT[0..OC-1] (int32 raw v, OC = 10) |
| img<n>/logit16.hex | 32 | same, padded to 16 entries with 0 (entries ≥ OC don't-care) |
| pred.hex | 8 | expected prediction per image (PS float32 dequant + argmax, `final_layer`) |
| label.hex | 8 | dataset label per image (predictions may differ from labels: model accuracy < 100 %) |

Generation cross-check: the tile model run layer by layer from the read-back files (garbage
elsewhere) and `gos_tile_model.run_net_mem` both reproduce LOGIT, and the prediction equals the
golden (`gos_golden.run_net`).

## 5. MANIFEST.json

`generator`, `git_commit`, `git_dirty`, `mode` (full/quick), `generated_dir`, `sizes`, `seeds`,
`image_indices`, `params` (quant_params / hw_requant path + SHA256 per net, from NET_CONFIGS),
`layouts`, `summary` (per-unit counts, near-tie list, per-layer T/K/T·K/streams, predictions and
labels), `n_files`, `total_bytes`, and `files`: for every generated file its path (relative to
`generated_dir`), `sha256`, `bytes`, `lines`, `width_bits`, `description`. No timestamps.

Tests: `v2/model/tests/test_gen_vectors.py` (quick generation in a tmp dir: format, read-back vs
golden, manifest hashes, near-ties, requant == legacy, unit recomputation, determinism; `slow`:
full-size determinism).
