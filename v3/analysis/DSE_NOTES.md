# V3 Phase 0 array DSE — model notes (WS4)

Producer: `v3/analysis/dse.py` (uses `v3/model/cycle_model.py`, `v3/model/mem_model.py`,
`v3/model/nets.py`, `v3/model/common.py`). Outputs: `v3/results/dse.csv` (one row per net x config,
`layer = total`) and `v3/results/dse_layers.csv` (per layer, key configs only: no packing, `phase_split`,
dense, fused residual/GAP). No result number is written in this file; read them from the CSV columns
named below.

Labels: `cycles`, `util`, `latency_us_*` are **model (projected)** — the V2 latency constants are
RTL-derived for the 8x8 V2 core only; every V3 generalisation is a candidate architecture choice, not a
fact. `*_bram36`, `*_uram`, `*_bytes` are **estimates** (capacity only). DSP counts are arithmetic on
the configuration (`pe_dsps`, `requant_dsps`), not synthesis results. Device totals and `*_pct` columns
are filled only when `v3/results/device_resources.csv` (Vivado part query) exists.

## Regression anchor

`v3/model/tests/test_cycle_model.py::test_v2_regression`: with R = C = 8, row mapping `x`, VALID,
stride 1, OS only, the model reproduces `v2/results/cycle_model.csv` (`T`, per-layer `cycles`, and the
total) for lenet5 and cifar10. The V2 file is read as data; nothing in `v3/` imports `v2/`.

## Assumptions (P1..P15 = `cycle_model.ASSUMPTIONS`; column `assumptions` in dse.csv)

- **P1 OS tiling / pipeline.** R rows = output positions, C columns = output channels, one k per cycle.
  `C_PIPE(D) = C_PIPE_V2 - (L_DRAIN_V2 - 1) + (D - 1)` as in `v2/analysis/projection_16x16.py`; all other
  V2 latencies kept; `extra_pipe_stages` (default 0) adds one cycle per layer per stage a larger array
  might need (broadcast fan-out). Column `c_pipe` in dse_layers.csv.
- **P2 Drain.** V2 drains one column (one OC x R positions) per cycle, so `D = C` and the requant has R
  lanes (`requant_lanes`). Tiles are spaced by `max(issue, D)`: tiles whose issue is shorter than the
  drain stall (`stall`, `drain_hidden`, `layers_drain_stalled`). Note: the task brief said "C requant
  lanes"; from the V2 drain scheme it is R lanes (identical for 8x8); `requant_lanes` is a parameter.
- **P3 Row mapping.** `x`: R consecutive ox of one output row, ACT bank = x mod R + rotator (V2).
  `xy`: R consecutive flattened positions, ACT bank = flattened index mod R + rotator; conflict-free only
  when IW = OW*stride (SAME, or phase-split for stride 2), so VALID layers fall back to `x`
  (`row_map_used`). `xy` changes the ACT layout (whole maps padded to R instead of rows).
- **P4 SAME padding.** Zero-fill by the read path (masked bank read) at no cycle cost;
  `pad_penalty_per_tile` exists for a per-tile cost (DSE default 0).
- **P5 Stride 2.** `half_rate`: s cycles per k-step (R strided outputs touch 2R input x, two per bank);
  `phase_split`: the producer writes an even/odd de-interleaved layout (layout flag), no penalty;
  `dual_port` (modelled, not swept): two reads per bank per cycle.
- **P6 Residual.** `fused`: the skip tensor is read in the drain/requant path, one value per lane per drain
  cycle (`skip_read_bytes_per_cycle`); an option-A strided skip under `half_rate` doubles the drain.
  `separate`: an element-wise pass `ceil(OC*OH*OW/eltwise_lanes) + eltwise_pipe` (rows with
  `residual = separate`, column `extra_pass_cycles`). The numeric contract of the add is not modelled.
- **P7 GAP.** `fused` into the last conv's drain (`gap_fused_cycles`, default 0) vs a separate pass.
- **P8 Split-K (B1).** S | R rows take k-slices (k interleaved over slices), R/S positions per tile,
  `issue = ceil(K/S)`, a log2(S) adder tree on the drain path (+log2 S per layer). It needs S*C distinct
  weight bytes per cycle (`splitk_wgt_bytes_per_cycle_max`) and an ic-interleaved input layout from the
  producer. Mode set `best` allows any power-of-two S (hybrid: it also repairs the `x`-mapping tail
  waste on small maps); `best_pure` allows only S = R (one position per tile, the B1 definition for FC /
  1x1-output layers). With packing, S <= R/2. `splitk_s_allowed` lists the factors.
- **P9 Depthwise (B2).** `dw` mode: columns = independent channels, issue = KH*KW per tile; it needs R*C
  distinct activation bytes per cycle (`dw_act_bytes_per_cycle`) or a kw-sliding reuse structure (open).
  In OS mode a dw layer uses one of C columns. With C > KH*KW the dw tiles are drain-bound (P2).
- **P10 Mode selection (B3).** Per layer the min-cycle applicable mode (`mode`, `modes_used`).
- **P11 Packing.** Two row positions share a column weight in one DSP48E2: same cycles, `pe_dsps` halved.
  Every `pack_kmax` issued k-steps the packed fields are extracted at `pack_extract_cycles` stall cycles.
  Both are PLACEHOLDERS (swept: unlimited and two finite values) until WS5 measures the real limit.
- **P12 C1 weight sparsity.** Block = C OCs x one k-step; an OC tile issues only its nonzero k-steps.
  The DSE uses the uniform `density` shortcut (ceil(d*K) per OC tile) for all conv/fc layers, not dw;
  real masks go through `cycle_model.block_mask` / `nnz_from_mask`. `util` keeps dense MACs in the
  numerator (dense-equivalent, can exceed 1). Storage: only nonzero words plus index metadata
  (`sparse_meta_bits`, the smaller of a K-bit mask or log2(K)-bit indices per OC tile).
- **P13 C2 activation skipping.** `cycle_model.zero_ksteps` / `layer_cycles_c2` count k-steps whose R
  broadcast activations are all zero (padding and masked tail rows count as zero), per image. Not in
  dse.csv (needs golden activations); tested on synthetic ReLU data.
- **P14 WS sketch (B4).** First principles only (column `ws_cycles_sketch` in dse_layers.csv); never
  selected by `best`.
- **P15 Requant / control.** `requant_dsps = requant_lanes * requant_dsp_per_lane` (V2: 2 per lane);
  C_START, C_DONE at V2 values.

## Memory estimate (mem_model.py)

Weights in V2 OS layout (C-byte words, OC padded to C), all resident, in URAM288 by default
(`weight_uram`; `weight_bram36_if_bram` for comparison). Activations: R byte-banks per buffer; 2 buffers,
3 with residual adds (skip tensor live across the block), 4 with a separate residual pass
(`act_n_buffers`); every buffer sized to the largest stored tensor (`act_largest_tensor_bytes`);
`act_pair_max_bytes` and `act_peak_live_bytes` are the liveness lower bounds. One BRAM36 per bank minimum,
so wide-R arrays pay bank granularity (`act_bram36`). Qparams: V2 layout (2 x 64-bit per channel).

## Not modelled

Timing/fmax, LUT/FF, power; DMA/host time; ports and routing of the wider memories that split-K and dw
mode need; numeric effects of packing, fused residual and fused GAP; the descriptor/config checker.
