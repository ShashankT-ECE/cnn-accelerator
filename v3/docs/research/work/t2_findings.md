# t2: Reconfigurable / flexible-dataflow accelerators and predictable latency

Scope: per-layer dataflow switching (OS/WS/IS, layouts, sub-array fission), split-K / K-partitioning,
depthwise support in systolic arrays, per-layer mode selection by cost model / mapper, FPGA run-time
per-layer reconfiguration, and whether any of these keep latency exactly predictable (model = RTL = board).
Already covered in LITERATURE.md and not repeated: Eyeriss v2, MAERI, FlexFlow, SIGMA, SCALE-Sim, DPU, etc.
All 30 entries in `t2_related.csv` were checked against Crossref or arXiv (verified=yes). Full text was read
(pdftotext + targeted sections) for FEATHER, Gemmini, Planaria, Herald, MAESTRO, Timeloop, Luebeck et al.,
Mei et al. (uniform latency), ZigZag, CoSA, Interstellar, SARA, FlexSA, Flex-TPU, Flexagon, HybridDNN,
DNNExplorer, dual-core OPU, HARFLOW3D and FlexCNN. For the rest only the abstract or metadata was read, as
noted per row.

## (a) What exists

1. **Per-layer dataflow switching is well established, in ASIC and on FPGA.**
   - Gemmini lets each PE run weight- or output-stationary, and the dataflow is "configured at design time
     and run time" (arXiv 1911.09925, p.2).
   - Flex-TPU switches a TPU-like array between IS, OS and WS per layer. The per-layer choice is made offline
     by running every layer in all three dataflows and keeping the one with the fewest cycles (arXiv
     2407.08700, Sec III, p.3). Cycle counts come from SCALE-Sim v2.
   - SARA/SAGAR picks sub-array shape and OS/WS/IS per layer at run time with an on-chip ML recommender. The
     recommender is trained on SCALE-Sim labels and reaches 95% recommendation accuracy (p.6).
   - FEATHER (ISCA 2024) switches dataflow and layout per layer. It does a per-layer co-search in Layoutloop
     (a Timeloop extension, EDP objective) and is deployed on ZCU104 (Sec VI-A, p.11).
   - Herald assigns layers to sub-accelerators with different dataflows at compile time, using a cost model
     built on MAESTRO.
   - Flexagon selects among IP, OP and Gustavson SpMSpM dataflows per layer offline in a mapper/compiler
     (p.4).
   - Planaria fissions a systolic array into up to 16 sub-arrays. Its compiler precompiles every fission
     configuration, with tiles and cycles per tile known at compile time (p.7).
   - ArrayFlex selects the pipeline depth per layer.
   - DyNNamic and Morph reshape the array or adapt tiling per layer (abstract only).

2. **Split-K / K-partitioning.**
   - FlexSA partitions the small-M/N, large-K weight-gradient GEMMs "across the accumulation dimension with
     one partition per group" and picks sub-systolic modes with a compile-time tiling heuristic (arXiv
     2004.13027, p.9).
   - No inference-oriented FPGA paper using split-K on an OS array was found in this search (see risks).

3. **Depthwise support in systolic arrays.**
   - RiSA (TECS 2021, abstract) adds a depthwise mechanism to a systolic array.
   - FuSeConv changes the network so that it fits the array (abstract).
   - The heterogeneous dual-core OPU (FCCM 2021) uses a separate depthwise-friendly core (full text).
   - FEATHER selects a separate layout for depthwise convolution on ZCU104 (p.11).
   - LITERATURE.md already lists separate DW engines on FPGA.

4. **Mapping search with cost models:** CoSA, dMazeRunner, Interstellar, ZigZag, MAGMA, Timeloop and
   MAESTRO. Their reported accuracy against RTL or silicon:

   | Model | Reference | Reported accuracy | Where |
   |---|---|---|---|
   | Timeloop | NVDLA-derived in-house simulator | 78–99%, mean 95% (performance); energy within 8% | Fig. 9, p.8 |
   | MAESTRO | MAERI RTL / Eyeriss | within 3.9% average absolute error | Sec 4.5, Fig. 9 |
   | Uniform latency model (ZigZag) | taped-out chip | average accuracy 94.3% | abstract |
   | ZigZag | in-house post-synthesis | 6% energy, 9% PE-utilization error | p.11 |
   | Interstellar | post-synthesis | energy error < 2% | p.9 |
   | CoSA | — | no RTL/silicon validation found; evaluated with Timeloop, a NoC simulator and a GPU | p.8 |

   Independent evidence on how far generic models sit from real RTL comes from Luebeck et al. (TECS 2025),
   comparing against Verilator RTL on a 16×16 Gemmini (Tables 2–4):

   | Model | TC-ResNet8 MAPE | AlexNet MAPE | EfficientNet MAPE |
   |---|---|---|---|
   | Timeloop | 28.93% | 48.25% | 14.02% |
   | Their AIDG model | 3.67% | 9.78% | 7.51% |

   For UltraTrail (fixed dataflow), AIDG gives 22,484 cycles against 22,481 in RTL, a +3-cycle error
   (Table 1).

5. **FPGA per-layer mode selection with a board-validated model (closest FPGA precedent).**

   | Work | Board | Reported model error | Where |
   |---|---|---|---|
   | HybridDNN (DAC 2020): per-layer Spatial vs Winograd mode | VU9P / PYNQ-Z1 | 4.27% / 4.03% | p.6 |
   | DNNExplorer (ICCAD 2020) | — | 1.15% (pipeline), 2.17% average over 36 CONV cases (generic) | pp.5–6 |
   | Heterogeneous dual-core OPU (FCCM 2021) | board | "<1% error on cycle count compared with … board-level FPGA"; e.g. MobileNet v1 757,149 vs 755,857 measured (−0.2%) | Table IV |
   | HARFLOW3D (FCCM 2023): runtime-parameterized blocks | ZCU106 | per-layer error shown in Fig. 6, divergence attributed to DMA burst delays | p.8 |
   | FlexCNN (TRETS 2023): per-layer dynamic tiling chosen by `latency_est()` | — | no accuracy number found | — |

6. **Real-time / deterministic FPGA DNN accelerators (counter-evidence for the framing).**
   - DERCA (RTSS 2025, VCK190; abstract only, full text embargoed until 2026-12-02) claims "cycle-level
     determinism", intra-layer preemption points, an on-chip EDF scheduler, a formal predictability and
     schedulability analysis, and "<5% overhead in WCET".
   - ART (GLSVLSI 2025) and FILCO (arXiv 2604.07523, per-layer runtime reconfiguration on VCK190 chosen by
     MILP at compile time) come from the same group.

## (b) Closest works to the thesis

The thesis part for t2: per-layer mode (OS / split-K / DW) chosen at compile time by a cycle model, with
cycle-exact model = RTL = board.

| Work | Does | Not found in sections read |
|---|---|---|
| FEATHER (ISCA'24) [tong2024feather] | per-layer dataflow+layout switching; per-layer co-search in Layoutloop; ZCU104 deployment incl. DW layout | Layoutloop vs board comparison; any exact/WCET latency claim (FPGA latency is a 100-run average, Fig. 12) |
| Flex-TPU [elbtity2024flextpu] | per-layer OS/WS/IS runtime switch, offline min-cycle selection | RTL- or board-validated cycles (SCALE-Sim only); split-K; DW |
| Dual-core OPU (FCCM'21) [zhao2021dualopu] | FPGA, DW-capable core, compile-time scheduling, cycle-accurate simulator vs board | exact agreement (−0.2%, <1%); dataflow-mode switching within one array |
| HybridDNN (DAC'20) [ye2020hybriddnn] | FPGA per-layer mode (Spatial/Winograd) chosen by analytical model; board-validated | exact agreement (4.27%/4.03%) |
| Planaria (MICRO'20) [ghodrati2020planaria] | compile-time per-layer fission configs with known cycles/tile; simulator "verified" with Verilog | error number; board/silicon; guarantee |
| DERCA (RTSS'25) [ji2025derca] | cycle-level determinism + WCET analysis on FPGA (VCK190) | dataflow flexibility, split-K/DW, sparsity, model=board check (abstract only) |
| Gemmini (DAC'21) [genc2021gemmini] | runtime OS/WS choice, cycle-exact FireSim of RTL | own cost model; per-layer choice by model (third parties report 4–48% MAPE on Gemmini) |
| SARA (DAC'22) [samajdar2022sara] | per-layer config incl. OS/WS/IS by learned recommender | RTL validation of labels; guarantee |
| FlexSA [lym2020flexsa] | sub-systolic modes + K-partitioning, compile-time heuristic | RTL validation; inference/latency guarantee |

## (c) Verdict for t2

**Partly done.** Each ingredient has prior art:

- per-layer dataflow switching in ASIC and on FPGA (FEATHER on ZCU104, Flex-TPU, SARA, Gemmini);
- per-layer mode selection at compile time from a cost model (HybridDNN, Herald, FEATHER/Layoutloop, Flex-TPU,
  FlexCNN, Planaria);
- split-K-style reduction partitioning (FlexSA, training);
- depthwise handling in or next to systolic arrays (RiSA, dual-core OPU, FEATHER);
- FPGA cost models validated on the board to within 0.2–4% (dual-core OPU, DNNExplorer, HybridDNN);
- cycle-level determinism on FPGA, in a different form (DERCA).

**Not found in any work read:** the combination of (i) a flexible array whose per-layer mode (OS / split-K /
DW) is chosen at compile time by a cycle model, with (ii) that model **cycle-exact** (zero error) against
both RTL and board counters on every layer and job.

- The best reported agreements are non-zero: −0.2% (dual-core OPU, Table IV), +3 cycles on a fixed-dataflow
  ASIC RTL (Luebeck, Table 1), and Planaria's "verified" with no number.
- Generic flexible-dataflow models are 4–48% off on real RTL (Luebeck, Tables 2–4).

So the defensible novelty is narrow: exact (zero-error) compile-time latency for a multi-mode FPGA array,
demonstrated on the board. It is not "reconfigurable dataflow" and not "predictable FPGA DNN accelerator" in
general.

## (d) Risks

- **DERCA** (RTSS 2025) already claims "cycle-level determinism" and WCET analysis for an FPGA DNN accelerator.
  The full text is embargoed, so we cannot yet say whether it reports model = board cycle equality. A
  reviewer from the real-time community will cite it. Position V3 against it explicitly: dataflow modes and
  sparsity versus preemption and EDF.
- **The claim must say "exact", not "accurate".** Dual-core OPU (<1%), DNNExplorer (1.15%) and HybridDNN (~4%)
  already show accurate board-validated models on FPGA. If V3 drops to "within X%", the novelty mostly
  disappears.
- **FEATHER is the closest flexible-dataflow FPGA design** (ZCU104, DW-specific layout, per-layer switching).
  Flex-TPU is the closest for per-layer OS/WS/IS selection by minimum cycles. Both lack exact or validated
  latency in the sections read, but they pre-empt "per-layer mode selection" as a contribution on its own.
- **Split-K for inference on small layers** was not found in an FPGA paper in this search, but split-K is
  standard on GPUs (Stream-K, Triton split-K) and FlexSA does it for training. Do not claim split-K itself
  as novel.
- **Depthwise mode in a systolic array** is covered by RiSA, FuSeConv and FEATHER. Only the abstracts of RiSA
  and FuSeConv were read, so their latency-modelling details are unknown.
- FILCO details came from a fetch-tool summary of the arXiv HTML. Re-read before citing specifics.

## (e) UNVERIFIED / not used as support

- "SqueezeJet-2 analytical model max error 8.3%, average 4.45%" and the "Systolic-CNN" runtime-flexible
  OpenCL design came from search snippets only. They were not fetched and are not in the CSV.
- MRSA (Microelectronics Journal 2026, multi-mode reconfigurable systolic array, channel-wise stationary)
  came up in Crossref search only and was not read.
- Light-OPU / OPU (overlay processors, per-layer runtime configuration): Light-OPU's DOI was seen in Crossref
  (10.1145/3373087.3375311) but the paper was not read. Excluded.
- Stream-K (GPU split-K): not looked up. Mentioned only as general knowledge in the risks above. Verify before
  citing.
- Exact venue of FlexSA: arXiv only in the sources checked. A later venue may exist but was not resolved.
