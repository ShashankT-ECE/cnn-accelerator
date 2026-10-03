# V3 literature review (WS1, Phase 0)

Date: 2026-10-03. Scope: workstream WS1 of V3 Phase 0 (decisions only, no RTL). Every number in this file is
labelled **literature**. Nothing here is a V3 requirement; open choices stay open decisions in `DECISIONS.md`.

Files:

- `v3/docs/refs.bib`: 51 BibTeX entries (DOI metadata pulled from api.crossref.org).
- `v3/results/literature.csv`: 65 rows, one per reference, plus one row per implementation result. 34 rows carry
  implementation numbers.
- This file: a synthesis for each topic, the comparison table, the gaps and positioning, and the flagged items.

Method. This is a narrative literature review. It was run in `lit-review` mode of the academic-research-skills
`deep-research` skill, with no full pipeline and no cross-model features. Sources were found by web search and by
following citations. Each DOI was resolved through the Crossref API, and the title, authors and year were compared
with the citation. Items without a DOI (AMD white papers, product guide, model zoo, arXiv) were fetched from the
official URL. Implementation numbers were read from the full text whenever it was accessible (arXiv versions, AMD
PDFs, PMC/MDPI). When only an abstract or a secondary source was available, the CSV says so.

Two flags in the CSV need explaining. `verified=yes` means two things: the reference resolved, and every number in
that row was read in the primary text named in `source_location`. `verified=no` means one of three things: the
reference could not be confirmed, the full text was not accessible, or the numbers come from a secondary source.
Cells keep the paper's own notation, for example `180.1k` LUTs or `320 (18Kb)` BRAM. Values are never converted.
Derived figures appear only in `notes`, marked "derived".

Provenance caveat (for the main session). `literature.csv` is curated by hand from the sources. It is not generated
by a script and has no `v3/model/common.py` metadata columns. `v3/scripts/check_results.py`, once written, needs an
explicit rule for this file (see the Flagged section).

## (a) CNN accelerators for ResNet / MobileNet on Zynq UltraScale+, including the AMD DPU

**AMD DPU (DPUCZDX8G).** PG338 v1.2 [xilinx2019pg338] describes the DPU_EU. Its DSP slices run at twice the
fabric clock ("DSP Double Data Rate", PDF p.11). The size names are B512 to B4096, where peak ops/clk =
PP·ICP·OCP·2; B4096 is PP=8, ICP=16, OCP=16 (Table 8). Table 9 gives one B4096 core on ZCU102 as 642 DSP, 40,865 LUT
and 249.5 BRAM (high-DSP mode). Table 10 gives 1.4 Tops peak for ZU5 B4096x1 at 350 MHz. The Vitis AI 2.5 model zoo
[amd2022vitisaimodelzoo] lists the KV260 with `1 * B4096F @ 300MHz`. At that clock the single-thread end-to-end
results are 94.21 fps for resnet_v1_50_tf and 290.62 fps for mobilenet_v2_1.0_224_tf. The multi-thread figures are
99.01 and 336.82 fps. Those tables have no power or latency columns. DPUV4E [li2025dpuv4e] reports a 3x B4096
ZCU102 DPU at 281 MHz: 1686 DSP, 160K LUT, 771 BRAM, 190.3 fps on ResNet50. Derived: 4096 ops/clk × 300 MHz =
1.2288 Tops peak for the KV260 configuration. This figure is computed, not stated in a source.

**Custom ZU+ designs.** There are three groups:

- **MobileNet-class engines with separate depthwise support.** Wu et al. [wu2019mobilenets] use separate Conv and
  Dwcv engines and reach 205.3 fps on ZU2 and 809.8 fps on ZU9; these figures come from the abstract and secondary
  tables. Li et al. [li2021dynamic] report 381.7 fps on ZCU102 with adaptive row-based dataflow. Jiang et al.
  [jiang2023fulldataflow] report 1910 fps with a full-dataflow, layer-pipelined MobileNetV2 (secondary). DeepDive
  [baharani2021deepdive] runs 4-bit MobileNet-V2 at 11 fps for 0.46 W, measured as board power minus idle.
- **Mixed or low precision.** FILM-QNN [sun2022filmqnn] runs on ZCU102 at 150 MHz with 2092 DSP and 12.9 W. It
  reaches 214.8, 109.1 and 537.9 fps on ResNet-18, ResNet-50 and MobileNet-V2, using DSP packing plus LUT MACs and
  95% 4-bit / 5% 8-bit weights. Synetgy [yang2019synetgy] is 4-bit on Ultra96. FracBNN [zhang2021fracbnn] and FINN-R
  [blott2018finnr] are binarized; FracBNN reaches 2806.9 fps on CIFAR-10 on ZU3EG at 4.1 W.
- **Weight compression.** unzipFPGA [venieris2021unzipfpga] reaches 71.71 inf/s on ResNet50 on ZU7EV with 16-bit
  data.

Typical ranges in the CSV (literature):

- **Clock:** ZU+ fabric clocks are 150 to 300 MHz in primary-read rows. Wu et al. (430 MHz) and DNNVM (500 MHz)
  are higher, but those values are secondary.
- **Power:** reported board-level power is 12.9 to 17.1 W for ZCU102 designs and 4.1 to 10.7 W for Ultra96 designs.
  The measurement methods differ and are listed per row.
- **KV260:** only the DPU rows are on the KV260 itself.

**What is open for V3.**

- No reviewed source reports an INT8 ResNet-20 or a MobileNet-style CIFAR-10 net on a ZU+ or KV260 custom
  accelerator. The checks were the comparison tables of all papers listed above.
- The CIFAR-10 FPGA rows we found are either binarized (FINN-R, FracBNN) or on a Kintex UltraScale device
  [kung2019packing].
- The DPU tables are ImageNet-only, so a DPU number for our CIFAR-10 nets has to be measured by us.

## (b) Reconfigurable / flexible dataflow

**ASIC flexible-dataflow designs:**

- Eyeriss v2 [chen2019eyerissv2] uses a hierarchical-mesh NoC that adapts to the different reuse and bandwidth needs
  of compact and sparse layers. It processes compressed weights and activations, and reaches 1470.6 inferences/s on
  sparse MobileNet in 65 nm (abstract).
- MAERI [kwon2018maeri] uses reconfigurable distribution and reduction trees to map arbitrary dataflows.
- FlexFlow [lu2017flexflow] mixes feature-map, neuron and synapse parallelism.
- SIGMA [qin2020sigma] targets irregular and sparse GEMM.
- The TPUv1 [jouppi2017indatacenter] is the fixed weight-stationary systolic baseline.

**Analysis tools.** Timeloop [parashar2019timeloop], MAESTRO [kwon2019understanding] and SCALE-Sim
[samajdar2020systematic] are the standard tools for evaluating dataflow and mapping choices. They are the analogue of
the V3 B3 plan, where the compiler picks a per-layer mode from a cycle model.

**On FPGA, flexibility is mostly about depthwise-separable support:**

- separate depthwise and pointwise engines [wu2019mobilenets, bai2018cnn];
- adaptive row-based scheduling and computation mapping [li2021dynamic];
- full or balanced layer pipelines [jiang2023fulldataflow, zhao2025balanced];
- architecture co-design [baharani2021deepdive].

Classic FPGA loop-tiling and dataflow work [zhang2015optimizing, ma2017optimizing] fixes one mapping per design.

**What is open for V3.** None of the FPGA papers read for this review documents three things together: a single
INT8 output-stationary array that switches at run time between OS, split-K and depthwise modes per layer; a choice
driven by a cycle model; and cycle-exact agreement between model, RTL and board. This statement is limited to the
architecture and evaluation sections read. It is not a systematic absence claim.

## (c) Structured weight sparsity and activation-zero skipping

**ASIC work:**

- Cnvlutin [albericio2016cnvlutin] skips zero activations with independent lanes and a co-designed storage format.
  It reports a 1.24x to 1.55x speedup with no accuracy loss (abstract).
- Cambricon-X [zhang2016cambriconx] uses an Indexing Module to select the neurons that match sparse weights
  (abstract).
- SCNN [parashar2017scnn] exploits both zero weights and zero activations in compressed form: 2.7x performance and
  2.3x energy versus a dense baseline (abstract).
- EIE [han2016eie] runs compressed sparse FC layers and also skips zero activations.
- Cambricon-S [zhou2018cambricons] argues for coarse-grained pruning to reduce irregularity and indexing cost.
- NVIDIA's 2:4 Sparse Tensor Cores [mishra2021accelerating] are the best-known fixed structured pattern, giving 2x
  math throughput.

**FPGA on ZU+:**

- Zhu et al. [zhu2020structured] use shape-wise structured pruning and a "sparse-wise dataflow" that skips MAC cycles
  with zero weights, plus clock gating of PEs on zero activations. In 8-bit mode on ZCU102 at 200 MHz they report
  96 img/s on VGG-16 with 2520 DSP, 990.8 effective GOP/s and 17.1 W. These are arXiv numbers; the TVLSI version
  reports ResNet-50 at 57 img/s (secondary).
- SPEC [xie2023spec] uses layer-wise N:16 sparsity with index decoders. On ZCU102 at 200 MHz with 1024 DSP and
  15.0 W it reports ResNet-50 at 83 or 150 img/s, depending on the sparsity setting (Table 5).
- Column combining [kung2019packing] packs sparse filter columns so that a systolic array stays dense. It reaches
  93.1% on ResNet-20 CIFAR-10, with a Kintex UltraScale implementation.

**Typical caveat.** Sparse FPGA papers report effective, dense-equivalent GOP/s. Any comparison with V3 must state
whether throughput counts skipped MACs.

**What is open for V3.** V3 C1 plans static, compile-time-scheduled block sparsity (array-width output channels ×
K-step), and C2 plans skipping of all-zero broadcast K-steps. That puts V3 between Zhu et al. and SPEC (static
structured patterns) and Cnvlutin-style activation skipping at coarse granularity. None of the sparse papers read
reports cycle-exact prediction of the skipped schedule, and none checks bit-exactness against a golden model; this
statement is limited to the sections read.

## (d) INT8 DSP48E2 packing (2 MACs per DSP)

**Packing scheme (WP486 [fu2017wp486], WP487 [fu2017wp487]).** The DSP48E2 has a 27-bit pre-adder, a 27×18
multiplier and a 48-bit post-adder (WP487 p.5; also Sommer et al. [sommer2022dsppacking] Eq. 1). Two INT8 operands
a and d are packed into the 27-bit pre-adder output. a is shifted left by G = 18 and sign-extended on port A; d is
sign-extended on port D. Both are multiplied by a shared INT8 operand b on port B. The product holds a·b·2^18 + d·b,
so a single post-adder accumulation carries both dot products at once.

**Operand sharing (exactness condition 1).** The two products must share one operand. WP487 (p.4) lists the valid
shares:

- two output channels that share one input vector;
- two inputs (pixels or patches) that share one weight vector.

WP486 (Fig. 7) shows the same two cases as "input sharing" and "weight sharing".

**Correction term (exactness condition 2).** In the packed word, the lower dot product sits in P[17:0] and is exact.
The upper dot product equals P[35:18] + P[17]: it is incremented by one only when the lower result is negative. The
increment is applied once, to the final packed word only (WP487 pp.8-10, worked example in Table 1). Sommer et al.
explain the origin: extracting a field by right shift floors a signed value. Without correction the extracted value
is off by −1. For the Xilinx INT4 four-multiplication scheme this gives MAE 0.37 and an error probability of 37.35%.
Full round-half-up correction (27 LUT + 32 FF) gives zero error; an approximate correction through port C reduces the
error probability to 3.13% (arXiv:2203.11028 Table I).

**Maximum accumulation length (exactness condition 3).**

- **Signed INT8 × INT8:** at most N = 7 products can be summed before the lower 18-bit field bleeds into the upper
  one (WP487 p.7, WP486 p.4). Beyond seven terms the two fields must be separated into wider words, for example
  two 24-bit fields in a further 48-bit DSP add, with the correction still applied only at the end (WP487 p.11). With
  DSP slices doing the extra adds, 8 DSPs compute 14 MACs, a 1.75x gain. With LUT adders and incrementer the gain is
  2x.
- **UINT8 data × INT8 weight (after ReLU):** G = 19, the A-port bias is removed through port C / WMUX, N ≤ 8, and 9
  DSPs compute 16 MACs, a 1.78x gain (WP487 pp.12-14).

Derived check (not from a source): the largest INT8×INT8 product magnitude is 128·128 = 16,384. Seven such terms sum
to 114,688, which is below 2^17 = 131,072; eight terms equal 131,072 and overflow. That is consistent with N = 7.
Sommer et al.'s general rule, that δ padding bits allow up to 2^δ accumulations, is more conservative for this case.

**Related techniques:**

- Overpacking [sommer2022dsppacking] puts more products into a DSP but is approximate, so it is unsuitable for a
  bit-exact design.
- Double MAC [lee2019doublemac] reports 14% to more than 80% network speedup. Its abstract only says the method works
  "without sacrificing the output quality significantly", so exactness is not established from what was read.
- PMSDS [huang2019efficient] is another multi-multiply packing.
- FILM-QNN [sun2022filmqnn] packs 2 W8A5 or 4 W4A5 multiplications per DSP.
- Zhu et al. [zhu2020structured] multiply two 8-bit activations by one weight using 2 DSPs per PE.
- Li et al. [li2024revealing] analyse how the DPUCZDX8G uses INT8 packing together with DSP double-pumping (Clk×2 =
  666 MHz on XCZU3EG, OOC). They propose in-DSP multiplexing and a ring accumulator: 158 LUT and 0.826 W, against
  1280 LUT and 1.03 W for their replica of the official engine (Table II).

**Implication for an output-stationary array (analysis, not a decision).** In the V2 OS array each PE accumulates
over the whole reduction length K = Cin·kh·kw, which is typically far more than 7. In-DSP packed accumulation
therefore needs one of these:

1. periodic field separation every ≤ 7 terms (≤ 8 for UINT8 activations) into wider accumulators;
2. a split of the P register into two independent wide accumulators fed by the packed product, with the
   inter-field borrow handled;
3. a different sharing axis.

The choice of sharing axis (two OCs per DSP sharing the broadcast activation, or two pixels sharing the weight), the
activation signedness (INT8 vs UINT8 after ReLU) and the LUT/FF cost all remain Phase 0 open decisions, as listed
under "Array R×C and INT8 DSP packing" in DECISIONS.md.

## Comparison table

The table below was rendered by script from `v3/results/literature.csv` on 2026-10-03. It includes every row with at
least one of fps, GOPS, DSP, LUT, latency, power or clock filled. Zynq UltraScale+ / KV260 rows come first. The
`ver.` column is the CSV `verified` flag. A `no` here usually means the numbers are secondary. Notes and caveats are
in the CSV `notes` column.

| key | ver. | topic | network | dataset | device | precision | MHz | DSP | LUT | BRAM | URAM | GOPS | lat. ms | fps | W | mJ/img | acc. % | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amd2022vitisaimodelzoo | yes | a | resnet_v1_50_tf (ResNet-50 v1, 6.97 GOPs) | ImageNet 224x224 | KV260 (XCK26), 1x DPU B4096F |  | 300 |  |  |  |  |  |  | 94.21 |  |  |  | README 'Performance on Kria KV260 SOM', row 63 |
| amd2022vitisaimodelzoo | yes | a | resnet50_pt (pruned ResNet-50, 4.1 GOPs) | ImageNet 224x224 | KV260 (XCK26), 1x DPU B4096F |  | 300 |  |  |  |  |  |  | 84.44 |  |  |  | README 'Performance on Kria KV260 SOM', row 70 |
| amd2022vitisaimodelzoo | yes | a | mobilenet_v2_1_0_224_tf (MobileNetV2 1.0, 602 MOPs) | ImageNet 224x224 | KV260 (XCK26), 1x DPU B4096F |  | 300 |  |  |  |  |  |  | 290.62 |  |  |  | README 'Performance on Kria KV260 SOM', row 39 |
| amd2022vitisaimodelzoo | yes | a | mobilenet_v1_1_0_224_tf (MobileNetV1 1.0, 1.14 GOPs) | ImageNet 224x224 | KV260 (XCK26), 1x DPU B4096F |  | 300 |  |  |  |  |  |  | 354.19 |  |  |  | README 'Performance on Kria KV260 SOM', row 38 |
| amd2022vitisaimodelzoo | yes | a | resnet_v1_50_tf | ImageNet 224x224 | ZCU102 (0432055-05), 3x DPU B4096 |  | 281 |  |  |  |  |  |  | 87.95 |  |  |  | README 'Performance on ZCU102', row 63 |
| amd2022vitisaimodelzoo | yes | a | mobilenet_v2_1_0_224_tf | ImageNet 224x224 | ZCU102 (0432055-05), 3x DPU B4096 |  | 281 |  |  |  |  |  |  | 268.07 |  |  |  | README 'Performance on ZCU102', row 39 |
| amd2022vitisaimodelzoo | yes | a | resnet_v1_50_tf | ImageNet 224x224 | ZCU104, 2x DPU B4096 |  | 300 |  |  |  |  |  |  | 93.84 |  |  |  | README 'Performance on ZCU104', row 63 |
| amd2022vitisaimodelzoo | yes | a | mobilenet_v2_1_0_224_tf | ImageNet 224x224 | ZCU104, 2x DPU B4096 |  | 300 |  |  |  |  |  |  | 284.18 |  |  |  | README 'Performance on ZCU104', row 39 |
| blott2018finnr | yes | a | CNV-6 | CIFAR-10 | Ultra96 (XCZU3EG) | W1A1 | 300 |  | 41733 | 283 (BRAM18) |  | 2318 |  |  | 10.7 |  | 80.10 | arXiv:1809.04570 Table 5 (p.18), Table 4 (p.17) |
| li2025dpuv4e | yes | a | ResNet50 (8.19 GOPs) | 224x224 input | ZCU102, AMD DPUCZDX8G B4096 x3 (CU3) |  | 281 | 1686 | 160K | 771 | 0 |  |  | 190.3 |  |  |  | arXiv:2506.11441 Table II and Table III |
| venieris2021unzipfpga | yes | a | ResNet50 (OVSF50 compressed weights) | ImageNet | ZCU104 (ZU7EV) | 16b fixed | 200 | 1728 | 230.0 kLUTs (device capacity) | 4.75 MB (device capacity) |  |  |  | 71.71 |  |  |  | arXiv:2103.05600 Table V |
| xing2020dnnvm | no | a | ResNet50 | ImageNet | ZU9 | 8b fixed | 500 | 2520 |  |  |  |  |  | 80.95 |  |  |  | secondary: unzipFPGA arXiv:2103.05600 Table V |
| yang2019synetgy | yes | a | DiracDeltaNet (ShuffleNetV2-derived, shift ops) | ImageNet | Ultra96 (ZU3EG) | 4-bit W / 4-bit A | 250 | 360 | 51776 | 159 |  | 47.09 |  | 66.3 | 5.5 |  | 68.30 | arXiv:1811.08634 Tables 5-7, Sec. 4 text p.8 |
| zhang2021fracbnn | yes | a | FracBNN CIFAR-10 model (binary, fractional activations) | CIFAR-10 | Ultra96 v2 (ZU3EG) | W1 / A1.4 (effective) | 250 | 126 | 51444 | 212 |  |  |  | 2806.9 | 4.1 |  | 89.1 | arXiv:2012.12206 Table 7, Table 8 |
| zhang2021fracbnn | yes | a | FracBNN ImageNet model | ImageNet | Ultra96 v2 (ZU3EG) | W1 / A1.4 (effective) | 250 | 224 | 50656 | 201 |  |  |  | 48.1 | 6.1 |  | 71.8 | arXiv:2012.12206 Table 6, Table 8 |
| baharani2021deepdive | yes | a;b | MobileNet-V2 (alpha=0.75, H=224) | ImageNet | ZCU102 (XCZU9EG) | 4-bit datapath (BW=4) | 200 |  |  |  |  |  | 88.49 | 11 | 0.46 |  |  | arXiv:2007.09490 Table 3, Table 4, Table 5 |
| jiang2023fulldataflow | no | a;b | MobileNetV2 | ImageNet | XCZU9EG (ZCU102) | 8-bit | 333 | 1283 | 170429 |  |  |  |  | 1910.0 |  |  |  | secondary: Zhao et al. arXiv:2407.19449 Tables IV-V |
| li2021dynamic | no | a;b | MobileNetV2 | ImageNet | ZCU102 (listed as XCZU9EQ in secondary) | 8-bit | 200 | 576 | 125470 |  |  |  |  | 381.7 |  |  |  | secondary: Zhao et al. arXiv:2407.19449 Tables IV-V |
| wu2019mobilenets | no | a;b | MobileNetV2 | ImageNet | ZU2EG | 8/8 | 430 | 212 | 31198 | 145 |  |  |  | 205.3 |  |  | 68.1 | secondary: FracBNN arXiv:2012.12206 Table 6 |
| wu2019mobilenets | no | a;b | MobileNetV2 | ImageNet | XCZU9EG | 8-bit | 333 | 2070 |  |  |  |  |  | 809.8 |  |  |  | secondary: Zhao et al. arXiv:2407.19449 Table IV |
| sun2022filmqnn | yes | a;d | ResNet-18 | ImageNet | ZCU102 | 95% W4A5 + 5% W8A5 (intra-layer mixed) | 150 | 2092 | 180.1k | 440.5 (BRAM36) |  | 778.9 |  | 214.8 | 12.9 |  | 70.47 | FILM-QNN (FPGA'22) Table 6 |
| sun2022filmqnn | yes | a;d | ResNet-50 | ImageNet | ZCU102 | 95% W4A5 + 5% W8A5 (intra-layer mixed) | 150 | 2092 | 180.1k | 440.5 (BRAM36) |  | 891.4 |  | 109.1 | 12.9 |  | 77.25 | FILM-QNN (FPGA'22) Table 6 |
| sun2022filmqnn | yes | a;d | MobileNet-V2 | ImageNet | ZCU102 | 95% W4A5 + 5% W8A5 (intra-layer mixed) | 150 | 2092 | 180.1k | 440.5 (BRAM36) |  | 320.1 |  | 537.9 | 12.9 |  | 65.67 | FILM-QNN (FPGA'22) Table 6 |
| xilinx2019pg338 | yes | a;d | ResNet50 | 224x224 input (dataset not named in table) | ZCU102 (ZU9), 3x DPU B4096_EU |  | 333 |  |  |  |  |  |  | 175 |  |  |  | PG338 v1.2 Table 11 (p.22-23) |
| xilinx2019pg338 | yes | a;d | (one DPU B4096 core, resources only) |  | ZCU102 |  |  | 642 | 40865 | 249.5 |  |  |  |  |  |  |  | PG338 v1.2 Table 9 (p.21) |
| li2024revealing | yes | b;d | (DPU B1024-style systolic engine replica, OOC) |  | XCZU3EG | INT8 (packed) | 666 |  | 158 |  |  |  |  |  | 0.826 |  |  | arXiv:2409.03508 Table II |
| xie2023spec | yes | c | ResNet-50 (layer-wise N:16 sparse) | ImageNet | ZCU102 | 8-bit | 200 | 1024 | 500K | 320 (18Kb) |  |  |  | 83 | 15.0 |  |  | Micromachines 14(3):528 Table 5 |
| xie2023spec | yes | c | ResNet-50 (layer-wise N:16 sparse) | ImageNet | ZCU102 | 8-bit | 200 | 1024 | 500K | 320 (18Kb) |  |  |  | 150 | 15.0 |  |  | Micromachines 14(3):528 Table 5 |
| zhu2020structured | yes | c | AlexNet |  | ZCU102 | 16bit fixed | 200 | 1350 | 390K | 1460 |  | 476.7 |  | 987 | 15.4 |  |  | arXiv:2001.01955 Table II-IV |
| zhu2020structured | yes | c;d | VGG-16 |  | ZCU102 | 8-bit int | 200 | 2520 | 405K (logic) | 1460 |  | 990.8 |  | 96 | 17.1 |  |  | arXiv:2001.01955 Table III, Table IV |
| bai2018cnn | yes | a;b | MobileNetV2 | ImageNet | Arria 10 SoC (10AS066N3F40E2SG) |  | 133 | 1278 | 81753 ALM | 1844 (M20K) |  | 170.6 |  | 266.2 |  |  |  | arXiv:1809.01536 Table II, Table III, Sec. V |
| zhao2025balanced | yes | a;b | MobileNetV2 | ImageNet | ZYNQ XC7Z045 | 8-bit | 200 | 844 | 163087 |  |  |  |  | 985.8 |  |  |  | arXiv:2407.19449 Tables IV-V |
| chen2019eyerissv2 | yes | b;c | sparse MobileNet | ImageNet | 65nm CMOS ASIC |  |  |  |  |  |  |  |  | 1470.6 |  |  |  | arXiv:1807.07928 abstract |
| kung2019packing | yes | b;c | ResNet-20 (column-combined, pruned) | CIFAR-10 | Kintex UltraScale XCKU035 | 8-bit (bit-serial MACs), 32-bit accumulation | 150 |  |  |  |  |  |  |  |  |  | 93.1 | arXiv:1811.04770 Table 2, Table 3, Sec. 7.3 |

## Gaps / positioning

This section gives a qualitative positioning only. No V3 numbers exist yet, so no comparison is claimed. There are
four places where a bit-exact, cycle-exact, reconfigurable KV260 design with structured sparsity could be competitive
or new:

1. **Small CIFAR-10 nets on the same board as the DPU.** Every DPU figure found is for ImageNet-size inputs. Our
   hypothesis, which is not established, is this: ResNet-20 (16 to 64 channels) and a small MobileNet-style net may
   under-fill a B4096 core (ICP = OCP = 16, PP = 8, PG338 Table 8). A right-sized INT8 array could then compete on
   latency and energy per CIFAR-10 image at much lower DSP, LUT and power cost. Testing this needs a DPU CIFAR-10
   measurement on our KV260; the V2 DPU flow exists. The model zoo lists the KV260 DPU at 300 MHz. V3 D2 records that
   300 MHz is not reachable from the boot-image PLL for our own PL designs, so the DPU's actual read-back clock must
   be recorded when it is measured.
2. **Verification as a contribution.** The custom accelerators read here report throughput, resources and accuracy.
   None of the sections read reports full-test-set bit-exact agreement with an integer golden model, and none reports
   cycle-exact agreement of a model with the board. Integer-only inference [jacob2018quantization] is the standard
   reference for the arithmetic. A design that is reconfigurable per layer and sparse, yet still bit- and cycle-exact,
   is a defensible positioning, within the stated limit of what was read.
3. **Sparsity with a static, predictable schedule.** FPGA structured-sparse work [zhu2020structured, xie2023spec]
   reports effective GOP/s on ImageNet-scale nets on ZCU102 at around 15 to 17 W. A block-structured scheme scheduled
   at compile time, aligned to the array width, fits a cycle-exact story: skip counts are known before run time.
   Activation-zero skipping (C2) is data-dependent and would need a cycle model fed with the actual activations.
4. **Exact INT8 packing in an OS array.** The sources give exact conditions for 2 MACs per DSP: a shared operand, at
   most 7 (INT8) or 8 (UINT8) terms per packed field, and a final +1 correction. The DPU itself uses packing plus DSP
   double-pumping [li2024revealing, xilinx2019pg338]. A packed OS array that is verified bit-exact against the golden
   model, with the field-separation cost reported, would be a concrete and checkable contribution. Whether it pays off
   at 250/333 MHz on XCK26 is a DSE question, not a literature one.

Risks to the positioning. FPGA papers differ in power method (board vs rail vs board-minus-idle), batch size and
"effective" throughput. Any comparison table in V3 must carry those labels per row, as the CSV `notes` do.

## Flagged / unverified

**Unverified references:**

- `xilinxug579` (UG579 DSP48E2 user guide): the URL loads the AMD JavaScript portal, so the title and version could
  not be confirmed. The DSP48E2 widths are cited from WP487 and Sommer et al. instead. `verified=no`.
- `xilinxwp521` (INT4 packing white paper): known only from Sommer et al.'s reference [4]; not fetched.
  `verified=no`.
- `costa2026throughput` (KV260 DPU multithreading, J. Real-Time Image Processing 2026): the DOI resolves, but the full
  text redirects to a Springer login. A web-search summary attributed KV260 throughput and FPS/W figures to it. Those
  figures were **not** used. `verified=no`.

**Rows whose numbers are secondary (`verified=no`).** The DOI resolves in each case and the abstract confirms the
headline fps:

- `wu2019mobilenets` (FracBNN Table 6; Zhao et al. Table IV);
- `jiang2023fulldataflow` (Zhao et al. Tables IV-V);
- `li2021dynamic` (Zhao et al.; the device is listed as "XCZU9EQ" in the secondary source and as ZCU102 in the
  abstract);
- `xing2020dnnvm` (unzipFPGA Table V; abstract not checked).

**Abstract-only references.** These rows record no numbers beyond what the abstract states: Cnvlutin, Cambricon-X,
Cambricon-S, EIE, SCNN, FlexFlow, MAERI, Eyeriss v2, Double MAC, PMSDS, Wu et al. FPGA'19 (compute-efficient), and
Angel-Eye.

**Metadata only.** Eyeriss, TPUv1, SIGMA, MAESTRO, Timeloop, SCALE-Sim, Zhang FPGA'15, Ma FPGA'17, FINN, ResNet,
MobileNet v1/v2 and Jacob et al. are cited for context only. No technical claim in this file depends on them beyond
their titles, except TPUv1's 256×256 8-bit WS array, which is quoted from li2024revealing.

**Version mismatches.** The numbers were read from arXiv versions, which can differ from the published papers:

- zhu2020structured: arXiv has AlexNet and VGG-16; TVLSI adds ResNet-50 (57 img/s, secondary).
- baharani2021deepdive: the GLSVLSI'21 version was not read.
- li2025dpuv4e: the FPGA'25 entry is pages 45-45 (abstract length); the numbers come from arXiv.
- li2024revealing (FPL'24) and sommer2022dsppacking (FPL'22): numbers from arXiv.

**Inconsistencies inside or between sources:**

- PG338 v4.1 "Features" page: "nomenclature indicates the total number of MACs per DPU clock cycle". PG338 v1.2
  Table 8: peak ops/clk = PP·ICP·OCP·2 = 4096 for B4096, i.e. 2048 MACs/clk. The v1.2 peak figures (1.4 Tops for
  B4096 at 350 MHz) match the ops reading.
- DeepDive: the abstract gives 47.4 FPS/W for MobileNet-V2, while Table 5 gives 23.91 FPS/W at the reported design
  point.
- Bai et al.: Table III gives 266.2 fps; the abstract and conclusion give 266.6 fps.
- WP486 says the upper operand "should start with at least the 17th bit" but uses an 18-bit shift. It also says
  "2-bits remaining ... up to 7 product terms". WP487 gives the exact derivation (G = 18, N ≤ 7). Sommer et al.'s
  generic "2^δ accumulations" bound is more conservative than WP487 for INT8×INT8.
- Sommer et al.'s reference [3] labels the INT8 white paper "wp485", but its URL is wp486.
- FINN-R Table 5: the column is headed "Perf.(predicted)" with a percentage in parentheses. Whether 2318 GOp/s is a
  measured or a predicted value was not resolved.
- xie2023spec Table 5: the meaning of "Sparsity (%)" (density kept vs fraction pruned) was not re-checked. The two
  ResNet-50 "Ours" columns are recorded with their MAC-reduction values.
- Vitis AI model zoo: the KV260 table states 300 MHz. That is the vendor configuration and has not been checked
  against our board's PLL read-back (see V3 D2 and V2 D19/D20).

**Process notes:**

- The deep-research skill's review-form note expects an author reply before a review starts. This run was a
  subagent task in which the caller explicitly asked for `lit-review` mode, which the skill maps to a
  narrative/integrative review. No systematic or PRISMA protocol was followed. The choice of review form remains
  with the user.
- `v3/results/literature.csv` is hand-curated and has no `common.py` metadata columns. `check_results.py` must
  exempt it or give it its own producer rule. That is a main-session decision.
