# t4: Analytical and cycle-level performance models validated against hardware

Scope: analytical or cycle-level models of DNN accelerators, how they were validated (against RTL sim, silicon or an
FPGA board), the reported error, and whether that error is per layer or for the whole network. Key question: has
anyone already shown "analytical model = RTL = board, to the cycle, per layer, for every input" for a CNN
accelerator?

Method: arXiv full texts were downloaded and converted with `pdftotext`. Validation sections were found by grep and
then read. The SCALE-Sim Fig. 4 page was rendered and inspected by eye. Every DOI was checked through Crossref
(title, first author, year), and BibTeX was built with DOI content negotiation. arXiv-only items were checked with
the arXiv export API. All 33 entries in `t4_related.csv` are verified=yes for their metadata. Numbers I took only
from an abstract or a secondary source are labelled as such in the `notes` column.

## (a) What exists

### General DNN-accelerator models (ASIC-oriented)

| Work | Kind | Reference compared against | Reported error | Granularity | Where |
|---|---|---|---|---|---|
| Timeloop [parashar2019timeloop] | analytical, throughput-based | detailed in-house simulator of an NVDLA-derived RTL design; Eyeriss published energy model | cycles: accuracy 78–99 %, mean 95 %; energy within 8 % (107 DeepBench workloads) | per workload (layer/kernel) | Sec. VII-C, Figs. 8–10 |
| MAESTRO [kwon2019understanding] | analytical, data-centric | MAERI RTL simulation (VGG16, 64 PEs); Eyeriss *reported* AlexNet runtime | runtime within 3.9 % absolute error on average; intro says "90-95% accuracy of actual open-source RTL" | per CONV layer | Sec. 4.5, Fig. 9 |
| SCALE-Sim v1 [samajdar2018scalesim] | cycle-accurate simulator (systolic) | in-house OS systolic-array RTL | no number given; bars coincide in Fig. 4 | single matmul "with same size as the array" (4×4 … 90×90) | Sec. III Validation, Fig. 4 p. 6 |
| SCALE-Sim v3 [raj2025scalesimv3] | cycle-accurate simulator | v2 "validated against RTL of systolic array (cycle-accurate)" (stated, not shown); Vegeta RTL; PnR energy | ≤ 5 % (N:M sparsity vs Vegeta RTL); energy 4.6 % / 4.8 %; 2:4 "100 %" vs the STC report | per layer | Sec. VIII, Table III |
| STONNE [munozmartinez2020stonnearxiv] | cycle-level simulator | BSV MAERI RTL (32 MSs, 3 layer types) | cycles: average 15 % (11–19 %); functional: identical outputs for every layer, and identical scores/labels on 50 ImageNet images × 5 DNNs | per layer | Sec. IV-A, Fig. 5 |
| Sparseloop [wu2022sparseloop] | analytical, statistical sparsity | SCNN/Eyeriss V2 simulators, Eyeriss silicon, DSTC/STC papers | 0.1–8 % average; STC 2:4 "exact 2× speedup ... 100% accuracy", with the error source listed as "None (structured sparsity introduces deterministic behaviors)" | per layer / speedup ratio | Sec. 6.3, Table 6, 6.3.5 |
| ZigZag [mei2021zigzag] | analytical | Eyeriss/ENVISION reported values; in-house accelerator post-synthesis RTL; Timeloop | energy 5 % / 7.5 %; vs RTL max 6 % energy, 9 % PE utilisation | per layer | Sec. VI, Figs. 14–16 |
| DeFiNES [mei2023defines] | analytical (depth-first) | DepFiN chip measurements | latency within 3 % (2 nets), 10 % (FSRCNN, controller stalls not modelled) | whole network | Sec. IV, Fig. 11a |
| Stream [symons2025stream] | analytical (layer-fused, multi-core) | DepFiN, Jia et al., DIANA silicon | latency accuracy 96 % / 97 % / 97 % | whole workload | Table 2 |
| ONNXim [ham2024onnxim] | cycle-level NPU simulator | Gemmini RTL (core only) | MAE 0.23 %, correlation 0.99 for GEMMs and convolutions | per op | Sec. V-C, Fig. 3b |
| LLMCompass [zhang2024llmcompass] | analytical/sim | A100, TPUv3, MI210 measured | 10.4 % (operators), 4.1 % (LLM inference) | per op / per stage | Sec. V-C, Fig. 5 |
| DNN-Chip Predictor [zhao2020dnnchip] | analytical | Eyeriss chip; SkyNet FPGA; 65 nm synthesis | latency ≤ 15.51 % (chip), ≤ 16.84 % (FPGA, 7 CONV layers), energy breakdown < 5.28 % | per layer | Sec. 4, Fig. 5, Table 2 |

### FPGA-specific models

| Work | Reference | Reported error | Granularity | Source quality |
|---|---|---|---|---|
| Ma et al. TCAD'20 [ma2020performance] | on-board tests, two FPGAs | "within 3%" | ? | abstract only; the 3 % figure is secondary (search snippet), **unverified** |
| DNNBuilder [zhang2018dnnbuilder] | ZC706/KU115 board | 1.15 % / 2.17 % average | ? | abstract only; numbers secondary, **unverified** |
| AutoDNNchip [xu2020autodnnchip] | Ultra96, edge TPU, edge GPU, ASIC | < 10 % (abstract) | per model | abstract + section head read |
| HARFLOW3D [toupas2023harflow3d] | ZCU106 measured | MAPE 6.64 % over C3D conv layers; the gap comes from DMA burst delays | per layer | Sec. VI, Fig. 6 |
| fpgaHART [toupas2023fpgahart] | board | 5.03–17.32 %, geometric mean 10.75 % | per graph type | grep of text |
| MCCM [qararyah2025mccm] | Vitis HLS synthesis reports (not the board) | accuracy 80.7–100 %, average > 90 %; off-chip access counts "exact since the accesses are deterministic" | per design | Sec. V-B, Table IV |
| Ferianc et al. [ferianc2020improving] | Arria 10 | ML correction gives 30.7 % lower MAE than the analytic model | per layer | abstract |
| FINN estimates (in FINNAS [chauffour2026finnas]) | RTL sim | Spearman ρ = 0.781; "substantial absolute error" | per design | validation paragraph |
| RealProbe [kim2025realprobe] | in-FPGA counters vs Vitis HLS co-sim | co-sim cycles up to 103.8 % off on the board | per design | Sec. I, Fig. 1 |

### Methodologies, not analytical models

- **Gemmini** [genc2021gemmini] runs end-to-end DNNs "on a complete cycle-exact simulated SoC" in FireSim
  [karandikar2018firesim]. Here "cycle-exact" means the simulation is derived from the RTL itself. It is not a
  model checked against RTL, and it is not board-measured performance of the accelerator as deployed.
- **VTA** [moreau2019hardware] uses only an analytical *peak* model to filter designs. TSIM (Verilator) is
  documented in a TVM RFC/README, not in a paper; it is not cited as evidence here.
- **Stand-alone VTA (Airbus/ONERA)** [faure2025standalone] targets certification (DO-254/DO-178C). It runs LeNet-5
  bit-accurate on the functional simulator and in the Chisel cycle-accurate simulator (2972 TensorGemm cycles, 6358
  cycles in total), with no board measurement.
- **FireBridge** [abarajithan2026firebridge]: firmware linked to RTL/gate-level simulation (xsim, VCS, Xcelium) with
  memory-congestion emulation. It is co-verification, not a predictive model.
- **Zeng et al. ICCAD'24** [zeng2024automatic] generate cycle-accurate timing models automatically from accelerator
  RTL. These match the RTL by construction. No board work appears in the abstract (full text was 403).

## (b) Closest works to "model = RTL = board, cycle-exact, per layer, every input"

1. **Kalagi et al. 2026** [kalagi2026fpga] (abstract only). 4×4 OS systolic array on a Basys 3. "Both Vivado XSim
   simulation and on-board runs confirm" a 320 ns tile, and the compute phase is 3N−1 = 11 cycles, so model = sim =
   board for **one 4×4 tile**. A per-layer CNN check, a full test set, and the model as a separate artefact were not
   found in the abstract.
2. **ONNXim** [ham2024onnxim]. Core-model cycles vs Gemmini RTL: 0.23 % MAE, which is nonzero. No board. Memory and
   NoC are excluded.
3. **SCALE-Sim v1/v2** [samajdar2018scalesim; raj2025scalesimv3]. The figure suggests visual cycle agreement with OS
   systolic RTL, for single-array-size matmuls only. v3 restates this as "cycle-accurate" against RTL. No board, and
   no error number.
4. **Sparseloop** [wu2022sparseloop]. States outright that structured sparsity is deterministic and reaches "100 %"
   on STC. That figure is a 2× speedup ratio against the vendor report, not cycle equality with RTL or hardware.
5. **Ma et al. TCAD'20** [ma2020performance] and **DNNBuilder** [zhang2018dnnbuilder]. FPGA analytical CNN models
   checked against on-board results, with errors of a few percent (secondary numbers). Not exact.

Also relevant: Rahman et al. 2026 [rahman2026edge] give a closed-form per-layer cycle model for an INT8 1D systolic
CNN on an iCE40, but I found no model-vs-RTL/board comparison in the sections read. STONNE is **bit-exact**
end-to-end (50 images × 5 DNNs) but only within 15 % on timing.

## (c) Verdict for this topic: partly done; the strong cycle-exact form appears novel

- **Already done:** analytical models validated against RTL, silicon and FPGA boards, with errors of roughly
  0.2–17 %, both per layer and whole network. Cycle-exact *RTL-derived* simulation (FireSim/Gemmini, Zeng'24,
  Verilator) is standard methodology. Bit-exact functional agreement on many inputs is also done (STONNE).
  Determinism of structured-sparsity speedup is stated (Sparseloop).
- **Not found in any sections read:** an analytical/Python cycle model that agrees with **both** RTL simulation
  **and** on-board hardware cycle counters with **0 cycles of error**, **per layer**, for **every input** of a full
  test set, on a complete CNN accelerator. The best reported agreement is 0.23 % MAE vs RTL (ONNXim), or 96–97 %
  against silicon (Stream). The only "sim = board" claim (Kalagi'26) covers one 4×4 tile and is abstract-only.
  Several papers explain their nonzero error by effects our design removes on purpose: DMA/DDR burst delays
  (HARFLOW3D, RealProbe), controller stalls (DeFiNES), and fill/drain assumptions (Timeloop).
- **Implication for V3:** state the novelty as the conjunction "exact (0-cycle) agreement model = RTL = board, per
  layer, for every input, kept while adding reconfigurable dataflow and sparsity skipping". Do not claim "the first
  accurate performance model". With data-dependent activation skipping, exactness requires the model to consume the
  actual activations; the "provable dense upper bound" is a separate claim (WCET topic).

## (d) Risks

- Kalagi'26, Ma TCAD'20, DNNBuilder and Zeng'24 were read only as abstract or metadata. Their full texts could
  contain per-layer board-cycle equality. Kalagi'26 is open access (Discover AI) and should be read in full before
  any claim of being first.
- Industrial practice is unpublished. Vendor IP such as the DPU, NVDLA and hls4ml latency-mode designs is often
  deterministic, and its HLS-reported latency may equal on-board latency for fixed-latency designs. hls4ml papers
  were not read here. A reviewer could argue that "cycle-exact for a statically scheduled FPGA pipeline is expected,
  not novel". Counter with RealProbe (co-sim up to 103.8 % off the board) and with the cost of keeping exactness
  under reconfiguration and sparsity.
- "Cycle-exact" collides with FireSim's usage, so define the term explicitly: an independent model, not RTL-derived.
- Safety and certification VTA work (Airbus/ONERA) and Restuccia & Biondi RTSS'21 are near the real-time framing.
  Coordinate with the WCET topic.

## (e) UNVERIFIED (not used as support)

- NVDLA performance model/estimator spreadsheet (nvdla.org). Not fetched.
- hls4ml latency estimate vs board (Duarte et al. 2018 JINST; hls4ml 2025, arXiv 2512.01463). Not read.
- fpgaConvNet TNNLS'19 model accuracy. Not read; the fpgaConvNet arXiv 1711.08740 text had no accuracy figure in grep.
- SqueezeJet-2 "max 8.3 %, avg 4.45 % vs hardware trace" (search snippet). Source paper not identified; arXiv
  2602.04044 did not contain these numbers.
- DNNBuilder 1.15 % / 2.17 %, Ma TCAD'20 "within 3 %", AutoDNNchip per-platform 4.85 / 3.73 / 6.57 %: secondary
  search snippets only.
- VTA TSIM (TVM RFC #3009 / README). Not peer-reviewed; not fetched.
- SCALE-Sim v2 [samajdar2020systematic, key in the existing refs.bib] "cycle-accurate vs RTL": seen only as restated
  in the v3 paper.
