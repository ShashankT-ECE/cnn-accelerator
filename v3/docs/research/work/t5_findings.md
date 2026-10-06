# t5 findings: ResNet / MobileNet accelerators on Zynq UltraScale+ (KV260 first)

Pass date 2026-10-06. Companion files: `t5_prior_numbers.csv` (45 design points, 41 verified),
`t5_related.csv` (25 works), `t5_refs.bib` (25 entries). Reused rows are copied from
`gos-v3/v3/results/literature.csv` with their source_location unchanged.

## (a) What exists

**KV260 / K26, custom (non-DPU) designs. These are the priority-1 results.**
- **bosio2025nn2fpga** (IEEE TCAD 44(5), 2025, open access, read in full). This is nn2fpga, an HLS
  compiler that builds a layer-pipelined dataflow design with a binary-integer program for
  per-layer parallelism. Board results (Table VI) and post-route resources (Table VII):
  - ResNet20 / CIFAR-10, INT8, KV260: 250 MHz, 7601 FPS, 0.318 ms, 6.10 W, 91.3 %, 636 DSP,
    65.0 kLUT, 60.5 BRAM, 12 URAM.
  - ResNet8, INT8, KV260: 30153 FPS, 0.046 ms, 6.67 W.
  - ResNet8, 4-bit, KV260: 61035 FPS, 6.12 W.
  - ResNet20 and ResNet8 on Ultra96, and MobileNetV2 / ImageNet on ZCU102 (2115 FPS, 13.5 W).
  - Power was "measured using sensors on the power rails, representative of the entire board".
  - The arXiv preprint (minnella2023resnet, 2309.15631) has the same FPS and latency but
    different clock, resources and power (274 MHz, 3.61 W). Cite the TCAD version.
- **zhang2022wsqaddernet** (ICCAD 2022). ResNet20 and AdderNet, INT8, on KV260 at 200 MHz:
  1.221 ms and 0.624 ms. These numbers come only from NN2FPGA Tables VI–VII, which also notes
  that the reported 1.07 W / 1.52 W is below the KV260 idle power. The primary was not read
  (closed access).
- **hamanaka2023exploration** (IEEE Access 2023, read in full). Same board and same model:
  FINN against the Vitis AI DPU for ResNet-8 / CIFAR-10 on a custom K26-SoM board, with a CT-3
  board meter.
  - DPU B4096, 1 core, 300 MHz: 0.955 ms, 1,024 FPS, 5.65 W, 690 DSP, 49,418 LUT.
  - DPU B512: 1.114 ms, 4.98 W.
  - 4×B512 at 200 MHz: 4,458 FPS, 6.42 W.
  - FINN INT4 at 225 MHz: 13,475 FPS at 5.89 W. Its 0.154 ms latency was **computed from
    Verilog-simulation cycle counts**, not measured on the board.
- **takahashi2024efficiera** (arXiv). W1A2 ResNet-style nets on KV260 with LeapMind's
  commercial Efficiera IP (0 DSP): ERNs18 3.60 ms, ERNs50 9.64 ms. No power and no clock
  reported.
- **saha2026potacc** (TCASAI 2026). CPU + FPGA TFLite delegates on KV260 at 250 MHz, with
  energy from the on-board sensors (inference minus idle). On KV260 the INT8 MobileNetV2
  delegate (110.2 ms) is slower than the 4-thread CPU (65.6 ms).
- **magalhaes2023benchmarking** (EAAI 2023). KV260 DPU RetinaNet-ResNet-50 at 13.13 FPS,
  board power measured with a multimeter. This is detection, not classification.
- Vendor: the Vitis AI 2.5 model zoo lists KV260 B4096F @ 300 MHz (ResNet-50 94.21 fps,
  MobileNetV2 290.62 fps, both reused). It gives no power and no latency columns.
- **costa2026throughput** (KV260 DPU multithreading, JRTIP 2026). The DOI is verified, but the
  full text is still behind a Springer login, so no numbers were used.

**Other ZU+ boards (priority 2):**
- **wu2019mobilenets** was upgraded to verified from the primary full text. Xilinx DPU
  lineage: MobileNetV2 8b at 205.3 fps on ZU2 (430 MHz, 212 DSP, 31198 LUT) and 809.8 fps on
  ZU9 (333 MHz, 2070 DSP, 161944 LUT), 68.1 %. No power reported.
- **xing2020dnnvm** was upgraded from the arXiv v2 full text; the TCAD version was not read.
  - ResNet50, ZU9, batch 3, 330 MHz: 1.38 TOPs/s.
  - ZU9, batch 1, 500 MHz: 0.68 TOPs/s.
  - ZU2, 330 MHz: 228.7 GOPs/s.
  - Platform power: ZU9 22.8 W, ZU2 7.5 W.
  - The old secondary row (500 MHz / 2520 DSP / 80.95 fps) mixed the batch-1 column with the
    device's DSP capacity. Corrected rows are in the CSV.
- **dong2021hao** (FCCM 2021): NAS + quantization + hardware search on Ultra96. 72.45 % at
  50 fps, 5.5 W (4.3 W with the PL idle), 360 DSP.
- **salami2020undervolting** (DSN 2020): ZCU102 3×B4096 DPU, PMBus rail power 12.59 W averaged
  over the benchmarks.
- **tong2024feather** (ISCA 2024): per-layer dataflow and layout switching, deployed end to end
  on ZCU104 at 100 MHz. It reports only normalized results against the DPU (2.65×).
- Reused verified ZU+ rows: Synetgy, FracBNN, FINN-R CNV (Ultra96); DeepDive and FILM-QNN
  (ZCU102); DPU numbers via DPUV4E (ZCU102); model-zoo ZCU102/ZCU104.

**Still secondary (verified=no):**
- **jiang2023fulldataflow**: MobileNetV2 ZCU102, 1910 FPS, 1283 DSP, 333 MHz. Two independent
  secondary sources now agree (NN2FPGA Tables VI–VII and Zhao et al.). The primary is paywalled.
- **li2021dynamic**: 381.7 fps on ZCU102. Paywalled, abstract only.

## (b) Closest works to the thesis

**bosio2025nn2fpga (closeness 3)**
- Does:
  - Custom INT8 ResNet-20 / CIFAR-10 on KV260, all weights on chip, measured board power.
  - It is the direct competitor on workload and platform for V3.
  - Uses an analytic per-layer cycle model (Eq. 14) for its design-space exploration.
- Not found in the sections read (abstract, III-F, IV, V, Tables V–VII):
  - any claim that model = RTL = board latency, cycle for cycle;
  - any latency guarantee or WCET;
  - sparsity skipping;
  - bit-exactness against a golden model.
- Eq. 14 explicitly ignores pipeline depth, so the model is approximate by construction.
- Results are throughput-oriented (batch 1000).

**hamanaka2023exploration (closeness 2)**
- Does: K26 DPU vs custom dataflow on the same board, with a meter.
- The FINN latency is derived from simulation cycles. This is the nearest thing to "cycle
  count from a model", but there is no board-side cycle validation.
- Not found: predictability or guarantees.

**zhao2021dualopu (closeness 3, but not ZU+: its own design is on XCK325T)**
- Does:
  - Per-layer heterogeneous engines for depthwise and regular convolution, with layer splitting.
  - A cycle-accurate instruction-level simulator validated against the board within <1 %
    (MobileNet v1: 757149 vs 755857 cycles, −0.2 %).
- This is the strongest counter-evidence on the "cycle model vs board" axis, but it shows close
  agreement, not exact agreement, and makes no guarantee.

**tong2024feather (closeness 3)**
- Does: per-layer dataflow (and layout) selection, running on a ZU+ board (ZCU104).
- This is direct prior art for the "reconfigurable dataflow / per-layer mode selection" part of
  the thesis.
- Not found in the sections read: timing guarantees, cycle-exact validation, sparsity.

**wu2019mobilenets, li2021dynamic (closeness 2)**
- Separate depthwise engines (Wu) and per-layer adaptive dataflow scheduling (Li, abstract only)
  on ZU+.
- These are prior art for "depthwise support + per-layer mode", without timing claims.

## (c) Verdict for t5's part of the thesis

- **Custom INT8 ResNet-20 on KV260 is already done** (nn2fpga; also WSQ-AdderNet, which is
  secondary). The earlier v3 note "no reviewed source reports an INT8 ResNet-20 … on a ZU+ or
  KV260 custom design" (LITERATURE.md §a) is now **wrong and must be corrected**.
- nn2fpga sets a strong performance bar: 0.318 ms, 7601 FPS, 6.10 W board power, 636 DSP at
  250 MHz.
- A V3 design will very likely not beat it on throughput; nn2fpga is a fully unrolled,
  layer-pipelined dataflow design.
- **MobileNet-style CIFAR-10 on KV260 with a custom accelerator:** no board-level latency result
  was found. PoTAcc reports only CIFAR-10 accuracy for MobileNetV2/ResNet20 plus
  ImageNet-size CPU+FPGA delegate latencies. This part is still open, but it is a workload gap,
  not a contribution by itself.
- **Timing-predictable / cycle-exact angle on ZU+ ResNet/MobileNet accelerators:** **not found**
  in any t5 paper read. Approximate models exist (nn2fpga Eq. 14, HAO, DNNVM evaluation, the
  dual-OPU simulator within <1 %), but no exact model = RTL = board agreement and no
  compile-time latency guarantee.
- So, within t5: the platform and workload are **already done**; the predictability claim is
  **novel as far as t5 sources go**. The verdict for t5 is **partly done**.
- t1 (real-time / WCET: Restuccia RTSS'21, EXPRESS) decides the predictability question overall.

## (d) Risks

1. **Throughput comparison.** A reviewer will put V3 next to nn2fpga's 7601 FPS and 0.318 ms on
   the same board. V3 must be framed on latency guarantees, exactness and reconfigurability,
   and should compare single-image latency (nn2fpga's 0.318 ms latency column; how its latency
   was measured, batch size and timer, was not found in the sections read).
2. **Power-method mismatch across papers.** Methods include:
   - nn2fpga: rail sensors, whole board.
   - Hamanaka: CT-3 USB meter.
   - Magalhães: multimeter at the board input.
   - PoTAcc: on-board sensors, minus idle.
   - Salami: PMBus PL rails only.
   - WSQ-AdderNet: unknown method, values below idle.

   Never compare our INA260 SOM-rail numbers to these without stating the method.
3. **Version drift.** nn2fpga's arXiv and TCAD versions disagree, and DNNVM was read only on
   arXiv. Cite published numbers only where the published version was read.
4. **Reconfigurable-dataflow novelty is weak on its own.** FEATHER (ZCU104, ISCA'24),
   Li 2021 TCAS-I and Wu 2019 already cover per-layer dataflow and depthwise engines on ZU+.
5. The KV260 is listed as "xczu5eg" in nn2fpga's own resource table. Our device string is
   XCK26; do not copy theirs.

## (e) UNVERIFIED (citation verified, numbers or content not read)

- **zhang2022wsqaddernet** (DOI ok, closed access): every number is secondary, via NN2FPGA.
- **jiang2023fulldataflow, li2021dynamic** (DOI ok, paywalled): numbers are secondary or come
  from the abstract.
- **costa2026throughput** (DOI ok, Springer login): no numbers used. Web summaries quote
  5.66–7.15 W and ResNet-50 26.4→83.9 img/s; these are unverified and must not be cited.
- **zhang2024addernet2** (DAC 2024, DOI ok): not read, and the device is unconfirmed.
- Leads not followed:
  - "MobileNetV2 Accelerator for Power and Speed Balanced Embedded Applications" (IEEE, doc
    9988258; CIFAR-10, 86.03 %, 2.741 W per a search snippet; device unknown).
  - "Efficient Dynamic Reconfigurable CNN Accelerator for Edge Intelligence Computing on FPGA"
    (Information 14(3):194, 2023; MDPI returned 403 to our requests).
  - "Dataflow-Reconfigurable CNN Accelerator Design" (IEEE Micro 2026, DOI
    10.1109/mm.2026.3671071; for t2).
  - Tensil ResNet-20 on Ultra96/ZCU104 (Hackster blog, 403).
- **Mix&Match and N3H-Core** were read but are Zynq-7000 (XC7Z020/045), not ZU+, so they were
  excluded. N3H-Core's Table 4 independently repeats Wu 2019's ZU2/ZU9 numbers.
