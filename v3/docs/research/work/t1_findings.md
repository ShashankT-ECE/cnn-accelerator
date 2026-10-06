# t1: Timing-predictable / WCET-analysable DNN accelerators and real-time DNN inference on FPGA/SoC

Date: 2026-10-06. All keys refer to `t1_refs.bib` / `t1_related.csv`. "Read" levels are recorded per row in the CSV.
Searches covered web search, arXiv API, Crossref and DataCite. Phrasings used: time-predictable, WCET, deterministic latency,
cycle-accurate / cycle-exact, response-time analysis, mixed-criticality, safety-critical/avionics/DO-254, static/compile-time
schedule, time-triggered, zero jitter, bit-exact + cycle-accurate, sparse + WCET. Semantic Scholar returned HTTP 429, so it was not used.

## (a) What exists: five separate lines of work

1. **Real-time systems analysis of a black-box FPGA DNN accelerator (the AMD DPU).** This line comes from the Pisa / Sant'Anna group.
   - `restuccia2021timepredictable` (RTSS'21) profiles the DPU on ZCU102 at 330 MHz with a custom AXI profiler. It builds a
     series-parallel model and a response-time analysis that gives *safe upper bounds*. A helper module (DICTAT, which fetches
     instructions from on-chip memory) makes the bounds at least 12% less pessimistic.
   - In their measurements, latency is *nearly* constant but not constant. Table II, 1000 runs: Lane Detect min 7.09 / avg 7.1 /
     max 7.12 ms. Pedestrian SSD 8.39 / 8.4 / 8.41 ms. The paper attributes the variation to the DRAM controller (Obs. 4).
   - `aromolo2026timepredictable` (RTAS'26, Outstanding Paper) extends this to multi-core DPUs (3 cores, 90 MHz, ZCU102). It uses
     AXI bandwidth reservation (ABU), DNN splitting and fixed-priority scheduling.
   - In the RTAS'26 paper the WCETs are **profiled** (10,000 inferences under memory-killer interference, Sec. VII-A), not derived
     from a cycle model.
   - Related, from the same field: `mattheeuws2021analyzing` (PL-to-CPU memory interference up to 16x, secondary source) and
     `cilardo2024systematic` (multitask DPU characterization, metadata only).
   - The 2026 MDPI review `hussein2026fpga` (abstract) states: "none of the surveyed FPGA-based AI accelerator studies provide WCET
     bounds, although recent analytical models for multi-DPU architectures demonstrate the feasibility". That statement covers
     only its 36 surveyed studies.

2. **Time-predictable processors and code generation applied to NNs.** This is the software / WCET-tool line.
   - Code generation for analysable NN code: ACETONE (`silva2022acetone`, `silva2023extending`), which generates C code and
     bounds it with the OTAWA WCET tool; Keras-to-C (`pearce2021designing`); and NeuralCasting on Patmos
     (`cerioli2025timepredictable`).
   - Vicuna (`platzer2021vicuna`) is a vector coprocessor free of timing anomalies, run on a Xilinx 7-series FPGA at 80 MHz. Its
     benchmarks include a 3x3 convolution, and "each benchmark program always executes in the exact same number of CPU cycles".
     However, the performance figures come from measurement and are *not* compared with an analytical prediction (Sec. 5).
   - Kirschner et al. use Vicuna cores with a compile-time static DMA schedule and compose a WCET from subtask WCETs
     (`kirschner2024wip`, RTSS'24 WiP, where the FPGA evaluation is future work).
   - The follow-up `kirschner2026multivic` runs on a VCU128 board. It reports "very low" but non-zero fluctuation on a matmul
     benchmark (median and std-dev, Fig. 4).

3. **Deterministic, compiler-scheduled ASIC dataflow processors.**
   - Groq TSP (`abts2020think`; determinism claim checked only through a secondary summary) and its multi-chip version
     (`abts2022software`, full text read). The 2022 paper says "Execution latency of all instructions is known statically (at
     compile time)".
   - The actual validation is end-to-end. In Sec. 5.4 / Fig. 17 (BERT-Large, 24,240 runs) the compiler estimate is "within 2% of
     the actual measured latency in the majority of cases", and the deviation is attributed to PCIe. The paper reports **no
     cycle-exact equality check on the device counters**.

4. **Real-time and certification-oriented custom FPGA DNN accelerators.**
   - The Siegen line, time-triggered VTA (`ezekiel2023optimization`, `bebawy2025superscalar`), targets temporal predictability.
     For these we saw metadata plus secondary snippets only (average inference times 53 vs 51 ms; STT-VTA 20-41% faster).
   - `ji2025art` (GLSVLSI'25, abstract only) turns FPGA accelerators into "real-time guaranteed" ones through schedulability
     analysis, preemption and hardware G-EDF on a VCK190.
   - The avionics line (ONERA/Airbus) builds a certification-minded stand-alone VTA compiler (`faure2025opensource`). It is
     evaluated on VTA simulators only, according to the abstract.
   - `garofalo2025reliable` is an ASIC SoC that bounds interference.
   - Daedalean's DO-254-aligned CNN accelerator is listed under UNVERIFIED below.

5. **Fixed-latency-by-construction dataflow (FINN / hls4ml) and latency models of FPGA CNN accelerators.**
   - hls4ml for the L1 trigger (`duarte2018fast`) gets its latency from HLS reports. A full-text keyword search found no on-board
     latency check against the estimate.
   - FINN's own documentation calls its critical-path latency analysis "very pessimistic" (`finndocs2026analysis`); measured
     latency comes separately from rtlsim.
   - Typical published model-versus-board accuracies are approximate:
     - 2.53% average deviation (`jiang2019superlinear`, which also says FPGAs give "deterministic timing characteristics")
     - 6.64% MAPE (`toupas2023harflow3d`)
     - max 8.3% (`mousouliotis2019software`, where HLS co-sim was off by up to 33.4%)
     - max 9.17% (`xu2020autodnnchip`)
     - <=11% (`zhao2020dnnchip`)
   - `kim2025realprobe` shows HLS C/RTL co-sim cycle counts up to 103.8% off in-FPGA execution.
   - The closest to exact that we found is `lubeck2025automatic`. Its model reproduces the UltraTrail ASIC's TC-ResNet8 latency
     "almost exactly to 22484 clock cycles", but overestimates by 3 cycles. That comparison is against the reference
     model/RTL, not a board.

## (b) Closest works to the thesis

The thesis has four parts:
- (i) a custom FPGA CNN accelerator
- (ii) latency known at compile time, with model = RTL = board cycle-exact
- (iii) reconfigurable dataflow
- (iv) sparsity with a provable dense upper bound

| key | does | not found in sections read |
|---|---|---|
| restuccia2021timepredictable | Predictability study on Zynq US+ (ZCU102); analytical RTA upper bound; measured min/avg/max | custom accelerator; exact predicted cycle count; equality model=board; sparsity; dataflow modes; bit-exactness |
| aromolo2026timepredictable | Multi-DPU WCET-based schedulability, bandwidth regulation, measured-WCET profiling on ZCU102 | analytical cycle model; exact equality; sparsity; custom RTL |
| abts2022software (+abts2020think) | Statically scheduled, deterministic, compile-time-known instruction latency; compiler estimate within 2% of measured | FPGA; CNN-on-edge; cycle-exact board-counter equality; sparsity |
| platzer2021vicuna | Same cycle count on every run on an FPGA, measured; WCET = measured since no data-dependent control flow | analytical prediction compared with measurement; CNN accelerator; whole networks |
| kirschner2024wip / kirschner2026multivic | Compile-time static schedule + compositional WCET for NN inference; FPGA prototype | exact equality (non-zero fluctuation reported); CNN-specific dataflow; sparsity |
| bebawy2025superscalar / ezekiel2023optimization | Time-triggered custom FPGA DNN accelerator (VTA) for safety-critical RT | not read in full; secondary snippets report averages only |
| lubeck2025automatic | Analytical latency within 3 cycles of the reference for an ASIC accelerator | board measurement; zero-cycle equality |

## (c) Verdict for t1 (the timing-predictability part of the thesis)

**Partly done. The specific combination is novel as far as our search reached.**

- **Already done.** Several pieces exist:
  - "Time-predictable DNN acceleration on Zynq UltraScale+ FPGA SoCs" is an established RTSS/RTAS topic, with WCET bounds and
    response-time analysis on ZCU102 using the DPU.
  - "Deterministic, compile-time-known latency" is a published architectural claim (Groq TSP).
  - "Same cycles on every run" was measured on an FPGA for a timing-predictable vector processor (Vicuna).
  - Compile-time static schedules for NN inference exist on predictable multicore processors (Kirschner).

  So V3 must not claim to be "the first time-predictable DNN accelerator on FPGA" or "the first with compile-time-known
  latency".
- **Not found (gap).** We found no work that combines all of the following:
  - a custom FPGA CNN accelerator whose per-layer and per-network latency is computed *exactly* (to the cycle) at compile time
    by a model
  - verification of *equality* (not a bound, not "within x%") between that model, RTL simulation and on-board hardware cycle
    counters, on every job
  - bit-exact outputs

  The best-documented prior works either bound the latency (DPU line), or measure constancy without predicting it (Vicuna), or
  predict within about 2-10% (Groq; FPGA performance models).
- **Sparsity and dataflow reconfiguration under a timing guarantee.** We found nothing that combines activation-sparsity
  skipping or structured weight sparsity, or per-layer dataflow mode selection, with a WCET or exact-latency guarantee. Sparse
  accelerators in the searched literature report speedups. One adversarial-sparsity paper (arXiv 2006.08020, seen as a search
  hit only) shows that activation density can be driven up by inputs, which motivates a dense upper bound. We found no
  sparse-accelerator paper that gives a provable dense worst case together with exact data-independent latency for the
  structured part.
- **The defensible novelty statement** therefore has four parts:
  - exactness rather than bounding
  - three-way equality (model = RTL = board) shown on every job
  - the guarantee kept under reconfigurable dataflow
  - a *two-level* guarantee under sparsity: exact for static structured weight sparsity, and a proven dense upper bound plus an
    exact post-hoc cycle count for activation skipping

## (d) Risks

1. **Scope of "exact".** Exactness holds only for the accelerator-internal cycle count, with weights in BRAM and no DRAM on the
   critical path.
   - Both the DPU line and MultiVic attribute their residual jitter to DRAM.
   - If V3 needs DRAM (ResNet-20 and MobileNet weights may not all fit; that is outside t1), board-level exactness may break.
     The claim would then have to become "exact compute cycles + bounded memory".
   - State the exact scope (counter start/stop points, host overhead excluded).
2. **Reviewers from the real-time community** will say that a fixed-function, data-independent pipeline is trivially
   deterministic (hls4ml and FINN are "fixed latency by construction").
   - The contribution must therefore lie in keeping exactness *with* runtime-configurable dataflow and sparsity.
   - It must also lie in the verification methodology (equality on every job, full datasets), not in determinism as such.
3. **Activation skipping makes latency data-dependent.** The guarantee then becomes a WCET bound (the dense upper bound) plus
   an exact *per-input* prediction. That prediction is only computable once the activations are known, so it is not available
   at compile time. Phrase it carefully: "compile-time bound, run-time exact accounting".
4. **Unread full texts that could undercut the novelty claim:**
   - TT-VTA / STT-VTA (`bebawy2025superscalar`): time-triggered implies a fixed schedule, so it may report exact cycles.
   - ART (`ji2025art`).
   - The Daedalean DTA paper.

   Read these before claiming novelty in writing.
5. **Groq TSP details** (ISCA'20) were confirmed only through a secondary summary. Cite the claim, not the numbers.

## (e) UNVERIFIED (not usable as supporting references)

- Haandbaek et al., "FPGA Design and Implementation of a High-Performance Avionics Certifiable CNN Accelerator", Embedded World
  2024 (Daedalean Tensor Accelerator, Intel Agilex, DO-254-aligned).
  - Source: futuretransport-news.com pages (secondary). No DOI found in Crossref. verified=no.
- The original TT-VTA paper (Obermaisser group), cited inside `bebawy2025superscalar` as the earlier work.
  - A Crossref search did not find it separately. verified=no.
- "A Uniform Latency Model for DNN Accelerators" (KU Leuven, lirias PDF): search hit only, not fetched. verified=no.
- FPGN (arXiv 2607.08427): a search snippet claims "predicted cycle counts ... precisely matching simulated hardware
  behavior".
  - The arXiv abstract was fetched and does not say this. It is a LUT-network accelerator; full text not read. verified=no for
    that claim.
- "Sparsity Turns Adversarial: Energy and Latency Attacks" (arXiv 2006.08020): search hit only. verified=no.
- `costa2026throughput` (KV260 DPU multithreading): already flagged in the earlier LITERATURE.md. Not re-checked here.
