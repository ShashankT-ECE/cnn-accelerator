# t3: Sparsity-skipping accelerators and latency predictability under sparsity

Scope: static structured weight sparsity (N:M, block, channel, pattern), activation zero skipping, and above
all work that bounds, predicts or guarantees latency under sparsity. This extends `v3/docs/LITERATURE.md`
(Cnvlutin, SCNN, Cambricon-X/S, EIE, Zhu2020, SPEC, column combining and NVIDIA 2:4 are not repeated here).
All keys are in `t3_related.csv` and `t3_refs.bib`. Page numbers are arXiv PDF pages unless stated otherwise.

## (a) What exists

1. **Structured sparsity is widely described as "deterministic", but nobody turns that into a stated guarantee.**
   - Sparseloop [wu2022sparseloop] validates against the NVIDIA Sparse Tensor Core at "100%" accuracy. Its
     Table 6 (p.16) explains why: "structured sparsity introduces deterministic behaviors". Sec. 6.3.5 (p.18)
     says the same.
   - SCALE-Sim v3 [raj2025scalesimv3] (Sec. VIII, p.10) matches 2:4 at 100% and VEGETA RTL for 2:4 and 1:4
     with an error of 5% or less. It also notes: "While Sparseloop claims compute cycles are deterministic,
     memory stalls are not."
   - In S2TA [liu2022s2ta] (Sec. 3.1, p.4), DBB "constrains the maximum number of NNZs in a block such that
     the maximum workload is known at design time."
   - None of these is a real-time paper.
2. **Static compile-time schedules for weight sparsity already exist on FPGAs.**
   - HPIPE [hall2020hpipe] compiles zero-weight skipping per layer for Stratix 10. Its analytic throughput
     model for sparse layers is "within 1% of the actual throughput" (Sec. IV, p.5).
   - SparHiXcel-v2 [zarei2026sparhixcel, arXiv Jul 2026] uses a Python scheduler that "statically determines
     all PE assignments ... and data movement schedules" (Sec. III, p.6). It computes speedup "with
     cycle-accurate modeling" (Sec. IV-A, p.6). The target is an XCKU19P.
   - Bit-Tactical [delmaslascorz2019bittactical] schedules zero-weight skipping statically in software,
     ahead of execution (arXiv precursor, Sec. 1, p.2).
   - hls4ml [duarte2018fast] optimises zero-weight multiplications out of the FPGA implementation
     (Sec. 3, p.18), so latency is fixed by the HLS schedule.
   - Load-balance-aware pruning (ESE [han2017ese]) and bank-balanced sparsity [cao2019bankbalanced] make the
     work per PE equal. This is from secondary summaries only.
3. **Activation zero skipping is data-dependent by construction.**
   - NullHop [aimar2019nullhop] runs on a Zynq 7100. Its pipeline is "though data-dependent" (Sec. V, p.10).
   - SparTen, GoSPA, Sparse-PE and Véstias et al. were seen only as metadata or abstracts.
   - None of these works states a worst-case or dense upper bound in the text read.
4. **Variable latency under activation sparsity is a known real-time and security problem.**
   - Krithivasan et al. [krithivasan2020sparsity] (TCAD 2020) cut activation sparsity by 1.16x–1.82x and
     slow a Cnvlutin cycle-accurate simulator by 1.12x–1.43x.
   - They state: "even if designers account for the worst case sparsity present in the clean inputs, the
     attack ... causing inference time and energy to exceed the limits of even a conservative design"
     (Sec. VI, p.9). Sec. I (p.2) gives a frame-loss example under a real-time budget.
   - Sponge examples [shumailov2021sponge] are the same class of attack (metadata only).
   - SparsePixels [tsoi2025sparsepixels] says that "methods with input-dependent runtime ... would fail to
     meet the strict latency bound required by trigger systems" (Sec. 1, p.2).
5. **Analytical sparse models are approximate when sparsity is data-dependent.**
   - Sparseloop reports 0.1% to 8% average error. Against DSTC it reaches 92.4% accuracy (7.6% error,
     Fig. 13). For the Eyeriss V2 PE under a uniform density model the error is up to 7% (Sec. 6.3.2,
     p.17). These baselines are simulators or paper numbers, not its own RTL or a board.
   - TeAAL [nayak2023teaal] measures Sparseloop at 187% average error on ExTensor (Fig. 10a, p.12). TeAAL's
     own error is 2.5% to 9.0% against published simulators.

## (b) Closest works to the thesis

| Work | Latency under sparsity known at compile time? | Stated dense upper bound? | Prediction validated against RTL/board, with error |
|---|---|---|---|
| **S2TA** [liu2022s2ta] HPCA'22 | Yes, per layer: NNZ per block is fixed per layer, and the time-unrolled TPE spends NNZ cycles per block (Sec. 5, p.6). This holds for activations too, because DAP prunes them to top-NNZ (lossy). | A maximum per block "known at design time" (p.4). Dense mode exists. A dense bound for exact skipping was not found. | RTL simulation gives the performance figures (Sec. 7). No model-vs-RTL or silicon/board error was found. ASIC 16/65 nm, no FPGA. |
| **SparsePixels** [tsoi2025sparsepixels] arXiv 2025/26 | Yes. Latency is "frozen at synthesis, exact constants for every input regardless of the input sparsity" (p.13). | Constant by design. It is not a bound: pixels beyond N_active are discarded (lossy) and the data gives no speedup. | Only HLS C-synth and logic-synth reports (Sec. 4). No board measurement was found. |
| **Sparse by Command** [ahmad2026sparsebycommand] MICRO'26 | The tile mask is fixed per task command before inference, so the work is static per command. | Not stated. | Measured on an Alveo U50 with HW cycle counters (Table 2, p.8): HBM stalls of 0.57–0.77 M cycles. No analytical prediction compared with the counters was found. |
| **HPIPE** [hall2020hpipe] FPGA'20 | Yes for weight sparsity (static per layer). | Not stated. | Analytic model "within 1% of actual throughput" (p.5). Whether "actual" means simulation or board was not determined. |
| **SparHiXcel-v2** [zarei2026sparhixcel] arXiv'26 | Yes for weights (static scheduler with cycle-accurate model). | Not stated. | No comparison of scheduler cycles with RTL or board was found. |
| Sparseloop [wu2022sparseloop] MICRO'22 | Statistical expectation, not a guarantee. Deterministic for STC. | No (it mentions average and worst-case *format overhead*, Sec. 5.3.3). | Against other simulators/papers: 0.1% to 8% average. Not against its own RTL or a board. |
| Krithivasan [krithivasan2020sparsity] TCAD'20 | No (it shows the problem). | No. It argues that bounds from clean inputs fail. | Cycle-accurate simulator of Cnvlutin, no board. |

Not found in any work read: an accelerator that does **all** of the following:
- exact (lossless) data-dependent activation skipping with a proven, stated dense upper bound per job;
- static structured weight sparsity with compile-time exact cycles;
- a cycle model checked against RTL simulation **and** board counters with zero error (cycle-exact);
- a real-time/WCET framing.

Searches covered: web and arXiv with phrasings including time-predictable, WCET, worst-case, deterministic,
upper bound, real-time, static schedule, load imbalance and predictable sparse, plus Crossref. Semantic
Scholar and DBLP API calls failed (rate limited or no JSON). The bounding relies mainly on full-text reading
of the 12 papers in the "sections_read" column.

## (c) Verdict for t3: partly done

- **Static structured weight sparsity with a compile-time-known schedule is already done.** HPIPE,
  SparHiXcel-v2, Bit-Tactical's weight scheduling, hls4ml, ESE/BBS and the N:M/2:4 line all do it. Sparseloop
  and SCALE-Sim v3 state explicitly that structured sparsity is deterministic. V3 cannot claim this part as
  novel. It can claim a cycle-exact model = RTL = board check of it on the KV260, which none of these report
  (HPIPE comes closest, at 1%).
- **Activation skipping with a bounded or known latency is partly done, in two lossy forms.**
  - S2TA bounds activation work by pruning activations (DBB/DAP).
  - SparsePixels makes latency constant by capping active pixels.
  - Both give up exactness. Neither is validated on a board.
- **Not found:** exact zero-skipping, for example whole all-zero K-steps (V3 C2), with a stated per-job dense
  upper bound (latency ≤ dense cycles, hence WCET = dense model), plus exact per-input cycle prediction from
  the actual activations, validated cycle-exact against board counters. Read narrowly, this is novel.
- **What the novelty rests on:** it is mainly verification and predictability methodology. The
  architectural idea of skipping zeros is not new. Sparse by Command is the closest hardware-and-measurement
  peer (SystemVerilog, INT8, FPGA, cycle counters, autonomous-driving framing), but it neither predicts
  latency nor gives a bound.

## (d) Risks

1. **The dense bound may look trivial.** Reviewers may say that any skipping design is never slower than
   dense. That is false whenever skipping adds detection or bubble cycles, or memory stalls (Sparse by
   Command: about 0.58 M stall cycles per inference; SCALE-Sim v3: memory stalls are not deterministic).
   V3 must prove the bound in the RTL, including control overhead and DMA, and show it on the board.
2. **"WCET" framing invites real-time-systems reviewers.** DICTAT [restuccia2021timepredictable] and the
   RTSS/RTAS line on DPU predictability (topic t1) will be compared against V3. Interference from PS/DDR
   traffic on the KV260 is outside a PL-only cycle bound.
3. **SparsePixels and S2TA can be cited as "already data-independent".** V3 should name the difference
   explicitly: lossless skipping that gives a speedup *and* has a bound, rather than lossy constant time.
4. **Weak speedup.** Coarse all-zero K-step skipping after ReLU may give small gains at array-width
   granularity. Column combining and Cnvlutin-style fine skipping get more speedup but are less predictable.
   Measure the real gain on ResNet-20/MobileNet CIFAR-10 before claiming it.
5. **Recent arXiv-only peers** (Sparse by Command at MICRO 2026, SparHiXcel-v2, SparsePixels v4 of Aug 2026)
   may change. Re-check before submission.

## (e) UNVERIFIED and secondary items (not supporting references)

- The STONNE vs MAERI "about 15% difference in total cycles" comes from a search summary only. The paper was
  not read, so the number is verified=no. The DOI itself is verified.
- The claims for SparTen greedy balancing, GoSPA speedups, Sparse-PE speedups, ESE and BBS load balancing,
  HighLight, VEGETA and DSTC come from search snippets or secondary sources. Their DOIs are verified; their
  numbers must not be cited.
- I knew of "S3DNN (RTAS 2018) bounds WCRT for GPU DNNs" and the "ART" real-time accelerator framework only
  from a search snippet. Neither was looked up or verified, and both are outside t3.
- Véstias et al. 2019: full text blocked (HTTP 403), abstract only.
- Bit-Tactical: the numbers and quotes come from the arXiv precursor 1803.03688, not the ASPLOS'19 text.
