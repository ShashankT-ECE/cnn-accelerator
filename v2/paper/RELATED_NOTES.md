# Related-work notes (ROCS 2026 short paper)

Bib file: `v2/paper/tex/refs.bib` (IEEEtran). Gathered 2026-10-02 with Claude Code
(academic-research-skills deep-research, lit-review scope only; no full pipeline, no
cross-model features).

**Verification method.** Every DOI was resolved through the Crossref REST API
(`api.crossref.org/works/<doi>`). Title, full author list, venue, year, volume/issue
and pages in `refs.bib` were copied from that record, and the abstract was read from
Crossref or Semantic Scholar where one exists. One candidate was rejected: the DOI first
recalled for Ma et al. (10.1109/TVLSI.2017.2688340) resolves to a different paper (Tu et al.),
so the entry uses the DOI that Crossref returns for the title (10.1109/TVLSI.2018.2815603).

**Writing rule 6 (PAPER_PLAN.md): "Every citation must be a paper someone on the team has
actually read."** None of these has been read in full by a team member yet. The summaries
below come from abstracts and the general standing of each paper, not from a full reading.
Before submission, a team member must read each paper they keep (at least the parts the
sentence in Sec. II relies on) and tick the box, or the citation goes.

| # | Key | DOI | Verified | Read by team |
|---|---|---|---|---|
| 1 | sze2017efficient | 10.1109/JPROC.2017.2761740 | Crossref ✓ | ☐ |
| 2 | jacob2018quantization | 10.1109/CVPR.2018.00286 | Crossref ✓ | ☐ |
| 3 | zhang2015optimizing | 10.1145/2684746.2689060 | Crossref ✓ (no abstract in Crossref/S2) | ☐ |
| 4 | umuroglu2017finn | 10.1145/3020078.3021744 | Crossref ✓ | ☐ |
| 5 | guo2018angeleye | 10.1109/TCAD.2017.2705069 | Crossref ✓ (no abstract in Crossref/S2) | ☐ |
| 6 | ma2018optimizing | 10.1109/TVLSI.2018.2815603 | Crossref ✓ | ☐ |
| 7 | parashar2019timeloop | 10.1109/ISPASS.2019.00042 | Crossref ✓ | ☐ |
| 8 | kwon2019maestro | 10.1145/3352460.3358252 | Crossref ✓ | ☐ |
| 9 | samajdar2020scalesim | 10.1109/ISPASS48437.2020.00016 | Crossref ✓ | ☐ |
| 10 | genc2021gemmini | 10.1109/DAC18074.2021.9586216 | Crossref ✓ | ☐ |
| 11 | costa2026kv260 | 10.1007/s11554-026-01889-x | Crossref ✓ | ☐ |
| 12 | amd_pg338 | — (vendor manual) | **UNVERIFIED revision**: the portal page is script-rendered, so the PG338 revision matching the DPUCZDX8G in the pynq-dpu 2.5 overlay was not confirmed | ☐ |

## Summaries and relation to this work

**(a) FPGA / edge CNN accelerators and dataflow**

1. **Sze et al. 2017, Proc. IEEE.** A tutorial and survey of efficient DNN processing, covering the dataflow taxonomy (weight-, output-, row-stationary) and benchmarking metrics. *Relation:* the standard reference for the term "output-stationary" and for its utilization trade-offs on small feature maps (our Sec. IV).
2. **Jacob et al. 2018, CVPR.** An integer-only quantization scheme: INT8 weights and activations, integer accumulation, and requantization by a fixed-point multiplier and shift. *Relation:* our numeric contract follows this family. We add an exhaustive check that the integer (m, s) pipeline reproduces the float64 reference (Sec. V).
3. **Zhang et al. 2015, FPGA.** Roofline-model-based design-space exploration of loop tiling and unrolling for an FPGA CNN accelerator, with a board implementation. *Relation:* a classic analytical model that guided an FPGA design. It bounds throughput (roofline) rather than predicting cycles exactly.
4. **Umuroglu et al. 2017 (FINN), FPGA.** A framework for streaming binarized-NN accelerators on FPGAs, sized per layer to a throughput target. It reports MNIST and CIFAR-10 throughput and latency on a ZC706. *Relation:* a small-network FPGA work on the same benchmarks, but with binarized numerics and a dataflow architecture. We use INT8 and one shared 8×8 array.
5. **Guo et al. 2018 (Angel-Eye), TCAD.** A complete flow (quantization, compiler, instruction-driven accelerator) for mapping CNNs onto embedded Zynq FPGAs. *Relation:* the embedded instruction-based approach, close in organization to the vendor DPU. We cite it as the programmable-overlay alternative to our fixed-function design. (Whether Angel-Eye is the direct ancestor of the DPU is NOT stated in the paper text; we don't claim it.)
6. **Ma et al. 2018, TVLSI.** A quantitative analysis of convolution loop optimizations (unrolling, tiling, interchange) and a dataflow chosen to minimize data movement, demonstrated on Intel FPGAs with end-to-end CNNs. *Relation:* an analytical-model-driven FPGA design validated by implementation, the closest in spirit to our "model first" approach. Its model targets design choices and throughput, not exact per-layer cycle identity.

**(b) Analytical and simulation performance models**

7. **Parashar et al. 2019 (Timeloop), ISPASS.** A mapper plus model that projects the performance and energy of a DNN accelerator from a unified architecture description. *Relation:* a general model for design-space exploration. Ours is a closed-form model for one fixed design, held to exact equality with RTL and the board.
8. **Kwon et al. 2019 (MAESTRO), MICRO.** Data-centric directives and an analytical cost model for dataflow reuse, runtime and energy, used for large design-space sweeps. *Relation:* same as Timeloop. The general models trade exactness for breadth; we take the opposite trade-off.
9. **Samajdar et al. 2020 (SCALE-Sim), ISPASS.** A cycle-accurate simulator for systolic-array DNN inference, plus an analytical model for scale-up vs scale-out. *Relation:* a cycle-level model of systolic arrays. Our array has no skew (broadcast operands), so a closed form suffices, and we check it on silicon.
10. **Genc et al. 2021 (Gemmini), DAC.** A full-stack accelerator generator that integrates accelerators into SoCs to capture system effects (OS, contention, software stack). *Relation:* it motivates measuring on a real system rather than in isolation. Our end-to-end vs compute split (Python host vs PL counter) shows the same effect on the KV260.

**(c) AMD DPU on Zynq UltraScale+ / Kria**

11. **Costa and Brito 2026, J. Real-Time Image Processing.** Measures the throughput gain from multithreaded Vitis AI runners driving the DPU on the KV260 for three ImageNet classifiers. *Relation:* the same board and vendor IP as our DPU baseline. Their focus is throughput scaling; ours is single-image latency, determinism and energy.
12. **AMD PG338 (DPUCZDX8G product guide).** The vendor documentation of the DPU IP (B4096 configuration in the pynq-dpu 2.5 overlay: 8 × 16 × 16 = 2,048 MAC lanes at 300 MHz, per `dpu_overlay_params.csv`). *Relation:* the reference for the DPU baseline's architecture. **No DOI, revision unverified.**

## Gap statement (as used in Sec. II, hedged)

The general models (7–9) are built for exploration across designs, and FPGA design papers
(3, 6) use models to choose a design. In the works listed here, we did not find a per-layer,
exact (zero-error) agreement between an analytical model, RTL simulation and board
measurement. The paper therefore says "few works report", not "no work".
