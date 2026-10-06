# Category 2 notes: reconfigurable / flexible-dataflow accelerators

11 rows (target 8 + 3 spares), ranked by closeness to V3 (per-layer modes on a systolic array, sub-array
fission, split-K). All DOIs resolved at Crossref; title, first author and year match the row for each.

## Sources and searches
- Reused: `research/work/t2_findings.md`, `t2_related.csv` (Planaria, FEATHER, Herald, SARA, Gemmini, ArrayFlex
  DOIs and PDF locations), `docs/refs.bib` (Eyeriss v2, MAERI, FlexFlow, SIGMA keys and DOIs).
- Crossref `works/<DOI>` for all 11; Crossref bibliographic search "Flex-TPU Flexible TPU Runtime Reconfigurable
  Dataflow" (no peer-reviewed version found).
- Full text fetched (arXiv PDF or author PDF, pdftotext): Eyeriss v2 (1807.07928), FEATHER (2405.13170), Herald
  (1909.07437), Gemmini (1911.09925), SARA (2101.04799), Stream-K (2301.03598), ArrayFlex (2211.12600), Planaria
  (UCSD author PDF), MAERI (Georgia Tech Synergy lab PDF).
- Abstract only: SIGMA (Semantic Scholar API abstract; OpenAlex says closed, author PDF URLs tried returned 404),
  FlexFlow (OpenAlex abstract; Semantic Scholar abstract elided; no open full text).

## Considered and dropped / not included
- Flex-TPU (arXiv 2407.08700): arXiv only, no peer-reviewed version found in Crossref; SARA and Gemmini cover
  per-layer OS/WS/IS switching with peer review.
- FlexSA (arXiv 2004.13027): split-K-like K partitioning on sub-systolic arrays, but arXiv only and training-
  focused; Stream-K (PPoPP 2023) used as the peer-reviewed split-K reference instead.
- HybridDNN (DAC 2020, 10.1109/dac18072.2020.9218684, Crossref verified): FPGA per-layer Spatial/Winograd mode;
  not annotated, closer to FPGA design-flow / Winograd than to dataflow flexibility. Possible extra spare.
- DyNNamic (IEEE TC 2023), Morph (MICRO 2018), Flexagon (ASPLOS 2023, sparse SpMSpM -> cat 3), dual-core OPU
  (FCCM 2021, DW core -> cat 6), RiSA/FuSeConv (DW -> cat 6), MAESTRO/Timeloop/ZigZag/CoSA (cat 5), DERCA/ART/FILCO
  (real-time -> cat 4): out of scope or owned elsewhere.
- Light-OPU, MRSA (2026), SqueezeJet-2: not read in prior work; not checked here.

## Caveats
- SIGMA has a sparsity component; kept here for its flexible distribution/reduction interconnect as assigned. If
  cat 3 also takes it, drop from one list.
- Gemmini (rank 11) may overlap with cat 7 (systolic generators/classics).
- SARA: arXiv title differs from the DAC 2022 title ("Learning Flexible GEMM Accelerator Configuration and
  Mapping-space using ML"); content matches, Crossref DOI/title used.
- Planaria states cycle counts are verified against Verilog; no agreement figure was found in the sections read.
- FlexFlow: evaluation technology/method not checked (abstract only).
