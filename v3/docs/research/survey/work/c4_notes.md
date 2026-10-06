# c4 notes: timing-predictable / real-time / WCET-aware DNN inference and hardware

Date: 2026-10-07. Output: `c4.csv` (10 rows, ranked; target 7 + 3 spares).

## Reuse
Started from `../../work/t1_findings.md` and `t1_related.csv` (keys reused). Each annotation was re-read for this
survey (PDFs or abstracts fetched again; see `read_locator`).

## Searches / sources
- Web search: "DERCA RTSS 2025 DNN accelerator real-time"; "PREMA predictive multi-task scheduling preemptible neural
  processing unit HPCA 2020"; "DERCA deterministic cycle-level accelerator arxiv pdf"; "AXI-REALM predictable bandwidth
  AXI4 interconnect Benz DOI".
- Crossref bibliographic queries: DERCA, STT-VTA, ART, AXI HyperConnect (and PREMA, where the search query failed; the DOI was
  then checked directly at /works/<DOI>).
- OpenAlex (abstract_inverted_index) for abstracts not reachable on IEEE Xplore: STT-VTA, Ezekiel 2023, DERCA,
  AXI HyperConnect.
- Full texts: RTAS26 and RTSS21 (author pages, Sant'Anna), ART accepted manuscript (NSF PAR), Vicuna and ACETONE
  (Dagstuhl DROPS), Kirschner WiP and PREMA (arXiv).

## Verification
- All 10 DOIs: title, first author and year compared with Crossref, except the two below.
- **DataCite DOIs (Crossref API returns 404):** `10.4230/LIPIcs.ECRTS.2021.1` (Vicuna) and `10.4230/LIPIcs.ECRTS.2022.3`
  (ACETONE). Both resolve at https://doi.org/ (HTTP 302 to drops.dagstuhl.de), and the DataCite API title, first author
  and year match. The bibliography script must take their fields from DataCite, not Crossref.

## Basis
Full text: 7 (Aromolo, Restuccia, ART, Vicuna, Kirschner WiP, ACETONE, PREMA). Abstract only: 3.
- DERCA: the full text is under NSF PAR embargo until 2026-12-02, and no arXiv version was found. **Read it before claiming
  novelty against "cycle-level determinism".**
- STT-VTA: IEEE Xplore returned no content to the fetcher; the abstract was taken from OpenAlex.
- AXI HyperConnect: closed access; the author-page URL tried returned no PDF.

## Considered, not ranked (spares beyond the 10, or dropped)
- `ezekiel2023optimization` (10.1109/DSD60849.2023.00030, abstract via OpenAlex): the time-triggered VTA load module.
  Covered by its successor bebawy2025superscalar; first spare if another category takes STT-VTA.
- `kirschner2026multivic` (10.1007/978-3-032-19099-4_14): FPGA measurements for the Kirschner line. Not re-read this
  round; spare.
- `garofalo2025reliable` (TCAS-II 2025): a time-predictable heterogeneous ASIC SoC. ASIC with a mixed-criticality focus,
  so it ranks below the FPGA/DNN-specific works. Spare.
- AXI-REALM (Benz et al., arXiv 2311.09662 / 2501.10161): an interconnect regulation extension. It is not DNN-specific,
  the peer-reviewed DOI was not checked, and AXI HyperConnect covers the FPGA-SoC interconnect point instead.
- `silva2023extending`, `pearce2021designing`, `cerioli2025timepredictable`: software WCET for NN code. ACETONE
  represents this line.
- `abts2020think` / `abts2022software` (Groq TSP, deterministic ASIC): left for the deterministic-architecture or
  ASIC category. Not cat 4 hardware on FPGA.
- `jiang2019superlinear`, `lubeck2025automatic`, `toupas2023harflow3d`, etc.: performance models, so they belong to
  cat 5 (owned elsewhere).
- `hussein2026fpga` (SLR), `buttazzo2025toward`, `lee2025timing`: survey and position papers. Useful for statements
  about the gap, not as primary works.
- `faure2025opensource`: the certification-minded VTA compiler. Simulator-only per its abstract, with no timing
  guarantee claim.

## Unverifiable / open
- How DERCA derives WCET, and whether it compares predicted cycles with board cycles, is unknown until the full text
  is released.
- STT-VTA: the abstract describes simulator-based evaluation. Whether any board results exist was not checked.
