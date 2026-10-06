# Category 5 notes: performance models / DSE validated against hardware

## Method
- Reused the candidate list from `work/t4_findings.md` and `t4_related.csv`. I re-read each paper myself: the full text
  was downloaded (arXiv, author/open-access PDF), converted with `pdftotext`, and its validation section found by grep and read.
- Every DOI was checked through the Crossref API (title, first author, year). All 9 matched. Crossref titles are
  truncated for MAESTRO ("...DNN Dataflow"), DNNExplorer ("DNNExplorer") and AutoDNNchip ("AutoDNNchip").
- `title_seen` is the title in the text I read. For ZigZag that is the arXiv v3 title, which differs from the TC title
  (noted in the row).

## Search queries used
- "A Systematic Methodology for Characterizing Scalability of DNN Accelerators using SCALE-Sim" pdf
- "Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks" Zhang Li Sun Guan Xiao Cong pdf
- Ma Cao Vrudhula Seo "Performance Modeling for CNN Inference Accelerators on FPGA" pdf

## Sources read (full text, all 9)
| rank | key | source |
|---|---|---|
| 1 | parashar2019timeloop | accelergy.mit.edu/timeloop.pdf |
| 2 | kwon2019understanding | arXiv 1805.02566v6 |
| 3 | samajdar2020systematic | horizon-lab.org/pubs/ispass20.pdf |
| 4 | ma2020performance | par.nsf.gov/servlets/purl/10165009 (IEEE published version) |
| 5 | zhang2020dnnexplorer | arXiv 2008.12745v2 |
| 6 | zhang2015optimizing | cl.cam.ac.uk course reading-list copy |
| 7 | xu2020autodnnchip | arXiv 2001.03535v4 |
| 8 | mei2021zigzag | arXiv 2007.11360v3 |
| 9 | wu2019accelergy | accelergy.mit.edu/paper.pdf |

## Corrections to the earlier t4 work
- The 1.15 % (pipeline model, ZC706/KU115) and 2.17 % (generic structure, 36 CONV layers, VU9P) board errors appear
  in **DNNExplorer's** own text (Sec. 6.1–6.2). t4 had attributed them to DNNBuilder from a secondary snippet. Do not
  cite them for DNNBuilder unless the DNNBuilder text is read.
- Ma et al. TCAD'20 "within 3 %" vs on-board (Arria 10, Stratix 10) is now confirmed from the full text (Sec. VIII-C).
- AutoDNNchip per-platform latency errors (avg 4.85 / 3.73 / 6.57 %, max 9.75 %) are confirmed from the full text.
- Zhang FPGA'15: in the text I found no explicit model-vs-board error figure (I grepped for estimat/measur/error/actual).
  The paper uses the roofline model for DSE and reports measured board timings per layer.

## Considered and not included (spares or out of scope)
- SCALE-Sim v1 (arXiv 1811.02883): not peer-reviewed. It has the same validation as the ISPASS 2020 paper, so the
  ISPASS paper is used.
- SCALE-Sim v3 (ISPASS 2025, 10.1109/ISPASS64960.2025.00026): t4 verified it, but the Crossref API returned no JSON
  in this pass. It is a good spare (N:M sparsity vs Vegeta RTL ≤ 5 %, per t4) once it is re-verified and re-read.
- DNNBuilder (ICCAD'18): I could not read the text (closed access), and its error figures were secondary. Dropped.
- DNN-Chip Predictor (ICASSP'20), STONNE, Sparseloop, DeFiNES, Stream, ONNXim, HARFLOW3D, LLMCompass: valid
  candidates from t4, but not re-read in this pass. Sparseloop may belong to the sparsity category.
- I excluded WCET/timing-guarantee works (cat 4) and dataflow-flexible accelerators (cat 2: e.g. HybridDNN, Herald,
  FEATHER).

## Unverifiable / caveats
- ZigZag was read on arXiv v3. The published TC version may differ in its numbers.
- Zhang FPGA'15 was read from a third-party hosted copy, not the ACM DL.
