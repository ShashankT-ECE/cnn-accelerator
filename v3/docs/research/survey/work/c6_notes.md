# Category 6 notes: INT8 quantization, DSP packing, depthwise / residual hardware

Date: 2026-10-07. Output: `c6.csv` (9 rows, ranked). Mix of ranks 1-6: 2 quantization (jacob2018quantization,
nagel2021white), 2 DSP packing (sommer2022dsppacking, lee2019doublemac), 2 depthwise/residual (cho2021risa,
nguyen2022shortcutfusion). Spares 7-9: li2024revealing (DSP), wu2019mobilenets (depthwise), azizimazreah2019shortcut
(residual).

## Verification
- All 8 DOIs checked at api.crossref.org/works/<DOI>: title, first author and year match.
- nagel2021white has no DOI; arXiv API record 2106.08295 fetched (title, first author Nagel, 2021-06-15); peer_reviewed=no.
- Full text read (arXiv PDFs via pdftotext): 1712.05877, 2106.08295, 2203.11028, 2409.03508, 2106.08167 (arXiv v4 of
  the TCAS-I ShortcutFusion paper).
- Abstract only: cho2021risa (Crossref abstract), lee2019doublemac, wu2019mobilenets, azizimazreah2019shortcut
  (Semantic Scholar API abstracts; publisher full texts closed).

## Searches
- Crossref bibliographic queries: "ShortcutFusion ...", "Shortcut Mining ...", "Design and Optimization of Residual
  Neural Network Accelerators for Low-Power FPGAs", "Semi-Streaming Architecture ...".
- arXiv API title search: DSP-Packing, Revealing Untapped DSP, RiSA, Double MAC.
- Web search: "FPGA accelerator residual connection shortcut element-wise addition ResNet hardware support paper".
- Reused from refs.bib / LITERATURE.md / research/work: jacob2018quantization, sommer2022dsppacking, lee2019doublemac,
  li2024revealing, huang2019efficient, wu2019mobilenets, bai2018cnn, li2021dynamic, cho2021risa, selvam2021fuseconv.

## Considered and dropped
- Xilinx WP486 / WP487 / WP521: no DOI, not eligible (per brief).
- krishnamoorthi2018quantizing (arXiv 1806.08342, read in full): good INT8 white paper (per-channel weights, per-layer
  activations, within 2% of float post-training) but arXiv-only like Nagel; Nagel kept as it has the more detailed
  hardware/accumulator discussion. Usable replacement if needed.
- huang2019efficient (IEEE Access, PMSDS packing): DOI verified; dropped in favour of Sommer/Lee/Li (Sommer already
  discusses it). Usable replacement.
- bai2018cnn (TCAS-II, arXiv 1809.01536) and li2021dynamic (TCAS-I): DOIs verified; depthwise accelerators on FPGA but
  less close to a systolic-array depthwise mode than RiSA. Replacements.
- selvam2021fuseconv (DATE 2021): changes the network operator rather than the hardware; replacement only.
- minnella2023resnet (arXiv 2309.15631): residual-focused, but its peer-reviewed version is the NN2FPGA KV260 system
  paper (bosio2025nn2fpga, TCAD 2025), which belongs to category 1 (full ZU+ system).
- weng2021hardware ("Hardware-efficient Residual Networks for FPGAs", arXiv 2102.01351): removes skip connections by
  training; no peer-reviewed version found in this pass.
- Semi-Streaming Architecture (arXiv 2006.08759): has an ADD engine, but no Crossref match found; not verified.

## Problems
- wu2019mobilenets: earlier pass (t5) read a course-hosted full text; that URL did not download from this machine, so
  basis = abstract here.
- jacob2018quantization uses uint8 with zero-points; the "int8" framing in V3 is a variant (noted in disadvantages).
