# Category 3 notes: sparsity-skipping accelerators

Output: `c3.csv`, 11 rows ranked (target 8 + 3 spares). All 11 are peer-reviewed. All DOIs were checked at Crossref (title, first author and year match). Crossref stores the short titles "SCNN", "SparTen" and "Bit-Tactical", and gives Cambricon-X in lower case. These are the same papers.

## Balance
- Structured or static weight sparsity: S2TA, Zhu 2020, SPEC (N:M), HPIPE, Cambricon-X, Bit-Tactical (static schedule).
- Activation zero skipping (value or bit): S2TA (DBB activations), Cnvlutin, SCNN, Bit-Tactical (zero bits), SparTen, NullHop, EIE.
- FPGA designs: Zhu 2020 (ZCU102), SPEC (ZCU102/VCU118), HPIPE (Stratix 10), NullHop (Zynq 7100). SparTen also reports an FPGA implementation (abstract).

## Sources and searches
- Reused the candidates and keys from `research/work/t3_related.csv`, `t3_findings.md` and `v3/docs/refs.bib`.
- Crossref works API was used for every DOI.
- Abstracts came from the arXiv export API (2107.07983, 2001.01955, 1708.04485, 2007.10451, 1602.01528, 1706.01406), from OpenAlex for Cnvlutin and Cambricon-X, from Semantic Scholar for SparTen and Bit-Tactical, and from Crossref for SPEC.
- Full text was fetched and read in part (sections are listed in `read_locator`) for S2TA, Zhu 2020 (arXiv preprint), SPEC (Europe PMC PMC10057003, because the MDPI site returned Access Denied), HPIPE, SCNN and NullHop.
- Basis: 6 full text, 5 abstract.

## Considered and dropped
- Systolic Sparse Tensor Slices (Taka et al., FPGA 2025, DOI 10.1145/3706628.3708867, verified, arXiv abstract read). This is a good N:M systolic fit, but it proposes in-fabric FPGA architecture blocks modelled in COFFE, not an accelerator on an existing device. It is held as the next spare if one is needed.
- Cambricon-S (MICRO 2018). Covered by Cambricon-X, and the abstract was not re-read.
- VEGETA, HighLight, DSTC, GoSPA, Sparse-PE. The DOIs were verified earlier (t3), but for this pass I read only secondary snippets. They are lower priority.
- Bit-Pragmatic (zero-bit). Bit-Tactical covers zero-bit skipping together with a static weight schedule.
- Sparseloop, SCALE-Sim v3, Krithivasan 2020, SparsePixels and Sparse by Command were left out. They are modelling, security or arXiv-only works, or they fit the timing-predictability category (cat 4) better.
- SIGMA, Eyeriss v2 and surveys are excluded by ownership.

## Caveats / unverifiable
- The Cnvlutin and Cambricon-X abstracts were rebuilt from the OpenAlex inverted index, because the IEEE Xplore pages returned nothing.
- Zhu 2020 annotations come from the arXiv preprint. The TVLSI version may add ResNet-50 (per LITERATURE.md, secondary).
- Bit-Tactical: only the ASPLOS abstract was used. The arXiv precursor 1803.03688 is a different text and none of its numbers are used.
- Semantic Scholar rate-limited (HTTP 429) one search. It was not needed after OpenAlex.
