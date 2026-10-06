# Category 7 notes: surveys and foundational works

## Searches / sources used
- Reused keys from v3/docs/refs.bib: chen2016eyeriss, jouppi2017indatacenter.
- Crossref `api.crossref.org/works/<DOI>` for every DOI (title, first author, year all match):
  10.1109/JPROC.2017.2761740, 10.1109/MC.1982.1653825, 10.1109/ISCA.2016.40, 10.1145/3079856.3080246,
  10.1145/3289185 (found via Crossref query "A Survey of FPGA-based Neural Network Inference Accelerators Guo"),
  10.1145/3186332, 10.1109/JSSC.2016.2616357.
- Web searches: "Eyeriss spatial architecture energy-efficient dataflow ISCA 2016 pdf";
  "Eyeriss: An Energy-Efficient Reconfigurable Accelerator ... JSSC pdf dspace".
- Full texts: arXiv 1703.09039 (Sze), 1704.04760 (TPU), 1712.08934 (Guo), 1803.05900 (Venieris), 1806.01683
  (Abdelouahab); Eyeriss JSSC accepted manuscript (UW course page); Kung 1982 author-hosted scanned PDF
  (pp. 37-39 read as page images, no text layer).

## Considered and dropped
- Mittal, "A survey of FPGA-based accelerators for CNNs", Neural Comput. Appl. (10.1007/s00521-018-3761-1):
  Crossref verified but no abstract accessible (Crossref none, Springer redirects to login); dropped.
- Zhang et al. FPGA 2015 (10.1145/2684746.2689060) and DianNao ASPLOS 2014 (10.1145/2541940.2541967): Crossref
  verified, but abstracts elided in Crossref/Semantic Scholar and likely owned by other categories; not used.
- Eyeriss v2, MAERI, FlexFlow, SIGMA, Timeloop, MAESTRO: owned by other categories, excluded.

## Caveats
- chen2016eyeriss (ISCA): only the abstract on the MIT DSpace record page was read (scripted PDF download returned
  HTML; the Cornell ISCA@50 PDF is the authors' retrospective, not the paper). basis = abstract.
- kung1982why: no abstract exists in Crossref/Semantic Scholar/IEEE (Xplore page is script-rendered). basis = full
  text for pp. 37-39 only; pages 40-45 (detailed convolution designs) not read.
- guo2019survey: arXiv title says "Accelerator" (singular) and 2017 in its ACM reference line; Crossref final
  title/year is "...Accelerators", 2019. Sections read: abstract, headings, conclusion.
- abdelouahab2018accelerating is arXiv-only (peer_reviewed = no); rank 8 spare.
- chen2017eyeriss (rank 7) overlaps chen2016eyeriss; use only one of them if space is short.
