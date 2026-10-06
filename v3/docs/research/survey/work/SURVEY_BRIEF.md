# Brief for the 50-paper survey subagents (read fully before starting)

Project context (for judging relevance only): V3 is a timing-predictable INT8 CNN accelerator on the AMD Kria
KV260 (XCK26, Zynq UltraScale+). Latency is known exactly before running (Python cycle model = RTL = board,
cycle-exact). Planned: R x C output-stationary systolic array (16x16 expected), 250 MHz, ResNet-20 and a
MobileNet-style net on CIFAR-10, weights in URAM, split-K mode, per-layer mode selection by a compiler, structured
weight sparsity (static skip), depthwise mode, activation zero-step skipping with the dense schedule as the
worst-case bound. Baselines: AMD DPU and ONNX Runtime on the same board. Source of truth:
/home/shashankt/gos-v3/v3/docs/V3_PLAN.md.

## Your job
Find and annotate the papers for ONE category (given in your prompt). Deliver `target + 3` candidates ranked best
first (the 3 extra are spares used if another category already has the paper or a check fails).

## Reuse first
Earlier, partly verified work: `/home/shashankt/gos-v3/v3/docs/research/work/` (`merged_related.csv` = 131 works,
`verify_report.csv` = Crossref re-check, `t1..t5_findings.md`, `t*_refs.bib`), plus
`/home/shashankt/gos-v3/v3/docs/refs.bib` and `/home/shashankt/gos-v3/v3/docs/LITERATURE.md`. Reuse those papers where
they fit, but you must still read the abstract or full text yourself for the annotation columns.

## Hard rules
1. Never invent a paper, author, venue, page or number. Every paper must have a DOI that resolves at Crossref
   (`curl -s https://api.crossref.org/works/<DOI>`: compare title, first author, year) or, only if no DOI exists,
   an official arXiv/IEEE/ACM page you actually fetched. If you cannot verify it, drop it and pick another.
2. Prefer peer-reviewed journal/conference papers from 2016-2026 plus key classics. Use arXiv-only papers only if
   nothing peer-reviewed covers the point (set peer_reviewed=no).
3. `work_done`, `advantages`, `disadvantages` must come from what you read (abstract or full text). Each is 1-2
   sentences (max ~300 characters), neutral wording. Disadvantages may be limits the authors state, or clear
   scope limits visible in the text (device, network, dataset, precision, simulation-only evaluation). Never claim
   an absence you did not check. Numbers only if they appear in what you read.
4. `basis` = `full text` only if you actually fetched and read the paper body (arXiv, open-access PDF, publisher
   HTML); otherwise `abstract`. `read_locator` says what you read (e.g. "abstract (Crossref)", "abstract (IEEE
   Xplore page)", "full text: Sec. III-V, Table 4"). `read_source_url` is the page/PDF you read.
5. `relevance` = one line on how it relates to V3 (e.g. baseline, closest prior work, contrast, method we reuse).
6. Privacy: send only search queries, titles, DOIs and URLs to web tools. Never paste repository text or file
   contents into a web request or external service.
7. Write ONLY your output file. Do not commit, do not edit other files, do not run Vivado.

## Output: `/home/shashankt/gos-v3/v3/docs/research/survey/work/c<N>.csv` (UTF-8, csv module quoting)
Columns, exactly in this order:
`category,rank,key,doi,arxiv_id,title_seen,first_author_seen,year_seen,peer_reviewed,work_done,advantages,disadvantages,relevance,basis,read_locator,read_source_url,notes`
- `key`: lowercase firstauthorYEARfirstword (reuse the existing key if the paper is already in the files above).
- `doi`: bare DOI (no https://doi.org/), empty only for arXiv-only papers (then fill `arxiv_id`).
- Bibliographic fields (authors, venue, volume, pages) are NOT needed: they are filled from Crossref by script.
Also write `c<N>_notes.md`: search queries used, papers considered and dropped (with the reason), anything unverifiable.
Final message: number of rows, how many full-text vs abstract, and any problems (short).
