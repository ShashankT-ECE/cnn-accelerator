# Shared brief for literature subagents (read fully before starting)

## Context (our project, so you can judge closeness)
V2 (done, paper-final): INT8 8x8 output-stationary CNN accelerator on AMD Kria KV260 (XCK26), LeNet-5 and
CIFAR-10, all weights in BRAM, 250 MHz. Thesis of V2: bit-exact inference AND cycle-exact agreement between a
Python cycle model, RTL simulation and the board (cycle counters on the KV260 equal the model on every job).
V3 (proposed, college major project): larger reconfigurable INT8 accelerator for ResNet-20 and a MobileNet-style
CIFAR-10 net on KV260.

Proposed thesis being tested for novelty:
"A timing-predictable CNN accelerator on FPGA (KV260): reconfigurable dataflow (split-K, depthwise, per-layer
mode selection) and sparsity skipping (static structured weight sparsity; activation skipping with a provable
dense upper bound) give speedups while keeping exact, compile-time latency guarantees (model = RTL = board,
cycle-exact), for real-time/WCET-sensitive embedded use."

Already covered by an earlier review (do NOT just repeat; extend, and you may reuse keys): see
/home/shashankt/gos-v3/v3/docs/LITERATURE.md and /home/shashankt/gos-v3/v3/docs/refs.bib
(DPU, FILM-QNN, Eyeriss v2, MAERI, FlexFlow, SIGMA, Timeloop, MAESTRO, SCALE-Sim, Cnvlutin, SCNN, Cambricon-X/S,
EIE, Zhu2020 structured, SPEC N:16, column combining, DSP packing papers, etc.).

## Hard rules
1. NEVER invent a citation. Every reference must be verified: DOI resolves via
   `curl -s https://api.crossref.org/works/<DOI>` (compare title, first author, year) or an arXiv / official
   publisher / vendor URL that you actually fetched. If you cannot verify it, it goes into the "UNVERIFIED"
   section only, with verified=no, never as a supporting reference.
2. Numbers only from the paper itself (full text: arXiv, open-access PDF, publisher HTML), with table/figure/page.
   If you only saw the abstract or a secondary source, say so (verified=no for that number, note "abstract only"
   or "secondary: <where>"). Never convert units; keep the paper's notation.
3. Absence claims ("X does not do Y") must name what was read (abstract only / sections read). Use
   "not found in <sections read>" rather than "does not".
4. Do not write anything in the git repo. Write ONLY your output files in
   /tmp/claude-1001/-home-shashankt-cnn-accelerator/4e1fa6c7-6c53-4b35-b5f0-0c68041ac829/scratchpad/research/
   Do not commit, do not run Vivado. Do not send our repo content to any external service beyond search queries.
5. Search actively for counter-evidence: work that ALREADY does what our thesis claims. Being wrong toward
   "novel" is the worst outcome. Try many phrasings (e.g. "deterministic latency", "WCET", "time-predictable",
   "cycle-accurate", "worst-case execution time", "real-time", "predictable", "mixed-criticality", "certifiable",
   "safety-critical", "static schedule", "compile-time scheduling") and look at Google Scholar-style results,
   arXiv, IEEE, ACM, DBLP, Semantic Scholar API (https://api.semanticscholar.org/graph/v1/paper/search?query=...).
   Follow citations both ways for the closest hits.
6. Aim for quality over quantity: ~10-25 verified works per topic, with the 3-6 closest read in full text.

## Output files (prefix = your topic id t1..t6)
- `<tid>_findings.md`: synthesis for your topic: (a) what exists, (b) closest works to the thesis with
  "does / not found in sections read", (c) explicit verdict for your topic's part of the thesis
  (novel / partly done / already done) with reasons, (d) risks, (e) UNVERIFIED list.
- `<tid>_related.csv` with header exactly:
  key,title,authors,year,venue,doi,url,verified,verify_method,topic,closeness_1to5,what_it_does,not_found_in_sections_read,sections_read,notes
  (key = bibtex-style firstauthorYEARword lowercase; doi blank if none; verified yes/no; verify_method e.g.
  "crossref title+author+year match" or "arXiv abs page fetched"; closeness 5 = nearly our thesis)
- `<tid>_refs.bib`: BibTeX for every verified entry (build from Crossref metadata; include doi and/or url field).
