# Interim research work (NOT the final deliverables)

Raw outputs from the 2026-10-06 novelty check (six topic subagents t1-t6; the t7 full-text "threat read" was stopped
before finishing because the usage limit was reached). Final NOVELTY.md / related.csv / refs.bib / prior_numbers.csv /
VENUES.md are still to be written from these files.

- t1..t5_{findings.md,related.csv,refs.bib}: per-topic synthesis, verdicts, closest works, UNVERIFIED lists.
- t5_prior_numbers.csv: ZU+ ResNet/MobileNet design points (45 rows, 41 verified per the subagent).
- t6_venues.{md,csv}: 40 venues with deadlines (many 2027 dates "expected from last year").
- merged_related.csv: 131 unique works (t1-t5 merged, DOI-deduplicated).
- verify_report.csv: independent re-check by verify_refs.py (Crossref / arXiv API). The 39 flagged rows still need a
  manual look. Most are false positives: Crossref stores a short title such as "SparTen", or the first-author check
  misparses the authors field. Two LIPIcs DOIs return 404 at Crossref because they are DataCite DOIs; check them via doi.org.

Open items before the final write-up:
1. Read in full text: DERCA (RTSS 2025, "cycle-level determinism" + WCET on FPGA), time-triggered VTA
   (Bebawy TCAD 2025, Ezekiel 2023), ART (GLSVLSI 2025), Kalagi 2026 (tile-level XSim = board), FPGN arXiv 2607.08427.
2. v3/docs/LITERATURE.md section (a) is wrong: NN2FPGA (Bosio et al., TCAD 2025) runs INT8 ResNet-20 on CIFAR-10 on the
   KV260 (see t5_findings.md).
