# V3 50-paper literature survey

Deliverables (in `v3/docs/research/`): `survey50.csv` (source data, one row per paper), `survey50.bib`
(IEEEtran), `Literature_Survey_V3.docx` (generated from the CSV). The scope follows `v3/docs/V3_PLAN.md`.

## Rebuild

```bash
cd v3/docs/research/survey
../../../../.venv/bin/python build_survey.py --offline   # verify + select -> survey50.csv/.bib, work/verify_log.csv
../../../../.venv/bin/python make_docx.py                # survey50.csv + intro.md -> Literature_Survey_V3.docx
```
`--offline` uses only the cached API responses in `work/cache/`. Without it, missing records are fetched
from doi.org, Crossref, DataCite and arXiv. `make_docx.py` needs `python-docx` (1.2.0).

## How the list was made (2026-10-07)

1. One subagent per category, all following `work/SURVEY_BRIEF.md`, wrote ranked candidates (target + 3 spares)
   with annotations to `work/c<N>.csv`, and their search log and dropped papers to `work/c<N>_notes.md`.
   Earlier novelty-check records in `../work/` were reused where they fit.
2. `build_survey.py` verifies every candidate:
   - the DOI must redirect at doi.org, and a Crossref record must exist (DataCite for LIPIcs DOIs, the arXiv API
     for arXiv-only papers);
   - the record's title and first author must match what the subagent read.

   All bibliographic fields are copied from that record. The year is the Crossref print year when one exists.
   Duplicates across categories stay in the category that ranked them higher. Within a category, peer-reviewed
   papers come before arXiv-only ones.
3. Manual review. Exclusions are listed with reasons in `work/exclusions.csv`, and the next spare replaces each one.
   Rows whose full text was only keyword-checked are labelled basis = abstract, with a note in `work/c1.csv`.

`work/verify_log.csv` records the result for every candidate.
