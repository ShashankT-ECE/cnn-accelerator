# t6 — Candidate venues for the V3 timing-predictable FPGA CNN accelerator

Fetch date for all sources: **2026-10-06**. Machine-readable version: `t6_venues.csv` (40 rows).

Labels used:
- **announced** means the date was read on an official page.
- **expected based on last year** means the date was not found. The previous edition's date and its source are given instead.
- **unknown** means neither was found.
- Acceptance rates come only from official pages or the aggregator csconfstats.xoveexu.com (which cites its sources per conference). The FPGA 2026 rate is UNVERIFIED (see the note).

## Timing constraints that drive the ranking

- V3 does not exist yet. It is a college major project. Realistic first results come no earlier than about Q1 2027. So deadlines in Oct–Dec 2026 can only take V2-derived material or a 2-page early result.
- V2 was submitted to ROCS 2026 (deadline 2026-10-09). Any venue that receives V2 content (ISORC, RTAS 2027, ARC) must respect dual-submission rules. Treat those as V3-only unless the ROCS outcome is known.
- Most PhD forums and the DAC Young Fellows program are **not open to undergraduates**:
  - DAC Young Fellows (2026 rules) takes MSc/PhD only.
  - DAC PhD Forum takes PhD students with at least one published paper.
  - The DATE PhD Forum takes PhD students who are near completion or recently graduated.
  - The FPL PhD forum takes PhD students.
- SRC@ICCAD explicitly allows undergraduates (up to 5 poster spots; ACM + SIGDA membership needed). Its 2026 deadline was 2026-08-16, so the 2027 deadline is expected around Aug 2027, which is outside the window.

## Ranked shortlist (for V3)

| Rank | Venue | Next deadline | Status | Format | Why |
|---|---|---|---|---|---|
| 1 | **FPL 2027** | ~late Mar 2027 (2026: abstract 03-20, paper 03-27) | expected | long 8 pp + ≤2 refs, short 5 pp + ≤1 ref; double-blind | FPGA community that values working board systems; 'AI/ML acceleration' is in scope; the timing matches V3 |
| 2 | **ESWEEK 2027 journal track** (EMSOFT for real-time, CASES for architecture) → IEEE TCAD | ~late Mar 2027 (2026: abstract 03-23, paper 03-30) | expected | 12 pp IEEE Trans. (14 after revision); double-blind | Journal publication plus presentation. EMSOFT is the natural home for "predictable accelerator". 2025 rates: EMSOFT 27.0%, CASES 28.2%, CODES+ISSS 27.2% |
| 3 | **ECRTS 2027** | ~late Feb 2027 (2026: 02-26) | expected | 20 pp LIPIcs, double-blind, open access | The CFP explicitly lists "time-predictable hardware architecture, GPU and accelerators, FPGA prototyping". 2026 rate 36.5% (23/63) |
| 4 | **DATE 2027 Late Breaking Results** | **2026-11-29** | announced | 2 pp + 1 p refs | The only near-term, low-cost published slot for a first V3 predictability result. Also the DATE University Fair on 2027-01-15 (1-p demo) |
| 5 | **FCCM 2027** (short paper) | ~mid Jan 2027 (2026: abstract 01-10, paper 01-17); site shows TBD | expected | short 4 pp / long 8 pp, double-blind | FPGA-architecture angle; deterministic latency is a differentiator there. Conference 2027-05-09..12, Provo UT |

Fallbacks:
- **RTAS 2027 Brief Presentations** (2027-02-11, announced).
- **FPT 2027** (~mid Jun 2027). The FPT journal track feeds ACM TRETS.
- **ESWEEK late-breaking → IEEE ESL** (~early Jun 2027).
- **DAC 2027 LBR** (~early Mar 2027, based on DAC 2026's 03-06).
- **RTCSA 2027** (~Mar 2027).
- Journals: **ACM TRETS**, **JSA** (Elsevier), **IEEE ESL** (all rolling).

## Notes per venue

### Real-time

- **RTAS 2027** (CPS-IoT Week). Paper deadline **2026-11-05** (firm). Notification 2027-01-28 (tentative). Conference 2027-05-17..20.
  - Format: 11 pp of technical content plus references, IEEEtran 10pt, double-anonymous. Links to a personal GitHub are forbidden.
  - Artifact evaluation is optional (deadline 2027-02-18).
  - The CFP names "FPGAs and specialized accelerators" and "real-time inference".
  - Fit is 5 on topic, but the timing is too early for V3.
  - Rate 25.2% (29/115, 2026, csconfstats; one secondary source says 27/115).
  - Sources: https://2027.rtas.org/ , https://2027.rtas.org/cfp , https://2027.rtas.org/submission
- **RTSS 2027.** No CFP yet.
  - RTSS 2026 had abstract 2026-05-21 and paper 2026-05-26, with staged review and double-blind.
  - The official RTSS 2026 page reports 350 submissions, 32 accepted and 18 conditionally accepted. The aggregator reports 48/350 = 13.7%.
  - The 2026 Brief Presentations (09-14) and RTSS@Work demos (09-30) are closed.
  - Very selective. It needs an analysis contribution, not only engineering.
  - Source: https://2026.rtss.org/
- **ECRTS 2027.** No 2027 CFP yet; the site still shows 2026.
  - 2026 dates: deadline 2026-02-26, notification 04-20, conference Jul 7–10 in Lund.
  - Format: LIPIcs, 20 pp of technical content excluding the bibliography, double-blind, optional artifact evaluation.
  - Co-located OSPERT/RTSOPS workshops and a WiP/pitches session are lower-bar options.
  - I did not find a page about the LITES journal link.
  - Sources: https://www.ecrts.org/call-for-papers/ , https://www.ecrts.org/submission-instructions/
- **ISORC 2027** (IIT Ropar, India, 2027-04-07..09). Deadline **2026-10-15**; notification 2026-12-01.
  - Format: IEEE 2-col. Regular 10 pp (+2 purchasable), short 6 pp, dissertation digest 2 pp, lightning talks.
  - The CFP lists "neural network inference WCET analysis" under AI/ML on Edge. It excludes ML papers that have no timing dimension.
  - Review type: not on the fetched page.
  - Sources: https://isorc-conference.org/important-dates/ , https://isorc-conference.org/call-for-papers/
- **RTCSA.** 2026 was in Qingdao, Aug 11–13.
  - The Ada-Belgium CFP mirror gives submission 2026-03-25, notification 05-04, full 10+2 pp, short 6+2 pp, **single-blind**. A secondary summary gave 03-06/03-13. The official site was not fetched, so the 2027 date is "~Mar 2027, expected".
  - Source: https://people.cs.kuleuven.be/~dirk.craeynest/ada-belgium/events/26/260811-rtcsa.html
- **WCET workshop.** The last edition I found is WCET 2024 (at ECRTS). There is no 2026/2027 CFP, so the deadline is unknown.
  - Source: https://www.ecrts.org/wcet-2024-cfp/
- **ESWEEK** (CASES / CODES+ISSS / EMSOFT).
  - 2026 dates: journal track abstract 2026-03-23, paper 03-30 (firm), first-round notification 05-22. Late-breaking track 06-05, notification 07-17. Artifacts 07-24.
  - The 2026 author-information page states:
    - Journal track goes to **IEEE TCAD** (12 pp, 14 after revision).
    - Late-breaking track goes to **IEEE ESL** (4 pp plus a 3-minute video).
    - Both are double-blind.
  - The CODES CFP text also says the journal is TCAD. Older editions used ACM TECS. The 2027 dates and venue are unknown.
  - Sources: https://esweek.org/ , https://esweek.org/author-information/ , http://esweek.org/codes_isss_cfp/

### FPGA

- **FPGA 2027** (ISFPGA, California, 2027-03-14..16). Abstract **2026-10-01 (passed, mandatory)**; paper 2026-10-08; notification 11-24.
  - Format: long 10 pp, short 6 pp, excluding references. Strict double-blind. Optional artifact evaluation with ACM badges. Top papers are invited to a TRETS special issue.
  - The 2026 rate of 24/94 (~26%) appears only in a search snippet attributed to the ACM proceedings front matter. The DL page returned 403, so it is **UNVERIFIED**.
  - Sources: https://www.isfpga.org/ , https://www.isfpga.org/call-for-papers/
- **FCCM 2027** (Provo, UT, 2027-05-09..12). All deadlines are TBD; the site launched 2026-09-29. There is a PhD Forum and artifact evaluation.
  - 2026 rules: abstract 01-10, paper 01-17, notification 03-16. Long 8 pp, short 4 pp, poster 1 p. Double-blind.
  - Sources: https://www.fccm.org/ , https://www.fccm.org/call-for-papers-2026/
- **FPL.** The 2027 edition has not been announced.
  - 2026 (Ghent, Sep 7–11): abstract 03-20, paper 03-27 per EasyChair (a secondary summary says extended to 03-27/04-03), notification 06-01.
  - Format: long 8 + ≤2 pp references, short 5 + ≤1 p, strict.
  - The double-blind status comes from a secondary search summary, because 2026.fpl.org is JS-rendered.
  - PhD forum deadline 2026-05-29 (secondary).
  - Sources: https://easychair.org/cfp/fpl2026 , https://2026.fpl.org/calls/call-for-papers/
- **FPT.** 2026 (Honolulu, Dec 7–10): conference track 2026-06-19, notification 08-21.
  - Format: 8 pp + references, short 4 pp total, double-blind. The journal track (TRETS, ≤32 pp) closed 2026-05-01.
  - Source: https://fpt2026.uark.edu/call-for-papers/
- **ARC 2027** (Berlin, 2027-03-31..04-02). Deadline **2026-10-25**; notification 12-07.
  - Format: LNCS, long 14 pp, short 8 pp. Anonymous review. Selected papers are invited to a TRETS special issue.
  - Source: https://amor.cms.hu-berlin.de/~hubnermi/index.html

### EDA / design automation

- **DATE 2027** (Dresden, 2027-03-22..24).
  - Main track: abstract 09-13 and paper 09-20, **passed**. 6 + 1 pp, double-blind.
  - **LBR: 2026-11-29** (2 + 1 pp).
  - PhD Forum: 2026-11-30 (PhD only).
  - Multi-Partner Projects: abstract 10-18, paper 10-25.
  - University Fair: 2027-01-15.
  - No acceptance rate is published on the site.
  - Source: https://www.date-conference.com/call-for-papers
- **DAC 2027** (starts 2027-07-11). Abstract **2026-11-11**, manuscript **2026-11-18** (5 PM PT). Accepted IDs 2027-02-22, notification 03-08.
  - Format: 6 + 1 pp, double-blind.
  - DAC itself states the rate is 20–25% for several years. The aggregator gives 22.6% (420/1,862) for 2025.
  - DAC 2026 LBR: 2026-03-06, 2 pp including references.
  - Sources: https://dac.com/2027/program/research-manuscript-submissions , https://dac.com/2026/late-breaking-results , https://dac.com/2026/dac-young-fellows , https://dac.com/2026/phd-forum
- **ICCAD.** 2026: abstract 04-07, paper 04-14, notification 07-11, Nov 8–12 in San Jose.
  - Format: 8 + 1 pp, double-blind.
  - 2025 rate 24.7% (266/1,078, aggregator).
  - SRC@ICCAD (undergraduates allowed) had its 2026 deadline on 08-16.
  - Sources: iccad-2026-cfp.pdf (https://ieee-cas.org/files/ieeecass/2026-02/iccad-2026-cfp.pdf) , https://iccad.com/2026/student-research-competition
- **ASP-DAC 2027 cycle.** Abstract 2026-07-11, PDF 07-18, notification 09-04. Format 6 + 1 pp, double-blind. ASP-DAC 2028 is expected around Jul 2027.
  - Source: https://www.aspdac.com/aspdac2027/cfp/

### Architecture / ML systems (stretch)

These venues are a weak fit. An 8×8 INT8 KV260 design does not contribute anything new to architecture.

- **ISCA 2027.** Expected ~Nov 10/17 2026, based on ISCA 2026 (abstract 2025-11-10, paper 2025-11-17). The official 2027 page could not be fetched. 2026 rate 18.9%.
- **MICRO 2027.** Expected ~Mar 31 / Apr 7 2027, based on MICRO 2026. 2025 rate 20.8%.
- **HPCA 2027.** Closed (07-24/07-31). 2026 rate 21.1%.
- **ASPLOS 2027.** Both cycles are closed. Format 11 pp. 2026 rate 14.5%.
- **MLSys 2027.** Paper due **2026-10-30**, 10 pp, double-blind.
- **tinyML / EDGE AI symposium.** No current CFP found.
- Sources:
  - https://www.sigarch.org/call-contributions/isca-2026/
  - https://www.sigarch.org/call-contributions/micro-2026/
  - https://www.sigarch.org/call-contributions/hpca-2027/
  - https://www.asplos-conference.org/asplos2027/cfp/
  - https://mlsys.org/Conferences/2027/CallForResearchPapers
  - rates from https://csconfstats.xoveexu.com/conferences/<name>/

### India / student-accessible

- **VLSID 2027** (Bengaluru, Jan 4–6, 2027). The regular deadline has passed (08-07).
  - User Design Track deadline is today, 2026-10-06. The fellowship deadline is 2026-10-15.
  - Format: 6 pp, double-blind. AI-use disclosure is mandatory.
  - Source: https://vlsid.org/call-for-papers/
- **ISVLSI.** 2026 (Kolkata): 6 pp, deadline 03-15. ISVLSI 2027 is expected ~Mar 2027.
  - Source: https://ieee-isvlsi.github.io/ISVLSI_2026_Website/ieee-isvlsi/index.html
- **IEEE iSES.** 2026 (NIT Goa, Dec 15–17): hard deadline 08-31. The 2027 edition is unknown.
  - Source: https://isesconf.github.io/iSES_2026_Website/index.html

### Journals (rolling)

- **ACM TRETS.** ≤32 pp. This comes from a search summary of the ACM author guidelines; the page returned 403.
- **IEEE ESL.** 4 pp including references (max 8 pp at $125/p). Single-anonymous. First decision targeted within about 1 month.
  - Source: https://ieee-ceda.org/publication/esl/esl-author-instructions
- **IEEE TCAD.** 14 pp. Single-anonymous.
  - Source: https://ieee-ceda.org/publications/tcad/tcad-paper-submissions
- **IEEE CAL.** 4 pp including references.
  - Source: https://www.computer.org/digital-library/journals/ca/cfp-ieee-computer-architecture-letters
- **IEEE TCAS-II.** 5 pp (4.5 + 0.5). Single-blind.
  - Source: https://ieee-cas.org/publication/TCAS-II/guidelines-author
- **IEEE Access.** No hard limit; under 20 pp recommended. Open access with APC.
  - Source: https://ieeeaccess.ieee.org/authors/submission-guidelines/
- **ACM TECS** (normally ≤25 pp), **JSA** (single-anonymized) and **IEEE TVLSI** (8–14 pp, $200/p above 9 pp): details come from search summaries only. The official pages returned 403 or 418.
- **Real-Time Systems** (Springer), **IEEE TC**, **TCAS-I** and **Microprocessors and Microsystems**: not verified in this pass.

## Fit dimensions

- **Real-time angle.** The core claim is a compile-time exact latency, with model = RTL = board.
  - Venues: ECRTS, RTAS, EMSOFT, RTCSA, ISORC, RTSS.
  - Reviewers there will ask how this compares with WCET analysis and with a predictable memory/interconnect, and what the bounds are under dynamic sparsity.
- **FPGA-architecture angle.** The core claims are reconfigurable dataflow and sparsity speedups on KV260.
  - Venues: FPL, FCCM, FPT, ARC, TRETS.
  - Reviewers will compare against FILM-QNN, the DPU and similar designs on throughput and resources.
- **EDA/methodology angle.** The core claim is the cycle-exact model as a verification and timing oracle.
  - Venues: DATE (Track E/D), DAC, ICCAD, TCAD.

## UNVERIFIED / caveats

- FPGA 2026 acceptance 24/94 (~26%): taken from a search snippet only.
- The double-blind status of FPL 2026 and the extended deadlines come from a secondary search summary.
- RTCSA 2026 deadline: sources conflict (03-25 on the Ada-Belgium mirror vs 03-06/03-13 in a secondary summary).
- ISCA 2027 dates (Jun 5–11, Atlanta) and MLSys 2027 location (Bellevue): from secondary trackers or summaries only.
- A secondary search summary claimed "ESWEEK 2027 = Mar 22–24, Dresden". That is almost certainly a conflation with DATE 2027 and was ignored.
- Page limits for TRETS, TECS, TVLSI and JSA: from search summaries; the official pages could not be fetched.
