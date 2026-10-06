#!/usr/bin/env python3
"""Build the V3 50-paper literature survey from the per-category subagent files.

Inputs : survey/work/c1.csv .. c7.csv (ranked candidates with annotations, written by the
         category subagents; bibliographic fields are NOT taken from them).
Outputs: research/survey50.csv, research/survey50.bib, survey/work/verify_log.csv.

Every candidate is verified here:
  * DOI: https://doi.org/<doi> must redirect (30x), and the metadata record must exist at
    Crossref (DataCite as fallback for DataCite DOIs, e.g. LIPIcs / arXiv);
  * arXiv-only papers: the arXiv API record must exist.
Title, all authors, venue, volume, issue, pages and year are copied from that record
(year = published-print year if Crossref has one, else the issued year); "n/a" means the
record has no such field. The record's title and first author are compared with what the
subagent read; a clear mismatch rejects the candidate and the next spare is used.
Raw API responses are cached in survey/work/cache/ so the build is reproducible offline.

Usage: python build_survey.py [--offline]
"""
import argparse
import csv
import difflib
import html
import json
import re
import sys
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
CACHE = WORK / "cache"
OUT_DIR = HERE.parent
UA = {"User-Agent": "gos-v3-literature-survey/1.0 (python-requests)"}

CATEGORIES = {
    "1": ("FPGA CNN accelerators on Zynq UltraScale+ (KV260 / ZCU102 / ZCU104)", 10),
    "2": ("Reconfigurable and flexible dataflow accelerators", 8),
    "3": ("Sparsity-skipping accelerators (structured weight and activation sparsity)", 8),
    "4": ("Timing-predictable, real-time and WCET-aware DNN inference", 7),
    "5": ("Analytical performance models and DSE validated against hardware", 6),
    "6": ("INT8 quantization, DSP packing, depthwise/MobileNet and ResNet hardware support", 6),
    "7": ("Surveys and foundational works", 5),
}
TOTAL = 50

OUT_COLS = [
    "s_no", "category", "category_name", "key", "title", "authors", "venue", "volume", "issue",
    "pages", "year", "doi", "url", "pub_type", "peer_reviewed", "work_done", "advantages",
    "disadvantages", "relevance", "basis", "read_locator", "read_source_url", "metadata_source",
    "check_title_sim", "check_first_author", "selection_note",
]


# ---------------------------------------------------------------- helpers
def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def clean(s):
    """Strip JATS/HTML tags and collapse whitespace (Crossref titles sometimes carry markup)."""
    s = re.sub(r"<[^>]+>", "", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def na(v):
    v = clean(str(v)) if v not in (None, "") else ""
    return v if v else "n/a"


def cached_get(name, url, offline, accept_404=False):
    path = CACHE / name
    if path.exists():
        rec = json.loads(path.read_text())
        return rec["status"], rec["body"]
    if offline:
        raise SystemExit(f"offline and not cached: {url}")
    r = requests.get(url, headers=UA, timeout=30, allow_redirects=False)
    rec = {"url": url, "status": r.status_code, "body": r.text,
           "location": r.headers.get("Location", "")}
    if r.status_code in (200, 301, 302, 303, 307, 308) or (accept_404 and r.status_code == 404):
        path.write_text(json.dumps(rec, ensure_ascii=False))
    return r.status_code, r.text if not rec["location"] else rec["location"]


def safe(doi):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", doi)


# ---------------------------------------------------------------- metadata sources
def from_crossref(doi, offline):
    st, body = cached_get(f"crossref_{safe(doi)}.json",
                          "https://api.crossref.org/works/" + urllib.parse.quote(doi), offline,
                          accept_404=True)
    if st != 200:
        return None
    m = json.loads(body)["message"]
    title = clean(" ".join(m.get("title") or []))
    sub = clean(" ".join(m.get("subtitle") or []))
    if sub and norm(sub) not in norm(title):
        title = f"{title}: {sub}"
    authors = []
    for a in m.get("author") or []:
        if a.get("family"):
            authors.append((clean(a.get("given", "")), clean(a["family"])))
        elif a.get("name"):
            authors.append(("", clean(a["name"])))
    venue = clean(" ".join(m.get("container-title") or []))
    if not venue and m.get("event", {}).get("name"):
        venue = clean(m["event"]["name"])
    pp = m.get("published-print") or m.get("issued") or {}
    year = (pp.get("date-parts") or [[None]])[0][0]
    pages = m.get("page") or ""
    if not pages and m.get("article-number"):
        pages = "Art. no. " + m["article-number"]
    return {
        "title": title, "authors": authors, "venue": venue, "volume": m.get("volume", ""),
        "issue": m.get("issue", ""), "pages": pages, "year": year or "",
        "type": m.get("type", ""), "url": "https://doi.org/" + doi,
        "source": "Crossref (published-print year)" if m.get("published-print") else "Crossref (issued year)",
    }


def from_datacite(doi, offline):
    st, body = cached_get(f"datacite_{safe(doi)}.json",
                          "https://api.datacite.org/dois/" + urllib.parse.quote(doi), offline,
                          accept_404=True)
    if st != 200:
        return None
    a = json.loads(body)["data"]["attributes"]
    authors = []
    for c in a.get("creators") or []:
        if c.get("familyName"):
            authors.append((clean(c.get("givenName", "")), clean(c["familyName"])))
        else:
            authors.append(("", clean(c.get("name", ""))))
    cont = a.get("container") or {}
    pages = ""
    if cont.get("firstPage"):
        pages = cont["firstPage"] + ("-" + cont["lastPage"] if cont.get("lastPage") else "")
    return {
        "title": clean(a["titles"][0]["title"]), "authors": authors,
        "venue": clean(cont.get("title", "")), "volume": cont.get("volume", ""),
        "issue": cont.get("issue", ""), "pages": pages, "year": a.get("publicationYear", ""),
        "type": (a.get("types") or {}).get("resourceTypeGeneral", ""),
        "url": "https://doi.org/" + doi, "source": "DataCite",
    }


def from_arxiv(aid, offline):
    aid = re.sub(r"^(arxiv:)", "", aid.strip(), flags=re.I)
    st, body = cached_get(f"arxiv_{safe(aid)}.xml",
                          "https://export.arxiv.org/api/query?id_list=" + aid, offline)
    if st != 200:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = ET.fromstring(body).find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return None
    authors = []
    for au in e.findall("a:author/a:name", ns):
        parts = au.text.strip().split()
        authors.append((" ".join(parts[:-1]), parts[-1]))
    return {
        "title": clean(e.find("a:title", ns).text), "authors": authors,
        "venue": f"arXiv preprint arXiv:{re.sub(r'v[0-9]+$', '', aid)}", "volume": "", "issue": "",
        "pages": "", "year": e.find("a:published", ns).text[:4], "type": "posted-content",
        "url": f"https://arxiv.org/abs/{aid}", "source": "arXiv API",
    }


def doi_resolves(doi, offline):
    st, _ = cached_get(f"doiorg_{safe(doi)}.json", "https://doi.org/" + urllib.parse.quote(doi),
                       offline)
    return st in (301, 302, 303, 307, 308)


# ---------------------------------------------------------------- verification
def verify(row, offline):
    doi = row["doi"].strip().removeprefix("https://doi.org/").removeprefix("http://dx.doi.org/")
    row["doi"] = doi
    meta = None
    if doi:
        if not doi_resolves(doi, offline):
            return None, "doi.org does not resolve"
        meta = from_crossref(doi, offline) or from_datacite(doi, offline)
        if meta is None:
            return None, "no Crossref/DataCite record"
    elif row.get("arxiv_id", "").strip():
        meta = from_arxiv(row["arxiv_id"], offline)
        if meta is None:
            return None, "arXiv record not found"
        aid = re.sub(r"v[0-9]+$", "", row["arxiv_id"].strip())
        row["doi"] = f"10.48550/arXiv.{aid}"
    else:
        return None, "no DOI and no arXiv id"
    sim = difflib.SequenceMatcher(None, norm(meta["title"]), norm(row["title_seen"])).ratio()
    # Crossref sometimes stores only the short name ("SparTen"); accept a prefix match.
    if norm(row["title_seen"]).startswith(norm(meta["title"])) and len(norm(meta["title"])) >= 4:
        sim = max(sim, 0.99)
    fam = norm(meta["authors"][0][1]) if meta["authors"] else ""
    fa_ok = bool(fam) and fam in norm(row["first_author_seen"]) or (
        bool(norm(row["first_author_seen"])) and norm(row["first_author_seen"]).split()[-1] in fam)
    meta["sim"], meta["fa_ok"] = round(sim, 3), fa_ok
    if sim < 0.6 and not fa_ok:
        return None, f"metadata mismatch (title_sim={sim:.2f}, first author differs)"
    if sim < 0.6:
        return None, f"title mismatch (title_sim={sim:.2f})"
    if not meta["authors"]:
        return None, "record has no authors"
    return meta, "ok" if (sim >= 0.85 and fa_ok) else f"ok, flagged (title_sim={sim:.2f}, first_author_ok={fa_ok})"


# ---------------------------------------------------------------- outputs
def authors_str(authors):
    return ", ".join((f"{g} {f}" if g else f) for g, f in authors)


def pages_human(p):
    return p.replace("--", "-").replace("-", "–") if p else ""


BIB_ESC = str.maketrans({"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$"})


def bib_entry(r, meta):
    t = meta["type"]
    kind = {"journal-article": "article", "proceedings-article": "inproceedings",
            "book-chapter": "incollection"}.get(t, "misc")
    f = {"author": " and ".join((f"{fam}, {g}" if g else f"{{{fam}}}") for g, fam in meta["authors"]),
         "title": "{" + meta["title"].translate(BIB_ESC) + "}"}
    venue = meta["venue"].translate(BIB_ESC)
    if kind == "article":
        f["journal"] = venue
    elif kind in ("inproceedings", "incollection"):
        f["booktitle"] = venue
    else:
        f["howpublished"] = venue
    if meta["volume"]:
        f["volume"] = meta["volume"]
    if meta["issue"]:
        f["number"] = meta["issue"]
    if meta["pages"]:
        f["pages"] = meta["pages"].replace("--", "-").replace("-", "--")
    f["year"] = str(meta["year"])
    f["doi"] = r["doi"].translate(BIB_ESC)
    body = ",\n".join(f"  {k} = {{{v}}}" if k != "title" else f"  {k} = {v}" for k, v in f.items())
    return f"@{kind}{{{r['key']},\n{body}\n}}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use only cached API responses")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    cands = []
    for c in CATEGORIES:
        p = WORK / f"c{c}.csv"
        if not p.exists():
            sys.exit(f"missing {p}")
        for r in csv.DictReader(p.open(encoding="utf-8")):
            r["category"] = c
            r["rank"] = int(r["rank"])
            cands.append(r)

    # Integrator exclusions (key, reason): candidates that passed the script checks but were
    # rejected on manual review; each one is replaced by the next spare.
    excl = {}
    ex = WORK / "exclusions.csv"
    if ex.exists():
        excl = {e["key"]: e["reason"] for e in csv.DictReader(ex.open(encoding="utf-8"))}

    log, ok = [], []
    for r in cands:
        if r["key"] in excl:
            log.append({"category": r["category"], "rank": r["rank"], "key": r["key"],
                        "doi": r["doi"], "result": "excluded: " + excl[r["key"]], "selected": ""})
            continue
        meta, why = verify(r, args.offline)
        log.append({"category": r["category"], "rank": r["rank"], "key": r["key"], "doi": r["doi"],
                    "result": why, "selected": ""})
        if meta:
            r["_meta"], r["_why"], r["_log"] = meta, why, log[-1]
            ok.append(r)

    # Duplicates across categories: keep the instance with the better (lower) rank.
    best = {}
    for r in sorted(ok, key=lambda r: (r["rank"], r["category"])):
        ident = r["doi"].lower()
        if ident in best:
            r["_log"]["result"] += f"; duplicate of category {best[ident]['category']} row"
            continue
        best[ident] = r
    # Within a category, peer-reviewed papers come before arXiv-only ones (user preference),
    # otherwise the subagent's rank order is kept.
    pool = sorted(best.values(),
                  key=lambda r: (r["category"], r["peer_reviewed"].strip().lower() != "yes", r["rank"]))

    chosen = []
    for c, (_, target) in CATEGORIES.items():
        mine = [r for r in pool if r["category"] == c][:target]
        for r in mine:
            r["_note"] = "selected"
        if len(mine) < target:
            print(f"WARNING: category {c} has {len(mine)}/{target} verified papers", file=sys.stderr)
        chosen += mine
    if len(chosen) < TOTAL:  # fill from the best remaining spares, any category (logged)
        rest = sorted((r for r in pool if r not in chosen), key=lambda r: r["rank"])
        for r in rest[: TOTAL - len(chosen)]:
            r["_note"] = "fill (category short elsewhere)"
            chosen.append(r)
    chosen.sort(key=lambda r: (r["category"], r["rank"]))

    keys = set()
    rows, bib = [], []
    for i, r in enumerate(chosen, 1):
        m = r["_meta"]
        k = r["key"]
        while k in keys:
            k += "b"
        keys.add(k)
        r["key"] = k
        r["_log"]["selected"] = r["_note"]
        rows.append({
            "s_no": i, "category": r["category"], "category_name": CATEGORIES[r["category"]][0],
            "key": k, "title": m["title"], "authors": authors_str(m["authors"]),
            "venue": na(m["venue"]), "volume": na(m["volume"]), "issue": na(m["issue"]),
            "pages": na(pages_human(m["pages"])), "year": na(m["year"]), "doi": r["doi"],
            "url": m["url"], "pub_type": na(m["type"]), "peer_reviewed": r["peer_reviewed"],
            "work_done": r["work_done"].strip(), "advantages": r["advantages"].strip(),
            "disadvantages": r["disadvantages"].strip(), "relevance": r["relevance"].strip(),
            "basis": r["basis"].strip().lower(), "read_locator": r["read_locator"].strip(),
            "read_source_url": r["read_source_url"].strip(), "metadata_source": m["source"],
            "check_title_sim": m["sim"], "check_first_author": m["fa_ok"],
            "selection_note": r["_note"] + ("" if r["_why"] == "ok" else f"; {r['_why']}"),
        })
        bib.append(bib_entry(r, m))

    with (OUT_DIR / "survey50.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(rows)
    hdr = ("% V3 literature survey (50 papers), generated by v3/docs/research/survey/build_survey.py.\n"
           "% Bibliographic fields copied from Crossref / DataCite / arXiv records. Use with IEEEtran.bst.\n\n")
    (OUT_DIR / "survey50.bib").write_text(hdr + "\n".join(bib), encoding="utf-8")
    with (WORK / "verify_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category", "rank", "key", "doi", "result", "selected"])
        w.writeheader()
        w.writerows(log)

    from collections import Counter
    print("rows:", len(rows))
    print("per category:", dict(sorted(Counter(r["category"] for r in rows).items())))
    print("basis:", dict(Counter(r["basis"] for r in rows)))
    print("rejected:", [(l["category"], l["key"], l["result"]) for l in log if not l["result"].startswith("ok")])
    print("flagged:", [(r["key"], r["selection_note"]) for r in rows if r["selection_note"] != "selected"])


if __name__ == "__main__":
    main()
