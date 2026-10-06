"""Independent re-verification of related.csv rows: DOI via Crossref, arXiv via export API, else URL HTTP status.
Writes verify_report.csv: key,doi,url,method,status,crossref_title,crossref_year,title_sim,first_author_ok"""
import csv, sys, json, re, time, urllib.request, urllib.parse, difflib
def get(url, accept=None):
    req = urllib.request.Request(url, headers={"User-Agent": "lit-check/1.0 (mailto:none)", **({"Accept": accept} if accept else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read().decode("utf-8", "replace")
def norm(s): return re.sub(r"[^a-z0-9 ]", "", re.sub(r"<[^>]+>", "", (s or "").lower())).split()
def sim(a, b): return difflib.SequenceMatcher(None, " ".join(norm(a)), " ".join(norm(b))).ratio()
rows = list(csv.DictReader(open(sys.argv[1])))
out = csv.writer(open(sys.argv[2], "w", newline=""))
out.writerow(["key","doi","url","method","status","found_title","found_year","title_sim","first_author_ok"])
for r in rows:
    doi, url, title, authors = r["doi"].strip(), r["url"].strip(), r["title"], r["authors"]
    fa = norm(authors.split(";")[0].split(" and ")[0].split(",")[0])
    fa = fa[-1] if fa else ""
    res = ["", "", 0.0, ""]; method = ""; status = ""
    try:
        if doi:
            method = "crossref"
            s, body = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
            m = json.loads(body)["message"]
            t = (m.get("title") or [""])[0]
            y = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
            fams = " ".join(a.get("family", "") for a in m.get("author", [])).lower()
            res = [t, y, round(sim(t, title), 3), str(bool(fa) and fa in norm(fams))]
            status = "ok"
        else:
            ax = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", url)
            if ax:
                method = "arxiv"
                s, body = get("http://export.arxiv.org/api/query?id_list=" + ax.group(1))
                t = re.findall(r"<title>(.*?)</title>", body, re.S)
                t = re.sub(r"\s+", " ", t[1]) if len(t) > 1 else ""
                y = (re.findall(r"<published>(\d{4})", body) or [""])[0]
                names = " ".join(re.findall(r"<name>(.*?)</name>", body)).lower()
                res = [t, y, round(sim(t, title), 3), str(bool(fa) and fa in norm(names))]
                status = "ok" if t else "notfound"
                time.sleep(3)
            elif url:
                method = "http"
                s, body = get(url); status = f"http{s}"
            else:
                method = "none"; status = "no-identifier"
    except Exception as e:
        status = "error:" + type(e).__name__ + ":" + str(e)[:60]
    out.writerow([r["key"], doi, url, method, status] + res)
    time.sleep(0.3)
