"""Shared machinery for the paper table/figure pipeline (V2 step 10).

Honesty contract (v2/CLAUDE.md, EXPERIMENTS.md "CSV rule", DECISIONS D9/D16):
  * every number written into a table or figure annotation is read from v2/results/*.csv
    (or computed by this code from such numbers) and is registered with its origin
    (file, row, column / formula) through Artifact.num(); check_tex_provenance() refuses a
    table that contains a numeric token that was not registered;
  * rows with git_dirty != False are rejected (cifar10_retrain_log.csv exempt, D9);
  * board rows are accepted only with source == hw (cpu_board for the CPU baseline) and, when the
    row carries a paper_grade column (board runs through run_sessions.py), paper_grade == True
    (environment pre-flight passed: governor fixed, process pinned, no package manager running);
  * v2/results/dryrun/ is read only with --dryrun, and everything produced then is
    watermarked "DRY RUN -- NOT DATA" and written to generated/dryrun/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

V2 = Path(__file__).resolve().parents[2]
REPO = V2.parent
DEFAULT_RESULTS = V2 / "results"
DEFAULT_OUT = V2 / "paper" / "generated"

EXEMPT_DIRTY = {"cifar10_retrain_log.csv"}          # DECISIONS D9
D16_SOURCE_PATHS = ("v2/rtl", "v2/vivado", "v2/model")
HW_SOURCES = {"hw", "cpu_board"}                   # board rows (FPGA / board CPU)
DRYRUN_SOURCES = {"dryrun_model", "cpu_laptop"}     # only with --dryrun

SOURCE_LABEL = {
    "model": "model",
    "rtl_sim": "RTL sim",
    "post_impl": "post-impl",
    "post_synth_ooc": "post-synth OOC",
    "post_synth_funcsim": "post-synth netlist sim",
    "hw": "measured on KV260",
    "cpu_board": "measured on KV260 (A53 CPU)",
    "dryrun_model": "DRY RUN (not data)",
    "cpu_laptop": "DRY RUN (laptop CPU, not data)",
}

PH_BOARD = "TBD (board)"
DRYRUN_MARK = "DRY RUN -- NOT DATA"

# Text with digits that may appear in a table without coming from a CSV: names only, never
# quantities. The provenance check removes these strings before scanning for numbers.
STATIC_TEXT = [
    "LeNet-5", "CIFAR-10", "INT8", "INT32", "FP32", "KV260", "Cortex-A53", "A53", "SHA-256",
    "sha256", "RAMB36", "RAMB18", "BRAM36", "BRAM18", "DSP48E2", "fp32", "int8",
    "A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "C1", "C2", "INA260", "p95", "p5", "p50",
    "12 V", "pl\\_clk0", "pl_clk0", "xck26-sfvc784-2LV-c", "e2e", "r2",
    "95PCT CI", "p99", "DPUCZDX8G",
]


class ProvenanceError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def git_head() -> str:
    r = _git("rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else ""


# --------------------------------------------------------------------------------------------
# Data store: loads CSVs once, applies the row rules, remembers what was rejected.
# --------------------------------------------------------------------------------------------
class Store:
    def __init__(self, results_dir: Path = DEFAULT_RESULTS, dryrun: bool = False):
        self.results_dir = Path(results_dir)
        self.dryrun = dryrun
        self.hw_dir = self.results_dir / "dryrun" if dryrun else self.results_dir
        self._cache: dict[tuple[str, bool], list[dict] | None] = {}
        self.files: dict[str, dict] = {}          # path -> info (sha256, rows, rejected ...)
        self._stale: dict[str, str] = {}
        self.head = git_head()

    def path(self, name: str, hw: bool) -> Path:
        return (self.hw_dir if hw else self.results_dir) / name

    def glob(self, pattern: str, hw: bool = True, exclude: tuple = ("_sensorcheck",)) -> list[str]:
        d = self.hw_dir if hw else self.results_dir
        return sorted(p.name for p in d.glob(pattern) if p.is_file() and not any(x in p.name for x in exclude))

    def reject_reason(self, name: str, r: dict, hw: bool) -> str:
        src = r.get("source", "")
        dirty_ok = self.dryrun and hw                 # dry-run rows are never data anyway
        if name not in EXEMPT_DIRTY and not dirty_ok:
            if str(r.get("git_dirty", "")).strip() != "False":
                return f"git_dirty={r.get('git_dirty', '')!r}"
            if "data_git_dirty" in r and str(r["data_git_dirty"]).strip() not in ("False", ""):
                return f"data_git_dirty={r['data_git_dirty']!r}"
        if hw:
            allowed = DRYRUN_SOURCES if self.dryrun else HW_SOURCES
            if src not in allowed:
                return f"source={src!r} not accepted as board data"
            if not self.dryrun and "paper_grade" in r and str(r["paper_grade"]).strip() != "True":
                return f"paper_grade={r['paper_grade']!r} (environment pre-flight not passed)"
        elif src in HW_SOURCES | DRYRUN_SOURCES:
            return f"source={src!r} in a non-board file"
        return ""

    def load(self, name: str, hw: bool = False) -> list[dict] | None:
        """Clean rows of a CSV (None if the file does not exist)."""
        key = (name, hw)
        if key in self._cache:
            return self._cache[key]
        p = self.path(name, hw)
        if not p.exists():
            self._cache[key] = None
            return None
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        clean, rejected = [], []
        for i, r in enumerate(rows):
            r["_file"] = str(p)
            r["_line"] = i + 2          # 1-based line in the file (header = line 1)
            why = self.reject_reason(name, r, hw)
            (rejected if why else clean).append((r, why))
        info = {
            "path": _rel(p),
            "sha256": sha256_file(p),
            "rows_total": len(rows),
            "rows_clean": len(clean),
            "rejected": [{"line": r["_line"], "reason": why} for r, why in rejected],
            "matches_git_HEAD": _matches_head(p),
            "dryrun_dir": hw and self.dryrun,
        }
        self.files[str(p)] = info
        out = [r for r, _ in clean]
        self._cache[key] = out
        return out

    def stale_reason(self, commit: str) -> str:
        """DECISIONS D16: '' if v2/rtl, v2/vivado, v2/model at `commit` equal HEAD."""
        if commit not in self._stale:
            if not commit:
                self._stale[commit] = "no git_commit"
            elif _git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
                self._stale[commit] = f"unknown commit {commit[:8]}"
            else:
                rc = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", commit, "HEAD",
                                     "--", *D16_SOURCE_PATHS]).returncode
                self._stale[commit] = "" if rc == 0 else f"v2/rtl|vivado|model differ @ {commit[:8]}"
        return self._stale[commit]


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _matches_head(p: Path):
    """True if the file equals its committed version at HEAD, False if modified/untracked,
    None if outside the repository."""
    try:
        rel = Path(p).resolve().relative_to(REPO)
    except ValueError:
        return None
    if _git("ls-files", "--error-unmatch", str(rel)).returncode != 0:
        return False
    return _git("diff", "--quiet", "HEAD", "--", str(rel)).returncode == 0


# --------------------------------------------------------------------------------------------
# Artifact: one table or figure; tracks inputs, registered numbers, placeholders, checks.
# --------------------------------------------------------------------------------------------
@dataclass
class Artifact:
    store: Store
    name: str
    kind: str                        # "table" | "figure"
    outputs: list[str] = field(default_factory=list)
    inputs: dict[str, dict] = field(default_factory=dict)
    numbers: list[dict] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    sources: set = field(default_factory=set)

    # ---- rows -------------------------------------------------------------------------
    def rows(self, name: str, hw: bool = False, where: Callable[[dict], bool] | None = None,
             **eq) -> list[dict] | None:
        """Clean rows of `name` matching eq/where; records them as used. None if file absent."""
        rows = self.store.load(name, hw)
        path = str(self.store.path(name, hw))
        if rows is None:
            self.inputs.setdefault(path, {"file": _rel(Path(path)), "exists": False,
                                          "rows_used": 0, "git_commits": [], "sources": []})
            return None
        sel = [r for r in rows if all(str(r.get(k, "")) == str(v) for k, v in eq.items())
               and (where is None or where(r))]
        rec = self.inputs.setdefault(path, {"file": _rel(Path(path)), "exists": True,
                                            "rows_used": 0, "git_commits": [], "sources": [],
                                            "lines": []})
        for r in sel:
            if r["_line"] not in rec["lines"]:
                rec["lines"].append(r["_line"])
                rec["rows_used"] += 1
            c = r.get("git_commit", "")
            if c not in rec["git_commits"]:
                rec["git_commits"].append(c)
            s = r.get("source", "")
            if s not in rec["sources"]:
                rec["sources"].append(s)
            if s:
                self.sources.add(s)
        return sel

    # ---- numbers ----------------------------------------------------------------------
    def num(self, value, fmt: str = "{}", origin: str = "", row: dict | None = None,
            col: str | None = None) -> str:
        """Format a CSV-derived value and register the resulting text with its origin."""
        text = fmt_value(value, fmt)
        if row is not None:
            origin = f"{_rel(Path(row['_file']))}:{row['_line']}:{col or ''} {origin}".strip()
        self.numbers.append({"text": text, "origin": origin})
        return text

    def cell(self, row: dict, col: str, fmt: str = "{}", conv=None) -> str:
        v = row.get(col, "")
        if v in ("", None):
            raise ProvenanceError(f"{self.name}: empty {col} in {row['_file']}:{row['_line']}")
        v = conv(v) if conv else v
        return self.num(v, fmt, row=row, col=col)

    def label(self, text: str, origin: str) -> str:
        """Register a CSV-derived label that may contain digits (layer names, build ids)."""
        self.numbers.append({"text": text, "origin": origin})
        return text

    def placeholder(self, what: str, text: str = PH_BOARD) -> str:
        if what not in self.placeholders:
            self.placeholders.append(what)
        return text

    def check(self, msg: str):
        if msg not in self.checks:
            self.checks.append(msg)

    def registered_tokens(self) -> set[str]:
        toks = set()
        for n in self.numbers:
            toks |= set(num_tokens(n["text"]))
        return toks

    def manifest(self) -> dict:
        commits = sorted({c for rec in self.inputs.values() for c in rec.get("git_commits", [])})
        inputs = []
        for path, rec in self.inputs.items():
            info = self.store.files.get(path, {})
            inputs.append({**{k: v for k, v in rec.items() if k != "lines"},
                           "lines": sorted(rec.get("lines", [])),
                           "sha256": info.get("sha256"),
                           "rows_rejected_in_file": info.get("rejected", []),
                           "matches_git_HEAD": info.get("matches_git_HEAD"),
                           "dryrun_dir": info.get("dryrun_dir", False)})
        return {
            "kind": self.kind,
            "outputs": self.outputs,
            "sources": sorted(self.sources),
            "row_git_commits": commits,
            "stale_row_commits_D16": {c: self.store.stale_reason(c) for c in commits
                                      if self.store.stale_reason(c)},
            "inputs": inputs,
            "placeholders": self.placeholders,
            "checks": self.checks,
            "numbers": self.numbers,
        }


# --------------------------------------------------------------------------------------------
# Formatting and the provenance check
# --------------------------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def num_tokens(text: str) -> list[str]:
    return [t.rstrip(",") for t in _NUM_RE.findall(text)]


def fmt_value(value, fmt: str) -> str:
    if fmt == "int":
        s = f"{int(round(float(value))):,}"
    elif fmt == "raw":
        s = str(value)
    elif fmt.startswith("f"):                      # "f2" -> 2 decimals with thousands sep
        s = f"{float(value):,.{int(fmt[1:])}f}"
    elif fmt.startswith("pct"):                    # "pct2": signed error in %, 0 -> 0.00
        d = int(fmt[3:])
        v = float(value)
        s = f"{abs(v):.{d}f}"
        if float(s) != 0:
            s = ("+" if v > 0 else "$-$") + s
    else:
        s = fmt.format(value)
    if s.startswith("-"):
        s = "$-$" + s[1:]
    return s


_STRUCT_RE = [
    re.compile(r"\\multicolumn\{\d+\}"),
    re.compile(r"\\cmidrule(\([a-z]*\))?\{\d+-\d+\}"),
    re.compile(r"\\cline\{\d+-\d+\}"),
    re.compile(r"\\(label|ref)\{[^}]*\}"),
    re.compile(r"[A-Za-z0-9_*\\]+\.(csv|json|npz|py)"),        # file names in notes
    re.compile(r"\\begin\{tabular\*?\}\{[^}]*\}"),
    re.compile(r"\\begin\{table\*?\}(\[[a-zA-Z!]*\])?"),
    re.compile(r"\\hspace\{[^}]*\}|\\vspace\{[^}]*\}|\\rule\{[^}]*\}\{[^}]*\}"),
]


def strip_for_scan(tex: str) -> str:
    lines = [ln.split("%", 1)[0] if not ln.lstrip().startswith("%") else ""
             for ln in tex.replace("\\%", "PCT").splitlines()]
    s = "\n".join(lines)
    for rx in _STRUCT_RE:
        s = rx.sub(" ", s)
    for t in sorted(STATIC_TEXT, key=len, reverse=True):
        s = s.replace(t, " ")
    return s


def unregistered_numbers(tex: str, art: Artifact) -> list[str]:
    ok = art.registered_tokens()
    return [t for t in num_tokens(strip_for_scan(tex)) if t not in ok]


def check_tex_provenance(tex: str, art: Artifact):
    bad = unregistered_numbers(tex, art)
    if bad:
        body = strip_for_scan(tex)
        ctx = []
        for t in sorted(set(bad)):
            for m in re.finditer(r"(?<![\d.,])" + re.escape(t) + r"(?![\d.,]*\d)", body):
                ctx.append(repr(body[max(0, m.start() - 25):m.end() + 10]))
                break
        raise ProvenanceError(f"{art.name}: numbers without a CSV origin: {sorted(set(bad))} "
                              f"context: {'; '.join(ctx)}")


# --------------------------------------------------------------------------------------------
# LaTeX table builder (IEEEtran; booktabs by default)
# --------------------------------------------------------------------------------------------
def latex_table(art: Artifact, colspec: str, header: list[str], body: list[list[str] | str],
                caption: str, label: str, notes: list[str] | None = None,
                booktabs: bool = True, wide: bool = False, dryrun: bool = False) -> str:
    top, mid, bot = (r"\toprule", r"\midrule", r"\bottomrule") if booktabs else (r"\hline", r"\hline", r"\hline")
    ncol = _ncols(colspec)
    env = "table*" if wide else "table"
    out = [f"% Generated by v2/paper/scripts/make_all.py -- DO NOT EDIT BY HAND.",
           f"% Artifact: {art.name}. Inputs and row commits: generated/MANIFEST.json.",
           "% Requires \\usepackage{booktabs}." if booktabs else "% Plain tabular (no booktabs).",
           f"\\begin{{{env}}}[t]", r"\centering",
           f"\\caption{{{caption}{' (' + DRYRUN_MARK + ')' if dryrun else ''}}}",
           f"\\label{{{label}}}", r"\footnotesize",
           f"\\begin{{tabular}}{{{colspec}}}", top]
    if dryrun:
        out.append(f"\\multicolumn{{{ncol}}}{{c}}{{\\textbf{{{DRYRUN_MARK}}}}} \\\\")
        out.append(mid)
    for h in header:
        done = h.rstrip().endswith("\\\\") or h.lstrip().startswith(("\\cmidrule", "\\midrule", "\\cline"))
        out.append(h if done else h + r" \\")
    out.append(mid)
    for r in body:
        if isinstance(r, str):
            out.append(r)
        else:
            span = sum(int(m.group(1)) if (m := re.match(r"\\multicolumn\{(\d+)\}", str(x))) else 1 for x in r)
            if span != ncol:
                raise ValueError(f"{art.name}: row spans {span} columns, table has {ncol}: {r}")
            out.append(" & ".join(r) + r" \\")
    out.append(bot)
    out.append(r"\end{tabular}")
    if notes:
        out.append(r"\par\vspace{2pt}\parbox{\linewidth}{\scriptsize " + " ".join(notes) + "}")
    out.append(f"\\end{{{env}}}")
    tex = "\n".join(out) + "\n"
    check_tex_provenance(tex, art)
    return tex


def _ncols(colspec: str) -> int:
    s = re.sub(r"@\{[^}]*\}", "", colspec)
    s = re.sub(r"[pmb]\{[^}]*\}", "l", s)
    s = re.sub(r"S\[[^\]]*\]", "S", s)
    return sum(1 for ch in s if ch in "lcrS")


def tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False, default=str) + "\n")


def fnum(x) -> float:
    return float(x)


def inum(x) -> int:
    f = float(x)
    if f != int(f):
        raise ValueError(f"expected an integer, got {x!r}")
    return int(f)


def pretty_net(net: str) -> str:
    return {"lenet5": "LeNet-5", "cifar10": "CIFAR-10"}.get(net, tex_escape(net))


NETS = ("lenet5", "cifar10")
