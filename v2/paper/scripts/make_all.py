#!/usr/bin/env python3
"""V2 step 10: generate every paper table (.tex) and figure (.pdf) from v2/results/*.csv.

    .venv/bin/python v2/paper/scripts/make_all.py            # paper data -> v2/paper/generated/
    .venv/bin/python v2/paper/scripts/make_all.py --dryrun   # LAYOUT TEST ONLY -> generated/dryrun/

Rules (v2/CLAUDE.md honesty; DECISIONS D9, D16): numbers come only from the CSVs; rows with
git_dirty != False are rejected (retrain log exempt); board data only from hw_*.csv rows with
source=hw (cpu_board for the CPU baseline). Missing board data -> "TBD (board)" cells and
hatched "board data pending" placeholders; the switch to real data is automatic. --dryrun
reads v2/results/dryrun/ (never paper data) and watermarks every output "DRY RUN -- NOT DATA".
Writes generated/MANIFEST.json: per artifact, the input CSVs (sha256), rows used, their
git_commits, sources, placeholders, consistency checks and every registered number's origin.

Extra artifacts: if scripts/tables_extra.py exists, tables_extra.make(tctx, failures) is called
(one guarded import) with the same table context tables.Ctx (see its docstring: store, out,
booktabs, dryrun, art(), table(), emit(), board_label, fctx = the figures.FCtx) and must return a
list of paperlib.Artifact (or one Artifact); failures it appends are reported like the others.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figures  # noqa: E402
import tables  # noqa: E402
from paperlib import DEFAULT_OUT, DEFAULT_RESULTS, Store, _rel, write_json  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default v2/paper/generated (generated/dryrun with --dryrun)")
    ap.add_argument("--dryrun", action="store_true",
                    help="use v2/results/dryrun/ board rows for LAYOUT TESTING ONLY (watermarked)")
    ap.add_argument("--no-booktabs", action="store_true", help="plain \\hline tables")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    out = a.out_dir or (DEFAULT_OUT / "dryrun" if a.dryrun else DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(a.results_dir, dryrun=a.dryrun)

    arts, failures = [], []
    tctx = tables.Ctx(store, out, booktabs=not a.no_booktabs, dryrun=a.dryrun)
    for fn in tables.ALL:
        try:
            arts.append(fn(tctx))
        except Exception as e:                       # report, keep going, exit nonzero
            failures.append({"artifact": fn.__name__, "error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc()})
    fctx = figures.FCtx(store, out, dryrun=a.dryrun)
    tctx.fctx = fctx
    figures.style()
    for fn, kw in [(figures.fig_a3, {"wide": False}), (figures.fig_a3, {"wide": True}), (figures.fig_a4, {}),
                   (figures.fig_b2, {}), (figures.fig_b3, {})]:
        try:
            arts.append(fn(fctx, **kw))
        except Exception as e:
            failures.append({"artifact": fn.__name__, "error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc()})

    try:
        import tables_extra  # noqa: PLC0415 - optional module (publication extras, other owner)
    except ImportError:
        tables_extra = None
    if tables_extra is not None:
        try:
            got = tables_extra.make(tctx, failures)
            arts.extend(got if isinstance(got, (list, tuple)) else [got])
        except Exception as e:
            failures.append({"artifact": "tables_extra.make", "error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc()})

    manifest = {
        "generator": "v2/paper/scripts/make_all.py",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_HEAD": store.head,
        "results_dir": _rel(store.results_dir),
        "dryrun": a.dryrun,
        "dryrun_warning": "DRY RUN -- NOT DATA: board rows from v2/results/dryrun/" if a.dryrun else None,
        "files": list(store.files.values()),
        "artifacts": {x.name: x.manifest() for x in arts},
        "failures": failures,
    }
    write_json(out / "MANIFEST.json", manifest)

    if not a.quiet:
        print(f"make_all: {len(arts)} artifacts -> {_rel(out)}{'  [DRY RUN -- NOT DATA]' if a.dryrun else ''}")
        for x in arts:
            ph = f"  placeholders: {len(x.placeholders)}" if x.placeholders else ""
            print(f"  {x.name:22s} {', '.join(Path(o).name for o in x.outputs)}  "
                  f"sources: {','.join(sorted(x.sources)) or '-'}{ph}")
        for f in store.files.values():
            flag = []
            if f["rejected"]:
                flag.append(f"{len(f['rejected'])} rows rejected ({f['rejected'][0]['reason']} ...)")
            if f["matches_git_HEAD"] is False:
                flag.append("differs from git HEAD (uncommitted)")
            if flag:
                print(f"  NOTE {f['path']}: {'; '.join(flag)}")
        checks = sorted({c for x in arts for c in x.checks})
        stale = sorted({f"{c[:8]}: {r}" for x in arts for c, r in x.manifest()["stale_row_commits_D16"].items()})
        for c in checks:
            print(f"  CHECK {c}")
        for s in stale:
            print(f"  STALE (D16) {s}")
        for f in failures:
            print(f"  FAILED {f['artifact']}: {f['error']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
