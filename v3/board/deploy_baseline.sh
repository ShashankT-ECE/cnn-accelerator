#!/usr/bin/env bash
# Deploy the V3 baseline package to a KV260:  v3/board/data/package/ -> <user>@<host>:~/gos3/
# Adapted from v2/board/deploy.sh + v2/dpu/deploy_dpu.sh @ 28dd2ad. ~/gos/ (the V2 deployment and its
# results) is never touched: everything goes to ~/gos3/, results stay in ~/gos3/results/ (kept across deploys).
#
#   v3/board/deploy_baseline.sh <host> [--user ubuntu] [--dest gos3] [--print-only]
#
# The package is built by make_baseline_package.py (refuses a dirty tree unless --allow-dirty; the board session
# refuses a dirty package unless its own --allow-dirty). Every file is checked against MANIFEST.json before copying.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$HERE/data/package"
[ $# -ge 1 ] || { sed -n '2,10p' "$0"; exit 2; }
HOST="$1"; shift
RUSER=ubuntu
DEST=gos3
PRINT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --user) RUSER="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --print-only) PRINT=1; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
[ "$DEST" != gos ] || { echo "REFUSED: ~/gos is the V2 deployment; V3 deploys to ~/gos3"; exit 2; }
[ -f "$PKG/MANIFEST.json" ] || { echo "no $PKG/MANIFEST.json: run v3/board/make_baseline_package.py"; exit 1; }
python3 - "$PKG" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1]); m = json.loads((d / "MANIFEST.json").read_text())
for n, e in m["files"].items():
    if hashlib.sha256((d / n).read_bytes()).hexdigest() != e["sha256"]:
        sys.exit(f"MANIFEST mismatch: {n}")
kinds = {k: v.get("checkpoint_kind") for k, v in m["nets"].items()}
print(f"package OK: {len(m['files'])} files, commit {m['git_commit'][:8]} dirty={m['git_dirty']}, checkpoints {kinds}")
if m["git_dirty"]:
    print("WARNING: dirty package -- the board session refuses it without --allow-dirty (never paper-grade)")
if any(v != "external" for v in kinds.values()):
    print("WARNING: dry-run throwaway checkpoint(s) -- board rows will be paper_grade=False (pipeline check only)")
PYEOF
R="$RUSER@$HOST"
run() { if [ "$PRINT" = 1 ]; then printf '+'; printf ' %q' "$@"; echo; else "$@"; fi; }
run ssh "$R" "mkdir -p $DEST/results"
run rsync -a --delete --exclude '/results*/' --exclude '__pycache__/' "$PKG/" "$R:$DEST/"
run ssh "$R" "chmod +x $DEST/session.sh"
if [ "$PRINT" = 1 ]; then WHAT="PRINT-ONLY (nothing copied) -- target"; else WHAT="Deployed to"; fi
cat <<MSG

$WHAT $R:~/$DEST. On the board (ssh $R; tmux), cd ~/$DEST:
  ./session.sh py power_log.py --list-sensors          # INA260 present?
  ./session.sh 1 --plan                                 # pre-flight + step list, runs nothing
  ./session.sh 1 --limit 200 --no-power --allow-non-paper-grade --results-dir ~/$DEST/results_smoke   # smoke
  ./session.sh 1                                        # session 1 (rerun the same command to resume)
  ./session.sh 2 ; ./session.sh 3                       # other days, then: python3 aggregate_sessions.py
Copy back (laptop):  rsync -av $R:$DEST/results/ v3/results/   (only hw_baseline_*.csv are results rows)
MSG
