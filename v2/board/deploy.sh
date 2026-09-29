#!/usr/bin/env bash
# Deploy the board package to a KV260:  v2/board (+ data/) + bitstreams -> <user>@<host>:~/gos/
#
#   v2/board/deploy.sh <host> [--bit B.bit]... [--user ubuntu] [--dest gos] [--print-only]
#
# Default bitstreams: the performance builds v2/vivado/out/gos_300/gos_300.bit AND
# v2/vivado/out/gos_250/gos_250.bit (Session 1 tries 300 MHz first and falls back to 250 MHz if
# its smoke test fails: clock_fallback.py via run_sessions.py). --bit (repeatable) replaces the list.
# Remote layout (dest relative to the remote home):  ~/gos/*.py, ~/gos/data/<net>/,
# ~/gos/bit/<name>/<name>.bit|.hwh|.bit.sha256|.hwh.sha256|summary.json, ~/gos/DEPLOY_INFO.json,
# ~/gos/results/ (kept). DEPLOY_INFO.json records the repo commit + dirty flag of the deployed
# scripts, the data package SHA256 and, per bitstream ("bits", highest closed clock first), the
# SHA256s, BUILD_ID / Vivado version / closed pl_clk0 / WNS from that build's summary.json; the
# top-level bit fields describe the first (highest-clock) bitstream. The board has no git checkout.
# --print-only prints the ssh/rsync commands instead of running them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
V2="$(dirname "$HERE")"
REPO="$(dirname "$V2")"

[ $# -ge 1 ] || { sed -n '2,16p' "$0"; exit 2; }
HOST="$1"; shift
BITS=()
RUSER=ubuntu
DEST=gos
PRINT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --bit) BITS+=("$2"); shift 2 ;;
    --user) RUSER="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --print-only) PRINT=1; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
[ ${#BITS[@]} -gt 0 ] || BITS=("$V2/vivado/out/gos_300/gos_300.bit" "$V2/vivado/out/gos_250/gos_250.bit")

ABS=()
for b in "${BITS[@]}"; do
  b="$(cd "$(dirname "$b")" && pwd)/$(basename "$b")"
  h="${b%.bit}.hwh"
  d="$(dirname "$b")"
  [ -f "$b" ] && [ -f "$h" ] || { echo "ERROR: need $b and $h side by side"; exit 1; }
  [ -f "$d/summary.json" ] || { echo "ERROR: $d/summary.json missing (closed clock / timing of the build)"; exit 1; }
  for f in "$b" "$h"; do
    if [ -f "$f.sha256" ]; then
      want=$(cut -d' ' -f1 "$f.sha256")
      got=$(sha256sum "$f" | cut -d' ' -f1)
      [ "$want" = "$got" ] || { echo "ERROR: $f SHA256 $got != $f.sha256 $want"; exit 1; }
    fi
  done
  ABS+=("$b")
done

python3 "$HERE/board_common.py" verify "$HERE/data" || { echo "ERROR: data package incomplete (run make_board_data.py)"; exit 1; }
PKG_SHA=$(sha256sum "$HERE/data/PACKAGE.json" | cut -d' ' -f1)

COMMIT=$(git -C "$REPO" rev-parse HEAD)
if [ -n "$(git -C "$REPO" status --porcelain -- . ':!v2/results' ':!v2/vectors/MANIFEST.json')" ]; then
  DIRTY=true
  echo "WARNING: repo is dirty — board scripts will refuse to write hardware results without --allow-dirty"
else
  DIRTY=false
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
python3 - "$STAGE/DEPLOY_INFO.json" "$COMMIT" "$DIRTY" "$PKG_SHA" "$HOST" "$HERE/data/PACKAGE.json" \
    "${ABS[@]}" <<'PYEOF'
import datetime, hashlib, json, os, sys
out, commit, dirty, pkg, host, pkgjson, *bits = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
p = json.load(open(pkgjson))
entries = []
for b in bits:
    d, name = os.path.dirname(b), os.path.basename(b)[:-4]
    s = json.load(open(os.path.join(d, "summary.json")))
    entries.append({"bit": f"bit/{name}/{name}.bit", "hwh": f"bit/{name}/{name}.hwh",
                    "bit_sha256": sha(b), "hwh_sha256": sha(b[:-4] + ".hwh"),
                    "build_id": s.get("build_id", ""), "vivado_version": s.get("vivado_version", ""),
                    "bit_clock_mhz": s.get("pl_clk0_mhz_actual"),
                    "bit_clock_requested_mhz": s.get("pl_clk0_mhz_requested"),
                    "timing_wns_ns": s.get("wns_ns"), "timing_whs_ns": s.get("whs_ns")})
entries.sort(key=lambda e: -float(e["bit_clock_mhz"] or 0))
info = {"commit": commit, "dirty": dirty == "true",
        "deployed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": host, **{k: v for k, v in entries[0].items() if k != "timing_whs_ns"},
        "bits": entries, "data_package_sha256": pkg,
        "data_git_commit": p.get("git_commit"), "data_git_dirty": p.get("git_dirty")}
if info["data_git_commit"] != commit:
    print(f"NOTE: data package built at {str(info['data_git_commit'])[:8]}, scripts at {commit[:8]}")
json.dump(info, open(out, "w"), indent=1)
print(json.dumps(info, indent=1))
PYEOF

R="$RUSER@$HOST"
run() { if [ "$PRINT" = 1 ]; then printf '+'; printf ' %q' "$@"; echo; else "$@"; fi; }
run ssh "$R" "mkdir -p $DEST/bit $DEST/results"
run rsync -a --delete --exclude '/results/' --exclude '/bit/' --exclude '/DEPLOY_INFO.json' \
    --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '/dpu/' "$HERE/" "$R:$DEST/"
NAMES=""
for b in "${ABS[@]}"; do
  name="$(basename "${b%.bit}")"
  d="$(dirname "$b")"
  FILES=("$b" "${b%.bit}.hwh" "$d/summary.json")
  for f in "$b.sha256" "${b%.bit}.hwh.sha256"; do [ -f "$f" ] && FILES+=("$f"); done
  run ssh "$R" "mkdir -p $DEST/bit/$name"
  run rsync -a "${FILES[@]}" "$R:$DEST/bit/$name/"
  NAMES="$NAMES $name"
done
run rsync -a "$STAGE/DEPLOY_INFO.json" "$R:$DEST/DEPLOY_INFO.json"

if [ "$PRINT" = 1 ]; then WHAT="PRINT-ONLY (nothing copied) — target"; else WHAT="Deployed to"; fi
cat <<MSG

$WHAT $R:~/$DEST (commit ${COMMIT:0:8}, dirty=$DIRTY, bitstreams:$NAMES).
On the board (ssh $R; tmux; source /etc/profile.d/pynq_venv.sh), cd ~/$DEST, then:
  Pre-flight: sudo -E ./session.sh all --plan
  Session 1:  sudo -E ./session.sh 1        (clock fallback: first bitstream, else the next; choice recorded)
  Session 2:  sudo -E ./session.sh 2 [--budget-min N]
  Session 3:  sudo -E ./session.sh 3        (B1/B2 power = on-board INA260 SOM-rail logger only)
  Repeat on other days with --session-index 2 / 3, then: python3 aggregate_sessions.py
  (rerun the same command to resume; --fresh after a redeploy with other bitstreams)
Copy results back (laptop):
  rsync -av $R:$DEST/results/ $V2/results/
MSG
