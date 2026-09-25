#!/usr/bin/env bash
# Deploy the board package to a KV260:  v2/board (+ data/) + bitstream -> <user>@<host>:~/gos/
#
#   v2/board/deploy.sh <host> [--bit v2/vivado/out/gos_200/gos_200.bit] [--user ubuntu]
#                             [--dest gos] [--print-only]
#
# Remote layout (dest relative to the remote home):  ~/gos/*.py, ~/gos/data/<net>/,
# ~/gos/bit/<name>.bit|.hwh|.sha256|summary.json, ~/gos/DEPLOY_INFO.json, ~/gos/results/ (kept).
# DEPLOY_INFO.json records the repo commit + dirty flag of the deployed scripts, the bitstream
# SHA256s, BUILD_ID / Vivado version / closed pl_clk0 from the build's summary.json, and the data
# package SHA256 — the board scripts read it (the board has no git checkout).
# --print-only prints the ssh/rsync commands instead of running them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
V2="$(dirname "$HERE")"
REPO="$(dirname "$V2")"

[ $# -ge 1 ] || { sed -n '2,13p' "$0"; exit 2; }
HOST="$1"; shift
BIT="$V2/vivado/out/gos_200/gos_200.bit"
RUSER=ubuntu
DEST=gos
PRINT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --bit) BIT="$2"; shift 2 ;;
    --user) RUSER="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --print-only) PRINT=1; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
BIT="$(cd "$(dirname "$BIT")" && pwd)/$(basename "$BIT")"
HWH="${BIT%.bit}.hwh"
BITDIR="$(dirname "$BIT")"
[ -f "$BIT" ] && [ -f "$HWH" ] || { echo "ERROR: need $BIT and $HWH side by side"; exit 1; }

BIT_SHA=$(sha256sum "$BIT" | cut -d' ' -f1)
HWH_SHA=$(sha256sum "$HWH" | cut -d' ' -f1)
for f in "$BIT" "$HWH"; do
  if [ -f "$f.sha256" ]; then
    want=$(cut -d' ' -f1 "$f.sha256")
    got=$(sha256sum "$f" | cut -d' ' -f1)
    [ "$want" = "$got" ] || { echo "ERROR: $f SHA256 $got != $f.sha256 $want"; exit 1; }
  fi
done

python3 "$HERE/board_common.py" verify "$HERE/data" || { echo "ERROR: data package incomplete (run make_board_data.py)"; exit 1; }
PKG_SHA=$(sha256sum "$HERE/data/PACKAGE.json" | cut -d' ' -f1)
# fclk0 limit for the smoke tests = the build's closed pl_clk0 (summary.json) + 0.5 MHz
MAXF=$(python3 -c 'import json,sys; print(round(float(json.load(open(sys.argv[1]))["pl_clk0_mhz_actual"]) + 0.5, 3))' \
       "$BITDIR/summary.json" 2>/dev/null || echo 200.5)

COMMIT=$(git -C "$REPO" rev-parse HEAD)
if [ -n "$(git -C "$REPO" status --porcelain -- . ':!v2/results' ':!v2/vectors/MANIFEST.json')" ]; then
  DIRTY=true
  echo "WARNING: repo is dirty — board scripts will refuse to write hardware results without --allow-dirty"
else
  DIRTY=false
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
python3 - "$STAGE/DEPLOY_INFO.json" "$COMMIT" "$DIRTY" "$(basename "$BIT")" "$(basename "$HWH")" \
    "$BIT_SHA" "$HWH_SHA" "$BITDIR/summary.json" "$PKG_SHA" "$HOST" "$HERE/data/PACKAGE.json" <<'PYEOF'
import datetime, json, os, sys
out, commit, dirty, bit, hwh, bsha, hsha, summ, pkg, host, pkgjson = sys.argv[1:]
s = json.load(open(summ)) if os.path.isfile(summ) else {}
p = json.load(open(pkgjson))
info = {
    "commit": commit, "dirty": dirty == "true",
    "deployed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "host": host, "bit": f"bit/{bit}", "hwh": f"bit/{hwh}", "bit_sha256": bsha, "hwh_sha256": hsha,
    "build_id": s.get("build_id", ""), "vivado_version": s.get("vivado_version", ""),
    "bit_clock_mhz": s.get("pl_clk0_mhz_actual"), "bit_clock_requested_mhz": s.get("pl_clk0_mhz_requested"),
    "timing_wns_ns": s.get("wns_ns"), "data_package_sha256": pkg,
    "data_git_commit": p.get("git_commit"), "data_git_dirty": p.get("git_dirty"),
}
if info["data_git_commit"] != commit:
    print(f"NOTE: data package built at {str(info['data_git_commit'])[:8]}, scripts at {commit[:8]}")
json.dump(info, open(out, "w"), indent=1)
print(json.dumps(info, indent=1))
PYEOF

R="$RUSER@$HOST"
run() { if [ "$PRINT" = 1 ]; then printf '+'; printf ' %q' "$@"; echo; else "$@"; fi; }
run ssh "$R" "mkdir -p $DEST/bit $DEST/results"
run rsync -a --delete --exclude '/results/' --exclude '/bit/' --exclude '/DEPLOY_INFO.json' \
    --exclude '__pycache__/' --exclude '.pytest_cache/' "$HERE/" "$R:$DEST/"
BITFILES=("$BIT" "$HWH")
for f in "$BIT.sha256" "$HWH.sha256" "$BITDIR/summary.json"; do [ -f "$f" ] && BITFILES+=("$f"); done
run rsync -a "${BITFILES[@]}" "$R:$DEST/bit/"
run rsync -a "$STAGE/DEPLOY_INFO.json" "$R:$DEST/DEPLOY_INFO.json"

if [ "$PRINT" = 1 ]; then WHAT="PRINT-ONLY (nothing copied) — target"; else WHAT="Deployed to"; fi
cat <<MSG

$WHAT $R:~/$DEST (commit ${COMMIT:0:8}, dirty=$DIRTY, bit $(basename "$BIT") ${BIT_SHA:0:12}).
On the board (ssh $R), cd ~/$DEST, then:
  Session 1:  sudo -E python3 test_shell.py --bit bit/$(basename "$BIT") --expect-version 0x474F5302 --skip-scratch --max-fclk0 $MAXF
              sudo -E python3 test_core_smoke.py --bit bit/$(basename "$BIT") --max-fclk0 $MAXF
  Session 2:  sudo -E ./run_all.sh 2>&1 | tee results/run_all_\$(date +%Y%m%d_%H%M%S).log
  Session 3:  sudo -E python3 exp_b1_power.py --modes idle fpga cpu --net lenet5
              sudo -E python3 exp_b2_clock.py --net lenet5
Copy results back (laptop):
  rsync -av $R:$DEST/results/ $V2/results/
MSG
