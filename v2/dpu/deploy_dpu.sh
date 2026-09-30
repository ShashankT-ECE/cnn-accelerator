#!/usr/bin/env bash
# Copy the DPU package (v2/dpu/build/package/) to <user>@<host>:~/gos/dpu/ on the KV260.
#   v2/dpu/deploy_dpu.sh <board-ip> [user]
# Prerequisite: the V2 board deployment (v2/board/deploy.sh) is already on the board in ~/gos/
# (board_common.py, power_log.py, data/<net>/); dpu_session.py imports those from ~/gos.
set -euo pipefail
HOST="${1:?usage: deploy_dpu.sh <board-ip> [user]}"
USER_="${2:-ubuntu}"
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build/package"
[[ -f "$PKG/DPU_INFO.json" ]] || { echo "no $PKG/DPU_INFO.json: run make_dpu_package.py" >&2; exit 1; }
ssh "$USER_@$HOST" 'test -f ~/gos/board_common.py -a -f ~/gos/power_log.py -a -d ~/gos/data' \
  || { echo "~/gos is not a V2 board deployment: run v2/board/deploy.sh first" >&2; exit 1; }
rsync -av --delete "$PKG/" "$USER_@$HOST:gos/dpu/"
cat <<EOF
Deployed to $USER_@$HOST:~/gos/dpu/. On the board:
  source /etc/profile.d/pynq_venv.sh
  python3 -c "import pynq_dpu; print(pynq_dpu.__file__)"        # pynq-dpu 2.5 installed?
  cd ~/gos/dpu && ~/gos/session.sh py dpu_session.py --limit 200 --no-power --out-dir ~/gos/results/dpu_quick
  ~/gos/session.sh py dpu_session.py                                 # full: 10k/net + INA260 power
Copy back: rsync -av $USER_@$HOST:gos/results/hw_dpu_* v2/results/
EOF
