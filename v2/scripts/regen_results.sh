#!/usr/bin/env bash
# Regenerate V2 model results CSVs from a clean, committed tree (v2/CLAUDE.md "Honesty").
# Refuses to run on a dirty tree so every row carries git_dirty=False and the
# commit that produced it. Commit the regenerated CSVs in a separate commit.
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1

if [[ -n "$(git status --porcelain)" ]]; then
    echo "regen_results.sh: REFUSED — working tree is dirty. Commit first:" >&2
    git status --porcelain >&2
    exit 1
fi
echo "regen_results.sh: clean tree at $(git rev-parse --short HEAD)"

HW_NPZ=(v2/model/frozen/lenet5_int8/hw_requant.npz v2/model/frozen/cifar10_int8/hw_requant.npz)
before="$(sha256sum "${HW_NPZ[@]}")"

cd v2/model
"$PY" reference_accuracy.py
"$PY" requant_check.py --nets lenet5 cifar10
"$PY" final_layer.py
"$PY" gos_cycle_model.py
"$PY" golden_crosscheck.py
cd "$REPO_ROOT"

after="$(sha256sum "${HW_NPZ[@]}")"
if [[ "$before" != "$after" ]]; then
    echo "regen_results.sh: STOP — hw_requant.npz changed on regeneration:" >&2
    diff <(echo "$before") <(echo "$after") >&2 || true
    exit 2
fi

# Every row must be clean and from HEAD.
"$PY" - <<'PYEOF'
import csv, glob, subprocess
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
for path in ("v2/results/reference_accuracy.csv", "v2/results/requant_equivalence.csv",
             "v2/results/final_layer_check.csv", "v2/results/cycle_model.csv",
             "v2/results/golden_crosscheck.csv"):
    rows = list(csv.DictReader(open(path)))
    bad = [r for r in rows if r["git_dirty"] != "False" or r["git_commit"] != head]
    assert rows and not bad, f"{path}: {len(bad)} rows not clean/HEAD"
    print(f"  {path}: {len(rows)} rows, clean @ {head[:7]}")
PYEOF
echo "regen_results.sh: done. Commit v2/results/*.csv separately."
