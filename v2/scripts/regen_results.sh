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

# NET_CONFIGS hw_requant files (cifar10 = r2 since Step 2.1c); must not change on regeneration.
HW_NPZ=(v2/model/frozen/lenet5_int8/hw_requant.npz v2/model/frozen/cifar10_int8_r2/hw_requant.npz)
before="$(sha256sum "${HW_NPZ[@]}")"

cd v2/model
"$PY" reference_accuracy.py
"$PY" requant_check.py --nets lenet5 cifar10
"$PY" final_layer.py
"$PY" gos_cycle_model.py
"$PY" golden_crosscheck.py
"$PY" retrain/check_r2.py          # r1 vs r2 acceptance record (reads the CSVs above)
cd "$REPO_ROOT"

after="$(sha256sum "${HW_NPZ[@]}")"
if [[ "$before" != "$after" ]]; then
    echo "regen_results.sh: STOP — hw_requant.npz changed on regeneration:" >&2
    diff <(echo "$before") <(echo "$after") >&2 || true
    exit 2
fi

# Every row of every results CSV must be clean and from HEAD. Sole exemption:
# cifar10_retrain_log.csv is a training artifact tied to the r2 checkpoint SHA256
# (DECISIONS D9), not regenerable without retraining; it is checked for that tie.
"$PY" - <<'PYEOF'
import csv, glob, hashlib, json, subprocess
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
TRAINING_ARTIFACTS = {"v2/results/cifar10_retrain_log.csv"}
REGEN = {"reference_accuracy", "requant_equivalence", "final_layer_check", "cycle_model",
         "golden_crosscheck", "cifar10_r2_accuracy", "cifar10_r2_summary"}
# CSVs written by the RTL / Vivado result scripts (checked by v2/scripts/check_results.py
# after those scripts run; they may still be from the previous commit at this point).
OTHER_PRODUCERS = {"unit_tb": "run_unit_all.sh", "ooc_synth": "ooc_all.sh",
                   "rtl_cycles": "run_core.sh", "rtl_network": "run_core.sh",
                   "rtl_checker": "run_core.sh", "impl_shell": "vivado/build_shell.sh"}
paths = sorted(glob.glob("v2/results/*.csv"))
found = {p.split("/")[-1][:-4] for p in paths if p not in TRAINING_ARTIFACTS} - set(OTHER_PRODUCERS)
assert found == REGEN, f"results CSVs not produced by any known script: {sorted(found - REGEN)}; missing {sorted(REGEN - found)}"
paths = [p for p in paths if p.split("/")[-1][:-4] not in OTHER_PRODUCERS]
for path in paths:
    rows = list(csv.DictReader(open(path)))
    if path in TRAINING_ARTIFACTS:
        meta = json.load(open("v2/model/retrain/cifar10_fp32_r2_meta.json"))
        sha = hashlib.sha256(open(meta["checkpoint"]["path"], "rb").read()).hexdigest()
        assert meta["log_csv"] == path and sha == meta["checkpoint"]["sha256"]
        assert len(rows) == meta["training"]["epochs_run"]
        assert [r["epoch"] for r in rows if r["selected"] == "1"] == [str(meta["selected_epoch"])]
        print(f"  {path}: {len(rows)} rows, training artifact of checkpoint {sha[:12]} (exempt)")
        continue
    bad = [r for r in rows if r["git_dirty"] != "False" or r["git_commit"] != head]
    assert rows and not bad, f"{path}: {len(bad)} rows not clean/HEAD"
    print(f"  {path}: {len(rows)} rows, clean @ {head[:7]}")
PYEOF
echo "regen_results.sh: done. Commit v2/results/*.csv separately."
