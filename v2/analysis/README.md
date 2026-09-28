# analysis — model-only paper analyses of the frozen V2 design (source=model)

Scripts here import `v2/model` (and, through it, legacy `python/v2_architecture_model.py`)
**read-only** and never write under `v2/model` (DECISIONS D16). They add no architecture:
every number is computed from the frozen cycle model (`gos_cycle_model.py`,
`net_config.py`). None of it is an RTL-simulation, post-implementation or KV260 measurement.

| script | CSV (`--out-dir`, default `v2/results/`) | what |
|---|---|---|
| `utilization.py` | `utilization_model.csv` (+ `v2/results/utilization_model.md`, `--md`) | EXPERIMENTS A4, model side: per-layer spatial / temporal / total PE utilization of the 8x8 array, exact loss decomposition (OC tail, x tail, 1x1-output effect, pipeline fill); asserts the 1/8 spatial bound for OH = OW = 1 layers |
| `ablation.py` | `schedule_ablation.csv` | DECISIONS D7: legacy V1-style generalized OS schedule (core + lead-in + pass drain + clear + result drain) vs V2 `LAYER_CYC`, LeNet-5 and CIFAR-10 |
| `projection_16x16.py` | `projection_16x16.csv` | cycle model parameterized by array size N (N=8 asserted equal to the 8x8 model; N=16 labeled "projected (model, not implemented)"). **Not EXPERIMENTS C1** (C1 is post-implementation only). Assumptions P1–P5 are in the script docstring and the CSV `assumptions` column; no resource or fmax claim. |

Paper CSVs (and `v2/results/utilization_model.md`) are produced only by `v2/scripts/regen_results.sh`
from a clean committed tree. For experiments write elsewhere:

```bash
.venv/bin/python v2/analysis/utilization.py --out-dir /some/scratch --md /some/scratch/v2/results/utilization_model.md
.venv/bin/python v2/analysis/ablation.py --out-dir /some/scratch
.venv/bin/python v2/analysis/projection_16x16.py --out-dir /some/scratch
.venv/bin/python -m pytest v2/analysis          # tests/test_analysis_*.py
```

The tests compare against the committed `v2/results/cycle_model.csv` (per-layer and total
cycles) and the D7 LeNet decomposition.
