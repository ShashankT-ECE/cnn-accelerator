# LeNet-5 Mapping Baseline — Frozen Manifest (Step 4.3)

> **Status: FROZEN (2026-09-23).** This is the authoritative, reproducibility-
> gated hardware-mapping baseline for the LeNet-5 workload on the frozen 8×8
> reconfigurable systolic array. It records the corrected cycle model (Step 4.1
> FC schedule, Step 4.2 correction) and the verification status of every timing
> assumption. No RTL, sparsity, or L1/L2 artifacts are modified by this manifest.

## Baseline numbers (cycle model, un-pipelined correctness-first)

| Layer | IC→OC | MACs | OS cycles | WS cycles | Optimal |
|---|---|---:|---:|---:|---|
| conv1 | 1→6 | 117,600 | 9,184 | 28,224 | OS |
| conv3 | 6→16 | 240,000 | 16,280 | 51,840 | OS |
| conv5 | 16→120 | 48,000 | 15,855 | 49,200 | OS |
| fc1 | 120→84 | 10,080 | 1,510 | 10,248 | OS |
| fc2 | 84→10 | 840 | 192 | 900 | OS |
| **TOTAL** | | **416,520** | **43,021** | **140,412** | **OS** |

- **OS is optimal for every layer.** There is **no OS/WS crossover** on LeNet-5.
- **Reconfiguration:** sequence `OS→OS→OS→OS→OS`, 0 switches, 0 overhead,
  reconfigurable == fixed OS = 43,021 cycles. **No reconfiguration benefit**
  (speedup vs OS = 1.000×; vs WS = 3.264×).

## Cycle formulas

- **OS conv** `= 17 + IC·K·(K+8)` per 8-pixel group; groups `= ceil(OC/8)·H'·ceil(W'/8)`.
- **OS FC** `= ceil(OC/8)·(IC + 2) + 2·OC`.
- **WS conv** `= 8·ceil(T/8) + 10` per group, `T = K·K·IC`; groups `= OC·H'·ceil(W'/8)`.
- **WS FC** `= OC·(8·ceil(IC/8) + 2)`.

## Verification status (distinction preserved)

| Timing | Status | Evidence |
|---|---|---|
| OS conv group 82 cyc (IC=1, K=5) | **RTL-verified** | `PHASE2_ARCHITECTURE.md` §3.1 |
| WS conv group 42 cyc (25 taps, 4 tiles) | **RTL-verified** | `SYSTOLIC_ARRAY_V2_SPEC.md` §9.9 |
| OS FC `IC + 2·rows + 2` | **RTL-verified** (bit-exact) | `FC_SCHEDULE_VALIDATION.md`; `fc_schedule_sim.py` vs frozen golden |
| WS FC `8·ceil(IC/8) + 2` | **analytically derived** | conv-WS anchor; `FC_SCHEDULE_VALIDATION.md` §3 |
| Conv3/Conv5 (IC>1, OC>8, 32×32) | **unverified generalization** | controller gap G1–G3 (`LENET5_MAPPING_SPEC.md` §4) |

## Artifacts

- `python/mapping_model.py`, `python/reconfig_model.py`, `python/activity_model.py`.
- `data/benchmark/mapping_results.json`, `reconfig_results.json`, `activity_results.json`.
- Tests: `python/tests/test_mapping_model.py`, `test_reconfig_model.py`, `test_activity_model.py` (all PASS).
- Frozen L2 golden (tensor shapes) cross-checked: `data/lenet5_int8/manifest.json`.

## Reproducibility gate result

Gates 1–9 all PASS (tensor-shape match, MACs 416,520, OS/WS per-layer + totals,
OS optimality, reconfig = OS×5 with zero overhead, no stale Step-3 values, full
test suite green). This baseline is frozen for Step 5 (sparsity) and any RTL work.
