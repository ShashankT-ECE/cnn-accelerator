# v2/shapes — A3-general random-shape experiment (RTL side)

About 300 seeded random multi-layer `gos_` jobs inside the V2 architecture envelope. Each job runs
through `gos_core` in RTL simulation. The harness checks:

- outputs bit-exact against the golden model;
- `LAYER_CYC`, `TOTAL_CYC` and `MAC_ACTIVE` equal to `gos_cycle_model`;
- `STALL` = 0.

A few jobs are labeled **expected refusal**. For those, the config checker must refuse the job and
report the expected `ERR_CODE`. Output: `v2/results/shapes_rtl.csv`, one row per job,
`source=rtl_sim`. Label every number from it **RTL sim**. The same shape set goes to the KV260 (`v2/board/exp_shapes.py`, loader
`v2/board/shapeset.py`) → `hw_shapes.csv` (source=hw). Paper: `tab_shapes`, `fig_shapes_cycles`
(`v2/paper/scripts/tables_extra.py`).

RTL, `v2/model`, `v2/vivado`, `v2/fullsim` and existing TBs are not touched (model code is imported
read-only). Everything generated goes under `v2/build/shapes/` (gitignored).

## Entry point / final run

    v2/scripts/run_shapes.sh            # from a clean, committed tree: defaults = the paper run

Options: `[--seed S] [--n-jobs N] [--sim verilator|xsim] [--jobs-per-shard M] [--limit K]
[--max-par P] [--xcheck-min X] [--force] [--allow-dirty] [--no-xval] [--ignore-vivado] [--csv PATH]`.
Defaults are seed 20260929, 300 jobs, Verilator, 20 jobs per shard, P ≤ 8, and an xsim
cross-check of at least 20 jobs.

`v2/results/shapes_rtl.csv` and `v2/results/shapes_rtl_logs_<commit12>-<shapeset12>.tar.xz` are
written only if all of these hold:
- the tree is clean;
- the seed and job count are the defaults;
- every job ran (no `--limit`, no `--csv`).

Otherwise the CSV goes to `<run dir>/shapes_rtl.csv`. A dirty tree is refused unless
`--allow-dirty` is given; those rows carry `git_dirty=True` and are rejected by the paper.

| step | what | where |
|---|---|---|
| 1 data | `gen_shapes.py --if-stale`. The set is regenerated unless its `shapes.json` has the same generator + model-source sha256, commit and dirty state, and its files verify. | `v2/build/shapes/<seed>/` |
| 2 build | `tb_shapes.sv` + RTL, compiled once per commit/source hash and simulator. | `v2/build/shapes/builds/<tag>/<sim>/` |
| 3 xval | Verilator runs only. The set's xsim cross-check jobs (`hex/xcheck.hex`: per category the 2 cheapest + the median job, 39 in the default set) run on **both** xsim and Verilator. Every RESULT field must be identical and every job exact, otherwise the run is **refused**. The result is cached per commit + shape set. | `<set>/runs/<key>/xval/` |
| 4 shards | Jobs `[0, K)` in shards, P processes. A shard is skipped on rerun if its log is complete (`collect_shapes.py check-shard`). | `<set>/runs/<key>/<sim>/shard_a_b/run.log` |
| 5 collect | `collect_shapes.py collect`: one row per job via `common.write_results_csv`, per-shard CSV, compressed log archive (sha256 in the CSV). | see above |

Resources: xsim and Verilator runs are not Vivado jobs. The script still prints `free -h`,
refuses to start when less than 8 GB is available or while a `vivado` process runs, and caps P
at min(8, nproc − 4, (avail − 8 GB)/per-process MB).

**Measured runtime** (laptop, dirty-tree validation, all 300 jobs):
- Verilator shards: 4.7 s wall at P = 8, about 26 s summed. The default set has 16.4 M model
  cycles; Verilator runs about 0.5–0.7 M cycles/s per process.
- xval of 39 jobs on xsim: about 25 s.
- Generation: about 10 s.
- The whole command: about 53 s, with builds cached.
- A cold Verilator build takes about 5 s; add the xsim compile on a fresh commit.

**Expected final run: about 1–2 min.**

## Shape set (`gen_shapes.py`, label: model)

Structures are sampled per category and validated before any data is made:
- **Descriptors:** `gos_pack.derive_fields` / `encode_descriptor` (host-side widths and
  consistency).
- **Checker:** `check_descriptor` must accept every valid layer, and `job_err_code` must return 0
  for every valid job.
- **Chaining:** layer l's output map is layer l+1's input; `in_sel` alternates 0, 1, 0, …; the
  layers are packed consecutively in WGT and QPARAM from a random base.
- **Envelope limits:**
  - ACT depth ≤ 4096 words for the input and every stored output;
  - WGT ≤ 16384 words, QPARAM ≤ 256 channels;
  - K ≥ 8, and K ≤ 1800 (V_MUL_W: K·16384 + |q_bias| < 2^25);
  - KH, KW ≤ 8;
  - pool only with even OH/OW;
  - `out_raw` only on the last layer, with OH = OW = 1, OC ≤ 16, and no relu/pool.
- **Harness caps (sim time):** T·K ≤ 150 k per layer and 400 k per job. Category `large` uses
  800 k / 1.2 M per job, with at least 100 k.

Per job:
- **Inputs and weights:** random int8 inputs (30 % of jobs use nonnegative "ReLU-like" inputs)
  and random int8 weights.
- **Requant parameters:**
  - `q_bias` = U(−1.5, 1.5) × std(acc) of that layer, clipped to the V_MUL_W bound;
  - (m, s) come from `requant_check.select_m_s(M, B=32)` with M = 10^U(−0.45, 0.45) · 48 / std(v).
    They are calibrated on the layer's own accumulators, so outputs are neither all-zero nor
    saturated (s ∈ 1..63, m ∈ [2^31, 2^32)).
- **Golden output:** `gos_golden.gos_layer`, chained.
- **Cycles:** `gos_cycle_model.net_cycles`.
- **Tile-model check:** every job is also re-run through the address-level
  `gos_tile_model.run_layer_mem` on garbage-filled memories, and it must equal the golden.

**Refused jobs.** Their descriptors still pass the host-side field checks, but the RTL checker
must refuse them. Each job has one mode, which cycles over the refusal rules the host can
produce "naturally":

- pool with odd OH (rule 1) or odd OW (rule 2);
- K < 8 (rule 3);
- WGT_END (rule 4), IN_END (rule 5), OUT_END (rule 6) or QP_END (rule 7) out of range;
- out_raw with OC > 16 (rule 27);
- N_LAYERS = 0 or 9 (rule 32).

The bad layer is the last layer of a short valid chain, so ERR_CODE also carries layer indices
> 0. The expected ERR_CODE comes from `gos_pack.job_err_code`. For the refusal cycles, the model
counts busy only during the config check: TOTAL_CYC = C_START + C_DONE and MAC_ACTIVE = 0.

Default set (seed 20260929, 300 jobs):

| category | jobs | what |
|---|---|---|
| conv | 26 | 1 layer, K×K / KH×KW conv (kernel > 1×1), no pool |
| conv_pool | 30 | 1 layer conv + fused 2×2/2 max pool |
| pointwise | 20 | 1 layer 1×1 conv on a spatial map (some pooled) |
| fc | 24 | 1–3 FC layers on 1×1 maps; last one out_raw when OC ≤ 16 |
| head_raw | 20 | 1 layer out_raw: global conv to 1×1 → LOGIT |
| tails | 30 | 1–2 layers, OC % 8 ≠ 0 and OW % 8 ≠ 0 forced, narrow maps (W ≤ 23) |
| tiny | 18 | minimal K (8..16) or 1×1 outputs (non-raw) |
| depth2_3 / depth4_6 / depth7_8 | 32 / 36 / 24 | random chains (an out_raw head is attempted with p = 0.45) |
| netlike | 14 | conv+pool, conv+pool, conv→1×1, 0–2 FC, FC raw (LeNet/CIFAR-like) |
| large | 12 | 1–2 big layers, T·K 1e5..8e5 per layer |
| refuse | 14 | **expected refusal** (all 10 modes above) |

`shapes.json` records per-job coverage flags: `out_raw`, `has_pool`, `has_oc_tail`,
`has_x_tail`, `has_1x1_out`. It also records the n_layers histogram, the sampler tries and the
model cycle totals.

**Envelope finding (not an RTL change).** The conflict-free read
`rd[b] = rowbase + ox0/8 + (b < kx)` with `rot = kx[2:0]` is exact only for kx ≤ 8, that is
KW ≤ 9. The config checker does not test KW. With KW = 10 and 12, `gos_tile_model` differs from
`gos_golden`, and the RTL implements the same formula. The generator therefore stays at KW ≤ 8.
FORMATS.md does not document the limit; it is an open item for DECISIONS.md.

## Set format (`v2/build/shapes/<seed>/`)

- `shapes.json` holds:
  - `format`, `seed`, `n_jobs`;
  - `generator_sha256`, `model_sources_sha256`, `git_commit`, `git_dirty`;
  - `limits`, `plan`, `categories`, `coverage`, `xcheck_jobs`;
  - `files` (sha256 + bytes of every `jobs/*.npz`);
  - `jobs` (per-job summary);
  - `shapeset_sha256`: sha256 over the sorted (file, sha256) list of the job files. The npz
    files are written deterministically (fixed zip timestamps), so the same seed and model code
    give the same value.
- `jobs/J0000.npz` holds:
  - `desc` uint32 [nd,16], `n_layers`;
  - `wgt` uint64 (at WGT word `wgt_base`);
  - `qparam` uint64 (combined words at `2*qp_base`);
  - `act_in` uint64 (ACT0 words 0..IN_END);
  - `out_buf`, `out_raw`, `out_expected` uint64, `out_mask` uint64 (byte-enable bits 0..255 per
    word);
  - `logits_expected` int32[16], `n_logits`;
  - `model_layer_cycles`, `model_T`, `model_total`, `model_mac`;
  - `category`, `expect_refuse`, `err_code`, `refuse_mode`;
  - `shape`, `fields` (+ `field_names`), `tk_total`, `macs`.
- `hex/` holds the TB inputs: `jobs.hex`, `xcheck.hex`, and per job
  `J0000/{meta,desc,wgt,qparam,act_in,exp_cyc,out_exp,out_mask,logit}.hex`.

**Loader** (numpy only, deployable), `v2/board/shapeset.py`:
- `verify_manifest(dir) -> dict` checks the format, sha256 + size of every listed file, that
  there are no unlisted npz, and `shapeset_sha256`. It raises `ShapesetError`.
- `load_shapeset(dir, verify=True, jobs=None) -> list[Shape]`.
- `shapeset_sha256(manifest)`.
- `Shape` fields are as in the npz, plus:
  - `id` (= job), `qp_word_base`, `out_mask64`, `layer(l)`;
  - `compare_output(words) -> (ok, n_bad)`;
  - `compare_logits(logits16)`;
  - `compare_cycles(layer_cyc, total, mac)`.

## Testbench `tb_shapes.sv` (drives `gos_core`, PS_RD_LAT = 1; simulator-portable)

Setup: one reset, then every memory is filled with random garbage. Nothing is cleared between
jobs, so earlier jobs' data stays behind as garbage.

Per valid job, the TB:
1. Loads WGT at `wgt_base`, QPARAM at `2*QP_BASE`, and ACT0.
2. Writes DESC slots 0..nd−1. The other slots get `$urandom`.
3. Pulses start and waits for `!busy`. On timeout it issues a soft_reset and marks the job bad.
4. Checks `done && !error`.
5. Checks the output:
   - out_raw jobs: LOGIT[0..OC−1] must equal the golden, and LOGIT[OC..15] must be 0 (LOGIT is
     cleared at start, DECISIONS).
   - Other jobs: ACT[out_buf] words 0..OUT_END must match under the byte mask, and LOGIT must be
     all 0.
6. Checks LAYER_CYC/TOTAL/MAC against the model, STALL = 0, and sticky flags = 0.

Refuse jobs: the TB writes DESC + N_LAYERS and pulses start. It then checks:
- the job is refused within 64 cycles: `!busy`, `error`, and ERR_CODE equal to the expected code;
- TOTAL_CYC and MAC_ACTIVE equal the model.

Each job prints one `RESULT kind=shape …` line. It includes `out_sig`, a 64-bit FNV-style fold
of the masked output words, which the collector recomputes from the npz. The run ends with
`SHARD_DONE` and then `TEST PASSED` / `TEST FAILED`.

Selection: `+DATA=<set>/hex` plus either `+JOB_START/+JOB_END` or `+JOBLIST=<hex>`.

**xsim notes.**
- A string ternary inside `$display` crashes the xsim 2023.1 kernel, so the strings are prepared
  before the call.
- `$readmemh` into automatic arrays is avoided.

## Collector `collect_shapes.py` (re-checks every job independently from the npz)

Valid jobs pass when all of these hold:
- status: done, no error, ERR_CODE = 0, no timeout;
- outputs:
  - ACT: the TB compare passes, `out_sig` equals the value recomputed from
    `out_expected & mask`, and `n_bad = 0`;
  - raw: the logits equal the npz values and the unused LOGIT are 0;
- cycles: the layer list, total and MAC equal the npz model values, and STALL = 0;
- sticky flags = 0.

Refuse jobs pass when all of these hold:
- error is set and ERR_CODE equals the expected code;
- the refusal cycles equal the model.

Other subcommands: `check-shard` (completeness: SHARD_DONE, exactly one RESULT per selected job),
`xval` (two simulators: identical fields, all exact, at least `--min-jobs` jobs), and `archive`.

CSV columns (after the metadata; `net=shapes`, `layer=J0000`):

| group | columns |
|---|---|
| job and shape | `job_id, category, n_layers, expect_refuse, refuse_mode, shape, in_shape, out_shape, out_raw, has_pool, has_oc_tail, has_x_tail, has_1x1_out, K_list, T_list, T_total, tk_total, macs` |
| model cycles | `model_layer_cycles, model_total, model_mac` |
| RTL cycles and errors | `rtl_layer_cycles, rtl_total, rtl_mac, rtl_stall, rtl_flags, total_err, abs_total_err, max_abs_layer_err` |
| refusal | `err_code_expected, err_code_rtl` |
| verdicts | `err_ok, outputs_match, cycles_exact, all_ok, timeout` |
| simulator | `simulator, simulator_version, xsim_crosschecked` |
| provenance | `seed, shapeset_sha256, data_git_commit, data_git_dirty, shard, run_dir, logs_archive, logs_archive_sha256` |

**Paper contract for `hw_shapes.csv`** (read with column aliases; the first present column wins):

| quantity | column(s) |
|---|---|
| job id | `job_id` |
| refuse flag | `expect_refuse` |
| totals | `model_total`, `hw_total` |
| layer lists | `model_layer_cycles`, `hw_layer_cycles` |
| outputs exact | `output_exact` or `outputs_match` |
| cycles exact | `cycles_eq_model` or `cycles_exact` |
| refusal ok | `refuse_ok` or `err_ok` |
| all ok | `pass` or `all_ok` |
| shape-set id | `shapeset_sha256`; the paper checks it equals the RTL CSV's |

Rows with `layer=summary` or `label=summary` are ignored. For each job id, the latest row is
used.

## Tests

    .venv/bin/python -m pytest v2/shapes/tests v2/paper/tests/test_tables_extra.py -q -p no:cacheprovider

The tests cover:
- **Generator:**
  - determinism: same seed gives the same sha256 and files; another seed differs;
  - every limit, the checker validity of each descriptor, chaining and bases;
  - requant parameter ranges;
  - model cycles recomputed from the descriptors;
  - an independent round trip: the npz WGT/QPARAM/ACT images are unpacked and re-run through
    `gos_golden`, and the result must equal the expected outputs;
  - refusal codes and modes.
- **Loader:** round trip, masks and compare helpers, tamper detection, and an AST check that it
  imports numpy only.
- **Collector** (synthetic logs):
  - one test per mismatch kind: signature, n_bad, logits, unused LOGIT, cycles, layer list, MAC,
    STALL, flags, timeout, error, refusal code, refusal cycles;
  - shard completeness (missing, duplicate, truncated, list mode);
  - end-to-end collect with a mismatch and a missing job;
  - xval identity and the minimum job count.
- **Paper:** RTL-only, RTL + board, and rejected/inconsistent rows.

## Boundary cases (`gen_limits.py`, `run_limits.sh`, `limits_rtl.csv`)

The random set stays inside the envelope (`gen_shapes.LIMITS`). The boundary-case set tests
descriptor values at and beyond it, e.g. KW = 8, 9, 10, 11, 12, 16 and KH = 9, 10, 16
(DECISIONS OC-3: the ACT read `rd[b] = rowbase + ox0/8 + (b < kx)`, `rot = kx[2:0]` is exact
only for KW <= 9 by analysis, and the checker has no KW bound).

- **Cases:** every module `limit_cases_*.py` exposes `CASES: list[dict]`. The schema is
  `gen_limits.CASE_SCHEMA`: `id`, `field`, `value`, `layers` (`gen_shapes.layer(...)` dicts),
  `expect` (`exact | mismatch | refuse | unknown`), `note`, and optionally `err_code` and
  `overrides` (`wgt_base`, `qp_base`, `n_layers`, `in_sel`, `desc_patch`, `raw_words`).
  `expect` is a prediction only. The RTL result is recorded in `observed`.
  `limit_cases_kw.py` holds the KW/KH cases.
- **Set:** `gen_limits.py gen [--cases kw,...]` writes `v2/build/limits/<all|cases>-s<seed>/`
  in the gen_shapes format (`shapes.json` with `"kind": "limits"` and per-case model
  predictions, plus `jobs/*.npz` and `hex/`). No envelope rejects are applied. Descriptors go
  through `derive_fields`/`encode_descriptor`, then any patches are applied. Per case it
  records these predictions without asserting them: the golden output, the checker verdict
  (`job_err_code`), the model cycles, and `tile_model_matches_golden` / `tile_model_n_bad`.
  The tile model runs on garbage memories loaded like the TB loads them.
  Cases the checker model refuses become refuse jobs.
- **Run:** `v2/scripts/run_limits.sh [--cases kw] [--allow-dirty]` runs every case on both
  **Verilator and xsim**, with one simulator process per case under `timeout` (default
  1800 s), so a hang cannot stop the other cases. The TB also has its own timeout and soft
  reset. The builds are cached in `v2/build/limits/builds/`.
- **Collect:** `gen_limits.py collect` writes one row per case (source `rtl_sim`, label RTL sim)
  with these columns:
  - `case_id, field, value, expect`
  - `observed` (`exact | mismatch | refuse | timeout | no_result | disagree`)
  - `outputs_match, cycles_exact, err_code_expected, err_code_rtl`
  - `tile_model_matches_golden, checker_accepts`
  - `sim_agree`. Every control field (done, error, ERR_CODE, LAYER_CYC/TOTAL_CYC/MAC_ACTIVE,
    STALL, flags, timeout) must be identical between Verilator and xsim. Output *values*
    (`out_sig`, `n_bad`, `out_ok`, `logits`) may differ only when both simulators observe the
    same `mismatch` or `timeout`. In that case the wrong output reads memory that tb_shapes
    filled with `$urandom` garbage, and that sequence is simulator-specific.
    `sim_agree_strict`, `outputs_identical` and `sim_diff_keys` record the exact comparison.
  - `simulators, n_bad, note`
  - per-simulator verdicts, cycles and the set sha256
  
  The output goes to `v2/results/limits_rtl.csv`, plus `limits_rtl_logs_<key>.tar.xz` with
  its sha256 in the CSV, only from a clean tree with all case modules and the default seed.
  Otherwise it goes to the run dir. The script exits 0 when every case has a result from both
  simulators and they agree. RTL mismatches are data, not failures. `check_results.py` covers
  the CSV, with `v2/shapes` as an extra source.
- **Tests:** `tests/test_limits.py` covers the loader (every module is picked up, ids are
  unique, the schema is enforced), determinism, the recorded predictions, the overrides, and
  the collector verdicts: exact, mismatch, refuse, timeout, no_result, simulator
  disagreement.
