# v2/fullsim — full-test-set RTL simulation (all 10,000 test images per net)

Runs every MNIST / CIFAR-10 test image (LeNet-5, CIFAR-10 r2) through `gos_core` in RTL
simulation. Each image gets a bit-exact LOGIT check and an exact cycle check against the frozen
golden model. Output: `v2/results/rtl_full10k.csv`, one row per net, `source=rtl_sim`.
Label every number from it **RTL sim**, never hardware.

RTL, `v2/model`, `v2/vivado` and the existing TBs are not touched. Everything generated
(10k-image vectors, builds, logs) goes under `v2/build/fullsim/`, which is gitignored.

## Entry point

    v2/scripts/run_full10k.sh [--nets lenet5 cifar10] [--sim verilator|xsim]
                              [--images-per-shard M | --shards N] [--limit K] [--max-par P]
                              [--force] [--allow-dirty] [--no-xval] [--ignore-vivado]

Defaults: both nets, `--sim verilator`, 500 images per shard, K = 10000, max-par 12.
The run writes `v2/results/rtl_full10k.csv` only if the tree is clean and K = 10000. Otherwise
the CSV goes to `<run dir>/rtl_full10k.csv`. `--allow-dirty` is for harness tests only.

## Pipeline

| step | what | where |
|---|---|---|
| 1 data | `gen_full10k_data.py`: `gos_golden.load_test_set` + `quantize_input`, packed with `gos_pack.pack_act_words`. Golden `gos_golden.run_net` gives the INT32 v, then PS float32 argmax. Net images come from `gos_pack.pack_wgt/pack_qparam/make_descriptors` and cycles from `gos_cycle_model`, the same calls `gen_vectors.py` makes. Self-checks: (a) images 0..9 and wgt/qparam/desc/n_layers/expect_cyc are **byte-identical** to `v2/vectors/generated` (sha256 matches `v2/vectors/MANIFEST.json`); (b) 4 images re-run through `gos_tile_model.run_net_mem`; (c) full-set golden accuracy = accuracy of record (D3). | `v2/build/fullsim/data/<net>/` |
| 2 build | Compiles `tb_full10k.sv` + RTL once per commit and simulator. The key is a hash of TB + RTL + `run_shard.sh`. | `v2/build/fullsim/runs/<commit12>[-dirty-<hash>]/<sim>/build/` |
| 3 xval | Verilator only. Images 0..9 of each net run on **both** xsim and Verilator. Per image, LOGIT[0..15], LAYER_CYC, TOTAL_CYC, MAC_ACTIVE, STALL, logits_ok and cycles_ok must be identical. TOTAL/LAYER_CYC must also equal `v2/results/rtl_network.csv` (xsim `tb_gos_core SUITE=net`). If any check fails, the run refuses to start. The result is cached per source hash. | `runs/<tag>/xval/xval.json` |
| 4 shards | Images [0, K) per net are split into shards and run P at a time. A shard is skipped on rerun when `collect_full10k.py check-shard` says it is COMPLETE (details below). Changing the shard plan discards that net's old shards. | `runs/<tag>/<sim>/<net>/shard_<a>_<b>/run.log` |
| 5 collect | `collect_full10k.py collect` writes one row per net via `common.write_results_csv`. A per-shard CSV is also written. | `v2/results/rtl_full10k.csv`, `runs/<tag>/<sim>/shards.csv` |

A shard is COMPLETE when its log has these lines:
- a `SHARD_DONE` line for exactly this net/range, with `images = end - start`;
- exactly one `RESULT` line for every image in the range.

A complete shard can still contain mismatches. It is not re-run, because the result is
deterministic, and the collector reports it. Use `--force` to rerun everything.

**Parallelism.** P = min(`--max-par`, nproc − 4, (available MB − 8192) / per-process MB), and
at least 1. Per-process MB is 800 for xsim, which covers the measured xsim front-end + kernel
RSS, and 64 for Verilator. The script prints `free -h`. It refuses to start while a `vivado`
process runs (one heavy job at a time).

## Testbench `tb_full10k.sv`

It uses the same procedure as `tb_gos_core SUITE=net`, but over an image range inside one
simulation:
1. Fill all memories with random garbage through the PS ports (PS_RD_LAT = 1).
2. Load WGT, QPARAM and descriptors once.
3. For each image: write ACT0 words 0..IN_END, pulse `start`, then wait for `!busy`.
4. Check `done && !error`.
5. Check LOGIT[0..15] bit-exact against `logit16` (unused = 0).
6. Check LAYER_CYC[0..NL-1], TOTAL_CYC and MAC_ACTIVE against the model, and STALL == 0.

A mismatch does not stop the run. File sizes come from `sizes.hex`, and nothing relies on
4-state `'x`, so the TB runs unchanged on xsim and Verilator. The collector re-checks every
image independently:
- the LOGITs against `golden.npz` v;
- the prediction, computed with `final_layer.predict_from_raw` (PS float32 argmax, D2), against
  the golden prediction.

## Simulators

- **xsim** (Vivado 2023.1) is the simulator of record (root `CLAUDE.md`).
- **Verilator 5.028** was built from source without sudo. Because of the Makefile's man-page
  step (needs `help2man`), `make install` fails, so it is used in-tree:
  `git clone --depth 1 --branch v5.028 https://github.com/verilator/verilator ~/tools/verilator-src`
  then `autoconf && ./configure --prefix=$HOME/tools/verilator && make -j4`. The script uses
  `VERILATOR_ROOT=${VERILATOR_ROOT:-~/tools/verilator-src}`.
  - Build: `--binary --timing -O3`.
  - The RTL compiles unmodified, with no warnings at default lint level.
  - Verilator is 2-state. X-propagation is covered only by the xsim suites; the xval step ties
    the two simulators together on the committed net images.
- Every CSV row carries `simulator` and `simulator_version`. xsim rows also set
  `vivado_version=2023.1`.
- Timing: per-shard seconds are in `shards.csv`. Per-net `wall_s` (first shard start to last
  shard end) and `shard_seconds_sum` are in the CSV.

## Tests

    .venv/bin/python -m pytest v2/fullsim/tests -q -p no:cacheprovider

The tests use synthetic logs to cover shard completeness (truncated, missing, duplicate or
wrong-range output), the resume decision (`check-shard` exit codes), and mismatch detection:
- a logit that differs from golden even when the TB says ok;
- a TB mismatch;
- a cycle mismatch;
- a prediction mismatch;
- a non-zero unused LOGIT;
- incomplete or missing shards.

They also cover the xval comparison and an end-to-end `collect` run.

## Files

`gen_full10k_data.py` (data), `tb_full10k.sv` (TB), `run_shard.sh` (one shard, one process),
`collect_full10k.py` (check-shard / xval / collect), `tests/`, and `v2/scripts/run_full10k.sh`
(entry point).
