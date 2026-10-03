# v3/feas — Phase 0 feasibility (EXPLORATORY, not design RTL)

WS5 of the V3 Phase 0 plan. Nothing here is reused in the design without review (v3/CLAUDE.md).

- `rtl/feas_row.sv` — one OS PE row of W columns + activation broadcast; PACK=0 (1 MAC/DSP, accumulate
  in the DSP, V2 PE template) or PACK=1 (2 MACs/DSP sharing the activation, pre-adder packing,
  fabric accumulators).
- `rtl/feas_mem_path.sv` — URAM weight bank -> broadcast replicas; W-bank BRAM ACT read + rotator ->
  broadcast replicas; residual-buffer read -> skip term in the accumulator domain -> V2 requant lanes.
- `tb/` — self-checking xsim TBs (PASS/FAIL, $fatal on failure).
- `packing_model.py` — exhaustive INT8 packing exactness + in-DSP accumulation limit -> `feas_packing.csv`.
- `run_feas_tb.sh` + `tb_collect.py` -> `feas_tb.csv` (RTL sim).
- `run_feas_ooc.sh` (one Vivado job at a time, guard before each) + `ooc_collect.py` ->
  `feas_ooc.csv` (OOC synth at 4.000 / 3.000 ns, V3 D2) and `device_resources.csv` (Vivado part query).

Producers run from a clean committed tree (a detached worktree of the commit) so rows carry
git_dirty=False.
