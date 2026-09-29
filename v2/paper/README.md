# paper — Paper drafts, figures and tables generated from v2/results.

## Step 10 — table/figure pipeline

```bash
.venv/bin/python v2/paper/scripts/make_all.py              # -> v2/paper/generated/ (gitignored)
.venv/bin/python -m pytest v2/paper/tests -q               # pipeline tests (temp copies of the CSVs)
.venv/bin/python v2/paper/scripts/make_all.py --dryrun     # LAYOUT TEST ONLY -> generated/dryrun/, watermarked
```
Options: `--results-dir`, `--out-dir`, `--no-booktabs` (plain `\hline` tables). Exit code 1 if any artifact failed.

Rules enforced by the code (`scripts/paperlib.py`):
- Every number comes from `v2/results/*.csv` (or is computed from such numbers) and is registered with
  its origin (file:line:column or formula); a table containing an unregistered numeric token is refused
  (`ProvenanceError`). Only names (LeNet-5, INT8, pl_clk0, …, `STATIC_TEXT`) may contain digits.
- Rows with `git_dirty != False` (or `data_git_dirty` True) are rejected; `cifar10_retrain_log.csv` exempt (D9).
- Board data only from `hw_*.csv` rows with `source=hw` (`cpu_board` for the CPU baseline) and, if the row has a
  `paper_grade` column (board runs through `run_sessions.py`), `paper_grade=True` (environment pre-flight passed:
  governor fixed, process pinned, no package manager; EXPERIMENTS.md "Measurement rigor"). Missing →
  `TBD (board)` cells / hatched "board data pending" series; switches to real data automatically.
- `v2/results/dryrun/` is read only with `--dryrun`; every output is then watermarked "DRY RUN -- NOT DATA".
- `generated/MANIFEST.json`: per artifact the input CSVs (sha256, whether they equal git HEAD), rows used
  (lines), row `git_commit`s, D16 staleness of those commits, sources, placeholders, consistency checks and
  every registered number with its origin. Figures also get `<name>.data.csv` with the plotted values.

| output | content | sources |
|---|---|---|
| `tab_t1_impl.tex` | A6/T1: LUT/FF/LUTRAM/DSP/RAMB, gos_core share, WNS/WHS, pl_clk0 req./actual, BUILD_ID, bitstream SHA-256 prefix; one column per clock that met timing (D14) — a 300 MHz row appears automatically | impl_gos.csv (post_impl) |
| `tab_a1_accuracy.tex` | A1: FP32/INT8 accuracy of record, RTL bit-exact images, KV260 mismatches/accuracy | reference_accuracy.csv (model; r2 cross-checked with cifar10_r2_accuracy.csv), rtl_network.csv (rtl_sim), hw_a1_accuracy.csv (hw) |
| `tab_a2_latency.tex` | A2: total cycles model/RTL/KV260, µs at each post-impl clock (RTL cycles / f, computed), KV260 PL µs at the measured clock f_meas, f_meas, read-back clock, wall-clock median [95% CI] | cycle_model, rtl_network, impl_gos, hw_a2_a3_cycles |
| `tab_a3_cycles.tex`, `fig_a3_cycles{,_wide}.pdf` | A3: per-layer model/RTL/KV260 cycles + error % vs model | cycle_model, rtl_network (+ rtl_cycles cross-check), hw_a2_a3_cycles |
| `tab_a4_util.tex`, `fig_a4_util.pdf` | A4: MAC_ACTIVE/cycles (model, RTL, KV260), theoretical PE utilization | cycle_model, rtl_cycles, rtl_network, hw_a4_util |
| `tab_a5_cpu.tex` | A5: A53 CPU baselines (compute / e2e, per thread count) vs accelerator, median [95% CI] where the CI columns exist | hw_cpu_baseline (cpu_board), hw_b3_breakdown |
| `tab_b1_power.tex` | B1 "SOM-rail power (INA260)": P_idle, ΔP accel, ΔP CPU, time/image, energy/image accel vs CPU (mean ± std over repeats), repeats, achieved sample rate, CPU workload; E_comp / t_PL at f_meas; the INA260 is the only power source | `hw_b1_power_ina260_summary*.csv` (power_log.py; any tag, net/clock read from the rows; `_sensorcheck` ignored) |
| `tab_b2_clock.tex`, `fig_b2_clock.pdf` | B2 per clock: INA260 ΔP, energy/image, time/image (mean ± std) + PL latency; figure = latency / ΔP / energy vs pl_clk0 (RTL cycles / post-impl clock also plotted, labeled) | `hw_b2_power_ina260_summary*.csv`, `hw_b2_clock.csv` (clocks matched within 0.5 MHz) |
| `tab_b3_breakdown.tex`, `fig_b3_breakdown.pdf` | B3 host-side phases, safe vs fast (interleaved), median [95% CI] + p95 | hw_b3_breakdown, hw_b3_breakdown_fast |
| `tab_repeatability.tex` | 3-session repeatability: mean ± between-session SD (n sessions) of f_meas, A2 wall-clock, B3 end-to-end, B1 ΔP / E_sys / E_comp; `TBD (board)` until aggregated | hw_repeatability (board/aggregate_sessions.py) |
| extras | `tables_extra.make(ctx, failures)` if `scripts/tables_extra.py` exists (soak, B2 fit, layer spread; other owner) | see tables_extra.py |
| `tab_verification.tex` | generic render of verification_stats.csv (placeholder if absent) | verification_stats.csv |

Tables are complete `table` environments (IEEEtran, `\footnotesize`, booktabs); figures are 3.5 in
(single column) or 7.16 in (`_wide`) wide, 8-pt STIX serif, grayscale + hatching, TrueType-embedded fonts.
