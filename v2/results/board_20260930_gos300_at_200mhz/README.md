# Board Session 1 of 2026-09-30 — gos_300 bitstream at 200 MHz (archived, NOT a 300 MHz result)

Measured on the KV260 (scripts / data commit 11aa4f0, bitstream gos_300 build fd880d43,
sha256 2d40e2c8…). Every Session 1 check passed, but pl_clk0 read back **199.998 MHz**: PYNQ does
not program the Vivado PS PLL settings and the session only required read-back ≤ closed clock
(DECISIONS D19). Nothing here may be reported as a 300 MHz measurement, and no row of this
directory is paper data (`check_results.py` reads only `v2/results/*.csv`).

| file | what |
|---|---|
| `hw_clock_choice.json` | **corrected** clock choice: `ok=false`, `ran_at_closed_clock=false`, read-back clock from `session_state.json`; written by `correct_clock_choice.py` (no typed numbers) |
| `hw_clock_choice.original.json` | the file as the board wrote it ("299.997 MHz passed the smoke test") |
| `session_state.json` | session state: provenance (`clock_readback_mhz` 199.998, `closed_clock_mhz` 299.997009), steps, environment |
| `hw_fclk_cal_s1.csv`, `.npz` | PL clock calibration with the former median estimator: 201.992 MHz at a 199.998 MHz read-back (+1.0 %); raw per-job times in the `.npz` (cause and fix: DECISIONS D20) |
| `hw_b1_power_ina260_{samples,phases}_sensorcheck.csv` | `power_log.py --sample-only --seconds 20` before the session: INA260 hwmon `ina260_u14`, 200 samples at 10 Hz, idle SOM-rail power (before the session loaded any overlay; PL state at that time not recorded) |
| `raw_board_results.tar.xz` | verbatim copy of `~/gos/results/` after the session (all of the above as written by the board, step logs, orchestrator logs, `env_log.jsonl`, the pre-state archive); every file's SHA256 was checked against the board after the copy |
