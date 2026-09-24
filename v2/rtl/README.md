# rtl — Synthesizable SystemVerilog for the V2 generalized output-stationary accelerator (module prefix gos_)

Contract: `v2/docs/ARCH_SPEC.md`, `v2/docs/FORMATS.md`, `v2/docs/DECISIONS.md` (D8, D10, D11 binding).
Shared parameters and the integration token: `gos_pkg.sv` (compile it first).

## Conventions (every module)

- `` `timescale 1ns/1ps ``; single clock `clk`; synchronous, active-high `rst`; synthesizable SystemVerilog only
  (`always_ff` / `always_comb`, no `initial`, no `#` delays, explicit widths and signedness).
- **Token pipeline.** Every pipelined module has `in_valid` + `in_tok[TW-1:0]` and produces `out_valid` + `out_tok`
  with EXACTLY the same fixed latency as its data. `TW` is a parameter. The latency is a localparam `L_<MODULE>`
  documented in the module header. No module infers data timing by counting cycles; control travels with the data.
  Only `valid` (and sticky error flags) are reset; data/token registers need no reset.
- **BRAMs** use the UG901 inference templates (true dual port where two ports are needed, byte-write enables where
  needed, `(* ram_style = "block" *)`). The accelerator-side read latency is fixed at 2 (output register,
  `gos_pkg::L_MEM_ACC`). PS-side ports have a parameter `PS_RD_LAT` ∈ {1, 2}, default 1 (no output register; the
  AXI BRAM Controller expects latency 1 by default). No XPM.
- **Multipliers** are written so Vivado infers DSP48E2 with pipeline registers (AREG/BREG, MREG, PREG). Small
  operands (8×8) need `(* use_dsp = "yes" *)` — without it Vivado maps them to LUTs (legacy recon, PROJECT_STATE).
- **No runtime multipliers in address generation** (counters / incremental adds only).
- Sticky error flags (e.g. `err_overrun`) are cleared only by `rst`.

## Tests and synthesis

- Every module has a self-checking TB `v2/tb/tb_<module>.sv` that prints `TEST PASSED checks=<n>` or
  `TEST FAILED ...` and then calls `$fatal(1, "<message>")` — always with a message: xsim 2023.1 silently
  ignores a bare `$fatal(1);`, and xsim exits 0 even on `$fatal`, so `run_xsim.sh` decides PASS/FAIL from the log
  ("TEST PASSED" present, no ERROR/FATAL/"TEST FAILED"). The first line of each TB lists its sources:
  `// GOS_UNIT_TB: gos_pkg.sv <module>.sv [...]` (paths relative to `v2/rtl/`), read by `v2/scripts/run_unit_all.sh`.
- Vectors: `$value$plusargs("VEC_DIR=%s", ...)` → `v2/vectors/generated` (passed by `v2/scripts/run_xsim.sh`).
- Simulations run in `v2/build/sim/<tb_name>/`; OOC synthesis in `v2/build/ooc/<top>/` (`v2/scripts/ooc_all.sh`).
