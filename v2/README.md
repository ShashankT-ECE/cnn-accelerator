# V2 — Generalized INT8 8×8 Output-Stationary Accelerator (KV260)

A small INT8 output-stationary CNN accelerator for the AMD Kria KV260
(`xck26-sfvc784-2LV-c`, Vivado 2023.1). It runs LeNet-5 and CIFAR-10
end to end, bit-exact against a frozen reference, and targets cycle-exact
agreement between the analytical model, RTL simulation and the board.

- Branch: `v2-dev`. The pre-V2 state is preserved on `v1-snapshot` / tag `v1-baseline`.
- Rules for working here: [`CLAUDE.md`](CLAUDE.md)
- Frozen architecture: [`docs/ARCH_SPEC.md`](docs/ARCH_SPEC.md)
- Decisions and open conflicts: [`docs/DECISIONS.md`](docs/DECISIONS.md)
- Experiment plan: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
- Pre-V2 repo recon: [`docs/RECON_2026-09-24.md`](docs/RECON_2026-09-24.md)

| Folder | Purpose |
|---|---|
| `model/` | Python golden model, requant equivalence check, cycle model, vector generation |
| `rtl/` | Synthesizable SystemVerilog (`gos_` prefix) |
| `tb/` | Self-checking xsim testbenches |
| `vectors/` | `$readmemh` hex vectors (`generated/` ignored) |
| `scripts/` | Build, sim and results scripts |
| `vivado/` | Tcl, XDC, block design |
| `board/` | KV260 PYNQ overlay and measurement code |
| `results/` | Script-generated CSVs (`raw/` ignored) |
| `paper/` | Paper drafts and figures |
| `build/` | Generated outputs (ignored) |

Legacy files outside `v2/` (`rtl/`, `sim/`, `python/`, `data/`, `docs/`,
`scripts/`, `software/`) are read-only references.
