# Phase 3 — Board-facing top: Synthesis / Implementation / Timing

> **Status:** VERIFIED (2026-08-21). Vivado ML 2023.1, target
> `xck26-sfvc784-2LV-c` (AMD Kria KV260 / XCK26). Top = `cnn_top` (AXI4-Lite
> slave + result BRAM + `cnn_accelerator_v2`), OOC (out-of-context),
> 200 MHz (5.000 ns) baseline constraint (`build/top/baseline.xdc`).

## 1. Result summary

| Metric | Value |
|---|---|
| Clock | 200 MHz (5.000 ns) on `s_axi_aclk` |
| **WNS (setup)** | **+0.202 ns** (0 failing endpoints) |
| **TNS** | **0.000 ns** |
| **WHS (hold)** | **+0.058 ns** (0 failing endpoints) |
| Pulse width | 0 failing (worst slack +1.738 ns) |
| Unrouted nets | 0 (route completed) |

Adding the AXI4-Lite slave, the 8-bank result BRAM, and the `% 28` result-decode
logic did **not** measurably reduce the timing margin: the hardened
`cnn_accelerator_v2` closed at +0.189 ns; the board-facing `cnn_top` closes at
+0.202 ns. The AXI/result path is off the critical path, which remains inside
the accelerator's FSM → result-emission logic.

## 2. Resource utilization (post-route)

| Resource | cnn_top | hardened accel (no AXI) | Δ |
|---|---|---|---|
| CLB LUTs | 9,462 (8.08%) | 9,372 | +90 |
| — LUT as Logic | 8,812 | — | — |
| — LUT as Memory | 650 (image dist-RAM) | 650 | 0 |
| Flip-flops (FDRE) | 5,903 | 5,795 | +108 |
| DSP48E2 | 64 | 64 | 0 |
| CARRY8 | 24 | 20 | +4 |
| BRAM (RAMB36E2) | 8 | 0 | +8 |
| URAM | 0 | 0 | 0 |

Cost attribution: **+8 BRAM36** = the 8 result banks (one per column residue,
1024×32 each, `ram_style="block"`); **+4 CARRY8** = the result address/decode
arithmetic (`% 28`, bank/dense decode); **+90 LUT / +108 FF** = the AXI4-Lite
slave (5-channel handshake, register file, result read mux) and the
image/weight auto-increment write logic. **DSP48E2 unchanged at 64** — the
result-address arithmetic was kept off the DSPs (shifts/`%28` in LUTs), and all
64 DSPs remain the PE MACs (1/PE, `use_dsp` attribute in the frozen `pe_v2`).

## 3. Critical path

```
Slack (MET) : +0.202 ns
Source      : u_ctrl/u_accel/phase_reg[1]_inv/C          (controller FSM)
Destination : u_ctrl/u_accel/result_data_reg[7][7]/D     (result emission register)
Data Path   : 4.793 ns  (logic 0.875 ns [18.3%]  route 3.918 ns [81.7%])
Logic Levels: 5  (LUT2=1 LUT5=1 LUT6=3)
```

Routing-dominated (81.7%), 5 logic levels — the controller FSM state → result
emission datapath. This is the same class of path as the pre-AXI hardening
report; the AXI slave and result BRAM sit off this path.

## 4. Conclusion

- **200 MHz closes with +0.202 ns** (~208 MHz extrapolated max), unchanged from
  the pre-AXI hardened margin. The research plan's §7 conservative estimate of
  **150 MHz** for the reconfigurable system remains a comfortable fallback if a
  later block-design change (e.g. clocking path) eats margin on the board.
- The result BRAM + AXI slave cost 8 BRAM36 + ~90 LUT + ~108 FF — negligible on
  the KV260 (144 BRAM36, 117k LUT, 234k FF available).

## 5. Reproducibility

```bash
source ~/Xilinx/Vivado/2023.1/settings64.sh
vivado -mode batch -source build/top/synth_impl.tcl
# reports -> build/top/00..11_*.rpt / *.txt
```
