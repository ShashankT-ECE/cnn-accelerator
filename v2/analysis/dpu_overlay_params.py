"""DPU overlay metadata (pynq-dpu 2.5 KV260 prebuilt overlay): what its .hwh states about the DPU.

Source file: the overlay's hardware handoff file `pynqdpu.dpu.kv260_som.2.5.0.hwh`
(= `pynq_dpu/dpu.hwh` on the board, md5 dc81379b8d742b8ac6700f2bc45d6fd1, checked against
`overlay_hwh_md5_expected` in v2/dpu/build/package/DPU_INFO.json), read from
v2/dpu/build/pynq_dpu_2.5/kv260/ (v2/dpu/fetch_overlay_ref.sh). The xclbin (`dpu.xclbin`, built by
v++ 2022.1) carries the same kernel/memory metadata and no resource counts either.

Rows (source = overlay_hwh, never a measurement):
  kind hwh_parameter  every PARAMETER of the DPUCZDX8G_1 module (name, value verbatim)
  kind hwh_clock      the module's clock ports with their CLKFREQUENCY
  kind derived        ARCH_PP * ARCH_ICP * ARCH_OCP = MAC lanes (INT8 MACs per aclk cycle of the
                      convolution engine), ops/cycle = 2 * MAC lanes; the formula is in `note`
  kind unavailable    DSP / LUT / FF / BRAM / URAM counts: the .hwh and .xclbin carry no
                      utilization report, so no resource count of the DPU can be read from them
                      (DPUCZDX8G_ISA1_B4096 = 4096 ops/cycle matches the derived value)

Run:  .venv/bin/python v2/analysis/dpu_overlay_params.py [--hwh PATH] [--out-dir DIR]
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import _setup

from common import base_meta, write_results_csv

CSV_NAME = "dpu_overlay_params.csv"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_HWH = REPO / "v2/dpu/build/pynq_dpu_2.5/kv260/pynqdpu.dpu.kv260_som.2.5.0.hwh"
DPU_INFO = REPO / "v2/dpu/build/package/DPU_INFO.json"
MODULE = "DPUCZDX8G_1"
FIELDS = ["kind", "name", "value", "unit", "note", "hwh_file", "hwh_md5", "hwh_sha256"]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract(hwh: Path) -> list[dict]:
    root = ET.parse(hwh).getroot()
    mod = next(m for m in root.iter("MODULE") if m.get("INSTANCE") == MODULE)
    params = {p.get("NAME"): p.get("VALUE") for p in mod.find("PARAMETERS").findall("PARAMETER")}
    rows = [{"kind": "hwh_parameter", "name": k, "value": v} for k, v in params.items()]
    for port in mod.iter("PORT"):
        if port.get("CLKFREQUENCY"):
            rows.append({"kind": "hwh_clock", "name": port.get("NAME"),
                         "value": f"{float(port.get('CLKFREQUENCY')) / 1e6:g}", "unit": "MHz",
                         "note": f"PORT CLKFREQUENCY of {MODULE}, net {port.get('SIGNAME')}"})
    pp, icp, ocp = (int(params[k]) for k in ("ARCH_PP", "ARCH_ICP", "ARCH_OCP"))
    lanes = pp * icp * ocp
    rows += [
        {"kind": "derived", "name": "mac_lanes", "value": str(lanes), "unit": "INT8 MAC/aclk cycle",
         "note": "ARCH_PP * ARCH_ICP * ARCH_OCP (pixel x input-channel x output-channel parallelism)"},
        {"kind": "derived", "name": "ops_per_cycle", "value": str(2 * lanes), "unit": "op/aclk cycle",
         "note": "2 * mac_lanes (1 MAC = 2 ops); DPUCZDX8G_ISA1_B4096 = 4096 op/cycle"},
    ]
    for res in ("DSP48E2", "LUT", "FF", "BRAM36", "URAM"):
        rows.append({"kind": "unavailable", "name": f"resource_{res}", "value": "",
                     "note": "no utilization report in the .hwh / .xclbin of the prebuilt overlay"})
    return rows


def main(argv=None) -> int:
    ap = _setup.out_dir_parser(__doc__.split("\n")[0])
    ap.add_argument("--hwh", type=Path, default=DEFAULT_HWH)
    a = ap.parse_args(argv)
    exp = json.loads(DPU_INFO.read_text()).get("overlay_hwh_md5_expected") if DPU_INFO.exists() else None
    got = md5(a.hwh)
    if exp and got != exp:
        raise SystemExit(f"hwh md5 {got} != overlay_hwh_md5_expected {exp} (DPU_INFO.json)")
    rel = str(a.hwh.resolve().relative_to(REPO)) if a.hwh.resolve().is_relative_to(REPO) else str(a.hwh)
    meta = {"hwh_file": rel, "hwh_md5": got, "hwh_sha256": sha256(a.hwh)}
    rows = [{**base_meta("", "dpu_overlay", source="overlay_hwh"), **meta, **r} for r in extract(a.hwh)]
    p = write_results_csv(a.out_dir / CSV_NAME, rows, FIELDS)
    lanes = next(r["value"] for r in rows if r["name"] == "mac_lanes")
    print(f"wrote {p} ({len(rows)} rows); MAC lanes {lanes}; md5 {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
