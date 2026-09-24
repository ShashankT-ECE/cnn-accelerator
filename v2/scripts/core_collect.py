#!/usr/bin/env python3
"""Collect tb_gos_core RESULT lines into results CSVs (source=rtl_sim).

Usage: core_collect.py <suite>:<rc>:<seconds> ...   (called by run_core.sh)
  v2/results/rtl_cycles.csv   one row per layer run (kind=layer: net layers x images;
                              kind=fuzz: random shapes) — model vs RTL cycles, output match
  v2/results/rtl_network.csv  one row per network image — LOGIT match, prediction via
                              final_layer (PS float32 dequant + argmax) vs golden, cycles
  v2/results/rtl_checker.csv  one row per config-checker case
Exit 1 if any row fails.
"""
import re
import sys
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "model"))
from common import RESULTS_DIR, base_meta, write_results_csv  # noqa: E402
import final_layer  # noqa: E402
import gos_golden as gg  # noqa: E402

VEC = V2 / "vectors" / "generated"


def parse(log: Path) -> list[dict]:
    out = []
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("RESULT "):
            out.append(dict(kv.split("=", 1) for kv in line.split()[1:]))
    return out


def read_hex(rel: str) -> list[int]:
    return [int(x, 16) for x in (VEC / rel).read_text().split()]


def main() -> int:
    runs = {}
    for a in sys.argv[1:]:
        s, rc, secs = a.split(":")
        log = V2 / "build" / "sim" / f"tb_gos_core_{s}" / "run.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        ver = re.search(r"Vivado Simulator v?(\d{4}\.\d)", text)
        runs[s] = {"rc": int(rc), "secs": float(secs), "rows": parse(log) if log.exists() else [],
                   "passed": int(rc) == 0 and "TEST PASSED" in text,
                   "ver": ver.group(1) if ver else ""}
    ok = all(r["passed"] for r in runs.values())

    # ---- cycles (layer + fuzz) ----
    rows = []
    for s in ("layer", "fuzz"):
        if s not in runs:
            continue
        for r in runs[s]["rows"]:
            net = r.get("net", "fuzz")
            layer = r.get("layer", f"F{int(r.get('case', 0)):03d}")
            m = base_meta(net=net, layer=layer, source="rtl_sim", duration_s=runs[s]["secs"],
                          num_inferences=1)
            m["vivado_version"] = runs[s]["ver"]
            mc, rcyc = int(r["model_cycles"]), int(r["rtl_cycles"])
            row = {**m, "kind": s, "img": r.get("img", ""), "model_cycles": mc, "rtl_cycles": rcyc,
                   "err": rcyc - mc, "model_total": int(r["model_total"]), "rtl_total": int(r["rtl_total"]),
                   "model_mac": int(r["model_mac"]), "rtl_mac": int(r["mac"]), "stall": int(r["stall"]),
                   "out_ok": r["out_ok"] == "1", "issue_ok": r.get("issue_ok", ""),
                   "write_ok": r.get("write_ok", "")}
            ok &= row["err"] == 0 and row["model_total"] == row["rtl_total"] and row["out_ok"] \
                and row["stall"] == 0 and row["model_mac"] == row["rtl_mac"]
            rows.append(row)
    if rows:
        write_results_csv(RESULTS_DIR / "rtl_cycles.csv", rows,
                          ["kind", "img", "model_cycles", "rtl_cycles", "err", "model_total", "rtl_total",
                           "model_mac", "rtl_mac", "stall", "out_ok", "issue_ok", "write_ok"])

    # ---- network ----
    if "net" in runs:
        nrows = []
        for r in runs["net"]["rows"]:
            net, img = r["net"], int(r["img"])
            G = gg.load_net(net)
            OC = G.final.cfg["OC"]
            v = np.array([int(x) for x in r["logits"].split(",")][:OC], dtype=np.int32)
            pred = int(final_layer.predict_from_raw(v[None], {"S_a": G.final.S_a, "S_w": G.final.S_w})[0])
            gold = read_hex(f"{net}/net/pred.hex")[img]
            label = read_hex(f"{net}/net/label.hex")[img]
            exp = [x - (1 << 32) if x >= 1 << 31 else x for x in read_hex(f"{net}/net/img{img}/logit10.hex")]
            m = base_meta(net=net, layer="all", source="rtl_sim", duration_s=runs["net"]["secs"],
                          num_inferences=1)
            m["vivado_version"] = runs["net"]["ver"]
            row = {**m, "img": img, "logits_match": list(v) == exp and r["logits_ok"] == "1",
                   "pred_rtl": pred, "pred_golden": gold, "pred_match": pred == gold, "label": label,
                   "model_total": int(r["model_total"]), "rtl_total": int(r["rtl_total"]),
                   "err": int(r["rtl_total"]) - int(r["model_total"]), "cycles_ok": r["cycles_ok"] == "1",
                   "rtl_layer_cycles": r["layer_cycles"], "model_layer_cycles": r["model_layer_cycles"],
                   "mac": int(r["mac"]), "stall": int(r["stall"])}
            ok &= row["logits_match"] and row["pred_match"] and row["cycles_ok"] and row["err"] == 0
            nrows.append(row)
        write_results_csv(RESULTS_DIR / "rtl_network.csv", nrows,
                          ["img", "logits_match", "pred_rtl", "pred_golden", "pred_match", "label",
                           "model_total", "rtl_total", "err", "cycles_ok", "rtl_layer_cycles",
                           "model_layer_cycles", "mac", "stall"])
        for net in ("lenet5", "cifar10"):
            rr = [x for x in nrows if x["net"] == net]
            print(f"  network {net}: images {len(rr)}, logits match {sum(x['logits_match'] for x in rr)}, "
                  f"pred match {sum(x['pred_match'] for x in rr)}, cycles ok {sum(x['cycles_ok'] for x in rr)}")

    # ---- checker ----
    if "checker" in runs:
        crows = []
        for r in runs["checker"]["rows"]:
            m = base_meta(layer="", source="rtl_sim", duration_s=runs["checker"]["secs"])
            m["vivado_version"] = runs["checker"]["ver"]
            row = {**m, "case": int(r["case"]), "exp_code": int(r["exp_code"]),
                   "rtl_code": int(r["rtl_code"]), "rtl_error": r["rtl_error"],
                   "accepted": int(r["exp_code"]) == 0, "ok": r["ok"] == "1"}
            ok &= row["ok"]
            crows.append(row)
        write_results_csv(RESULTS_DIR / "rtl_checker.csv", crows,
                          ["case", "exp_code", "rtl_code", "rtl_error", "accepted", "ok"])
        print(f"  checker: {len(crows)} cases, ok {sum(x['ok'] for x in crows)}, "
              f"accepted {sum(x['accepted'] for x in crows)}")
    if rows:
        for kind in ("layer", "fuzz"):
            rr = [x for x in rows if x["kind"] == kind]
            if rr:
                print(f"  {kind}: {len(rr)} runs, out ok {sum(x['out_ok'] for x in rr)}, "
                      f"cycles exact {sum(x['err'] == 0 for x in rr)}")
    print(f"core_collect: {'ALL OK' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
