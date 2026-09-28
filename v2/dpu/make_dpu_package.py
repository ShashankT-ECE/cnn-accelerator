#!/usr/bin/env python3
"""DPU baseline step 4 (laptop): collect the compiled xmodels into the board package + CSV.

    .venv/bin/python v2/dpu/make_dpu_package.py [--nets lenet5 cifar10] [--results] [--allow-dirty]

Reads v2/dpu/build/<net>/{export_info.json, quant_info_*.json, vaiq_pred.npy,
compiled/<net>_kv260.xmodel, compiled/xmodel_info.json} and writes (all gitignored):

  build/package/DPU_INFO.json          git state, Vitis AI version, image, arch/fingerprint,
                                       per net: xmodel file + sha256, subgraph counts, accuracies
  build/package/<net>/                 <net>_kv260.xmodel, xmodel_info.json, vaiq_pred.npy
  build/package/dpu_session.py         the board session script
  build/dpu_model_accuracy.csv         per net: FP32 (model) and "DPU quantized (vai_q), model"
                                       accuracy rows (EXPERIMENTS.md CSV rule columns)

--results also writes v2/results/dpu_model_accuracy.csv; refused on a dirty tree (v2/CLAUDE.md:
results CSVs only from a clean, committed tree) unless --allow-dirty (then build/ only).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

DPU = Path(__file__).resolve().parent
V2 = DPU.parent
BUILD = DPU / "build"
PKG = BUILD / "package"
sys.path.insert(0, str(V2 / "model"))
import common  # noqa: E402
from export_data import git_state, sha256  # noqa: E402

VAI_VERSION = "2.5.0"
DOCKER_IMAGE = "xilinx/vitis-ai-cpu:2.5.0"
DPU_ARCH = "DPUCZDX8G_ISA1_B4096"
QUANTIZER = "vai_q_pytorch PTQ, calib train[0:1024], per-tensor power-of-2 (Vitis AI default)"
FIELDS = ["reference_version", "precision", "label", "vitis_ai_version", "docker_image",
          "dpu_arch", "dpu_fingerprint", "quantizer", "xmodel_sha256", "dpu_subgraphs",
          "cpu_subgraphs", "correct", "total", "accuracy_pct"]


def overlay_md5(link: str) -> str:
    """KV260 md5 of the prebuilt overlay file from the pynq-dpu 2.5 sdist .link file."""
    p = BUILD / "pynq_dpu_2.5" / "pynq_dpu-2.5" / "pynq_dpu" / link
    return json.loads(p.read_text())["KV260"]["md5sum"]


def fp_hex(e: dict) -> str:
    return ",".join("0x%x" % int(f) for f in e["dpu_fingerprint"])


def collect(net: str) -> tuple[dict, list[dict]]:
    nd = BUILD / net
    ex = json.loads((nd / "export_info.json").read_text())
    qt = json.loads((nd / "quant_info_test.json").read_text())
    xi = json.loads((nd / "compiled" / "xmodel_info.json").read_text())
    xm = nd / "compiled" / xi["xmodel"]
    if sha256(xm) != xi["xmodel_sha256"]:
        raise SystemExit(f"{xm}: sha256 != xmodel_info.json")
    out = PKG / net
    out.mkdir(parents=True, exist_ok=True)
    for f in (xm, nd / "compiled" / "xmodel_info.json", nd / "vaiq_pred.npy"):
        shutil.copyfile(f, out / f.name)
    if not xi.get("fingerprint_matches_overlay"):
        raise SystemExit(f"{net}: xmodel fingerprint {xi.get('dpu_fingerprint_hex')} != overlay "
                         f"{xi.get('overlay_fingerprint')}")
    fp = ",".join(xi["dpu_fingerprint_hex"])
    ent = {"xmodel": xi["xmodel"], "xmodel_sha256": xi["xmodel_sha256"],
           "xmodel_bytes": xi["xmodel_bytes"], "dpu_fingerprint": xi["dpu_fingerprint"],
           "arch_json": xi["arch_json"], "arch_json_sha256": xi["arch_json_sha256"],
           "n_dpu_subgraphs": xi["n_dpu_subgraphs"], "n_cpu_subgraphs": xi["n_cpu_subgraphs"],
           "all_compute_on_dpu": xi["all_compute_on_dpu"], "cpu_ops": xi.get("cpu_ops"),
           "fingerprint_matches_overlay": xi["fingerprint_matches_overlay"],
           "dpu_input": xi["dpu_input"],
           "dpu_output": xi["dpu_output"], "versions": xi.get("versions", {}),
           "reference_version": ex["reference_version"], "checkpoint": ex["checkpoint"],
           "checkpoint_sha256": ex["checkpoint_sha256"], "fp32_correct": ex["fp32_correct"],
           "vaiq_correct": qt["vaiq_correct"], "vaiq_total": qt["vaiq_total"],
           "vai_q_pytorch_version": qt.get("vai_q_pytorch_version"),
           "container_torch_version": qt.get("torch_version")}
    rows = []
    for prec, lab, c in (("FP32", "model, FP32", ex["fp32_correct"]),
                         ("vai_q INT8", "DPU quantized (vai_q), model", qt["vaiq_correct"])):
        r = common.base_meta(net=net, layer="all", source="model", num_inferences=10000)
        r.update(reference_version=ex["reference_version"], precision=prec, label=lab,
                 vitis_ai_version=VAI_VERSION, docker_image=DOCKER_IMAGE, dpu_arch=DPU_ARCH,
                 dpu_fingerprint=fp, quantizer=QUANTIZER if prec != "FP32" else "",
                 xmodel_sha256=xi["xmodel_sha256"] if prec != "FP32" else "",
                 dpu_subgraphs=xi["n_dpu_subgraphs"], cpu_subgraphs=xi["n_cpu_subgraphs"],
                 correct=c, total=10000, accuracy_pct=f"{c / 100:.2f}")
        rows.append(r)
    return ent, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nets", nargs="+", default=["lenet5", "cifar10"])
    ap.add_argument("--results", action="store_true", help="also write v2/results/dpu_model_accuracy.csv")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    gs = git_state()
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True)
    nets, rows = {}, []
    for n in a.nets:
        nets[n], r = collect(n)
        rows += r
    info = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            **gs, "vitis_ai_version": VAI_VERSION, "docker_image": DOCKER_IMAGE,
            "dpu_arch": "DPUCZDX8G_ISA1_B4096", "quantizer": QUANTIZER,
            "board_runtime": "Kria-PYNQ (pynq 3.0.1) + pynq-dpu==2.5 (DPU-PYNQ v2.5, Vitis AI 2.5.0)",
            "overlay": "prebuilt pynq-dpu 2.5 KV260 overlay (dpu.bit = pynqdpu.dpu.kv260_som.2.5.0.bit)",
            "overlay_arch": json.loads((BUILD / "overlay_arch_info.json").read_text()),
            "overlay_bit_md5_expected": overlay_md5("dpu.bit.link"),
            "overlay_hwh_md5_expected": overlay_md5("dpu.hwh.link"),
            "label_rule": "DPU numbers are a Vitis AI vai_q quantization of the same FP32 nets; "
                          "never our INT8 reference", "nets": nets}
    (PKG / "DPU_INFO.json").write_text(json.dumps(info, indent=1) + "\n")
    shutil.copyfile(DPU / "dpu_session.py", PKG / "dpu_session.py")
    common.write_results_csv(BUILD / "dpu_model_accuracy.csv", rows, FIELDS)
    if a.results:
        if gs["git_dirty"]:
            print("NOT writing v2/results/dpu_model_accuracy.csv: dirty tree (commit first)",
                  file=sys.stderr)
            return 0 if a.allow_dirty else 1
        common.write_results_csv(V2 / "results" / "dpu_model_accuracy.csv", rows, FIELDS)
    for n, e in nets.items():
        print(f"{n}: {e['xmodel']} sha256 {e['xmodel_sha256']} ({e['xmodel_bytes']} B), "
              f"DPU subgraphs {e['n_dpu_subgraphs']}, CPU subgraphs {e['n_cpu_subgraphs']}, "
              f"fingerprint {fp_hex(e)}, FP32 {e['fp32_correct']}/10000, "
              f"vai_q {e['vaiq_correct']}/10000")
    print(f"package: {PKG} (git {gs['git_commit'][:8]} dirty={gs['git_dirty']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
