#!/usr/bin/env python3
"""Build the V3 baseline board package (laptop, repo .venv) -> v3/board/data/package/ (gitignored).
Combines v2/dpu/make_dpu_package.py and the data/manifest part of v2/board/deploy.sh @ 28dd2ad.

    .venv/bin/python v3/board/make_baseline_package.py [--nets resnet20_b mobilenet_s]
        [--allow-dirty] [--allow-throwaway] [--allow-partial] [--results]

Prerequisites per net (same checkpoint for both):
  v3/dpu/export_data.py + v3/dpu/run_docker.sh quantize/compile   -> v3/dpu/build/<net>/...
  v3/board/export_onnx.py                                          -> v3/board/data/<net>/onnx/...
Checks: checkpoint sha256 and preprocessing identical in the DPU export and the ONNX export; DPU xmodel
sha256 == xmodel_info.json; xmodel DPU fingerprint == the prebuilt pynq-dpu 2.5 KV260 overlay's.
REFUSES a dirty tree (V3 rule: board data only from a clean, committed tree) unless --allow-dirty, and a
dry-run throwaway checkpoint unless --allow-throwaway (both recorded in MANIFEST.json; the board session
then marks every row paper_grade=False and refuses a dirty package without its own --allow-dirty).

Package layout (deploy_baseline.sh copies it to ~/gos3/ on the board):
  *.py, session.sh                   the board scripts (copies of v3/board/)
  data/test_u8.npz                   CIFAR-10 test[0:10000] uint8 HWC (x_u8) + labels (y)
  <net>/fp32.onnx, int8_qdq.onnx     ORT models; <net>/ref/*.npy laptop reference predictions
  <net>/dpu/<net>_kv260.xmodel, xmodel_info.json, vaiq_pred.npy
  baseline_model_accuracy.csv        laptop/container model accuracies (label "model")
  MANIFEST.json                      sha256 + size of every file, git state, per-net provenance
--results also writes v3/results/baseline_model_accuracy.csv (clean tree + trained checkpoints only).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

BOARD = Path(__file__).resolve().parent
V3 = BOARD.parent
DPU_BUILD = V3 / "dpu" / "build"
sys.path.insert(0, str(BOARD))
sys.path.insert(0, str(V3 / "model"))
import laptop_common as lc  # noqa: E402
import baseline_common as bc  # noqa: E402
import common  # noqa: E402  (v3/model/common.py: results metadata)

PKG = bc.PACKAGE_DIR_REPO
SCRIPTS = ("baseline_session.py", "baseline_common.py", "power_log.py", "stats.py", "board_env.py",
           "login_spikes.py", "aggregate_sessions.py", "session.sh")
VAI_VERSION = "2.5.0"
DOCKER_IMAGE = "xilinx/vitis-ai-cpu:2.5.0"
DPU_ARCH = "DPUCZDX8G_ISA1_B4096"
QUANTIZER = "vai_q_pytorch PTQ, calib train[0:1024], per-tensor power-of-2 (Vitis AI default)"
ACC_FIELDS = ["system", "label", "checkpoint_sha256", "checkpoint_kind", "model_sha256", "dpu_fingerprint",
              "dpu_subgraphs", "cpu_subgraphs", "all_compute_on_dpu", "correct", "total", "accuracy_pct"]


def overlay_md5(link: str, dpu_build: Path = DPU_BUILD) -> str:
    p = dpu_build / "pynq_dpu_2.5" / "pynq_dpu-2.5" / "pynq_dpu" / link
    return json.loads(p.read_text())["KV260"]["md5sum"]


def collect_dpu(net: str, out: Path, dpu_build: Path = DPU_BUILD) -> tuple[dict, dict]:
    nd = dpu_build / net
    ex = json.loads((nd / "export_info.json").read_text())
    qt = json.loads((nd / "quant_info_test.json").read_text())
    xi = json.loads((nd / "compiled" / "xmodel_info.json").read_text())
    xm = nd / "compiled" / xi["xmodel"]
    if bc.sha256_file(xm) != xi["xmodel_sha256"]:
        raise SystemExit(f"{xm}: sha256 != xmodel_info.json")
    if not xi.get("fingerprint_matches_overlay"):
        raise SystemExit(f"{net}: xmodel fingerprint {xi.get('dpu_fingerprint_hex')} != overlay "
                         f"{xi.get('overlay_fingerprint')}")
    if qt.get("checkpoint_sha256") != ex["checkpoint_sha256"]:
        raise SystemExit(f"{net}: vai_q test run used another checkpoint than export_info")
    d = out / net / "dpu"
    d.mkdir(parents=True, exist_ok=True)
    for f in (xm, nd / "compiled" / "xmodel_info.json", nd / "vaiq_pred.npy"):
        shutil.copyfile(f, d / f.name)
    ent = {"xmodel": f"{net}/dpu/{xi['xmodel']}", "xmodel_sha256": xi["xmodel_sha256"],
           "xmodel_info": f"{net}/dpu/xmodel_info.json", "vaiq_pred": f"{net}/dpu/vaiq_pred.npy",
           "dpu_fingerprint": xi["dpu_fingerprint_hex"], "n_subgraphs": xi["n_subgraphs"],
           "n_dpu_subgraphs": xi["n_dpu_subgraphs"], "n_cpu_subgraphs": xi["n_cpu_subgraphs"],
           "cpu_compute_ops": xi["cpu_compute_ops"], "all_compute_on_dpu": xi["all_compute_on_dpu"],
           "dpu_op_counts": xi.get("dpu_op_counts"), "versions": xi.get("versions", {}),
           "vaiq_correct": qt["vaiq_correct"], "vaiq_total": qt["vaiq_total"],
           "vai_q_pytorch_version": qt.get("vai_q_pytorch_version"), "fp32_correct": ex["fp32_correct"]}
    return ent, ex


def collect_onnx(net: str, out: Path, onnx_root: Path = BOARD / "data") -> tuple[dict, dict]:
    src = onnx_root / net / "onnx"
    oi = json.loads((src / "ONNX_INFO.json").read_text())
    for name, meta in oi["files"].items():
        if bc.sha256_file(src / name) != meta["sha256"]:
            raise SystemExit(f"{src / name}: sha256 != ONNX_INFO.json")
    (out / net / "ref").mkdir(parents=True, exist_ok=True)
    for f in ("fp32.onnx", "int8_qdq.onnx"):
        shutil.copyfile(src / f, out / net / f)
    for f in ("torch_pred.npy", "ort_fp32_pred.npy", "ort_int8_pred.npy"):
        shutil.copyfile(src / f, out / net / "ref" / f)
    ent = {"fp32": f"{net}/fp32.onnx", "fp32_sha256": oi["files"]["fp32.onnx"]["sha256"],
           "int8_qdq": f"{net}/int8_qdq.onnx", "int8_qdq_sha256": oi["files"]["int8_qdq.onnx"]["sha256"],
           "ref_pred_fp32": f"{net}/ref/torch_pred.npy", "ref_pred_int8_qdq": f"{net}/ref/ort_int8_pred.npy",
           "ref_pred_ort_fp32": f"{net}/ref/ort_fp32_pred.npy", "onnx_opset": oi["onnx_opset"],
           "ort_int8_quantization": oi["ort_int8_quantization"], "export_versions": oi["versions"],
           "laptop_verification": oi["verification"]}
    return ent, oi


def acc_rows(net: str, n: dict) -> list[dict]:
    ck, kind = n["checkpoint_sha256"], n["checkpoint_kind"]
    rows = []

    def row(system, label, correct, total, **kw):
        r = common.base_meta(net=net, layer="all", source="model", num_inferences=total)
        r.update(system=system, label=label, checkpoint_sha256=ck, checkpoint_kind=kind, correct=correct,
                 total=total, accuracy_pct=f"{100.0 * correct / total:.2f}", **kw)
        rows.append(r)
    if "onnx" in n:
        v = n["onnx"]["laptop_verification"]
        row("torch_fp32", "model, FP32 (PyTorch, laptop)", v["torch_fp32_correct"], v["n_images"])
        row("ort_fp32", "model, ORT FP32 (laptop)", v["ort_fp32_correct"], v["n_images"],
            model_sha256=n["onnx"]["fp32_sha256"])
        row("ort_int8", "model, ORT INT8 QDQ S8S8 per-channel MinMax (laptop)", v["ort_int8_correct"],
            v["n_images"], model_sha256=n["onnx"]["int8_qdq_sha256"])
    if "dpu" in n:
        d = n["dpu"]
        row("dpu_vaiq", "DPU quantized (vai_q), model (Vitis AI 2.5.0 container)", d["vaiq_correct"], d["vaiq_total"],
            model_sha256=d["xmodel_sha256"], dpu_fingerprint=",".join(d["dpu_fingerprint"]),
            dpu_subgraphs=d["n_dpu_subgraphs"], cpu_subgraphs=d["n_cpu_subgraphs"],
            all_compute_on_dpu=d["all_compute_on_dpu"])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nets", nargs="+", default=["resnet20_b", "mobilenet_s"])
    ap.add_argument("--out", default=str(PKG))
    ap.add_argument("--dpu-build", default=str(DPU_BUILD), help="v3/dpu/build (DPU exports + compiled xmodels)")
    ap.add_argument("--onnx-root", default=str(BOARD / "data"), help="dir holding <net>/onnx/ (export_onnx.py)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-throwaway", action="store_true", help="package a dry-run throwaway checkpoint")
    ap.add_argument("--allow-partial", action="store_true", help="a net may lack its DPU or ONNX part")
    ap.add_argument("--results", action="store_true", help="also write v3/results/baseline_model_accuracy.csv")
    a = ap.parse_args(argv)
    gs = lc.git_state()
    if gs["git_dirty"] and not a.allow_dirty:
        print("REFUSED: dirty tree (commit first; --allow-dirty builds a package the board refuses without "
              "its own --allow-dirty and never marks paper-grade)", file=sys.stderr)
        return 2
    out = Path(a.out)
    dpu_build, onnx_root = Path(a.dpu_build), Path(a.onnx_root)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    nets, rows = {}, []
    for net in a.nets:
        n: dict = {}
        exs = []
        if (dpu_build / net / "compiled" / "xmodel_info.json").is_file():
            n["dpu"], ex = collect_dpu(net, out, dpu_build)
            exs.append(("dpu", ex))
        if (onnx_root / net / "onnx" / "ONNX_INFO.json").is_file():
            n["onnx"], oi = collect_onnx(net, out, onnx_root)
            exs.append(("onnx", oi))
        if not exs or (len(exs) < 2 and not a.allow_partial):
            raise SystemExit(f"{net}: need both the DPU build and the ONNX export (have {[k for k, _ in exs]}); "
                             "--allow-partial to package what exists")
        for key in ("checkpoint_sha256", "checkpoint_kind", "preprocess"):
            vals = {json.dumps(e[key], sort_keys=True) for _, e in exs}
            if len(vals) != 1:
                raise SystemExit(f"{net}: {key} differs between the DPU and ONNX exports: {vals}")
        e0 = exs[0][1]
        if e0["checkpoint_kind"] != bc.CHECKPOINT_TRAINED and not a.allow_throwaway:
            raise SystemExit(f"{net}: checkpoint kind {e0['checkpoint_kind']} (dry run only): --allow-throwaway")
        n.update(checkpoint=e0["checkpoint"], checkpoint_sha256=e0["checkpoint_sha256"],
                 checkpoint_kind=e0["checkpoint_kind"], preprocess=e0["preprocess"])
        nets[net] = n
        rows += acc_rows(net, n)
    x_u8, y = lc.cifar_test()
    np.savez(out / "data" / "test_u8.npz", x_u8=x_u8, y=y)
    for s in SCRIPTS:
        shutil.copy2(BOARD / s, out / s)
    acc_path = common.write_results_csv(out / "baseline_model_accuracy.csv", rows, ACC_FIELDS)
    files = {str(p.relative_to(out)): {"sha256": bc.sha256_file(p), "bytes": p.stat().st_size}
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != bc.MANIFEST}
    man = {"created_utc": bc.utc_now(), **gs, "nets": nets,
           "test_set": {"file": "data/test_u8.npz", "what": "CIFAR-10 test[0:10000] uint8 HWC + labels",
                        "n": int(len(y))},
           "dpu": {"vitis_ai_version": VAI_VERSION, "docker_image": DOCKER_IMAGE, "dpu_arch": DPU_ARCH,
                   "quantizer": QUANTIZER,
                   "board_runtime": "Kria-PYNQ (pynq 3.0.1) + pynq-dpu==2.5 (DPU-PYNQ v2.5, Vitis AI 2.5.0)",
                   "overlay": "prebuilt pynq-dpu 2.5 KV260 overlay (dpu.bit = pynqdpu.dpu.kv260_som.2.5.0.bit)",
                   "overlay_bit_md5_expected": overlay_md5("dpu.bit.link", dpu_build),
                   "overlay_hwh_md5_expected": overlay_md5("dpu.hwh.link", dpu_build),
                   "overlay_arch": json.loads((dpu_build / "overlay_arch_info.json").read_text())},
           "label_rule": "DPU and ORT numbers are their own quantizations of the same FP32 checkpoint; never our "
                         "INT8 reference and never our accelerator", "files": files}
    (out / bc.MANIFEST).write_text(json.dumps(man, indent=1) + "\n")
    for net, n in nets.items():
        d = n.get("dpu", {})
        print(f"{net}: checkpoint {n['checkpoint_sha256'][:12]} ({n['checkpoint_kind']}); DPU: "
              + (f"{d['n_subgraphs']} subgraphs (DPU {d['n_dpu_subgraphs']}, CPU {d['n_cpu_subgraphs']}, CPU compute "
                 f"{d['cpu_compute_ops']}), fingerprint {d['dpu_fingerprint']}, vai_q {d['vaiq_correct']}/"
                 f"{d['vaiq_total']}" if d else "none") + ("; ONNX: yes" if "onnx" in n else "; ONNX: none"))
    print(f"package: {out} ({len(files)} files, git {gs['git_commit'][:8]} dirty={gs['git_dirty']}); "
          f"model accuracies -> {acc_path}")
    if a.results:
        kinds = {n["checkpoint_kind"] for n in nets.values()}
        if gs["git_dirty"] or kinds != {bc.CHECKPOINT_TRAINED}:
            print(f"NOT writing v3/results/baseline_model_accuracy.csv: dirty={gs['git_dirty']}, checkpoint kinds "
                  f"{sorted(kinds)}", file=sys.stderr)
            return 1
        common.write_results_csv(common.RESULTS_DIR / "baseline_model_accuracy.csv", rows, ACC_FIELDS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
