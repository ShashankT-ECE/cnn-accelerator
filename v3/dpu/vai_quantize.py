#!/usr/bin/env python3
"""DPU baseline step 2 (inside the Vitis AI 2.5.0 docker, conda env vitis-ai-pytorch).
Adapted from v2/dpu/vai_quantize.py @ 28dd2ad (V3: model = v3/model/nets.build(net) with BN; vai_q
folds BN into the convolutions).

    python /workspace/repo/v3/dpu/vai_quantize.py --net resnet20_b --mode calib|test|deploy

Run through v3/dpu/run_docker.sh (repo read-only at /workspace/repo, v3/dpu/build read-write at
/workspace/build). Python 3.7 compatible (container python).

Model: v3/model/nets.py build(net) (unchanged class, imported from the read-only repo mount) with
the exported state_dict (build/<net>/fp32_state_dict.pt, sha256 checked against export_info.json).
Inputs: build/<net>/dpu_inputs.npz from export_data.py (test[0:10000], calibration train[0:1024],
preprocessing recorded in export_info.json).

  --mode calib  FP32 accuracy on x_test (must equal export_info fp32_correct), then vai_q_pytorch
                PTQ calibration (torch_quantizer quant_mode='calib') over x_calib, export the quant config.
  --mode test   quant_mode='test': accuracy of the vai_q-quantized model on all 10,000 test images
                (label "DPU quantized (vai_q), model" -- a PyTorch simulation, NOT a DPU measurement),
                predictions saved.
  --mode deploy fresh quant_mode='test' quantizer, exactly one batch-1 forward (vai_q requires batch 1
                for export), export_xmodel -> build/<net>/quantized/<Class>_int.xmodel; checks the
                batch-1 prediction of image 0 equals the test-mode one.

Outputs (gitignored, build/<net>/): quantized/, vaiq_pred.npy, vaiq_logits.npy, quant_info_<mode>.json.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.environ.get("REPO", "/workspace/repo")
BUILD = os.environ.get("DPU_BUILD", "/workspace/build")
sys.path.insert(0, os.path.join(REPO, "v3", "model"))

BATCH = 500


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_model(net, sd_path):
    import nets
    m = nets.build(net)
    m.load_state_dict(torch.load(sd_path, map_location="cpu"))
    m.eval()
    return m


def predict(model, x, batch=BATCH):
    logits = []
    with torch.no_grad():
        for s in range(0, x.shape[0], batch):
            logits.append(model(torch.from_numpy(x[s:s + batch])).float().numpy())
    lg = np.concatenate(logits).astype(np.float32)
    return lg, lg.argmax(1).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", required=True, choices=("resnet20_a", "resnet20_b", "mobilenet_s"))
    ap.add_argument("--mode", required=True, choices=("calib", "test", "deploy"))
    a = ap.parse_args()

    from pytorch_nndct.apis import torch_quantizer
    import pytorch_nndct

    t0 = time.time()
    nd = os.path.join(BUILD, a.net)
    info = json.load(open(os.path.join(nd, "export_info.json")))
    sd = os.path.join(nd, "fp32_state_dict.pt")
    if sha256(sd) != info["state_dict_sha256"]:
        sys.exit("state_dict sha256 != export_info state_dict_sha256")
    npz = os.path.join(nd, "dpu_inputs.npz")
    if sha256(npz) != info["dpu_inputs_sha256"]:
        sys.exit("dpu_inputs.npz sha256 != export_info")
    d = np.load(npz)
    x_test, y_test, x_cal = d["x_test"], d["y_test"], d["x_calib"]
    qdir = os.path.join(nd, "quantized")
    model = build_model(a.net, sd)
    dummy = torch.from_numpy(x_test[:1])
    out = {"net": a.net, "mode": a.mode, "python": sys.version.split()[0],
           "torch_version": torch.__version__,
           "vai_q_pytorch_version": getattr(pytorch_nndct, "__version__", "unknown"),
           "checkpoint_sha256": info["checkpoint_sha256"], "checkpoint_kind": info["checkpoint_kind"],
           "dpu_inputs_sha256": info["dpu_inputs_sha256"]}

    if a.mode == "calib":
        _, p = predict(model, x_test)
        c = int((p == y_test).sum())
        print("[%s] FP32 in container: %d/%d (export %d)" % (a.net, c, len(y_test), info["fp32_correct"]))
        # torch 1.10 (container) vs 2.x (laptop) may differ in the last float bit; a prediction flip is
        # reported, more than 0.1 % of the images stops the flow
        out["fp32_container_correct"] = c
        out["fp32_container_vs_export_diff"] = c - info["fp32_correct"]
        if abs(c - info["fp32_correct"]) > len(y_test) // 1000:
            sys.exit("FP32 accuracy in container differs from export by more than 0.1 %")
        q = torch_quantizer("calib", model, (dummy,), device=torch.device("cpu"), output_dir=qdir)
        qm = q.quant_model
        with torch.no_grad():
            for s in range(0, x_cal.shape[0], 256):
                qm(torch.from_numpy(x_cal[s:s + 256]))
        q.export_quant_config()
        out["calib_images"] = int(x_cal.shape[0])
        out["calib_set"] = info["calib_set"]
    elif a.mode == "test":
        q = torch_quantizer("test", model, (dummy,), device=torch.device("cpu"), output_dir=qdir)
        qm = q.quant_model
        lg, p = predict(qm, x_test)
        c = int((p == y_test).sum())
        print("[%s] DPU quantized (vai_q), model: %d/%d = %.2f%%" % (a.net, c, len(y_test), 100.0 * c / len(y_test)))
        np.save(os.path.join(nd, "vaiq_pred.npy"), p)
        np.save(os.path.join(nd, "vaiq_logits.npy"), lg)
        out.update(vaiq_correct=c, vaiq_total=int(len(y_test)),
                   vaiq_accuracy_pct="%.2f" % (100.0 * c / len(y_test)),
                   label="DPU quantized (vai_q), model",
                   vaiq_pred_sha256=sha256(os.path.join(nd, "vaiq_pred.npy")))
    else:  # deploy: fresh test-mode quantizer, exactly one batch-1 forward, then export_xmodel
        for f in os.listdir(qdir):
            if f.endswith(".xmodel"):
                os.remove(os.path.join(qdir, f))
        q = torch_quantizer("test", model, (dummy,), device=torch.device("cpu"), output_dir=qdir)
        qm = q.quant_model
        with torch.no_grad():
            o = qm(dummy)
        q.export_xmodel(output_dir=qdir, deploy_check=False)
        xm = [f for f in os.listdir(qdir) if f.endswith("_int.xmodel")]
        if len(xm) != 1:
            sys.exit("expected one *_int.xmodel, got %s" % xm)
        vp = np.load(os.path.join(nd, "vaiq_pred.npy"))
        out.update(quantized_xmodel=xm[0],
                   quantized_xmodel_sha256=sha256(os.path.join(qdir, xm[0])),
                   deploy_forward_pred0=int(o.argmax(1)[0]), vaiq_pred0=int(vp[0]))
        if out["deploy_forward_pred0"] != out["vaiq_pred0"]:
            sys.exit("batch-1 deploy forward disagrees with the test-mode prediction of image 0")
    out["duration_s"] = round(time.time() - t0, 3)
    with open(os.path.join(nd, "quant_info_%s.json" % a.mode), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
