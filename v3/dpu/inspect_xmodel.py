#!/usr/bin/env python3
# copied from v2/dpu/inspect_xmodel.py @ 28dd2ad (V3: + per-net DPU op summary: depthwise / residual add / pool)
"""DPU baseline step 3b (inside the Vitis AI docker, after vai_c_xir): describe the compiled xmodel.

    python /workspace/repo/v3/dpu/inspect_xmodel.py --net resnet20_b --arch <arch.json>

Reads build/<net>/compiled/<net>_kv260.xmodel with the xir Python API and writes
build/<net>/compiled/xmodel_info.json: sha256 + size, target arch.json (content + sha256),
every child subgraph in topological order (device DPU / CPU / USER, op types, input/output
tensors with dims, dtype and fix_point), the DPU fingerprint, the number of DPU and CPU
subgraphs, the tool versions found in the container (conda packages), and the quantize info.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import subprocess
import sys

BUILD = os.environ.get("DPU_BUILD", "/workspace/build")
CONVERSION_OPS = {"fix2float", "float2fix", "data", "data-fix"}
# xir op types of the V3 layer kinds (Vitis AI 2.5 DPUCZDX8G): depthwise conv, residual add, pooling (GAP)
KEY_OPS = {"depthwise_conv": ("depthwise-conv2d-fix",), "residual_add": ("eltwise-fix",),
           "pool_gap": ("pool-fix",), "conv": ("conv2d-fix",)}


def op_counts(g, device):
    """{op_type: count} over the ops of every child subgraph on `device`."""
    out = {}
    for sg in g.get_root_subgraph().toposort_child_subgraph():
        if attr(sg, "device") != device:
            continue
        for op in sg.get_ops():
            out[op.get_type()] = out.get(op.get_type(), 0) + 1
    return dict(sorted(out.items()))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def attr(obj, name):
    try:
        return obj.get_attr(name) if obj.has_attr(name) else None
    except Exception:  # noqa: BLE001
        return None


def tdesc(t):
    return {"name": t.name, "dims": list(t.dims), "dtype": str(getattr(t, "dtype", "")),
            "fix_point": attr(t, "fix_point")}


def versions():
    out = {"image": os.environ.get("VAI_IMAGE", "xilinx/vitis-ai-cpu:2.5.0")}
    try:
        txt = subprocess.check_output(["conda", "list"], universal_newlines=True,
                                      stderr=subprocess.STDOUT)
        keep = ("xir", "xcompiler", "vart", "target-factory", "unilog", "pytorch-nndct",
                "pytorch_nndct", "vai_q_pytorch", "pytorch ", "torch ", "python ")
        out["conda"] = [" ".join(l.split()[:2]) for l in txt.splitlines()
                        if any(l.startswith(k.strip()) and l.split()[0] == k.strip() for k in keep)]
    except Exception as e:  # noqa: BLE001
        out["conda_error"] = str(e)
    for f in ("/etc/VAI_VERSION", "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/VAI_VERSION"):
        if os.path.isfile(f):
            out[f] = open(f).read().strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--expect-overlay", default=None,
                    help="overlay_arch_info.json: fail unless the DPU fingerprint equals it")
    a = ap.parse_args()
    import xir

    cdir = os.path.join(BUILD, a.net, "compiled")
    xm = os.path.join(cdir, "%s_kv260.xmodel" % a.net)
    g = xir.Graph.deserialize(xm)
    subs = []
    for sg in g.get_root_subgraph().toposort_child_subgraph():
        dev = attr(sg, "device")
        ops = sorted({op.get_type() for op in sg.get_ops()})
        subs.append({"name": sg.get_name(), "device": dev, "op_types": ops,
                     "n_ops": len(sg.get_ops()),
                     "inputs": [tdesc(t) for t in sg.get_input_tensors()],
                     "outputs": [tdesc(t) for t in sg.get_output_tensors()],
                     "dpu_fingerprint": attr(sg, "dpu_fingerprint"),
                     "dpu_name": attr(sg, "dpu_name")})
    dpu = [s for s in subs if s["device"] == "DPU"]
    cpu = [s for s in subs if s["device"] == "CPU"]
    fps = sorted({str(s["dpu_fingerprint"]) for s in dpu})
    arch = json.load(open(a.arch))
    qi = {}
    for m in ("calib", "test"):
        p = os.path.join(BUILD, a.net, "quant_info_%s.json" % m)
        if os.path.isfile(p):
            qi[m] = json.load(open(p))
    ov = json.load(open(a.expect_overlay)) if a.expect_overlay else None
    ov_fp = ov["arch"]["fingerprint"] if ov else None
    fps_hex = sorted("0x%x" % int(f) for f in fps if f not in ("None", ""))
    info = {"net": a.net, "dpu_fingerprint_hex": fps_hex, "overlay_fingerprint": ov_fp,
            "fingerprint_matches_overlay": (fps_hex == [ov_fp]) if ov else None,
            "overlay_arch_info": ov, "xmodel": os.path.basename(xm), "xmodel_sha256": sha256(xm),
            "xmodel_bytes": os.path.getsize(xm), "arch_json_path": a.arch, "arch_json": arch,
            "arch_json_sha256": sha256(a.arch), "dpu_fingerprint": fps,
            "n_subgraphs": len(subs), "n_dpu_subgraphs": len(dpu), "n_cpu_subgraphs": len(cpu),
            "cpu_ops": sorted({o for s in cpu for o in s["op_types"]}),
            "cpu_compute_ops": sorted({o for s in cpu for o in s["op_types"]} - CONVERSION_OPS),
            "all_compute_on_dpu": len(dpu) == 1 and not ({o for s in cpu for o in s["op_types"]}
                                                          - CONVERSION_OPS),
            "note": "CPU subgraphs holding only data-type conversion ops (fix2float / "
                    "float2fix) are not compute; pynq_dpu runs the single DPU subgraph and the "
                    "session script does the conversion itself",
            "dpu_op_types": sorted({o for s in dpu for o in s["op_types"]}),
            "dpu_op_counts": op_counts(g, "DPU"),
            "dpu_has": {k: any(o in {x for s in dpu for x in s["op_types"]} for o in v)
                        for k, v in KEY_OPS.items()},
            "dpu_input": dpu[0]["inputs"] if dpu else None,
            "dpu_output": dpu[-1]["outputs"] if dpu else None,
            "subgraphs": subs, "versions": versions(), "quant_info": qi}
    with open(os.path.join(cdir, "xmodel_info.json"), "w") as f:
        json.dump(info, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in info.items() if k not in ("subgraphs", "quant_info")},
                     indent=1, default=str))
    if ov is not None and not info["fingerprint_matches_overlay"]:
        print("FINGERPRINT MISMATCH: xmodel %s != overlay %s" % (fps_hex, ov_fp))
        return 1
    for s in subs:
        print("subgraph %-40s device=%-5s ops=%s" % (s["name"], s["device"], ",".join(s["op_types"])))
    print("SUMMARY %s: subgraphs %d (DPU %d, CPU %d), CPU compute ops %s, all compute on DPU: %s, "
          "fingerprint %s (overlay %s, match %s), DPU op counts %s" % (
              a.net, info["n_subgraphs"], info["n_dpu_subgraphs"], info["n_cpu_subgraphs"],
              info["cpu_compute_ops"], info["all_compute_on_dpu"], fps_hex, ov_fp,
              info["fingerprint_matches_overlay"], info["dpu_op_counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
