#!/usr/bin/env python3
"""Derive the compile target (arch.json) from the PREBUILT pynq_dpu 2.5 KV260 overlay.

Runs inside the Vitis AI 2.5.0 container (needs xir):

    python /workspace/repo/v2/dpu/overlay_arch.py        (via run_docker.sh arch)

pynq-dpu 2.5 (the version Kria-PYNQ installs) ships no arch.json. Its KV260 overlay
(dpu.bit/.hwh/.xclbin, pynqdpu.dpu.kv260_som.2.5.0.*) comes with Xilinx-compiled example
xmodels for that board; their DPU subgraph carries the fingerprint the overlay's DPU accepts
(VART refuses a mismatch). fetch_overlay_ref.sh downloads, md5-checks (against the pynq-dpu 2.5
.link files) and stores in build/pynq_dpu_2.5/kv260/:
  pynqdpu.dpu.kv260_som.2.5.0.hwh                               (overlay hardware description)
  pynqdpu.tf2_mnist_classifier.DPUCZDX8G_ISA1_B4096.2.5.0.xmodel (KV260 example xmodel)

This script reads the example xmodel's DPU fingerprint and the DPU parameters from the .hwh and
writes build/arch_kv260_pynqdpu25.json = {"fingerprint": "0x..."} (the DPU-PYNQ convention for
custom arch.json files, e.g. arch_ultra96.json) + build/overlay_arch_info.json (provenance, and
the fingerprint the docker's own KV260 arch.json target resolves to, for comparison).
"""
from __future__ import print_function

import hashlib
import json
import os
import re
import sys

BUILD = os.environ.get("DPU_BUILD", "/workspace/build")
REF = os.path.join(BUILD, "pynq_dpu_2.5", "kv260")
XM = os.path.join(REF, "pynqdpu.tf2_mnist_classifier.DPUCZDX8G_ISA1_B4096.2.5.0.xmodel")
HWH = os.path.join(REF, "pynqdpu.dpu.kv260_som.2.5.0.hwh")
DOCKER_ARCH = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"
HWH_KEYS = ("ARCH_ICP", "ARCH_OCP", "ARCH_PP", "ARCH_IMG_BKGRP", "ARCH_DATA_BW", "ARCH_HP_BW",
            "UBANK_IMG_N", "UBANK_WGT_N", "UBANK_BIAS", "LOAD_AUGM", "LOAD_IMG_MEAN",
            "LOAD_PARALLEL", "SAVE_PARALLEL", "CONV_LEAKYRELU", "CONV_WR_PARALLEL",
            "CONV_DSP_CASC_MAX", "CONV_DSP_ACCU_ENA", "ALU_PARALLEL", "ALU_LEAKYRELU")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def docker_target_fingerprint():
    """Fingerprint the docker's KV260 arch.json target resolves to (target_factory)."""
    try:
        import target_factory  # noqa: PLC0415
        t = json.load(open(DOCKER_ARCH))["target"]
        tmp = "/tmp/docker_kv260_target.prototxt"
        target_factory.dump_proto_txt(target_factory.get_target_by_name(t), tmp)
        txt = open(tmp).read()
        isa = re.search(r"isa_version:\s*(\d+)", txt)
        fc = re.search(r"feature_code:\s*(\d+)", txt)
        if not (isa and fc):
            return "not in proto: %s" % txt[:200]
        # DPUCZDX8G fingerprint layout (as in the overlay's 0x0101000016010407):
        # type code 0x01 << 56 | isa_version << 48 | feature_code (low 32 bits)
        return "0x%x" % ((0x01 << 56) | (int(isa.group(1)) << 48) | int(fc.group(1)))
    except Exception as e:  # noqa: BLE001
        return "unavailable: %s" % e


def main():
    import xir
    g = xir.Graph.deserialize(XM)
    fps = set()
    names = set()
    for s in g.get_root_subgraph().toposort_child_subgraph():
        if s.has_attr("device") and s.get_attr("device") == "DPU":
            fps.add(int(s.get_attr("dpu_fingerprint")))
            names.add(s.get_attr("dpu_name"))
    if len(fps) != 1:
        sys.exit("expected one DPU fingerprint in the reference xmodel, got %s" % fps)
    fp = "0x%x" % fps.pop()
    hwh = open(HWH).read()
    params = {}
    for k in HWH_KEYS:
        m = re.search(r'NAME="%s" VALUE="([^"]*)"' % k, hwh)
        params[k] = m.group(1) if m else None
    arch = {"fingerprint": fp}
    out = os.path.join(BUILD, "arch_kv260_pynqdpu25.json")
    with open(out, "w") as f:
        json.dump(arch, f)
    dfp = docker_target_fingerprint()
    info = {"arch_json": out, "arch": arch, "arch_json_sha256": sha256(out),
            "source": "DPU fingerprint of the Xilinx-compiled KV260 example xmodel of the "
                      "pynq-dpu 2.5 overlay (DPU-PYNQ v2.5, Vitis AI 2.5.0)",
            "reference_xmodel": os.path.basename(XM), "reference_xmodel_md5": md5(XM),
            "reference_dpu_name": sorted(names), "overlay_hwh": os.path.basename(HWH),
            "overlay_hwh_md5": md5(HWH), "overlay_dpu_params": params,
            "ops_per_cycle": (int(params["ARCH_ICP"]) * int(params["ARCH_OCP"]) *
                              int(params["ARCH_PP"]) * 2) if params["ARCH_ICP"] else None,
            "docker_kv260_arch_json": json.load(open(DOCKER_ARCH)),
            "docker_kv260_target_fingerprint": dfp,
            "docker_kv260_target_fingerprint_note": "assembled from the target proto "
            "(0x01<<56 | isa_version<<48 | feature_code); the authoritative check is the "
            "dpu_fingerprint vai_c_xir writes into each compiled xmodel (inspect_xmodel.py)",
            "docker_target_matches_overlay": dfp == fp}
    with open(os.path.join(BUILD, "overlay_arch_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
