#!/usr/bin/env python3
"""Compare two compiled xmodels beyond their bytes (inside the Vitis AI container, needs xir).

    python /workspace/repo/v2/dpu/xmodel_diff.py A.xmodel B.xmodel

vai_c_xir output is not byte-reproducible; this reports whether the parts that determine what
the DPU computes are identical: per DPU subgraph the machine code (mc_code), the register id
map, the fingerprint, and every const op's data (weights/biases) + fix_point; plus the list of
attributes that differ (typically build metadata).
"""
from __future__ import print_function

import hashlib
import sys

import xir


def h(b):
    return hashlib.sha256(bytes(b)).hexdigest()[:16]


def summary(path):
    g = xir.Graph.deserialize(path)
    out = {"graph_attrs": {k: h(repr(g.get_attr(k)).encode()) for k in g.get_attrs()}}
    for s in g.get_root_subgraph().toposort_child_subgraph():
        if not (s.has_attr("device") and s.get_attr("device") == "DPU"):
            continue
        d = {}
        for k in s.get_attrs():
            v = s.get_attr(k)
            try:
                d[k] = h(v) if isinstance(v, (bytes, bytearray)) else h(repr(v).encode())
            except Exception:  # noqa: BLE001
                d[k] = "?"
        consts = {}
        for op in s.get_ops():
            if op.get_type() == "const-fix":
                t = op.get_output_tensor()
                consts[op.get_name()] = (h(op.get_attr("data")) if op.has_attr("data") else "",
                                         t.get_attr("fix_point") if t.has_attr("fix_point") else None)
        out[s.get_name()] = {"attrs": d, "consts": consts}
        print(path.split("/")[-1], s.get_name(), "attrs:", sorted(d))
    return out


a, b = summary(sys.argv[1]), summary(sys.argv[2])
same = True
for k in sorted(set(a) | set(b)):
    if k == "graph_attrs":
        diff = sorted(x for x in set(a[k]) | set(b[k]) if a[k].get(x) != b[k].get(x))
        print("graph attrs differing:", diff)
        continue
    sa, sb = a.get(k), b.get(k)
    if sa is None or sb is None:
        print("DPU subgraph only in one:", k); same = False; continue
    da = sorted(x for x in set(sa["attrs"]) | set(sb["attrs"]) if sa["attrs"].get(x) != sb["attrs"].get(x))
    print(k, "attrs differing:", da)
    print(k, "consts identical:", sa["consts"] == sb["consts"], "(%d const ops)" % len(sa["consts"]))
    if sa["consts"] != sb["consts"] or any(x.startswith("mc_code") or x == "dpu_fingerprint" for x in da):
        same = False
print("DPU COMPUTE IDENTICAL" if same else "DPU COMPUTE DIFFERS")
