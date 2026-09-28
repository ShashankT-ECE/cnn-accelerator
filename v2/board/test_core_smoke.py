#!/usr/bin/env python3
"""Smoke test of the real gos_ core on the KV260 (or a dry run with --backend model).

    sudo -E python3 test_core_smoke.py [--bit bit/gos_200.bit] [--img 0] [--max-fclk0 200.5]
    python3 test_core_smoke.py --backend model          # laptop dry run (dryrun_model)

Checks, in order (each prints PASS/FAIL; exit 0 only if all pass):
  1. overlay loaded; VERSION == 0x474F5302 (GosDevice raises otherwise); BUILD_ID printed and
     compared with DEPLOY_INFO build_id when known; pl_clk0 read back <= --max-fclk0
  2. per net (LeNet-5, CIFAR-10): load_net (WGT/QPARAM/DESC written + read back), one image:
     LOGIT bit-exact vs golden_logits, prediction == golden_pred, LAYER_CYC / TOTAL_CYC /
     MAC_ACTIVE cycle-exact vs model_cycles.json, STALL == 0, PS_BUSY_VIOLATION == 0
  3. refused job (LeNet descriptors with layer 1 K = 4, refuse_test.json): STATUS.error,
     ERR_CODE == expected (rule 3, layer 1), TOTAL_CYC == C_START; then soft_reset -> STATUS 0;
     then the good descriptors again and one image, bit- and cycle-exact.
  4. with --host-path fast (Session 1 step s1.fast): GosDevice.check_fast_path() before and
     after the images (whole ACT0 window written with the fast vectorized copy and read back
     word by word + by block read, 3 patterns; LOGIT / LAYER_CYC block reads == per-word reads;
     memoryview scalar reads == MMIO reads), then checks 2-3 run on the fast path. PASS here is
     the bring-up condition for using --host-path fast in Sessions 2/3.
No CSV is written (bring-up check, not a result).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import board_common as bc
import gos_driver as D


def check(ok: bool, what: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}: {what}")
    return ok


def run_image(dev, pkg, i: int, tag: str) -> bool:
    r = dev.infer(pkg.x_act[i])
    mc = pkg.model_cycles
    ok = check(np.array_equal(r.logits, pkg.golden_logits[i]),
               f"{tag} img {i} logits {r.logits.tolist()} == golden")
    ok &= check(int(bc.predict(r.logits, pkg.dequant)) == int(pkg.golden_pred[i]),
                f"{tag} img {i} prediction {int(bc.predict(r.logits, pkg.dequant))} == golden "
                f"{int(pkg.golden_pred[i])} (label {int(pkg.labels[i])})")
    exp_layers = [L["cycles"] for L in mc["layers"]]
    ok &= check(r.layer_cyc == exp_layers, f"{tag} LAYER_CYC {r.layer_cyc} == model {exp_layers}")
    ok &= check(r.total_cyc == mc["total"]["cycles"],
                f"{tag} TOTAL_CYC {r.total_cyc} == model {mc['total']['cycles']}")
    ok &= check(r.mac_active == mc["total"]["mac_active"],
                f"{tag} MAC_ACTIVE {r.mac_active} == model sum(T*K) {mc['total']['mac_active']}")
    ok &= check(r.stall == 0, f"{tag} STALL {r.stall} == 0")
    ok &= check(r.violation == 0, f"{tag} PS_BUSY_VIOLATION 0x{r.violation:X} == 0 "
                f"{D.decode_violation(r.violation)}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap, nets=False)
    ap.add_argument("--img", type=int, default=0)
    ap.add_argument("--max-fclk0", type=float, default=200.5)
    a = ap.parse_args(argv)

    ok = True
    try:
        dev, be = bc.open_device(a)
    except D.GosVersionError as e:
        print(f"  FAIL: {e}")
        print("TEST FAILED")
        return 1
    info = bc.deploy_info()
    print(f"test_core_smoke: backend={be.kind} ({be.source}) {be.info()}")
    if be.source == bc.SOURCE_DRYRUN:
        print("  DRY RUN: model backend — checks the flow, not the hardware")
    ok &= check(dev.version == D.VERSION_CORE, f"VERSION 0x{dev.version:08X}")
    print(f"  BUILD_ID = 0x{dev.build_id:08X}")
    if info.get("build_id") and be.kind == "pynq":
        ok &= check(dev.build_id_hex == str(info["build_id"]).lower(),
                    f"BUILD_ID {dev.build_id_hex} == DEPLOY_INFO build_id {info['build_id']}")
    fclk = dev.fclk0_mhz()
    ok &= check(fclk <= a.max_fclk0, f"pl_clk0 {fclk:.3f} MHz <= {a.max_fclk0} MHz "
                f"({'readback' if be.kind == 'pynq' else 'nominal'})")
    st = dev.status()
    print(f"  STATUS at start = 0x{st:X}")
    print(f"  host path: {dev.host_path_desc}")

    def fast_check(when: str) -> bool:
        try:
            chk = dev.check_fast_path()
        except Exception as e:  # noqa: BLE001 - any failure means: fast path not validated
            return check(False, f"fast path check ({when}): {type(e).__name__}: {e}")
        return check(True, f"fast path check ({when}): {chk['patterns']} patterns x "
                     f"{chk['act0_words_checked'] // chk['patterns']} ACT0 words written (fast "
                     f"{chk['fast_store']}) + read back per word and by block; "
                     f"{chk['csr_block_words_checked']} CSR block words == per-word ({chk['windows']})")
    if dev.host_path == "fast":
        ok &= fast_check("before any job")
        if not ok:
            print("TEST FAILED")
            return 1

    pkgs = {net: bc.load_package(a.data_dir, net) for net in bc.NETS}
    for net, pkg in pkgs.items():
        print(f"[{net}] manifest OK ({pkg.manifest_sha256[:12]}), commit "
              f"{pkg.manifest['git_commit'][:8]} dirty={pkg.manifest['git_dirty']}")
        dev.load_net(pkg)
        ok &= check(True, f"{net} load_net: WGT {pkg.wgt.size} + QPARAM {pkg.qparam.size} words, "
                    f"{pkg.desc.shape[0]} descriptors written and read back")
        ok &= run_image(dev, pkg, a.img, net)
        if dev.host_path == "fast":
            ok &= fast_check(f"after a {net} job (LOGIT/LAYER_CYC non-zero)")

    # refused job, soft_reset, good job
    pkg = pkgs["lenet5"]
    rt = json.loads((pkg.net_dir / "refuse_test.json").read_text())
    print(f"[refuse] {rt['note']}")
    dev.load_net(pkg)
    bad = pkg.desc.copy()
    bad[rt["layer"], rt["word"]] = rt["value"]
    dev.write_descriptors(bad, rt["n_layers"])
    try:
        dev.start_and_wait()
        ok &= check(False, "refused job: expected STATUS.error, got done")
    except D.GosJobError as e:
        print(f"  {e}")
        ok &= check(e.err_code == rt["expected_err_code"],
                    f"ERR_CODE 0x{e.err_code:08X} == expected 0x{rt['expected_err_code']:08X} "
                    f"(rule {e.info['rule_id']} '{e.info['rule']}', layer {e.info['layer']})")
        ok &= check(e.status & D.ST_ERROR and not e.status & (D.ST_BUSY | D.ST_DONE),
                    f"STATUS 0x{e.status:X} = error only")
        tc = dev.read64(D.OFF_TOTAL_CYC)
        ok &= check(tc == rt["expected_total_cyc"], f"TOTAL_CYC {tc} == C_START "
                    f"{rt['expected_total_cyc']} (model)")
    except D.GosTimeout as e:
        ok &= check(False, f"refused job timed out: {e}")
    dev.soft_reset()
    ok &= check(dev.status() == 0, "soft_reset -> STATUS 0")
    ok &= check(dev.err_code() == 0, "soft_reset -> ERR_CODE 0")
    ok &= run_image(dev, pkg, a.img, "after-reset lenet5")   # infer() rewrites the descriptors

    print("TEST PASSED" if ok else "TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
