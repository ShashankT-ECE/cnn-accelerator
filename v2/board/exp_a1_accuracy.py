#!/usr/bin/env python3
"""A1 correctness: every test image of each net on the accelerator vs the golden model.

    sudo -E python3 exp_a1_accuracy.py [--nets lenet5 cifar10] [--limit N]
    python3 exp_a1_accuracy.py --backend model        # dry run -> results/dryrun/

Per net: images run, images whose INT32 logits differ from golden_logits (and the number of
differing logit values), prediction mismatches (PS float32 dequant + argmax, D2), golden accuracy
(model, from the package), accelerator accuracy, job errors. Writes hw_a1_accuracy.csv and
hw_logits_<net>.npz (logits, predictions, per-image TOTAL_CYC) next to it.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import board_common as bc

FIELDS = ["images", "logit_mismatch_images", "logit_mismatch_values", "pred_mismatches",
          "golden_correct", "golden_accuracy_pct", "hw_correct", "hw_accuracy_pct",
          "job_errors", "first_mismatch_img", "golden_accuracy_of_record", "golden_matches_record",
          "wall_s"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap)
    a = ap.parse_args(argv)
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "A1")
    ctx.check_clean()
    ctx.banner()
    rows, ok = [], True
    for net in a.nets:
        pkg = ctx.package(net)
        dev.load_net(pkg)
        clk = dev.fclk0_mhz()
        idx = bc.image_range(pkg, a.limit)
        res = bc.infer_loop(dev, pkg, idx, read_counters=True, tag=f"A1 {net}")
        n = len(idx)
        gl, gp_, lab = pkg.golden_logits[:n], pkg.golden_pred[:n], pkg.labels[:n]
        diff = res["logits"] != gl
        bad_img = diff.any(axis=1)
        pred = np.where(res["ok"], bc.predict(res["logits"], pkg.dequant), -1)
        pred_mm = int((pred != gp_).sum())
        g_corr, h_corr = int((gp_ == lab).sum()), int((pred == lab).sum())
        rec = bc.EXPECTED_GOLDEN_CORRECT[net] if n == 10000 else ""
        row = ctx.meta(net, "all", res["duration_s"], n, clock_mhz=clk)
        row.update(images=n, logit_mismatch_images=int(bad_img.sum()),
                   logit_mismatch_values=int(diff.sum()), pred_mismatches=pred_mm,
                   golden_correct=g_corr, golden_accuracy_pct=f"{100.0 * g_corr / n:.2f}",
                   hw_correct=h_corr, hw_accuracy_pct=f"{100.0 * h_corr / n:.2f}",
                   job_errors=len(res["errors"]),
                   first_mismatch_img=int(np.flatnonzero(bad_img)[0]) if bad_img.any() else "",
                   golden_accuracy_of_record=rec,
                   golden_matches_record=(g_corr == rec) if rec != "" else "",
                   wall_s=f"{res['duration_s']:.3f}")
        rows.append(row)
        np.savez_compressed(ctx.path(f"hw_logits_{net}.npz"), idx=res["idx"], logits=res["logits"],
                            pred=pred, ok=res["ok"], total_cyc=res["total_cyc"],
                            source=np.array(ctx.source))
        print(f"[A1 {net}] ({ctx.source}) images {n}; logit-mismatch images {int(bad_img.sum())}; "
              f"pred mismatches {pred_mm}; golden {g_corr}/{n} = {row['golden_accuracy_pct']}% "
              f"(record {rec}); accelerator {h_corr}/{n} = {row['hw_accuracy_pct']}%; "
              f"job errors {len(res['errors'])}; {res['duration_s']:.1f} s")
        ok &= not bad_img.any() and pred_mm == 0 and not res["errors"] and (rec == "" or g_corr == rec)
    ctx.csv("hw_a1_accuracy.csv", rows, FIELDS)
    print("A1:", "PASS (0 mismatches)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
