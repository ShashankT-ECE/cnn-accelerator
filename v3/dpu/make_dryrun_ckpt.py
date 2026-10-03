#!/usr/bin/env python3
"""THROWAWAY checkpoint for the baseline DRY RUN (laptop, repo .venv). NEVER data.

    .venv/bin/python v3/dpu/make_dryrun_ckpt.py [--nets resnet20_b mobilenet_s] [--steps 1]

Random init (seeded) of v3/model/nets.build(net); BatchNorm running statistics are set by forward
passes in train mode over train[0:1024] (no gradient), plus --steps SGD steps of one 128-image batch
(default 1), so activations have sane ranges for vai_q / ORT calibration. Accuracy is ~chance.
Writes v3/dpu/build/dryrun_ckpt/<net>.pt (gitignored) as {"net", "state_dict", "dryrun_throwaway":
True, "note"}; every downstream manifest carries checkpoint_kind = dryrun_throwaway and the board
session marks such rows paper_grade = False.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DPU = Path(__file__).resolve().parent
V3 = DPU.parent
sys.path.insert(0, str(V3 / "board"))
sys.path.insert(0, str(V3 / "model"))

import laptop_common as lc  # noqa: E402
import baseline_common as bc  # noqa: E402
import nets  # noqa: E402

OUT = DPU / "build" / "dryrun_ckpt"


def make(net: str, steps: int, seed: int) -> Path:
    import torch
    torch.manual_seed(seed)
    m = nets.build(net)
    x_u8, y = lc.cifar_train_head(lc.CALIB_N)
    x = torch.from_numpy(bc.preprocess(x_u8))
    m.train()
    with torch.no_grad():
        for s in range(0, x.shape[0], 128):
            m(x[s:s + 128])
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    for k in range(steps):
        xb, yb = x[(k * 128) % 1024:(k * 128) % 1024 + 128], torch.from_numpy(y[(k * 128) % 1024:][:128])
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(xb), yb)
        loss.backward()
        opt.step()
    m.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{net}.pt"
    torch.save({"net": net, "state_dict": m.state_dict(), "dryrun_throwaway": True,
                "note": f"DRY RUN throwaway: seed {seed}, BN stats from train[0:1024], {steps} SGD step(s); "
                        "not trained, never data"}, p)
    print(f"[{net}] wrote {p} (THROWAWAY, dry run only)")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nets", nargs="+", default=["resnet20_b", "mobilenet_s"], choices=nets.NETS)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    for n in a.nets:
        make(n, a.steps, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
