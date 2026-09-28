#!/usr/bin/env python3
"""Full-test-set RTL-sim data: ACT0 input images + expected logits for all test images.

    .venv/bin/python v2/fullsim/gen_full10k_data.py [--nets lenet5 cifar10] [--limit N]
                                                    [--out v2/build/fullsim/data] [--block 100]

Everything comes from the frozen v2/model (nothing is re-implemented here), exactly as
v2/scripts/gen_vectors.py gen_net_common / gen_network and v2/board/make_board_data.py:
  inputs   gos_golden.load_test_set + quantize_input, packed with gos_pack.pack_act_words
  golden   gos_golden.run_net -> raw INT32 v (LOGIT), PS float32 argmax (final_layer, D2)
  images   gos_pack.pack_wgt / pack_qparam / make_descriptors, written with gos_pack.write_hex
  cycles   gos_cycle_model.net_cycles (same expect_cycles() as gen_vectors.py)

Output per net, <out>/<net>/:
  wgt.hex qparam.hex desc.hex n_layers.hex expect_cyc.hex    net-level images (identical bytes
                                                            to v2/vectors/generated/<net>/...)
  sizes.hex         32-bit: n_in, wgt words, qparam words, n_layers, OC, block, n_images
  act0_b<k>.hex     images [k*B, (k+1)*B): n_in 64-bit lines per image, image-major
                    (image i = lines (i - k*B)*n_in ..; each image's lines == net/img<i>/act0.hex)
  logit16_b<k>.hex  same images: 16 32-bit lines per image (== net/img<i>/logit16.hex)
  golden.npz        v [n,OC] int32, pred [n], labels [n]  (collector: prediction check)
  meta.json         n_images, block, n_in, n_layers, OC, git commit/dirty, sha256 of every file
Self-checks: images 0..9 byte-identical to the committed-generator 10-image net vectors in
v2/vectors/generated (if present; their sha256 is in v2/vectors/MANIFEST.json), a few images
re-run through the tile model (gos_tile_model.run_net_mem), full-set golden accuracy equals the
accuracy of record (board_common.EXPECTED_GOLDEN_CORRECT) when all 10000 images are generated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

FULLSIM = Path(__file__).resolve().parent
V2 = FULLSIM.parent
sys.path.insert(0, str(V2 / "model"))

import common  # noqa: E402  (legacy python/ on sys.path)
from common import git_commit, git_dirty, sha256_file  # noqa: E402
import final_layer  # noqa: E402
import gos_cycle_model as cm  # noqa: E402
import gos_golden as gg  # noqa: E402
import gos_pack as gp  # noqa: E402
import gos_tile_model as tm  # noqa: E402
from net_config import NET_CONFIGS, NETS  # noqa: E402

DEFAULT_OUT = V2 / "build" / "fullsim" / "data"
REF_VEC = V2 / "vectors" / "generated"
REF_MANIFEST = V2 / "vectors" / "MANIFEST.json"
EXPECTED_GOLDEN_CORRECT = {"lenet5": 9879, "cifar10": 7852}   # = v2/board/board_common.py (D3)
BATCH = 500
FORMAT_VERSION = 1


def u32(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    assert x.size == 0 or (x.min() >= -(2**31) and x.max() < 2**31)
    return x.astype(np.int32).view(np.uint32)


def expect_cycles(layers) -> list[int]:
    """== v2/scripts/gen_vectors.py expect_cycles: LAYER_CYC[0..n-1], TOTAL_CYC, MAC_ACTIVE."""
    r = cm.net_cycles(layers)
    assert all(v is not None for v in (cm.C_PIPE, cm.C_START, cm.C_DONE))
    return [r["layers"][L["name"]]["cycles"] for L in layers] + [r["total"]["cycles"],
                                                                 r["total"]["mac_active"]]


def hex_text(values, width: int) -> str:
    return "\n".join(gp.hex_lines(values, width)) + "\n"


def build_net(net: str, out: Path, n: int, block: int) -> dict:
    t0 = time.time()
    d = out / net
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        if f.is_file():
            f.unlink()
    G = gg.load_net(net)
    OC = G.final.cfg["OC"]
    layers = list(NET_CONFIGS[net]["layers"])

    # ---- net-level images (as gen_vectors.gen_net_common) ----
    wgt, wb = gp.pack_wgt(net)
    qp, qb = gp.pack_qparam(net)
    descs, words = gp.make_descriptors(net)
    assert [x["WGT_BASE"] for x in descs] == wb and [x["QP_BASE"] for x in descs] == qb
    gp.write_hex(d / "wgt.hex", wgt, 64)
    gp.write_hex(d / "qparam.hex", qp, 64)
    gp.write_hex(d / "desc.hex", words.reshape(-1), 32)
    gp.write_hex(d / "n_layers.hex", [len(descs)], 32)
    gp.write_hex(d / "expect_cyc.hex", expect_cycles(layers), 32)
    n_in = gp.decode_descriptor(words[0])["IN_END"] + 1
    # simulator-portable sizes (no reliance on 'x fill to find a file's length)
    gp.write_hex(d / "sizes.hex", [n_in, wgt.size, qp.size, len(descs), OC, block, n], 32)

    # ---- inputs + golden ----
    print(f"[{net}] loading {n} test images ...", flush=True)
    x_f32, labels = gg.load_test_set(net, n)
    q = gg.quantize_input(G, x_f32)
    assert q.dtype == np.int8
    print(f"[{net}] golden (gos_golden.run_net) ...", flush=True)
    v = np.empty((n, OC), dtype=np.int32)
    pred = np.empty(n, dtype=np.int64)
    for s in range(0, n, BATCH):
        o = gg.run_net(G, q[s:s + BATCH])
        v[s:s + BATCH], pred[s:s + BATCH] = o["v"], o["pred"]
    ref_pred = final_layer.predict_from_raw(v, {"S_a": G.final.S_a, "S_w": G.final.S_w})
    assert np.array_equal(ref_pred, pred)
    np.savez_compressed(d / "golden.npz", v=v, pred=pred, labels=labels.astype(np.int64))

    # ---- per-block hex files ----
    print(f"[{net}] writing ACT0 / LOGIT blocks (block {block}) ...", flush=True)
    n_blocks = (n + block - 1) // block
    for k in range(n_blocks):
        a, b = k * block, min(n, (k + 1) * block)
        act = np.concatenate([gp.pack_act_words(q[i], n_in) for i in range(a, b)])
        v16 = np.zeros((b - a, gp.N_LOGITS), dtype=np.int64)
        v16[:, :OC] = v[a:b]
        (d / f"act0_b{k}.hex").write_text(hex_text(act, 64))
        (d / f"logit16_b{k}.hex").write_text(hex_text(u32(v16.reshape(-1)), 32))

    # ---- self-checks ----
    checks = {}
    checks["ref_vectors"] = check_against_ref(net, d, n, block, n_in)
    rng = np.random.default_rng([20260928, NETS.index(net)])
    sample = sorted({0, n - 1, *rng.integers(0, n, size=2).tolist()})
    for i in sample:                                  # tile model (hardware-order model) agrees
        r = tm.run_net_mem(net, q[i])
        assert np.array_equal(np.asarray(r["logits"], dtype=np.int64), v[i].astype(np.int64)), (net, i)
    checks["tile_model_images"] = sample
    correct = int((pred == labels).sum())
    checks["golden_correct"] = correct
    if n == 10000:
        assert correct == EXPECTED_GOLDEN_CORRECT[net], (net, correct)
        checks["golden_correct_matches_record"] = True

    files = sorted(p.name for p in d.iterdir() if p.is_file() and p.name != "meta.json")
    meta = {"format_version": FORMAT_VERSION, "net": net, "n_images": n, "block": block,
            "n_blocks": n_blocks, "n_in": int(n_in), "n_layers": len(descs), "OC": int(OC),
            "git_commit": git_commit(), "git_dirty": git_dirty(),
            "generator": "v2/fullsim/gen_full10k_data.py",
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "source": "model (gos_golden/gos_pack)",
            "checks": checks,
            "files": {f: sha256_file(d / f) for f in files}}
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[{net}] {n} images, {n_blocks} blocks, n_in {n_in}, golden correct {correct} [model]; "
          f"ref vectors: {checks['ref_vectors']}; {time.time() - t0:.1f} s", flush=True)
    return meta


def check_against_ref(net: str, d: Path, n: int, block: int, n_in: int) -> str:
    """Images 0..9 and the net-level files must equal gen_vectors.py output byte for byte."""
    ref = REF_VEC / net
    if not ref.is_dir():
        return "skipped (v2/vectors/generated absent; run gen_vectors.py)"
    man = json.loads(REF_MANIFEST.read_text())
    msha = {f["path"]: f["sha256"] for f in man["files"]}
    pairs = [(d / "wgt.hex", f"{net}/wgt.hex"), (d / "qparam.hex", f"{net}/qparam.hex"),
             (d / "desc.hex", f"{net}/desc.hex"), (d / "n_layers.hex", f"{net}/n_layers.hex"),
             (d / "expect_cyc.hex", f"{net}/net/expect_cyc.hex")]
    for mine, rel in pairs:
        assert mine.read_bytes() == (REF_VEC / rel).read_bytes(), f"{mine.name} != {rel}"
        assert sha256_file(mine) == msha[rel], f"{rel}: MANIFEST sha256 mismatch"
    act_lines = (d / "act0_b0.hex").read_text().splitlines()
    lg_lines = (d / "logit16_b0.hex").read_text().splitlines()
    nimg = 0
    for i in range(min(n, block, 10)):
        ra, rl = f"{net}/net/img{i}/act0.hex", f"{net}/net/img{i}/logit16.hex"
        mine_a = "\n".join(act_lines[i * n_in:(i + 1) * n_in]) + "\n"
        mine_l = "\n".join(lg_lines[i * 16:(i + 1) * 16]) + "\n"
        assert mine_a.encode() == (REF_VEC / ra).read_bytes(), ra
        assert mine_l.encode() == (REF_VEC / rl).read_bytes(), rl
        assert sha256_file(REF_VEC / ra) == msha[ra] and sha256_file(REF_VEC / rl) == msha[rl]
        nimg += 1
    return f"identical ({nimg} images + wgt/qparam/desc/n_layers/expect_cyc; MANIFEST sha256 ok)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nets", nargs="+", default=list(NETS), choices=NETS)
    ap.add_argument("--limit", type=int, default=10000, help="first N test images (default 10000)")
    ap.add_argument("--block", type=int, default=100, help="images per hex block file")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    assert 1 <= a.limit <= 10000 and a.block >= 1
    for net in a.nets:
        build_net(net, Path(a.out), a.limit, a.block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
