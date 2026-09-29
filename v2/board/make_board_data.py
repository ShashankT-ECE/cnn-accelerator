#!/usr/bin/env python3
"""Build the board data package v2/board/data/<net>/ (laptop only; repo .venv).

    ~/cnn-accelerator/.venv/bin/python v2/board/make_board_data.py [--nets lenet5 cifar10]
                                                                    [--limit N] [--allow-dirty]

Everything comes from v2/model (no format re-implementation here):
  inputs    gos_golden.load_test_set + quantize_input (legacy preprocessing + quantizer),
            packed to the ACT0 image with gos_pack.pack_act_words
  golden    gos_golden.run_net -> raw INT32 v; PS float32 dequant/argmax (final_layer, D2)
  images    gos_pack.pack_wgt / pack_qparam / make_descriptors
  cycles    gos_cycle_model.net_cycles (label: model); RTL cycles = copies of
            v2/results/rtl_cycles.csv and rtl_network.csv (label: rtl_sim)
  shapes    (A3-general, exp_shapes.py) the random shape set v2/build/shapes/<seed>/ (other owner,
            v2/shapes/gen_shapes.py + shapeset.py): shapes.json + every jobs/*.npz it lists (not
            hex/, runs/) copied to data/shapes/ with SHIP.json (SHA256 + size of every shipped
            file, seed, n_jobs, categories, the set's shapeset_sha256, git provenance) + a copy of
            v2/results/shapes_rtl.csv if present (label: rtl_sim); PACKAGE.json "shapes" records
            SHIP.json's SHA256.
              --shapes auto (default: the only v2/build/shapes/<digits>/ holding shapes.json, none
              -> skipped with a note, several -> error) | none | <seed> | <dir>;  --shapes-only: ship the shape set
              and rewrite PACKAGE.json without rebuilding the net packages.
Layout: v2/board/README.md "Data package". Refuses a dirty git tree unless --allow-dirty (the
manifest then records git_dirty=true and hardware runs using it refuse without --allow-dirty).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

BOARD = Path(__file__).resolve().parent
V2 = BOARD.parent
sys.path.insert(0, str(V2 / "model"))
sys.path.insert(0, str(BOARD))

import board_common as bc  # noqa: E402
import common  # noqa: E402,F401  (legacy python/ on sys.path)
import final_layer  # noqa: E402
import gos_cycle_model as cm  # noqa: E402
import gos_golden as gg  # noqa: E402
import gos_pack as gp  # noqa: E402
from net_config import NET_CONFIGS, NETS  # noqa: E402

BATCH = 500


def _save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=1) + "\n")


def _fp32_params(net: str) -> dict:
    import torch
    sd = torch.load(common.REPO_ROOT / NET_CONFIGS[net]["checkpoint"], map_location="cpu",
                    weights_only=True)
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}


def refuse_test(net: str, desc: np.ndarray) -> dict:
    """A descriptor set the RTL checker must refuse: layer 1 K (w7) = 4 (< 8, rule 3)."""
    bad = desc.copy()
    layer, word, value = 1, 7, 4
    bad[layer, word] = value
    code = gp.job_err_code(bad.shape[0], bad)
    assert code == (3 << 8) | layer, hex(code)
    return {"layer": layer, "word": word, "value": value, "n_layers": int(bad.shape[0]),
            "expected_err_code": code, "expected_rule_id": 3,
            "expected_total_cyc": cm.C_START,
            "note": "descriptor w7 (K) of layer 1 set to 4; checker rule 3 (K < 8); "
                    "expected from gos_pack.job_err_code"}


def build_net(net: str, out: Path, limit: int | None, commit: str, dirty: bool) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        if f.is_file():
            f.unlink()
    n = 10000 if limit is None else limit
    G = gg.load_net(net)
    print(f"[{net}] loading {n} test images ...", flush=True)
    x_f32, labels = gg.load_test_set(net, n)
    q = gg.quantize_input(G, x_f32)
    assert q.dtype == np.int8

    print(f"[{net}] golden (gos_golden.run_net) ...", flush=True)
    v = np.empty((n, G.final.cfg["OC"]), dtype=np.int32)
    lf32 = np.empty(v.shape, dtype=np.float32)
    pred = np.empty(n, dtype=np.int64)
    for s in range(0, n, BATCH):
        o = gg.run_net(G, q[s:s + BATCH])
        v[s:s + BATCH], lf32[s:s + BATCH], pred[s:s + BATCH] = o["v"], o["logits"], o["pred"]

    dq = {"layer": G.final.name, "S_a": float(G.final.S_a),
          "S_w": [float(s) for s in G.final.S_w],
          "scale": [float(s) for s in (G.final.S_a * G.final.S_w)],
          "formula": "logits = (v.astype(int64) * (S_a*S_w).reshape(1,-1)).astype(float32); "
                     "pred = logits.argmax(1)  (legacy int8_model fc expression, DECISIONS D2)"}
    # board-side dequant == v2/model/final_layer, bit for bit
    ref = final_layer.logits_from_raw(v, {"S_a": G.final.S_a, "S_w": G.final.S_w})
    mine = bc.final_dequant(v, json.loads(json.dumps(dq)))
    assert np.array_equal(ref.view(np.uint32), mine.view(np.uint32)), "board dequant differs"
    assert np.array_equal(ref.view(np.uint32), lf32.view(np.uint32))
    assert np.array_equal(bc.predict(v, dq), pred)

    print(f"[{net}] packing ACT images ...", flush=True)
    D = gp.act_depth(*q.shape[1:])
    x_act = np.empty((n, 8 * D), dtype=np.int8)
    for i in range(n):
        x_act[i] = gp.pack_act_words(q[i]).view(np.int8)
    for i in {0, n // 2, n - 1}:
        assert np.array_equal(gp.unpack_act(x_act[i].view(np.uint64), *q.shape[1:]), q[i])
    np.savez_compressed(out / "inputs_act.npz", x=x_act, x_nchw=q, x_f32=x_f32.astype(np.float32))

    np.save(out / "labels.npy", labels.astype(np.int64))
    np.save(out / "golden_logits.npy", v)
    np.save(out / "golden_logits_f32.npy", lf32)
    np.save(out / "golden_pred.npy", pred)

    wgt, wb = gp.pack_wgt(net)
    qpar, qb = gp.pack_qparam(net)
    descs, words = gp.make_descriptors(net)
    np.save(out / "wgt.npy", wgt.astype(np.uint64))
    np.save(out / "qparam.npy", qpar.astype(np.uint64))
    np.save(out / "desc.npy", words.astype(np.uint32))

    layers = [{k: (bool(L[k]) if k in ("relu", "pool", "final") else L[k])
               for k in ("name", "IC", "OC", "IH", "IW", "KH", "KW", "OH", "OW", "K",
                         "relu", "pool", "final")}
              | {"in_sel": d["in_sel"], "WGT_BASE": d["WGT_BASE"], "QP_BASE": d["QP_BASE"]}
              for L, d in zip(NET_CONFIGS[net]["layers"], descs)]
    L0 = layers[0]
    _save_json(out / "net.json", {
        "net": net, "reference_version": NET_CONFIGS[net]["reference_version"],
        "dataset": NET_CONFIGS[net]["dataset"], "n_images": n, "n_layers": len(layers),
        "OC": layers[-1]["OC"], "input_shape": [L0["IC"], L0["IH"], L0["IW"]],
        "act_in_words": D, "S_input": G.S_input, "B": G.B, "layers": layers,
        "wgt_words": int(wgt.size), "qparam_words": int(qpar.size)})
    _save_json(out / "final_dequant.json", dq)
    shutil.copyfile(NET_CONFIGS[net]["quant_params"], out / "quant_params.npz")
    shutil.copyfile(NET_CONFIGS[net]["hw_requant"], out / "hw_requant.npz")
    np.savez(out / "fp32_params.npz", **_fp32_params(net))

    mc = cm.net_cycles(NET_CONFIGS[net]["layers"])
    _save_json(out / "model_cycles.json", {
        "label": "model (gos_cycle_model; LAYER_CYC = T*K + C_PIPE)",
        "C_PIPE": cm.C_PIPE, "C_START": cm.C_START, "C_DONE": cm.C_DONE, "PE_COUNT": cm.PE_COUNT,
        "layers": [{"name": nm, **{k: r[k] for k in ("T", "K", "compute_cycles", "mac_active",
                                                    "stall", "macs", "util_theoretical",
                                                    "cycles")}}
                   for nm, r in mc["layers"].items()],
        "total": {k: mc["total"][k] for k in ("T", "compute_cycles", "mac_active", "stall", "macs",
                                              "util_theoretical", "cycles")}})
    for f in ("rtl_cycles.csv", "rtl_network.csv"):
        shutil.copyfile(V2 / "results" / f, out / f)
    _save_json(out / "refuse_test.json", refuse_test(net, words))

    correct = int((pred == labels).sum())
    files = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "MANIFEST.json")
    man = {"net": net, "n_images": n, "limit": limit,
           "git_commit": commit, "git_dirty": dirty, "created_utc": bc.utc_now(),
           "generator": "v2/board/make_board_data.py",
           "source": "model (gos_golden / gos_pack / gos_cycle_model); rtl_*.csv = rtl_sim copies",
           "golden_correct": correct, "golden_accuracy_pct": round(100.0 * correct / n, 2),
           "files": {f: {"sha256": bc.sha256_file(out / f), "bytes": (out / f).stat().st_size}
                     for f in files}}
    _save_json(out / "MANIFEST.json", man)
    bc.verify_manifest(out)
    dt = time.time() - t0
    exp = bc.EXPECTED_GOLDEN_CORRECT[net] if limit is None else None
    print(f"[{net}] {n} images; golden correct {correct} ({man['golden_accuracy_pct']}%) "
          f"[model]; expected (D3) {exp}; ACT depth {D} words; WGT {wgt.size} words; "
          f"QPARAM {qpar.size} words; {len(layers)} layers; {dt:.1f} s")
    if exp is not None and correct != exp:
        raise SystemExit(f"[{net}] golden accuracy {correct} != accuracy of record {exp}")
    return man


SHAPES_BUILD = V2 / "build" / "shapes"
SHAPES_RTL_CSV = V2 / "results" / "shapes_rtl.csv"


def resolve_shapes_src(arg: str) -> Path | None:
    if arg == "none":
        return None
    if arg == "auto":
        subs = sorted(d for d in SHAPES_BUILD.iterdir() if d.is_dir() and d.name.isdigit()
                      and (d / "shapes.json").is_file()) if SHAPES_BUILD.is_dir() else []
        if not subs:
            print(f"NOTE: no shape set under {SHAPES_BUILD}: data/shapes not shipped "
                  "(A3-general step s2.shapes will not be registered)")
            return None
        if len(subs) > 1:
            raise SystemExit(f"several shape sets under {SHAPES_BUILD} ({[d.name for d in subs]}): "
                             "pass --shapes <seed>")
        return subs[0]
    p = Path(arg)
    return p if p.is_dir() else SHAPES_BUILD / arg


def ship_shapes(src: Path, out: Path, commit: str, dirty: bool,
                rtl_csv: Path = SHAPES_RTL_CSV) -> dict:
    """Copy the shape set src into out (= data/shapes) and write SHIP.json."""
    import exp_shapes as xs
    import shapeset
    if not (src / xs.SHAPES_JSON).is_file():
        raise SystemExit(f"{src}: no {xs.SHAPES_JSON} (not a shape set)")
    man = shapeset.verify_manifest(src)
    shapes = shapeset.load_shapeset(src)
    if out.is_dir():
        shutil.rmtree(out)                      # generated copy (gitignored data package)
    out.mkdir(parents=True)
    for rel in [xs.SHAPES_JSON, *sorted(man["files"])]:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, out / rel)
    rtl_rows = None
    if rtl_csv.is_file():
        shutil.copyfile(rtl_csv, out / xs.RTL_CSV)
        rtl_rows = sum(1 for _ in rtl_csv.open()) - 1
    set_dirty = man.get("git_dirty")
    cats = {}
    for sh in shapes:
        c = str(xs._f(sh, "category", ""))
        cats[c] = cats.get(c, 0) + 1
    files = sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file()
                   and f.name != xs.SHIP_NAME)
    rel = src.resolve()
    rel = str(rel.relative_to(V2.parent)) if rel.is_relative_to(V2.parent) else str(rel)
    ship = {"seed": str(man.get("seed", src.name)), "source_dir": rel,
            "n_jobs": len(shapes), "categories": cats,
            "shapeset_sha256": man["shapeset_sha256"],
            "shapes_json_sha256": bc.sha256_file(out / xs.SHAPES_JSON),
            "set_git_commit": man.get("git_commit", ""),
            "set_git_dirty": set_dirty,
            "git_commit": commit, "git_dirty": bool(dirty) or bool(set_dirty),
            "created_utc": bc.utc_now(), "generator": "v2/board/make_board_data.py (ship_shapes)",
            "rtl_csv": xs.RTL_CSV if rtl_rows is not None else "", "rtl_csv_rows": rtl_rows,
            "source": "shape set = model (golden outputs, model cycles); shapes_rtl.csv = rtl_sim copy",
            "files": {f: {"sha256": bc.sha256_file(out / f), "bytes": (out / f).stat().st_size}
                      for f in files}}
    _save_json(out / xs.SHIP_NAME, ship)
    xs.verify_ship(out)
    print(f"[shapes] shipped {len(shapes)} jobs ({len(cats)} categories) from {src} -> {out}; "
          f"set {ship['shapeset_sha256'][:12]}; RTL csv: "
          f"{'%d rows' % rtl_rows if rtl_rows is not None else 'not present'}")
    return ship


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nets", nargs="+", default=list(NETS), choices=NETS)
    ap.add_argument("--limit", type=int, default=None, help="first N test images (default 10000)")
    ap.add_argument("--out", default=str(bc.DEFAULT_DATA_DIR))
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--shapes", default="auto",
                    help="shape set to ship: auto | none | <seed> | <dir> (A3-general)")
    ap.add_argument("--shapes-only", action="store_true",
                    help="only ship the shape set and rewrite PACKAGE.json (net packages kept)")
    a = ap.parse_args(argv)
    commit, dirty = bc.git_state()
    if dirty and not a.allow_dirty:
        print("REFUSED: git tree is dirty (commit first, or --allow-dirty to build a package "
              "marked git_dirty=true)")
        return 1
    out = Path(a.out)
    if not a.shapes_only:
        for net in a.nets:
            build_net(net, out / net, a.limit, commit, dirty)
    pk = {"nets": {}, "created_utc": bc.utc_now(), "git_commit": commit, "git_dirty": dirty}
    for net in NETS:
        m = out / net / "MANIFEST.json"
        if m.is_file():
            pk["nets"][net] = {"manifest_sha256": bc.sha256_file(m)}
            if a.shapes_only:     # nets not rebuilt here: keep their own dirty flag
                pk["git_dirty"] = pk["git_dirty"] or bool(json.loads(m.read_text()).get("git_dirty", True))
    src = resolve_shapes_src(a.shapes)
    if src is not None:
        ship = ship_shapes(src, out / "shapes", commit, dirty)
        pk["shapes"] = {"ship_sha256": bc.sha256_file(out / "shapes" / "SHIP.json"),
                        "shapeset_sha256": ship["shapeset_sha256"], "n_jobs": ship["n_jobs"],
                        "seed": ship["seed"]}
        pk["git_dirty"] = pk["git_dirty"] or ship["git_dirty"]
    elif (out / "shapes").is_dir():
        print(f"WARNING: {out / 'shapes'} exists but is not listed in PACKAGE.json: s2.shapes and "
              "deploy.sh will refuse it (remove the directory or ship a set with --shapes)")
    _save_json(out / "PACKAGE.json", pk)
    print(f"wrote {out}/PACKAGE.json (commit {commit[:8]}, dirty={dirty})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
