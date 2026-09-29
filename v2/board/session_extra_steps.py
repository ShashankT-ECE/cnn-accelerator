"""Extra session steps for run_sessions.py (publication experiments; agreed interface (a)).

    extra_steps(opts) -> list[dict]

Each dict: id, session (2 or 3), group ("A3","A1","latency","shapes","baselines","energy","sweep",
"soak","A4"), title, cmd (argv relative to the board dir; the orchestrator appends `out_flag <staging>`),
out_flag ("--out-dir"), outputs (result file names that must exist after an OK run), est_s,
timeout_s, requires (step ids that must be recorded OK first), required (False = informational),
params (dict; part of the resume key), kind ("fixed" / "infer"), images (accelerator inferences,
for rate learning).

Registered here:
  s3.soak         session 3, group "soak"  exp_soak.py (30 min back-to-back, both nets alternating)
  s2.layerspread  session 2, group "A3"    exp_layer_spread.py (reuses s2.A2A3's
                                           hw_cycles_<net>.npz from the results dir when it covers
                                           every image at the same clock; otherwise runs 10k images
                                           per net itself). Not `requires` s2.A2A3, so that a failed
                                           A2/A3 does not block the determinism evidence; schedule
                                           it after s2.A2A3 (group order) to avoid a duplicate run.
  s2.shapes      session 2, group "shapes" exp_shapes.py (A3-general: the shipped random multi-layer
                                           shape set, bit/cycle-exact vs model and RTL sim; hard
                                           15 min cap inside the script, stratified job order).
                                           Registered only if <data_dir>/shapes/SHIP.json exists
                                           (make_board_data.py --shapes); requires s1.smoke (+ s1.fast
                                           on the fast host path).
The B2 sweep stays the orchestrator's s3.B2 (exp_b2_clock.py); it now also writes
hw_b2_cycles.csv and hw_b2_fit.csv (B2_EXTRA_OUTPUTS, to add to s3.B2's expected outputs).

opts: a dict, an argparse Namespace or run_sessions.Config-like object; attributes/keys read
(missing -> default): backend ('pynq'|'model'), data_dir, bit, closed_mhz (or deploy
{'bit_clock_mhz'}), results_dir, allow_dirty, quick, write_mode ('elem'), host_path ('safe'),
fast_store ('block'), board_id, dev_args (optional: prebuilt device argv, replaces the one built
here), soak_s (default 1800 s; --quick 300 s; dry run 60 s), soak_nets (both), soak_block_s (60),
spread_limit (None = all images; --quick 200), shapes_budget_s (900 s; --quick 300 s; never above
900), shapes_seed (job-order seed; default order_seed + 30), clock_choice (path of the clock_fallback decision
JSON; default <results_dir>/hw_clock_choice.json).
"""
from __future__ import annotations

from pathlib import Path

import clock_fallback as cf

NETS = ("lenet5", "cifar10")
SOAK_S_DEFAULT = {"hw": 1800.0, "quick": 300.0, "dry": 60.0}
SOAK_OUTPUTS = ("hw_soak.csv", "hw_soak_minutes.csv")
SPREAD_OUTPUTS = ("hw_layer_spread.csv",)
B2_EXTRA_OUTPUTS = ("hw_b2_cycles.csv", "hw_b2_fit.csv")
SETUP_S = 60.0                  # overlay load, package verify, net loads
RATE_S = {"pynq": 0.02, "model": 0.006}   # = run_sessions.DEFAULT_RATE_S (pynq value ASSUMED)
SHAPES_OUTPUTS = ("hw_shapes.csv",)
SHAPES_CAP_S = 900.0            # = exp_shapes.BUDGET_MAX_S (hard cap inside the script)
SHAPES_QUICK_S = 300.0
# per-job estimate: WGT/QPARAM/ACT0 writes + per-word readback + output readback. pynq value
# ASSUMED (safe path, up to 16k WGT words written and verified), replaced by the recorded duration
# after the first run (run_sessions.estimate "previous run").
SHAPES_JOB_S = {"pynq": 1.5, "model": 0.3}
SHAPES_TIMEOUT_MARGIN_S = 180.0  # script setup (overlay, set verify/load) + the job in flight


def _opt(opts, name, default=None):
    if opts is None:
        return default
    if isinstance(opts, dict):
        v = opts.get(name, default)
    else:
        v = getattr(opts, name, default)
    return default if v is None else v


def _closed(opts):
    c = _opt(opts, "closed_mhz")
    if c is None:
        dep = _opt(opts, "deploy", {}) or {}
        c = dep.get("bit_clock_mhz")
    return None if c is None else float(c)


def device_args(opts) -> list[str]:
    """--backend/--data-dir/(--clock-mhz | --bit)/--board-id/--allow-dirty/--write-mode/host path."""
    be = _opt(opts, "backend", "pynq")
    dev = list(_opt(opts, "dev_args", []) or [])
    if not dev:
        dev = ["--backend", be, "--data-dir", str(_opt(opts, "data_dir", "data"))]
        closed = _closed(opts)
        if be == "model":
            if closed is not None:
                dev += ["--clock-mhz", f"{closed:.6f}"]
        elif _opt(opts, "bit"):
            dev += ["--bit", str(_opt(opts, "bit"))]
        if _opt(opts, "board_id"):
            dev += ["--board-id", str(_opt(opts, "board_id"))]
    if _opt(opts, "allow_dirty", False) and "--allow-dirty" not in dev:
        dev += ["--allow-dirty"]
    dev += ["--write-mode", _opt(opts, "write_mode", "elem"),
            "--host-path", _opt(opts, "host_path", "safe"),
            "--fast-store", _opt(opts, "fast_store", "block")]
    return dev


def n_images(opts, net: str) -> int:
    import json
    try:
        return int(json.loads((Path(_opt(opts, "data_dir", "data")) / net / "MANIFEST.json")
                              .read_text())["n_images"])
    except (OSError, KeyError, ValueError):
        return 10000


def soak_step(opts) -> dict:
    be = _opt(opts, "backend", "pynq")
    dry, quick = be == "model", bool(_opt(opts, "quick", False))
    dur = float(_opt(opts, "soak_s", SOAK_S_DEFAULT["dry" if dry else "quick" if quick else "hw"]))
    nets = list(_opt(opts, "soak_nets", NETS))
    block = float(_opt(opts, "soak_block_s", 60.0))
    rd = _opt(opts, "results_dir")
    choice = _opt(opts, "clock_choice", str(Path(rd) / cf.CHOICE_FILE) if rd else None)
    closed = _closed(opts)
    params = {"duration_s": dur, "nets": nets, "block_s": block, "bucket_s": 60.0,
              "expect_clock_mhz": closed, "host_path": _opt(opts, "host_path", "safe")}
    cmd = ["exp_soak.py", *device_args(opts), "--nets", *nets, "--duration-s", f"{dur:g}",
           "--block-s", f"{block:g}", "--bucket-s", "60"]
    if closed is not None:
        cmd += ["--expect-clock-mhz", f"{closed:.6f}"]
    if choice:
        cmd += ["--clock-choice", str(choice)]
    est = dur + SETUP_S
    return {"id": "s3.soak", "session": 3, "group": "soak",
            "title": f"soak: {dur:g} s back-to-back inference ({'/'.join(nets)} alternating every "
                     f"{block:g} s), bit/cycle-exact every job, per-minute throughput, die temperature",
            "cmd": cmd, "out_flag": "--out-dir", "outputs": list(SOAK_OUTPUTS), "est_s": est,
            "timeout_s": dur * 1.2 + 300.0, "requires": ["s1.fast"] if params["host_path"] == "fast" else [],
            "required": True, "params": params, "kind": "fixed", "images": 0}


def spread_step(opts) -> dict:
    be = _opt(opts, "backend", "pynq")
    quick = bool(_opt(opts, "quick", False))
    lim = _opt(opts, "spread_limit", 200 if quick else None)
    rd = _opt(opts, "results_dir")
    n = sum(min(lim, n_images(opts, net)) if lim else n_images(opts, net) for net in NETS)
    cmd = ["exp_layer_spread.py", *device_args(opts)]
    if rd:
        cmd += ["--npz-dir", str(rd)]
    if lim:
        cmd += ["--limit", str(lim)]
    params = {"limit": lim, "host_path": _opt(opts, "host_path", "safe"), "npz_dir": str(rd or "")}
    # estimate = the worst case (no reusable npz: own run of every image)
    est = n * RATE_S.get(be, RATE_S["pynq"]) + SETUP_S
    return {"id": "s2.layerspread", "session": 2, "group": "A3",
            "title": "per-layer cycle spread over every image (min/max/distinct per layer; reuses "
                     "s2.A2A3 hw_cycles_<net>.npz when complete)",
            "cmd": cmd, "out_flag": "--out-dir", "outputs": list(SPREAD_OUTPUTS), "est_s": est,
            "timeout_s": max(300.0, 3 * est + 120.0),
            "requires": ["s1.fast"] if params["host_path"] == "fast" else [],
            "required": True, "params": params, "kind": "infer", "images": n}


def shapes_ship(opts) -> dict | None:
    import json
    p = Path(_opt(opts, "data_dir", "data")) / "shapes" / "SHIP.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def shapes_step(opts) -> dict | None:
    """s2.shapes (A3-general) or None when no shape set is shipped in the data package."""
    ship = shapes_ship(opts)
    if ship is None:
        return None
    be = _opt(opts, "backend", "pynq")
    quick = bool(_opt(opts, "quick", False))
    budget = min(float(_opt(opts, "shapes_budget_s", SHAPES_QUICK_S if quick else SHAPES_CAP_S)),
                 SHAPES_CAP_S)
    seed = int(_opt(opts, "shapes_seed", int(_opt(opts, "order_seed", 1)) + 30))
    n = int(ship.get("n_jobs") or 300)
    est = min(SETUP_S + n * SHAPES_JOB_S.get(be, SHAPES_JOB_S["pynq"]), budget)
    hp = _opt(opts, "host_path", "safe")
    cmd = ["exp_shapes.py", *device_args(opts), "--budget-s", f"{budget:g}",
           "--order-seed", str(seed)]
    params = {"budget_s": budget, "order_seed": seed, "host_path": hp, "n_jobs": n,
              "shapeset_sha256": ship.get("shapeset_sha256", ""), "seed": ship.get("seed", "")}
    return {"id": "s2.shapes", "session": 2, "group": "shapes",
            "title": f"A3-general: {n} random multi-layer shape jobs (set seed {ship.get('seed')}), "
                     f"bit/cycle-exact vs model + RTL sim, stratified order, cap {budget:g} s",
            "cmd": cmd, "out_flag": "--out-dir", "outputs": list(SHAPES_OUTPUTS), "est_s": est,
            "timeout_s": budget + SHAPES_TIMEOUT_MARGIN_S,
            "requires": ["s1.smoke"] + (["s1.fast"] if hp == "fast" else []),
            "required": True, "params": params, "kind": "fixed", "images": 0}


def extra_steps(opts) -> list[dict]:
    return [s for s in (spread_step(opts), shapes_step(opts), soak_step(opts)) if s is not None]
