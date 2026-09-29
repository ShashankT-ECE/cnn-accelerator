"""Extra session steps for run_sessions.py (publication experiments; agreed interface (a)).

    extra_steps(opts) -> list[dict]

Each dict: id, session (2 or 3), group ("A3","A1","latency","baselines","energy","sweep","soak",
"A4"), title, cmd (argv relative to the board dir; the orchestrator appends `out_flag <staging>`),
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
The B2 sweep stays the orchestrator's s3.B2 (exp_b2_clock.py); it now also writes
hw_b2_cycles.csv and hw_b2_fit.csv (B2_EXTRA_OUTPUTS, to add to s3.B2's expected outputs).

opts: a dict, an argparse Namespace or run_sessions.Config-like object; attributes/keys read
(missing -> default): backend ('pynq'|'model'), data_dir, bit, closed_mhz (or deploy
{'bit_clock_mhz'}), results_dir, allow_dirty, quick, write_mode ('elem'), host_path ('safe'),
fast_store ('block'), board_id, dev_args (optional: prebuilt device argv, replaces the one built
here), soak_s (default 1800 s; --quick 300 s; dry run 60 s), soak_nets (both), soak_block_s (60),
spread_limit (None = all images; --quick 200), clock_choice (path of the clock_fallback decision
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


def extra_steps(opts) -> list[dict]:
    return [spread_step(opts), soak_step(opts)]
