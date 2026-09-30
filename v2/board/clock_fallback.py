#!/usr/bin/env python3
"""Performance-clock bitstream choice with automatic fallback (300 -> 250 MHz; 200 only on request).

    choose(ctx, bits, run_smoke, *, allow_lower=False, min_clock_mhz=FALLBACK_FLOOR_MHZ) -> dict

bits       [{"bit": path, "clock_mhz": closed_clock}, ...] (any order; sorted highest first here)
run_smoke  run_smoke(bit_entry) -> bool: load that bitstream and run the core smoke test
           (e.g. subprocess_smoke(...) below = test_core_smoke.py --bit <bit> with a timeout).
           False, an exception or a timeout (TimeoutError / subprocess.TimeoutExpired) = FAIL.

Policy (agreed with the session orchestrator):
  * try the highest closed clock first (300 MHz build), then the next one (250 MHz build);
  * candidates below min_clock_mhz (default 250 MHz - tolerance, i.e. the 200 MHz build) are tried
    ONLY with allow_lower=True (explicit option) -- never silently;
  * if every allowed candidate fails, the result is a FAILURE (ok=False, bit=None): the caller
    must stop, not run the performance experiments at an unverified clock;
  * every attempt is recorded (bit, closed clock, sha256 if readable, result, reason, duration).

Returns {"ok", "bit", "clock_mhz", "fell_back", "reason", "attempts": [...], "skipped": [...],
"policy", "decided_utc"}. fell_back = the chosen bitstream is not the highest candidate.

CLI (board, after deploy; writes the decision JSON, exit 0 = a bitstream was chosen):
    sudo -E python3 clock_fallback.py --bits bit/gos_300.bit:299.997009 bit/gos_250.bit:249.997498 \
        [--allow-lower] [--smoke-timeout-s 300] [--out results/hw_clock_choice.json]
    python3 clock_fallback.py --backend model --bits ... --out ../results/dryrun/hw_clock_choice.json
The closed clock of each entry may be omitted (bit/gos_300.bit): it is then read from the
summary.json next to the bitstream (pl_clk0_mhz_actual).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

BOARD_DIR = Path(__file__).resolve().parent
CLOCK_TOL_MHZ = 0.5                     # = board_common.CLOCK_TOL_MHZ
FALLBACK_FLOOR_MHZ = 250.0              # lowest clock tried without allow_lower
CHOICE_FILE = "hw_clock_choice.json"    # default decision file name (results dir)
SMOKE_SCRIPT = "test_core_smoke.py"


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256(path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _say(ctx):
    if ctx is None:
        return print
    if isinstance(ctx, dict):
        return ctx.get("say", print)
    return getattr(ctx, "say", print)


def choose(ctx, bits: list[dict], run_smoke, *, allow_lower: bool = False,
           min_clock_mhz: float = FALLBACK_FLOOR_MHZ, tol: float = CLOCK_TOL_MHZ) -> dict:
    """Try the candidates highest clock first; return the first whose smoke passes (see module doc)."""
    say = _say(ctx)
    cands = sorted(bits, key=lambda b: float(b["clock_mhz"]), reverse=True)
    floor = float(min_clock_mhz) - tol
    allowed = [b for b in cands if allow_lower or float(b["clock_mhz"]) >= floor]
    skipped = [{"bit": str(b["bit"]), "clock_mhz": float(b["clock_mhz"]),
                "reason": f"below {min_clock_mhz:g} MHz; needs the explicit allow_lower option"}
               for b in cands if b not in allowed]
    policy = (f"highest closed clock first; fall back down to {min_clock_mhz:g} MHz"
              + ("; lower clocks allowed (explicit option)" if allow_lower else
                 "; lower clocks never tried without the explicit option"))
    res = {"ok": False, "bit": None, "clock_mhz": None, "fell_back": False, "reason": "",
           "attempts": [], "skipped": skipped, "policy": policy, "decided_utc": ""}
    if not allowed:
        res["reason"] = "no candidate bitstream at or above the fallback floor"
        res["decided_utc"] = _utc()
        say(f"[clock_fallback] FAIL: {res['reason']} (candidates {[b['clock_mhz'] for b in cands]})")
        return res
    top = float(allowed[0]["clock_mhz"])
    for b in allowed:
        att = {"bit": str(b["bit"]), "clock_mhz": float(b["clock_mhz"]),
               "bit_sha256": _sha256(b["bit"]), "start_utc": _utc(), "ok": False, "reason": ""}
        t0 = time.monotonic()
        say(f"[clock_fallback] smoke test {att['bit']} (closed {att['clock_mhz']:g} MHz)")
        try:
            ok = run_smoke(b)
            att["ok"] = ok is True
            att["reason"] = "smoke PASS" if att["ok"] else f"smoke FAIL (returned {ok!r})"
            # optional facts the runner leaves on the entry: the pl_clk0 actually read back during
            # the smoke (must equal the closed clock) and why a pre-flight failed
            for k in ("clock_readback_mhz", "detail"):
                if isinstance(b, dict) and b.get(k) is not None:
                    att[k] = b[k]
            if att.get("detail") and not att["ok"]:
                att["reason"] += f": {att['detail']}"
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            att["reason"] = f"smoke TIMEOUT: {e}"
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # noqa: BLE001 - a crashing smoke is a failed smoke
            att["reason"] = f"smoke EXCEPTION {type(e).__name__}: {e}"
        att["duration_s"] = round(time.monotonic() - t0, 3)
        res["attempts"].append(att)
        say(f"[clock_fallback]   -> {att['reason']} ({att['duration_s']} s)")
        if att["ok"]:
            fb = float(b["clock_mhz"]) < top
            res.update(ok=True, bit=str(b["bit"]), clock_mhz=float(b["clock_mhz"]), fell_back=fb,
                       reason=(f"fell back to {b['clock_mhz']:g} MHz: " + "; ".join(
                           f"{a['clock_mhz']:g} MHz {a['reason']}" for a in res["attempts"][:-1]))
                       if fb else f"highest candidate {b['clock_mhz']:g} MHz passed the smoke test")
            break
    else:
        res["reason"] = ("every allowed candidate failed: " + "; ".join(
            f"{a['clock_mhz']:g} MHz {a['reason']}" for a in res["attempts"])
            + ("" if allow_lower else f" (lower clocks not tried: pass the explicit option)"))
    res["decided_utc"] = _utc()
    say(f"[clock_fallback] {'CHOSEN ' + str(res['bit']) if res['ok'] else 'FAILURE'}: {res['reason']}")
    return res


def write_choice(result: dict, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=1) + "\n")
    return p


def read_choice(path) -> dict | None:
    """The decision JSON, or None if absent / unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def closed_clock_of(bit) -> float | None:
    """pl_clk0_mhz_actual from the summary.json next to the bitstream (None if unavailable)."""
    p = Path(bit).parent / "summary.json"
    try:
        return float(json.loads(p.read_text())["pl_clk0_mhz_actual"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def parse_bits(specs: list[str]) -> list[dict]:
    out = []
    for s in specs:
        path, _, clk = s.partition(":")
        c = float(clk) if clk else closed_clock_of(path)
        if c is None:
            raise SystemExit(f"closed clock of {path} unknown: pass {path}:<MHz>")
        out.append({"bit": path, "clock_mhz": c})
    return out


def subprocess_smoke(extra_args: list[str], timeout_s: float = 300.0, python: str = sys.executable,
                     cwd=BOARD_DIR, tol: float = CLOCK_TOL_MHZ):
    """run_smoke for choose(): test_core_smoke.py --bit <bit> --max-fclk0 <closed + tol> <extra_args>
    as a subprocess (own process group) with a hard timeout (raises subprocess.TimeoutExpired)."""
    def run(entry: dict) -> bool:
        cmd = [python, str(Path(cwd) / SMOKE_SCRIPT), "--bit", str(entry["bit"]),
               "--max-fclk0", f"{float(entry['clock_mhz']) + tol:.6f}", *extra_args]
        p = subprocess.run(cmd, cwd=str(cwd), timeout=timeout_s)
        return p.returncode == 0
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bits", nargs="+", required=True, help="<bit>[:<closed MHz>] ... (any order)")
    ap.add_argument("--allow-lower", action="store_true",
                    help=f"also try bitstreams below {FALLBACK_FLOOR_MHZ:g} MHz (e.g. the 200 MHz build)")
    ap.add_argument("--min-clock-mhz", type=float, default=FALLBACK_FLOOR_MHZ)
    ap.add_argument("--smoke-timeout-s", type=float, default=300.0)
    ap.add_argument("--backend", choices=("pynq", "model"), default="pynq")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None, help=f"decision JSON (default: print only); e.g. "
                    f"results/{CHOICE_FILE}")
    a = ap.parse_args(argv)
    bits = parse_bits(a.bits)
    extra = ["--backend", a.backend] + (["--data-dir", a.data_dir] if a.data_dir else [])
    if a.backend == "model":
        extra_fn = lambda e: [*extra, "--clock-mhz", f"{e['clock_mhz']:.6f}"]  # noqa: E731
    else:
        extra_fn = lambda e: extra  # noqa: E731

    def smoke(e):
        return subprocess_smoke(extra_fn(e), a.smoke_timeout_s)(e)

    res = choose(None, bits, smoke, allow_lower=a.allow_lower, min_clock_mhz=a.min_clock_mhz)
    res["backend"] = a.backend
    if a.backend == "model":
        res["note"] = "DRY RUN: ModelBackend smoke, not a hardware bring-up"
    if a.out:
        out = Path(a.out)
        in_dry = "dryrun" in Path(out.resolve()).parent.parts
        if (a.backend == "model") != in_dry:
            raise SystemExit(f"REFUSED: {'model' if a.backend == 'model' else 'hardware'} decision "
                             f"file {'must' if a.backend == 'model' else 'must not'} go under a "
                             f"'dryrun' directory: {out}")
        print(f"wrote {write_choice(res, out)}")
    print(json.dumps({k: res[k] for k in ("ok", "bit", "clock_mhz", "fell_back", "reason")}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
