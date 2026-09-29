"""clock_fallback.choose: 300 -> 250 fallback policy, failures, exceptions, timeouts, records."""
import json
import subprocess

import pytest

import clock_fallback as cf

B300 = {"bit": "bit/gos_300.bit", "clock_mhz": 299.997009}
B250 = {"bit": "bit/gos_250.bit", "clock_mhz": 249.997498}
B200 = {"bit": "bit/gos_200.bit", "clock_mhz": 199.998001}
BITS = [B300, B250, B200]


def smoke(results: dict):
    calls = []

    def run(e):
        calls.append(e["clock_mhz"])
        r = results[e["clock_mhz"]]
        if isinstance(r, BaseException):
            raise r
        return r
    run.calls = calls
    return run


def quiet():
    return {"say": lambda *_: None}


def test_300_passes_chooses_300():
    s = smoke({B300["clock_mhz"]: True})
    r = cf.choose(quiet(), BITS, s)
    assert r["ok"] and r["bit"] == B300["bit"] and r["clock_mhz"] == B300["clock_mhz"]
    assert r["fell_back"] is False
    assert s.calls == [B300["clock_mhz"]]
    assert len(r["attempts"]) == 1 and r["attempts"][0]["ok"]


def test_300_fails_falls_back_to_250():
    s = smoke({B300["clock_mhz"]: False, B250["clock_mhz"]: True})
    r = cf.choose(quiet(), BITS, s)
    assert r["ok"] and r["bit"] == B250["bit"] and r["fell_back"] is True
    assert [a["clock_mhz"] for a in r["attempts"]] == [B300["clock_mhz"], B250["clock_mhz"]]
    assert [a["ok"] for a in r["attempts"]] == [False, True]
    assert "fell back to 249.997" in r["reason"] and "299.997 MHz smoke FAIL" in r["reason"]


def test_both_fail_is_failure_and_200_not_tried():
    s = smoke({B300["clock_mhz"]: False, B250["clock_mhz"]: False, B200["clock_mhz"]: True})
    r = cf.choose(quiet(), BITS, s)
    assert not r["ok"] and r["bit"] is None and r["clock_mhz"] is None
    assert B200["clock_mhz"] not in s.calls          # never silently lower
    assert len(r["attempts"]) == 2
    assert r["skipped"] and r["skipped"][0]["clock_mhz"] == B200["clock_mhz"]
    assert "every allowed candidate failed" in r["reason"]


def test_200_only_with_explicit_option():
    s = smoke({B300["clock_mhz"]: False, B250["clock_mhz"]: False, B200["clock_mhz"]: True})
    r = cf.choose(quiet(), BITS, s, allow_lower=True)
    assert r["ok"] and r["bit"] == B200["bit"] and r["fell_back"]
    assert len(r["attempts"]) == 3 and not r["skipped"]


def test_exception_and_timeout_are_failures():
    s = smoke({B300["clock_mhz"]: RuntimeError("overlay load failed"),
               B250["clock_mhz"]: subprocess.TimeoutExpired(["smoke"], 300)})
    r = cf.choose(quiet(), BITS, s)
    assert not r["ok"]
    assert "EXCEPTION RuntimeError" in r["attempts"][0]["reason"]
    assert "TIMEOUT" in r["attempts"][1]["reason"]
    s2 = smoke({B300["clock_mhz"]: TimeoutError("hung"), B250["clock_mhz"]: True})
    r2 = cf.choose(quiet(), BITS, s2)
    assert r2["ok"] and r2["clock_mhz"] == B250["clock_mhz"] and "TIMEOUT" in r2["attempts"][0]["reason"]


def test_non_true_return_is_failure():
    s = smoke({B300["clock_mhz"]: 1, B250["clock_mhz"]: None})       # only `True` passes
    r = cf.choose(quiet(), BITS, s)
    assert not r["ok"] and all(not a["ok"] for a in r["attempts"])


def test_keyboard_interrupt_propagates():
    s = smoke({B300["clock_mhz"]: KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        cf.choose(quiet(), BITS, s)


def test_unsorted_input_highest_first():
    s = smoke({B300["clock_mhz"]: False, B250["clock_mhz"]: True})
    r = cf.choose(quiet(), [B250, B200, B300], s)
    assert s.calls == [B300["clock_mhz"], B250["clock_mhz"]] and r["clock_mhz"] == B250["clock_mhz"]


def test_only_250_available_is_not_a_fallback():
    s = smoke({B250["clock_mhz"]: True})
    r = cf.choose(quiet(), [B250], s)
    assert r["ok"] and not r["fell_back"]


def test_no_candidate_at_floor():
    r = cf.choose(quiet(), [B200], smoke({}))
    assert not r["ok"] and not r["attempts"] and r["skipped"]


def test_attempt_records_sha_and_duration(tmp_path):
    bit = tmp_path / "gos_300.bit"
    bit.write_bytes(b"x")
    r = cf.choose(quiet(), [{"bit": str(bit), "clock_mhz": 300.0}], lambda e: True)
    a = r["attempts"][0]
    assert len(a["bit_sha256"]) == 64 and a["duration_s"] >= 0 and a["start_utc"]
    assert r["decided_utc"] and r["policy"]


def test_write_read_choice_and_parse_bits(tmp_path):
    d = tmp_path / "b300"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"pl_clk0_mhz_actual": 299.997009}))
    bits = cf.parse_bits([str(d / "gos_300.bit"), "x/gos_250.bit:249.997498"])
    assert bits == [{"bit": str(d / "gos_300.bit"), "clock_mhz": 299.997009},
                    {"bit": "x/gos_250.bit", "clock_mhz": 249.997498}]
    with pytest.raises(SystemExit):
        cf.parse_bits([str(tmp_path / "nosummary" / "a.bit")])
    r = cf.choose(quiet(), bits, lambda e: True)
    p = cf.write_choice(r, tmp_path / "dryrun" / cf.CHOICE_FILE)
    assert cf.read_choice(p)["bit"] == r["bit"]
    assert cf.read_choice(tmp_path / "missing.json") is None


def test_subprocess_smoke_builds_command(monkeypatch):
    seen = {}

    def fake_run(cmd, cwd=None, timeout=None):
        seen.update(cmd=cmd, timeout=timeout)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    run = cf.subprocess_smoke(["--backend", "model"], timeout_s=12.0)
    assert run(B300) is True
    assert seen["timeout"] == 12.0 and "--bit" in seen["cmd"] and B300["bit"] in seen["cmd"]
    assert f"{B300['clock_mhz'] + cf.CLOCK_TOL_MHZ:.6f}" in seen["cmd"]


def test_cli_output_dir_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "subprocess_smoke", lambda *a, **k: (lambda e: True))
    with pytest.raises(SystemExit):     # model decision outside a dryrun dir is refused
        cf.main(["--backend", "model", "--bits", "a.bit:300", "--out", str(tmp_path / "c.json")])
    rc = cf.main(["--backend", "model", "--bits", "a.bit:300", "b.bit:250",
                  "--out", str(tmp_path / "dryrun" / "c.json")])
    d = json.loads((tmp_path / "dryrun" / "c.json").read_text())
    assert rc == 0 and d["clock_mhz"] == 300.0 and "DRY RUN" in d["note"]
