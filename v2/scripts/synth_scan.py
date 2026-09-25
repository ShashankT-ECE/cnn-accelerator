#!/usr/bin/env python3
"""Synthesis-log scan (V2 step 4.5, Part B static checks).

Usage: synth_scan.py <synth.log> [--out <report.txt>] [--waivers <file>]
Classifies Vivado synthesis messages into
  latch        Synth 8-327 (inferring latch), 8-4767, 8-5972
  multidriven  Synth 8-6859, 8-6858, 8-3352, 8-5559 (multi-driven net / pin)
  undriven     Synth 8-3848 (net without driver), 8-3295 (tying undriven pin), 8-6104
  removed      Synth 8-3332, 8-6014, 8-3936, 8-7129, 8-3917, 8-3886, 8-4471, 8-7067
               (unused / unconnected / trimmed / merged logic)
and groups each message by (id, module-or-file, signal name without bit/array indices).
The log must be written with no per-ID message limit (messaging.defaultLimit), otherwise the scan
refuses: a truncated log cannot prove the absence of a message.
Exit 0 only if latch, multidriven and undriven are all empty AND every removed group matches a
waiver line "<id> <file-or-module glob> <signal glob> D<n>" (reason recorded in docs/DECISIONS.md).
"""
import argparse
import fnmatch
import re
import sys
from collections import defaultdict
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
CATS = {
    "latch": {"8-327", "8-4767", "8-5972"},
    "multidriven": {"8-6859", "8-6858", "8-3352", "8-5559"},
    "undriven": {"8-3848", "8-3295", "8-6104"},
    "removed": {"8-3332", "8-6014", "8-3936", "8-7129", "8-3917", "8-3886", "8-4471", "8-7067"},
}
MSG = re.compile(r"^(?:CRITICAL WARNING|WARNING|INFO): \[Synth (8-\d+)\] (.*)$")
LIMIT = re.compile(r"\[Common 17-14\] Message '(Synth 8-\d+)' appears 100 times and further instances")


def norm(sig):
    sig = re.sub(r"\[[^\]]*\]", "", sig)          # bit / array indices
    sig = re.sub(r"_reg\b", "", sig)
    return sig


def key_of(mid, text):
    loc = re.search(r"\[([^\]]+\.s?v):\d+\]", text)
    where = Path(loc.group(1)).name if loc else ""
    m = (re.search(r"Port (\S+) in module (\S+)", text) or
         re.search(r"(?:Sequential element|sequential element|register|net|Net|pin) \(?'?([^\s')]+)'?\)?", text))
    if mid == "8-7129" and m:
        return where or m.group(2), norm(m.group(1))
    if mid == "8-3332":
        mm = re.search(r"Sequential element \(([^)]+)\) is unused and will be removed from module (\S+)", text)
        if mm:
            return mm.group(2).rstrip("."), norm(mm.group(1).split("/")[-1])
    sig = m.group(1) if m else text[:60]
    return where, norm(sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--waivers", type=Path, default=V2 / "vivado" / "synth_waivers.txt")
    a = ap.parse_args()
    text = a.log.read_text(errors="replace")
    truncated = sorted(set(LIMIT.findall(text)))
    groups = {c: defaultdict(int) for c in CATS}
    for line in text.splitlines():
        m = MSG.match(line)
        if not m:
            continue
        mid, body = m.groups()
        for c, ids in CATS.items():
            if mid in ids:
                groups[c][(mid,) + key_of(mid, body)] += 1
    waivers = []
    if a.waivers.exists():
        for l in a.waivers.read_text().splitlines():
            l = l.split("#", 1)[0].split()
            if l:
                assert len(l) == 4 and re.fullmatch(r"D\d+(-\d+)?", l[3]), f"bad waiver line {l}"
                waivers.append(l)
    unwaived = []
    lines = [f"synth_scan: {a.log}"]
    if truncated:
        lines.append(f"TRUNCATED LOG (per-ID message limit hit): {', '.join(truncated)}")
    for c in CATS:
        n = sum(groups[c].values())
        lines.append(f"{c}: {n} messages in {len(groups[c])} groups")
        for (mid, where, sig), cnt in sorted(groups[c].items()):
            w = next((w for w in waivers if w[0] == mid and fnmatch.fnmatch(where, w[1])
                      and fnmatch.fnmatch(sig, w[2])), None) if c == "removed" else None
            if c == "removed" and w is None:
                unwaived.append((mid, where, sig))
            lines.append(f"  {mid:7s} {where:24s} {sig:40s} x{cnt:<5d} {'waived ' + w[3] if w else 'NOT WAIVED'}")
    ok = not truncated and not unwaived and not any(groups[c] for c in ("latch", "multidriven", "undriven"))
    lines.append(f"SYNTH_SCAN {'PASS' if ok else 'FAIL'} latch={sum(groups['latch'].values())} "
                 f"multidriven={sum(groups['multidriven'].values())} undriven={sum(groups['undriven'].values())} "
                 f"removed={sum(groups['removed'].values())} unwaived_groups={len(unwaived)} truncated={len(truncated)}")
    rep = "\n".join(lines) + "\n"
    print(rep, end="")
    if a.out:
        a.out.write_text(rep)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
