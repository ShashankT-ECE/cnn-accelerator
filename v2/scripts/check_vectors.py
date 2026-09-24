#!/usr/bin/env python3
"""Verify v2/vectors/generated against v2/vectors/MANIFEST.json (SHA256 + size).

Exit 0 if every manifest file exists with the recorded hash; 1 otherwise
(then run: .venv/bin/python v2/scripts/gen_vectors.py --require-clean).
"""
import hashlib
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
m = json.loads((V2 / "vectors" / "MANIFEST.json").read_text())
gen = V2 / "vectors" / "generated"
bad = []
for f in m["files"]:
    p = gen / f["path"]
    if not p.exists():
        bad.append(f"missing {f['path']}")
    elif hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
        bad.append(f"hash mismatch {f['path']}")
if bad:
    print("check_vectors: FAIL\n  " + "\n  ".join(bad[:20]), file=sys.stderr)
    print("  run: .venv/bin/python v2/scripts/gen_vectors.py --require-clean", file=sys.stderr)
    sys.exit(1)
print(f"check_vectors: {len(m['files'])} files match MANIFEST (generator commit {m['git_commit'][:7]})")
