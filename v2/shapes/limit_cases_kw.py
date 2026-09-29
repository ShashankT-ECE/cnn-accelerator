"""Boundary cases for the kernel width / height (DECISIONS OC-3), loaded by gen_limits.py.

OC-3: the conflict-free ACT read rd[b] = rowbase + ox0/8 + (b < kx), rotate = kx[2:0]
(gos_ctrl.sv issue register) is exact for kx <= 8 (KW <= 9) by analysis and wrong from
kx = 9 (KW >= 10). The config checker has no KW bound. These cases measure it in RTL sim:
KW = 8, 9, 10 (>= 2 cases each: different IC, IW, pool on/off, square and rectangular
kernels, OW spanning several 8-wide tiles with OW % 8 != 0), KW = 9 in a 2-layer chain and
as an out_raw head, KW = 11, 12, 16 for the shape of the failure, and KH = 9, 10, 16 with a
small KW (the ky path is rowbase += IN_WPR per ky: no analytical limit below the 8-bit
field; recorded, not assumed).

"expect" is a PREDICTION only (recorded next to the RTL result); the verdict is what the RTL
does. Case schema: gen_limits.py (CASE_SCHEMA).
"""
from __future__ import annotations

from gen_shapes import layer


def _kw_pred(kw: int) -> str:
    return "exact" if kw <= 9 else "mismatch"


CASES: list[dict] = []


def _add(cid, field, value, layers, note, expect=None):
    CASES.append({"id": cid, "field": field, "value": value, "layers": layers,
                  "expect": expect or (_kw_pred(value) if field == "KW" else "exact"),
                  "note": note})


# ---- KW 8 / 9 / 10: two single-layer shapes each (a: no pool, rect KH=3; b: pool, KH=2),
# plus a square KHxKW case (c). OW = 30 (4 tiles, OW % 8 = 6) / 20 (3 tiles, % 8 = 4) / 17.
for kw in (8, 9, 10):
    _add(f"KW{kw:02d}_conv_a", "KW", kw, [layer(2, 6, kw + 29, 5, 3, kw, pool=0, relu=1)],
         f"1 layer 2x6x{kw + 29} k3x{kw} -> 5, OW=30 (4 tiles, OW%8=6), no pool")
    _add(f"KW{kw:02d}_pool_b", "KW", kw, [layer(3, 7, kw + 19, 9, 2, kw, pool=1, relu=1)],
         f"1 layer 3x7x{kw + 19} k2x{kw} -> 9, OH=6 OW=20 (3 tiles, OW%8=4), 2x2 pool")
    _add(f"KW{kw:02d}_sq_c", "KW", kw, [layer(1, kw + 2, kw + 16, 12, kw, kw, pool=0, relu=0)],
         f"1 layer 1x{kw + 2}x{kw + 16} k{kw}x{kw} -> 12 (OC tail), OW=17 (3 tiles, OW%8=1), no ReLU")

# ---- KW = 9 in a 2-layer chain (both layers KW = 9; layer 1 reads layer 0's pooled output)
_add("KW09_chain2", "KW", 9,
     [layer(2, 8, 30, 8, 3, 9, pool=1, relu=1),          # -> 8x6x22 -> pool 8x3x11
      layer(8, 3, 11, 6, 2, 9, pool=0, relu=1)],         # -> 6x2x3
     "2 layers: 2x8x30 k3x9 -> 8 pool (8x3x11), then k2x9 -> 6 (OW=3)")

# ---- KW = 9 as the out_raw head (global 9x9 conv to 1x1 LOGIT; OW = 1)
_add("KW09_raw_head", "KW", 9, [layer(2, 9, 9, 10, 9, 9, raw=1)],
     "1 layer out_raw: 2x9x9 k9x9 -> LOGIT[10]")

# ---- KW 11 / 12 / 16: shape of the failure (OW = 30, 4 tiles)
for kw in (11, 12, 16):
    _add(f"KW{kw:02d}_conv_a", "KW", kw, [layer(2, 6, kw + 29, 5, 3, kw, pool=0, relu=1)],
         f"1 layer 2x6x{kw + 29} k3x{kw} -> 5, OW=30 (4 tiles, OW%8=6), no pool")

# ---- KH boundary with a small KW (ky path: rowbase += IN_WPR per ky)
_add("KH09_kw3", "KH", 9, [layer(2, 12, 21, 5, 9, 3, pool=0, relu=1)],
     "1 layer 2x12x21 k9x3 -> 5, OH=4, OW=19 (3 tiles)")
_add("KH09_kw1_pool", "KH", 9, [layer(3, 14, 18, 9, 9, 1, pool=1, relu=1)],
     "1 layer 3x14x18 k9x1 -> 9, OH=6 OW=18 (3 tiles), 2x2 pool")
_add("KH10_kw2", "KH", 10, [layer(2, 13, 20, 7, 10, 2, pool=0, relu=1)],
     "1 layer 2x13x20 k10x2 -> 7, OH=4, OW=19 (3 tiles)")
_add("KH16_kw1", "KH", 16, [layer(1, 20, 19, 6, 16, 1, pool=0, relu=0)],
     "1 layer 1x20x19 k16x1 -> 6 (K=16), OH=5, OW=19, no ReLU")
_add("KH16_kw3", "KH", 16, [layer(2, 18, 21, 9, 16, 3, pool=0, relu=1)],
     "1 layer 2x18x21 k16x3 -> 9, OH=3, OW=19 (3 tiles)")
