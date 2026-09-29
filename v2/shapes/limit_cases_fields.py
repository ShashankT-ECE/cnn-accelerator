"""Boundary cases for every descriptor field the RTL config checker does not fully enforce
(v2/docs/LIMITS.md), loaded by gen_limits.py. KW is excluded (limit_cases_kw.py, OC-3).

Groups:
  * at-limit shapes the checker accepts (expected exact) + the just-over shape (expected refuse,
    showing the limit IS enforced, indirectly, through an END bound given consistent fields);
  * descriptors that bypass the host (desc_patch / raw_words after encode): inconsistent OH/OW,
    K field, understated END fields, ignored fields (IH, OUT_W), reserved bits, in_sel, and
    out_raw combinations the checker accepts (OH/OW > 1, relu_en, pool_en).
The golden is always that of the INTENDED (unpatched) layers (gen_limits.structure), so a
patch that changes what the RTL computes shows up as "mismatch".

"expect" is a PREDICTION only; the verdict is what the RTL does. Not covered here (cannot be
built by gen_limits / tb_shapes, analysed in LIMITS.md instead): the V_MUL_W bound with
adversarial data (K >= 2048 on a requant layer: gen_shapes.make_layer_params asserts
K <= 2047), out_raw on a non-final layer (gen_limits refuses to chain it), N_LAYERS >= 16
(the CSR keeps bits 3:0; tb_shapes drives the core port directly), WGT/QPARAM overlap.
"""
from __future__ import annotations

from gen_shapes import layer

CASES: list[dict] = []


def _add(cid, field, value, layers, expect, note, overrides=None, err_code=None):
    c = {"id": f"F_{cid}", "field": field, "value": int(value), "layers": layers,
         "expect": expect, "note": note}
    if overrides:
        c["overrides"] = overrides
    if err_code is not None:
        c["err_code"] = int(err_code)
    CASES.append(c)


def _err(rule: int, layer_idx: int = 0) -> int:
    return (rule << 8) | layer_idx


def _w2(KH: int, KW: int) -> int:
    return KH | (KW << 8)


def _w6(relu: int, pool: int, raw: int, in_sel: int) -> int:
    return relu | (pool << 1) | (raw << 2) | (in_sel << 3)


# ---- IC / K: 16-bit ic and k counters, K field used as f_k[15:0] --------------------------------
_add("IC_2047_req", "IC", 2047, [layer(2047, 1, 1, 8, 1, 1)], "exact",
     "requant FC K=2047 (largest K with K*16384 < 2^25, V_MUL_W); random data, |v| not stressed")
_add("IC_4096_raw", "IC", 4096, [layer(4096, 1, 1, 16, 1, 1, raw=1)], "exact",
     "out_raw FC IC=4096: IN_END=4095, ic counter to 4095, 2 OC tiles, WGT 8192 words")
_add("IC_4097_raw", "IC", 4097, [layer(4097, 1, 1, 16, 1, 1, raw=1)], "refuse",
     "IC=4097 on a 1x1 map -> IN_END=4096 -> rule 5 (IC bounded only through IN_END)",
     err_code=_err(5))
_add("K_16384_raw", "K", 16384, [layer(1024, 4, 4, 8, 4, 4, raw=1)], "exact",
     "out_raw 1024x4x4 k4x4 -> 8: K=16384 (f_k[15:0], k counter to 16383), IN_END=4095, WGT_END=16383")
_add("K_16384_oc9", "K", 16384, [layer(1024, 4, 4, 9, 4, 4, raw=1)], "refuse",
     "same with OC=9: 2 OC tiles x 16384 -> WGT_END=32767 -> rule 4", err_code=_err(4))

# ---- OC: 8-bit QPARAM channel, oc_tile counter, ch_rem --------------------------------------
_add("OC_256", "OC", 256, [layer(8, 1, 1, 256, 1, 1)], "exact",
     "FC 8 -> 256: 32 OC tiles, QP_END=255, OUT_END=255 (OC bounded by QP_END)")
_add("OC_257", "OC", 257, [layer(8, 1, 1, 257, 1, 1)], "refuse",
     "FC 8 -> 257 -> QP_END=256 -> rule 7", err_code=_err(7))

# ---- IH / IW: only through IN_PLANE/IN_WPR (IN_END) and OH/OW (OUT_END) -----------------------
_add("IH_4096", "IH", 4096, [layer(1, 4096, 8, 1, 1, 8)], "exact",
     "1x4096x8 k1x8 -> 1: 4096 output rows (oy/out_row 16-bit), IN_END=OUT_END=4095, T*K=32768")
_add("IH_4097", "IH", 4097, [layer(1, 4097, 8, 1, 1, 8)], "refuse",
     "IH=4097 -> IN_END=4096 -> rule 5", err_code=_err(5))
_add("IW_32768", "IW", 32768, [layer(1, 1, 32768, 1, 1, 8)], "exact",
     "1x1x32768 k1x8 -> 1: OW=32761, 4096 x tiles, IN_END=OUT_END=4095; rotator over-read "
     "reaches word 4096 and wraps to 0 (masked rows only)")
_add("IW_32776", "IW", 32776, [layer(1, 1, 32776, 1, 1, 8)], "refuse",
     "IW=32776 -> IN_WPR=4097 -> IN_END=4096 -> rule 5", err_code=_err(5))

# ---- KH: 8-bit ky counter, kh_m1 (other agent: KH 9/10/16 in limit_cases_kw.py) ---------------
_add("KH_255", "KH", 255, [layer(1, 255, 8, 8, 255, 1)], "exact",
     "1x255x8 k255x1 -> 8: KH at the 8-bit field maximum, OH=1, K=255")

# ---- WGT_BASE / QP_BASE at the top of their memories -----------------------------------------
_add("bases_top", "WGT_BASE", 16312, [layer(4, 6, 10, 12, 3, 3)], "exact",
     "k3x3 4->12 (2 OC tiles, K=36): WGT_BASE=16312 (WGT_END=16383), QP_BASE=244 (QP_END=255); "
     "masked OC-tail QPARAM reads wrap past 255",
     overrides={"wgt_base": 16312, "qp_base": 244})

# ---- OH / OW = IH-KH+1 / IW-KW+1: host-only (checker has OH <= IH, OW <= IW) -------------------
_add("OH_over", "OH", 7, [layer(2, 8, 12, 8, 3, 3)], "mismatch",
     "intended OH=6; OH patched to 7 (<= IH): an extra output row reads past the channel plane "
     "and is written over row 0 of the next output channel",
     overrides={"desc_patch": {0: {"OH": 7}}})
_add("OW_over", "OW", 17, [layer(1, 4, 18, 8, 3, 3)], "mismatch",
     "intended OW=16 (2 tiles); OW=17, OW_TILES=3 patched (<= IW): x=16 is written to word 0 of "
     "the next output row",
     overrides={"desc_patch": {0: {"OW": 17, "OW_TILES": 3}}})

# ---- K field vs IC*KH*KW: host-only consistency ---------------------------------------------
_add("K_field_oc8", "K", 100, [layer(3, 6, 10, 8, 3, 3)], "exact",
     "K=27 layer, K field patched to 100 with 1 OC tile: the K field is only the WGT step "
     "between OC tiles (and rule 3), so nothing changes",
     overrides={"desc_patch": {0: {"K": 100}}})
_add("K_field_oc16", "K", 100, [layer(3, 6, 10, 16, 3, 3)], "mismatch",
     "same with 2 OC tiles: OC tile 1 reads weights at WGT_BASE+100 instead of +27",
     overrides={"desc_patch": {0: {"K": 100}}})
_add("K_real_4", "K", 8, [layer(1, 6, 12, 8, 2, 2)], "mismatch",
     "real IC*KH*KW=4 (< 8, rule 3 would refuse), K field patched to 8: the checker accepts, "
     "tiles are 4 cycles apart < the 8-cycle drain -> array overrun (err_flags[4])",
     overrides={"desc_patch": {0: {"K": 8}}})

# ---- fields read only by the checker: understated END / ignored raw fields --------------------
_add("IN_END_0", "IN_END", 0, [layer(2, 8, 12, 8, 3, 3)], "exact",
     "IN_END patched to 0: END fields feed only the checker bounds (rules 4-7), so the bound "
     "holds only if the host keeps END consistent",
     overrides={"desc_patch": {0: {"IN_END": 0}}})
_add("WGT_END_0", "WGT_END", 0, [layer(2, 8, 12, 8, 3, 3)], "exact",
     "WGT_END patched to 0 (< WGT_BASE): not used by the datapath",
     overrides={"desc_patch": {0: {"WGT_END": 0}}})
_add("IH_field_ignored", "IH", 11, [layer(2, 8, 12, 8, 3, 3)], "exact",
     "IH patched 8 -> 11: IH/IW are used only by rules 10/11/25/26 (addresses come from IN_WPR/IN_PLANE)",
     overrides={"desc_patch": {0: {"IH": 11}}})
_add("OUT_W_field_ignored", "OUT_W", 3, [layer(2, 8, 12, 8, 3, 3)], "exact",
     "OUT_W patched 10 -> 3: OUT_W is used only by rule 21",
     overrides={"desc_patch": {0: {"OUT_W": 3}}})

# ---- reserved bits (ignored by checker and datapath) -----------------------------------------
_add("w2_reserved", "reserved_w2", 0xFFFF, [layer(2, 8, 12, 8, 3, 3)], "exact",
     "w2[31:16] = 0xFFFF",
     overrides={"raw_words": {0: {2: 0xFFFF0000 | _w2(3, 3)}}})
_add("w6_reserved", "reserved_w6", 0xFFFFFFF, [layer(2, 8, 12, 8, 3, 3)], "exact",
     "w6[31:4] all ones (flags relu_en=1, in_sel=0 kept)",
     overrides={"raw_words": {0: {6: 0xFFFFFFF0 | _w6(1, 0, 0, 0)}}})

# ---- in_sel: host chaining contract ----------------------------------------------------------
_add("in_sel_same", "in_sel", 0, [layer(2, 8, 12, 8, 3, 3), layer(8, 6, 10, 5, 3, 3)], "mismatch",
     "2-layer chain with in_sel = [0, 0]: layer 1 reads ACT0 (the input image) instead of "
     "layer 0's output in ACT1",
     overrides={"in_sel": [0, 0]})

# ---- out_raw combinations the checker accepts ------------------------------------------------
_add("raw_OH2", "out_raw", 1, [layer(8, 2, 1, 4, 1, 1, raw=1)], "mismatch",
     "out_raw with OH=2: LOGIT[oc] is written by every tile (row 0); the oy=1 tile overwrites oy=0",
     )
_add("raw_OW2", "out_raw", 1, [layer(8, 1, 2, 4, 1, 1, raw=1)], "exact",
     "out_raw with OW=2 (one x tile): LOGIT = row 0 = ox 0, which is what the harness compares; "
     "ox=1 is silently dropped")
_add("raw_OW9", "out_raw", 1, [layer(8, 1, 9, 4, 1, 1, raw=1)], "mismatch",
     "out_raw with OW=9 (2 x tiles): LOGIT = row 0 of the last tile = ox 8")
_add("raw_relu", "relu_en", 1, [layer(8, 1, 1, 12, 1, 1, raw=1)], "exact",
     "out_raw with relu_en=1: LOGIT takes v_raw, independent of relu_en (gos_requant.sv)",
     overrides={"desc_patch": {0: {"relu_en": 1}}})
_add("raw_pool", "pool_en", 1, [layer(8, 2, 2, 4, 1, 1, raw=1)], "mismatch",
     "out_raw with pool_en=1 (OUT 1x1 patched): both dy tiles write LOGIT; the dy=1 one "
     "(oy=1, ox=0) wins",
     overrides={"desc_patch": {0: {"pool_en": 1, "OUT_H": 1, "OUT_W": 1, "OUT_PLANE": 1,
                                   "OUT_END": 3}}})
