"""Step 3 phase 0: host-side proof that the requant multiply may use a 26-bit signed v."""
import numpy as np
import pytest

import gos_pack as gp
from common import V2_ROOT
from net_config import NET_CONFIGS, NETS


@pytest.mark.parametrize("net", NETS)
def test_v_bound_fits_v_mul_w(net):
    rows = gp.v_mul_margin(net)
    assert [r["layer"] for r in rows] == [L["name"] for L in NET_CONFIGS[net]["layers"] if not L["final"]]
    for r in rows:
        print(net, r)
        assert r["v_abs_bound"] < 2 ** (gp.V_MUL_W - 1)
        assert r["headroom_bits"] >= 0


def test_bound_formula_and_guard():
    L = {"K": 800, "final": False}
    assert gp.v_abs_bound(L, np.array([-5, 3], dtype=np.int32)) == 800 * 16384 + 5
    # a layer whose bound reaches 2^25 must be rejected by the width check
    K_bad = (2 ** 25) // 16384
    assert gp.v_abs_bound({"K": K_bad}, np.array([0], dtype=np.int32)) >= 2 ** (gp.V_MUL_W - 1)


def test_rtl_pkg_matches():
    text = (V2_ROOT / "rtl" / "gos_pkg.sv").read_text()
    assert f"V_MUL_W   = {gp.V_MUL_W};" in text
