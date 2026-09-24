"""Step 4: the cycle-model latency constants equal the RTL localparams.

Each L_* used in gos_cycle_model's C_PIPE/C_START/C_DONE derivation is read from
the RTL source (`localparam int NAME = expr;`, evaluated over the previously
parsed names), so the model cannot silently drift from the RTL.
"""
import re

import pytest

import gos_cycle_model as cm
from common import V2_ROOT

RTL = V2_ROOT / "rtl"
_LP = re.compile(r"localparam\s+int\s+(\w+)\s*=\s*([^;]+);")


def rtl_params(*files) -> dict:
    env = {}
    for f in ("gos_pkg.sv",) + files:
        for name, expr in _LP.findall((RTL / f).read_text()):
            expr = re.sub(r"//.*", "", expr).strip()
            try:
                env[name] = int(eval(expr, {"__builtins__": {}}, dict(env)))
            except Exception:
                pass            # $bits / $clog2 etc. (not needed here)
    return env


@pytest.mark.parametrize("model_name,rtl_file,rtl_name", [
    ("L_ISSUE", "gos_ctrl.sv", "L_ISSUE"),
    ("L_MEM_ACC", "gos_act_buf.sv", "L_ACT_BUF"),
    ("L_MEM_ACC", "gos_wgt_mem.sv", "L_WGT_MEM"),
    ("L_ROT", "gos_rotator.sv", "L_ROT"),
    ("L_ARRAY", "gos_array.sv", "L_ARRAY"),
    ("L_DRAIN", "gos_array.sv", "L_DRAIN"),
    ("L_QPARAM", "gos_qparam_mem.sv", "L_QPARAM_MEM"),
    ("L_RQ", "gos_requant.sv", "L_RQ"),
    ("L_POOL", "gos_pool.sv", "L_POOL"),
    ("L_LOAD", "gos_core.sv", "L_LOAD"),
    ("L_RETIRE", "gos_core.sv", "L_RETIRE"),
])
def test_latency_matches_rtl(model_name, rtl_file, rtl_name):
    env = rtl_params(rtl_file)
    assert rtl_name in env, f"{rtl_name} not found in {rtl_file}"
    assert getattr(cm, model_name) == env[rtl_name], (model_name, rtl_file, env[rtl_name])


def test_constants_are_the_derivation():
    assert cm.C_PIPE == (cm.L_LOAD + cm.L_ISSUE + cm.L_MEM_ACC + cm.L_ROT + cm.L_ARRAY
                         + (cm.L_DRAIN - 1) + cm.L_QPARAM + cm.L_RQ + cm.L_POOL + cm.L_RETIRE)
    env = rtl_params("gos_core.sv")
    assert cm.C_START == env["N_CHK"] and cm.C_DONE == env["N_DONE"]


def test_c_start_is_the_check_state_count():
    # C_START must equal the number of S_CHK<n> states in the gos_core state enum
    import re
    src = (RTL / "gos_core.sv").read_text()
    enum = re.search(r"typedef enum[^{]*\{([^}]*)\}\s*state_t", src).group(1)
    n_chk = len(re.findall(r"\bS_CHK\d\b", enum))
    assert n_chk == cm.C_START == rtl_params("gos_core.sv")["N_CHK"]
