#!/usr/bin/env bash
#==============================================================================
# run_verification.sh — single PASS/FAIL gate for the CNN accelerator
#
# Runs the whole board-independent verification ladder and prints one table:
#   1. RTL integration regression (tb_cnn_accelerator_v2) — bit-exact vs the
#      independent Python integer golden, OS/WS/sparse/reconfig/boundary.
#   2. AXI board-integration regression (tb_cnn_top_axi) — the full AXI4-Lite
#      path (image/weight load, start/done, cycle counter, result readback,
#      runtime OS<->WS switch) bit-exact vs the same golden.
#   3. Python golden-model self-checks (python/tests/test_pipeline.py).
#
# Exit code 0 only if every row passes. Requires Vivado ML 2023.1 on PATH
# (asserts it — no substitute simulator, per CLAUDE.md).
#
# Usage: scripts/run_verification.sh
#==============================================================================
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== CNN accelerator verification ==="
echo "repo: $REPO_ROOT"

# --- Vivado ML 2023.1 assertion -------------------------------------------
if ! command -v xsim >/dev/null 2>&1; then
    if [ -f "$HOME/Xilinx/Vivado/2023.1/settings64.sh" ]; then
        # shellcheck disable=SC1090
        source "$HOME/Xilinx/Vivado/2023.1/settings64.sh"
    else
        echo "ERROR: xsim not on PATH and Vivado 2023.1 not found." >&2
        exit 1
    fi
fi

VER="$(xsim --version 2>/dev/null | head -1 || true)"
case "$VER" in
    *2023.1*) ;;
    *) echo "ERROR: xsim must be Vivado ML 2023.1 (got: ${VER:-unknown})" >&2
       exit 1 ;;
esac
echo "simulator: $VER"

# --- Scratch build dir -----------------------------------------------------
SIM_DIR="$REPO_ROOT/build/verify"
mkdir -p "$SIM_DIR"

overall=0

run_xsim() {
    local name="$1"; shift
    local top="$1"; shift
    local files=("$@")
    echo
    echo "--- $name ---"
    if ! xvlog -sv "${files[@]}" > "$SIM_DIR/${name}_xvlog.log" 2>&1; then
        echo "  $name: xvlog FAILED (see build/verify/${name}_xvlog.log)"
        return 1
    fi
    if ! xelab -debug typical -s "${name}_snap" "$top" > "$SIM_DIR/${name}_xelab.log" 2>&1; then
        echo "  $name: xelab FAILED (see build/verify/${name}_xelab.log)"
        return 1
    fi
    if ! xsim "${name}_snap" -runall > "$SIM_DIR/${name}_xsim.log" 2>&1; then
        echo "  $name: xsim FAILED (see build/verify/${name}_xsim.log)"
        return 1
    fi
    if grep -q "RESULT: PASS" "$SIM_DIR/${name}_xsim.log"; then
        echo "  $name: PASS"
        return 0
    else
        echo "  $name: FAIL (result marker absent)"
        return 1
    fi
}

RTL_COMMON="$REPO_ROOT/rtl/common"
RTL_TOP="$REPO_ROOT/rtl/top"
SIM="$REPO_ROOT/sim"

# 1. Integration regression
run_xsim tb_cnn_acc \
    tb_cnn_accelerator_v2 \
    "$RTL_COMMON/pe_v2.sv" "$RTL_COMMON/systolic_array_v2.sv" \
    "$RTL_COMMON/cnn_accelerator_v2.sv" "$SIM/tb_cnn_accelerator_v2.sv"
[ $? -eq 0 ] || overall=1

# 2. AXI board-integration regression
run_xsim tb_cnn_top_axi \
    tb_cnn_top_axi \
    "$RTL_COMMON/pe_v2.sv" "$RTL_COMMON/systolic_array_v2.sv" \
    "$RTL_COMMON/cnn_accelerator_v2.sv" "$RTL_COMMON/cnn_axi_ctrl.sv" \
    "$RTL_TOP/cnn_top.sv" "$SIM/tb_cnn_top_axi.sv"
[ $? -eq 0 ] || overall=1

# 3. Python golden-model self-checks
echo
echo "--- python golden-model self-checks ---"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    if "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/python/tests/test_pipeline.py" > "$SIM_DIR/python_test.log" 2>&1; then
        echo "  python/test_pipeline.py: PASS"
    else
        echo "  python/test_pipeline.py: FAIL (see build/verify/python_test.log)"
        overall=1
    fi
else
    echo "  python/test_pipeline.py: SKIPPED (.venv not found)"
    overall=1
fi

echo
echo "=============================================="
if [ "$overall" -eq 0 ]; then
    echo "VERIFICATION: ALL PASS"
    exit 0
else
    echo "VERIFICATION: FAIL"
    exit 1
fi
