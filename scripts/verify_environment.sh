#!/usr/bin/env bash
# scripts/verify_environment.sh
# Verify that the development environment matches the CNN Accelerator
# project baseline documented in docs/DEVELOPMENT_SETUP.md.
#
# Usage: ./scripts/verify_environment.sh   (run from the repository root)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

VIVADO_SETTINGS_HINT='source ~/Xilinx/Vivado/2023.1/settings64.sh'

FAILED=0

# report <name> <status> [detail]  (status: PASS | FAIL | SKIP)
report() {
    local name="$1" status="$2" detail="${3:-}"
    if [ "$status" = "PASS" ]; then
        printf '  %-24s PASS' "$name"
    elif [ "$status" = "SKIP" ]; then
        printf '  %-24s SKIP' "$name"
    else
        printf '  %-24s FAIL' "$name"
        FAILED=1
    fi
    if [ -n "$detail" ]; then
        printf '  (%s)' "$detail"
    fi
    printf '\n'
}

echo "=== CNN Accelerator Environment Verification ==="
echo "Baseline: docs/DEVELOPMENT_SETUP.md"
echo

# --- Python -------------------------------------------------------------
echo "[Python]"

# Prefer the project virtualenv; otherwise fall back to python3 on PATH.
PYTHON_BIN=""
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "$cand")"
            break
        fi
    done
fi

if [ -z "$PYTHON_BIN" ]; then
    report "Python interpreter" FAIL "python3 not found on PATH (activate the project venv first)"
else
    pyver="$("$PYTHON_BIN" --version 2>&1)"
    case "$pyver" in
        *"Python 3.10."*) report "Python version" PASS "$pyver" ;;
        *) report "Python version" FAIL "$pyver (expected Python 3.10.x)" ;;
    esac

    # check_module <module> <expected version>
    check_module() {
        local module="$1" expected="$2"
        local got
        got="$("$PYTHON_BIN" -c "import $module; print($module.__version__)" 2>/dev/null)"
        if [ -n "$got" ] && [ "$got" = "$expected" ]; then
            report "python: $module" PASS "$got"
        else
            report "python: $module" FAIL "${got:-not importable} (expected $expected)"
        fi
    }

    check_module torch 2.1.2+cpu
    check_module torchvision 0.16.2+cpu
    check_module numpy 1.26.4
fi
echo

# --- Vivado -------------------------------------------------------------
echo "[Vivado]"

VIVADO_BIN="$(command -v vivado 2>/dev/null || true)"

for tool in vivado xvlog xelab xsim; do
    if command -v "$tool" >/dev/null 2>&1; then
        report "tool: $tool" PASS
    else
        report "tool: $tool" FAIL "not on PATH ($VIVADO_SETTINGS_HINT)"
    fi
done

if [ -n "$VIVADO_BIN" ]; then
    vver="$("$VIVADO_BIN" -version 2>&1 | head -n 1)"
    if printf '%s' "$vver" | grep -q '2023\.1'; then
        report "Vivado version" PASS "$vver"
    else
        report "Vivado version" FAIL "$vver (expected 2023.1)"
    fi
else
    report "Vivado version" FAIL "vivado not on PATH ($VIVADO_SETTINGS_HINT)"
fi

# Target device availability, only when Vivado is available.
if [ -n "$VIVADO_BIN" ]; then
    dev_out="$(
        printf '%s\n' \
            'if {[llength [get_parts xck26-sfvc784-2LV-c*]] > 0} {puts DEVICE_OK} else {puts DEVICE_MISSING}' |
        "$VIVADO_BIN" -mode tcl -nolog -nojournal -notrace 2>&1
    )"
    if printf '%s' "$dev_out" | grep -q 'DEVICE_OK'; then
        report "Target device" PASS "xck26-sfvc784-2LV-c"
    else
        report "Target device" FAIL "xck26-sfvc784-2LV-c not found in the Vivado device database"
    fi
else
    report "Target device" SKIP "vivado not available"
fi
echo

# --- Result -------------------------------------------------------------
if [ "$FAILED" -eq 0 ]; then
    echo "Result: PASS"
else
    echo "Result: FAIL"
fi

exit "$FAILED"
