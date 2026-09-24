#!/usr/bin/env bash
# vivado_guard.sh — run before EVERY Vivado batch job (user rule after the 2026-09-24 OOM freeze):
#   * at most ONE Vivado job at a time: refuse if another vivado process is running;
#   * print `free -h`; refuse if available memory is below ${GOS_MIN_AVAIL_GB:-8} GB.
# Exit 0 = OK to start; 1 = refused (the caller must not start Vivado).
set -uo pipefail
MIN_GB="${GOS_MIN_AVAIL_GB:-8}"
free -h
if pgrep -x vivado > /dev/null 2>&1; then
    echo "vivado_guard: REFUSED — another Vivado job is running:" >&2
    pgrep -ax vivado >&2
    exit 1
fi
avail_mb="$(free -m | awk '/^Mem:/ {print $7}')"
if (( avail_mb < MIN_GB * 1024 )); then
    echo "vivado_guard: REFUSED — available memory ${avail_mb} MB < ${MIN_GB} GB" >&2
    exit 1
fi
echo "vivado_guard: OK (available ${avail_mb} MB, no other Vivado job)"
