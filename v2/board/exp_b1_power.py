#!/usr/bin/env python3
"""B1 entry point: the INA260 SOM-rail power protocol for each net (thin wrapper of power_log.py).

    ./session.sh py exp_b1_power.py [--nets lenet5 cifar10] [power_log.py protocol options]
    python3 exp_b1_power.py --backend model --phase-s 2 --sensor mock   # dry run -> results/dryrun/

B1 power is measured ONLY with the on-board INA260 on the SOM rail (VCC_SOM), label
"SOM-rail power (INA260)" (EXPERIMENTS.md B1, decision 2026-09-29). This
script runs `power_log.py --protocol --net <net> --tag _<net>` once per net with the remaining
arguments passed through; run_sessions.py schedules power_log.py directly (steps s3.B1.<net>).
Outputs: hw_b1_power_ina260_{samples,phases,summary}_<net>.csv per net.
"""
from __future__ import annotations

import sys

SENSOR_LABEL = "SOM-rail power (INA260)"      # = power_log.LABEL
NETS = ("lenet5", "cifar10")


def split_nets(argv: list[str]) -> tuple[list[str], list[str]]:
    nets, rest, i = list(NETS), [], 0
    while i < len(argv):
        if argv[i] == "--nets":
            j = i + 1
            nets = []
            while j < len(argv) and not argv[j].startswith("--"):
                nets.append(argv[j])
                j += 1
            i = j
            continue
        rest.append(argv[i])
        i += 1
    return nets, rest


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    nets, rest = split_nets(argv)
    import power_log                              # after argv parsing (thread env set by caller)
    rc = 0
    for net in nets:
        print(f"[B1] {SENSOR_LABEL} protocol, net {net}")
        rc = max(rc, power_log.main(["--protocol", "--net", net, "--tag", f"_{net}", *rest]) or 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
