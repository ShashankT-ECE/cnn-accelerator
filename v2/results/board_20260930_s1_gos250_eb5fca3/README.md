# Board Session 1 rerun of 2026-09-30 — gos_250 at 249.9975 MHz (scripts/data commit eb5fca3)

Measured on the KV260 with `./session.sh 1 --fresh`. gos_300 failed its pre-flight (300 MHz is not
reachable from the boot image's PLLs, DECISIONS D19) and the session fell back to gos_250 (build
fd880d43, closed 249.997498 MHz, read-back 249.997500 MHz, set after the overlay load from
166.665 MHz). All five steps passed (shell, core smoke, slice smoke, fast-path check, fcal); the
fcal cross-check read 249.886 MHz (-0.0445 % vs read-back, tolerance 0.1 %).
This is bring-up evidence, superseded by the Session 1 that precedes the Session 2 run at the
final commit; it is not paper data (`check_results.py` reads only `v2/results/*.csv`).
`raw_board_results.tar.xz` = verbatim `~/gos/results/` (step and orchestrator logs, env log),
SHA256-verified against the board after the copy.
