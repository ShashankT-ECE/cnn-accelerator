#!/usr/bin/env python3
"""KV260 / PYNQ smoke test of the V2 PL shell (Step 4 empty shell; reusable for the real design).

    sudo -E python3 test_shell.py [--bit gos_shell.bit] [--expect-version 0x474F5300]
                                  [--seed 1] [--skip-scratch] [--mems ACT0,ACT1,WGT,QPARAM]
                                  [--set-fclk0 MHZ]

<name>.bit and <name>.hwh must sit side by side (same basename). Checks, in order:
  1. load the overlay; print pl_clk0 as configured by PYNQ (Clocks.fclk0_mhz) — PYNQ programs the
     divisors from the .hwh onto the source PLL the boot firmware set up, so this is the number to
     trust on the board, not the Vivado value; FAIL if it exceeds --max-fclk0 (default 200.5 MHz);
  2. CSR VERSION (0x0F8) == --expect-version (repeatable; shell 0x474F5300, real core 0x474F5302);
     print BUILD_ID (0x0FC);
  3. SCRATCH0..3 (0x0E0..0x0EC, shell only; --skip-scratch for the real design): write/read;
  4. each memory: fill every 64-bit word with seeded random data, read everything back.

Memory access (FORMATS.md "PL top ports and address map"): every BRAM word is 64 bit and is
accessed as two 32-bit MMIO words, little-endian: word w lives at byte offset 8*w;
bits 31:0 at offset 8*w, bits 63:32 at offset 8*w + 4. QPARAM odd words (combined word 2i+1) keep
only bits 5:0 (s); bits 63:6 always read 0 (FORMATS.md section 3), so the expected value is masked.

Prints PASS/FAIL and exits 0 on PASS, 1 on FAIL (first mismatch per test is reported).
"""
import argparse
import sys

import numpy as np

CSR_BASE = 0xA000_0000
MEMS = {  # name: (base, size in bytes)
    "QPARAM": (0xA001_0000, 4 * 1024),
    "ACT0":   (0xA002_0000, 32 * 1024),
    "ACT1":   (0xA004_0000, 32 * 1024),
    "WGT":    (0xA008_0000, 128 * 1024),
}
CSR_SIZE = 4 * 1024
OFF_SCRATCH = (0x0E0, 0x0E4, 0x0E8, 0x0EC)
OFF_VERSION = 0x0F8
OFF_BUILD_ID = 0x0FC


def expected_words(name, words):
    if name == "QPARAM":
        words = words.copy()
        words[1::2] &= np.uint64(0x3F)
    return words


def test_mem(MMIO, name, rng):
    base, size = MEMS[name]
    n = size // 8
    mm = MMIO(base, size)
    words = rng.integers(0, 2**64, size=n, dtype=np.uint64)
    lo = (words & np.uint64(0xFFFF_FFFF)).astype(np.uint32)
    hi = (words >> np.uint64(32)).astype(np.uint32)
    for w in range(n):
        mm.write(8 * w, int(lo[w]))
        mm.write(8 * w + 4, int(hi[w]))
    exp = expected_words(name, words)
    bad = 0
    first = None
    for w in range(n):
        got = (mm.read(8 * w + 4) << 32) | mm.read(8 * w)
        if got != int(exp[w]):
            bad += 1
            if first is None:
                first = (w, got, int(exp[w]))
    if bad:
        w, got, e = first
        print(f"  {name}: FAIL {bad}/{n} words mismatch; first at word {w} (0x{base + 8 * w:08X}): "
              f"got 0x{got:016X} expected 0x{e:016X}")
        return False
    print(f"  {name}: PASS {n} x 64-bit words at 0x{base:08X}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bit", default="gos_shell.bit")
    ap.add_argument("--expect-version", action="append", type=lambda s: int(s, 0),
                    help="accepted VERSION value (repeatable); default 0x474F5300")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-scratch", action="store_true")
    ap.add_argument("--mems", default="ACT0,ACT1,WGT,QPARAM")
    ap.add_argument("--set-fclk0", type=float, default=None, help="set pl_clk0 (MHz) after loading")
    ap.add_argument("--max-fclk0", type=float, default=200.5)
    a = ap.parse_args()
    versions = a.expect_version or [0x474F5300]

    from pynq import MMIO, Overlay
    from pynq.ps import Clocks

    ok = True
    print(f"test_shell: loading {a.bit}")
    ol = Overlay(a.bit)
    if a.set_fclk0 is not None:
        Clocks.fclk0_mhz = a.set_fclk0
    fclk0 = Clocks.fclk0_mhz
    print(f"  pl_clk0 (PYNQ Clocks.fclk0_mhz) = {fclk0:.3f} MHz")
    if fclk0 > a.max_fclk0:
        print(f"  FAIL: pl_clk0 {fclk0:.3f} MHz > {a.max_fclk0} MHz (timing closed for 200 MHz only)")
        ok = False

    # address map cross-check against the .hwh (informational)
    try:
        segs = {k: (v["phys_addr"], v["addr_range"]) for k, v in ol.ip_dict.items()}
        print("  ip_dict:", ", ".join(f"{k}@0x{b:08X}/{r // 1024}K" for k, (b, r) in sorted(segs.items())))
    except Exception as e:  # noqa: BLE001 - informational only
        print(f"  (ip_dict not available: {e})")

    csr = MMIO(CSR_BASE, CSR_SIZE)
    ver = csr.read(OFF_VERSION)
    bid = csr.read(OFF_BUILD_ID)
    vok = ver in versions
    ok &= vok
    print(f"  VERSION  = 0x{ver:08X} ({'PASS' if vok else 'FAIL, expected ' + '/'.join(f'0x{v:08X}' for v in versions)})")
    print(f"  BUILD_ID = 0x{bid:08X} (short git commit {bid:08x})")

    rng = np.random.default_rng(a.seed)
    if not a.skip_scratch:
        pats = [int(x) for x in rng.integers(0, 2**32, size=4, dtype=np.uint64)]
        for off, p in zip(OFF_SCRATCH, pats):
            csr.write(off, p)
        sok = True
        for off, p in zip(OFF_SCRATCH, pats):
            got = csr.read(off)
            if got != p:
                print(f"  SCRATCH 0x{off:03X}: FAIL got 0x{got:08X} expected 0x{p:08X}")
                sok = False
        if csr.read(OFF_VERSION) != ver:
            print("  SCRATCH: FAIL VERSION changed after scratch writes")
            sok = False
        print(f"  SCRATCH0..3: {'PASS' if sok else 'FAIL'}")
        ok &= sok

    for name in [m.strip().upper() for m in a.mems.split(",") if m.strip()]:
        if name not in MEMS:
            print(f"  unknown memory {name}")
            ok = False
            continue
        ok &= test_mem(MMIO, name, rng)

    print("TEST PASSED" if ok else "TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
