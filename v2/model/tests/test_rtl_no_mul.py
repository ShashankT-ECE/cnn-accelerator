"""Step 4: no runtime multiply / divide / modulo in the RTL address and counter paths.

gos_ctrl.sv (issue-stage counters and addresses) is checked entirely; in gos_core.sv the
drain stage + writer (output word / QPARAM index / LOGIT index) section is checked. Comments
are stripped. Elaboration-time constants are allowed: declaration lines (localparam /
parameter / logic width brackets / .TW(...) overrides) and products of two ALL-CAPS
parameters (e.g. N*ACC_W in a part-select width). `timescale / import pkg::* are skipped.
"""
import re

from common import V2_ROOT

RTL = V2_ROOT / "rtl"
OPS = re.compile(r"(?<![*/])(\*|/|%)(?![*/])")


def code_lines(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    for i, line in enumerate(text.splitlines(), 1):
        yield i, line.split("//", 1)[0]


def is_constant_decl(line):
    s = line.strip()
    return (s.startswith(("localparam", "parameter")) or ".TW(" in s
            or re.match(r"^(input|output)?\s*logic\s*(signed\s*)?\[", s) is not None)


CONST_PROD = re.compile(r"\b[A-Z_][A-Z0-9_]*\s*\*\s*[A-Z_][A-Z0-9_]*\b")


def offending(text):
    out = []
    for i, l in code_lines(text):
        s = l.strip()
        if s.startswith("`timescale") or s.startswith("import") or is_constant_decl(l):
            continue
        if OPS.search(CONST_PROD.sub("", l.replace("::*", ""))):
            out.append((i, s))
    return out


def test_gos_ctrl_has_no_mul_div_mod():
    assert offending((RTL / "gos_ctrl.sv").read_text()) == []


def test_gos_core_drain_writer_has_no_mul_div_mod():
    t = (RTL / "gos_core.sv").read_text()
    s = t.index("\n", t.index("---- drain stage"))
    e = t.index("sticky error flags")
    assert offending(t[s:e]) == []


def test_checker_is_comparison_only():
    assert offending((RTL / "gos_cfg_check.sv").read_text()) == []


def test_detector_catches_a_multiply():
    assert offending("  assign w = a * b;\n") and offending("x <= y % 8;") and not offending("// a*b")
    assert offending("  assign w = N * b;") and not offending("  x = y[3 +: N*ACC_W];")
