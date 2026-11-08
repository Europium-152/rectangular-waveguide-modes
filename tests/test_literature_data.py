"""Format checks of the digitized literature data in benchmarks/literature/ (card L1, CONVENTIONS.md section 9).

Every final CSV must parse, carry the required columns and a descriptive header, contain finite values with
positive uncertainties, and be listed in benchmarks/literature/README.md. The raw digitizations of both agents must
be kept next to them for traceability.
"""
import csv
import math
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIT = os.path.join(ROOT, "benchmarks", "literature")
DOCS = os.path.join(ROOT, "docs", "literature")

CURVE_COLUMNS = ["curve", "x", "y", "uncertainty", "x_uncertainty", "source"]
TABLE_COLUMNS = ["steps", "case", "quantity", "value", "uncertainty", "source"]
QUOTE_COLUMNS = ["id", "source", "page", "pdf_page", "quantity", "value", "unit", "conditions", "quote", "notes",
                 "transcribed_by", "l1b_id"]
SPECIAL = {"ma1997_table1.csv": TABLE_COLUMNS, "butterweck1968_numbers.csv": QUOTE_COLUMNS}
CSV_FILES = sorted(f for f in os.listdir(LIT) if f.endswith(".csv"))
CURVE_FILES = [f for f in CSV_FILES if f not in SPECIAL]


def read(name):
    with open(os.path.join(LIT, name), newline="") as fh:
        lines = fh.read().splitlines()
    header = [ln for ln in lines if ln.startswith("#")]
    body = [ln for ln in lines if not ln.startswith("#")]
    rows = list(csv.reader(body))
    return header, rows[0], rows[1:]


def test_expected_files_present():
    expected = {"alessandri1994_fig4.csv", "alessandri1994_fig5.csv", "doane1989_fig9.csv", "ma1997_fig5.csv",
                "ma1997_fig6.csv", "ma1997_fig8.csv", "ma1997_fig9.csv", "ma1997_fig11.csv", "ma1997_table1.csv",
                "reiter1995_fig3a.csv", "reiter1995_fig3b.csv", "reiter1995_fig3c.csv", "reiter1995_fig3d.csv",
                "reiter1995_fig4.csv", "butterweck1968_numbers.csv"}
    assert expected <= set(CSV_FILES)
    for doc in ("marcuvitz-steps-and-bends.md", "doane1989.md"):
        assert os.path.isfile(os.path.join(DOCS, doc))


@pytest.mark.parametrize("name", CURVE_FILES)
def test_curve_file(name):
    header, cols, rows = read(name)
    assert cols == CURVE_COLUMNS
    text = "\n".join(header)
    for key in ("# Source:", "# Figure:", "# Axes:", "# Merge", "Calibration L1a", "Calibration L1b", "Agreement"):
        assert key in text, f"{name}: header lacks {key!r}"
    assert len(rows) > 5
    for r in rows:
        assert len(r) == len(CURVE_COLUMNS), r
        assert r[0].strip()
        x, y, u, xu = (float(v) for v in r[1:5])
        assert all(math.isfinite(v) for v in (x, y, u, xu)), r
        assert u > 0, r
        assert xu >= 0, r
        assert r[5] in ("ab", "a", "b", "L1"), r


@pytest.mark.parametrize("name", CURVE_FILES)
def test_curve_x_sorted_within_line_curves(name):
    """Line curves (x_uncertainty = 0) are on a grid in increasing x, without duplicates."""
    _, _, rows = read(name)
    by_curve = {}
    for r in rows:
        if float(r[4]) == 0.0:
            by_curve.setdefault(r[0], []).append(float(r[1]))
    for curve, xs in by_curve.items():
        assert all(b > a for a, b in zip(xs, xs[1:])), f"{name}: {curve}"


def test_table_file():
    header, cols, rows = read("ma1997_table1.csv")
    assert cols == TABLE_COLUMNS
    assert any(h.startswith("# Source:") for h in header)
    assert len(rows) == 36
    for r in rows:
        assert math.isfinite(float(r[3])) and float(r[4]) > 0, r


def test_quote_file():
    header, cols, rows = read("butterweck1968_numbers.csv")
    assert cols == QUOTE_COLUMNS
    assert len(rows) >= 14
    for r in rows:
        d = dict(zip(cols, r))
        assert d["quote"].strip() and d["value"].strip(), r
        assert int(d["page"]) > 0 and int(d["pdf_page"]) > 0, r


def test_readme_lists_every_file():
    with open(os.path.join(LIT, "README.md")) as fh:
        readme = fh.read()
    for name in CSV_FILES:
        assert f"`{name}`" in readme, f"README.md does not list {name}"


@pytest.mark.parametrize("name", [f for f in CSV_FILES if f != "butterweck1968_numbers.csv"])
def test_raw_digitizations_kept(name):
    for s in ("a", "b"):
        assert os.path.isfile(os.path.join(LIT, "raw", s, name)), f"raw/{s}/{name} missing"
