"""WP7b: the wave 2 components against the stored WP7b references (benchmarks/validation_v2/data), at cheap points.

Tolerances of CONVENTIONS.md section 8 for two independent methods: 0.01 dB on the main transmission, 1e-4 of the
input power for every mode converted from the incident one (reflected and transmitted). The full comparisons are
benchmarks/validation_v2/compare_v2.py and REPORT.md; known disagreements are strict xfails pointing to their
discrepancy report.
"""
import json
import os

import numpy as np
import pytest

import wrmodes as wr
from wrmodes.components import Guide

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "benchmarks", "validation_v2", "data")
W90, H90 = 22.86e-3, 10.16e-3
INCH = 0.0254


def load(name):
    with np.load(os.path.join(DATA, name), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def gsm_ref(d, key):
    return dict(S11=d[f"{key}_S11"], S21=d[f"{key}_S21"], n1=[str(s) for s in d[f"{key}_names1"]],
                n2=[str(s) for s in d[f"{key}_names2"]])


def check(ref, g, inc, db_tol=0.01, p_tol=1e-4):
    i_r, i_t = ref["n1"].index(inc), g.basis1.names.index(inc)
    main_r = ref["S21"][ref["n2"].index(inc), i_r]
    main_t = g.S21[g.basis2.names.index(inc), i_t]
    assert abs(20 * np.log10(abs(main_t) / abs(main_r))) <= db_tol
    for blk, nr, nt in (("S21", ref["n2"], g.basis2.names), ("S11", ref["n1"], g.basis1.names)):
        S = getattr(g, blk)
        for j, n in enumerate(nr):
            if blk == "S21" and n == inc:
                continue
            assert abs(abs(S[nt.index(n), i_t]) ** 2 - abs(ref[blk][j, i_r]) ** 2) <= p_tol, (blk, n)


def finest(d, f):
    runs = [r for r in d["meta"]["runs"] if r["f"] == f]
    dmax = max(r["density"] for r in runs)
    nmax = max(r["n_seg"] for r in runs)
    return [r for r in runs if r["density"] == dmax and r["n_seg"] == nmax][0]["key"]


@pytest.mark.parametrize("case,L", [("H_linear_L100", 0.1), ("H_linear_L200", 0.2)])
def test_t2b_linear_taper_against_fem(case, L):
    d = load(f"taper_fem_{case}.npz")
    ref = gsm_ref(d, finest(d, 15e9))
    t = wr.Taper(L, 45e-3, H90, profile="linear", guide=Guide(W90, H90))
    b1, b2 = t.default_basis(15e9)
    check(ref, t.gsm(15e9, b1, b2), "TE1,0")


@pytest.mark.parametrize("f", [18e9, 30e9])
def test_raised_cosine_e_taper_against_fem(f):
    d = load("taper_fem_E_rc50.npz")
    ref = gsm_ref(d, finest(d, f))
    t = wr.Taper(0.05, 1.6e-3, W90, profile="raised_cosine", plane="E", guide=Guide(W90, H90))
    b1, b2 = t.default_basis(f)
    check(ref, t.gsm(f, b1, b2), "TE0,1")


def test_strangled_bend_against_fem_18GHz():
    d = load("taper_fem_strangled.npz")
    ref = gsm_ref(d, finest(d, 18e9))
    s = wr.StrangledBend(90, 0.05, 1.6e-3, 0.05, guide=Guide(W90, H90))
    b1, b2 = s.default_basis(18e9)
    check(ref, s.gsm(18e9, b1, b2), "TE0,1")


def _twist_ref(name, f):
    d = load(f"twist_{name}.npz")
    key = f"{name}_f{f / 1e9:g}_main"
    names = [str(s) for s in d[f"{key}_names"]]
    return dict(S11=d[f"{key}_S11"], S21=d[f"{key}_S21"], n1=names, n2=names), d["meta"]["geometry"]


@pytest.mark.parametrize("f", [15e9, 16.5e9, 18e9])
def test_doane_triangular_twist_against_helical_fem(f):
    ref, c = _twist_ref("doane_triangular", f)
    tw = wr.Twist(90, c["L"], profile="triangular", guide=Guide(c["a"], c["b"]))
    check(ref, tw.gsm(f), "TE1,0")


def test_doane_constant_twist_against_helical_fem_15GHz():
    """Strict xfail until WF2 (docs/discrepancies/2026-11-15-wp7b-twist-truncation.md, Resolution): 4.1e-4 at WP6's
    default, 5e-6 at WF2's (internal list to 3 f, quasi-static tail)."""
    ref, c = _twist_ref("doane_constant", 15e9)
    tw = wr.Twist(90, c["L"], profile="constant", guide=Guide(c["a"], c["b"]))
    check(ref, tw.gsm(15e9), "TE1,0")


@pytest.mark.parametrize("plane,d,f", [("H", 0.7071, 26.3), ("H", 0.8, 52.5), ("E", 0.7071, 30.0), ("E", 0.9, 10.0)])
def test_miter_mm_against_fem_sweep(plane, d, f):
    z = load(f"miter_fem_{plane}_d{d:.4f}.npz")
    ref = gsm_ref(z, f"{plane}_d{d:.4f}_f{f:g}_main")
    m = wr.MiterMM(None if d < 0.71 else d, plane=plane, guide=Guide(W90, H90))
    basis = Guide(W90, H90).basis(f * 1e9, 1.0, plane)
    check(ref, m.gsm(f * 1e9, basis, basis), "TE1,0" if plane == "H" else "TE0,1")
