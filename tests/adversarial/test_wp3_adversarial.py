"""Adversarial tests of WP3 (2D modal-port solver, H-plane miter), from tasks/WP3-2d-solver.md and CONVENTIONS.md.

They skip until wrmodes.fem2d is merged and its optional extra is installed. The miter is used through a WP1 Line
(Line(Guide(w, h), [Miter(...)])) so that dimensions and frames come from the guide. Weak points: frequencies just
above a cutoff (port modes with tiny beta), many propagating modes, zero bend angle, symmetry and reference
planes, and the one precise literature number (Ma et al. 1997, Table 1).
"""
import numpy as np
import pytest
import scipy.constants as cst

import ref_interfaces as I
import ref_literature as L
from wrmodes.modes import Mode

WR90 = (22.86e-3, 10.16e-3)
TE10 = Mode("TE", 1, 0)


def _miter_line(angle=90, chamfer=None, plane="H", w=WR90[0], h=WR90[1]):
    Miter, Guide, Line = I.require("Miter", "Guide", "Line")
    return Miter, Line(Guide(w, h), [Miter(angle, chamfer=chamfer, plane=plane)])


def _checks(g, unit_tol=1e-5):
    assert np.all(np.isfinite(g.S))
    assert g.reciprocity_defect() <= 1e-10 * np.max(np.abs(g.S))
    assert g.unitarity_defect() <= unit_tol
    assert np.max(np.abs(g.S11 - g.S22)) <= 1e-10            # symmetric miter, no sign matrix (CONVENTIONS 2)


@pytest.mark.parametrize("m,rel", [(3, 1e-3), (3, 1e-5), (8, 1e-4)])
def test_h_miter_just_above_a_cutoff(m, rel):
    """TE_m0 just above cutoff: its port mode has beta ~ sqrt(2 rel) k; DtN ports must stay well posed."""
    fc = m * cst.c / (2 * WR90[0])
    _, line = _miter_line()
    _checks(line.gsm(fc * (1 + rel), cutoff_factor=1.0))


def test_h_miter_with_13_propagating_modes():
    _, line = _miter_line()
    g = line.gsm(89.9e9, cutoff_factor=1.0)
    assert sum(1 for m in g.basis1.modes if m.n == 0) == 13
    _checks(g)


@pytest.mark.xfail(strict=True, reason=(
    "Ma et al. (1997) Table 1 disagrees with three independent methods: the FEM (WP3, open issue 3), mode matching "
    "(WP4b) and finite differences (WP4b) all give |S11| = 0.2778, 6 % above 0.2622; see "
    "docs/discrepancies/2026-11-11-alimenti-chamfered-miters.md"))
def test_h_miter_against_ma1997_table1():
    """Ma et al. (1997) Table 1: 15.799 x 7.899 mm guide, c/a = 1 (diagonal miter), 16 GHz: |S11| = 0.2622 and
    |S21| = 0.9650, with a spread of 0.57 % and 0.04 % under displacement of their staircase."""
    t = L.MA1997_TABLE1
    _, line = _miter_line(chamfer=L.ma_chamfer(t["c_over_a"]), w=t["a"], h=t["b"])
    g = line.gsm(t["f"])
    i, j = g.basis1.index(TE10), g.basis2.index(TE10)
    assert abs(abs(g.S11[i, i]) / t["S11"] - 1) <= t["rel_unc_S11"]
    assert abs(abs(g.S21[j, i]) / t["S21"] - 1) <= t["rel_unc_S21"]


@pytest.mark.parametrize("angle", [30, 60, 90, 120])
def test_arm_length_is_half_a_tan_half_angle(angle):
    """CONVENTIONS.md 6: the reference planes are (a/2) tan(theta/2) from the axes' crossing point."""
    Miter, Guide = I.require("Miter", "Guide")
    a = WR90[0]
    arm = Miter(angle, guide=Guide(*WR90)).arm_length   # metres; the miter needs its guide (WP3 interface)
    expected = a / 2 * np.tan(np.radians(angle) / 2)
    assert abs(arm - expected) <= 1e-12 * a


def test_zero_angle_miter_is_a_zero_length_straight():
    """theta = 0: no discontinuity and arm_length 0, so S must be the identity (or a clear ValueError)."""
    try:
        _, line = _miter_line(angle=0)
        g = line.gsm(30e9, cutoff_factor=1.0)
    except ValueError:
        pytest.skip("Miter(0) refused with ValueError")
    n = len(g.basis1)
    assert np.allclose(g.S21, np.eye(n), atol=1e-5) and np.allclose(g.S11, 0, atol=1e-5)


def test_reversed_h_miter_equals_itself():
    """A symmetric miter turned around (GSM.reversed) is the same component: S equal to 1e-10."""
    _, line = _miter_line()
    g = line.gsm(45e9, cutoff_factor=1.0)
    r = g.reversed()
    assert np.max(np.abs(r.S - g.S)) <= 1e-10
