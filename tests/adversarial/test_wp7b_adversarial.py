"""WP7b adversarial tests of the wave 2 components without the FEM extra: WP5 (Taper, VariableGuide, StrangledBend),
WP6 (Twist) and WP4b (MiterMM, Step parity).

Written from the task cards, CONVENTIONS.md and the components' docstrings only (card WP7b item 7). Weak points:
modes exactly at or within 1e-6 of a cutoff inside or at the end of a taper, a flat guide just below and just above a
cutoff, very long tapers at 90 GHz, a 360 degree twist, nearly degenerate pairs in a twisted nearly square guide,
reversal symmetries, and the symmetric miter with its port frames solved without the symmetry shortcut.
"""
import numpy as np
import pytest
import scipy.constants as cst

import wrmodes as wr
from wrmodes.components import Guide
from wrmodes.modes import Mode, propagation_constant

W90, H90 = 22.86e-3, 10.16e-3
TE10 = Mode("TE", 1, 0)


def _finite_lossless(g, unit=1e-8, recip=1e-10):
    assert np.all(np.isfinite(g.S))
    assert g.reciprocity_defect() <= recip * max(1.0, np.max(np.abs(g.S)))
    assert g.unitarity_defect() <= unit


def _differ(b1, b2):
    return not (b1 == b2)


# ----------------------------------------------------------------------------------------------- WP5: tapers
def test_taper_mode_exactly_at_cutoff_inside():
    """TE3,0 reaches its cutoff exactly at z = L/2 of a linear H-plane taper (a cutoff plane inside the component).
    The V-I method must stay regular there (T2 section 8.1); the result must be continuous in f across the value
    for which the cutoff plane sits exactly at the slice midpoint."""
    f = 15e9
    a_c = 3 * cst.c / (2 * f)                       # TE3,0 cutoff width at f
    a1 = 22.86e-3
    a2 = 2 * a_c - a1                               # linear: a(L/2) = a_c exactly
    L = 0.1
    t = wr.Taper(L, a2, H90, profile="linear", guide=Guide(a1, H90))
    g0 = t.gsm(f)
    _finite_lossless(g0)
    g1 = t.gsm(f * (1 + 1e-9))
    i, j = g0.basis2.index(Mode("TE", 3, 0)), g0.basis1.index(TE10)
    assert abs(g1.S21[i, j] - g0.S21[i, j]) < 1e-5
    assert abs(g1.S11[j, j] - g0.S11[j, j]) < 1e-5


@pytest.mark.parametrize("rel", [1e-6, 2e-7])
def test_taper_end_within_1e_6_of_a_cutoff(rel):
    """Port 2 of an H-plane taper with TE3,0 at f_c = f (1 - rel): either a clean ValueError (documented: port modes
    within 1e-6 k^2 of their cutoff raise) or a finite, unitary GSM. Never NaN or a silent blow-up."""
    f = 15e9
    a2 = 3 * cst.c / (2 * f * (1 - rel))
    t = wr.Taper(0.1, a2, H90, profile="raised_cosine", guide=Guide(W90, H90))
    try:
        g = t.gsm(f)
    except ValueError:
        assert (1 - (1 - rel) ** 2) < 1e-6 * 1.0001   # only allowed inside the documented band
        return
    _finite_lossless(g, unit=1e-7)


@pytest.mark.parametrize("side", [-1, +1])
def test_strangled_bend_flat_guide_just_below_and_above_a_cutoff(side):
    """Flat height b1 with the local TE1,0 (E frame; E across the flat side) 0.1 % below / above cutoff at f in the
    flat guide, so that the mode coupled by the E-plane curvature is barely propagating or barely evanescent there.
    The component is symmetric (port 2 frame transported along the centre line): S11 = S22 without a sign matrix."""
    f = 60e9
    b1 = cst.c / (2 * f) * (1 + side * 1e-3) ** -1    # local a = b1: f_c(TE1,0) = f (1 + side 1e-3)
    sb = wr.StrangledBend(90, 0.05, b1, 0.03, guide=Guide(W90, H90))
    g = sb.gsm(f)
    _finite_lossless(g, unit=1e-7)
    perm = [g.basis2.names.index(n) for n in g.basis1.names]   # compare by name (see the next test)
    assert np.max(np.abs(g.S11 - g.S22[np.ix_(perm, perm)])) <= 1e-7
    i = g.basis1.index(Mode("TE", 0, 1))              # global TE1,0 in the local E frame
    assert abs(g.S21[i, i]) ** 2 + np.sum(np.abs(g.S11[:, i]) ** 2) <= 1 + 1e-8


@pytest.mark.xfail(strict=True, reason="docs/discrepancies/2026-11-15-wp7b-strangled-bend-port-basis-order.md")
def test_strangled_bend_port_bases_are_equal():
    """Both ports of a StrangledBend are the same full-height guide, so basis2 == basis1 (CONVENTIONS 3) and two
    GSMs cascade directly. WR-90 in the E frame has exactly degenerate pairs (22.86 = 2.25 x 10.16 mm: local TE4,0
    and TE0,9 at 59.02 GHz); a port-2 height that differs from h by one rounding error reorders them."""
    f = 60e9
    unequal = [b1 for b1 in np.linspace(1.5e-3, 2.6e-3, 200)
               if _differ(*wr.StrangledBend(90, 0.05, b1, 0.03, guide=Guide(W90, H90)).default_basis(f))]
    assert not unequal, f"{len(unequal)} of 200 flat heights give basis2 != basis1, e.g. b1 = {unequal[0]!r}"


@pytest.mark.slow
def test_long_taper_1m_at_90GHz_is_adiabatic():
    """1 m raised-cosine H-plane taper 22.86 -> 25 mm at 90 GHz (slope 2e-3): TE1,0 must pass with |S21|^2 > 0.999
    and the WKB phase exp(-i int beta dz) within 0.02 rad; lossless and reciprocal with ~120 port modes."""
    f, L, a2 = 90e9, 1.0, 25e-3
    t = wr.Taper(L, a2, H90, profile="raised_cosine", guide=Guide(W90, H90))
    g = t.gsm(f)
    _finite_lossless(g, unit=1e-7)
    i1, i2 = g.basis1.index(TE10), g.basis2.index(TE10)
    s = g.S21[i2, i1]
    assert abs(s) ** 2 > 0.999
    z = np.linspace(0, L, 4001)
    a = W90 + (a2 - W90) * 0.5 * (1 - np.cos(np.pi * z / L))
    beta = np.sqrt((2 * np.pi * f / cst.c) ** 2 - (np.pi / a) ** 2)
    phi = np.trapezoid(beta, z) if hasattr(np, "trapezoid") else np.trapz(beta, z)
    assert abs(np.angle(s * np.exp(1j * phi))) < 0.02


def test_taper_turned_around_equals_the_reverse_taper():
    """Turning a taper around (CONVENTIONS 2: P S_swapped P, `GSM.reversed`) is the taper from a2 back to a1 with
    the mirrored profile; for a linear profile the same profile. Checks frames, signs and both port bases."""
    f = 20e9
    a2 = 30e-3
    fwd = wr.Taper(0.08, a2, H90, profile="linear", guide=Guide(W90, H90)).gsm(f)
    back = wr.Taper(0.08, W90, H90, profile="linear", guide=Guide(a2, H90)).gsm(f)
    rev = fwd.reversed()
    assert rev.basis1 == back.basis1 and rev.basis2 == back.basis2
    assert np.max(np.abs(rev.S - back.S)) < 1e-5


# ----------------------------------------------------------------------------------------------- WP6: twists
def test_360_degree_twist_is_two_180_degree_twists():
    """A constant-rate 360 deg twist equals the cascade of two 180 deg halves (exact expm for a constant rate;
    the co-rotating frame needs no frame change between them)."""
    g = Guide(W90, H90)
    f = 30e9
    full = wr.Twist(360, 1.0, guide=g).gsm(f)
    half = wr.Twist(180, 0.5, guide=g)
    two = half.gsm(f) @ half.gsm(f)
    _finite_lossless(full, unit=1e-9)
    assert np.max(np.abs(full.S - two.S)) < 1e-8


def test_slow_360_degree_twist_is_nearly_adiabatic():
    """A long, slow 360 deg twist (tau a = 0.036 at 30 GHz in WR-90): TE1,0 follows the guide (co-rotating port
    frame), |S21|^2 > 0.99, reflection below 1e-4."""
    g = Guide(W90, H90)
    f = 30e9
    s = wr.Twist(360, 4.0, guide=g).gsm(f)
    i = s.basis1.index(TE10)
    assert abs(s.S21[i, i]) ** 2 > 0.99
    assert np.sum(np.abs(s.S11[:, i]) ** 2) < 1e-4


def test_twist_of_nearly_square_guide_is_continuous():
    """20 x 20.00002 mm (TE1,0 and TE0,1 split by 1e-6 relative): the nearly degenerate pair must give the square
    guide's polarization rotation (T3b box 5), entry by entry, with the modes matched by name."""
    f, L = 12e9, 0.3
    sq = wr.Twist(90, L, guide=Guide(20e-3, 20e-3)).gsm(f)
    ns = wr.Twist(90, L, guide=Guide(20e-3, 20.00002e-3)).gsm(f)
    _finite_lossless(ns, unit=1e-8)
    names = [n for n in sq.basis1.names if n in ns.basis1.names]
    i_s = [sq.basis1.names.index(n) for n in names]
    i_n = [ns.basis1.names.index(n) for n in names]
    d = np.abs(sq.S21[np.ix_(i_s, i_s)] - ns.S21[np.ix_(i_n, i_n)])
    assert np.max(d) < 1e-3


def test_symmetric_twist_turned_around_is_itself():
    """T3 box T3-5: turning a twist around gives the profile theta'(L - z) with the same handedness; a triangular
    (symmetric) twist is therefore its own reverse: P S_swapped P = S."""
    s = wr.Twist(90, 0.3, profile="triangular", guide=Guide(W90, H90)).gsm(25e9)
    assert np.max(np.abs(s.reversed().S - s.S)) < 1e-7


def test_twist_mode_just_above_cutoff():
    """A frequency 1e-5 above the cutoff of TE0,1 in WR-90 (coupled to TE1,0 at first order): finite and unitary."""
    f = cst.c / (2 * H90) * (1 + 1e-5)
    s = wr.Twist(90, 0.4, guide=Guide(W90, H90)).gsm(f)
    _finite_lossless(s, unit=1e-8)


# ----------------------------------------------------------------------------------------------- WP4b: MiterMM
@pytest.mark.parametrize("plane,chamfer", [("H", None), ("E", 0.87), ("H", 1.0)])
def test_miter_mm_port_frames_without_symmetry(plane, chamfer):
    """The full two-port solve (symmetry=False) must give S11 = S22 (no sign matrix, CONVENTIONS 2) and agree with
    the symmetric solve."""
    m = wr.MiterMM(chamfer, plane=plane, guide=Guide(W90, H90))
    f = 33e9
    a = m.gsm(f, symmetry=False)
    b = m.gsm(f)
    assert np.max(np.abs(a.S11 - a.S22)) < 1e-8
    assert np.max(np.abs(a.S - b.S)) < 1e-6


@pytest.mark.parametrize("plane", ["H", "E"])
def test_miter_mm_just_above_a_cutoff(plane):
    """1e-6 above the cutoff of the third in-plane mode (H: TE3,0; E: local TE3,0 = global TE0,3 family): finite,
    unitary, symmetric."""
    a = W90 if plane == "H" else H90
    f = 3 * cst.c / (2 * a) * (1 + 1e-6)
    g = wr.MiterMM(None, plane=plane, guide=Guide(W90, H90)).gsm(f)
    _finite_lossless(g, unit=1e-8)
    assert np.max(np.abs(g.S11 - g.S22)) < 1e-10


def test_miter_mm_negative_angle_is_mirror_image():
    """angle_deg = -90 is the mirror u -> -u: D S D with D = (-1)^(m+1)."""
    g = Guide(W90, H90)
    f = 27e9
    p = wr.MiterMM(0.9, guide=g).gsm(f)
    n = wr.MiterMM(0.9, guide=g, angle_deg=-90).gsm(f)
    D = np.array([(-1.0) ** (m.m + 1) for m in p.basis1.modes])
    DD = np.concatenate([D, D])
    assert np.max(np.abs(n.S - DD[:, None] * p.S * DD[None, :])) < 1e-10


def test_two_miters_back_to_back_zero_spacer_conserve_power():
    """Two mode-matching miters joined with no straight (a U-turn, zero-length cascade) on a basis with evanescent
    modes (cutoff_factor 3): lossless and reciprocal; and the result with cutoff_factor 3 and 4 agree for TE1,0
    (the zero-length cascade needs the evanescent entries, CONVENTIONS 8)."""
    g = Guide(W90, H90)
    line = wr.Line(g, [wr.MiterMM(None), wr.MiterMM(None)])
    f = 20e9
    a = line.gsm(f, cutoff_factor=3.0)
    b = line.gsm(f, cutoff_factor=4.0)
    assert a.unitarity_defect() < 1e-8 and a.reciprocity_defect() < 1e-10
    ia, ib = a.basis1.index(TE10), b.basis1.index(TE10)
    assert abs(abs(a.S21[ia, ia]) - abs(b.S21[ib, ib])) < 1e-3


def test_step_parity_reduction_on_a_flush_offset_step():
    """WP4b parity reduction must not be applied to an offset step (du != 0 breaks the m parity): reduce=True and
    reduce=False agree to 1e-12 for an offset 2D step."""
    st = wr.Step(W90, H90, 15e-3, 6e-3, du=2e-3, dv=1e-3)
    f = 25e9
    b1, b2 = st.default_basis(f)
    x = st.solve(f, b1, b2, reduce=True)
    y = st.solve(f, b1, b2, reduce=False)
    assert np.max(np.abs(x.S - y.S)) < 1e-12
