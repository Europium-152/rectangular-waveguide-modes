"""Tests of wrmodes.components.Twist (WP6, card tasks/WP6-twists.md).

Theory: docs/theory/T3-twist-coupling.md and docs/theory/T3b-twist-coupling.md (the closed forms used are T3b's).
Report: docs/reports/WP6.md. Slow cases are marked `slow`; timing limits are strict only with WRMODES_PERF=1.
"""
import os
import time

import numpy as np
import pytest
import scipy.constants as cst

import wrmodes as wr
from wrmodes.components import Guide, Straight, SmoothBend, SechCurvature, Twist, TwistProfile
from wrmodes.components.twist import (twist_coupling_matrix, twist_system_matrices, helical_propagation_constants,
                                      zero_length_twist_transfer_matrix, twist_gsm_blocks, twist_tail_matrix,
                                      twist_tail_system_matrix, DEFAULT_MODE_FACTOR)
from wrmodes.gsm import GSM, from_transfer_matrix
from wrmodes.modes import ModeBasis, Mode, eplane_frame_signs

INCH = 0.0254
WR187_TALL = Guide(0.872 * INCH, 1.872 * INCH)          # a = 22.149 mm along u, b = 47.549 mm; TE1,0 E along b
L20 = 20 * INCH
C = 8 / np.pi ** 2
TOL = 1e-10
LEGACY = dict(mode_factor=1.0, tail=False)
"""WP6's model: the port basis completed, plain truncation (T3/T3b tables, Doane Fig. 9 discrepancy numbers)."""
QUICK = dict(mode_factor=1.5)
"""A cheaper internal list for property tests of variable rates (the tail is on)."""


def assert_lossless(g, tol=TOL):
    assert g.reciprocity_defect() <= tol * np.abs(g.S).max(), g.reciprocity_defect()
    assert g.unitarity_defect() <= tol, g.unitarity_defect()


def spurious(g, mode="TE1,0"):
    i = g.basis1.index(mode)
    return 1 - abs(g.S21[i, i]) ** 2


# ======================================================================================== coupling coefficients
def test_coupling_matrix_closed_forms_of_t3b():
    """T3b section 3a, examples (exact sympy values, symbolic a and b) and the selection rule."""
    for a, b in ((0.03, 0.02), (0.872 * INCH, 1.872 * INCH), (0.02, 0.02), (0.1, 0.01)):
        basis = ModeBasis.below(3.2 * cst.c / (2 * min(a, b)), a, b)
        K = twist_coupling_matrix(basis)
        k = lambda p, q: K[basis.index(p), basis.index(q)]
        assert abs(k("TE1,0", "TE0,1") - C) < 1e-15 and abs(k("TE0,1", "TE1,0") + C) < 1e-15
        assert abs(k("TE1,0", "TE2,1") - (-8 * np.sqrt(2 * a * a + 8 * b * b) / (3 * np.pi ** 2 * a))) < 1e-14
        assert abs(k("TM2,1", "TE1,0") - (-32 * np.sqrt(2) * a * a / (9 * np.pi ** 2 * b * np.sqrt(a * a + 4 * b * b)))) < 1e-14
        assert k("TE1,0", "TM2,1") == 0.0
        ref = -128 * (4 * a ** 4 + 5 * a * a * b * b + b ** 4) / (27 * np.pi ** 2 * a * b * np.sqrt(4 * a ** 4 + 17 * a * a * b * b + 4 * b ** 4))
        assert abs(k("TM1,2", "TM2,1") - ref) < 1e-14 * abs(ref)
        assert abs(k("TM1,1", "TE2,2") + 64 / (9 * np.pi ** 2)) < 1e-14
        m = np.array([md.m for md in basis]); n = np.array([md.n for md in basis])
        allowed = ((m[:, None] + m[None, :]) % 2 == 1) & ((n[:, None] + n[None, :]) % 2 == 1)
        assert not np.any(K[~allowed])
        te = np.array([md.kind == "TE" for md in basis])
        assert not np.any(K[np.ix_(te, ~te)])                       # K_TE,TM = 0
    # T3 quotes for WR187 tall: C(TE21, TE10) = -1.6845 = K(TE10, TE21) and C(TE10, TE21) = 1.4534
    basis = ModeBasis.below(20e9, WR187_TALL.w, WR187_TALL.h)
    K = twist_coupling_matrix(basis)
    assert abs(K[basis.index("TE1,0"), basis.index("TE2,1")] + 1.6845) < 1e-4
    assert abs(K[basis.index("TE2,1"), basis.index("TE1,0")] - 1.4534) < 1e-4
    assert abs(K[basis.index("TE1,0"), basis.index("TE0,3")] - 0.2702) < 1e-4


def test_system_matrix_is_lossless_reciprocal_and_frame_covariant():
    f = 16.5e9
    basis = ModeBasis.propagating(f, WR187_TALL.w, WR187_TALL.h)
    d, X = twist_system_matrices(basis, f)
    N = len(basis)
    A, B = X[:N, :N], X[:N, N:]
    assert np.abs(A + A.T).max() < 1e-13 and np.abs(B - B.T).max() < 1e-13 and np.abs(np.imag(X)).max() == 0
    # the twist commutes with the H <-> E frame change (T3 box T3-5): K_E = F K_H F^T
    be = ModeBasis(basis.b, basis.a, [Mode(m.kind, m.n, m.m) for m in basis])
    F = eplane_frame_signs(basis, be)
    assert np.abs(twist_coupling_matrix(be) - F @ twist_coupling_matrix(basis) @ F.T).max() < 1e-14


# ======================================================================================== exact two-mode law
@pytest.mark.parametrize("profile", ["constant", "cosine", "triangular", "callable"])
@pytest.mark.parametrize("angle", [90.0, 154.7, -60.0])
def test_square_guide_two_mode_law(profile, angle):
    """T3 box T3-4 / T3b box 5: a 20 mm square guide at 10 GHz (only TE1,0 and TE0,1 propagate): for any profile,
    a_10 = cos(c theta) e^{-i beta L}, a_01 = -sin(c theta) e^{-i beta L}, no reflection."""
    g0 = Guide(0.02, 0.02)
    f, L = 10e9, 0.137
    prof = (lambda t: 1 + 0.7 * np.sin(3 * t) ** 2) if profile == "callable" else profile
    tw = Twist(angle, L, prof, guide=g0, atol=1e-13, **LEGACY)          # the law of the two-mode system
    g = tw.gsm(f)
    assert g.basis1.names == ["TE0,1", "TE1,0"]
    beta = g.basis1.beta(f)[0].real
    ct, st = np.cos(C * np.radians(angle)), np.sin(C * np.radians(angle))
    ref = np.exp(-1j * beta * L) * np.array([[ct, -st], [st, ct]])        # rows/cols TE0,1, TE1,0
    assert np.abs(g.S21 - ref).max() < 1e-12
    assert np.abs(g.S12 - ref.T).max() < 1e-12
    assert np.abs(g.S11).max() < 1e-12 and np.abs(g.S22).max() < 1e-12


# ======================================================================================== zero length, +-90
@pytest.mark.parametrize("guide,f", [(Guide(0.03, 0.03), 9e9), (WR187_TALL, 9e9), (WR187_TALL, 16.5e9)])
def test_zero_length_inverse_pair_is_identity(guide, f):
    """Z(-theta) Z(theta) = I at zero length (T3 box T3-5, T3b box 6), through transfer matrices and through GSMs."""
    basis = ModeBasis.propagating(f, guide.w, guide.h)
    Zp = zero_length_twist_transfer_matrix(basis, f, 90.0)
    Zm = zero_length_twist_transfer_matrix(basis, f, -90.0)
    I = np.eye(2 * len(basis))
    assert np.abs(Zm @ Zp - I).max() < 1e-12 and np.abs(Zp @ Zm - I).max() < 1e-12
    S = (from_transfer_matrix(Zp, basis, f) @ from_transfer_matrix(Zm, basis, f)).S
    n = len(basis)
    ident = np.block([[np.zeros((n, n)), np.eye(n)], [np.eye(n), np.zeros((n, n))]])     # S of nothing
    assert np.abs(S - ident).max() < 1e-12


def test_finite_inverse_pair_tends_to_identity_for_long_twists():
    """+90 deg then -90 deg constant twists (tall WR187, 9 GHz, 5 modes): the TE1,0 conversion vanishes as L grows
    (adiabatic limit). T3b box 6 note: two 20-lambda twists give |S21(TE10, TE10)|^2 = 0.980."""
    f = 9e9
    lam = cst.c / f
    out = []
    for n in (5, 20, 80):
        L = n * lam
        g = Twist(90, L, guide=WR187_TALL, **LEGACY).gsm(f) @ Twist(-90, L, guide=WR187_TALL, **LEGACY).gsm(f)
        assert_lossless(g)
        out.append(spurious(g))
    assert out[0] > out[1] > out[2] and out[2] < 2e-4
    assert abs(out[1] - 0.020) < 0.0005


# ======================================================================================== straight limit
@pytest.mark.parametrize("sigma", [None, 3.5e7])
@pytest.mark.parametrize("cf", [1.0, 1.5])
@pytest.mark.parametrize("profile", ["constant", "triangular"])
def test_zero_angle_is_a_straight(sigma, cf, profile):
    g0 = Guide(WR187_TALL.w, WR187_TALL.h, sigma=sigma)
    f, L = 15e9, 0.3
    basis = g0.basis(f, cf)
    g = Twist(0.0, L, profile, guide=g0, **QUICK).gsm(f, basis)
    s = Straight(L, guide=g0).gsm(f, basis)
    assert np.abs(g.S - s.S).max() < 1e-12


# ======================================================================================== GSM properties
@pytest.mark.parametrize("profile", ["constant", "cosine", "triangular"])
@pytest.mark.parametrize("f", [15e9, 18e9])
def test_doane_twists_are_reciprocal_and_lossless(profile, f):
    g = Twist(90, L20, profile, guide=WR187_TALL, **({} if profile == "constant" else QUICK)).gsm(f)
    assert_lossless(g)
    assert g.info["tail"] and g.info["route"] == "gsm"
    assert g.basis1 == g.basis2 == WR187_TALL.basis(f)


@pytest.mark.parametrize("profile", ["constant", "triangular"])
def test_evanescent_basis_gsm_is_symmetric_and_unitary(profile):
    """GSM built directly with evanescent modes of all families (CONVENTIONS.md section 6)."""
    f = 15e9
    tw = Twist(90, 0.2, profile, guide=WR187_TALL, cutoff_factor=1.3, **QUICK)
    g = tw.gsm(f, WR187_TALL.basis(f, 1.3, evanescent_families="all"))
    assert g.info["route"] == "gsm"
    assert_lossless(g)


def test_lossy_twist_is_passive_and_reciprocal():
    g0 = Guide(WR187_TALL.w, WR187_TALL.h, sigma=3.5e7)
    for profile in ("constant", "cosine"):
        g = Twist(90, L20, profile, guide=g0, **QUICK).gsm(16e9)
        assert g.reciprocity_defect() <= TOL * np.abs(g.S).max()
        m = g.passivity_margin()
        assert 0 < m < 1e-2
        # the TE1,0 attenuation is that of the straight guide to first order (the twist only redistributes power)
        s = Straight(L20, guide=g0).gsm(16e9)
        lossless = Twist(90, L20, profile, guide=WR187_TALL, **QUICK).gsm(16e9)
        j = g.basis1.index("TE1,0")
        p_loss = 1 - np.sum(np.abs(g.S21[:, j]) ** 2) - np.sum(np.abs(g.S11[:, j]) ** 2)
        p_straight = 1 - np.abs(s.S21[j, j]) ** 2
        assert abs(p_loss - p_straight) < 0.05 * p_straight
        assert abs(spurious(lossless) - (1 - abs(g.S21[j, j]) ** 2 - p_loss)) < 2e-4


def test_gsm_route_agrees_with_transfer_matrix_route():
    f = 18e9
    basis = ModeBasis.propagating(f, WR187_TALL.w, WR187_TALL.h)
    rate = TwistProfile("cosine", np.pi / 2, L20)
    S1, i1 = twist_gsm_blocks(basis, f, rate, L20, slices=512)
    S2, i2 = twist_gsm_blocks(basis, f, rate, L20, slices=512, loss=np.zeros((2 * len(basis),) * 2))
    assert i1["route"] == "transfer matrix" and i2["route"] == "gsm"
    assert max(np.abs(a - b).max() for a, b in zip(S1, S2)) < 1e-11


def test_turning_a_twist_around_reverses_the_profile_and_keeps_the_angle():
    """T3 box T3-5: P S_swapped P of a twist is the twist with profile theta'(L - z) and the same angle."""
    f = 16e9
    ramp, pmar = (lambda t: 0.2 + t), (lambda t: 1.2 - t)
    kw = dict(guide=WR187_TALL, atol=1e-10, mode_factor=1.3)            # with the tail (even in theta')
    g = Twist(90, 0.3, ramp, **kw).gsm(f).reversed()
    h = Twist(90, 0.3, pmar, **kw).gsm(f)
    assert np.abs(g.S - h.S).max() < 1e-9
    h_neg = Twist(-90, 0.3, pmar, **kw).gsm(f)
    assert np.abs(g.S - h_neg.S).max() > 1e-2


def test_named_profiles_and_callable_normalization():
    L = 0.4
    z = np.linspace(0, L, 1001)
    for kind in ("constant", "cosine", "triangular"):
        r = TwistProfile(kind, 1.3, L)
        assert abs(np.trapezoid(r(z), z) - 1.3) < 1e-5
    tri = TwistProfile("triangular", np.pi / 2, L)
    assert tri.breakpoints == (L / 2,) and abs(tri(L / 2) - 2 * (np.pi / 2) / L) < 1e-12      # peak = 2 theta / L
    cos = TwistProfile("cosine", np.pi / 2, L)
    assert abs(cos(L / 2) - np.pi * (np.pi / 2) / (2 * L)) < 1e-12 and abs(cos(0.0)) < 1e-15
    # a callable shape is normalized: r = 3 gives the constant profile, sin(pi t) the cosine one
    f = 17e9
    for shape, kind, bp in ((lambda t: 3 + 0 * t, "constant", None), (lambda t: np.sin(np.pi * t), "cosine", None),
                            (lambda t: np.minimum(t, 1 - t), "triangular", [0.5])):
        g1 = Twist(90, L, shape, guide=WR187_TALL, breakpoints=bp, atol=1e-10, mode_factor=1.3).gsm(f)
        g2 = Twist(90, L, kind, guide=WR187_TALL, atol=1e-10, mode_factor=1.3).gsm(f)
        assert np.abs(g1.S - g2.S).max() < 1e-8
    with pytest.raises(ValueError):
        TwistProfile("raised", 1.0, 1.0)
    with pytest.raises(ValueError):
        TwistProfile(lambda t: 0 * t, 1.0, 1.0)


def test_argument_and_frame_checks():
    with pytest.raises(ValueError):
        Twist(90, 0.0)
    tw = Twist(90, 0.2, guide=WR187_TALL)
    f = 15e9
    b = WR187_TALL.basis(f)
    with pytest.raises(ValueError):
        tw.gsm(f, b.with_frame("E"))
    with pytest.raises(ValueError):
        tw.gsm(f, b, WR187_TALL.basis(f, 1.2))
    assert Twist.plane == "H" and tw.gsm(f).basis2 == b
    with pytest.raises(ValueError):
        Twist(90, 0.2).default_basis(f)


# ======================================================================================== T3 / T3b tables
# Total spurious power 1 - |S21(TE10, TE10)|^2 and the three largest converted modes, 20-inch 90 deg twists in tall
# WR187, all propagating modes (WP6's model: LEGACY, plain truncation). Computed with T3's prototype (docs/theory/T3/twist_formulas.py, transfer_matrix,
# 2400 Magnus slices); T3 section 6 prints 1.387 / 0.697 / 2.140, 0.352 / 0.867 / 0.943, 0.114 / 0.247 / 0.603 %.
T3_TABLE = {
    ("constant", 15.0): (1.386475e-02, {"TE0,1": 1.086379e-02, "TE0,3": 1.165684e-03, "TE2,1": 8.999142e-04}),
    ("constant", 16.5): (6.974107e-03, {"TE2,1": 6.088169e-03, "TE0,1": 3.204770e-04, "TE1,2": 2.915546e-04}),
    ("constant", 18.0): (2.139478e-02, {"TE0,1": 1.820326e-02, "TE0,3": 1.826375e-03, "TE2,1": 7.554088e-04}),
    ("cosine", 15.0): (3.518769e-03, {"TE0,1": 3.451069e-03, "TE0,3": 4.186329e-05, "TE2,1": 1.289316e-05}),
    ("cosine", 16.5): (8.672249e-03, {"TE0,1": 8.483257e-03, "TE0,3": 1.336769e-04, "TE1,2": 4.296561e-05}),
    ("cosine", 18.0): (9.431512e-03, {"TE0,1": 9.054085e-03, "TE1,2": 2.291124e-04, "TE2,1": 5.276318e-05}),
    ("triangular", 15.0): (1.142100e-03, {"TE1,2": 7.974294e-04, "TE0,3": 1.882948e-04, "TM1,2": 1.059243e-04}),
    ("triangular", 16.5): (2.474181e-03, {"TE1,2": 1.686919e-03, "TE0,3": 4.304407e-04, "TM1,2": 2.295123e-04}),
    ("triangular", 18.0): (6.029584e-03, {"TE1,2": 3.835751e-03, "TE0,3": 1.575648e-03, "TM1,2": 4.639081e-04}),
}
T3_PRINTED = {("constant", 15.0): 1.387, ("constant", 16.5): 0.697, ("constant", 18.0): 2.140,
              ("cosine", 15.0): 0.352, ("cosine", 16.5): 0.867, ("cosine", 18.0): 0.943,
              ("triangular", 15.0): 0.114, ("triangular", 16.5): 0.247, ("triangular", 18.0): 0.603}


@pytest.mark.parametrize("key", sorted(T3_TABLE))
def test_t3_full_basis_tables(key):
    profile, fghz = key
    g = Twist(90, L20, profile, guide=WR187_TALL, **LEGACY).gsm(fghz * 1e9)
    total, top = T3_TABLE[key]
    assert abs(spurious(g) / total - 1) < 1e-3
    assert abs(100 * spurious(g) - T3_PRINTED[key]) <= 0.001             # one unit of T3's last printed digit
    j = g.basis1.index("TE1,0")
    for name, p in top.items():
        assert abs(abs(g.S21[g.basis1.index(name), j]) ** 2 / p - 1) < 1e-3, name


def test_t3b_square_guide_full_basis():
    """T3b section 6 table (a = b = 30 mm, 90 deg twists, all propagating modes): local polarization angle."""
    a = 0.03
    g0 = Guide(a, a)
    rows = [(2.5, 5, "constant", -71.138), (2.5, 20, "constant", -72.849), (2.5, 80, "triangular", -72.940),
            (4.5, 20, "triangular", -67.285), (2.5, 80, "constant", -72.945), (4.5, 80, "constant", -72.836)]
    for two_a_over_lam, L_over_lam, prof, psi_ref in rows:
        lam = 2 * a / two_a_over_lam
        f = cst.c / lam
        g = Twist(90, L_over_lam * lam, prof, guide=g0, **LEGACY).gsm(f)
        j = g.basis1.index("TE1,0")
        # field at the centre: e_10 along +v, e_01 along -u (T3b box 5); angle of the major axis of the polarization
        # ellipse, from +v towards -u (T3b's "local angle")
        E = np.array([-g.S21[g.basis1.index("TE0,1"), j], g.S21[j, j]])
        v = (E * np.exp(-0.5j * np.angle(E @ E))).real
        psi = np.degrees(np.arctan2(-v[0], v[1]))
        psi = (psi + 90) % 180 - 90                                       # angles modulo 180 deg
        assert abs(psi - psi_ref) < 2e-3, (two_a_over_lam, L_over_lam, prof, psi)


# ======================================================================================== Doane Fig. 9 (literature)
def _fig9():
    path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "literature", "doane1989_fig9.csv")
    import csv
    with open(path) as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    return {"curve": np.array([r["curve"] for r in rows]), "x": np.array([float(r["x"]) for r in rows]),
            "y": np.array([float(r["y"]) for r in rows]), "uncertainty": np.array([float(r["uncertainty"]) for r in rows])}


def _two_mode_spurious(f, profile):
    basis = ModeBasis(WR187_TALL.w, WR187_TALL.h, ["TE1,0", "TE0,1"])
    S, _ = twist_gsm_blocks(basis, f, TwistProfile(profile, np.pi / 2, L20), L20)
    return 1 - abs(S[2][basis.index("TE1,0"), basis.index("TE1,0")]) ** 2


def test_doane_fig9_cosine_curve():
    """Cosine curve of Fig. 9. The full multimode solution (cutoff_factor 1) is within the digitizing uncertainty from 15.1
    to 16.35 GHz and falls below the curve above (by 0.10 % at 18 GHz, 8.5 sigma); the two-mode solution (TE1,0 + TE0,1)
    follows the whole curve within 3.5 sigma (rms 0.022 %). Not met over the whole band: see
    docs/discrepancies/2026-11-12-doane-fig9-twists.md and docs/reports/WP6.md."""
    d = _fig9()
    sel = d["curve"] == "cosine"
    x, y, u = d["x"][sel], d["y"][sel], d["uncertainty"][sel]
    tw = Twist(90, L20, "cosine", guide=WR187_TALL, **LEGACY)          # WF2 moves these by < 2e-5 of the power
    for fx in (15.25, 15.5, 16.0, 16.25):                             # 15.1-16.35 GHz within 1 sigma (15.025: 1.1)
        i = np.argmin(abs(x - fx))
        p = 100 * spurious(tw.gsm(x[i] * 1e9))
        assert abs(p - y[i]) <= u[i], (x[i], p, y[i], u[i])
    i = np.argmin(abs(x - 17.975))
    assert y[i] - 100 * spurious(tw.gsm(x[i] * 1e9)) > 0.08          # the documented discrepancy
    z = [(100 * _two_mode_spurious(xx * 1e9, "cosine") - yy) / uu for xx, yy, uu in zip(x[::4], y[::4], u[::4])]
    assert np.max(np.abs(z)) < 3.5


def test_doane_fig9_constant_curve():
    """Constant curve of Fig. 9: it follows our TE1,0 -> TE0,1 conversion (rms difference 0.11 % over the 119 grid
    points), but not our total spurious power, which has lobes of TE2,1 (up to 0.71 %, 0.61 % at 16.5 GHz) that Fig. 9
    does not show. Not met: see docs/discrepancies/2026-11-12-doane-fig9-twists.md."""
    d = _fig9()
    sel = d["curve"] == "constant"
    x, y = d["x"][sel], d["y"][sel]
    tw = Twist(90, L20, "constant", guide=WR187_TALL, **LEGACY)        # WF2: TE2,1 +0.04 % at 15 / 18 GHz
    te01, tot = [], []
    for xx in x:
        g = tw.gsm(xx * 1e9)
        j = g.basis1.index("TE1,0")
        te01.append(100 * abs(g.S21[g.basis1.index("TE0,1"), j]) ** 2)
        tot.append(100 * spurious(g))
    assert np.sqrt(np.mean((np.array(te01) - y) ** 2)) < 0.12
    i = np.argmin(abs(x - 16.5))
    assert tot[i] - y[i] > 0.6                                       # the documented TE2,1 discrepancy


def test_doane_fig9_triangular_curve_open_issue():
    """Triangular curve of Fig. 9: ours is 2.4-4.0 times higher at every frequency (T3 / T3b open issue), almost
    entirely TE1,2, TE0,3 and TM1,2. To be settled by COMSOL campaign 2, case E4."""
    d = _fig9()
    sel = d["curve"] == "triangular"
    x, y = d["x"][sel], d["y"][sel]
    tw = Twist(90, L20, "triangular", guide=WR187_TALL, **LEGACY)      # WF2 moves these by < 2.3e-5
    for fx, lo, hi in ((15.025, 2.2, 2.6), (17.975, 3.8, 4.2)):
        i = np.argmin(abs(x - fx))
        assert lo < 100 * spurious(tw.gsm(x[i] * 1e9)) / y[i] < hi


# ======================================================================================== near cutoff (16.5 GHz)
def test_near_cutoff_sensitivity_at_16_5_ghz():
    """TE2,3/TM2,3 (cutoff 16.512 GHz) are 0.07 % below cutoff at 16.5 GHz. With cutoff_factor 1 they are left out
    and the triangular twist's spurious power jumps by about 5 % at their cutoff; with them kept (cutoff_factor >=
    1.001) the result is continuous through the cutoff. The constant twist's TE2,1 (0.61 %) is insensitive."""
    f = 16.5e9
    port = WR187_TALL.basis(f)
    tri = Twist(90, L20, "triangular", guide=WR187_TALL, **LEGACY)
    p1 = spurious(tri.gsm(f, port))
    p2 = spurious(tri.replace(cutoff_factor=1.001).gsm(f, port))
    assert abs(p1 - 2.4742e-3) < 2e-7 and abs(p2 - 2.6026e-3) < 2e-6
    con = Twist(90, L20, "constant", guide=WR187_TALL, **LEGACY)
    for cf in (1.0, 1.001, 1.5):
        g = con.replace(cutoff_factor=cf).gsm(f, port)
        p = abs(g.S21[port.index("TE2,1"), port.index("TE1,0")]) ** 2
        assert abs(p - 6.07e-3) < 0.06e-3


# ======================================================================================== Galerkin / helical modes
# Exact helical modes of a uniformly twisted guide (T3b check_helical_eigen.out, spectral Nedelec solver converged to
# 3e-10): a = 2, b = 1, tau = 0.5, gamma = 4; the k at which the twisted guide has gamma = 4.
HELICAL_K = np.array([4.1528675745, 4.7062051643, 5.1629896646, 5.2696835618, 5.5431540615, 5.6122772221])


def _gamma_cm(chi_factor):
    a, b, tau, gamma = 2.0, 1.0, 0.5, 4.0
    basis = ModeBasis.below(chi_factor * HELICAL_K.max() * cst.c / (2 * np.pi), a, b)
    out = []
    for k in HELICAL_K:
        g = helical_propagation_constants(basis, k * cst.c / (2 * np.pi), tau)
        out.append(g[np.argmin(abs(g - gamma))].real)
    return np.array(out) / gamma - 1, len(basis)


def test_helical_modes_against_spectral_solution():
    """T3b section 5: the coupled-mode gamma converges to the exact one like 1/chi_max; at chi_max = 3 k_max (89
    modes) the largest error is 5.37e-3 (T3b), reproduced here. The spectral solution itself is converged to 3e-10,
    far below the modal truncation error; the comparison is limited by the latter."""
    d3, n3 = _gamma_cm(3.0)
    assert n3 == 89 and abs(np.max(np.abs(d3)) - 5.37e-3) < 0.005e-3


@pytest.mark.slow
def test_helical_modes_converge_like_one_over_chi():
    d3, _ = _gamma_cm(3.0)
    d6, n6 = _gamma_cm(6.0)
    assert n6 == 359 and abs(np.max(np.abs(d6)) - 2.93e-3) < 0.005e-3
    rich = 2 * d6 - d3                                                   # Richardson, error ~ 1/chi_max
    assert np.max(np.abs(rich)) < 7.5e-4


# ======================================================================================== Line
def test_line_twist_bend_twist():
    """[Twist(+90), SmoothBend(H), Twist(-90)]: an H-plane bend whose bend plane is, in space, the input's E plane
    (the y-z plane), returned to the input orientation. Runs with no new Line rule and conserves power."""
    f = 15e9
    bend = SmoothBend(0.3, SechCurvature(90, 0.3), plane="H")
    line = wr.Line(WR187_TALL, [Twist(90, 0.4, "cosine", **QUICK), bend, Twist(-90, 0.4, "cosine", **QUICK)])
    assert not any(line.inserted)
    g = line.gsm(f)
    assert_lossless(g)
    # the line is the cascade of the parts in the line's (global) basis
    b = WR187_TALL.basis(f)
    ref = Twist(90, 0.4, "cosine", guide=WR187_TALL, **QUICK).gsm(f, b) @ bend.with_guide(WR187_TALL).gsm(f, b) @ \
        Twist(-90, 0.4, "cosine", guide=WR187_TALL, **QUICK).gsm(f, b)
    assert np.abs(g.S - ref.S).max() < 1e-13
    # with an E-plane component the line inserts frame changes around it, not around the twist
    line2 = wr.Line(WR187_TALL, [Twist(90, 0.4), SmoothBend(0.3, SechCurvature(90, 0.3), plane="E"), Twist(-90, 0.4)])
    assert sum(line2.inserted) == 2
    assert_lossless(line2.gsm(f))


def test_line_with_evanescent_modes_and_loss():
    f = 15e9
    lossy = Guide(WR187_TALL.w, WR187_TALL.h, sigma=3.5e7)
    line = wr.Line(lossy, [Straight(0.05), Twist(45, 0.2, "triangular", **QUICK), Straight(0.05)])
    g = line.gsm(f, cutoff_factor=1.2)
    assert g.reciprocity_defect() <= TOL * np.abs(g.S).max() and g.passivity_margin() > 0


# ======================================================================================== WF2: tail elimination
def test_tail_matrix_properties():
    """twist_tail_matrix: real symmetric, block diagonal in the parity classes (m mod 2, n mod 2), independent of
    the order of the basis, converged in the outer radius after the 1/R extrapolation; refuses incomplete lists."""
    f = 15e9
    a, b = WR187_TALL.w, WR187_TALL.h
    basis = ModeBasis.below(2.0 * f, a, b)
    Q = twist_tail_matrix(basis, f)
    assert Q.dtype == float and np.array_equal(Q, Q.T) and np.abs(Q).max() > 0
    m = np.array([md.m for md in basis]) % 2
    n = np.array([md.n for md in basis]) % 2
    other = (m[:, None] != m[None, :]) | (n[:, None] != n[None, :])
    assert not np.any(Q[other])
    perm = ModeBasis(a, b, list(reversed(basis.modes)), sort=False)
    Qp = twist_tail_matrix(perm, f)
    idx = [perm.index(md) for md in basis.modes]
    assert np.abs(Qp[np.ix_(idx, idx)] - Q).max() < 1e-12 * np.abs(Q).max()
    Q2 = twist_tail_matrix(basis, f, radii=(12.0, 24.0))
    assert np.abs(Q2 - Q).max() < 1e-3 * np.abs(Q).max()
    assert np.abs(twist_tail_matrix(basis, f, dynamic=False) - Q).max() > 1e-3 * np.abs(Q).max()
    with pytest.raises(ValueError):
        twist_tail_matrix(ModeBasis(a, b, [md for md in basis.modes if md.name != "TE0,1"]), f)


def test_tail_term_is_lossless_and_reciprocal():
    """X2 = (i/2)[[W, W], [-W, -W]] with W real symmetric on propagating modes: X2^H J + J X2 = 0 (J = diag(I, -I),
    power) and X2^T K + K X2 = 0 (K = [[0, I], [-I, 0]], reciprocity), the two conditions D and X of T3b box 4 meet."""
    f = 16e9
    basis = WR187_TALL.basis(f)
    X2 = twist_tail_system_matrix(basis, f, twist_tail_matrix(basis, f))
    N = len(basis)
    J = np.diag(np.r_[np.ones(N), -np.ones(N)])
    assert np.abs(X2.conj().T @ J + J @ X2).max() < 1e-12 * np.abs(X2).max()
    K = np.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])
    assert np.abs(X2.T @ K + K @ X2).max() < 1e-12 * np.abs(X2).max()
    d, X = twist_system_matrices(basis, f)
    assert np.abs(X.T @ K + K @ X).max() < 1e-12 * np.abs(X).max()


def test_tail_constant_rate_expm_equals_sliced_route():
    """A constant rate given as a callable shape goes through the Magnus slices with the theta'^2 X2 term; it must
    agree with the exact expm route (and both carry the tail)."""
    f, L = 16e9, 0.3
    kw = dict(guide=WR187_TALL, mode_factor=1.3, atol=1e-10)
    g1 = Twist(90, L, "constant", **kw).gsm(f)
    g2 = Twist(90, L, lambda t: 1 + 0 * t, **kw).gsm(f)
    assert g1.info["n_slices"] != g2.info["n_slices"]
    assert np.abs(g1.S - g2.S).max() < 1e-8


def test_default_mode_factors():
    assert Twist(90, 0.2).mode_factor == DEFAULT_MODE_FACTOR["constant"] == 3.0
    assert Twist(90, 0.2, "cosine").mode_factor == DEFAULT_MODE_FACTOR["variable"] == 2.0
    assert Twist(90, 0.2, mode_factor=1.7).replace(angle_deg=45).mode_factor == 1.7
    with pytest.raises(ValueError):
        Twist(90, 0.2, mode_factor=0.5)


def _helical_reference(name, f):
    """WP7b's stored GSM (main setting) of case `name` at the frequency closest to f: (data, key, names, geometry,
    exact frequency)."""
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "validation_v2", "data", f"twist_{name}.npz")
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(str(d["meta"]))
    fe = [r["f"] for r in meta["rows"] if r["tag"] == "main" and abs(r["f"] - f) < 1e7][0]
    key = f"{name}_f{fe / 1e9:g}_main"
    return d, key, [str(s) for s in d[f"{key}_names"]], meta["geometry"], fe


def _worst_dP(g, d, key, names):
    """Worst | |S|^2 difference | over the modes converted from TE1,0 (transmitted and reflected), and the TE1,0
    transmission difference in dB."""
    i_r, i_t = names.index("TE1,0"), g.basis1.index("TE1,0")
    out = 0.0
    for blk in ("S21", "S11"):
        for j, n in enumerate(names):
            if blk == "S21" and n == "TE1,0":
                continue
            out = max(out, abs(abs(getattr(g, blk)[g.basis1.index(n), i_t]) ** 2 - abs(d[f"{key}_{blk}"][j, i_r]) ** 2))
    main = 20 * np.log10(abs(g.S21[i_t, i_t]) / abs(d[f"{key}_S21"][i_r, i_r]))
    return out, main


@pytest.mark.parametrize("name,f,tol", [("doane_constant", 15e9, 1.5e-5), ("doane_constant", 18e9, 1.5e-5),
                                        ("square_const_5lam", 12.49e9, 8e-5), ("square_const_20lam", 12.49e9, 1e-5),
                                        ("strong_tau0p1", 15e9, 2e-6), ("strong_tau0p1", 30e9, 3e-6)])
def test_default_against_helical_fem_reference(name, f, tol):
    """WF2 item 1: at the default settings the constant-rate twists are within the CONVENTIONS section 8 tolerance
    (1e-4 of the input power per mode, 0.01 dB) of WP7b's helical-coordinate FEM (benchmarks/validation_v2), with
    the margins measured here (tol); WP6's default was 4e-4 (Doane), 4.6e-3 (square, 5 lambda), 2.4e-4 (tau a = 0.1,
    30 GHz) off."""
    d, key, names, c, fe = _helical_reference(name, f)
    g = Twist(90, c["L"], "constant", guide=Guide(c["a"], c["b"])).gsm(fe)
    dP, ddb = _worst_dP(g, d, key, names)
    assert dP < tol and abs(ddb) < 0.01, (dP, ddb)


@pytest.mark.slow
@pytest.mark.parametrize("profile", ["cosine", "triangular"])
def test_default_variable_rates_against_helical_fem_reference_18GHz(profile):
    """Cosine and triangular rates at 18 GHz (the worst frequency): 1.2e-5 and 2.3e-5 at the default (WP6's: 1.6e-4
    and 7.4e-5; the triangular twist's unexplained 2 % in TE1,2 was the truncation)."""
    d, key, names, c, _ = _helical_reference(f"doane_{profile}", 18e9)
    g = Twist(90, c["L"], profile, guide=Guide(c["a"], c["b"])).gsm(18e9)
    dP, ddb = _worst_dP(g, d, key, names)
    assert dP < 4e-5 and abs(ddb) < 0.01, dP


@pytest.mark.slow
def test_strong_twist_option_reaches_the_tolerance():
    """tau a = 0.5 (WR-90, 90 deg in 71.8 mm, 30 GHz): 1.9e-4 at the default mode_factor 3, 2e-5 with 6."""
    d, key, names, c, _ = _helical_reference("strong_tau0p5", 30e9)
    tw = Twist(90, c["L"], "constant", guide=Guide(c["a"], c["b"]))
    dP3, _ = _worst_dP(tw.gsm(30e9), d, key, names)
    dP6, ddb = _worst_dP(tw.replace(mode_factor=6.0).gsm(30e9), d, key, names)
    assert 1e-4 < dP3 < 3e-4 and dP6 < 4e-5 and abs(ddb) < 0.01


# ======================================================================================== convergence, timing
def test_convergence_study_runs():
    """Rows per mode factor with and without the tail, and the Richardson extrapolation of the plain series in
    1/chi_max (WF2); the LEGACY row is T3's table value."""
    tw = Twist(90, L20, "constant", guide=WR187_TALL)
    res = tw.convergence_study(15e9, mode_factors=(1.0, 3.0, 4.0))
    rows = res["modes"]
    assert [(r["mode_factor"], r["tail"]) for r in rows] == [(1.0, True), (3.0, True), (4.0, True), (1.0, False),
                                                             (3.0, False), (4.0, False)]
    assert rows[1]["n_modes"] > rows[0]["n_modes"]
    assert abs(rows[3]["powers"]["spurious"] - T3_TABLE[("constant", 15.0)][0]) < 1e-4
    ref = 1.31251e-3                                                  # WP7b helical FEM, TE2,1
    assert abs(res["best"]["TE2,1"] - ref) < 2e-6
    assert abs(res["extrapolated"]["TE2,1"] - ref) < 2e-5 and res["change"] < 3e-5
    assert abs(rows[5]["powers"]["TE2,1"] - ref) > 1e-4               # the plain series at 4 f is still far off


@pytest.mark.slow
def test_convergence_study_variable_rate():
    tw = Twist(90, L20, "triangular", guide=WR187_TALL)
    res = tw.convergence_study(15e9, mode_factors=(1.5, 2.0), tails=(True,), slice_factors=(1, 2))
    p = [r[1]["spurious"] for r in res["slices"]]
    assert abs(p[1] - p[0]) < 1e-6
    q = [r["powers"]["spurious"] for r in res["modes"]]
    assert abs(q[1] / q[0] - 1) < 0.05 and res["change"] < 2e-5


@pytest.mark.slow
def test_timing_doane_twist():
    """One 20-inch triangular twist at 18 GHz. WP6's model (23 modes): about 0.6 s on a laptop; WF2's default
    (internal list to 2 f, 96 modes, tail): about 32-45 s on the 2-core container, 4.4 s after WF4. Strict limits
    (2 s, 15 s) only with WRMODES_PERF=1; otherwise 10x and 4x guards (CI runners are 2-3x slower)."""
    perf = bool(os.environ.get("WRMODES_PERF"))
    t = time.perf_counter()
    Twist(90, L20, "triangular", guide=WR187_TALL, **LEGACY).gsm(18e9)
    assert time.perf_counter() - t < (2.0 if perf else 20.0)
    t = time.perf_counter()
    Twist(90, L20, "constant", guide=WR187_TALL).gsm(18e9)
    Twist(90, L20, "triangular", guide=WR187_TALL).gsm(18e9)
    assert time.perf_counter() - t < (15.0 if perf else 60.0)


# ======================================================================================== WF4: run time
def _wf4_baseline():
    path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "twist", "data", "wf4_twist_baseline.npz")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _max_power_change(g, ref, key):
    names = [str(s) for s in ref[f"{key}_names"]]
    assert names == [m.name for m in g.basis1.modes]
    return max(float(np.max(np.abs(np.abs(getattr(g, b)) ** 2 - np.abs(ref[f"{key}_{b}"]) ** 2))) for b in ("S11", "S21"))


def test_coupled_groups_are_the_parity_classes():
    """WF4 item 1: the system splits into {(e, e), (o, o)} and {(e, o), (o, e)} of (m mod 2, n mod 2), with the
    tail and the wall loss; the group-wise solution equals the solution of the whole system."""
    from wrmodes.components.twist import coupled_groups, _sliced_solution
    from wrmodes.transfer_matrix import _slice_edges
    f = 16.5e9
    tw = Twist(90, L20, "triangular", guide=Guide(WR187_TALL.w, WR187_TALL.h, sigma=5.8e7), mode_factor=1.5)
    sb = tw.solve_basis(f, tw.default_basis(f)[0])
    K, Q = twist_coupling_matrix(sb), twist_tail_matrix(sb, f)
    groups = coupled_groups(K, [Q])
    cls = [{((m.m % 2) + (m.n % 2)) % 2 for m in (sb.modes[i] for i in g)} for g in groups]
    assert len(groups) == 2 and sorted(len(c) for c in cls) == [1, 1] and cls[0] != cls[1]
    from wrmodes.loss import loss_matrix_full
    Lf = loss_matrix_full(sb, f, tw.guide.wall_impedance(f))
    N = len(sb)
    assert len(coupled_groups(K, [Q, Lf[:N, :N], Lf[:N, N:], Lf[N:, :N], Lf[N:, N:]])) == 2
    S, info = twist_gsm_blocks(sb, f, tw.rate, L20, loss=Lf, slices=64, tail=Q)
    assert info["n_groups"] == 2
    d, X = twist_system_matrices(sb, f)
    X2 = twist_tail_system_matrix(sb, f, Q)
    edges = _slice_edges(L20, [32, 32], [0.0, L20 / 2, L20], 0)
    beta = sb.beta(f)
    hmax = min(4.0 / np.max(np.abs(beta[~sb.propagating_mask(f)].imag)), 0.5 / np.linalg.norm(Lf, 1))
    full = _sliced_solution(np.diag(d) - Lf, X, tw.rate, edges, True, N, X2, hmax)
    nogroup = _sliced_solution(np.diag(d) - Lf, X, tw.rate, edges, True, N, X2, 1e-300)   # no grouping
    for x, y, z in zip(S, full, nogroup):
        assert np.max(np.abs(x - y)) < 1e-12 and np.max(np.abs(y - z)) < 1e-12


@pytest.mark.parametrize("profile", ["cosine", "triangular"])
def test_wf4_defaults_agree_with_wf2_defaults_15GHz(profile):
    """WF4 item 1: the faster solver (groups, grouped slice products, earlier acceptance of the adaptive slicing)
    changes no power |S_pq|^2 of the propagating GSM by more than 2e-6 against the WF2 defaults (stored in
    benchmarks/twist/data/wf4_twist_baseline.npz by benchmarks/twist/wf4_speed.py; measured <= 8.2e-10)."""
    ref = _wf4_baseline()
    port = WR187_TALL.basis(15e9, 1.0)
    g = Twist(90, L20, profile, guide=WR187_TALL).gsm(15e9, port)
    assert _max_power_change(g, ref, f"{profile}_f15") < 2e-6
    assert_lossless(g)


@pytest.mark.slow
@pytest.mark.parametrize("profile", ["constant", "cosine", "triangular"])
@pytest.mark.parametrize("f", [16.5e9, 18e9])
def test_wf4_defaults_agree_with_wf2_defaults(profile, f):
    ref = _wf4_baseline()
    port = WR187_TALL.basis(f, 1.0)
    g = Twist(90, L20, profile, guide=WR187_TALL).gsm(f, port)
    assert _max_power_change(g, ref, f"{profile}_f{f / 1e9:g}") < 2e-6


# ======================================================================================== WF4 item 7
def test_doane_fig9_with_doanes_five_modes():
    """WF4 item 7: restricted to Doane's five modes (his TE10, TE01, TE30, TE21, TM21 = wrmodes' TE1,0, TE0,1,
    TE0,3, TE1,2, TM1,2), the plain truncated system reproduces Fig. 9's triangular curve within the digitizing
    uncertainty (stored: all 119 points within 1 sigma, rms z 0.52); adding TE2,1 (Doane's TE12) moves it to the full
    solution (0.55 % at 18 GHz against 0.15 %)."""
    import csv
    path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "literature", "doane1989_fig9.csv")
    with open(path) as fh:
        rows = [r for r in csv.DictReader(l for l in fh if not l.startswith("#")) if r["curve"] == "triangular"]
    five = ["TE1,0", "TE0,1", "TE0,3", "TE1,2", "TM1,2"]
    for r in rows[::24]:
        f = float(r["x"]) * 1e9
        basis = ModeBasis(WR187_TALL.w, WR187_TALL.h, five)
        S, _ = twist_gsm_blocks(basis, f, TwistProfile("triangular", np.pi / 2, L20), L20)
        i = basis.index("TE1,0")
        assert abs(100 * (1 - abs(S[2][i, i]) ** 2) - float(r["y"])) <= float(r["uncertainty"]), r
    basis = ModeBasis(WR187_TALL.w, WR187_TALL.h, five + ["TE2,1"])
    S, _ = twist_gsm_blocks(basis, 18e9, TwistProfile("triangular", np.pi / 2, L20), L20)
    i = basis.index("TE1,0")
    assert 0.53 < 100 * (1 - abs(S[2][i, i]) ** 2) < 0.57
    path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "twist", "data", "doane_fig9_five_modes.npz")
    import json
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        assert np.all(np.abs(z["triangular_five_z"]) <= 1.0 + 1e-9) and len(z["triangular_five_z"]) == 119
    s = meta["parameters"]["summary"]
    assert s["constant_five"]["rms_z"] < 1.3 and s["cosine_five"]["rms_z"] < 1.2 and s["triangular_six"]["rms_z"] > 10
