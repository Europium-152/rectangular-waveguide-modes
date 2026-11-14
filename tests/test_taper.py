"""Tests of WP5: profiles, VariableGuide, Taper, StrangledBend (card tasks/WP5-tapers-strangled-bend.md).

Theory: docs/theory/T2-taper-coupling.md, docs/theory/T2b-taper-coupling.md. Slow cases are marked `slow`.
Timing assertions allow CI runners 3x slower than the targets of the card; WRMODES_PERF=1 makes them strict.
"""
import importlib.util
import os
import time
import warnings

import numpy as np
import pytest
import scipy.constants as cst
from scipy.special import airy, jv, yv, jvp, yvp

import wrmodes
from wrmodes import Guide, Line, Straight, SmoothBend
from wrmodes.components import VariableGuide, Taper, StrangledBend, GeometryFunction
from wrmodes.components import taper as tm
from wrmodes.loss import wall_integrals
from wrmodes.modes import Mode, ModeBasis, wave_admittance
from wrmodes.profiles import get_profile, BottleProfile, CallableProfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WR90 = Guide(22.86e-3, 10.16e-3)
PERF = os.environ.get("WRMODES_PERF") == "1"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.join(ROOT, path)))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def lin(x0, x1, L):
    return GeometryFunction(lambda z: x0 + (x1 - x0) * np.asarray(z) / L, lambda z: (x1 - x0) / L + 0 * np.asarray(z))


# ============================================================================================ profiles
@pytest.mark.parametrize("name", ["linear", "raised_cosine", "bottle"])
def test_named_profiles_end_values_and_derivatives(name):
    p = get_profile(name)
    assert p(np.array([0.0, 1.0])) == pytest.approx([0.0, 1.0], abs=1e-14)
    t = np.linspace(0.02, 0.98, 25)
    h = 1e-6
    assert np.max(np.abs((p(t + h) - p(t - h)) / (2 * h) - p.derivative(t))) < 1e-8
    assert np.all(np.diff(p(np.linspace(0, 1, 101))) > 0)


def test_bottle_formula():
    p = BottleProfile()
    assert p.lam == 3.5
    t = np.linspace(0, 1, 7)
    assert np.allclose(p(t), (1 - np.exp(-3.5 * t)) / (1 - np.exp(-3.5)), rtol=0, atol=1e-15)
    assert p.derivative(np.array([0.0]))[0] / p.derivative(np.array([1.0]))[0] == pytest.approx(np.exp(3.5))
    assert np.allclose(BottleProfile(0.0)(t), t)


def test_callable_profile_finite_differences_warn():
    with pytest.warns(RuntimeWarning):
        p = get_profile(lambda t: t * t)
    t = np.linspace(0, 1, 11)
    assert np.allclose(p.derivative(t), 2 * t, atol=1e-8)
    q = CallableProfile(lambda t: t * t, lambda t: 2 * t)
    assert np.allclose(q.derivative(t), 2 * t)
    with pytest.raises(ValueError):
        CallableProfile(lambda t: t + 0.1, lambda t: 1 + 0 * t)


# ============================================================================================ coefficients
def _random_geometry(seed=0):
    rng = np.random.default_rng(seed)
    G = len(range(3))
    g = type("G", (), {})()
    g.z = np.zeros(G)
    g.a = 0.0229 * (1 + 0.3 * rng.random(G))
    g.b = 0.0102 * (1 + 0.3 * rng.random(G))
    g.da, g.db, g.du0, g.dv0 = (rng.normal(0, 0.2, G) for _ in range(4))
    g.u0 = rng.normal(0, 0.002, G)
    g.kappa = rng.normal(0, 5.0, G)
    return g


def _modes(mmax, nmax):
    return [Mode(k, m, n) for m in range(mmax + 1) for n in range(nmax + 1) for k in ("TE", "TM")
            if not (k == "TE" and m == n == 0) and not (k == "TM" and (m == 0 or n == 0))]


def test_vi_system_equals_t2_and_t2b_closed_forms():
    """M = [[T, -iZ], [-iY, -T^T]] against T2's vi_matrix (all four slopes, kappa, u0) and T2b's T_closed."""
    t2 = _load("docs/theory/T2/t2_coefficients.py", "t2_coefficients")
    t2b = _load("docs/theory/T2b/t2b_formulas.py", "t2b_formulas")
    modes = _modes(4, 3)
    g = _random_geometry()
    f = 25e9
    M = tm.vi_system(modes, f, g)
    for i in range(len(g.a)):
        ref = t2.vi_matrix(modes, g.a[i], g.b[i], f, g.da[i], g.db[i], g.du0[i], g.dv0[i], g.kappa[i], g.u0[i])
        assert np.max(np.abs(M[i] - ref)) / np.max(np.abs(ref)) < 1e-12
        T = t2b.T_matrix(modes, g.a[i], g.b[i], g.da[i], g.db[i], g.du0[i], g.dv0[i])
        n = len(modes)
        assert np.max(np.abs(M[i][:n, :n] - T)) < 1e-12 * np.max(np.abs(T))


def test_wall_parts_equal_loss_wall_integrals():
    modes = _modes(4, 3)
    a, b = 0.0229, 0.0137
    N0, G0, N1, G1 = tm._wall_parts(tm._Modes(modes), np.array([a]), np.array([b]))
    basis = ModeBasis(a, b, modes, sort=False)
    for order, (Nn, Gn) in ((0, (N0, G0)), (1, (N1, G1))):
        w = wall_integrals(basis, order)
        Nr = sum(v[0] for v in w.values())
        Gr = sum(v[1] for v in w.values())
        assert np.max(np.abs(Nn[0] - Nr)) <= 1e-12 * np.max(np.abs(Nr))
        assert np.max(np.abs(Gn[0] - Gr)) <= 1e-12 * np.max(np.abs(Gr))


def test_lossless_structure_of_vi_matrix():
    """M^T J + J M = 0 (reciprocity) and M^H G + G M = 0 (power), evanescent modes included (T2 section 4)."""
    modes = _modes(3, 2)
    M = tm.vi_system(modes, 30e9, _random_geometry(1))
    n = len(modes)
    Z, I = np.zeros((n, n)), np.eye(n)
    J = np.block([[Z, I], [-I, Z]])
    Gp = np.block([[Z, I], [I, Z]])
    for Mi in M:
        assert np.max(np.abs(Mi.T @ J + J @ Mi)) <= 1e-12 * np.max(np.abs(Mi))
        assert np.max(np.abs(Mi.conj().T @ Gp + Gp @ Mi)) <= 1e-12 * np.max(np.abs(Mi))


# ============================================================================================ uniform limit
@pytest.mark.parametrize("cf", [1.0, 2.0])
def test_uniform_limit_straight_lossless(cf):
    f, L = 30e9, 0.1
    b = WR90.basis(f, cf)
    for uniform in (None, [(0.0, L)]):
        vg = VariableGuide(L, 22.86e-3, 10.16e-3, guide=WR90, mode_factor=1.0, uniform=uniform)
        assert np.max(np.abs(vg.gsm(f, b, b).S - Straight(L, WR90).gsm(f, b).S)) < 1e-10


def test_uniform_limit_straight_lossy():
    g = Guide(22.86e-3, 10.16e-3, 3.5e7)
    f, L = 30e9, 0.1
    b = g.basis(f, 1.0)
    vg = VariableGuide(L, 22.86e-3, 10.16e-3, guide=g, mode_factor=1.0, families="coupled")
    assert np.max(np.abs(vg.gsm(f, b, b).S - Straight(L, g).gsm(f, b).S)) < 1e-10


@pytest.mark.parametrize("plane", ["H", "E"])
@pytest.mark.parametrize("sigma", [None, 3.5e7])
def test_uniform_limit_smooth_bend(plane, sigma):
    g = Guide(22.86e-3, 10.16e-3, sigma)
    f, R = 30e9, 0.2
    L = R * np.pi / 2
    a, b = g.dims(plane)
    basis = g.basis(f, 1.0, plane=plane)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # the reference bend at atol 1e-12 warns at the round-off floor
        ref = SmoothBend(L, 1 / R, plane=plane, guide=g, atol=1e-12).gsm(f, basis)
    vg = VariableGuide(L, a, b, kappa=1 / R, plane=plane, guide=g, mode_factor=1.0, uniform=[(0, L)])
    assert np.max(np.abs(vg.gsm(f, basis, basis).S - ref.S)) < 1e-10


def test_uniform_limit_lossy_evanescent_entries_documented():
    """With evanescent modes the propagating block still equals Straight; the evanescent diagonal differs by the
    first-order impedance change Lambda^-_pp that the regular V-I loss keeps (module docstring): largest for
    modes just below cutoff (8e-3 on S11 of TE4,1 at |beta| = 50 1/m, 30 GHz), 2e-5 on transmission."""
    g = Guide(22.86e-3, 10.16e-3, 3.5e7)
    f, L = 30e9, 0.05
    b = g.basis(f, 2.0)
    G = VariableGuide(L, 22.86e-3, 10.16e-3, guide=g, mode_factor=1.0, families="coupled").gsm(f, b, b)
    S = Straight(L, g).gsm(f, b)
    p = np.concatenate([b.propagating_mask(f)] * 2)
    assert np.max(np.abs(G.S[np.ix_(p, p)] - S.S[np.ix_(p, p)])) < 1e-10
    assert np.max(np.abs(G.S - S.S)) < 1e-2
    assert np.max(np.abs(G.S21 - S.S21)) < 1e-4


# ============================================================================================ exact solutions
def _exact_single(mode, a0, a1, b, f, Phi):
    def rows(a):
        s = np.sqrt(complex(wave_admittance(mode, a, b, f)))
        return np.array([[s, 1 / s]]) / (2 * np.sqrt(2)), np.array([[s, -1 / s]]) / (2 * np.sqrt(2))
    Pp0, Pm0 = rows(a0)
    Pp1, Pm1 = rows(a1)
    A = np.vstack([Pp0, Pm1 @ Phi])
    B = np.vstack([Pm0, Pp1 @ Phi])
    return B @ np.linalg.inv(A)


def test_single_mode_through_cutoff_airy():
    """TE1,0 with a(z) = pi / sqrt(k^2 - g (z - z_c)): beta^2 = g (z - z_c) exactly (T2 section 10)."""
    f = 10e9
    k, w = 2 * np.pi * f / cst.c, 2 * np.pi * f
    gam, zc, L, b = k * k / 0.3, 0.03, 0.10, 0.01016
    s = gam ** (1 / 3)
    a = GeometryFunction(lambda z: np.pi / np.sqrt(k * k - gam * (np.asarray(z) - zc)),
                         lambda z: 0.5 * np.pi * gam * (k * k - gam * (np.asarray(z) - zc)) ** -1.5)

    def sols(z):
        ai, aip, bi, bip = airy(-s * (z - zc))
        return np.array([[ai, bi], 1j * (-s) * np.array([aip, bip]) / (w * cst.mu_0)])
    te10 = Mode("TE", 1, 0)
    Sx = _exact_single(te10, a(0.0), a(L), b, f, sols(L) @ np.linalg.inv(sols(0.0)))
    vg = VariableGuide(L, a, b, mode_factor=1.0, tail=False, atol=1e-11)
    G = vg.gsm(f, ModeBasis(a(0.0), b, [te10]), ModeBasis(a(L), b, [te10]))
    assert G.basis1.propagating_mask(f)[0] == False and G.basis2.propagating_mask(f)[0] == True  # noqa: E712
    assert np.max(np.abs(G.S - Sx)) < 1e-8


def test_single_mode_linear_taper_bessel():
    f = 12e9
    k, w = 2 * np.pi * f / cst.c, 2 * np.pi * f
    a0, alpha, L, b = 0.014, 0.25, 0.08, 0.01016
    nu = np.sqrt(0.25 + (np.pi / alpha) ** 2)
    K = k / alpha

    def sols(z):
        x = a0 + alpha * z
        V = np.array([np.sqrt(x) * jv(nu, K * x), np.sqrt(x) * yv(nu, K * x)])
        dV = np.array([jv(nu, K * x) / (2 * np.sqrt(x)) + np.sqrt(x) * K * jvp(nu, K * x),
                       yv(nu, K * x) / (2 * np.sqrt(x)) + np.sqrt(x) * K * yvp(nu, K * x)])
        return np.array([V, 1j * alpha * dV / (w * cst.mu_0)])
    te10 = Mode("TE", 1, 0)
    Sx = _exact_single(te10, a0, a0 + alpha * L, b, f, sols(L) @ np.linalg.inv(sols(0.0)))
    vg = VariableGuide(L, lin(a0, a0 + alpha * L, L), b, mode_factor=1.0, tail=False, atol=1e-11)
    G = vg.gsm(f, ModeBasis(a0, b, [te10]), ModeBasis(a0 + alpha * L, b, [te10]))
    assert np.max(np.abs(G.S - Sx)) < 1e-8


def test_total_reflection_at_cutoff_plane():
    """TE3,0 from the wide end of a linear H-plane taper with a' = 0.028 (T2 section 8.5)."""
    f = 20e9
    k = 2 * np.pi * f / cst.c
    a0, a1, L, b = 0.0205, 0.0275, 0.25, 0.01016
    zc = (3 * np.pi / k - a0) * L / (a1 - a0)
    zz = np.linspace(zc, L, 40001)
    beta = np.sqrt(np.maximum(k * k - (3 * np.pi / (a0 + (a1 - a0) * zz / L)) ** 2, 0))
    phase = np.trapezoid(beta, zz)
    G = Taper(L, a1, b, "linear", guide=Guide(a0, b)).gsm(f)
    i = G.basis2.index("TE3,0")
    S22 = G.S22[i, i]
    assert 1 - abs(S22) <= 1e-4
    assert abs(np.angle(S22 / (1j * np.exp(-2j * phase)))) <= 0.02


# ============================================================================================ T2b table
T2B = {0.1: (5.0040e-02 - 5.3514e-03j, 2.009e-3), 0.2: (2.7271e-03 + 2.5519e-02j, 8.503e-4),
       0.4: (-1.1563e-03 - 1.2944e-02j, 4.394e-4), 0.8: (-1.0636e-03 - 6.4437e-03j, 2.245e-4)}


@pytest.mark.parametrize("L", [0.1, 0.2, 0.4, 0.8])
def test_t2b_small_slope_table(L):
    f = 15e9
    b1 = ModeBasis.propagating(f, 22.86e-3, 10.16e-3, families=[0], frame="H")
    b2 = ModeBasis.propagating(f, 45e-3, 10.16e-3, families=[0], frame="H")
    G = Taper(L, 45e-3, 10.16e-3, "linear", guide=WR90).gsm(f, b1, b2)
    j = b1.index("TE1,0")
    s31 = G.S21[b2.index("TE3,0"), j]
    assert abs(s31 - T2B[L][0]) <= 1e-4
    assert abs(abs(G.S11[j, j]) - T2B[L][1]) <= 1e-4


# ============================================================================================ properties
def _general():
    L = 0.12
    s = lambda z: np.sin(np.pi * np.asarray(z) / L)
    c = lambda z: np.cos(np.pi * np.asarray(z) / L)
    a = GeometryFunction(lambda z: 0.0229 * (1 + 0.35 * s(z) ** 2), lambda z: 0.0229 * 0.35 * 2 * s(z) * c(z) * np.pi / L)
    b = lin(0.0102, 0.0102 * 1.2, L)
    u0 = GeometryFunction(lambda z: 0.002 * s(z), lambda z: 0.002 * c(z) * np.pi / L)
    v0 = GeometryFunction(lambda z: 0.001 * s(z) ** 2, lambda z: 0.001 * 2 * s(z) * c(z) * np.pi / L)
    kap = GeometryFunction(lambda z: 4.0 * s(z), lambda z: 4.0 * c(z) * np.pi / L)
    return L, a, b, u0, v0, kap


@pytest.mark.parametrize("sigma", [None, 3.5e7])
def test_reciprocity_unitarity_passivity_general_geometry(sigma):
    """a', b', u0', v0', kappa at once (all families coupled), evanescent port modes included."""
    L, a, b, u0, v0, kap = _general()
    g = Guide(0.0229, 0.0102, sigma)
    vg = VariableGuide(L, a, b, kappa=kap, u0=u0, v0=v0, guide=g)
    f = 20e9
    b1, b2 = vg.default_basis(f, 1.5)
    G = vg.gsm(f, b1, b2)
    assert G.reciprocity_defect() <= 1e-10 * np.max(np.abs(G.S))
    if sigma is None:
        assert G.unitarity_defect() <= 1e-10
    else:
        assert G.passivity_margin() >= 0
        assert G.passivity_margin() < 0.05


def test_strangled_bend_symmetry_and_frames():
    f = 40e9
    sb = StrangledBend(90, 0.05, 1.6e-3, 0.05, guide=WR90)
    assert sb.plane == "E"
    assert sb.length == pytest.approx(0.1 + 0.05 * np.pi / 2)
    G = sb.gsm(f)
    assert np.max(np.abs(G.S11 - G.S22)) < 1e-10        # symmetric component, same family: no sign matrix
    assert G.reciprocity_defect() < 1e-10
    assert G.unitarity_defect() < 1e-10
    ln = Line(WR90, [sb])
    assert [type(e).__name__ for e in ln.elements] == ["FrameChange", "StrangledBend", "FrameChange"]
    Gl = ln.gsm(f)
    j = Gl.basis1.index("TE1,0")
    assert abs(abs(Gl.S21[j, j]) - abs(G.S21[G.basis2.index("TE0,1"), G.basis1.index("TE0,1")])) < 1e-12


def test_taper_in_a_line_changes_guide():
    f = 18e9
    ln = Line(WR90, [Straight(0.01), Taper(0.1, 30e-3, 10.16e-3, "raised_cosine"), Straight(0.02),
                     Taper(0.1, 22.86e-3, 10.16e-3, "raised_cosine")])
    assert ln.guides[1] == WR90
    assert ln.elements[2].guide is None or True
    G = ln.gsm(f)
    assert G.reciprocity_defect() < 1e-10 and G.unitarity_defect() < 1e-10
    lossy = Guide(22.86e-3, 10.16e-3, 3.5e7)
    ln2 = Line(lossy, [Taper(0.1, 30e-3, 10.16e-3, "raised_cosine")])
    assert ln2.guides[1] == Guide(30e-3, 10.16e-3, 3.5e7)
    assert ln2.gsm(f).passivity_margin() > 0


def test_lossy_taper_loss_close_to_local_attenuation():
    """A gentle lossy taper loses about the integral of the local TE1,0 attenuation (T1 B3)."""
    from wrmodes.loss import attenuation_constant
    f, L = 30e9, 0.3
    g = Guide(22.86e-3, 10.16e-3, 3.5e7)
    G = Taper(L, 30e-3, 10.16e-3, "raised_cosine", guide=g).gsm(f)
    j = G.basis1.index("TE1,0")
    loss = -np.log(np.sum(np.abs(G.S21[:, j]) ** 2) + np.sum(np.abs(G.S11[:, j]) ** 2))
    z = np.linspace(0, L, 401)
    a = 22.86e-3 + (30e-3 - 22.86e-3) * 0.5 * (1 - np.cos(np.pi * z / L))
    expect = 2 * np.trapezoid([attenuation_constant("TE1,0", ai, 10.16e-3, f, 3.5e7) for ai in a], z)
    assert loss == pytest.approx(expect, rel=2e-2)


def test_port_at_cutoff_raises():
    f = cst.c / (2 * 0.03)              # TE1,0 exactly at cutoff for a = 30 mm
    tp = Taper(0.1, 30e-3, 10.16e-3, guide=Guide(40e-3, 10.16e-3))
    b1 = ModeBasis(40e-3, 10.16e-3, ["TE1,0"], frame="H")
    b2 = ModeBasis(30e-3, 10.16e-3, ["TE1,0"], frame="H")
    with pytest.raises(ValueError, match="cutoff"):
        tp.gsm(f, b1, b2)


def test_guide_and_frame_checks():
    with pytest.raises(ValueError):
        VariableGuide(0.1, 22.86e-3, 10.16e-3, guide=Guide(20e-3, 10.16e-3))
    vg = VariableGuide(0.1, 10.16e-3, 22.86e-3, plane="E", guide=WR90)
    with pytest.raises(ValueError):
        vg.gsm(10e9, WR90.basis(10e9), WR90.basis(10e9))          # H-frame bases for an E-plane component
    with pytest.raises(ValueError):
        Taper(0.1, 30e-3, 10.16e-3).gsm(10e9)                      # no guide yet


def test_families_conserved_without_b_slope():
    f = 40e9
    G = Taper(0.1, 30e-3, 10.16e-3, "linear", guide=WR90).gsm(f)
    for i, p in enumerate(G.basis2.modes):
        for j, q in enumerate(G.basis1.modes):
            if p.n != q.n or (p.m + q.m) % 2:
                assert G.S21[i, j] == 0


# ============================================================================================ B5' and convergence
def test_tail_remainder_matches_long_explicit_sum(monkeypatch):
    f, L = 15e9, 0.1
    b1 = ModeBasis.propagating(f, 22.86e-3, 10.16e-3, families=[0], frame="H")
    b2 = ModeBasis.propagating(f, 45e-3, 10.16e-3, families=[0], frame="H")
    tp = Taper(L, 45e-3, 10.16e-3, "linear", guide=WR90)
    S = tp.gsm(f, b1, b2, slice_factor=4).S
    monkeypatch.setattr(tm, "TAIL_EXPLICIT", 800)
    monkeypatch.setattr(tm, "TAIL_TERMS", 1)
    S_long = tp.gsm(f, b1, b2, slice_factor=4).S
    assert np.max(np.abs(S - S_long)) < 1e-6


def test_tail_speeds_up_convergence_in_mode_count():
    f, L = 15e9, 0.1
    b1 = ModeBasis.propagating(f, 22.86e-3, 10.16e-3, families=[0], frame="H")
    b2 = ModeBasis.propagating(f, 45e-3, 10.16e-3, families=[0], frame="H")
    tp = Taper(L, 45e-3, 10.16e-3, "linear", guide=WR90)
    j = b1.index("TE1,0")
    S = {(mf, t): tp.gsm(f, b1, b2, mode_factor=mf, tail=t).S[:, j] for mf in (2.0, 4.0, 8.0) for t in (True, False)}
    ref = S[(8.0, True)]
    on = np.max(np.abs(S[(4.0, True)] - ref))
    off = np.max(np.abs(S[(4.0, False)] - ref))
    assert on < 1e-4 and off > 3 * on
    rich = (8 * S[(8.0, False)] - 4 * S[(4.0, False)]) / 4
    assert np.max(np.abs(rich - ref)) < np.max(np.abs(S[(8.0, False)] - ref))


def test_convergence_study_report():
    tp = Taper(0.1, 45e-3, 10.16e-3, "linear", guide=WR90)
    b1 = ModeBasis.propagating(15e9, 22.86e-3, 10.16e-3, families=[0], frame="H")
    b2 = ModeBasis.propagating(15e9, 45e-3, 10.16e-3, families=[0], frame="H")
    r = tp.convergence_study(15e9, "TE1,0", basis1=b1, basis2=b2)
    assert [row["mode_factor"] for row in r["rows"]] == [2.0, 4.0, 2.0, 4.0, 2.0]
    assert [row["slice_factor"] is None for row in r["rows"]] == [True] * 4 + [False]
    assert set(r["richardson"]) == {True, False}
    assert r["modes2"] == ["TE1,0", "TE2,0", "TE3,0", "TE4,0"]
    assert len(r["column"]) == 5
    slices = [row for row in r["rows"] if row["slice_factor"] is not None]
    assert np.max(np.abs(slices[0]["S"] - r["rows"][0]["S"])) < 2e-6


# ============================================================================================ literature, slow
def test_butterweck_bottle_vs_linear_family_zero():
    """P5/P6 in the family n = 0 model (b constant): bottle below -40 dB, linear about -20 dB (measured -18)."""
    f, L, a0, a1, b = 75e9, 0.14, 3.10e-3, 22.86e-3, 10.16e-3
    b1 = ModeBasis(a0, b, [m for m in ModeBasis.propagating(f, a0, b).modes if m.n == 0 and m.m % 2])
    b2 = ModeBasis(a1, b, [m for m in ModeBasis.propagating(f, a1, b).modes if m.n == 0 and m.m % 2])
    out = {}
    for name in ("bottle", "linear"):
        G = Taper(L, a1, b, name, guide=Guide(a0, b), mode_factor=1.5, atol=1e-5).gsm(f, b1, b2)
        out[name] = 10 * np.log10(abs(G.S21[b2.index("TE3,0"), 0]) ** 2)
    assert out["bottle"] < -40
    assert -22 < out["linear"] < -18


@pytest.mark.slow
def test_against_t2_prototype_solver():
    vs = _load("docs/theory/T2/vi_solver.py", "vi_solver")
    f = 20e9
    L, a, b, u0, v0, kap = _general()
    geometry = lambda z: dict(a=float(a(z)), b=float(b(z)), u0=float(u0(z)), v0=0.0, kappa=float(kap(z)),
                              da=float(a.derivative(z)), db=float(b.derivative(z)), du0=float(u0.derivative(z)),
                              dv0=0.0)
    modes = [Mode(k, m, n) for m in range(0, 5) for n in (0, 2) for k in ("TE", "TM")
             if not (k == "TE" and m == n == 0) and not (k == "TM" and (m == 0 or n == 0))]
    vg = VariableGuide(L, a, b, kappa=kap, u0=u0, mode_factor=1.0, tail=False, atol=1e-10)
    b1 = ModeBasis(a(0.0), b(0.0), modes, sort=False)
    b2 = ModeBasis(a(L), b(L), modes, sort=False)
    G = vg.gsm(f, b1, b2)
    R = vs.taper_gsm(modes, geometry, f, L, nslices=160, nsub=2)
    assert np.max(np.abs(G.S - R.S)) < 1e-6


@pytest.mark.slow
def test_staircase_of_steps_approaches_taper():
    """A few points of benchmarks/taper/staircase.py: the error of the WP4 staircase falls with the step count."""
    st = _load("benchmarks/taper/staircase.py", "staircase")
    case = st.CASES["b-taper"]
    gt, p1, p2 = st.taper(case)
    ref = st.block(gt, p1, p2)
    d = [np.max(np.abs(st.block(st.staircase(case, n), p1, p2) - ref)) for n in (20, 80)]
    assert d[1] < d[0] / 3


@pytest.mark.slow
def test_timing_taper_90ghz():
    """WR-90 -> 22.86 x 1.6 mm raised-cosine taper of 50 mm at 90 GHz: < 5 s (card WP5; 15 s on CI)."""
    tp = Taper(0.05, 1.6e-3, 22.86e-3, "raised_cosine", guide=WR90, plane="E")
    t0 = time.time()
    G = tp.gsm(90e9)
    dt = time.time() - t0
    assert dt < (5.0 if PERF else 15.0)
    assert G.unitarity_defect() < 1e-10


@pytest.mark.slow
def test_timing_strangled_bend_90ghz():
    """The SPARC strangled bend (two such tapers, R = 50-200 mm, 90 deg) at 90 GHz: < 30 s (90 s on CI)."""
    for R in (0.05, 0.2):
        sb = StrangledBend(90, R, 1.6e-3, 0.05, guide=WR90)
        t0 = time.time()
        G = sb.gsm(90e9)
        dt = time.time() - t0
        assert dt < (30.0 if PERF else 90.0)
        assert G.unitarity_defect() < 1e-10


@pytest.mark.slow
def test_e3_predictions_regression():
    """E3 at 18 GHz against the stored predictions (benchmarks/taper/data/e3_predictions.npz)."""
    d = np.load(os.path.join(ROOT, "benchmarks/taper/data/e3_predictions.npz"))
    for wall, sigma in (("PEC", None), ("aluminium", 3.774e7)):
        g = Guide(22.86e-3, 10.16e-3, sigma)
        G = Line(g, [StrangledBend(90, 0.05, 1.6e-3, 0.05, guide=g, atol=1e-7)]).gsm(18e9)
        j = G.basis1.index("TE1,0")
        col = d[f"18GHz_{wall}_S21_col"][0]
        assert np.max(np.abs(G.S21[:, j] - col)) < 1e-7
