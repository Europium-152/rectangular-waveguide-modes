"""Tests of the 2D modal-port FEM solver (wrmodes.fem2d, WP3). Skipped without the optional extra wrmodes[fem]."""
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import jv, jvp, yv, yvp
import scipy.constants as cst

try:
    import wrmodes.fem2d as fem2d
except ImportError as exc:                      # the extra is missing (or gmsh cannot load its libraries)
    pytest.skip(f"wrmodes[fem] not available: {exc}", allow_module_level=True)

from wrmodes.fem2d import Coplanar2D, Miter, arc_bend_outline, straight_outline, convergence_study
from wrmodes.fem2d.normalization import ls_amplitude_factors, closed_form_factor
from wrmodes.gsm import block_diag, from_transfer_matrix, straight
from wrmodes.modes import ModeBasis
from wrmodes.transfer_matrix import coupled_mode_matrices

W, H = 0.9 * 0.0254, 0.4 * 0.0254              # WR-90
ACCURATE = dict(order=3, mesh_density=10)      # settings for the phase-accurate comparisons (see docs/reports/WP3.md)


def local(plane):
    """(a, b) of the local frame (CONVENTIONS.md section 2): a is the in-plane side."""
    return (W, H) if plane == "H" else (H, W)


def rel_reciprocity(g):
    return g.reciprocity_defect() / np.max(np.abs(g.S))


# ----------------------------------------------------------------------------------------- normalization
@pytest.mark.parametrize("f, cutoff_factor", [(10e9, 1.0), (23e9, 1.0), (23e9, 1.6), (41e9, 1.3)])
@pytest.mark.parametrize("plane", ["H", "E"])
def test_scalar_factors_match_closed_forms(plane, f, cutoff_factor):
    """g and h computed from the library fields equal the closed forms of docs/theory/WP3-normalization.md,
    sign and branch included, for propagating and evanescent modes; g^2 beta and h^2 beta are constant in a family."""
    a, b = local(plane)
    basis = ModeBasis.propagating(f, a, b, cutoff_factor=cutoff_factor)
    Q, labels, fac = ls_amplitude_factors(basis, f)
    beta = basis.beta(f)
    for (pol, m, n), x in zip(labels, fac):
        ref = closed_form_factor(pol, m, n, a, b, f)
        assert abs(x - ref) <= 1e-12 * abs(ref), (pol, m, n, x, ref)
    for pol in ("LSM", "LSE"):
        for n in basis.families():
            # beta of LS row l is that of the TE/TM pair of row l
            vals = [fac[l] ** 2 * beta[basis.index(("TE", m, n))]
                    for l, (p, m, nn) in enumerate(labels) if p == pol and nn == n]
            if vals:
                assert np.allclose(vals, vals[0], rtol=1e-12, atol=0)


# ----------------------------------------------------------------------------------------- straight channel
def test_straight_channel_equals_straight_section():
    """No discontinuity: S equals gsm.straight to 1e-6, signs included (all propagating modes, families 0 and 1)."""
    f, L = 20e9, 0.01
    c = Coplanar2D(straight_outline(W, L), H, frame="H", order=3, mesh_density=15, port_distance=W / 4)
    b1, b2 = c.default_basis(f)
    assert len(b1) == 8 and b1.families() == [0, 1]
    g = c.gsm(f)
    assert np.max(np.abs(g.S - straight(b1, f, L).S)) < 1e-6
    assert rel_reciprocity(g) < 1e-10 and g.unitarity_defect() < 1e-10


def test_straight_channel_with_evanescent_modes():
    """Evanescent modes in the basis are de-embedded over the arms: still exact for a straight channel. (Their
    entries carry the discretization error times exp(2 |beta| L_p), about 30 for TE3,1 here.)"""
    f, L = 20e9, 0.01
    c = Coplanar2D(straight_outline(W, L), H, frame="H", order=3, mesh_density=20, port_distance=W / 4)
    b1, _ = c.default_basis(f, cutoff_factor=1.3)
    assert not b1.propagating_mask(f).all()
    g = c.gsm(f, b1, b1)
    assert np.max(np.abs(g.S - straight(b1, f, L).S)) < 1e-5


# ----------------------------------------------------------------------------------------- smooth bends
def wrmodes_bend(f, a, b, frame, radius, angle):
    """Constant-curvature bend from the wrmodes 0.1 coupled-mode matrices (exact exponential), all families."""
    basis = ModeBasis.propagating(f, a, b, frame=frame)
    gs = []
    for n in basis.families():
        bn = ModeBasis.propagating(f, a, b, families=[n], frame=frame)
        D, K, _ = coupled_mode_matrices(f, a, b, bn.legacy_modes())
        gs.append(from_transfer_matrix(expm((D + K / radius) * radius * angle), bn, f))
    return block_diag(*gs, basis1=basis, basis2=basis)


@pytest.mark.parametrize("f", [10e9, 17.5e9, 22e9])
@pytest.mark.parametrize("plane", ["H", "E"])
def test_smooth_bend_matches_wrmodes(plane, f):
    """Complex S21 of every propagating mode (all pairs) equals wrmodes to 1e-3: gauge and sign of every coupling.

    kappa/beta <= 0.03 for every propagating mode. The frequencies avoid the neighbourhood of evanescent cut-offs,
    where the propagating-mode coupled-mode theory itself errs by more than 1e-3 (docs/reports/WP3.md).
    """
    a, b = local(plane)
    R, angle = 0.5, np.radians(10)
    ref = wrmodes_bend(f, a, b, plane, R, angle)
    beta = ref.basis1.beta(f).real
    assert np.max(1 / R / beta) <= 0.03
    c = Coplanar2D(arc_bend_outline(a, R, angle), b, frame=plane, port_distance=a / 4, **ACCURATE)
    g = c.gsm(f, ref.basis1, ref.basis2)
    assert np.max(np.abs(g.S21 - ref.S21)) < 1e-3
    assert rel_reciprocity(g) < 1e-10 and g.unitarity_defect() < 1e-10


def bend_mode_order(kt, r1, r2, pol, guess):
    """Order nu of the bend mode of an annulus (Dirichlet: LSM, Neumann: LSE) closest to guess."""
    if pol == "LSM":
        F = lambda nu: jv(nu, kt * r1) * yv(nu, kt * r2) - jv(nu, kt * r2) * yv(nu, kt * r1)
    else:
        F = lambda nu: jvp(nu, kt * r1) * yvp(nu, kt * r2) - jvp(nu, kt * r2) * yvp(nu, kt * r1)
    nus = np.linspace(guess - 2, guess + 2, 801)
    v = np.array([F(x) for x in nus])
    roots = [brentq(F, nus[i], nus[i + 1]) for i in range(len(nus) - 1) if np.sign(v[i]) != np.sign(v[i + 1])]
    return min(roots, key=lambda r: abs(r - guess))


@pytest.mark.parametrize("plane, pol", [("H", "LSM"), ("E", "LSE")])
def test_single_mode_bend_matches_exact_bend_mode(plane, pol):
    """Single-mode bend (TE10 in H, TE01 in E, local frame): S21 = exp(-i nu angle) of the exact bend mode,
    including the second-order phase that the propagating-mode coupled-mode theory leaves out."""
    a, b = local(plane)
    f, R, angle = 10e9, 0.25, np.radians(20)
    n = 0 if plane == "H" else 1
    k = 2 * np.pi * f / cst.c
    kt = np.sqrt(k * k - (n * np.pi / b) ** 2)
    beta0 = np.sqrt(kt ** 2 - (np.pi / a) ** 2) if pol == "LSM" else kt
    nu = bend_mode_order(kt, R - a / 2, R + a / 2, pol, beta0 * R)
    c = Coplanar2D(arc_bend_outline(a, R, angle), b, frame=plane, port_distance=a / 4, **ACCURATE)
    g = c.gsm(f)
    assert len(g.basis1) == 1
    s21 = g.S21[0, 0]
    assert abs(s21 - np.exp(-1j * nu * angle)) < 5e-5
    if plane == "H":    # here the straight-guide phase is off by more than 1e-3 (wrmodes' truncation)
        assert abs(np.exp(-1j * beta0 * R * angle) - np.exp(-1j * nu * angle)) > 1e-3


# ----------------------------------------------------------------------------------------- miters
@pytest.mark.parametrize("angle_deg, chamfer", [(90, None), (60, None), (90, 0.968), (90, 0.5), (45, 1.2)])
def test_miter_reciprocity_unitarity_symmetry(angle_deg, chamfer):
    """Default settings, default basis (all families of propagating modes): reciprocity and S11 = S22 to 1e-10,
    unitarity to 1e-5 (in fact to rounding: the discrete modal-port system conserves power exactly)."""
    m = Miter(np.radians(angle_deg), chamfer=chamfer, guide=(W, H))
    g = m.gsm(30e9)
    assert g.basis1.families() == [0, 1, 2]
    assert rel_reciprocity(g) < 1e-10
    assert g.unitarity_defect() < 1e-5
    assert np.max(np.abs(g.S11 - g.S22)) < 1e-10
    assert g.basis1.frame == "H"
    assert np.isclose(m.arm_length, W / 2 * np.tan(np.radians(angle_deg) / 2))


def test_miter_continuous_across_geometry_cases():
    """The outline changes form at d = cos(theta/2) (plain miter; deeper cuts are de-embedded from the arm) and at
    d = 1/cos(theta/2) (no cut): S must be continuous there, phases included."""
    theta, f = np.radians(90), 12e9
    S = lambda d: Miter(theta, chamfer=d, guide=(W, H)).gsm(f).S
    # plain miter: S is smooth in d through the change of outline; one-sided differences of +-1e-3 and +-2e-3
    d0 = np.cos(theta / 2)
    S0 = S(d0)
    dm2, dm1, dp1, dp2 = (S(d0 + s) - S0 for s in (-2e-3, -1e-3, 1e-3, 2e-3))
    assert np.max(np.abs(dm1 + dp1)) < 0.05 * np.max(np.abs(dp1))           # same slope on both sides
    assert np.max(np.abs(dm2 - 2 * dm1)) < 0.05 * np.max(np.abs(dm2))       # linear, i.e. no jump
    # sharp corner: a vanishing cut gives a vanishing change
    d1 = 1 / np.cos(theta / 2)
    assert np.max(np.abs(S(d1 - 1e-3) - S(d1))) < 1e-3


def test_e_plane_miter_smoke():
    """E-plane miter: local frame a = h, b = w, frame tag 'E'; TE01 (global TE10) is LSE."""
    m = Miter(np.pi / 2, plane="E", guide=(W, H))
    g = m.gsm(20e9)
    assert g.basis1.frame == "E" and np.isclose(g.basis1.a, H) and np.isclose(g.basis1.b, W)
    assert rel_reciprocity(g) < 1e-10 and g.unitarity_defect() < 1e-10
    assert np.max(np.abs(g.S11 - g.S22)) < 1e-10
    assert 0 < abs(g.transmission("TE0,1")) < 1


def test_family_without_propagating_mode_is_refused():
    m = Miter(np.pi / 2, guide=(W, H))
    f = 10e9                                    # family n = 1 is below cutoff (14.76 GHz)
    basis = ModeBasis.propagating(f, W, H, cutoff_factor=2.0, evanescent_families="all", frame="H")
    assert 1 in basis.families()
    with pytest.raises(ValueError):
        m.gsm(f, basis, basis)


def test_basis_of_other_frame_is_refused():
    m = Miter(np.pi / 2, guide=(W, H))
    basis = ModeBasis.propagating(10e9, W, H, frame="E")
    with pytest.raises(ValueError):
        m.gsm(10e9, basis, basis)


def test_convergence_study_runs():
    m = Miter(np.pi / 2, guide=(W, H))
    out = convergence_study(m, 10e9, mesh_densities=(8, 12, 16), n_evanescent=(2, 6, 10))
    mesh = [x[2] for x in out["mesh"]]
    assert mesh[-1] == 0 and mesh[0] < 1e-3
    assert out["reciprocity"] < 1e-10


def test_core_imports_without_fem_extra():
    """`import wrmodes` must not import scikit-fem or gmsh (CONVENTIONS.md section 10)."""
    code = "import sys, wrmodes; print('skfem' in sys.modules, 'gmsh' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "False False"
