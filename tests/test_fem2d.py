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

from wrmodes import Line, Guide, Straight, FrameChange
from wrmodes.fem2d import (Coplanar2D, Miter, Outline, PortPlane, arc_bend_outline, straight_outline, miter_outline,
                           mirror_outline, convergence_study)
from wrmodes.fem2d.normalization import ls_amplitude_factors, closed_form_factor
from wrmodes.gsm import block_diag, from_transfer_matrix, straight
from wrmodes.modes import Mode, ModeBasis
from wrmodes.transfer_matrix import coupled_mode_matrices

W, H = 0.9 * 0.0254, 0.4 * 0.0254              # WR-90
WR90 = Guide(W, H)
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


def eplane_step_outline(a1, a2, ell):
    """Symmetric E-plane step from height a1 to a2 < a1 (local frame), port 1 a distance ell before the step and
    de-embedded to the step plane, port 2 on the step plane."""
    V = [(-a1 / 2, -ell), (a1 / 2, -ell), (a1 / 2, 0.0), (a2 / 2, 0.0), (-a2 / 2, 0.0), (-a1 / 2, 0.0)]
    ports = (PortPlane(V[0], V[1], deembed=ell), PortPlane(V[4], V[3]))
    return Outline(vertices=V, ports=ports)


@pytest.mark.parametrize("ratio, static, measured", [(2, -1 / 3, -0.3336 - 0.0145j), (4, -0.6, -0.6005 - 0.0136j)])
def test_lse_reflection_sign_on_e_plane_step(ratio, static, measured):
    """Pins the sign of the LSE reflection (docs/theory/WP3-normalization.md section 3): at 7 GHz a symmetric E-plane
    step a -> a/ratio reflects TE01 (global TE10) with S11 close to the static TEM value (Z2 - Z1)/(Z2 + Z1) =
    (1/ratio - 1)/(1/ratio + 1), i.e. negative. `measured` is the reviewer's independent value."""
    c = Coplanar2D(eplane_step_outline(H, H / ratio, H / 2), W, plane="E")
    g = c.gsm(7e9)
    assert g.basis1.names == ["TE0,1"]
    s11 = g.S11[0, 0]
    assert abs(s11.real - static) < 0.01 and abs(s11.imag) < 0.05
    assert abs(s11 - measured) < 2e-3
    assert rel_reciprocity(g) < 1e-10 and g.unitarity_defect() < 1e-10


# ----------------------------------------------------------------------------------------- straight channel
def test_straight_channel_equals_straight_section():
    """No discontinuity: S equals gsm.straight to 1e-6, signs included (all propagating modes, families 0 and 1)."""
    f, L = 20e9, 0.01
    c = Coplanar2D(straight_outline(W, L), H, plane="H", order=3, mesh_density=15)
    b1, b2 = c.default_basis(f)
    assert len(b1) == 8 and b1.families() == [0, 1]
    g = c.gsm(f)
    assert np.max(np.abs(g.S - straight(b1, f, L).S)) < 1e-6
    assert rel_reciprocity(g) < 1e-10 and g.unitarity_defect() < 1e-10


def test_straight_channel_with_evanescent_modes_and_dead_families():
    """Evanescent modes in the basis, including families without a propagating mode (k_t^2 < 0, CONVENTIONS.md
    section 3, the line's 'live' rule): still exactly a straight section, default settings."""
    f, L = 20e9, 0.01
    c = Coplanar2D(straight_outline(W, L), plane="H", guide=WR90)
    b, _ = c.default_basis(f, cutoff_factor=2.0)
    k = 2 * np.pi * f / cst.c
    dead = [n for n in b.families() if (n * np.pi / H) ** 2 > k * k]
    assert dead and not b.propagating_mask(f).all()
    g = c.gsm(f, b, b)
    err = np.abs(g.S - straight(b, f, L).S)
    p = np.concatenate([b.propagating_mask(f)] * 2)
    assert np.max(err[np.ix_(p, p)]) < 1e-5
    # evanescent entries: discretization error times at most evanescent_gain (FEMOptions)
    assert np.max(err) < 1e-3
    assert rel_reciprocity(g) < 1e-10


def test_family_at_kt_zero_is_uncoupled():
    """Exactly at k_t = 0 the LS rotation does not exist: the family is returned as straight modes over `length`."""
    f = cst.c / (2 * H)                         # k_t = 0 for n = 1
    basis = ModeBasis(W, H, [Mode("TE", 1, 0), Mode("TE", 1, 1), Mode("TM", 1, 1)], frame="H")
    m = Miter(90, guide=WR90)
    g = m.gsm(f, basis, basis)
    assert g.fem_info["uncoupled_families"] == [1]
    i = [basis.index("TE1,1"), basis.index("TM1,1")]
    P = np.exp(-1j * basis.beta(f)[i] * 2 * m.arm_length)
    assert np.allclose(np.diag(g.S21)[i], P, rtol=0, atol=1e-14) and np.allclose(g.S11[np.ix_(i, i)], 0)


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
    c = Coplanar2D(arc_bend_outline(a, R, angle), b, plane=plane, **ACCURATE)
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
    c = Coplanar2D(arc_bend_outline(a, R, angle), b, plane=plane, **ACCURATE)
    g = c.gsm(f)
    assert len(g.basis1) == 1
    s21 = g.S21[0, 0]
    assert abs(s21 - np.exp(-1j * nu * angle)) < 5e-5
    if plane == "H":    # here the straight-guide phase is off by more than 1e-3 (wrmodes' truncation)
        assert abs(np.exp(-1j * beta0 * R * angle) - np.exp(-1j * nu * angle)) > 1e-3


# ----------------------------------------------------------------------------------------- miters
@pytest.mark.parametrize("angle_deg, chamfer", [(90, None), (60, None), (60, np.cos(np.radians(30))),
                                                (90, 0.968), (90, 0.5), (45, 1.2), (120, None)])
def test_miter_reciprocity_unitarity_symmetry(angle_deg, chamfer):
    """Default settings, default basis (all families of propagating modes): reciprocity and S11 = S22 to 1e-10,
    unitarity to 1e-5 (in fact to rounding: the discrete modal-port system conserves power exactly)."""
    m = Miter(angle_deg, chamfer=chamfer, guide=WR90)
    g = m.gsm(30e9)
    assert g.basis1.families() == [0, 1, 2]
    assert rel_reciprocity(g) < 1e-10
    assert g.unitarity_defect() < 1e-5
    assert np.max(np.abs(g.S11 - g.S22)) < 1e-10
    assert g.basis1.frame == "H"
    assert np.isclose(m.arm_length, W / 2 * np.tan(np.radians(angle_deg) / 2))


def test_default_chamfer_is_the_optical_mirror():
    """chamfer=None: mirror through the intersection of the arm axes, d = 1/(2 cos(theta/2)); at 90 degrees this is
    the plain miter d = cos(45) of Ma (c/a = 1) and Alessandri (d/b = 0.707)."""
    for ang in (30, 60, 90, 120, 150):
        assert np.isclose(Miter(ang).d, 0.5 / np.cos(np.radians(ang) / 2))
    assert np.isclose(Miter(90).d, np.cos(np.pi / 4))
    # the optical 60-degree miter is a good bend in the overmoded regime, the outer-corner one is not
    f = 60e9
    b = ModeBasis.propagating(f, W, H, families=[0], frame="H")
    good = Miter(60, guide=WR90).gsm(f, b, b).loss_db("TE1,0")
    bad = Miter(60, chamfer=np.cos(np.radians(30)), guide=WR90).gsm(f, b, b).loss_db("TE1,0")
    assert good < 0.3 < 5 < bad


def test_negative_angle_is_the_mirror_image():
    """A miter turning towards +u (negative angle, D S D) equals the FEM of the mirrored outline (independent mesh)."""
    f = 20e9
    b = WR90.basis(f, 1.0, plane="H")
    o, _ = miter_outline(W, np.pi / 2, 1 / np.sqrt(2))
    mirrored = Coplanar2D(mirror_outline(o), plane="H", guide=WR90).gsm(f, b, b)
    g = Miter(-90, guide=WR90).gsm(f, b, b)
    assert np.max(np.abs(g.S - mirrored.S)) < 2e-4
    assert np.max(np.abs(g.S - Miter(90, guide=WR90).gsm(f, b, b).S)) > 0.1      # odd conversions change sign


def test_evanescent_entries_are_accurate():
    """Evanescent modes in the basis (30 GHz, family 0, cutoff_factor 1.6: TE5,0-TE7,0 evanescent): the default
    settings agree with two fine references of very different arm lengths (a/40 and a/10, i.e. different
    exp(|beta| L) shifts), which agree with each other."""
    f = 30e9
    b = ModeBasis.propagating(f, W, H, families=[0], cutoff_factor=1.6, frame="H")
    assert (~b.propagating_mask(f)).sum() == 3
    m = Miter(90, guide=WR90)
    ref1 = m.with_options(mesh_density=16, port_distance=W / 40).gsm(f, b, b)
    ref2 = m.with_options(mesh_density=16, port_distance=W / 10).gsm(f, b, b)
    g = m.gsm(f, b, b)
    ev = ~np.concatenate([b.propagating_mask(f)] * 2)
    assert np.max(np.abs(ref1.S[ev])) > 0.2                       # the evanescent entries are not small
    assert np.max(np.abs(ref1.S - ref2.S)) < 2e-5
    assert np.max(np.abs(g.S - ref1.S)) < 2e-3
    assert rel_reciprocity(g) < 1e-10 and g.unitarity_defect() < 1e-10


def test_miter_continuous_across_geometry_cases():
    """The outline changes form at d = cos(theta/2) (deeper cuts are de-embedded from the arm) and at
    d = 1/cos(theta/2) (no cut): S must be continuous there, phases included."""
    f = 12e9
    S = lambda d: Miter(90, chamfer=d, guide=WR90).gsm(f).S
    d0 = np.cos(np.pi / 4)
    S0 = S(d0)
    dm2, dm1, dp1, dp2 = (S(d0 + s) - S0 for s in (-2e-3, -1e-3, 1e-3, 2e-3))
    assert np.max(np.abs(dm1 + dp1)) < 0.05 * np.max(np.abs(dp1))           # same slope on both sides
    assert np.max(np.abs(dm2 - 2 * dm1)) < 0.05 * np.max(np.abs(dm2))       # linear, i.e. no jump
    d1 = 1 / np.cos(np.pi / 4)
    assert np.max(np.abs(S(d1 - 1e-3) - S(d1))) < 1e-3


def test_e_plane_miter_smoke():
    """E-plane miter: local frame a = h, b = w, frame tag 'E'; TE01 (global TE10) is LSE."""
    m = Miter(90, plane="E", guide=WR90)
    g = m.gsm(20e9)
    assert g.basis1.frame == "E" and np.isclose(g.basis1.a, H) and np.isclose(g.basis1.b, W)
    assert rel_reciprocity(g) < 1e-10 and g.unitarity_defect() < 1e-10
    assert np.max(np.abs(g.S11 - g.S22)) < 1e-10
    assert 0 < abs(g.transmission("TE0,1")) < 1


# ----------------------------------------------------------------------------------------- component protocol / Line
def test_component_protocol():
    m = Miter(90)
    assert m.plane == "H" and m.guide is None
    with pytest.raises(ValueError):
        m.gsm(10e9)                                           # no guide yet
    mg = m.with_guide(WR90)
    assert mg.guide == WR90 and m.guide is None and mg.with_guide(WR90) is mg
    b1, b2 = mg.default_basis(10e9, None)
    assert b1 == b2 == WR90.basis(10e9, 1.0, plane="H")
    with pytest.raises(NotImplementedError):                  # wall loss is WP2's
        Miter(90, guide=Guide(W, H, sigma=5.8e7)).gsm(10e9)
    with pytest.raises(ValueError):
        mg.gsm(10e9, *Miter(90, plane="E", guide=WR90).default_basis(10e9))   # basis of another frame
    with pytest.raises(ValueError):                           # an outline that does not fit the guide
        Coplanar2D(straight_outline(H, 0.01), plane="H", guide=WR90)


@pytest.mark.parametrize("f", [10e9, 15e9, 20e9])
def test_line_with_miters_equals_manual_cascade(f):
    """[Straight, Miter(90), Straight, Miter(-90), Straight] in WR-90, and E-plane miters with the frame changes
    the Line inserts, equal the manual cascade of the components' GSMs."""
    m1, m2 = Miter(90), Miter(-90)
    line = Line(WR90, [Straight(0.05), m1, Straight(0.1), m2, Straight(0.03)])
    G = line.gsm(f)
    b = WR90.basis(f, 1.0, plane="H")
    st = lambda L: Straight(L, WR90).gsm(f, b, b)
    man = st(0.05) @ m1.with_guide(WR90).gsm(f, b, b) @ st(0.1) @ m2.with_guide(WR90).gsm(f, b, b) @ st(0.03)
    assert np.max(np.abs(G.S - man.S)) < 1e-12
    assert rel_reciprocity(G) < 1e-10 and G.unitarity_defect() < 1e-10

    lineE = Line(WR90, [Straight(0.02), Miter(90, plane="E"), Miter(-90, plane="E"), Straight(0.02)])
    assert sum(lineE.inserted) == 2
    bE = WR90.basis(f, 1.0, plane="E")
    mE1, mE2 = Miter(90, plane="E", guide=WR90), Miter(-90, plane="E", guide=WR90)
    manE = (st(0.02) @ FrameChange("H", "E", WR90).gsm(f, b, bE) @ mE1.gsm(f, bE, bE) @ mE2.gsm(f, bE, bE)
            @ FrameChange("E", "H", WR90).gsm(f, bE, b) @ st(0.02))
    assert np.max(np.abs(lineE.gsm(f).S - manE.S)) < 1e-12


def test_line_with_live_evanescent_families():
    """With the line's 'live' rule (cutoff_factor 1.5 at 30 GHz) the basis has a family without propagating modes
    (n = 3); the line computes and stays reciprocal and lossless."""
    f = 30e9
    line = Line(WR90, [Straight(0.02), Miter(90), Straight(0.02)])
    b = WR90.basis(f, 1.5, plane="H")
    k = 2 * np.pi * f / cst.c
    assert any((n * np.pi / H) ** 2 > k * k for n in b.families())
    G = line.gsm(f, cutoff_factor=1.5)
    assert rel_reciprocity(G) < 1e-10 and G.unitarity_defect() < 1e-10
    G1 = line.gsm(f)
    ip = [b.index(m) for m in G1.basis1.modes]
    assert np.max(np.abs(G.S21[np.ix_(ip, ip)] - G1.S21)) < 1e-3       # evanescent modes barely matter here


def test_convergence_study_runs():
    m = Miter(90, guide=WR90)
    out = convergence_study(m, 10e9, mesh_densities=(4, 6, 10), n_evanescent=(2, 6, 10))
    mesh = [x[2] for x in out["mesh"]]
    assert mesh[-1] == 0 and mesh[0] < 1e-3
    assert out["reciprocity"] < 1e-10


def test_core_imports_without_fem_extra():
    """`import wrmodes` must not import scikit-fem or gmsh (CONVENTIONS.md section 10)."""
    code = "import sys, wrmodes; print('skfem' in sys.modules, 'gmsh' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "False False"
