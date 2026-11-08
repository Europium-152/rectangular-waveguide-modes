"""Tests of WP2: ohmic wall loss (wrmodes.loss, lossy Straight / SmoothBend / Line) and the mode filters.

References: docs/theory/T1-wall-impedance.md (boxes B1-B10) with its machine-checkable form docs/theory/T1/t1lib.py,
and the independent derivation docs/theory/T1b (t1b_coefficients.py). Both are used here as references only.
"""
import os
import sys

import numpy as np
import pytest
import scipy.constants as cst
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

import wrmodes as wr
from wrmodes import loss
from wrmodes.components import Guide, Straight, SmoothBend, FrameChange, SechCurvature, ConstantCurvature, map_basis
from wrmodes.filters import LossyWallFilter, ResistiveSheetFilter, impedance_wall_modes, sheet_modes
from wrmodes.gsm import straight, from_transfer_matrix
from wrmodes.modes import Mode, ModeBasis, ETA0, eplane_frame_signs
from wrmodes.transfer_matrix import solve_transfer_matrix, coupled_mode_matrices

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("T1", "T1b"):
    p = os.path.join(HERE, "..", "docs", "theory", sub)
    if p not in sys.path:
        sys.path.insert(0, p)
import t1lib                      # noqa: E402  (T1 reference)
import t1b_coefficients as t1b    # noqa: E402  (T1b reference)

W, H = 22.86e-3, 10.16e-3
AL, CU = 3.5e7, 5.8e7
WR90 = Guide(W, H)
WR90_AL = Guide(W, H, sigma=AL)
DB = 20 / np.log(10)


def check_lossy(g, rtol=1e-10):
    """Reciprocity (relative) and passivity (CONVENTIONS.md section 6)."""
    assert g.reciprocity_defect() <= rtol * np.max(np.abs(g.S)), g.reciprocity_defect()
    assert g.passivity_margin() >= -1e-12, g.passivity_margin()


# ============================================================================ wrmodes.loss
def test_surface_resistance_and_no_default_material():
    f = 30e9
    assert np.isclose(loss.surface_resistance(f, CU), np.sqrt(2 * np.pi * f * cst.mu_0 / (2 * CU)), rtol=1e-15)
    assert loss.surface_impedance(f, AL) == (1 + 1j) * loss.surface_resistance(f, AL)
    with pytest.raises(TypeError):
        loss.surface_resistance(f)                       # sigma is explicit: no default material
    with pytest.raises(ValueError):
        loss.surface_resistance(f, -1.0)


@pytest.mark.parametrize("f", [18e9, 30e9, 45e9, 60e9, 75e9, 90e9])
def test_te10_attenuation_wr90_aluminium(f):
    """Card: textbook formula to 1e-6; 0.12 dB/m at 18 GHz, 0.23 dB/m at 90 GHz."""
    al = loss.attenuation_constant("TE1,0", W, H, f, AL)
    ref = t1lib.attenuation_te10_card(W, H, f, t1lib.surface_resistance(f, AL))
    assert abs(al / ref - 1) < 1e-12
    if f == 18e9:
        assert round(al * DB, 2) == 0.12
    if f == 90e9:
        assert round(al * DB, 2) == 0.23


@pytest.mark.parametrize("sigma", [AL, CU])
def test_attenuation_of_every_mode_against_marcuvitz(sigma):
    for f in (18e9, 54e9, 90e9):
        basis = ModeBasis.propagating(f, W, H)
        Rs = t1lib.surface_resistance(f, sigma)
        got = loss.attenuation_constants(basis, f, sigma)
        ref = np.array([t1lib.attenuation_marcuvitz(m, W, H, f, Rs) for m in basis.modes])
        refb = np.array([(t1b.alpha_te_marcuvitz if m.kind == "TE" else t1b.alpha_tm_marcuvitz)(m.m, m.n, W, H, f, Rs)
                         for m in basis.modes])
        assert np.max(np.abs(got / ref - 1)) < 1e-12
        assert np.max(np.abs(got / refb - 1)) < 1e-12
        lam, _ = loss.loss_matrices(basis, f, loss.surface_impedance(f, sigma))
        assert np.allclose(np.diag(lam), (1 + 1j) * got, rtol=1e-13, atol=0)   # beta grows by alpha


def test_attenuation_broadcasts_over_frequency():
    fs = np.array([20e9, 40e9, 80e9])
    al = loss.attenuation_constant("TM2,1", W, H, fs, AL)
    assert al.shape == (3,)
    assert np.allclose(al, [loss.attenuation_constant("TM2,1", W, H, f, AL) for f in fs], rtol=1e-15)


def test_wall_integrals_against_quadrature_of_the_mode_fields():
    basis = ModeBasis.below(80e9, W, H)
    for order in (0, 1):
        got = loss.wall_integrals(basis, order)
        for wall in loss.WALLS:
            Nq, Gq = t1lib.wall_matrices_quadrature(basis, wall, order, npts=300)
            Nw, Gw = got[wall]
            assert np.max(np.abs(Nw - Nq)) <= 1e-11 * np.max(np.abs(Nq))
            assert np.max(np.abs(Gw - Gq)) <= 1e-11 * np.max(np.abs(Gq))


def test_loss_matrices_against_t1_and_t1b_random_walls():
    rng = np.random.default_rng(7)
    for _ in range(6):
        a, b = rng.uniform(5e-3, 30e-3, 2)
        f = rng.uniform(15e9, 95e9)
        basis = ModeBasis.below(rng.uniform(1.05, 1.6) * f, a, b)
        if np.any(np.abs(basis.cutoff_frequencies() / f - 1) < 1e-6):
            continue
        Z = {w: complex(rng.uniform(0, 300), rng.uniform(-100, 100)) for w in loss.WALLS}
        for order in (0, 1):
            Lp, Lm = t1lib.loss_matrices(basis, f, Z, order)
            Mp, Mm = loss.loss_matrices(basis, f, Z, order)
            sc = np.max(np.abs(Lp))
            assert np.max(np.abs(Mp - Lp)) <= 1e-12 * sc and np.max(np.abs(Mm - Lm)) <= 1e-12 * sc
            assert np.max(np.abs(Mp - Mp.T)) <= 1e-14 * sc       # symmetric (reciprocity)
        modes = [(m.kind, m.m, m.n) for m in basis.modes]
        ref0 = t1b.lambda_full(modes, a, b, f, Z)
        ref1 = t1b.lambda1_full(modes, a, b, f, Z)
        L0, L1 = loss.bend_loss_matrices(basis, f, Z, restrict=False)
        assert np.max(np.abs(L0 - ref0)) <= 1e-12 * np.max(np.abs(ref0))
        assert np.max(np.abs(L1 - ref1)) <= 1e-12 * np.max(np.abs(ref1))


def test_coupling_rule_evanescent_modes_get_only_their_diagonal():
    f = 30e9
    basis = WR90_AL.basis(f, 2.0)
    Zs = loss.surface_impedance(f, AL)
    full = loss.loss_matrix_full(basis, f, Zs)
    raw = loss.loss_matrix_full(basis, f, Zs, restrict=False)
    N = len(basis)
    prop = basis.propagating_mask(f)
    ev = np.flatnonzero(~prop)
    assert len(ev) > 0
    for e in ev:
        for blk in (full[e], full[:, e], full[e + N], full[:, e + N]):
            nz = np.flatnonzero(blk)
            assert set(nz) <= {e, e + N}
        assert full[e, e] == raw[e, e] and full[e + N, e + N] == raw[e + N, e + N] and full[e, e + N] == 0
    pp = np.r_[prop, prop]
    assert np.array_equal(full[np.ix_(pp, pp)], raw[np.ix_(pp, pp)])


# ============================================================================ lossless limit
def test_lossless_limit_is_exactly_wrmodes_0_1_and_wp1():
    f = 40e9
    b = WR90.basis(f, 1.5)
    assert np.array_equal(Straight(2.0, guide=WR90).gsm(f, b).S, straight(b, f, 2.0).S)
    bend = SmoothBend(0.4, SechCurvature(90, 0.4), guide=WR90)
    g = bend.gsm(f, b)
    # the 0.1 solver with and without the new argument
    sub = b.subset(b.family(0))
    sub = sub.subset([i for i, p in enumerate(sub.propagating_mask(f)) if p])
    modes = sub.legacy_modes(f)
    kw = dict(coupled_modes=modes, atol=1e-6, error_control="full", profile=False)
    _, r0 = solve_transfer_matrix(f, W, H, ("TE", 1, 0), 0.4, SechCurvature(90, 0.4), **kw)
    _, r1 = solve_transfer_matrix(f, W, H, ("TE", 1, 0), 0.4, SechCurvature(90, 0.4), loss_matrix=None, **kw)
    assert np.array_equal(r0.transfer_matrix, r1.transfer_matrix)
    idx = [b.index(m) for m in sub.modes]
    assert np.array_equal(g.S21[np.ix_(idx, idx)], from_transfer_matrix(r0.transfer_matrix, sub, f).S21)
    # zero loss matrices give the same result to round-off
    Z0 = np.zeros((2 * len(modes),) * 2)
    _, r2 = solve_transfer_matrix(f, W, H, ("TE", 1, 0), 0.4, SechCurvature(90, 0.4), loss_matrix=(Z0, Z0), **kw)
    assert np.max(np.abs(r2.transfer_matrix - r0.transfer_matrix)) < 1e-13
    line = [Straight(1.0), SmoothBend(0.4, SechCurvature(90, 0.4), plane="E"), Straight(0.5)]
    assert np.array_equal(wr.Line(WR90, line).gsm(f).S, wr.Line(Guide(W, H, sigma=None), line).gsm(f).S)


# ============================================================================ lossy straights
def test_lossy_straight_equals_the_exponential_of_the_full_system():
    f, L = 45e9, 0.7
    b = WR90_AL.basis(f)
    lam = loss.loss_matrix_full(b, f, loss.surface_impedance(f, AL))
    D = np.diag(np.concatenate([-1j * b.beta(f), 1j * b.beta(f)]))
    ref = from_transfer_matrix(expm((D - lam) * L), b, f)
    g = Straight(L, guide=WR90_AL).gsm(f, b)
    assert np.max(np.abs(g.S - ref.S)) < 1e-12
    check_lossy(g)


@pytest.mark.parametrize("f", [18e9, 54e9, 90e9])
def test_20m_aluminium_straight(f):
    """Long line: finite, passive, reciprocal; degenerate TE_mn -> TM_mn conversion of a few per cent (T1 sec. 5)."""
    g = Straight(20.0, guide=WR90_AL).gsm(f)
    b = g.basis1
    assert np.all(np.isfinite(g.S))
    check_lossy(g)
    i = b.index("TE1,0")
    alpha = loss.attenuation_constant("TE1,0", W, H, f, AL)
    # the coupling to the other propagating modes raises alpha above the diagonal (Marcuvitz) value by up to 1.4e-3
    # at 90 GHz, towards the exact Leontovich value (test_coupled_attenuation_approaches_the_exact_leontovich_value)
    assert abs(-np.log(abs(g.S21[i, i])) / (alpha * 20.0) - 1) < 2e-3
    j, k = b.index("TE1,1"), b.index("TM1,1")
    frac = abs(g.S21[k, j]) ** 2 / np.sum(np.abs(g.S21[:, j]) ** 2)
    assert 0.01 < frac < 0.34


@pytest.mark.parametrize("f", [18e9, 54e9, 90e9])
def test_coupled_attenuation_approaches_the_exact_leontovich_value(f):
    """TE1,0 and TE0,1 with one pair of aluminium walls, where the exact Leontovich modes are known (T1 box B6): the
    eigenvalue of the coupled propagating system (the components' rule) is within 4e-4 of the exact attenuation
    (T1 sec. 5: 3.5e-4), while the diagonal (Marcuvitz) value misses it by up to 3.8e-3 (TE0,1, side walls,
    90 GHz) and 1.7e-3 (TE1,0, broad walls, 90 GHz): a genuine O(ka Z/eta0) correction of the overmoded guide."""
    k = 2 * np.pi * f / cst.c
    Zs = loss.surface_impedance(f, AL)
    B = ModeBasis.propagating(f, W, H)
    D = np.diag(np.r_[-1j * B.beta(f), 1j * B.beta(f)])
    for mode in ("TE1,0", "TE0,1"):
        i = B.index(mode)
        b0 = B.beta(f)[i].real
        for walls in ("u", "v"):
            Lf = loss.loss_matrix_full(B, f, {walls + "+": Zs, walls + "-": Zs})
            ev = np.linalg.eigvals(D - Lf)
            lam = -(ev[np.argmin(np.abs(ev + 1j * b0))] + 1j * b0)
            # exact: LS^u modes; for the v walls in the frame rotated by 90 degrees (h x w guide, modes (n, m))
            kind = {("TE1,0", "u"): "LSE", ("TE1,0", "v"): "LSM", ("TE0,1", "u"): "LSM", ("TE0,1", "v"): "LSE"}
            a_, kt = (W, (np.pi / H) if mode == "TE0,1" else 0.0) if walls == "u" else \
                (H, (np.pi / W) if mode == "TE1,0" else 0.0)
            g = np.sqrt(k * k - kt ** 2 - impedance_wall_modes(f, a_, Zs, Zs, kind[(mode, walls)], 2)[0])
            exact = -g.imag
            assert abs(lam.real / exact - 1) < 4e-4, (mode, walls)
            if f == 90e9 and (mode, walls) in (("TE1,0", "v"), ("TE0,1", "u")):
                assert abs(Lf[i, i].real / exact - 1) > 1.5e-3


def test_lossy_straight_in_the_E_frame_equals_the_H_frame():
    """Uniform-sigma straights keep plane = None: same GSM in both frames, degenerate and cross-family coupling
    included (broad-wall coupling in one frame is side-wall coupling in the other)."""
    f, L = 60e9, 3.0
    s = Straight(L, guide=WR90_AL)
    assert s.plane is None
    bh = WR90_AL.basis(f, 1.3)
    be = map_basis(bh, "H", "E")
    gh, ge = s.gsm(f, bh), s.gsm(f, be)
    F = eplane_frame_signs(bh, be)
    for blk_h, blk_e in zip((gh.S11, gh.S12, gh.S21, gh.S22), (ge.S11, ge.S12, ge.S21, ge.S22)):
        assert np.max(np.abs(F.T @ blk_e @ F - blk_h)) < 1e-12
    j, k = bh.index("TE2,1"), bh.index("TM2,1")
    assert abs(gh.S21[k, j]) > 1e-2                       # the degenerate coupling is there
    j2 = bh.index("TE1,2")                                # cross-family (broad walls) coupling is there
    assert abs(gh.S21[bh.index("TE1,0"), j2]) > 0


def test_lossy_line_of_straights_and_bends():
    f = 54e9
    line = wr.Line(WR90_AL, [Straight(1.0), SmoothBend(0.4, SechCurvature(90, 0.4), plane="H"), Straight(2.0),
                             SmoothBend(0.4, SechCurvature(90, 0.4), plane="E"), Straight(1.0)])
    g = line.gsm(f)
    check_lossy(g)
    g15 = line.gsm(f, cutoff_factor=1.5)
    check_lossy(g15, rtol=1e-9)
    p = g15.basis1.propagating_mask(f)
    assert g15.basis1.subset(np.flatnonzero(p)) == g.basis1
    i = g.basis1.index("TE1,0")
    # the evanescent modes of the straights do not touch the propagating block (they are uncoupled there)
    assert abs(abs(g.S21[i, i]) - abs(g15.S21[np.ix_(p, p)][i, i])) < 1e-9
    g0 = wr.Line(WR90, [Straight(1.0), SmoothBend(0.4, SechCurvature(90, 0.4), plane="H"), Straight(2.0),
                        SmoothBend(0.4, SechCurvature(90, 0.4), plane="E"), Straight(1.0)]).gsm(f)
    alpha = loss.attenuation_constant("TE1,0", W, H, f, AL)
    assert abs(g0.S21[i, i]) > abs(g.S21[i, i]) > abs(g0.S21[i, i]) * np.exp(-alpha * 6.0)


# ============================================================================ lossy bends
def test_constant_curvature_lossy_bend_is_exact():
    """For constant kappa the Magnus product is exact: equals expm of (D - L0 + kappa (K - L1)) L."""
    f, L = 40e9, 0.6
    kappa = np.radians(60) / L
    bend = SmoothBend(L, ConstantCurvature(kappa), guide=WR90_AL)
    b = WR90_AL.basis(f)
    g = bend.gsm(f, b)
    for n in b.families():
        idx = [i for i in b.family(n) if b.propagating_mask(f)[i]]
        sub = b.subset(idx)
        D, K, _ = coupled_mode_matrices(f, W, H, sub.legacy_modes(f))
        L0, L1 = loss.bend_loss_matrices(sub, f, loss.surface_impedance(f, AL))
        ref = from_transfer_matrix(expm((D - L0 + kappa * (K - L1)) * L), sub, f)
        assert np.max(np.abs(g.S21[np.ix_(idx, idx)] - ref.S21)) < 1e-10
        assert np.max(np.abs(g.S11[np.ix_(idx, idx)] - ref.S11)) < 1e-10
    check_lossy(g)


def test_lossy_bend_adaptive_slicing_meets_its_tolerance():
    f = 60e9
    b = WR90_AL.basis(f)
    for plane in ("H", "E"):
        bb = WR90_AL.basis(f, plane=plane)
        g = SmoothBend(0.4, SechCurvature(90, 0.4), plane=plane, atol=1e-6, guide=WR90_AL).gsm(f, bb)
        ref = SmoothBend(0.4, SechCurvature(90, 0.4), plane=plane, atol=1e-11, max_slices=16384,
                         guide=WR90_AL).gsm(f, bb)
        assert all(v["success"] and v["error_estimate"] <= 1e-6 for v in g.info.values() if v)
        assert np.max(np.abs(g.S - ref.S)) <= 1e-6
        check_lossy(g)
    assert len(b) > 0


def test_kappa_lambda1_is_included():
    """The cross term kappa Lambda1 changes the bend result (T1 sec. 6), here at 90 GHz in the H plane."""
    f = 90e9
    b = WR90_AL.basis(f).subset(WR90_AL.basis(f).family(0))
    sub = b
    curv = SechCurvature(90, 0.4)
    D, K, _ = coupled_mode_matrices(f, W, H, sub.legacy_modes(f))
    L0, L1 = loss.bend_loss_matrices(sub, f, loss.surface_impedance(f, AL))
    kw = dict(coupled_modes=sub.legacy_modes(f), atol=1e-8, error_control="full", profile=False)
    _, r1 = solve_transfer_matrix(f, W, H, ("TE", 1, 0), 0.4, curv, loss_matrix=(L0, L1), **kw)
    _, r0 = solve_transfer_matrix(f, W, H, ("TE", 1, 0), 0.4, curv, loss_matrix=L0, **kw)
    g = SmoothBend(0.4, curv, atol=1e-8, guide=WR90_AL).gsm(f, sub)
    assert np.max(np.abs(g.S21 - r1.T)) < 1e-12
    dP = np.max(np.abs(np.abs(r1.T[:, 0]) ** 2 - np.abs(r0.T[:, 0]) ** 2))
    assert 1e-5 < dP < 1e-3                                  # T1 table: 1.3e-4
    in_loss = -DB * np.log(abs(r1.T[0, 0]))
    assert abs(in_loss - 2.4193) < 2e-4                       # T1 sec. 6 table (H, 90 GHz, 40 cm sech)


def test_straight_shortcut_of_a_bend_accounts_for_loss():
    """Below the TE2,0 cutoff the H-plane bend of TE1,0 has K = 0: a lossy straight over the centre line."""
    f, L = 11e9, 0.5
    b = WR90_AL.basis(f, 2.0)
    g = SmoothBend(L, SechCurvature(90, L), guide=WR90_AL).gsm(f, b)
    assert g.info[0] is None
    i = b.index("TE1,0")
    alpha = loss.attenuation_constant("TE1,0", W, H, f, AL)
    beta = b.beta(f)[i].real
    assert abs(g.S21[i, i] - np.exp(-(1j * beta + (1 + 1j) * alpha) * L)) < 1e-9      # Lambda^- adds ~1e-10
    sub = b.subset([i])
    lam = loss.loss_matrix_full(sub, f, loss.surface_impedance(f, AL))
    D = np.diag([-1j * beta, 1j * beta])
    ref = from_transfer_matrix(expm((D - lam) * L), sub, f)
    assert abs(g.S21[i, i] - ref.S21[0, 0]) < 1e-14 and abs(g.S11[i, i] - ref.S11[0, 0]) < 1e-14
    for e in np.flatnonzero(~b.propagating_mask(f)):   # evanescent modes: own attenuation added to their decay
        lam = loss.loss_matrices(b, f, loss.surface_impedance(f, AL))[0][e, e]
        assert abs(g.S21[e, e] - np.exp(-(1j * b.beta(f)[e] + lam) * L)) <= 1e-14 * abs(g.S21[e, e]) + 1e-300


def _dissipated_power(f, sub, res, curvature, Z):
    """Power into the walls, integral of (1/2) oint h Re(Z) (|H_t|^2 + |H_z|^2) dl dz, from the amplitude profile and
    the wall integrals by quadrature of the mode fields (t1lib, independent of wrmodes.loss)."""
    k = 2 * np.pi * f / cst.c
    N = len(sub)
    Y = sub.admittance(f).real
    z, y = res.x, res.y
    V = np.sqrt(2 / Y)[:, None] * (y[:N] + y[N:])
    I = np.sqrt(2 * Y)[:, None] * (y[:N] - y[N:])
    kap = curvature(z)
    p = np.zeros(len(z))
    for wall, Zw in Z.items():
        for order, wgt in ((0, 1.0), (1, kap)):
            Nw, Gw = t1lib.wall_matrices_quadrature(sub, wall, order, npts=200)
            p += 0.5 * np.real(Zw) * wgt * np.real(np.einsum("pz,pq,qz->z", I.conj(), Nw, I)
                                                    + np.einsum("pz,pq,qz->z", V.conj(), Gw, V) / (k * ETA0) ** 2)
    h = z[1] - z[0]
    return h / 3 * (p[0] + p[-1] + 4 * p[1:-1:2].sum() + 2 * p[2:-1:2].sum())     # Simpson


@pytest.mark.parametrize("plane_family", [("H", 0), ("E", 1)])
def test_lossy_bend_power_balance(plane_family):
    """Card: transmitted + reflected + dissipated = 1, the dissipation from the wall integral, to 1e-6."""
    plane, n = plane_family
    f, L = 30e9, 0.3
    curv = SechCurvature(90, L)
    a, b = WR90_AL.dims(plane)
    bas = WR90_AL.basis(f, plane=plane)
    sub = bas.subset([i for i in bas.family(n) if bas.propagating_mask(f)[i]])
    Zs = loss.surface_impedance(f, AL)
    L0, L1 = loss.bend_loss_matrices(sub, f, Zs)
    m0 = sub.modes[0]
    _, res = solve_transfer_matrix(f, a, b, (m0.kind, m0.m, m0.n), L, curv, coupled_modes=sub.legacy_modes(f),
                                   atol=1e-10, error_control="full", loss_matrix=(L0, L1),
                                   z_eval=np.linspace(0, L, 8001))
    j = res.input_index
    balance = 1 - np.sum(np.abs(res.T[:, j]) ** 2) - np.sum(np.abs(res.R[:, j]) ** 2)
    diss = _dissipated_power(f, sub, res, curv, loss.wall_impedances(Zs))
    assert balance > 1e-3
    assert abs(balance - diss) < 1e-6, (balance, diss)


def test_bend_families_separate_vs_coupled():
    """The default treats the local families separately; the neglected broad-wall coupling of families is small."""
    f = 30e9
    for plane in ("H", "E"):
        b = WR90_AL.basis(f, plane=plane)
        args = (0.4, SechCurvature(90, 0.4))
        g = SmoothBend(*args, plane=plane, guide=WR90_AL).gsm(f, b)
        gc = SmoothBend(*args, plane=plane, guide=WR90_AL, families="coupled").gsm(f, b)
        check_lossy(gc)
        dP = np.abs(np.abs(gc.S) ** 2 - np.abs(g.S) ** 2)
        assert dP.max() < 1e-5
        assert np.max(np.abs(gc.S - g.S)) < 1e-3


def test_steps_refuse_lossy_guides_clearly():
    step = wr.Step(W, H, 0.8 * W, H)
    with pytest.raises(NotImplementedError, match="perfectly conducting walls only"):
        wr.Line(WR90_AL, [Straight(0.1), step])
    wr.Line(WR90, [Straight(0.1), step])                     # lossless: fine
    with pytest.raises(ValueError):                           # a real mismatch stays a ValueError
        wr.Line(Guide(W, 0.9 * H, sigma=AL), [step])


# ============================================================================ lossy-wall filter
def test_impedance_wall_roots_against_t1():
    f = 75e9
    k = 2 * np.pi * f / cst.c
    for kind in ("LSE", "LSM"):
        for Zp, Zm in ((250.0, 250.0), (40 + 30j, 120.0), (0.0, 300.0)):
            s = impedance_wall_modes(f, W, Zp, Zm, kind, 40)
            ku = np.sqrt(s)
            ku = ku[np.argsort(ku.real)]                       # t1lib orders by Re k_u, wrmodes by Re k_u^2
            ref = t1lib.lsu_roots_all(40, k, W, Zp, Zm, kind)
            assert np.max(np.abs(ku[:36] - ref[:36]) / np.abs(ref[:36])) < 1e-12
            assert np.max(np.abs(t1lib.lsu_dispersion_scaled(ku, k, W, Zp, Zm, kind) / ku)) < 1e-6 * k ** 2


def test_butterweck_lossy_wall_filter():
    """X-band guide at 75 GHz, Zw = 250 ohm on both narrow walls, 15 cm (card; T1 section 7).

    Measured (Philips Tech. Rev. 29, p. 95): 0.6, 2.5, 5.4 dB for H10, H20, H30, within 10 %. H10 -> H30 conversion
    between -36 and -24.5 dB (Butterweck's band; his formula lacks a factor nu/mu, T1 section 7)."""
    f = 75e9
    basis = WR90.basis(f)
    g = LossyWallFilter(0.15, 250.0).gsm(f, basis)
    check_lossy(g, rtol=1e-12)
    t = [-DB * np.log(abs(g.transmission(f"TE{m},0"))) for m in (1, 2, 3)]
    for got, meas in zip(t, (0.6, 2.5, 5.4)):
        assert abs(got / meas - 1) < 0.10, t
    assert np.allclose(t, [0.5691, 2.3157, 5.3386], atol=2e-3)           # T1: 0.569 / 2.315 / 5.337 (160 modes)
    conv = DB * np.log(abs(g.transmission("TE1,0", "TE3,0")))
    assert -36 <= conv <= -24.5 and abs(conv + 33.55) < 0.05
    assert abs(g.transmission("TE0,1")) < 1                               # n >= 1: also attenuated (exactly)
    i3 = [basis.index(f"TE{m},0") for m in (1, 2, 3)]
    assert np.max(np.abs(g.S11[np.ix_(i3, i3)])) < 3.2e-3          # two junctions, each <= 1.6e-3 (T1 sec. 7)


def test_lossy_wall_filter_convergence_and_first_order_limit():
    f = 75e9
    b0 = ModeBasis.propagating(f, W, H, families=[0])
    res = {M: LossyWallFilter(0.15, 250.0, modes_per_family=M).gsm(f, b0) for M in (40, 80, 160)}
    t = {M: np.array([-DB * np.log(abs(g.transmission(f"TE{m},0"))) for m in (1, 2, 3)]) for M, g in res.items()}
    assert np.max(np.abs(t[160] - t[80])) < 3e-3 and np.max(np.abs(t[80] - t[40])) < 5e-3
    # small impedance: the exact filter approaches the first-order lossy straight (T1 B4) on the same walls
    Zw = 2.5
    ge = LossyWallFilter(0.15, Zw).gsm(f, b0)
    gf = loss.lossy_straight_gsm(b0, f, 0.15, {"u+": Zw, "u-": Zw})
    for m in (1, 2, 3):
        i = b0.index(f"TE{m},0")
        a_exact, a_first = -np.log(abs(ge.S21[i, i])), -np.log(abs(gf.S21[i, i]))
        assert abs(a_exact / a_first - 1) < 2e-3
    # Z = 0: the empty guide
    b = WR90.basis(f)
    assert np.max(np.abs(LossyWallFilter(0.15, 0.0).gsm(f, b).S - straight(b, f, 0.15).S)) < 1e-11


def test_lossy_wall_filter_frames():
    """v-walls in one frame are u-walls in the other: the H-plane filter on the broad walls equals the E-plane
    filter on its u-walls, after the frame change."""
    f = 40e9
    bh = WR90.basis(f)
    be = map_basis(bh, "H", "E")
    gv = LossyWallFilter(0.1, {"v+": 200.0, "v-": 150.0}, walls=("v-", "v+"), plane="H").gsm(f, bh)
    # local H 'v+' (y = +h/2) is E-frame u = +h/2 ('u+')
    gu = LossyWallFilter(0.1, {"u+": 200.0, "u-": 150.0}, plane="E").gsm(f, be)
    F = eplane_frame_signs(bh, be)
    assert np.max(np.abs(F.T @ gu.S21 @ F - gv.S21)) < 1e-12
    assert np.max(np.abs(F.T @ gu.S11 @ F - gv.S11)) < 1e-12
    check_lossy(gv)
    # in a Line, an E-plane filter gets its frame changes
    line = wr.Line(WR90, [Straight(0.1), LossyWallFilter(0.1, 250.0, plane="E"), Straight(0.1)])
    assert sum(line.inserted) == 2
    check_lossy(line.gsm(f))
    with pytest.raises(TypeError):
        LossyWallFilter(0.1)                                                  # Zw is required
    with pytest.raises(ValueError):
        LossyWallFilter(0.1, 250.0, walls=("u+", "v+"))


# ============================================================================ resistive-sheet filter
def test_sheet_dispersion_against_t1():
    f = 75e9
    k = 2 * np.pi * f / cst.c
    b = 10e-3
    for kind in ("LSM", "LSE"):
        for v0 in (0.5 * b, 0.3 * b):
            s = sheet_modes(f, b, 56.0, [v0], kind, 12)
            kv = np.sqrt(s)
            Fc = t1lib.sheet_dispersion_closed(kv, k, b, 56.0 / ETA0, v0, kind)
            scale = k * np.abs(np.sin(kv * b)) + np.abs(kv) + k
            assert np.max(np.abs(Fc) / scale) < 1e-9
        pos = [0.25 * b, 0.5 * b, 0.75 * b]
        s = sheet_modes(f, b, [70.0, 60.0, 80.0], pos, kind, 10)
        F = t1lib.sheet_dispersion(np.sqrt(s), k, b, np.array([70.0, 60.0, 80.0]) / ETA0, pos, kind)
        assert np.max(np.abs(F)) < 1e-7 * (1 + np.max(np.abs(np.sqrt(s))))
    s = sheet_modes(f, b, 56.0, [0.5 * b], "LSM", 3)
    assert abs(s[0]) < 1e-6                                  # TE_m0 untouched


def _eh11_alpha(f, b, R, kind):
    k = 2 * np.pi * f / cst.c
    s = sheet_modes(f, b, R, [b / 2], kind, 3)[1 if kind == "LSM" else 0]
    g = np.sqrt(k * k - (np.pi / W) ** 2 - s + 0j)
    return -(g.imag if g.imag < 0 else -g.imag)


@pytest.mark.parametrize("f", [75e9, cst.c / 4e-3])
def test_butterweck_sheet_optimum(f):
    """b = 10 mm, 75 GHz (lambda = 4 mm): optimum about 56 ohm, EH11^(2) about 1.5 dB/cm (theory values, 5 %).
    The dispersion is solved directly, so k_t = 0 of family n = 5 at exactly lambda = 4 mm does not enter."""
    b = 10e-3
    r = minimize_scalar(lambda x: -_eh11_alpha(f, b, np.exp(x), "LSM"), bounds=(np.log(10), np.log(1000)),
                        method="bounded", options={"xatol": 1e-7})
    R_opt, a_max = np.exp(r.x), -r.fun * DB / 100
    assert abs(R_opt / 56 - 1) < 0.05 and abs(a_max / 1.5 - 1) < 0.05
    assert abs(R_opt - 55.75) < 0.1 and abs(a_max - 1.554) < 2e-3          # T1 sec. 8 (lambda = 4 mm)
    r1 = minimize_scalar(lambda x: -_eh11_alpha(f, b, np.exp(x), "LSE"), bounds=(np.log(50), np.log(3000)),
                         method="bounded", options={"xatol": 1e-7})
    assert abs(np.exp(r1.x) / 420 - 1) < 0.05 and abs(-r1.fun * DB / 100 / 5 - 1) < 0.1   # his 420 ohm, ~5 dB/cm


def test_sheet_filter_convergence_and_properties():
    """10 cm, centred 56 ohm sheet, b = 10.16 mm, 75 GHz. Convergence in the number of modes per type is slow; the
    coordinator's check (T1/T1b) gave -12.72 / -12.89 / -12.97 dB for TE1,1 with 6 / 10 / 14 modes."""
    f = 75e9
    basis = WR90.basis(f)
    t = {}
    for Nn in (6, 10, 14, 20, 40):
        g = ResistiveSheetFilter(0.10, 56.0, modes_per_family=Nn).gsm(f, basis)
        check_lossy(g, rtol=1e-12)
        t[Nn] = -DB * np.log(abs(g.transmission("TE1,1")))
        assert abs(g.transmission("TE1,0")) > 1 - 1e-12                       # E normal to the sheet
        assert abs(g.transmission("TE1,2")) > 1 - 1e-12                       # node at the centred sheet
    assert np.allclose([t[6], t[10], t[14]], [12.72, 12.89, 12.97], atol=6e-3)
    assert abs(t[40] - t[20]) < 0.01
    g = ResistiveSheetFilter(0.10, 56.0).gsm(f, basis)
    assert abs(-DB * np.log(abs(g.transmission("TM1,1"))) - 16.56) < 0.02
    # very large sheet resistance: the empty guide
    g = ResistiveSheetFilter(0.10, 1e15).gsm(f, basis)
    assert np.max(np.abs(g.S - straight(basis, f, 0.10).S)) < 1e-9


def test_three_sheet_filter_in_a_line():
    """Butterweck's experimental filter: three equidistant 70 ohm sheets, 10 cm (Philips Tech. Rev. 29, p. 95)."""
    f = 75e9
    flt = ResistiveSheetFilter(0.10, 70.0, n_sheets=3)
    assert np.allclose(flt.sheet_positions(H), H * np.array([0.25, 0.5, 0.75]))
    line = wr.Line(WR90_AL, [Straight(0.5), flt, Straight(0.5)])
    g = line.gsm(f)
    check_lossy(g)
    b = g.basis1
    lossless = wr.Line(WR90_AL, [Straight(1.0)]).gsm(f)
    i = b.index("TE1,0")
    assert abs(abs(g.S21[i, i]) - abs(lossless.S21[i, i])) < 1e-6     # the sheets do not touch TE1,0
    for md in ("TE1,1", "TE1,2", "TM1,2", "TE0,1"):
        assert -DB * np.log(abs(g.transmission(md))) > 3.0
    # an off-centre sheet and its mirror image give the same attenuation
    g1 = ResistiveSheetFilter(0.05, 56.0, v0=0.2 * H).gsm(f, WR90.basis(f))
    g2 = ResistiveSheetFilter(0.05, 56.0, v0=-0.2 * H).gsm(f, WR90.basis(f))
    assert np.max(np.abs(np.abs(g1.S) - np.abs(g2.S))) < 1e-12


def test_sheet_filter_at_kt_zero():
    """Butterweck's b = 10 mm at lambda = 4 mm has k = 5 pi / b (k_t = 0 for n = 5, where ls_rotation is singular).
    The filter matches the TE/TM modes directly to the sheet modes, so no LS rotation of the empty guide enters, and
    the n = 5 sheet modes are moved away from k_v = k by the sheet. A sheet mode left exactly at k_v = k (a node at
    every sheet) is self-orthogonal and is refused with a clear message."""
    b = 10e-3
    f = cst.c / 4e-3
    g = ResistiveSheetFilter(0.10, 56.0).gsm(f, Guide(W, b).basis(f))
    check_lossy(g, rtol=1e-12)
    f6 = 6 * cst.c / (2 * b)              # k = 6 pi / b; sheets at b/3 and 2b/3 are nodes of n = 6
    with pytest.raises(ValueError, match="self-orthogonal"):
        ResistiveSheetFilter(0.10, 56.0, positions=[-b / 6, b / 6]).gsm(f6, Guide(W, b).basis(f6))


# ============================================================================ review fixes: surface waves
def _winding(F, R, npts=40000):
    """Number of zeros of the entire function F inside |s| = R (argument principle)."""
    th = np.linspace(0, 2 * np.pi, npts + 1)
    v = F(R * np.exp(1j * th))
    dphi = np.angle(v[1:] / v[:-1])
    assert np.max(np.abs(dphi)) < 1.0                      # sampled finely enough
    return int(round(np.sum(dphi) / (2 * np.pi)))


def _check_count(roots, F, exclude_zero=False):
    """All zeros of F with |s| < R must be among `roots` (the count lowest by Re s), for R < Re of the last root.
    exclude_zero: the root s = 0 is not a zero of F (removed by the division of F by k_t)."""
    if exclude_zero:
        roots = roots[np.abs(roots) > 1e-6 * np.max(np.abs(roots))]
    mags = np.sort(np.abs(roots))
    lim = 0.6 * roots[-1].real
    inner = mags[mags < lim]
    gaps = np.diff(inner)
    R = 0.5 * (inner[np.argmax(gaps)] + inner[np.argmax(gaps) + 1]) if len(gaps) else lim
    assert _winding(F, R) == np.sum(np.abs(roots) < R) > 0


@pytest.mark.parametrize("Z", [250 + 250j, 100 + 300j, 100 + 100j])
def test_inductive_walls_surface_waves(Z):
    """Inductive walls carry surface waves bound to each wall: an even/odd pair degenerate to about exp(-|Im k_u| a).
    They must all be found (argument principle on the B6 dispersion function of t1lib), and the filter must be
    passive (review of PR #16: family n = 1, 10 cm, 75 GHz gave singular values 1.545 and 1.169 before the fix)."""
    f = 75e9
    k = 2 * np.pi * f / cst.c
    sc = (np.pi / W) ** 2
    for kind in ("LSE", "LSM"):
        s = impedance_wall_modes(f, W, Z, Z, kind, 30)

        def F(x, kind=kind):
            ku = np.sqrt(x)
            return t1lib.lsu_dispersion_scaled(ku, k, W, Z, Z, kind) / ku
        _check_count(s, F)
        if Z == 250 + 250j and kind == "LSM":       # mpmath roots quoted by the reviewer: 115.2 i (pi/a)^2 (pair)
            assert np.sum(np.abs(s / sc - 115.2j) < 0.1) == 2
        if Z == 100 + 300j and kind == "LSM":       # -73.7 + 55.3 i (pi/a)^2 (pair)
            assert np.sum(np.abs(s / sc - (-73.7 + 55.3j)) < 0.1) == 2
    basis = WR90.basis(f)
    b1 = basis.subset(basis.family(1))
    g = LossyWallFilter(0.10, Z).gsm(f, b1)
    check_lossy(g, rtol=1e-12)
    g = LossyWallFilter(0.10, {"u+": Z, "u-": 0.5 * Z}).gsm(f, b1)      # unequal walls: one problem on [0, a]
    check_lossy(g, rtol=1e-12)
    for kind in ("LSE", "LSM"):
        s = impedance_wall_modes(f, W, Z, 0.5 * Z, kind, 30)

        def F(x, kind=kind):
            ku = np.sqrt(x)
            return t1lib.lsu_dispersion_scaled(ku, k, W, Z, 0.5 * Z, kind) / ku
        _check_count(s, F)


def test_nearly_equal_reactive_walls_raise_rather_than_return():
    """A pair of surface waves on nearly (not exactly) equal reactive walls is degenerate to round-off: the modes
    cannot be resolved, and the filter raises instead of returning a non-passive GSM."""
    from wrmodes.filters import RootFindingError
    f = 75e9
    basis = WR90.basis(f)
    b1 = basis.subset(basis.family(1))
    Z = 250 + 250j
    with pytest.raises(RootFindingError):
        LossyWallFilter(0.10, {"u+": Z, "u-": Z * (1 + 1e-9)}).gsm(f, b1)
    g = LossyWallFilter(0.10, {"u+": Z, "u-": Z * (1 + 1e-14)}).gsm(f, b1)     # treated as equal walls
    assert np.array_equal(g.S, LossyWallFilter(0.10, Z).gsm(f, b1).S)


@pytest.mark.parametrize("R", [20 + 150j, 56 - 300j, 5 + 500j])
def test_reactive_sheets(R):
    """Reactive sheets (complex R_sheet): all roots found (argument principle on the closed form B7 of t1lib) and the
    filter passive, for one centred sheet, one off-centre sheet and three equidistant sheets."""
    f = 75e9
    k = 2 * np.pi * f / cst.c
    xi = R / ETA0
    for v0 in (0.5 * H, 0.3 * H):
        for kind in ("LSM", "LSE"):
            s = sheet_modes(f, H, R, [v0], kind, 24)

            def F(x, kind=kind, v0=v0):
                kv = np.sqrt(x)
                val = t1lib.sheet_dispersion_closed(kv, k, H, xi, v0, kind)
                return val / kv if kind == "LSM" else val / x     # LSE: F has the trivial zero s = 0 (Y = 0)
            if kind == "LSM":           # s = 0 is an exact LSM^v root (TE_m0), divided out of F by kv
                assert np.min(np.abs(s)) < 1e-6 * (np.pi / H) ** 2
            _check_count(s, F, exclude_zero=(kind == "LSM"))
    basis = WR90.basis(f)
    for flt in (ResistiveSheetFilter(0.10, R), ResistiveSheetFilter(0.10, R, v0=0.2 * H),
                ResistiveSheetFilter(0.10, R, n_sheets=3)):
        check_lossy(flt.gsm(f, basis), rtol=1e-12)


def test_loss_matrix_argument_forms():
    """loss_matrix: an array or nested lists (Lambda0_full alone), or the pair (Lambda0_full, Lambda1_full)."""
    f, L = 40e9, 0.4
    sub = WR90_AL.basis(f).subset(WR90_AL.basis(f).family(0))
    L0, L1 = loss.bend_loss_matrices(sub, f, loss.surface_impedance(f, AL))
    kw = dict(coupled_modes=sub.legacy_modes(f), atol=1e-8, error_control="full", profile=False)
    curv = SechCurvature(90, L)
    run = lambda lm: solve_transfer_matrix(f, W, H, ("TE", 1, 0), L, curv, loss_matrix=lm, **kw)[1].transfer_matrix
    ref0, ref01 = run(L0), run((L0, L1))
    assert np.array_equal(run(L0.tolist()), ref0)               # nested lists are Lambda0, not a pair
    assert np.array_equal(run([L0, L1]), ref01) and np.array_equal(run((L0, None)), ref0)
    assert np.max(np.abs(ref0 - ref01)) > 0
    with pytest.raises(ValueError):
        run(L0[:-1, :-1])


def test_coupled_families_split_by_parity_of_n():
    """families='coupled' solves the even-n and odd-n modes separately (Lambda vanishes between them for equal
    walls); the result equals the single system of all propagating modes."""
    f = 30e9
    b = WR90_AL.basis(f)
    bend = SmoothBend(0.4, SechCurvature(90, 0.4), guide=WR90_AL, families="coupled")
    g = bend.gsm(f, b)
    assert set(g.info) == {"even", "odd"}
    L0, L1 = loss.bend_loss_matrices(b, f, loss.surface_impedance(f, AL))
    par = np.array([m.n % 2 for m in b.modes] * 2)
    assert not np.any(L0[np.not_equal.outer(par, par)]) and not np.any(L1[np.not_equal.outer(par, par)])
    idx = np.flatnonzero(b.propagating_mask(f))
    sel = np.r_[idx, idx + len(b)]
    gall, _ = bend._solve(b.subset(idx), f, (L0[np.ix_(sel, sel)], L1[np.ix_(sel, sel)]))
    assert np.max(np.abs(g.S - gall.S)) < 1e-12
