"""Adversarial tests of WP2 (ohmic loss), written from tasks/WP2-loss-and-filters.md and CONVENTIONS.md only.

Reference: the WP7 surface-integral loss matrix (ref_loss). Weak points: many modes (129 at 90 GHz), modes just
above cutoff (alpha ~ 1/sqrt(1 - fc^2/f^2)), degenerate TE/TM pairs and TE_mn/TE_nm in square guides, very long
lossy straights, passivity of lossy bends, and the lossless limit.
"""
import numpy as np
import pytest
import scipy.constants as cst

import ref_interfaces as I
import ref_line
import ref_loss
from wrmodes.modes import Mode, ModeBasis

WR90 = ref_line.WR90
TOL = 1e-6


def _alpha_fn():
    try:
        return I.attenuation_function()[1]
    except I.InterfaceMissing as exc:
        pytest.skip(f"WP7: WP2 attenuation function not available yet: {exc}")


def _lossy_line(segs, w, h, sigma):
    I.require("Guide", "Straight", "SmoothBend", "Line")
    line = I.build_line(segs, w, h, sigma=sigma, atol=1e-10)
    try:                                           # WP1's Guide accepts sigma but raises until WP2 lands: skip
        I.build_line([ref_line.straight(1e-3)], w, h, sigma=sigma, atol=1e-10).gsm(1.2 * cst.c / (2 * w))
    except NotImplementedError as exc:
        pytest.skip(f"WP7: ohmic loss not implemented yet (WP2): {exc}")
    return line



@pytest.mark.parametrize("sigma", [ref_loss.SIGMA_AL, ref_loss.SIGMA_CU])
def test_alpha_of_all_129_modes_at_90GHz(sigma):
    fn = _alpha_fn()
    f = 90e9
    basis = ModeBasis.propagating(f, *WR90)
    A, _ = ref_loss.loss_matrix(basis, f, sigma)
    got = np.array([float(np.real(fn(m, *WR90, f, sigma))) for m in basis.modes])
    assert np.max(np.abs(got / np.diag(A) - 1)) <= TOL


@pytest.mark.parametrize("rel", [1e-2, 1e-5])
def test_alpha_just_above_cutoff(rel):
    """TM3,1 at f = fc (1 + rel): alpha grows like (1 - fc^2/f^2)^-1/2 (about 160 x at rel = 1e-5)."""
    fn = _alpha_fn()
    a, b = WR90
    m = Mode("TM", 3, 1)
    fc = cst.c / 2 * np.hypot(3 / a, 1 / b)
    f = fc * (1 + rel)
    ref = ref_loss.alpha(m, a, b, f, ref_loss.SIGMA_AL)
    assert abs(float(np.real(fn(m, a, b, f, ref_loss.SIGMA_AL))) / ref - 1) <= TOL


def _group_alpha_check(g, A, basis, f, L):
    from compare_wp2_loss import degenerate_groups
    idx = np.array([g.basis1.index(m) for m in basis.modes])
    S21 = g.S21[np.ix_(idx, idx)]
    worst = 0.0
    for grp in degenerate_groups(basis, f):
        lam = np.sort(np.linalg.eigvalsh(A[np.ix_(grp, grp)]))
        sv = np.linalg.svd(S21[np.ix_(grp, grp)], compute_uv=False)
        worst = max(worst, np.max(np.abs(np.sort(-np.log(sv) / L) / lam - 1)))
    return worst


def test_square_guide_degenerate_groups_in_a_lossy_straight():
    """20 mm square guide at 30 GHz: TE1,2/TE2,1/TM1,2/TM2,1 are four-fold degenerate. The eigen-attenuations of
    each degenerate group must match the wall-integral matrix (whatever basis WP2 diagonalizes in)."""
    a = b = 0.02
    f, L = 30e9, 1.0
    g = _lossy_line([ref_line.straight(L)], a, b, ref_loss.SIGMA_AL).gsm(f, cutoff_factor=1.0)
    basis = ModeBasis.propagating(f, a, b)
    A, _ = ref_loss.loss_matrix(basis, f, ref_loss.SIGMA_AL)
    assert _group_alpha_check(g, A, basis, f, L) <= TOL


def test_20m_lossy_straight_at_90GHz():
    """20 m of aluminium WR-90 at 90 GHz: TE10 loses 4.6 dB; alpha from |S21| to 1e-6 for every group."""
    f, L = 90e9, 20.0
    g = _lossy_line([ref_line.straight(L)], *WR90, ref_loss.SIGMA_AL).gsm(f, cutoff_factor=1.0)
    basis = ModeBasis.propagating(f, *WR90)
    A, _ = ref_loss.loss_matrix(basis, f, ref_loss.SIGMA_AL)
    assert np.all(np.isfinite(g.S))
    assert _group_alpha_check(g, A, basis, f, L) <= TOL


def test_lossy_bend_is_passive_and_reciprocal_and_lossless_limit_is_exact():
    segs = [ref_line.bend(0.6, ref_line.sech_curvature(60, 0.6), "H"), ref_line.straight(1.0),
            ref_line.bend(0.4, ref_line.sech_curvature(90, 0.4), "E")]
    f = 60e9
    g = _lossy_line(segs, *WR90, ref_loss.SIGMA_AL).gsm(f, cutoff_factor=1.0)
    assert g.passivity_margin() >= -1e-12
    assert g.reciprocity_defect() <= 1e-10 * np.max(np.abs(g.S))
    g0 = I.build_line(segs, *WR90, atol=1e-10).gsm(f, cutoff_factor=1.0)
    gn = _lossy_line(segs, *WR90, None).gsm(f, cutoff_factor=1.0)
    assert np.array_equal(g0.S, gn.S)                       # sigma=None: exactly the lossless result


def test_lossy_bend_power_balance_against_wall_integral():
    """Short gentle bend (losses first order): 1 - |T|^2 - |R|^2 for TE10 input equals the dissipated power, which
    to first order is 2 alpha L for TE10 (conversion is tiny) -- checks loss and curvature are not double counted."""
    f, L = 30e9, 0.3
    segs = [ref_line.bend(L, ref_line.constant_curvature(5, L), "H")]
    g = _lossy_line(segs, *WR90, ref_loss.SIGMA_AL).gsm(f, cutoff_factor=1.0)
    j = g.basis1.index(Mode("TE", 1, 0))
    diss = 1 - np.sum(np.abs(g.S21[:, j]) ** 2) - np.sum(np.abs(g.S11[:, j]) ** 2)
    alpha = ref_loss.alpha(("TE", 1, 0), *WR90, f, ref_loss.SIGMA_AL)
    exact = 1 - np.exp(-2 * alpha * L)
    assert abs(diss / exact - 1) < 2e-3                      # conversion to TE2,0 (lossier) is ~1e-5 here
