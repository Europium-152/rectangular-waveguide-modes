"""Adversarial tests of WP2 (ohmic loss), written from tasks/WP2-loss-and-filters.md and CONVENTIONS.md only.

Reference: the WP7 surface-integral loss matrix (ref_loss) for alpha, and the WP7 first-order coupled system
(ref_loss.first_order_matrices / straight_gsm_blocks) for lossy straights. Weak points: many modes (129 at 90 GHz), modes just
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


TOL_S = 1e-8     # implementation agreement with the WP7 first-order coupled reference (same theory, absolute on S)


def _straight_vs_coupled_reference(g, a, b, f, L, sigma):
    """max |S - S_ref| over S11 and S21 (propagating modes), S_ref from ref_loss.straight_gsm_blocks: the full
    first-order coupled system (every propagating pair through the walls, forward and backward blocks), solved by its
    eigenmodes. Resolution of docs/discrepancies/2026-11-08-wp2-wp7-straight-attenuation.md."""
    basis = ModeBasis.propagating(f, a, b)
    Zs = (1 + 1j) * ref_loss.surface_resistance(f, sigma)
    Lp, Lm = ref_loss.first_order_matrices(basis, f, Zs)
    R, T = ref_loss.straight_gsm_blocks(basis, f, Lp, Lm, L)
    idx = np.array([g.basis1.index(m) for m in basis.modes])
    jdx = np.array([g.basis2.index(m) for m in basis.modes])
    return max(np.max(np.abs(g.S21[np.ix_(jdx, idx)] - T)), np.max(np.abs(g.S11[np.ix_(idx, idx)] - R)))


def test_square_guide_degenerate_groups_in_a_lossy_straight():
    """20 mm square guide at 30 GHz, 1 m: four-fold degenerate TE1,2/TE2,1/TM1,2/TM2,1 and TE0,4/TE4,0 at
    fc/f = 0.9993, where the forward-backward wall coupling changes the decay by 2.6 %."""
    a = b = 0.02
    f, L = 30e9, 1.0
    g = _lossy_line([ref_line.straight(L)], a, b, ref_loss.SIGMA_AL).gsm(f, cutoff_factor=1.0)
    assert _straight_vs_coupled_reference(g, a, b, f, L, ref_loss.SIGMA_AL) <= TOL_S


def test_20m_lossy_straight_at_90GHz():
    """20 m of aluminium WR-90 at 90 GHz, 129 modes: TE10 loses 4.6 dB; modes near cutoff lose hundreds of dB."""
    f, L = 90e9, 20.0
    g = _lossy_line([ref_line.straight(L)], *WR90, ref_loss.SIGMA_AL).gsm(f, cutoff_factor=1.0)
    assert np.all(np.isfinite(g.S))
    assert _straight_vs_coupled_reference(g, *WR90, f, L, ref_loss.SIGMA_AL) <= TOL_S


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
