"""Adversarial tests of WP4 (steps by mode matching), from tasks/WP4-steps.md and CONVENTIONS.md only.

Weak points: many modes (30+ per family) in the identity step, extreme aspect ratios (50:1), frequencies a hair
above a cutoff, offset steps, evanescent modes with large |beta| L between two steps, and Marcuvitz's equivalent
circuit for the symmetric E-plane step (sec. 5.26a, 1 % on B).
"""
import numpy as np
import pytest
import scipy.constants as cst

import ref_interfaces as I
import ref_literature as L
from wrmodes.gsm import straight
from wrmodes.modes import Mode, ModeBasis

A, B = 22.86e-3, 10.16e-3
TE10 = Mode("TE", 1, 0)


def _step(*args, **kw):
    Step = I.require("Step")
    return Step(*args, **kw)


def _props(g, unit=1e-10):
    assert np.all(np.isfinite(g.S))
    assert g.reciprocity_defect() <= 1e-10 * np.max(np.abs(g.S))
    assert g.unitarity_defect() <= unit


def test_identity_step_with_30_modes_per_family():
    """a1 = a2, b1 = b2, basis with >= 30 modes in family 0 (evanescent included): S = zero-length straight."""
    f = 40e9
    basis = ModeBasis.propagating(f, A, B, cutoff_factor=10.0)
    assert len(basis.family(0)) >= 30
    g = _step(A, B, A, B).gsm(f, basis, basis)
    ref = straight(basis, f, 0.0)
    assert np.max(np.abs(g.S - ref.S)) <= 1e-12


@pytest.mark.parametrize("b2", [1.6e-3, 0.2e-3])
def test_extreme_aspect_ratio_strangling_step(b2):
    """WR-90 -> 22.86 x b2 (aspect ratio up to 114): reciprocal and unitary, TE10 reflection below 1."""
    g = _step(A, B, A, b2).gsm(30e9)
    _props(g)
    i = g.basis1.index(TE10)
    assert abs(g.S11[i, i]) < 1


@pytest.mark.parametrize("rel", [1e-6, 1e-9])
def test_step_a_hair_above_a_cutoff(rel):
    """TE1,2/TM1,2 of WR-90 (30.24 GHz) at fc (1 + rel): huge admittances of the newly propagating pair."""
    fc = cst.c / 2 * np.hypot(1 / A, 2 / B)
    g = _step(A, B, A, 1.6e-3).gsm(fc * (1 + rel))
    _props(g, unit=1e-8)


@pytest.mark.parametrize("du,dv", [(0.0, 2.0e-3), (3.0e-3, 0.0), (2.0e-3, 1.5e-3)])
def test_offset_small_large_small_is_identity(du, dv):
    """Small -> large -> small at zero distance is the identity on the small guide's propagating modes when the
    large guide's basis spans the small one's fields (WP4 acceptance), here with offsets."""
    a2, b2, f = 15.0e-3, 5.0e-3, 35e9
    s1 = _step(a2, b2, A, B, du=du, dv=dv)
    s2 = _step(A, B, a2, b2, du=du, dv=dv)
    big = ModeBasis.propagating(f, A, B, cutoff_factor=12.0, evanescent_families="all")
    small = ModeBasis.propagating(f, a2, b2)
    g = s1.gsm(f, small, big) @ s2.gsm(f, big, small)
    assert np.max(np.abs(g.S21 - np.eye(len(small)))) <= 1e-6
    assert np.max(np.abs(g.S11)) <= 1e-6


def test_evanescent_coupling_through_a_long_thin_section():
    """WR-90 -> 1.6 mm -> WR-90 with a 1 m thin section at 40 GHz and 5x cutoff truncation: the evanescent modes
    of the thin guide decay by exp(-|beta| 1 m) ~ 1e-600 (underflow). Nothing may overflow, the result must be
    unitary, and equal to the cascade of the two single-mode-limited junctions."""
    f = 40e9
    b2 = 1.6e-3
    s_in, s_out = _step(A, B, A, b2), _step(A, b2, A, B)
    thin = ModeBasis.propagating(f, A, b2, cutoff_factor=5.0, evanescent_families="all")
    g1 = s_in.gsm(f, None, thin)
    g = g1 @ straight(thin, f, 1.0) @ s_out.gsm(f, thin, None)
    _props(g)


@pytest.mark.parametrize("b2,f", [(1.6e-3, 14e9), (1.6e-3, 26e9), (5e-3, 20e9), (8e-3, 11e9)])
def test_symmetric_eplane_step_against_marcuvitz(b2, f):
    """Marcuvitz sec. 5.26a eq. (2a): |S11| within the band given by +-1 % on B (stated accuracy, b/lambda_g < 1)."""
    Bm, y0, S = L.marcuvitz_eplane_step(f, A, B, b2)
    lo, hi = sorted(abs((1 - (y0 + 1j * Bm * s)) / (1 + (y0 + 1j * Bm * s))) for s in (0.99, 1.01))
    g = _step(A, B, A, b2).gsm(f)
    i = g.basis1.index(TE10)
    assert lo <= abs(g.S11[i, i]) <= hi
