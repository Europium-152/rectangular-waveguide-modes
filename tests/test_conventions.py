"""Checks of the mode-field convention of CONVENTIONS.md (wrmodes.modes).

The central test re-derives the curvature coupling matrix from the field definitions alone (the general
coupled-mode equations of docs/theory/T0-field-convention.md, evaluated by quadrature) and compares it with
the closed-form matrix of wrmodes 0.1. Agreement, signs included, shows that e_TM = grad Pi and
e_TE = grad Gamma x z_hat are the fields behind the existing solver, so every new component can be written
against them.
"""
import numpy as np
import pytest
import scipy.constants as cst

from wrmodes.modes import (Mode, ModeBasis, transverse_field, divergence, propagation_constant,
                           wave_admittance, ls_rotation, cutoff_wavenumber, ETA0)
from wrmodes.transfer_matrix import coupled_mode_matrices

WR90 = (0.9 * 0.0254, 0.4 * 0.0254)


def grid(a, b, nq=64):
    x, w = np.polynomial.legendre.leggauss(nq)
    u, wu = x * a / 2, w * a / 2
    v, wv = x * b / 2, w * b / 2
    U, V = np.meshgrid(u, v, indexing="ij")
    W = np.outer(wu, wv)
    return U, V, W


def fields(basis, U, V):
    E = [np.array(transverse_field(m, basis.a, basis.b, U, V)) for m in basis]
    Dv = [np.array(divergence(m, basis.a, basis.b, U, V)) for m in basis]
    return E, Dv


@pytest.mark.parametrize("a,b", [WR90, WR90[::-1], (0.044, 0.030), (0.02, 0.02)])
def test_fields_are_orthonormal(a, b):
    basis = ModeBasis.below(2.5 * cst.c / (2 * min(a, b)), a, b)
    U, V, W = grid(a, b)
    E, _ = fields(basis, U, V)
    G = np.array([[np.sum(W * (ei * ej).sum(0)) for ej in E] for ei in E])
    assert np.allclose(G, np.eye(len(basis)), atol=1e-12)


def derived_coupling(basis, f):
    """Curvature coupling matrices C+ and C- (paper convention, without the factor i) from the fields.

    From Maxwell's equations in the bend coordinates (h3 = 1 + kappa u), with V = a sqrt(2/Y) and
    I = a sqrt(2Y) (docs/theory/T0-field-convention.md):
        C+- = 1/2 [ +- sqrt(Yp Yq) (w mu U - Ud / (w eps)) + (w eps U - Ug / (w mu)) / sqrt(Yp Yq) ],
    with U = int u e_p.e_q, Ud = int u div(e_p) div(e_q), Ug = int u div(z x e_p) div(z x e_q).
    """
    a, b = basis.a, basis.b
    U, V, W = grid(a, b)
    E, Dv = fields(basis, U, V)
    N = len(basis)
    Uu = np.array([[np.sum(W * U * (E[p] * E[q]).sum(0)) for q in range(N)] for p in range(N)])
    Ud = np.array([[np.sum(W * U * Dv[p][0] * Dv[q][0]) for q in range(N)] for p in range(N)])
    Ug = np.array([[np.sum(W * U * Dv[p][1] * Dv[q][1]) for q in range(N)] for p in range(N)])
    k = 2 * np.pi * f / cst.c
    wmu, weps = k * ETA0, k / ETA0                 # omega mu0, omega eps0
    Y = basis.admittance(f)
    sY = np.sqrt(np.outer(Y, Y))
    A = sY * (wmu * Uu - Ud / weps)
    B = (weps * Uu - Ug / wmu) / sY
    return 0.5 * (A + B), 0.5 * (-A + B)


@pytest.mark.parametrize("a,b,n,f", [(*WR90, 0, 40e9), (*WR90[::-1], 1, 60e9), (*WR90, 1, 70e9),
                                     (0.044, 0.030, 2, 30e9)])
def test_closed_form_coupling_follows_from_the_field_convention(a, b, n, f):
    basis = ModeBasis.propagating(f, a, b, families=[n])
    assert len(basis) >= 3
    Cp, Cm = derived_coupling(basis, f)
    D, K, beta = coupled_mode_matrices(f, a, b, basis.legacy_modes())
    N = len(basis)
    # K = [[-Cp, -Cm], [Cm, Cp]] with the factor i included (wrmodes.mode_coupling.coupling_coefficient)
    scale = np.max(np.abs(K))
    assert np.max(np.abs(K[:N, :N] + 1j * Cp)) < 1e-10 * scale
    assert np.max(np.abs(K[:N, N:] + 1j * Cm)) < 1e-10 * scale
    # backward equations: da-/dz = +i beta a- + i kappa sum (C- a+ + C+ a-)
    assert np.max(np.abs(K[N:, :N] - 1j * Cm)) < 1e-10 * scale
    assert np.max(np.abs(K[N:, N:] - 1j * Cp)) < 1e-10 * scale
    assert np.allclose(beta, basis.beta(f))


def test_te_tm_sign_matters():
    """Flipping the sign of the TM fields changes the TE-TM couplings: the test above is sign-sensitive."""
    a, b, f = WR90[1], WR90[0], 60e9
    basis = ModeBasis.propagating(f, a, b, families=[1])
    Cp, _ = derived_coupling(basis, f)
    flip = np.array([-1.0 if m.kind == "TM" else 1.0 for m in basis])
    D, K, _ = coupled_mode_matrices(f, a, b, basis.legacy_modes())
    N = len(basis)
    flipped = flip[:, None] * Cp * flip[None, :]
    assert np.max(np.abs(K[:N, :N] + 1j * flipped)) > 1e-2 * np.max(np.abs(K))


def test_propagation_constant_branch():
    a, b = WR90
    te10, te01 = Mode("TE", 1, 0), Mode("TE", 0, 1)
    f = 10e9                                   # TE10 propagates, TE01 (cutoff 14.76 GHz) does not
    assert propagation_constant(te10, a, b, f).real > 0 and propagation_constant(te10, a, b, f).imag == 0
    beta = propagation_constant(te01, a, b, f)
    assert beta.real == 0 and beta.imag < 0    # exp(-i beta z) decays in +z
    assert np.isclose(abs(beta) ** 2, cutoff_wavenumber(te01, a, b) ** 2 - (2 * np.pi * f / cst.c) ** 2)
    y = wave_admittance(te10, a, b, f)
    assert np.isclose(y, propagation_constant(te10, a, b, f) / (2 * np.pi * f * cst.mu_0), rtol=1e-9)


def test_basis_order_and_legacy_compatibility():
    from wrmodes.transfer_matrix import select_coupled_modes
    a, b = WR90
    f = 60e9
    for n in (0, 1, 2):
        basis = ModeBasis.propagating(f, a, b, families=[n])
        legacy = select_coupled_modes(f, a, b, n=n)
        assert basis.names == [m[0] for m in legacy]
    full = ModeBasis.propagating(f, a, b)
    fc = full.cutoff_frequencies()
    assert np.all(np.diff(fc) >= -1e-6)
    assert full[0] == Mode("TE", 1, 0)


@pytest.mark.parametrize("f", [40e9, 75e9, 12e9])
def test_ls_rotation(f):
    """LSE modes have E_v = 0 and LSM modes H_v = 0, field by field, and Q is complex-orthogonal."""
    a, b = WR90
    basis = ModeBasis.propagating(max(f, 40e9), a, b, cutoff_factor=1.0)
    Q, labels = ls_rotation(basis, f)
    assert np.allclose(Q @ Q.T, np.eye(len(basis)), atol=1e-12)
    U, V, W = grid(a, b, 32)
    Y = basis.admittance(f)
    e = [np.array(transverse_field(m, a, b, U, V)) for m in basis]
    for i, (kind, m, n) in enumerate(labels):
        coeff = Q[i]                                   # TE/TM content of the LS mode (Q is orthogonal)
        Ev = sum(c * np.sqrt(2 / y) * ei[1] for c, y, ei in zip(coeff, Y, e))
        # z_hat x e = (-e_v, e_u): its v component is e_u
        Hv = sum(c * np.sqrt(2 * y) * ei[0] for c, y, ei in zip(coeff, Y, e))
        scale = max(np.max(np.abs(Ev)), np.max(np.abs(Hv)))
        if kind == "LSE":
            assert np.max(np.abs(Ev)) < 1e-12 * scale, (kind, m, n)
        else:
            assert np.max(np.abs(Hv)) < 1e-12 * scale, (kind, m, n)


def test_eplane_frame_signs_from_the_fields():
    """Global (H-plane) frame -> E-plane local frame (u = y, v = -x): the signed permutation of CONVENTIONS.md 2."""
    from wrmodes.modes import eplane_frame_signs
    w, h = WR90
    gb = ModeBasis.below(70e9, w, h)
    lb = ModeBasis.below(70e9, h, w)
    F = eplane_frame_signs(gb, lb)
    x, wt = np.polynomial.legendre.leggauss(48)
    X, Y = np.meshgrid(x * w / 2, x * h / 2, indexing="ij")
    W = np.outer(wt * w / 2, wt * h / 2)
    G = np.zeros_like(F)
    for i, loc in enumerate(lb):
        eu, ev = transverse_field(loc, h, w, Y, -X)
        lx, ly = -ev, eu                                  # u_hat = y_hat, v_hat = -x_hat
        for j, g in enumerate(gb):
            ex, ey = transverse_field(g, w, h, X, Y)
            G[i, j] = np.sum(W * (ex * lx + ey * ly))
    assert np.allclose(G, F, atol=1e-12)


def test_ls_rotation_refuses_the_singular_point():
    a, b = WR90
    f_sing = cst.c / (2 * b)                             # k = pi / b: k_t = 0 for family n = 1
    basis = ModeBasis.below(1.6 * f_sing, a, b, families=[1])
    with pytest.raises(ValueError):
        ls_rotation(basis, f_sing)
    Q, _ = ls_rotation(basis, 1.01 * f_sing)            # close to it: valid, complex-orthogonal
    assert np.allclose(Q @ Q.T, np.eye(len(basis)), atol=1e-10)


def test_propagating_basis_adds_evanescent_modes_only_to_live_families():
    a, b = WR90
    f = 10e9                                             # only TE1,0 propagates
    basis = ModeBasis.propagating(f, a, b, cutoff_factor=3.0)
    assert basis.families() == [0]
    full = ModeBasis.propagating(f, a, b, cutoff_factor=3.0, evanescent_families="all")
    assert len(full.families()) > 1


def test_cutoff_classification_and_legacy_guard():
    a, b = WR90
    te20 = Mode("TE", 2, 0)
    fc = cutoff_wavenumber(te20, a, b) * cst.c / (2 * np.pi)
    basis = ModeBasis.below(40e9, a, b)
    for f in (np.nextafter(fc, 0), fc, np.nextafter(fc, np.inf), 1.0001 * fc):
        mask = basis.propagating_mask(f)
        prop = ModeBasis.propagating(f, a, b)
        assert [m for m, p in zip(basis, mask) if p] == list(prop.modes)
    basis = ModeBasis.propagating(20e9, a, b, cutoff_factor=1.5)
    with pytest.raises(ValueError):
        basis.legacy_modes(20e9)
    with pytest.warns(RuntimeWarning):
        coupled_mode_matrices(20e9, a, b, basis.legacy_modes())
