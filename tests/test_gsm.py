"""Tests of the GSM core (wrmodes.gsm): definitions, star product, conversion from transfer matrices."""
import numpy as np
import pytest
from scipy.linalg import expm

import wrmodes as wr
from wrmodes.gsm import GSM, cascade, straight, from_transfer_matrix, block_diag
from wrmodes.modes import ModeBasis
from wrmodes.transfer_matrix import coupled_mode_matrices

WR90 = (0.9 * 0.0254, 0.4 * 0.0254)
rng = np.random.default_rng(1234)


def random_reciprocal_lossless(basis, f=None):
    """A random symmetric unitary matrix (Takagi form U U^T with U unitary) as a GSM."""
    N = 2 * len(basis)
    X = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    U, _ = np.linalg.qr(X)
    return GSM.from_matrix(U @ U.T, basis, basis, f)


def random_reciprocal_lossy(basis, f=None, scale=0.8):
    g = random_reciprocal_lossless(basis, f)
    return GSM.from_matrix(scale * g.S, basis, basis, f)


def bend_gsm(f, a, b, n, kappa, L):
    basis = ModeBasis.propagating(f, a, b, families=[n])
    D, K, beta = coupled_mode_matrices(f, a, b, basis.legacy_modes())
    return from_transfer_matrix(expm((D + kappa * K) * L), basis, f, "bend"), expm((D + kappa * K) * L)


# ------------------------------------------------------------------------------------ definitions
def test_shapes_are_checked():
    basis = ModeBasis.propagating(40e9, *WR90)
    N = len(basis)
    with pytest.raises(ValueError):
        GSM(np.zeros((N, N)), np.zeros((N, N + 1)), np.zeros((N, N)), np.zeros((N, N)), basis)


def test_straight_section():
    f = 30e9
    basis = ModeBasis.propagating(f, *WR90, cutoff_factor=1.5)       # includes evanescent modes
    g = straight(basis, f, 0.05)
    beta = basis.beta(f)
    assert np.allclose(np.diag(g.S21), np.exp(-1j * beta * 0.05))
    ev = ~basis.propagating_mask(f)
    assert ev.any() and np.all(np.abs(np.diag(g.S21))[ev] < 1)       # evanescent modes decay
    assert g.reciprocity_defect() == 0 and g.unitarity_defect() < 1e-14


def test_star_with_straight_adds_lengths():
    f = 30e9
    basis = ModeBasis.propagating(f, *WR90, cutoff_factor=1.3)
    g = straight(basis, f, 0.1) @ straight(basis, f, 0.25)
    assert np.allclose(g.S, straight(basis, f, 0.35).S, atol=1e-14)


# ------------------------------------------------------------------------------------ star product
def test_star_matches_explicit_multiple_reflection_sum():
    """Two-component cascade against the explicit sum over round trips between the components."""
    basis = ModeBasis.propagating(40e9, *WR90)
    A, B = random_reciprocal_lossy(basis, scale=0.9), random_reciprocal_lossy(basis, scale=0.9)
    C = A @ B
    # forward transmission: B21 (sum_k (A22 B11)^k) A21, summed until converged
    M = A.S22 @ B.S11
    term, total = np.eye(len(basis)), np.zeros((len(basis),) * 2, complex)
    for _ in range(2000):
        total += term
        term = term @ M
        if np.max(np.abs(term)) < 1e-16:
            break
    assert np.allclose(C.S21, B.S21 @ total @ A.S21, atol=1e-12)
    assert np.allclose(C.S11, A.S11 + A.S12 @ B.S11 @ total @ A.S21, atol=1e-12)


def test_star_is_associative_and_preserves_symmetry_and_unitarity():
    basis = ModeBasis.propagating(60e9, *WR90)
    A, B, C = (random_reciprocal_lossless(basis) for _ in range(3))
    left, right = (A @ B) @ C, A @ (B @ C)
    assert np.allclose(left.S, right.S, atol=1e-12)
    S = left.S
    assert np.allclose(S, S.T, atol=1e-12)
    assert np.allclose(S.conj().T @ S, np.eye(S.shape[0]), atol=1e-12)


def test_identity_element():
    basis = ModeBasis.propagating(40e9, *WR90)
    A = random_reciprocal_lossy(basis)
    N = len(basis)
    Z, I = np.zeros((N, N)), np.eye(N)
    E = GSM.from_matrix(np.block([[Z, I], [I, Z]]), basis)      # a section of zero length
    assert np.allclose((A @ E).S, A.S, atol=1e-14) and np.allclose((E @ A).S, A.S, atol=1e-14)


def test_mismatched_bases_are_refused():
    b1 = ModeBasis.propagating(40e9, *WR90)
    b2 = ModeBasis.propagating(50e9, *WR90)
    with pytest.raises(ValueError):
        random_reciprocal_lossy(b1) @ random_reciprocal_lossy(b2)


def test_reversed_swaps_ports_with_reversal_signs():
    basis = ModeBasis.propagating(40e9, *WR90)
    A = random_reciprocal_lossy(basis)
    R = A.reversed()
    P = np.array([(-1.0) ** (m.n + 1) for m in basis])
    assert np.allclose(R.S11, P[:, None] * A.S22 * P[None, :]) and np.allclose(R.S21, P[:, None] * A.S12 * P[None, :])
    assert np.allclose(R.reversed().S, A.S)
    # within one family the signs cancel
    g, _ = bend_gsm(60e9, *WR90, 1, 1 / 0.2, 0.1)
    assert np.allclose(g.reversed().S11, g.S22) and np.allclose(g.reversed().S21, g.S12)


def test_reversal_signs_from_the_fields():
    """P_p = overlap of mode p with itself rotated by pi about u: (u, v) -> (u, -v), (e_u, e_v) -> (e_u, -e_v)."""
    from wrmodes.modes import transverse_field, reversal_signs
    a, b = WR90
    basis = ModeBasis.below(80e9, a, b)
    x, w = np.polynomial.legendre.leggauss(48)
    U, V = np.meshgrid(x * a / 2, x * b / 2, indexing="ij")
    W = np.outer(w * a / 2, w * b / 2)
    P = reversal_signs(basis)
    for mode, p in zip(basis, P):
        eu, ev = transverse_field(mode, a, b, U, V)
        ru, rv = transverse_field(mode, a, b, U, -V)
        assert np.isclose(np.sum(W * (eu * ru - ev * rv)), p, atol=1e-12), mode


def test_frequency_and_cross_section_checks():
    f, (a, b) = 30e9, WR90
    g1 = straight(ModeBasis.propagating(f, a, b, families=[0]), f, 0.1)
    g2 = straight(ModeBasis.propagating(31e9, 1.1 * a, b, families=[1]), 31e9, 0.1)
    with pytest.raises(ValueError):
        block_diag(g1, g2)
    nof = GSM.from_matrix(g1.S, g1.basis1)
    with pytest.raises(ValueError):
        g1 @ nof
    hb = ModeBasis.propagating(f, 0.02, 0.02, frame="H")
    eb = hb.with_frame("E")
    with pytest.raises(ValueError):
        straight(hb, f, 0.1) @ straight(eb, f, 0.1)


# ------------------------------------------------------------------------------------ transfer matrices
@pytest.mark.parametrize("a,b,n,f", [(*WR90, 0, 60e9), (*WR90[::-1], 1, 60e9), (*WR90, 1, 70e9)])
def test_bend_gsm_is_reciprocal_and_unitary(a, b, n, f):
    g, _ = bend_gsm(f, a, b, n, 1 / 0.2, 0.1)
    assert g.reciprocity_defect() < 1e-12
    assert g.unitarity_defect() < 1e-12


def test_gsm_cascade_equals_transfer_matrix_product():
    """Lossless, propagating-only: the star product of two bends and a straight equals the product of the
    transfer matrices (the cascade used in wrmodes 0.1)."""
    f, (a, b) = 60e9, WR90
    g1, phi1 = bend_gsm(f, a, b, 0, 1 / 0.25, 0.15)
    g2, phi2 = bend_gsm(f, a, b, 0, -1 / 0.3, 0.12)
    s = straight(g1.basis2, f, 0.7)
    D, K, beta = coupled_mode_matrices(f, a, b, g1.basis1.legacy_modes())
    P = expm(D * 0.7)
    via_star = cascade(g1, s, g2)
    via_phi = from_transfer_matrix(phi2 @ P @ phi1, g1.basis1, f)
    assert np.allclose(via_star.S, via_phi.S, atol=1e-12)


def test_matches_legacy_transmission_matrix():
    """S21 and S11 of the GSM agree with T and R of wrmodes 0.1 for the same bend."""
    f, (a, b) = 50e9, WR90
    L = 0.3
    kappa = lambda z: np.ones_like(np.asarray(z, float)) * np.radians(60) / L
    modes, T, R, beta = wr.transmission_matrix(f, a, b, 0, L, kappa)
    g, _ = bend_gsm(f, a, b, 0, np.radians(60) / L, L)
    assert np.allclose(g.S21, T, atol=1e-9) and np.allclose(g.S11, R, atol=1e-9)


def test_block_diag_of_families():
    f, (a, b) = 60e9, WR90
    g0, _ = bend_gsm(f, a, b, 0, 1 / 0.2, 0.1)
    g1, _ = bend_gsm(f, a, b, 1, 1 / 0.2, 0.1)
    g = block_diag(g0, g1)
    assert len(g.basis1) == len(g0.basis1) + len(g1.basis1)
    i, j = g.basis1.index(("TE", 1, 0)), g.basis1.index(("TE", 2, 0))
    assert np.isclose(g.S21[j, i], g0.S21[g0.basis1.index(("TE", 2, 0)), g0.basis1.index(("TE", 1, 0))])
    k = g.basis1.index(("TE", 0, 1))
    assert g.S21[k, i] == 0                                   # different families do not couple
    assert g.reciprocity_defect() < 1e-12 and g.unitarity_defect() < 1e-12


# ------------------------------------------------------------------------------------ trapped modes
def _trapped_pair(f=None):
    """Two GSMs with a two-mode middle basis: mode 0 passes, mode 1 is totally reflected on both sides and
    decoupled from the outer ports, so I - A22 B11 is exactly singular (as for two junctions on the same plane)."""
    outer = ModeBasis(0.02, 0.01, [("TE", 1, 0)])
    middle = ModeBasis(0.02, 0.01, [("TE", 1, 0), ("TE", 2, 0)])
    t, r = 0.6, 0.8j                                   # |t|^2 + |r|^2 = 1, symmetric lossless 2x2 for the passing mode
    A11 = np.array([[r]]); A12 = np.array([[t, 0]]); A21 = A12.T.copy(); A22 = np.array([[r, 0], [0, -1]])
    A = GSM(A11, A12, A21, A22, outer, middle, f)
    B = GSM(A22, A21, A12, A11, middle, outer, f)
    return A, B, t, r


def test_star_handles_hidden_trapped_modes():
    A, B, t, r = _trapped_pair()
    M = np.eye(2) - B.S11 @ A.S22
    assert np.linalg.matrix_rank(M) == 1               # exactly singular
    C = A @ B
    # the passing mode alone: two lossless 2-ports, closed form with one round trip series
    S21 = t * t / (1 - r * r)
    S11 = r + t * r * t / (1 - r * r)
    assert np.isclose(C.S21[0, 0], S21) and np.isclose(C.S11[0, 0], S11)
    assert np.isfinite(C.S).all() and C.reciprocity_defect() < 1e-14


def test_star_raises_on_a_resonance_visible_from_outside():
    A, B, t, r = _trapped_pair()
    # couple the trapped mode weakly to the outer port of A: now the singular direction is visible
    A2 = GSM(A.S11, np.array([[t, 1e-3]]), np.array([[t], [1e-3]]), A.S22, A.basis1, A.basis2, None)
    with pytest.raises(np.linalg.LinAlgError):
        A2 @ B


def test_star_unchanged_for_regular_cascades():
    basis = ModeBasis.propagating(60e9, *WR90)
    A, B = random_reciprocal_lossy(basis), random_reciprocal_lossy(basis)
    M = np.eye(len(basis)) - B.S11 @ A.S22
    X = np.linalg.solve(M, B.S11)
    assert np.allclose((A @ B).S11, A.S11 + A.S12 @ X @ A.S21, atol=1e-13)
