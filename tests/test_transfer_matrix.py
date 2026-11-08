"""Unit tests for the transfer-matrix solver (run with `pytest` from the repository root)."""
import os
import warnings
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import wrmodes as wr
from wrmodes.transfer_matrix import coupled_mode_matrices, select_coupled_modes

WR90_BROAD = 0.9 * 0.0254
WR90_NARROW = 0.4 * 0.0254
HERE = os.path.dirname(os.path.abspath(__file__))


def sech_curvature(theta_deg, length, T=4.0):
    gamma = 2 * T / length
    R0 = 4 / np.radians(theta_deg) / gamma * np.arctan(np.tanh(T / 2))
    return lambda z: 1 / (R0 * np.cosh(gamma * (np.asarray(z) - length / 2)))


def constant_curvature(theta_deg, length):
    R = length / np.radians(theta_deg)
    return lambda z: np.ones_like(np.asarray(z, dtype=float)) / R


def loss_db(amp):
    return -20 * np.log10(np.abs(amp))


def reference_T(f, a, b, n, length, curvature):
    """Independent solution: fundamental matrix by DOP853 at tight tolerance, then the same algebra."""
    modes = select_coupled_modes(f, a, b, n=n)
    D, K, beta = coupled_mode_matrices(f, a, b, modes)
    N2 = D.shape[0]
    sol = solve_ivp(lambda z, y: ((D + curvature(z) * K) @ y.reshape(N2, N2)).ravel(), (0, length),
                    np.eye(N2, dtype=complex).ravel(), method='DOP853', rtol=1e-11, atol=1e-13)
    phi = sol.y[:, -1].reshape(N2, N2)
    N = N2 // 2
    R = -np.linalg.solve(phi[N:, N:], phi[N:, :N])
    return modes, phi[:N, :N] + phi[:N, N:] @ R, R


# ------------------------------------------------------------------------------------------- basics
def test_straight_guide_is_pure_phase():
    f, L = 40e9, 0.3
    modes, res = wr.solve_transfer_matrix(f, WR90_BROAD, WR90_NARROW, ('TE', 1, 0), L, lambda z: 0 * np.asarray(z))
    assert np.allclose(res.T, np.diag(np.exp(-1j * res.beta * L)), atol=1e-12)
    assert np.allclose(res.R, 0, atol=1e-14)


def test_constant_curvature_matches_single_matrix_exponential():
    f, L = 60e9, 0.3
    kappa = constant_curvature(60, L)
    modes, res = wr.solve_transfer_matrix(f, WR90_NARROW, WR90_BROAD, ('TE', 0, 1), L, kappa)
    D, K, beta = coupled_mode_matrices(f, WR90_NARROW, WR90_BROAD, modes)
    assert np.allclose(res.transfer_matrix, expm((D + kappa(0.0) * K) * L), atol=1e-10)


@pytest.mark.parametrize("a,b,mode_in", [(WR90_BROAD, WR90_NARROW, ('TE', 1, 0)), (WR90_NARROW, WR90_BROAD, ('TE', 0, 1)),
                                          (WR90_BROAD, WR90_NARROW, ('TE', 0, 1))])
@pytest.mark.parametrize("f", [25e9, 55e9, 89.5e9])
def test_power_conservation(a, b, mode_in, f):
    modes, res = wr.solve_transfer_matrix(f, a, b, mode_in, 0.6, sech_curvature(60, 0.6), error_control='full')
    S = np.vstack([res.T, res.R])            # columns: response to each forward input mode
    assert np.allclose(S.conj().T @ S, np.eye(len(modes)), atol=1e-10)


@pytest.mark.parametrize("a,b,mode_in", [(WR90_BROAD, WR90_NARROW, ('TE', 1, 0)), (WR90_BROAD, WR90_NARROW, ('TE', 0, 1))])
@pytest.mark.parametrize("f", [30e9, 70e9, 89.5e9])
def test_agrees_with_independent_reference(a, b, mode_in, f):
    L, kappa = 0.6, sech_curvature(60, 0.6)
    modes, res = wr.solve_transfer_matrix(f, a, b, mode_in, L, kappa)
    _, T_ref, _ = reference_T(f, a, b, mode_in[2], L, kappa)
    j = res.input_index
    assert np.max(np.abs(np.abs(res.T[:, j]) ** 2 - np.abs(T_ref[:, j]) ** 2)) < 5e-6
    assert abs(loss_db(res.T[j, j]) - loss_db(T_ref[j, j])) < 1e-4


def test_error_control_full_controls_all_entries():
    f, a, b, L, kappa = 70e9, WR90_BROAD, WR90_NARROW, 0.6, sech_curvature(60, 0.6)
    modes, res = wr.solve_transfer_matrix(f, a, b, ('TE', 0, 1), L, kappa, error_control='full', atol=1e-6)
    _, T_ref, R_ref = reference_T(f, a, b, 1, L, kappa)
    assert np.max(np.abs(res.T - T_ref)) < 1e-5
    assert np.max(np.abs(res.R - R_ref)) < 1e-5


# ------------------------------------------------------------------------------------------- profile / API
def test_profile_boundary_values_and_z_eval():
    f, L, kappa = 50e9, 0.15, sech_curvature(60, 0.15)
    modes, res = wr.solve_transfer_matrix(f, WR90_NARROW, WR90_BROAD, ('TE', 0, 1), L, kappa)
    N, j = len(modes), res.input_index
    assert res.x[0] == 0 and np.isclose(res.x[-1], L)
    e = np.zeros(N); e[j] = 1
    assert np.allclose(res.y[:N, 0], e)                      # prescribed input
    assert np.allclose(res.y[N:, -1], 0, atol=1e-12)          # no backward wave incident at z = L
    assert np.allclose(res.y[:N, -1], res.T[:, j])
    z = np.linspace(0, L, 37)
    _, res2 = wr.solve_transfer_matrix(f, WR90_NARROW, WR90_BROAD, ('TE', 0, 1), L, kappa, z_eval=z)
    assert np.allclose(res2.x, z)
    P = np.sum(np.abs(res2.y[:N]) ** 2, 0) - np.sum(np.abs(res2.y[N:]) ** 2, 0)
    assert np.allclose(P, 1, atol=1e-10)                      # net power flow conserved along z


def test_solve_mode_coupling_default_is_transfer_matrix_and_bvp_still_works():
    f, L, kappa = 40e9, 0.15, sech_curvature(60, 0.15)
    a, b, mode_in = WR90_NARROW, WR90_BROAD, ('TE', 0, 1)
    modes, res = wr.solve_mode_coupling(f, a, b, mode_in, L, kappa)
    assert isinstance(res, wr.TransferMatrixResult)
    modes_b, res_b = wr.solve_mode_coupling(f, a, b, mode_in, L, kappa, method='bvp', maximum_modes=50,
                                            max_nodes=10 ** 6, resolution='auto', tol=0.5)
    j = [m[0] for m in modes_b].index('TE0,1')
    assert abs(loss_db(res.y[res.input_index, -1]) - loss_db(res_b.y[j, -1])) < 0.01


def test_bvp_keywords_are_ignored_with_a_warning():
    with pytest.warns(UserWarning):
        wr.solve_mode_coupling(30e9, WR90_BROAD, WR90_NARROW, ('TE', 1, 0), 0.3, constant_curvature(60, 0.3), tol=0.5)


def test_maximum_modes_and_unknown_input():
    modes, res = wr.solve_transfer_matrix(80e9, WR90_BROAD, WR90_NARROW, ('TE', 1, 0), 0.4, sech_curvature(90, 0.4),
                                          maximum_modes=3)
    assert len(modes) == 3
    with pytest.raises(ValueError):
        wr.solve_transfer_matrix(10e9, WR90_BROAD, WR90_NARROW, ('TE', 2, 0), 0.4, sech_curvature(90, 0.4))


def test_single_mode_guide_has_no_conversion():
    # below the TE20 cutoff (13.1 GHz) only TE10 propagates in its family: no conversion, |T| = 1
    modes, res = wr.solve_transfer_matrix(10e9, WR90_BROAD, WR90_NARROW, ('TE', 1, 0), 0.2, sech_curvature(90, 0.2))
    assert len(modes) == 1 and np.isclose(abs(res.T[0, 0]), 1.0, atol=1e-12)


def test_transmission_matrix_and_cascade():
    f = 80e9
    modes, T1, R1, beta = wr.transmission_matrix(f, WR90_BROAD, WR90_NARROW, 0, 0.4, sech_curvature(90, 0.4))
    total = T1 @ wr.straight_section(beta, 1.0) @ T1
    assert total.shape == (len(modes), len(modes))
    assert np.all(np.sum(np.abs(total) ** 2, 0) <= 1 + 1e-9)


# ------------------------------------------------------------------------------------------- COMSOL
def _read_comsol(name):
    path = os.path.join(HERE, '..', 'benchmarks', 'comsol', name)
    lines = open(path).read().splitlines()
    hdr = [l for l in lines if l.startswith('% X')][0]
    f = np.array([float(s.split('freq=')[1].split()[0]) for s in hdr.split('emw.S21dB')[1:]]) * 1e9
    return f, -np.array([l for l in lines if not l.startswith('%')][0].split()[3:], dtype=float)


def test_agreement_with_comsol_within_0p01_dB():
    a, b, mode_in, th = WR90_NARROW, WR90_BROAD, ('TE', 0, 1), np.radians(60)
    r0, S = 0.055, 4.0
    g = 4 / th / r0 * np.arctan(np.tanh(S / 2)); L = 2 * S / g
    cases = [('15cm_60deg_hyperbolic_curve_side_a_short_e_field_horizontal', L,
              lambda z: 1 / (r0 * np.cosh(g * (np.asarray(z) - L / 2)))),
             ('15cm_60deg_constant_curve_side_a_short_e_field_horizontal', 0.15, constant_curvature(60, 0.15))]
    for fname, length, kappa in cases:
        freqs, comsol = _read_comsol(fname)
        ours = [loss_db(wr.solve_transfer_matrix(f, a, b, mode_in, length, kappa, profile=False)[1].T[0, 0])
                for f in freqs]
        assert np.max(np.abs(np.array(ours) - comsol)) < 0.01


def test_breakpoints_restore_fast_convergence_for_discontinuous_curvature():
    L, th = 0.6, np.radians(90)
    k1, k2 = 0.5 * th / L, 1.25 * th / L                      # step at L/3, total angle 90 deg
    kappa = lambda z: np.where(np.asarray(z) < L / 3, k1, k2)
    f, a, b = 70e9, WR90_BROAD, WR90_NARROW
    modes, res = wr.solve_transfer_matrix(f, a, b, ('TE', 1, 0), L, kappa, breakpoints=[L / 3])
    D, K, beta = coupled_mode_matrices(f, a, b, modes)
    exact = expm((D + k2 * K) * (2 * L / 3)) @ expm((D + k1 * K) * (L / 3))   # piecewise constant: exact
    N = len(modes)
    R = -np.linalg.solve(exact[N:, N:], exact[N:, :N]); T = exact[:N, :N] + exact[:N, N:] @ R
    assert res.success and res.n_slices <= 1024
    # without the breakpoint the convergence is only first order: the solver must hit max_slices and warn,
    # suggesting breakpoints
    with pytest.warns(RuntimeWarning, match="breakpoints"):
        _, res_nobp = wr.solve_transfer_matrix(f, a, b, ('TE', 1, 0), L, kappa, max_slices=1024)
    assert not res_nobp.success and res_nobp.n_slices >= res.n_slices
    assert np.allclose(np.abs(res.T[:, 0]) ** 2, np.abs(T[:, 0]) ** 2, atol=1e-8)


def test_blas_threads_option_gives_identical_results():
    import wrmodes as wr
    a, b = 0.4 * 0.0254, 0.9 * 0.0254
    kappa = lambda z: np.full_like(np.asarray(z, float), 1 / 0.2)
    _, r1 = wr.solve_transfer_matrix(40e9, a, b, ('TE', 0, 1), 0.1, kappa, profile=False, blas_threads=1)
    _, r0 = wr.solve_transfer_matrix(40e9, a, b, ('TE', 0, 1), 0.1, kappa, profile=False, blas_threads=None)
    assert np.allclose(r1.T, r0.T, atol=1e-12) and r1.n_slices == r0.n_slices
