"""Shared definitions for the validation of the transfer-matrix solver.

The reference solution integrates the fundamental matrix Phi' = (D + kappa(z) K) Phi, Phi(0) = I, with the
explicit 8th-order Runge-Kutta method DOP853 at rtol = 1e-11 (piecewise between curvature discontinuities),
or uses the exact matrix exponential for a constant curvature. It shares the coupling matrices D, K with the
solver under test (these are validated separately against quadrature, check_coupling_integrals.py) but none of
its numerics (no Magnus slicing, no adaptive doubling, no batched exponentials).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from wrmodes.transfer_matrix import coupled_mode_matrices, select_coupled_modes

INCH = 0.0254

# (name, a [in bend plane], b, input mode, frequencies) -- a in the bend plane, b perpendicular
GUIDES = {
    'WR-90 H':  (0.9 * INCH, 0.4 * INCH, ('TE', 1, 0), (12e9, 90e9)),
    'WR-90 E':  (0.4 * INCH, 0.9 * INCH, ('TE', 0, 1), (12e9, 90e9)),
    'WR-28 H':  (0.28 * INCH, 0.14 * INCH, ('TE', 1, 0), (25e9, 110e9)),
    'WR-28 E':  (0.14 * INCH, 0.28 * INCH, ('TE', 0, 1), (25e9, 110e9)),
    '44x30 H':  (44e-3, 30e-3, ('TE', 1, 0), (4e9, 20e9)),
    '44x30 E':  (30e-3, 44e-3, ('TE', 0, 1), (6e9, 20e9)),
    'sq20 H':   (20e-3, 20e-3, ('TE', 1, 0), (8e9, 40e9)),
    'sq20 E':   (20e-3, 20e-3, ('TE', 0, 1), (8e9, 40e9)),
}

# bend angle (deg) and length in units of the larger guide side D
GEOMETRIES = {
    'sharp 30deg 2D':   (30, 2),
    '90deg 7D':         (90, 7),
    '90deg 26D':        (90, 26),
    '180deg 90D':       (180, 90),
}


def profile(name, theta_deg, L):
    """Curvature profile kappa(z) on [0, L] with total angle theta; returns (kappa, breakpoints, is_constant)."""
    th = np.radians(theta_deg)
    if name == 'constant':
        k0 = th / L
        return (lambda z: np.full(np.shape(z), k0)), [], True
    if name == 'sech':
        T = 4.0; g = 2 * T / L; R0 = 4 / th / g * np.arctan(np.tanh(T / 2))
        return (lambda z: 1 / (R0 * np.cosh(g * (np.asarray(z) - L / 2)))), [], False
    if name == 'cosine':        # half-sine curvature, zero at both ends
        A = np.pi * th / (2 * L)
        return (lambda z: A * np.sin(np.pi * np.asarray(z) / L)), [], False
    if name == 'triangular':    # linear rise to L/2, linear fall
        km = 2 * th / L
        return (lambda z: km * (1 - np.abs(2 * np.asarray(z) / L - 1))), [L / 2], False
    if name == 'S-bend':        # +theta sech bend over [0, L/2] followed by -theta sech bend (net angle 0)
        T = 4.0; h = L / 2; g = 2 * T / h; R0 = 4 / th / g * np.arctan(np.tanh(T / 2))
        def k(z):
            z = np.asarray(z, dtype=float)
            return np.where(z < h, 1, -1) / (R0 * np.cosh(g * (np.where(z < h, z, z - h) - h / 2)))
        return k, [h], False
    if name == 'step':          # piecewise constant, discontinuity at L/3 (not on a dyadic slice boundary)
        k1 = 0.5 * th / L; k2 = 1.25 * th / L   # k1 L/3 + k2 2L/3 = th
        return (lambda z: np.where(np.asarray(z) < L / 3, k1, k2)), [L / 3], False
    raise ValueError(name)


PROFILES = ['constant', 'sech', 'cosine', 'triangular', 'S-bend', 'step']


def reference(f, a, b, n, L, kappa, breakpoints=(), constant=False, rtol=1e-11):
    """Reference T, R and transfer matrix for the mode family with vertical index n."""
    modes = select_coupled_modes(f, a, b, n=n)
    D, K, beta = coupled_mode_matrices(f, a, b, modes)
    N2 = D.shape[0]
    if constant:
        phi = expm((D + float(kappa(0.0)) * K) * L)
    else:
        phi = np.eye(N2, dtype=complex)
        pts = [0.0] + sorted(breakpoints) + [L]
        for z0, z1 in zip(pts[:-1], pts[1:]):
            sol = solve_ivp(lambda z, y: ((D + float(kappa(z)) * K) @ y.reshape(N2, N2)).ravel(), (z0, z1),
                            np.eye(N2, dtype=complex).ravel(), method='DOP853', rtol=rtol, atol=rtol * 1e-2)
            phi = sol.y[:, -1].reshape(N2, N2) @ phi
    N = N2 // 2
    R = -np.linalg.solve(phi[N:, N:], phi[N:, :N])
    T = phi[:N, :N] + phi[:N, N:] @ R
    return modes, beta, T, R, phi


def loss_db(x):
    return -20 * np.log10(np.abs(x))
