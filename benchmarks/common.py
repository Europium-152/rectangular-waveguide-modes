"""Shared definitions for the JINST paper benchmarks (D. Hachmeister et al., JINST 2026).

All calculations use `wrmodes.solve_mode_coupling` with the settings of the paper:
  * all propagating modes of the coupled family are retained (maximum_modes=50 is never reached),
  * grid of 10 points per guided wavelength of the fastest mode (resolution='auto'),
  * tol=0.5, i.e. no adaptive mesh refinement (see Section 6.1 of the paper).
"""
import numpy as np
import wrmodes as wr

WR90_BROAD = 0.9 * 0.0254   # m
WR90_NARROW = 0.4 * 0.0254  # m
PAPER_SETTINGS = dict(maximum_modes=50, max_nodes=1_000_000, resolution='auto', tol=5e-1)

# Orientations in the convention of Section 5 of the paper: a = side in the bend plane, b = side perpendicular to it.
# (name, a, b, input mode)
ORIENTATIONS = {
    'TE10 H-plane': (WR90_BROAD, WR90_NARROW, ('TE', 1, 0)),
    'TE10 E-plane': (WR90_NARROW, WR90_BROAD, ('TE', 0, 1)),
    'TE01 H-plane': (WR90_NARROW, WR90_BROAD, ('TE', 1, 0)),   # tall guide
    'TE01 E-plane': (WR90_BROAD, WR90_NARROW, ('TE', 0, 1)),   # tall guide
}


def sech_curvature(theta_deg, length, T=4.0):
    """Hyperbolic secant curvature profile of total angle theta over `length`, truncated at +-T (eqs. 4.6-4.7)."""
    theta = np.radians(theta_deg)
    gamma = 2 * T / length
    R0 = 4 / theta / gamma * np.arctan(np.tanh(T / 2))
    def kappa(z):
        return 1 / (R0 * np.cosh(gamma * (z - length / 2)))
    kappa.R0, kappa.gamma = R0, gamma
    return kappa


def constant_curvature(theta_deg, length):
    R0 = length / np.radians(theta_deg)
    def kappa(z):
        return np.ones_like(np.asarray(z, dtype=float)) / R0
    kappa.R0 = R0
    return kappa


def mode_conversion_loss_db(f, a, b, mode_in, length, curvature, **kwargs):
    """Mode-conversion loss -20 log10 |A_j(L)| of the input mode, in dB (eq. 6.1)."""
    settings = dict(PAPER_SETTINGS); settings.update(kwargs)
    modes, result = wr.solve_mode_coupling(f, a, b, mode_in, length=length, curvature=curvature, **settings)
    name = f"{mode_in[0]}{mode_in[1]},{mode_in[2]}"
    j = [m[0] for m in modes].index(name)
    return -20 * np.log10(np.abs(result.y[j, -1]))
