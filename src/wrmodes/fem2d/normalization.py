"""Normalization of the 2D scalar modal amplitudes to the LS and TE/TM amplitudes of CONVENTIONS.md.

Derivation: ``docs/theory/WP3-normalization.md``. In short, for family n of a port basis:

* LSM (Dirichlet): the scalar is the profile of E_v. A unit LS amplitude a_LS (forward or backward) gives
  E_v = g cos(n pi v'/b) phi_m(u), phi_m(u) = sqrt(2/a) sin(m pi u'/a), and the 2D amplitude is alpha = g a_LS
  for waves going either way.
* LSE (Neumann): the scalar is the profile of H_v. A unit forward LS amplitude gives
  H_v = h sin(n pi v'/b) phi_m(u), phi_m(u) = sqrt(eps_m/a) cos(m pi u'/a); the backward wave has -h.

g and h are computed here from ``wrmodes.modes.transverse_field``, ``wave_admittance`` (principal square roots,
CONVENTIONS.md section 5) and ``ls_rotation``; nothing about the fields is re-implemented. Closed forms, used only
by the tests: g = sqrt(eps_n) k_t sqrt(2 eta0 / (k beta b)), h = -2 k_t / sqrt(eta0 k beta b) (n >= 1).
"""
import numpy as np
import scipy.constants as cst

from ..modes import Mode, ls_rotation, transverse_field, wave_admittance

__all__ = ["scalar_mode", "ls_amplitude_factors"]

# generic sample points (fractions of a and b) at which the field shapes are compared
_SAMPLES = np.array([[0.2137, 0.1361], [0.4171, 0.3299], [0.7313, 0.6011], [0.0919, 0.8737]])


def scalar_mode(pol, m, a, s):
    """2D port mode phi_m at u' = s (0 <= s <= a), orthonormal on [0, a].

    LSM: sqrt(2/a) sin(m pi s / a) (m >= 1); LSE: sqrt(eps_m / a) cos(m pi s / a) (m >= 0).
    """
    if pol == "LSM":
        return np.sqrt(2.0 / a) * np.sin(m * np.pi * s / a)
    return np.sqrt((1.0 if m == 0 else 2.0) / a) * np.cos(m * np.pi * s / a)


def ls_amplitude_factors(basis, f):
    """LS labels and scalar factors of every mode of a port basis.

    Returns (Q, labels, factors): Q and labels as from ``ls_rotation`` (a_LS = Q a_TE/TM), and factors[l] = g (LSM)
    or h (LSE) of LS mode l, defined in the module docstring.
    """
    Q, labels = ls_rotation(basis, f)
    a, b = basis.a, basis.b
    Y = basis.admittance(f)
    w_e = np.sqrt(2.0 / Y)          # principal roots, CONVENTIONS.md section 5
    w_h = np.sqrt(2.0 * Y)
    factors = np.zeros(len(basis), dtype=complex)
    for l, (pol, m, n) in enumerate(labels):
        cols = np.flatnonzero(Q[l] != 0)
        vals = []
        for su, sv in _SAMPLES:
            u, v = su * a - a / 2, sv * b - b / 2
            if pol == "LSM":
                shape = scalar_mode("LSM", m, a, su * a) * np.cos(n * np.pi * sv)
                field = sum(Q[l, p] * w_e[p] * transverse_field(basis.modes[p], a, b, u, v)[1] for p in cols)
            else:
                shape = scalar_mode("LSE", m, a, su * a) * np.sin(n * np.pi * sv)
                # H_t = sqrt(2Y) z x e, and (z x e)_v = e_u
                field = sum(Q[l, p] * w_h[p] * transverse_field(basis.modes[p], a, b, u, v)[0] for p in cols)
            vals.append(field / shape)
        vals = np.array(vals)
        if np.max(np.abs(vals - vals[0])) > 1e-9 * abs(vals[0]):
            raise RuntimeError(f"LS mode {labels[l]} is not separable as expected (field convention mismatch)")
        factors[l] = vals[0]
    return Q, labels, factors


def closed_form_factor(pol, m, n, a, b, f):
    """Closed forms of g and h (see the module docstring); for the tests."""
    k = 2 * np.pi * f / cst.c
    from ..modes import ETA0, propagation_constant
    beta = propagation_constant(Mode("TE", m, n), a, b, f)
    kt = np.sqrt(k * k - (n * np.pi / b) ** 2 + 0j)
    if pol == "LSM":
        return np.sqrt(1.0 if n == 0 else 2.0) * kt * np.sqrt(2 * ETA0 / (k * beta * b))
    return -2 * kt / np.sqrt(ETA0 * k * beta * b)
