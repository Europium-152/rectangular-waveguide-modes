"""Closed-form overlap integrals between the modes of two rectangular guides, one inside the other.

The fields are those of ``wrmodes.modes.transverse_field`` (CONVENTIONS.md section 4). In local coordinates
u' = u + a/2, v' = v + b/2 every normalized transverse field is separable,

    e_u = A_u cos(m pi u'/a) sin(n pi v'/b),     e_v = A_v sin(m pi u'/a) cos(n pi v'/b),

with (A_u, A_v) = c (k_m, k_n) for TM (e = grad Pi) and c (-k_n, k_m) for TE (e = grad Gamma x z_hat), where
k_m = m pi / a, k_n = n pi / b and c the normalization of the potential (JINST paper eq. 5.4). The overlap over
the aperture of the small guide therefore reduces to products of one-dimensional integrals of the type
int_0^A cos(p x) cos(q (x + s)) dx, which are evaluated in closed form.
"""
import numpy as np

from ..modes import ModeBasis

__all__ = ["overlap_matrix", "field_amplitudes"]


def field_amplitudes(basis):
    """Per-mode arrays (m, n, A_u, A_v) of the separable form of the normalized fields of `basis`.

    e_u = A_u cos(m pi u'/a) sin(n pi v'/b) and e_v = A_v sin(m pi u'/a) cos(n pi v'/b), identical to
    ``wrmodes.modes.transverse_field`` (checked against it in tests/test_steps.py).
    """
    a, b = basis.a, basis.b
    m = np.array([md.m for md in basis.modes], dtype=float)
    n = np.array([md.n for md in basis.modes], dtype=float)
    te = np.array([md.kind == "TE" for md in basis.modes])
    eps_m, eps_n = np.where(m == 0, 1.0, 2.0), np.where(n == 0, 1.0, 2.0)
    s = np.pi * np.sqrt(m * m * b / a + n * n * a / b)
    c = np.where(te, np.sqrt(eps_m * eps_n), 2.0) / s
    km, kn = m * np.pi / a, n * np.pi / b
    Au = np.where(te, -c * kn, c * km)     # TE: d Gamma/dv;  TM: d Pi/du
    Av = np.where(te, c * km, c * kn)      # TE: -d Gamma/du; TM: d Pi/dv
    return m, n, Au, Av


def _cos_integral(k, phi, A):
    """int_0^A cos(k x + phi) dx = A cos(k A / 2 + phi) sinc(k A / 2), stable at k = 0."""
    return A * np.cos(0.5 * k * A + phi) * np.sinc(k * A / (2 * np.pi))


def _one_dim(ms, mL, As, AL, s):
    """(I_cc, I_ss) of shape (len(ms), len(mL)):

    I_cc = int_0^As cos(p x) cos(q (x + s)) dx,   I_ss = int_0^As sin(p x) sin(q (x + s)) dx,
    p = ms pi / As, q = mL pi / AL, where s is the distance from the wall of the large guide to the wall of the
    small one (both at local coordinate 0).
    """
    p = (np.asarray(ms) * np.pi / As)[:, None]
    q = (np.asarray(mL) * np.pi / AL)[None, :]
    minus = _cos_integral(p - q, -q * s, As)
    plus = _cos_integral(p + q, q * s, As)
    return 0.5 * (minus + plus), 0.5 * (minus - plus)


def overlap_matrix(small, large, du=0.0, dv=0.0):
    """Overlaps X[p, q] = int_small e^small_p . e^large_q ds of the normalized transverse fields.

    Parameters
    ----------
    small, large : ModeBasis
        Mode bases of the small guide (a_s x b_s) and of the large guide (a_l x b_l), with a_s <= a_l and
        b_s <= b_l. The fields are those of ``wrmodes.modes.transverse_field`` (CONVENTIONS.md section 4).
    du, dv : float
        Position (m) of the centre of the small guide in the local frame (u, v) of the large guide. The small
        cross-section must lie inside the large one.

    Returns
    -------
    X : ndarray, shape (len(small), len(large)), real
        The aperture is the cross-section of the small guide, where the small guide's fields live; outside it the
        tangential electric field of the large guide's side vanishes on the metal of the step.
    """
    if not isinstance(small, ModeBasis) or not isinstance(large, ModeBasis):
        raise TypeError("small and large must be ModeBasis objects")
    a_s, b_s, a_l, b_l = small.a, small.b, large.a, large.b
    su = du + 0.5 * (a_l - a_s)          # wall-to-wall offsets along u and v
    sv = dv + 0.5 * (b_l - b_s)
    tol = 1e-12 * max(a_l, b_l)
    if su < -tol or su > a_l - a_s + tol or sv < -tol or sv > b_l - b_s + tol:
        raise ValueError(f"the {a_s:g} x {b_s:g} guide offset by ({du:g}, {dv:g}) does not fit inside the "
                         f"{a_l:g} x {b_l:g} guide")
    ms, ns, Aus, Avs = field_amplitudes(small)
    mL, nL, AuL, AvL = field_amplitudes(large)
    # integrals over the distinct mode numbers only, then gathered
    um_s, iu_s = np.unique(ms, return_inverse=True)
    um_l, iu_l = np.unique(mL, return_inverse=True)
    vn_s, iv_s = np.unique(ns, return_inverse=True)
    vn_l, iv_l = np.unique(nL, return_inverse=True)
    Ucc, Uss = _one_dim(um_s, um_l, a_s, a_l, su)
    Vcc, Vss = _one_dim(vn_s, vn_l, b_s, b_l, sv)
    iu_s, iu_l, iv_s, iv_l = (np.ravel(x) for x in (iu_s, iu_l, iv_s, iv_l))
    Ucc, Uss = Ucc[np.ix_(iu_s, iu_l)], Uss[np.ix_(iu_s, iu_l)]
    Vcc, Vss = Vcc[np.ix_(iv_s, iv_l)], Vss[np.ix_(iv_s, iv_l)]
    return (Aus[:, None] * AuL[None, :]) * Ucc * Vss + (Avs[:, None] * AvL[None, :]) * Uss * Vcc
