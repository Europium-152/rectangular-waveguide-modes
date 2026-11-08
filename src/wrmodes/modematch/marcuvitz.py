"""Single-mode equivalent circuits of step junctions, transcribed from Marcuvitz (1951).

N. Marcuvitz, *Waveguide Handbook*, MIT Radiation Laboratory Series vol. 10, McGraw-Hill (1951):

* Sec. 5.24a, pp. 296-298 (PDF pp. 310-312): symmetric H-plane junction of two rectangular guides (widths a > a',
  equal heights), eqs. (1a)-(3c).
* Sec. 5.24b, pp. 298-302 (PDF pp. 312-316): asymmetric H-plane junction (one side wall continuous), eqs. (4a)-(6c).
* Sec. 5.26a, pp. 306-308 (PDF pp. 320-322): symmetric E-plane junction (change in height b > b', equal widths),
  eqs. (1), (2a)-(2d).
* Sec. 5.26b, p. 310 (PDF p. 324): asymmetric E-plane junction; same as 5.26a with lambda_g replaced by lambda_g/2.

Marcuvitz uses exp(+j omega t), the time convention of wrmodes (j = i). His reactances and susceptances are
therefore directly comparable with wrmodes amplitudes, with the voltage reflection coefficient
Gamma = S11 of the TE10 mode (CONVENTIONS.md section 5: the voltage of a mode is proportional to a+ + a-).

These formulas are for comparison only (tests/test_steps.py, docs/reports/WP4.md). They are accurate to the
percentages Marcuvitz states within his stated ranges, not to the tolerances of CONVENTIONS.md.

Notes on the transcription:

* F(k) and E(k) are the complete elliptic integrals of the first and second kinds of *modulus* k (Marcuvitz p. 298);
  scipy's ellipk/ellipe take the parameter m = k^2. The small-aperture limits (1b), (2b), (3b) reproduce (1a),
  (2a), (3a) with this reading (tests/test_steps.py).
* Sec. 5.24a defines X0 = X22 - X12^2 / X11 ~ A' - C^2 / A but gives no expression for X12; the approximation
  A' - C^2 / A is used, as printed.
* Eq. (2a) of Sec. 5.24 has lambda_g (not lambda_g') in its denominator, and N22 of Sec. 5.24b has 1 + 3 alpha^2
  (not 3 + alpha^2) under 16 alpha^2 R2. Both are transcribed as printed.
"""
import numpy as np
import scipy.constants as cst
from scipy.special import ellipk, ellipe

__all__ = ["guide_wavelength", "eplane_step_susceptance", "eplane_step_susceptance_approx", "hplane_step_circuit",
           "hplane_step_circuit_approx", "shunt_susceptance_s", "hplane_circuit_s", "eplane_from_s",
           "hplane_from_s"]


def guide_wavelength(a, f):
    """TE10 guide wavelength (m) of a guide of width a (m) at f (Hz)."""
    lam = cst.c / f
    x = 1 - (lam / (2 * a)) ** 2
    if x <= 0:
        raise ValueError(f"TE10 does not propagate in a {a:g} m wide guide at {f / 1e9:g} GHz")
    return lam / np.sqrt(x)


# ------------------------------------------------------------------------------------------ E plane
def eplane_step_susceptance(b, b_prime, lam_g, asymmetric=False):
    """Shunt susceptance B / Y0 of a change in height b -> b' (b' < b), Marcuvitz Sec. 5.26, eq. (2a).

    Equivalent circuit (Fig. 5.26-1): at the junction plane T, both guides; Y0 (height b) and Y0' (height b'),
    with Y0 / Y0' = b' / b = alpha (eq. 1), and a shunt capacitive susceptance jB. Valid for b / lambda_g < 1,
    "correct to within 1 per cent" (p. 308). The asymmetric case (one broad wall continuous, Sec. 5.26b,
    p. 310) replaces lambda_g by lambda_g / 2 everywhere, including the range of validity.
    """
    lg = lam_g / 2 if asymmetric else lam_g
    al = b_prime / b
    r = (1 + al) / (1 - al)
    sb = np.sqrt(1 - (b / lg) ** 2)
    sbp = np.sqrt(1 - (b_prime / lg) ** 2)
    A = r ** (2 * al) * (1 + sb) / (1 - sb) - (1 + 3 * al ** 2) / (1 - al ** 2)
    Ap = r ** (2 / al) * (1 + sbp) / (1 - sbp) + (3 + al ** 2) / (1 - al ** 2)
    C = (4 * al / (1 - al ** 2)) ** 2
    L = np.log((1 - al ** 2) / (4 * al) * r ** (0.5 * (al + 1 / al)))
    term3 = (b / (4 * lg)) ** 2 * r ** (-4 * al) * ((5 * al ** 2 - 1) / (1 - al ** 2) + 4 / 3 * al ** 2 * C / A) ** 2
    return 2 * b / lg * (L + 2 * (A + Ap + 2 * C) / (A * Ap - C ** 2) + term3)


def eplane_step_susceptance_approx(b, b_prime, lam_g, which, asymmetric=False):
    """Approximations of B / Y0, Marcuvitz Sec. 5.26a: which = '2b' (3 %), '2c' (alpha << 1), '2d' (delta << 1)."""
    lg = lam_g / 2 if asymmetric else lam_g
    al = b_prime / b
    d = 1 - al
    r = (1 + al) / (1 - al)
    if which == "2b":
        sb = np.sqrt(1 - (b / lg) ** 2)
        A = r ** (2 * al) * (1 + sb) / (1 - sb) - (1 + 3 * al ** 2) / (1 - al ** 2)
        return 2 * b / lg * (np.log((1 - al ** 2) / (4 * al) * r ** (0.5 * (al + 1 / al))) + 2 / A)
    if which == "2c":
        return 2 * b / lg * (np.log(np.e / (4 * al)) + al ** 2 / 3 + 0.5 * (b / lg) ** 2 * (1 - al ** 2) ** 4)
    if which == "2d":
        return 2 * b / lg * (d / 2) ** 2 * (2 * np.log(2 / d) / (1 - d) + 1 + 17 / 16 * (b / lg) ** 2)
    raise ValueError("which must be '2b', '2c' or '2d'")


# ------------------------------------------------------------------------------------------ H plane
def _hplane_coefficients(al, a, lam, asymmetric):
    """R1, R2, T, N11, N22, N12 of Marcuvitz Sec. 5.24a (p. 297) or 5.24b (pp. 301-302)."""
    ap = np.sqrt(1 - al ** 2)                 # alpha'
    rho = (1 - al) / (1 + al)
    a_s = al * a                              # a'
    pre = 2 * (a / lam) ** 2
    if not asymmetric:
        R1 = -rho ** al
        R2 = rho ** (1 / al)
        T = 4 * al / (1 - al ** 2) * rho ** (0.5 * (al + 1 / al))
        Fa, Ea = ellipk(al ** 2), ellipe(al ** 2)
        Fp, Ep = ellipk(ap ** 2), ellipe(ap ** 2)
        x, xp = 2 * a / (3 * lam), 2 * a_s / (3 * lam)
        Q, Qp = 1 - np.sqrt(1 - x ** 2), 1 - np.sqrt(1 - xp ** 2)
        q, qp = Q - 0.5 * x ** 2, Qp - 0.5 * xp ** 2
        w = 1 / (1 - al ** 2) ** 2
        N11 = (pre * (1 + 16 * R1 / (np.pi * (1 - al ** 2)) * (Ea - ap ** 2 * Fa) * (Ep - al ** 2 * Fp)
                      - R1 ** 2 - al ** 2 * T ** 2)
               + 12 * al ** 4 * w * rho ** (4 * al) * q + 48 * al ** 2 * w * rho ** (al + 3 / al) * qp)
        N22 = (pre * (al ** 2 + 16 * al ** 2 * R2 * Ea / (np.pi * (1 - al ** 2)) * (Fp - Ep) - al ** 2 * R2 ** 2
                      - T ** 2)
               + 48 * al ** 2 * w * rho ** (3 * al + 1 / al) * q + 12 * w * rho ** (4 / al) * qp)
        N12 = (pre * (ap ** 2 - 4 * Ea / np.pi * (Ep - al ** 2 * Fp) - R1 + al ** 2 * R2) * T
               + 24 * al ** 3 * w * rho ** (0.5 * (7 * al + 1 / al)) * q
               + 24 * al * w * rho ** (0.5 * (al + 7 / al)) * qp)
    else:
        R1 = -(1 + 3 * al ** 2) / (1 - al ** 2) * rho ** (2 * al)
        R2 = (3 + al ** 2) / (1 - al ** 2) * rho ** (2 / al)
        T = 16 * al ** 2 / (1 - al ** 2) ** 2 * rho ** (al + 1 / al)
        x, xp = a / lam, a_s / lam
        Q, Qp = 1 - np.sqrt(1 - x ** 2), 1 - np.sqrt(1 - xp ** 2)
        q, qp = Q - 0.5 * x ** 2, Qp - 0.5 * xp ** 2
        w = 1 / (1 - al ** 2) ** 2
        lr = np.log(rho)
        N11 = (pre * (1 + 8 * al * (1 - al ** 2) / (1 + 3 * al ** 2) * R1 * lr + 16 * al ** 2 / (1 + 3 * al ** 2) * R1
                      - R1 ** 2 - al ** 2 * T ** 2)
               + 2 * (32 / 3 * al ** 4 * w * rho ** (3 * al)) ** 2 * q
               + 2 * (32 * al ** 2 * w * rho ** (al + 2 / al)) ** 2 * qp)
        N22 = (pre * (al ** 2 - 8 * al * (1 - al ** 2) / (3 + al ** 2) * R2 * lr + 16 * al ** 2 / (1 + 3 * al ** 2) * R2
                      - al ** 2 * R2 ** 2 - T ** 2)
               + 2 * (32 * al ** 2 * w * rho ** (2 * al + 1 / al)) ** 2 * q
               + 2 * (32 / 3 * w * rho ** (3 / al)) ** 2 * qp)
        N12 = (pre * ((1 - al ** 2) ** 2 / (2 * al) * lr - R1 - al ** 2 * R2) * T
               + 2 / 3 * rho ** (5 * al + 1 / al) * (32 * al ** 3 * w) ** 2 * q
               + 2 / 3 * rho ** (al + 5 / al) * (32 * al * w) ** 2 * qp)
    return R1, R2, T, N11, N22, N12


def hplane_step_circuit(a, a_prime, f, asymmetric=False):
    """(Z0'/Z0, X/Z0, l/a) of an H-plane junction of widths a > a' (equal heights), Marcuvitz Sec. 5.24.

    Symmetric case: eqs. (1a), (2a), (3a) (pp. 296-297); asymmetric case (one side wall continuous): eqs. (4a),
    (5a), (6a) with the coefficients of pp. 301-302.

    Equivalent circuit (Figs. 5.24-1 and 5.24-4): line Z0 (wide guide) ending at the junction plane T, shunt
    inductive reactance jX at T, line Z0' (narrow guide) whose reference plane T' lies a distance l from T *on the
    wide-guide side* (top views of Figs. 5.24-1 and 5.24-4).

    Stated validity (p. 298, p. 302): symmetric 0.5 < a/lambda < 1.5 and 2a'/lambda > 1, error below 1 % over
    most of the range, up to 5 % at a/lambda = 1.5; asymmetric 0.5 < a/lambda < 1.0, below 1 % over most of the
    range, up to 5 % at a/lambda = 1.0.
    """
    lam = cst.c / f
    al = a_prime / a
    lg, lgp = guide_wavelength(a, f), guide_wavelength(a_prime, f)
    R1, R2, T, N11, N22, N12 = _hplane_coefficients(al, a, lam, asymmetric)
    den = (1 - R1) * (1 - R2) - T ** 2
    A = ((1 + R1) * (1 - R2) + T ** 2) / den
    Ap = ((1 - R1) * (1 + R2) + T ** 2) / den
    C = 2 * T / den
    X11 = A / (1 - ((A + 1) ** 2 * N11 + 2 * (A + 1) * C * N12 + C ** 2 * N22) / (4 * A))
    X22 = Ap / (1 - ((Ap + 1) ** 2 * N22 + 2 * (Ap + 1) * C * N12 + C ** 2 * N11) / (4 * Ap))
    X0 = Ap - C ** 2 / A
    g = 1 + (2 * a_prime * X0 / lgp) ** 2
    z_ratio = lgp * a_prime / (lg * a) * X11 / (al ** 2 * (X22 - X0)) * g
    x = 2 * a / lg * X11 * g / (1 + (2 * a_prime / lg) ** 2 * X0 * X22)
    l_over_a = al * X0 / np.pi
    return z_ratio, x, l_over_a


def hplane_step_circuit_approx(a, a_prime, f, which, asymmetric=False):
    """Small-aperture ('b', alpha << 1) or wide-aperture ('c', beta = 1 - alpha << 1) forms of Sec. 5.24:
    symmetric (1b)-(3b) / (1c)-(3c), asymmetric (4b)-(6b) / (4c)-(6c). Returns (Z0'/Z0, X/Z0, l/a)."""
    lam = cst.c / f
    al = a_prime / a
    be = 1 - al
    lg, lgp = guide_wavelength(a, f), guide_wavelength(a_prime, f)
    if not asymmetric:
        Q, Qp = 1 - np.sqrt(1 - (2 * a / (3 * lam)) ** 2), 1 - np.sqrt(1 - (2 * a_prime / (3 * lam)) ** 2)
        if which == "b":
            return (lgp * a_prime / (lg * a) * 1.750 * (1 - 0.58 * al ** 2),
                    2 * a / lg * 2.330 * al ** 2 * (1 + 1.56 * al ** 2) * (1 + 6.75 * al ** 2 * Q),
                    0.0084 * al * (1 - 1.5 * al ** 2))
        if which == "c":
            L2 = np.log(2 / be)
            inv_x = lg / (2 * a) * be ** 2 * (1 + be) * L2 / (1 - be / 2) * (1 - 27 / 8 * (Q + Qp) / (1 + 8 * L2))
            return lgp * a_prime / (lg * a) * (1 + be + be ** 2 / 2), 1 / inv_x, 0.0
    else:
        Q, Qp = 1 - np.sqrt(1 - (a / lam) ** 2), 1 - np.sqrt(1 - (a_prime / lam) ** 2)
        if which == "b":
            return (lgp * a_prime ** 3 / (lg * a ** 3) * 5.046 * (1 - 1.44 * al ** 2),
                    2 * a / lg * 5.764 * al ** 4 * (1 - 1.33 * al ** 2) * (1 + 41.3 * al ** 4 * Q),
                    0.0259 * al * (1 - 0.6 * al ** 2))
        if which == "c":
            L2 = np.log(2 / be)
            brace = (7 / 4 * L2 * (1 + be - 5 / 4 * be ** 2 * L2) * (1 + 4 / 7 * (1 - be) ** 2 * (a / lam) ** 2)
                     - 8 / 9 * (1 + 3 * be * L2) * (Q + Qp))
            inv_x = lg / (2 * a) * be ** 2 / (1 - be / 2 - be ** 2 * (L2 + 1.5)) * brace
            return (lgp / a_prime) / (lg / a) * (1 - be - 7 * be ** 2 / 4), 1 / inv_x, 0.0
    raise ValueError("which must be 'b' or 'c'")


# ------------------------------------------------------------------------------------------ circuits <-> S
def shunt_susceptance_s(y_ratio, b):
    """(S11, S22, S21) of the circuit Y0 | shunt jB | Y0' at one plane, with y_ratio = Y0'/Y0, b = B/Y0."""
    y_in = y_ratio + 1j * b                   # seen from guide 1, normalized to Y0
    s11 = (1 - y_in) / (1 + y_in)
    y_out = (1 + 1j * b) / y_ratio            # seen from guide 2, normalized to Y0'
    s22 = (1 - y_out) / (1 + y_out)
    s21 = 2 * np.sqrt(y_ratio) / (1 + y_ratio + 1j * b)
    return s11, s22, s21


def hplane_circuit_s(z_ratio, x, l, beta_prime):
    """(S11, S22) at the physical junction plane of Marcuvitz's H-plane circuit: Z0 | shunt jX | Z0', with the
    reference plane T' of the narrow guide a distance l on the wide-guide side of the junction. beta_prime is the
    propagation constant (1/m) of TE10 in the narrow guide; S22 is moved from T' to the junction plane."""
    y_in = 1 / z_ratio - 1j / x               # normalized to 1/Z0
    s11 = (1 - y_in) / (1 + y_in)
    y_out = z_ratio * (1 - 1j / x)            # normalized to 1/Z0'
    s22_tp = (1 - y_out) / (1 + y_out)
    return s11, s22_tp * np.exp(-2j * beta_prime * l)


def eplane_from_s(s11):
    """(Y0'/Y0, B/Y0) of the shunt-susceptance circuit at the junction plane from S11 of TE10 in the high guide."""
    y = (1 - s11) / (1 + s11)
    return y.real, y.imag


def hplane_from_s(s11, s22, beta_prime):
    """(Z0'/Z0, X/Z0, l) of Marcuvitz's H-plane circuit from S11 (wide guide) and S22 (narrow guide) of TE10, both
    at the junction plane. l is determined modulo pi / beta_prime; the value of smallest magnitude is returned."""
    y = (1 - s11) / (1 + s11)                 # = Z0/Z0' - j Z0/X
    z_ratio = 1 / y.real
    x = -1 / y.imag
    _, s22_tp = hplane_circuit_s(z_ratio, x, 0.0, beta_prime)
    dphi = np.angle(s22_tp / s22)             # = 2 beta' l  (mod 2 pi)
    return z_ratio, x, dphi / (2 * beta_prime)
