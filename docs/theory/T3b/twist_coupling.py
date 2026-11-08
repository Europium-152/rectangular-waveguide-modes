"""Closed-form twist coupling of a rectangular guide (T3b, docs/theory/T3b-twist-coupling.md).

Machine-checkable version of the boxed formulas of the note. Conventions: CONVENTIONS.md (fields of
wrmodes.modes, power-wave amplitudes, e^{+i omega t}). The cross-section rotates about its centre, theta(z) is the
rotation angle of the local frame (u, v) from the port-1 frame, positive counter-clockwise (from u towards v)
seen with z pointing at the observer, and tau = theta'(z).

Results (boxes of the note):

* K_pq = iint [ div(e_p) (w . e_q) + (r . e_p) div(z x e_q) ] ds,   r = (u, v), w = (v, -u)        (box 2)
  equivalently K_pq = G_qp = iint e_q . (d/dtheta) e_p ds, with the rotation generator
  G e = z x e - (u d_v - v d_u) e                                                                    (box 3)
* Telegrapher form:  dV/dz = -i Z I + tau K V,   dI/dz = -i Y V - tau K^T I   (+ straight-guide terms)  (box 1)
* Power waves: d/dz [a+; a-] = M [a+; a-],
      M = [[-i beta + tau A, tau B], [tau B, +i beta + tau A]],
      A_pq = (K_pq rho_pq - K_qp / rho_pq) / 2,  B_pq = (K_pq rho_pq + K_qp / rho_pq) / 2,
      rho_pq = sqrt(Y_p) / sqrt(Y_q)  (principal roots)                                             (box 4)
"""
import numpy as np
import scipy.constants as cst

from wrmodes.modes import Mode, ModeBasis, cutoff_wavenumber, wave_admittance, propagation_constant

__all__ = ["Z_sc", "X_cc", "X_ss", "norm", "K_element", "K_explicit", "K_matrix", "K_matrix_fast", "A_B_matrices", "M_matrix", "couples",
           "C_TE10_TE01"]

C_TE10_TE01 = 8 / np.pi ** 2


# ------------------------------------------------------------------ 1D integrals on [-a/2, a/2], x' = x + a/2
# c_m(x) = cos(m pi x'/a), s_m(x) = sin(m pi x'/a)
def Z_sc(m, mp, a):
    """int s_m c_m' dx = (a/pi) 2m/(m^2 - m'^2) if m + m' is odd, else 0."""
    if (m + mp) % 2 == 0:
        return 0.0
    return a / np.pi * 2.0 * m / (m * m - mp * mp)


def X_cc(m, mp, a):
    """int x c_m c_m' dx = -(2 a^2/pi^2) (m^2 + m'^2)/(m^2 - m'^2)^2 if m + m' is odd, else 0."""
    if (m + mp) % 2 == 0:
        return 0.0
    return -2.0 * a * a / np.pi ** 2 * (m * m + mp * mp) / (m * m - mp * mp) ** 2


def X_ss(m, mp, a):
    """int x s_m s_m' dx = -(4 a^2/pi^2) m m'/(m^2 - m'^2)^2 if m + m' is odd, else 0."""
    if (m + mp) % 2 == 0:
        return 0.0
    return -4.0 * a * a / np.pi ** 2 * m * mp / (m * m - mp * mp) ** 2


def norm(mode, a, b):
    """Prefactor N of Gamma (TE) or Pi (TM), CONVENTIONS.md section 4."""
    eps = lambda l: 1.0 if l == 0 else 2.0
    s = np.pi * np.sqrt(mode.m ** 2 * b / a + mode.n ** 2 * a / b)
    return (np.sqrt(eps(mode.m) * eps(mode.n)) if mode.kind == "TE" else 2.0) / s


def couples(p, q):
    """Selection rule: K_pq can be non-zero only if m + m' and n + n' are both odd."""
    p, q = Mode.parse(p), Mode.parse(q)
    return (p.m + q.m) % 2 == 1 and (p.n + q.n) % 2 == 1


def K_element(p, q, a, b):
    """K_pq (dimensionless) of box 2, closed form.

    TE p, TE q:  K = -chi_q^2 iint Gamma_q L Gamma_p
    TM p, TM q:  K = +chi_p^2 iint Pi_p L Pi_q
    TM p, TE q:  K = (chi_q^2 - chi_p^2) iint Pi_p D Gamma_q + 2 chi_q^2 iint Pi_p Gamma_q
    TE p, TM q:  K = 0
    with L = u d_v - v d_u and D = u d_u + v d_v.
    """
    p, q = Mode.parse(p), Mode.parse(q)
    if not couples(p, q):
        return 0.0
    m, n, mp, np_ = p.m, p.n, q.m, q.n
    al_m, ga_n = m * np.pi / a, n * np.pi / b
    al_mp, ga_np = mp * np.pi / a, np_ * np.pi / b
    NpNq = norm(p, a, b) * norm(q, a, b)
    chi2p, chi2q = cutoff_wavenumber(p, a, b) ** 2, cutoff_wavenumber(q, a, b) ** 2
    if p.kind == "TE" and q.kind == "TE":
        GqLGp = NpNq * (-ga_n * X_cc(m, mp, a) * Z_sc(n, np_, b) + al_m * Z_sc(m, mp, a) * X_cc(n, np_, b))
        return -chi2q * GqLGp
    if p.kind == "TM" and q.kind == "TM":
        PpLPq = NpNq * (ga_np * X_ss(m, mp, a) * Z_sc(n, np_, b) - al_mp * Z_sc(m, mp, a) * X_ss(n, np_, b))
        return chi2p * PpLPq
    if p.kind == "TM" and q.kind == "TE":
        PpDGq = NpNq * (-al_mp * X_ss(m, mp, a) * Z_sc(n, np_, b) - ga_np * Z_sc(m, mp, a) * X_ss(n, np_, b))
        PpGq = NpNq * Z_sc(m, mp, a) * Z_sc(n, np_, b)
        return (chi2q - chi2p) * PpDGq + 2.0 * chi2q * PpGq
    return 0.0                                              # TE p, TM q


def K_matrix(basis):
    """K (N x N, real, dimensionless) for a ModeBasis, rows and columns in basis order."""
    a, b = basis.a, basis.b
    return np.array([[K_element(p, q, a, b) for q in basis.modes] for p in basis.modes])


def A_B_matrices(basis, f):
    """A (forward-forward, = backward-backward) and B (forward-backward) blocks of box 4 at f (Hz).

    A is antisymmetric (lossless), B symmetric, both real for propagating modes. Multiply by theta'(z).
    """
    K = K_matrix(basis)
    sY = np.sqrt(basis.admittance(f).astype(complex))    # principal roots
    rho = sY[:, None] / sY[None, :]
    A = 0.5 * (K * rho - K.T / rho)
    B = 0.5 * (K * rho + K.T / rho)
    return A, B


def M_matrix(basis, f, tau):
    """System matrix of d/dz [a+; a-] = M [a+; a-] for a twist rate tau = theta' (rad/m)."""
    A, B = A_B_matrices(basis, f)
    beta = basis.beta(f)
    N = len(basis)
    M = np.zeros((2 * N, 2 * N), dtype=complex)
    M[:N, :N] = -1j * np.diag(beta) + tau * A
    M[:N, N:] = tau * B
    M[N:, :N] = tau * B
    M[N:, N:] = 1j * np.diag(beta) + tau * A
    return M


def K_matrix_fast(basis):
    """Vectorized K_matrix (same result; used for large bases)."""
    a, b = basis.a, basis.b
    kind = np.array([0 if md.kind == "TE" else 1 for md in basis.modes])
    m = np.array([md.m for md in basis.modes], dtype=float)
    n = np.array([md.n for md in basis.modes], dtype=float)
    Nn = np.array([norm(md, a, b) for md in basis.modes])
    chi2 = np.pi ** 2 * ((m / a) ** 2 + (n / b) ** 2)
    M1, M2 = m[:, None], m[None, :]
    N1, N2 = n[:, None], n[None, :]
    oddm = (M1 + M2) % 2 == 1
    oddn = (N1 + N2) % 2 == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        def Z(x, y, h, odd):
            return np.where(odd, h / np.pi * 2 * x / (x * x - y * y), 0.0)

        def Xc(x, y, h, odd):
            return np.where(odd, -2 * h * h / np.pi ** 2 * (x * x + y * y) / (x * x - y * y) ** 2, 0.0)

        def Xs(x, y, h, odd):
            return np.where(odd, -4 * h * h / np.pi ** 2 * x * y / (x * x - y * y) ** 2, 0.0)
        Zm, Zn = Z(M1, M2, a, oddm), Z(N1, N2, b, oddn)
        Xcm, Xcn = Xc(M1, M2, a, oddm), Xc(N1, N2, b, oddn)
        Xsm, Xsn = Xs(M1, M2, a, oddm), Xs(N1, N2, b, oddn)
    NN = Nn[:, None] * Nn[None, :]
    al1, ga1 = M1 * np.pi / a, N1 * np.pi / b
    al2, ga2 = M2 * np.pi / a, N2 * np.pi / b
    c1, c2 = chi2[:, None], chi2[None, :]
    EE = -c2 * NN * (-ga1 * Xcm * Zn + al1 * Zm * Xcn)
    MM = c1 * NN * (ga2 * Xsm * Zn - al2 * Zm * Xsn)
    ME = (c2 - c1) * NN * (-al2 * Xsm * Zn - ga2 * Zm * Xsn) + 2 * c2 * NN * Zm * Zn
    k1, k2 = kind[:, None], kind[None, :]
    K = np.where((k1 == 0) & (k2 == 0), EE, 0.0)
    K = np.where((k1 == 1) & (k2 == 1), MM, K)
    K = np.where((k1 == 1) & (k2 == 0), ME, K)
    return np.where(oddm & oddn, K, 0.0)


def K_explicit(p, q, a, b):
    """Fully explicit rational form of K_pq (box 2b of the note); P = b^2 m^2 + a^2 n^2, eps_0 = 1, eps_l = 2.

    TE-TE: -(4 s / (pi^2 a b)) sqrt(P_q / P_p) [a^2 n^2 (m^2+m'^2)(n^2-n'^2) - b^2 m^2 (n^2+n'^2)(m^2-m'^2)]
            / [(m^2-m'^2)^2 (n^2-n'^2)^2],                      s = sqrt(eps_m eps_n eps_m' eps_n')
    TM-TM: -(32 m n m' n' / (pi^2 a b)) sqrt(P_p / P_q) [a^2 (n^2-n'^2) - b^2 (m^2-m'^2)] / [(m^2-m'^2)^2 (n^2-n'^2)^2]
    TM-TE: -(16 m n sqrt(eps_m' eps_n') / (pi^2 sqrt(P_p P_q))) [a^2 m'^2/(m^2-m'^2)^2 + b^2 n'^2/(n^2-n'^2)^2]
    TE-TM: 0
    (all for m + m' and n + n' odd; zero otherwise).
    """
    p, q = Mode.parse(p), Mode.parse(q)
    if not couples(p, q) or (p.kind == "TE" and q.kind == "TM"):
        return 0.0
    eps = lambda l: 1.0 if l == 0 else 2.0
    m, n, mp, np_ = p.m, p.n, q.m, q.n
    Pp, Pq = b * b * m * m + a * a * n * n, b * b * mp * mp + a * a * np_ * np_
    dm, dn = m * m - mp * mp, n * n - np_ * np_
    if p.kind == "TE":
        s = np.sqrt(eps(m) * eps(n) * eps(mp) * eps(np_))
        num = a * a * n * n * (m * m + mp * mp) * dn - b * b * m * m * (n * n + np_ * np_) * dm
        return -4 * s / (np.pi ** 2 * a * b) * np.sqrt(Pq / Pp) * num / (dm * dm * dn * dn)
    if q.kind == "TM":
        return (-32 * m * n * mp * np_ / (np.pi ** 2 * a * b) * np.sqrt(Pp / Pq) * (a * a * dn - b * b * dm)
                / (dm * dm * dn * dn))
    s = np.sqrt(eps(mp) * eps(np_))
    return -16 * m * n * s / (np.pi ** 2 * np.sqrt(Pp * Pq)) * (a * a * mp * mp / dm ** 2 + b * b * np_ * np_ / dn ** 2)
