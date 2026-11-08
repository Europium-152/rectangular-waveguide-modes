"""Machine-checkable form of the boxed formulas of docs/theory/T3-twist-coupling.md.

Twisted rectangular guide: the cross-section at z is the port-1 cross-section rotated by theta(z) about +z
(counterclockwise seen from the side the wave travels to). All fields and amplitudes are in the local frame
(u, v) that co-rotates with the cross-section (CONVENTIONS.md sections 2-5, fields of wrmodes.modes).

Every function here is a closed form; docs/theory/T3/check_*.py compare them with quadrature, with an
independent Galerkin solution of Maxwell's equations, and with the literature.
"""
import numpy as np
from scipy.linalg import expm

from wrmodes.modes import Mode, ModeBasis, cutoff_wavenumber


# ------------------------------------------------------------------------------------- coupling coefficients
def _wavenumbers(mode, a, b):
    return mode.m * np.pi / a, mode.n * np.pi / b


def _nu(mode):
    """Normalization factor nu: sqrt(eps_m eps_n) for TE, 2 for TM (e = nu / (sqrt(ab) chi) * (...))."""
    if mode.kind == "TM":
        return 2.0
    return np.sqrt((1.0 if mode.m == 0 else 2.0) * (1.0 if mode.n == 0 else 2.0))


def coupling_C(p, q, a, b):
    """C_pq = iint e_p . G e_q ds, G e = z x e - (u d_v - v d_u) e  (generator of active rotations).

    Box T3-2 of the theory note. Zero unless m_p + m_q and n_p + n_q are both odd. Dimensionless.
    """
    p, q = Mode.parse(p), Mode.parse(q)
    if (p.m + q.m) % 2 == 0 or (p.n + q.n) % 2 == 0:
        return 0.0
    k, l = _wavenumbers(p, a, b)
    kq, lq = _wavenumbers(q, a, b)
    K, Lam = kq ** 2 - k ** 2, lq ** 2 - l ** 2
    chi, chiq = cutoff_wavenumber(p, a, b), cutoff_wavenumber(q, a, b)
    nn = _nu(p) * _nu(q) / (a * b * chi * chiq)
    if p.kind == "TE" and q.kind == "TE":
        br = (k ** 2 + kq ** 2) * lq ** 2 * Lam - kq ** 2 * (l ** 2 + lq ** 2) * K
        return -4 * nn * chi ** 2 * br / (K ** 2 * Lam ** 2)
    if p.kind == "TE" and q.kind == "TM":
        return -8 * nn * kq * lq * (l ** 2 / Lam ** 2 + k ** 2 / K ** 2)
    if p.kind == "TM" and q.kind == "TE":
        return 0.0
    return 8 * nn * k * kq * l * lq * chiq ** 2 * (K - Lam) / (K ** 2 * Lam ** 2)


def coupling_matrix(basis):
    """Matrix C (len(basis) x len(basis)) of box T3-2 (vectorized form of coupling_C)."""
    a, b = basis.a, basis.b
    te = np.array([m.kind == "TE" for m in basis])
    m = np.array([md.m for md in basis], float)
    n = np.array([md.n for md in basis], float)
    nu = np.where(te, np.sqrt(np.where(m == 0, 1.0, 2.0) * np.where(n == 0, 1.0, 2.0)), 2.0)
    k, l = m * np.pi / a, n * np.pi / b
    chi = np.hypot(k, l)
    sel = ((m[:, None] + m[None, :]) % 2 == 1) & ((n[:, None] + n[None, :]) % 2 == 1)
    kp, lp, kq, lq = k[:, None], l[:, None], k[None, :], l[None, :]
    K = np.where(sel, kq ** 2 - kp ** 2, 1.0)
    Lam = np.where(sel, lq ** 2 - lp ** 2, 1.0)
    nn = nu[:, None] * nu[None, :] / (a * b * chi[:, None] * chi[None, :])
    br = (kp ** 2 + kq ** 2) * lq ** 2 * Lam - kq ** 2 * (lp ** 2 + lq ** 2) * K
    c_ee = -4 * nn * chi[:, None] ** 2 * br / (K ** 2 * Lam ** 2)
    c_em = -8 * nn * kq * lq * (lp ** 2 / Lam ** 2 + kp ** 2 / K ** 2)
    c_mm = 8 * nn * kp * kq * lp * lq * chi[None, :] ** 2 * (K - Lam) / (K ** 2 * Lam ** 2)
    tp, tq = te[:, None], te[None, :]
    C = np.where(tp & tq, c_ee, np.where(tp & ~tq, c_em, np.where(~tp & ~tq, c_mm, 0.0)))
    return np.where(sel, C, 0.0)


def boundary_S(p, q, a, b):
    """Symmetric part S = (C + C^T)/2 = -1/2 oint (e_p.n)(e_q.n) ((z x rho).n) dl (box T3-3)."""
    p, q = Mode.parse(p), Mode.parse(q)
    if (p.m + q.m) % 2 == 0 or (p.n + q.n) % 2 == 0:
        return 0.0
    return 0.5 * (coupling_C(p, q, a, b) + coupling_C(q, p, a, b))


# ------------------------------------------------------------------------------------- coupled-mode system
def system_matrix(basis, f, theta_prime):
    """2N x 2N matrix M of y' = M y, y = [a+; a-] (CONVENTIONS.md section 5 amplitudes), box T3-1/T3-4.

        da+/dz = -i beta a+ + theta' (K+ a+ + K- a-)
        da-/dz = +i beta a- + theta' (K- a+ + K+ a-)
        K+-_pq = 1/2 (r_pq C_qp -+ r_qp C_pq),   r_pq = sqrt(Y_p) / sqrt(Y_q)   (principal roots)

    equivalent to  dV/dz = -i beta I / Y + theta' C^T V,   dI/dz = -i beta Y V - theta' C I.
    """
    C = coupling_matrix(basis)
    beta = basis.beta(f)
    sY = np.sqrt(basis.admittance(f))
    r = sY[:, None] / sY[None, :]
    Kp = 0.5 * (r * C.T - r.T * C)
    Km = 0.5 * (r * C.T + r.T * C)
    N = len(basis)
    M = np.zeros((2 * N, 2 * N), complex)
    M[:N, :N] = -1j * np.diag(beta) + theta_prime * Kp
    M[:N, N:] = theta_prime * Km
    M[N:, :N] = theta_prime * Km
    M[N:, N:] = 1j * np.diag(beta) + theta_prime * Kp
    return M


def vi_system_matrix(basis, f, theta_prime):
    """The same system for y = [V; I] (voltage/current form, box T3-1)."""
    C = coupling_matrix(basis)
    beta, Y = basis.beta(f), basis.admittance(f)
    N = len(basis)
    M = np.zeros((2 * N, 2 * N), complex)
    M[:N, :N] = theta_prime * C.T
    M[:N, N:] = -1j * np.diag(beta / Y)
    M[N:, :N] = -1j * np.diag(beta * Y)
    M[N:, N:] = -theta_prime * C
    return M


def transfer_matrix(basis, f, theta_prime, length, slices=400, extra=None):
    """Transfer matrix y(L) = Phi y(0) of a twist with rate profile theta_prime(z) (callable, rad/m).

    Fourth-order commutator-free Magnus integrator (exact for a constant rate with slices=1). Only for bases
    of propagating modes (CONVENTIONS.md section 6).
    """
    M0 = system_matrix(basis, f, 0.0)
    M1 = system_matrix(basis, f, 1.0) - M0          # M(z) = M0 + theta'(z) M1
    h = length / slices
    c1, c2 = 0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6
    a1, a2 = 0.25 + np.sqrt(3) / 6, 0.25 - np.sqrt(3) / 6
    Phi = np.eye(2 * len(basis), dtype=complex)
    for s in range(slices):
        z0 = s * h
        A1 = M0 + theta_prime(z0 + c1 * h) * M1
        A2 = M0 + theta_prime(z0 + c2 * h) * M1
        Phi = expm(h * (a2 * A1 + a1 * A2)) @ expm(h * (a1 * A1 + a2 * A2)) @ Phi
    return Phi


# ------------------------------------------------------------------------------------- rate profiles
def rate_profile(kind, theta_tot, L):
    """theta'(z) on [0, L] with integral theta_tot (box T3-6).

    'constant'  theta/L
    'triangular' Doane & Anderson (1989) eq. 17: (2 theta/L)(2z/L) for z < L/2, (2 theta/L)(2(L-z)/L) after
    'cosine'    half-cosine bump (pi theta / 2L) sin(pi z / L)  [= cos(pi (z - L/2)/L)]
    'cos2'      raised cosine (theta/L)(1 - cos(2 pi z / L))
    """
    if kind == "constant":
        return lambda z: theta_tot / L + 0.0 * np.asarray(z)
    if kind == "triangular":
        return lambda z: (2 * theta_tot / L) * np.where(np.asarray(z) < L / 2, 2 * np.asarray(z) / L,
                                                         2 * (L - np.asarray(z)) / L)
    if kind == "cosine":
        return lambda z: (np.pi * theta_tot / (2 * L)) * np.sin(np.pi * np.asarray(z) / L)
    if kind == "cos2":
        return lambda z: (theta_tot / L) * (1 - np.cos(2 * np.pi * np.asarray(z) / L))
    raise ValueError(kind)


def profile_transform(kind, theta_tot, L, dbeta):
    """|F(dbeta)| = |int_0^L theta'(z) exp(i dbeta z) dz| in closed form (box T3-6)."""
    x = np.asarray(dbeta, dtype=float) * L
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "constant":
            out = np.abs(np.sinc(x / (2 * np.pi)))
        elif kind == "triangular":
            out = np.sinc(x / (4 * np.pi)) ** 2
        elif kind == "cosine":
            out = np.where(np.isclose(np.abs(x), np.pi), np.pi / 4,
                           np.pi ** 2 * np.abs(np.cos(x / 2)) / np.abs(np.pi ** 2 - x ** 2))
        elif kind == "cos2":
            y = x / 2
            out = np.where(np.isclose(np.abs(y), np.pi), 0.5,
                           np.abs(np.sinc(y / np.pi)) * np.pi ** 2 / np.abs(np.pi ** 2 - y ** 2))
        else:
            raise ValueError(kind)
    return theta_tot * out


def first_order_power(basis, f, kind, theta_tot, L, incident):
    """First-order converted power |a+_p(L)|^2 = |K+_pq|^2 |F(beta_p - beta_q)|^2 for every mode p (box T3-6)."""
    q = basis.index(incident)
    C = coupling_matrix(basis)
    beta = basis.beta(f).real
    sY = np.sqrt(basis.admittance(f).real)
    Kp = 0.5 * (sY / sY[q] * C[q, :] - sY[q] / sY * C[:, q])
    P = np.abs(Kp) ** 2 * profile_transform(kind, theta_tot, L, beta - beta[q]) ** 2
    P[q] = 0.0
    return P


def doane_eq18(c, theta, dbeta_L):
    """Doane & Anderson (1989) eq. 18, transcribed from the scan: P_c ~ (8 c theta)^2 / (dbeta L_t)^4."""
    return (8 * c * theta) ** 2 / np.asarray(dbeta_L, dtype=float) ** 4


# ------------------------------------------------------------------------------------- frame changes
def rotate_mode_90(mode, a, b, quarter_turns):
    """Image of mode (m, n) of an a x b guide under a rotation of the local axes by quarter_turns * 90 deg.

    The new axes are u' = R(phi) u, v' = R(phi) v (phi = quarter_turns * pi/2, counterclockwise about +z). The
    same physical field has amplitude sign * a on mode `new` of the guide with sides (a', b') along (u', v').
    Returns (new_mode, sign, a', b') (box T3-5).
    """
    t = quarter_turns % 4
    m, n = mode.m, mode.n
    if t == 0:
        return mode, 1.0, a, b
    if t == 2:
        return mode, (-1.0) ** (m + n), a, b
    if t == 1:   # u' = R(90) u = old v axis, v' = -old u axis: old (u, v) = (-v', u')
        sign = (-1.0) ** m if mode.kind == "TE" else (-1.0) ** (m + 1)
        return Mode(mode.kind, n, m), sign, b, a
    # t == 3: u' = -old v axis, v' = old u axis: old (u, v) = (v', -u')
    sign = (-1.0) ** n if mode.kind == "TE" else (-1.0) ** (n + 1)
    return Mode(mode.kind, n, m), sign, b, a


def frame_change_matrix(basis, quarter_turns, frame=None):
    """(F, basis_to): a_to = F a_from for a rotation of the local axes by quarter_turns * 90 degrees."""
    imgs = [rotate_mode_90(m, basis.a, basis.b, quarter_turns) for m in basis]
    a2, b2 = imgs[0][2], imgs[0][3]
    to = ModeBasis(a2, b2, [im[0] for im in imgs], frame=frame)
    F = np.zeros((len(to), len(basis)))
    for j, (mode, s, _, _) in enumerate(imgs):
        F[to.index(mode), j] = s
    return F, to


def zero_length_twist(basis, theta_tot, f=1.0):
    """Transfer matrix of a twist by theta_tot in the limit L -> 0 (propagation negligible): exp(theta_tot M1)."""
    M1 = system_matrix(basis, f, 1.0) - system_matrix(basis, f, 0.0)
    return expm(theta_tot * M1)
