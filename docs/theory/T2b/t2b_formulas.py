"""Machine-checkable versions of the boxed formulas of docs/theory/T2b-taper-coupling.md (card T2b).

Conventions: CONVENTIONS.md (fields of wrmodes.modes, time dependence exp(+i w t), forward modes exp(-i beta z),
amplitudes of section 5). Local cross-section of the taper at abscissa z:

    u in [u0(z) - a(z)/2, u0(z) + a(z)/2],  v in [v0(z) - b(z)/2, v0(z) + b(z)/2],

with (u, v, z) the frame of the curved centre line, scale factor h = 1 + kappa(z) u (u measured from the
reference line of the curvature, so the guide axis is at u = u0). The local mode fields are those of
wrmodes.modes evaluated at (u - u0, v - v0) with the local a, b.

Equations (boxed results B1-B5 of the note):

    dV/dz = -i Z I + T V,          dI/dz = -i Y V - T^T I                                   (B1)
    Z = w mu0 H - H^d / (w eps0),   Y = w eps0 H - H^g / (w mu0)                              (B2)
    H_pq = int h e_p.e_q,  H^d_pq = int h d_p d_q,  H^g_pq = int h g_p g_q  (over the local rectangle)
    T_pq: closed forms in `T_closed` (B3)
    power waves: `power_wave_matrices` (B4)

Everything here is written for checking, not for speed.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.constants as cst

from wrmodes.modes import Mode, ETA0, transverse_field, divergence

MU0 = cst.mu_0


def eps(l):
    return 1 if l == 0 else 2


def _sqrt(x):
    """sqrt that also accepts sympy expressions (for the symbolic check)."""
    if type(x).__module__.startswith("sympy"):
        import sympy
        return sympy.sqrt(x)
    return np.sqrt(x)


def Nnorm(p, a, b):
    """N_p = sqrt(m^2 b^2 + n^2 a^2), so that chi_p = pi N_p / (a b)."""
    return _sqrt(p.m ** 2 * b ** 2 + p.n ** 2 * a ** 2)


def chi2(p, a, b):
    return np.pi ** 2 * (p.m ** 2 / a ** 2 + p.n ** 2 / b ** 2)


def Omega(i, j, d_size, d_off):
    """Wall-velocity combination Omega(i, j) = W_minus + (-1)^(i+j) W_plus (box B3).

    W_plus = d_off + d_size/2 and W_minus = d_size/2 - d_off are the outward normal velocities (wall slopes) of the
    two walls perpendicular to the direction considered (u: d_size = a', d_off = u0'; v: d_size = b', d_off = v0').
    Equal to d_size when i + j is even and to -2 d_off when i + j is odd."""
    return d_size if (i + j) % 2 == 0 else -2 * d_off


# --------------------------------------------------------------------------------------------- box B3
def T_closed(p, q, a, b, da, db, du0, dv0):
    """Cross-section coupling coefficient T_pq of dV_p/dz = ... + sum_q T_pq V_q (box B3).

    Frequency independent, real, linear in the four wall slopes a', b', u0', v0' (1/m).  Equals
    int_R(z) e_q . (d e_p/dz)|_(u,v fixed) du dv  (fixed integration domain R(z)), see the note.
    """
    p, q = Mode.parse(p), Mode.parse(q)
    m, n, mq, nq = p.m, p.n, q.m, q.n
    Np, Nq = Nnorm(p, a, b), Nnorm(q, a, b)
    if p == q:
        if p.kind == "TE":
            return -eps(m) * eps(n) / (4 * Np ** 2) * (n ** 2 * a * da + m ** 2 * b * db)
        return -(m ** 2 * b ** 2 * da / a + n ** 2 * a ** 2 * db / b) / Np ** 2
    du = (n == nq)            # coupling through the u-walls (a', u0') conserves n
    dv = (m == mq)            # coupling through the v-walls (b', v0') conserves m
    if not (du or dv):
        return 0              # selection rule: one of the two indices must be shared
    Ou = Omega(m, mq, da, du0) if du else 0.0
    Ov = Omega(n, nq, db, dv0) if dv else 0.0
    if p.kind == "TE" and q.kind == "TE":
        s = (_sqrt(eps(m) * eps(mq)) * b ** 2 * m ** 2 / a * Ou if du else 0.0) \
            + (_sqrt(eps(n) * eps(nq)) * a ** 2 * n ** 2 / b * Ov if dv else 0.0)
        return -Nq ** 2 / (Np ** 2 - Nq ** 2) * s / (Np * Nq)
    if p.kind == "TM" and q.kind == "TM":
        s = (m * mq * b ** 2 / a * Ou if du else 0.0) + (n * nq * a ** 2 / b * Ov if dv else 0.0)
        return -Np ** 2 / (Np ** 2 - Nq ** 2) * 2 * s / (Np * Nq)
    if p.kind == "TM" and q.kind == "TE":
        return a * b * _sqrt(eps(mq) * eps(nq)) * m * n / (Np * Nq) * \
            ((Ou / a if du else 0.0) - (Ov / b if dv else 0.0))
    return 0                  # p TE, q TM


def T_matrix(modes, a, b, da, db, du0, dv0):
    modes = [Mode.parse(x) for x in modes]
    return np.array([[T_closed(p, q, a, b, da, db, du0, dv0) for q in modes] for p in modes])


# --------------------------------------------------------------------------------------------- route of the note
def _gl(nq):
    x, w = leggauss(nq)
    return 0.5 * x, 0.5 * w        # nodes and weights on [-1/2, 1/2]


def T_transformed(p, q, a, b, da, db, du0, dv0, nq=64):
    """T_pq from the coordinate-transformation route (section 3 of the note), by quadrature on the unit square:

        T_pq = - int (w . e_q) d_p  + int g_q (w_v e_pu - w_u e_pv)  + int eps_q . d_zeta eps*_p  dxi deta,

    first two in physical measure du dv = a b dxi deta, w = (u0' + a' xi, v0' + b' eta), covariant fields
    eps = (a e_u, b e_v), density fields eps* = (b e_u, a e_v), d_zeta at fixed (xi, eta) (complex step in a, b).
    """
    p, q = Mode.parse(p), Mode.parse(q)
    x, wx = _gl(nq)
    XI, ETA = np.meshgrid(x, x, indexing="ij")
    W = np.outer(wx, wx)
    U, V = a * XI, b * ETA                  # centred local coordinates (the offset does not change the shapes)
    epu, epv = transverse_field(p, a, b, U, V)
    equ, eqv = transverse_field(q, a, b, U, V)
    dp, _ = divergence(p, a, b, U, V)
    _, gq = divergence(q, a, b, U, V)
    wu, wv = du0 + da * XI, dv0 + db * ETA
    t1 = -np.sum(W * (wu * equ + wv * eqv) * dp) * a * b
    t2 = np.sum(W * gq * (wv * epu - wu * epv)) * a * b

    def epsstar(aa, bb):
        fu, fv = transverse_field(p, aa, bb, aa * XI, bb * ETA)
        return bb * fu, aa * fv
    h = 1e-30
    sa = epsstar(a + 1j * h * da, b + 1j * h * db)
    dsu, dsv = sa[0].imag / h, sa[1].imag / h     # complex step: exact directional derivative a' d_a + b' d_b
    t3 = np.sum(W * (a * equ * dsu + b * eqv * dsv))
    return t1 + t2 + t3


def T_overlap(p, q, geom, z, nq=64, moving=False, method="complex"):
    """Cross-section route: derivative of the overlap O_pq(z') = int_D e_q(z) . e_p(z') du dv at z' = z.

    geom(z) -> (a, b, u0, v0) (complex-step safe). D = R(z) (fixed domain, default) or R(z') (moving=True).
    Fields outside their own rectangle are the analytic continuation of the trigonometric formulas.
    """
    p, q = Mode.parse(p), Mode.parse(q)
    x, wx = _gl(nq)

    def O(zp):
        a, b, u0, v0 = geom(z)
        ap, bp, u0p, v0p = geom(zp)
        A, B, U0, V0 = (ap, bp, u0p, v0p) if moving else (a, b, u0, v0)
        U = U0 + A * x[:, None]
        V = V0 + B * x[None, :]
        equ, eqv = transverse_field(q, a, b, U - u0, V - v0)
        epu, epv = transverse_field(p, ap, bp, U - u0p, V - v0p)
        return np.sum(np.outer(wx, wx) * (equ * epu + eqv * epv)) * A * B
    if method == "complex":
        h = 1e-30
        return O(z + 1j * h).imag / h
    hs = 2e-4
    d1 = (O(z + hs) - O(z - hs)) / (2 * hs)
    d2 = (O(z + hs / 2) - O(z - hs / 2)) / hs
    return ((4 * d2 - d1) / 3).real


# --------------------------------------------------------------------------------------------- box B2 (curvature)
def curvature_integrals(modes, a, b, u0, nq=64):
    """H, H^d, H^g with h = 1 + kappa u split as H = H0 + kappa H1 (u = u0 + local centred u)."""
    modes = [Mode.parse(x) for x in modes]
    x, wx = _gl(nq)
    U, V = np.meshgrid(a * x, b * x, indexing="ij")
    W = np.outer(wx, wx) * a * b
    E = [transverse_field(p, a, b, U, V) for p in modes]
    D = [divergence(p, a, b, U, V) for p in modes]
    M = len(modes)
    out = {k: np.zeros((M, M)) for k in ("H0", "H1", "Hd0", "Hd1", "Hg0", "Hg1")}
    for i in range(M):
        for j in range(M):
            ee = E[i][0] * E[j][0] + E[i][1] * E[j][1]
            dd = D[i][0] * D[j][0]
            gg = D[i][1] * D[j][1]
            for key, f in (("H", ee), ("Hd", dd), ("Hg", gg)):
                out[key + "0"][i, j] = np.sum(W * f)
                out[key + "1"][i, j] = np.sum(W * (u0 + U) * f)
    return out


def ZY(modes, f, a, b, u0=0.0, kappa=0.0, nq=64):
    """Z and Y of box B2 (ohm, S per unit... per mode), with h = 1 + kappa u over the local rectangle."""
    w = 2 * np.pi * f
    k = w / cst.c
    I = curvature_integrals(modes, a, b, u0, nq)
    H = I["H0"] + kappa * I["H1"]
    Hd = I["Hd0"] + kappa * I["Hd1"]
    Hg = I["Hg0"] + kappa * I["Hg1"]
    # w mu0 = k eta0, w eps0 = k / eta0 exactly (CONVENTIONS.md 1)
    Z = k * ETA0 * H - Hd * ETA0 / k
    Y = k / ETA0 * H - Hg / (k * ETA0)
    return Z, Y


def system_matrix(modes, f, a, b, da, db, u0=0.0, du0=0.0, dv0=0.0, kappa=0.0, nq=64):
    """2N x 2N matrix M of dy/dz = M y, y = [V; I] (box B1). Regular through cutoff (no beta anywhere)."""
    Z, Y = ZY(modes, f, a, b, u0, kappa, nq)
    T = T_matrix(modes, a, b, da, db, du0, dv0)
    return np.block([[T, -1j * Z], [-1j * Y, -T.T]])


# --------------------------------------------------------------------------------------------- box B4
def beta_Y(modes, f, a, b):
    k = 2 * np.pi * f / cst.c
    out_b, out_y = [], []
    for p in modes:
        p = Mode.parse(p)
        d = k * k - chi2(p, a, b)
        bt = np.sqrt(d) + 0j if d >= 0 else -1j * np.sqrt(-d)
        out_b.append(bt)
        out_y.append(bt / (k * ETA0) if p.kind == "TE" else k / (ETA0 * bt))
    return np.array(out_b), np.array(out_y)


def dlogY(p, f, a, b, da, db):
    """Y_p'/Y_p = -/+ (chi^2)' / (2 beta^2) for TE / TM."""
    p = Mode.parse(p)
    k = 2 * np.pi * f / cst.c
    dchi2 = -2 * np.pi ** 2 * (p.m ** 2 * da / a ** 3 + p.n ** 2 * db / b ** 3)
    beta2 = k * k - chi2(p, a, b)
    return (-1 if p.kind == "TE" else 1) * dchi2 / (2 * beta2)


def power_wave_matrices(modes, f, a, b, da, db, du0, dv0):
    """A, B of box B4: taper part of da+/dz = ... + A a+ + B a-,  da-/dz = ... + B a+ + A a-.

    P = s T s^-1 with s = diag(sqrt(Y)) (principal roots); A = (P - P^T)/2; B = diag(Y'/2Y) + (P + P^T)/2.
    """
    modes = [Mode.parse(x) for x in modes]
    T = T_matrix(modes, a, b, da, db, du0, dv0)
    _, Yv = beta_Y(modes, f, a, b)
    s = np.sqrt(Yv)
    P = s[:, None] * T / s[None, :]
    D = np.diag([dlogY(p, f, a, b, da, db) / 2 for p in modes])
    return 0.5 * (P - P.T), D + 0.5 * (P + P.T)


def T_block(rows, cols, a, b, da, db, du0, dv0):
    """Vectorized T_closed: matrix T[p, q] for p in rows, q in cols (same formulas, checked against T_closed)."""
    rows = [Mode.parse(x) for x in rows]
    cols = [Mode.parse(x) for x in cols]
    kp = np.array([p.kind == "TE" for p in rows])[:, None]
    kq = np.array([q.kind == "TE" for q in cols])[None, :]
    m = np.array([p.m for p in rows], float)[:, None]
    n = np.array([p.n for p in rows], float)[:, None]
    mq = np.array([q.m for q in cols], float)[None, :]
    nq = np.array([q.n for q in cols], float)[None, :]
    ep = lambda l: np.where(l == 0, 1.0, 2.0)
    Np = np.sqrt(m ** 2 * b ** 2 + n ** 2 * a ** 2)
    Nq = np.sqrt(mq ** 2 * b ** 2 + nq ** 2 * a ** 2)
    du = (n == nq)
    dv = (m == mq)
    Ou = np.where(du, np.where((m + mq) % 2 == 0, da, -2 * du0), 0.0)
    Ov = np.where(dv, np.where((n + nq) % 2 == 0, db, -2 * dv0), 0.0)
    same = du & dv & (kp == kq)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.where(same, 1.0, Np ** 2 - Nq ** 2)
        tete = -Nq ** 2 / den * (np.sqrt(ep(m) * ep(mq)) * b ** 2 * m ** 2 / a * Ou
                                 + np.sqrt(ep(n) * ep(nq)) * a ** 2 * n ** 2 / b * Ov) / (Np * Nq)
        tmtm = -Np ** 2 / den * 2 * (m * mq * b ** 2 / a * Ou + n * nq * a ** 2 / b * Ov) / (Np * Nq)
        tmte = a * b * np.sqrt(ep(mq) * ep(nq)) * m * n / (Np * Nq) * (Ou / a - Ov / b)
        dte = -ep(m) * ep(n) / (4 * Np ** 2) * (n ** 2 * a * da + m ** 2 * b * db)
        dtm = -(m ** 2 * b ** 2 * da / a + n ** 2 * a ** 2 * db / b) / Np ** 2
    T = np.where(kp & kq, tete, 0.0)
    T = np.where(~kp & ~kq, tmtm, T)
    T = np.where(~kp & kq, tmte, T)
    T = np.where(same & kp, dte, T)
    T = np.where(same & ~kp, dtm, T)
    return np.where(np.isfinite(T), T, 0.0) * (du | dv)
