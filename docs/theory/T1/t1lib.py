"""Reference implementation of the formulas of docs/theory/T1-wall-impedance.md (theory card T1).

This module is the machine-checkable form of the boxed results of the note. It is not part of the wrmodes
package: WP2 implements the same formulas in `wrmodes.loss`, and the scripts of this folder check both the
formulas (sympy, quadrature) and, later, WP2's implementation against these functions.

Conventions: CONVENTIONS.md (e^{+i omega t}, forward modes e^{-i beta z}, fields of wrmodes.modes, amplitudes
normalized with the unconjugated reaction). Local frame (u, v, z), u in [-a/2, a/2], v in [-b/2, b/2];
u' = u + a/2, v' = v + b/2.

Walls are labelled by their outward normal (pointing from the guide into the wall):
    'u+' : u = +a/2, n = +u_hat (outer wall of a bend with kappa > 0)
    'u-' : u = -a/2, n = -u_hat
    'v+' : v = +b/2, n = +v_hat
    'v-' : v = -b/2, n = -v_hat
A wall impedance is given as a dict {wall: Z} (ohm); walls that are not listed are perfect conductors.

Main results (first order in Z, exact in the bend factor h = 1 + kappa u):

    dV_p/dz = [T0 terms] - sum_q N_pq I_q ,   N_pq = sum_w Z_w  oint_w h (e_p . n)(e_q . n) dl
    dI_p/dz = [T0 terms] - sum_q G_pq V_q ,   G_pq = sum_w Z_w  oint_w h g_p g_q dl / (k eta0)^2

    g_p = div(z_hat x e_p) = -chi_p^2 Gamma_p (TE), 0 (TM).

    Lambda^{+-}_pq = 1/2 [ +- sqrt(Y_p) sqrt(Y_q) N_pq + G_pq / (sqrt(Y_p) sqrt(Y_q)) ]   (principal roots)

    da+/dz = -i beta a+ - Lambda^+ a+ - Lambda^- a-
    da-/dz = +i beta a- + Lambda^- a+ + Lambda^+ a-
    dy/dz  = (D + kappa K - Lambda_full) y,  Lambda_full = [[Lambda^+, Lambda^-], [-Lambda^-, -Lambda^+]]

i.e. in the card's notation Lambda^{++} = Lambda^{--} = Lambda^+ and Lambda^{+-} = Lambda^{-+} = Lambda^-.
With h = 1 + kappa u, Lambda = Lambda0 + kappa Lambda1, both constant.
"""
import numpy as np
import scipy.constants as cst
from numpy.polynomial.legendre import leggauss

from wrmodes.modes import (ETA0,  # noqa: F401
                            Mode, ModeBasis, cutoff_wavenumber, propagation_constant, wave_admittance,
                           transverse_field, divergence)

WALLS = ("u+", "u-", "v+", "v-")


def wavenumber(f):
    return 2 * np.pi * f / cst.c


def surface_resistance(f, sigma):
    """R_s = sqrt(omega mu0 / (2 sigma)) (ohm)."""
    return np.sqrt(2 * np.pi * f * cst.mu_0 / (2 * sigma))


def surface_impedance(f, sigma):
    """Z_s = (1 + i) R_s for e^{+i omega t} (CONVENTIONS.md section 1)."""
    return (1 + 1j) * surface_resistance(f, sigma)


def all_walls(Z):
    return {w: Z for w in WALLS}


# ------------------------------------------------------------------------------------------ mode constants
def _eps(l):
    return 1.0 if l == 0 else 2.0


def mode_constants(mode, a, b):
    """(c, km, kn, chi2): normalization constant of Pi or Gamma (c / chi = prefactor of eq. 5.4), k_m, k_n, chi^2."""
    mode = Mode.parse(mode)
    m, n = mode.m, mode.n
    chi = cutoff_wavenumber(mode, a, b)
    c = (2.0 if mode.kind == "TM" else np.sqrt(_eps(m) * _eps(n))) / (np.sqrt(a * b) * chi)
    return c, m * np.pi / a, n * np.pi / b, chi * chi


def _A_side(mode, a, b):
    """e_u at u' = 0 equals A sin(k_n v'); at u' = a it is (-1)^m A sin(k_n v')."""
    c, km, kn, _ = mode_constants(mode, a, b)
    return c * km if mode.kind == "TM" else -c * kn


def _B_broad(mode, a, b):
    """e_v at v' = 0 equals B sin(k_m u'); at v' = b it is (-1)^n B sin(k_m u')."""
    c, km, kn, _ = mode_constants(mode, a, b)
    return c * kn if mode.kind == "TM" else c * km


# 1D integrals over [0, L] of products of sin / cos with integer half periods, unweighted and weighted
# with (x - L/2).
def _ss0(p, q, L):
    return L / 2 if (p == q and p >= 1) else 0.0


def _cc0(p, q, L):
    return (L if p == 0 else L / 2) if p == q else 0.0


def _ss1(p, q, L):
    if (p + q) % 2 == 0:
        return 0.0
    return -(L / np.pi) ** 2 * 4.0 * p * q / (p * p - q * q) ** 2


def _cc1(p, q, L):
    if (p + q) % 2 == 0:
        return 0.0
    return -(L / np.pi) ** 2 * 2.0 * (p * p + q * q) / (p * p - q * q) ** 2


# ------------------------------------------------------------------------------------------ wall integrals
def wall_integrals(p, q, a, b, wall, order=0):
    """Closed forms of the wall integrals of one wall (boxed result B2 of the note).

    Returns (Nw, Gw):
        Nw = oint_wall u^order (e_p . n)(e_q . n) dl      (1/m for order 0, dimensionless for order 1)
        Gw = oint_wall u^order g_p g_q dl                  (1/m^3 for order 0)
    with u the centred coordinate (u = u' - a/2). order 0 is the straight guide; order 1 is the coefficient of
    kappa in a bend, where h = 1 + kappa u.
    """
    p, q = Mode.parse(p), Mode.parse(q)
    cp, _, _, chi2p = mode_constants(p, a, b)
    cq, _, _, chi2q = mode_constants(q, a, b)
    te = p.kind == "TE" and q.kind == "TE"
    if wall in ("u+", "u-"):
        if p.n != q.n:
            return 0.0, 0.0
        sig = (-1.0) ** (p.m + q.m) if wall == "u+" else 1.0
        w = 1.0 if order == 0 else (a / 2 if wall == "u+" else -a / 2)
        N = _A_side(p, a, b) * _A_side(q, a, b) * sig * _ss0(p.n, q.n, b) * w
        G = (chi2p * chi2q * cp * cq * sig * _cc0(p.n, q.n, b) * w) if te else 0.0
        return N, G
    if wall in ("v+", "v-"):
        tau = (-1.0) ** (p.n + q.n) if wall == "v+" else 1.0
        if order == 0:
            S, C = _ss0(p.m, q.m, a), _cc0(p.m, q.m, a)
        else:
            S, C = _ss1(p.m, q.m, a), _cc1(p.m, q.m, a)
        N = _B_broad(p, a, b) * _B_broad(q, a, b) * tau * S
        G = (chi2p * chi2q * cp * cq * tau * C) if te else 0.0
        return N, G
    raise ValueError(f"unknown wall {wall!r}")


def wall_integrals_quadrature(p, q, a, b, wall, order=0, npts=200):
    """The same integrals by Gauss-Legendre quadrature of the fields of wrmodes.modes (independent check)."""
    p, q = Mode.parse(p), Mode.parse(q)
    x, w = leggauss(npts)
    if wall in ("u+", "u-"):
        u0 = a / 2 if wall == "u+" else -a / 2
        nrm = (1.0, 0.0) if wall == "u+" else (-1.0, 0.0)
        v = 0.5 * b * x
        u = np.full_like(v, u0)
        wt = 0.5 * b * w * (u0 ** order)
    else:
        v0 = b / 2 if wall == "v+" else -b / 2
        nrm = (0.0, 1.0) if wall == "v+" else (0.0, -1.0)
        u = 0.5 * a * x
        v = np.full_like(u, v0)
        wt = 0.5 * a * w * (u ** order)
    ep = transverse_field(p, a, b, u, v)
    eq = transverse_field(q, a, b, u, v)
    epn = ep[0] * nrm[0] + ep[1] * nrm[1]
    eqn = eq[0] * nrm[0] + eq[1] * nrm[1]
    gp = divergence(p, a, b, u, v)[1]
    gq = divergence(q, a, b, u, v)[1]
    return float(np.sum(wt * epn * eqn)), float(np.sum(wt * gp * gq))


def wall_matrices_quadrature(basis, wall, order=0, npts=400):
    """All (Nw, Gw) of one wall for a basis at once, by Gauss-Legendre quadrature (vectorized)."""
    a, b = basis.a, basis.b
    x, w = leggauss(npts)
    if wall in ("u+", "u-"):
        u0 = a / 2 if wall == "u+" else -a / 2
        nrm = (1.0, 0.0) if wall == "u+" else (-1.0, 0.0)
        v = 0.5 * b * x
        u = np.full_like(v, u0)
        wt = 0.5 * b * w * (u0 ** order)
    else:
        v0 = b / 2 if wall == "v+" else -b / 2
        nrm = (0.0, 1.0) if wall == "v+" else (0.0, -1.0)
        u = 0.5 * a * x
        v = np.full_like(u, v0)
        wt = 0.5 * a * w * (u ** order)
    fields = [transverse_field(p, a, b, u, v) for p in basis.modes]
    En = np.array([e[0] * nrm[0] + e[1] * nrm[1] for e in fields])
    Gn = np.array([divergence(p, a, b, u, v)[1] for p in basis.modes])
    return (En * wt) @ En.T, (Gn * wt) @ Gn.T


def wall_matrices(basis, wall, order=0):
    """All (Nw, Gw) of one wall for a basis, from the closed forms."""
    n = len(basis)
    Nw, Gw = np.zeros((n, n)), np.zeros((n, n))
    for i, p in enumerate(basis.modes):
        for j, q in enumerate(basis.modes[: i + 1]):
            Nw[i, j], Gw[i, j] = wall_integrals(p, q, basis.a, basis.b, wall, order)
            Nw[j, i], Gw[j, i] = Nw[i, j], Gw[i, j]
    return Nw, Gw


# ------------------------------------------------------------------------------------------ N, G, Lambda
def NG_matrices(basis, f, Z, order=0, quadrature=False):
    """N and G matrices (N x N, complex) of a basis for wall impedances Z = {wall: Z_w}.

    N_pq = sum_w Z_w Nw_pq (ohm/m),  G_pq = sum_w Z_w Gw_pq / (k eta0)^2 (S/m).
    """
    k = wavenumber(f)
    n = len(basis)
    N = np.zeros((n, n), dtype=complex)
    G = np.zeros((n, n), dtype=complex)
    for wall, Zw in Z.items():
        if Zw == 0:
            continue
        Nw, Gw = wall_matrices_quadrature(basis, wall, order) if quadrature else wall_matrices(basis, wall, order)
        N += Zw * Nw
        G += Zw * Gw / (k * ETA0) ** 2
    return N, G


def loss_matrices(basis, f, Z, order=0, quadrature=False):
    """(Lambda^+, Lambda^-), N x N complex symmetric (boxed result B3).

    order=0: straight guide (Lambda0); order=1: coefficient of kappa in a bend (Lambda1).
    """
    N, G = NG_matrices(basis, f, Z, order, quadrature)
    sY = np.sqrt(basis.admittance(f))            # principal square roots, as in CONVENTIONS.md section 5
    YY = np.outer(sY, sY)
    Lp = 0.5 * (YY * N + G / YY)
    Lm = 0.5 * (-YY * N + G / YY)
    return Lp, Lm


def loss_matrix_full(basis, f, Z, kappa=0.0, quadrature=False):
    """Lambda_full (2N x 2N) such that dy/dz = (D + kappa K - Lambda_full) y, y = [a+; a-] (boxed result B4).

    Includes the bend factor h = 1 + kappa u on the wall integrals for a constant kappa (Lambda0 + kappa Lambda1).
    """
    Lp, Lm = loss_matrices(basis, f, Z, 0, quadrature)
    if kappa != 0.0:
        Lp1, Lm1 = loss_matrices(basis, f, Z, 1, quadrature)
        Lp, Lm = Lp + kappa * Lp1, Lm + kappa * Lm1
    return np.block([[Lp, Lm], [-Lm, -Lp]])


def D_matrix(basis, f):
    beta = basis.beta(f)
    return np.diag(np.concatenate([-1j * beta, 1j * beta]))


# ------------------------------------------------------------------------------------------ textbook values
def attenuation_marcuvitz(mode, a, b, f, Rs):
    """Attenuation constant (Np/m) of a mode in a guide with four walls of surface resistance Rs.

    Marcuvitz, Waveguide Handbook (1951), sec. 2.2, eq. (4) (E_mn = TM) and eq. (10) (H_mn = TE), with
    zeta = eta0 and the Neumann factors eps_0 = 1, eps_l = 2 (x along a, y along b).
    """
    mode = Mode.parse(mode)
    m, n = mode.m, mode.n
    x2 = (cutoff_wavenumber(mode, a, b) / wavenumber(f)) ** 2        # (lambda / lambda_c)^2
    s = np.sqrt(1 - x2)
    if mode.kind == "TM":
        return 2 * Rs / (ETA0 * a) * (m * m + n * n * a ** 3 / b ** 3) / (m * m + n * n * a * a / (b * b)) / s
    em, en = _eps(m), _eps(n)
    return Rs / (ETA0 * b) * ((en * m * m * b / a + em * n * n) / (m * m * b / a + n * n * a / b) * s
                              + (en + em * b / a) * x2 / s)


def attenuation_te10_card(a, b, f, Rs):
    """The card's TE10 formula: alpha = Rs / (eta0 b sqrt(1 - (fc/f)^2)) [1 + (2b/a)(fc/f)^2]."""
    fc = cst.c / (2 * a)
    x2 = (fc / f) ** 2
    return Rs / (ETA0 * b * np.sqrt(1 - x2)) * (1 + 2 * b / a * x2)


def ls_first_order(mode_m, mode_n, a, b, f, Z):
    """First-order loss eigenvalues Lambda of the LS modes for one pair of opposite walls (boxed result B5).

    Returns dict with keys 'LSE_u', 'LSM_u' (side walls u = +-a/2 with impedances Z['u+'], Z['u-']) and
    'LSE_v', 'LSM_v' (broad walls v = +-b/2), each the eigenvalue Lambda (da/dz = -(i beta + Lambda) a).
    """
    k = wavenumber(f)
    m, n = mode_m, mode_n
    km, kn = m * np.pi / a, n * np.pi / b
    beta = np.sqrt(k * k - km * km - kn * kn + 0j)
    Zu = Z.get("u+", 0) + Z.get("u-", 0)
    Zv = Z.get("v+", 0) + Z.get("v-", 0)
    out = {}
    out["LSE_u"] = km * km * Zu / (ETA0 * k * a * beta)                     # E_u = 0 (TE^u), m >= 1
    out["LSM_u"] = k * Zu / (ETA0 * a * beta) * (0.5 if m == 0 else 1.0)    # H_u = 0 (TM^u), n >= 1
    out["LSE_v"] = kn * kn * Zv / (ETA0 * k * b * beta)                     # E_v = 0 (TE^v), n >= 1
    out["LSM_v"] = k * Zv / (ETA0 * b * beta) * (0.5 if n == 0 else 1.0)    # H_v = 0 (TM^v), m >= 1
    return out


# ------------------------------------------------------------------------------------------ exact LS^u modes
def lsu_dispersion(ku, k, a, pp, pm):
    """F(k_u) = (k_u^2 + k^2 p+ p-) sin(k_u a) - i k k_u (p+ + p-) cos(k_u a) (boxed result B6).

    p = eta0 / Z for LSE^u (TE to u, E_u = 0), p = Z / eta0 for LSM^u (TM to u, H_u = 0); p+ at u = +a/2,
    p- at u = -a/2. Use pp = pm = np.inf (LSE^u) or 0 (LSM^u) for a perfect conductor.
    """
    return (ku * ku + k * k * pp * pm) * np.sin(ku * a) - 1j * k * ku * (pp + pm) * np.cos(ku * a)


def lsu_dispersion_scaled(ku, k, a, Zp, Zm, kind):
    """F scaled to stay finite when a wall impedance is 0: for LSE^u, F * Z+ Z- / eta0^2."""
    if kind == "LSE":
        zp, zm = Zp / ETA0, Zm / ETA0
        return (zp * zm * ku * ku + k * k) * np.sin(ku * a) - 1j * k * ku * (zm + zp) * np.cos(ku * a)
    return lsu_dispersion(ku, k, a, Zp / ETA0, Zm / ETA0)


def _newton(F, x0, tol=1e-14, maxit=100, h=1e-7):
    x = complex(x0)
    for _ in range(maxit):
        d = h * max(1.0, abs(x))
        fx = F(x)
        dfx = (F(x + d) - F(x - d)) / (2 * d)
        dx = fx / dfx
        x -= dx
        if abs(dx) < tol * max(1.0, abs(x)):
            return x
    raise RuntimeError("Newton did not converge")


def lsu_root(m, k, a, Zp, Zm, kind, steps=60):
    """k_u of the LS^u mode that tends to m pi / a for Z -> 0 (continuation in Z from 0)."""
    if m == 0 and kind != "LSM":
        raise ValueError("LSE^u modes need m >= 1")
    ts = np.linspace(0, 1, steps + 1)[1:]
    ku = m * np.pi / a
    if m == 0:   # first-order root k_u^2 = i k (Z+ + Z-) / (eta0 a) of the first step
        ku = np.sqrt(1j * k * ts[0] * (Zp + Zm) / (ETA0 * a))
    for t in ts:
        if m == 0:
            # k_u = 0 is a trivial root of F (no field); divide it out
            F = lambda x, t=t: lsu_dispersion_scaled(x, k, a, t * Zp, t * Zm, kind) / x
        else:
            F = lambda x, t=t: lsu_dispersion_scaled(x, k, a, t * Zp, t * Zm, kind)
        ku = _newton(F, ku)
    return ku


def _cheb(N, a):
    """Chebyshev points on [0, a] and first-derivative matrix (Trefethen, Spectral Methods in MATLAB, cheb.m)."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    Dm = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    Dm = Dm - np.diag(Dm.sum(axis=1))
    # map [-1, 1] -> [0, a]: u = a (1 - x) / 2 (so that u increases with the index)
    return a * (1 - x) / 2, -2 / a * Dm


def lsu_roots_all(M, k, a, Zp, Zm, kind, N=400):
    """The M LS^u roots k_u (Re k_u ascending) at the given wall impedances, robust for any |Z| and high orders.

    Chebyshev collocation of -X'' = k_u^2 X with the Leontovich (Robin) conditions
        LSE^u: Z- X'(0) = i k eta0 X(0),  Z+ X'(a) = -i k eta0 X(a)
        LSM^u: eta0 X'(0) = i k Z- X(0),  eta0 X'(a) = -i k Z+ X(a)
    (u' = u + a/2), then Newton polishing on the dispersion function F(k_u)/k_u.
    """
    u, D1 = _cheb(N, a)
    D2 = D1 @ D1
    if kind == "LSE":
        c0, d0 = Zm * D1[0], -1j * k * ETA0           # Z- X'(0) - i k eta X(0) = 0
        cN, dN = Zp * D1[-1], 1j * k * ETA0           # Z+ X'(a) + i k eta X(a) = 0
    else:
        c0, d0 = ETA0 * D1[0], -1j * k * Zm
        cN, dN = ETA0 * D1[-1], 1j * k * Zp
    # boundary rows: r0 . X = 0, rN . X = 0, solve for X_0, X_N in terms of the interior values
    r0 = c0.astype(complex).copy(); r0[0] += d0
    rN = cN.astype(complex).copy(); rN[-1] += dN
    Bb = np.array([[r0[0], r0[-1]], [rN[0], rN[-1]]])
    Bi = -np.array([r0[1:-1], rN[1:-1]])
    E = np.linalg.solve(Bb, Bi)                     # [X_0; X_N] = E X_int
    A = -(D2[1:-1, 1:-1] + np.outer(D2[1:-1, 0], E[0]) + np.outer(D2[1:-1, -1], E[1]))
    lam = np.linalg.eigvals(A)
    ku = np.sqrt(lam.astype(complex))
    ku = np.where(ku.real < 0, -ku, ku)
    ku = ku[np.argsort(ku.real)]
    roots = []
    for x0 in ku:
        if len(roots) == M:
            break
        try:
            r = _newton(lambda x: lsu_dispersion_scaled(x, k, a, Zp, Zm, kind) / x, x0)
        except RuntimeError:
            continue
        if r.real < 0:
            r = -r
        if abs(r) < 1e-9 / a or any(abs(r - q) < 1e-8 * max(1.0, abs(q)) for q in roots):
            continue
        roots.append(r)
    roots = np.array(sorted(roots, key=lambda r: r.real))
    if len(roots) < M:
        raise RuntimeError("not enough roots; increase N")
    return roots


def gamma_from_ku(ku, k, kn):
    """Propagation constant with Im <= 0 (CONVENTIONS.md branch): gamma = sqrt(k^2 - k_n^2 - k_u^2)."""
    g = np.sqrt(k * k - kn * kn - ku * ku + 0j)
    if abs(g.imag) <= 1e-12 * abs(g):          # (numerically) real: forward = positive real part
        return complex(abs(g.real), g.imag) if g.real < 0 else g
    return g if g.imag < 0 else -g


# ------------------------------------------------------------------------------------------ resistive sheets
def sheet_dispersion(kv, k, b, xis, v0s, kind):
    """Dispersion function of the LS^v modes of a guide with resistive sheets at v' = v0s (boxed result B8).

    xis = R_sheet / eta0 for each sheet, v0s = sheet positions v' in (0, b) (v' = v + b/2). kind 'LSM'
    (H_v = 0, Butterweck's EH^(2)) or 'LSE' (E_v = 0, EH^(1)). Transfer of (Y, Y') across [0, b]:
      LSM: Y' continuous, Y(v0+) = Y(v0-) - i Y'(v0) / (k xi);   Y'(0) = Y'(b) = 0, Y = cos(k_v v') at v' = 0
      LSE: Y continuous, Y'(v0+) = Y'(v0-) + i k Y(v0) / xi;     Y(0) = Y(b) = 0,  Y = sin(k_v v')/k_v at v' = 0
    Returns the boundary value at v' = b that must vanish (Y'(b) for LSM, Y(b) for LSE).
    """
    order = np.argsort(v0s)
    xis = np.asarray(xis, dtype=complex)[order]
    v0s = np.asarray(v0s, dtype=float)[order]
    if kind == "LSM":
        Y, dY = 1.0 + 0j, 0.0 + 0j
    else:
        Y, dY = 0.0 + 0j, 1.0 + 0j
    pos = 0.0
    for xi, v0 in zip(xis, v0s):
        Y, dY = _advance(Y, dY, kv, v0 - pos)
        pos = v0
        if kind == "LSM":
            Y = Y - 1j * dY / (k * xi)
        else:
            dY = dY + 1j * k * Y / xi
    Y, dY = _advance(Y, dY, kv, b - pos)
    return dY if kind == "LSM" else Y


def _advance(Y, dY, kv, d):
    c, s = np.cos(kv * d), np.sin(kv * d)
    sinc = d * np.sinc(kv * d / np.pi)          # sin(kv d) / kv, also for kv = 0
    return c * Y + sinc * dY, -kv * s * Y + c * dY


def sheet_root(kv0, k, b, xis, v0s, kind):
    return _newton(lambda x: sheet_dispersion(x, k, b, xis, v0s, kind), kv0)


def butterweck_sheet_eq(kv, k, b, xi, kind):
    """Butterweck (1968) eqs. (8) (LSE, EH^(1)) and (9) (LSM, EH^(2)) for one centred sheet, e^{j omega t}."""
    t = np.tan(kv * b / 2)
    if kind == "LSE":
        return 2 * xi + 1j * (k / kv) * t
    return 2 * xi + 1j * (kv / k) * t


def sheet_dispersion_closed(kv, k, b, xi, v0, kind):
    """One sheet at v' = v0 (d1 = v0, d2 = b - v0), closed form (boxed result B7):
       LSM (H_v = 0): xi k sin(k_v b) + i k_v sin(k_v d1) sin(k_v d2) = 0
       LSE (E_v = 0): xi k_v sin(k_v b) + i k sin(k_v d1) sin(k_v d2) = 0
    """
    d1, d2 = v0, b - v0
    if kind == "LSM":
        return xi * k * np.sin(kv * b) + 1j * kv * np.sin(kv * d1) * np.sin(kv * d2)
    return xi * kv * np.sin(kv * b) + 1j * k * np.sin(kv * d1) * np.sin(kv * d2)
