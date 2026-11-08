"""Ohmic wall loss in the coupled-mode equations (first order in the wall impedance).

Implements the boxed results B1-B4b of docs/theory/T1-wall-impedance.md (theory card T1, checked independently by
T1b). Conventions: CONVENTIONS.md (e^{+i omega t}, forward modes e^{-i beta z}, fields of `wrmodes.modes`,
amplitudes normalized with the unconjugated reaction, principal square roots of the wave admittances).

Walls are labelled by their outward normal (from the guide into the wall), in the local frame (u, v, z):

    'u+' : u = +a/2 (outer wall of a bend with kappa > 0),   'u-' : u = -a/2,
    'v+' : v = +b/2,                                          'v-' : v = -b/2.

A wall impedance argument ``Z`` is a scalar (the same on all four walls) or a dict {wall: Z_w} (ohm); walls that are
not listed are perfect conductors. Metal walls of conductivity sigma have Z_w = Z_s = (1 + i) R_s (B1,
`surface_impedance`).

Main result (B2-B4b). With h = 1 + kappa u, the wall integrals

    N_pq = sum_w Z_w oint_w h (e_p . n)(e_q . n) dl,      G_pq = (k eta0)^-2 sum_w Z_w oint_w h g_p g_q dl,

g_p = div(z x e_p) = -chi_p^2 Gamma_p (TE; 0 for TM), give the symmetric matrices

    Lambda^{+-}_pq = 1/2 [ +- sqrt(Y_p) sqrt(Y_q) N_pq + G_pq / (sqrt(Y_p) sqrt(Y_q)) ]      (1/m)

and the coupled-mode system for y = [a+; a-]

    dy/dz = [(D - Lambda0_full) + kappa(z) (K - Lambda1_full)] y,
    Lambda_full = [[Lambda^+, Lambda^-], [-Lambda^-, -Lambda^+]],   D = diag(-i beta, +i beta),

with Lambda = Lambda0 + kappa Lambda1 (straight part and coefficient of kappa). For a propagating mode with metal
walls, Re Lambda^+_pp is the textbook attenuation constant (Marcuvitz 1951, sec. 2.2, eqs. 4 and 10) and the phase
constant grows by the same amount.

Validity (T1 section 5): the first-order system has spurious O(Z^2) terms through the far off-diagonal entries,
which grow with the number of modes. The components therefore **couple only the propagating modes through Lambda**
and give each evanescent mode only its diagonal Lambda^+_pp (`coupling_mask`, the rule of T1 section 5). For
aluminium WR-90 the propagating block is then within 3.5e-4 of the exact Leontovich solution. For large wall
impedances (|Z| ~ eta0, dielectric absorbers) first order fails for modes with n >= 1; use the exact
`wrmodes.filters.LossyWallFilter` instead.
"""
import numpy as np
import scipy.constants as cst
from scipy.linalg import expm

from .modes import ETA0, Mode, ModeBasis, cutoff_wavenumber
from .gsm import GSM, from_transfer_matrix

__all__ = ["WALLS", "surface_resistance", "surface_impedance", "wall_impedances", "wall_integrals",
           "loss_matrices", "loss_matrix_full", "bend_loss_matrices", "coupling_mask", "attenuation_constant",
           "attenuation_constants", "lossy_straight_gsm", "straight_gsm_from_loss_matrix"]

WALLS = ("u+", "u-", "v+", "v-")


# ------------------------------------------------------------------------------------------ item 1
def surface_resistance(f, sigma):
    """Surface resistance R_s = sqrt(omega mu0 / (2 sigma)) (ohm) of a wall of conductivity sigma (S/m) at f (Hz).

    CONVENTIONS.md section 1. There is no default material: aluminium (SPARC baseline) is sigma = 3.5e7 S/m,
    copper 5.8e7 S/m. Broadcasts over f and sigma.
    """
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive (S/m)")
    return np.sqrt(np.pi * np.asarray(f, dtype=float) * cst.mu_0 / sigma)


def surface_impedance(f, sigma):
    """Leontovich surface impedance Z_s = (1 + i) R_s (ohm) for e^{+i omega t} (T1 box B1, CONVENTIONS.md 1)."""
    return (1 + 1j) * surface_resistance(f, sigma)


def wall_impedances(Z):
    """Dict {wall: Z_w} (complex, ohm) of the four walls from a scalar (all walls) or a partial dict (others 0)."""
    if isinstance(Z, dict):
        bad = set(Z) - set(WALLS)
        if bad:
            raise ValueError(f"unknown walls {sorted(bad)}; walls are {WALLS}")
        out = {w: complex(Z.get(w, 0.0)) for w in WALLS}
    else:
        out = {w: complex(Z) for w in WALLS}
    for w, z in out.items():
        if not np.isfinite(z):
            raise ValueError(f"wall impedance of {w} is not finite")
    return out


# ------------------------------------------------------------------------------------------ wall integrals
def _mode_arrays(basis):
    """Integer and geometric arrays of a basis: kind (True for TE), m, n, c, k_m, k_n, chi^2."""
    a, b = basis.a, basis.b
    te = np.array([md.kind == "TE" for md in basis.modes], dtype=bool)
    m = np.array([md.m for md in basis.modes], dtype=int)
    n = np.array([md.n for md in basis.modes], dtype=int)
    km, kn = m * np.pi / a, n * np.pi / b
    chi2 = km * km + kn * kn
    eps_m, eps_n = np.where(m == 0, 1.0, 2.0), np.where(n == 0, 1.0, 2.0)
    c = np.where(te, np.sqrt(eps_m * eps_n), 2.0) / (np.sqrt(a * b) * np.sqrt(chi2))
    return te, m, n, c, km, kn, chi2


def wall_integrals(basis, order=0):
    """Closed forms of the wall integrals of every pair of modes of a basis (T1 box B2a), one entry per wall.

    Returns a dict {wall: (Nw, Gw)} of real symmetric N x N matrices,

        Nw_pq = oint_wall u^order (e_p . n)(e_q . n) dl      (1/m for order 0, dimensionless for order 1),
        Gw_pq = oint_wall u^order g_p g_q dl                  (1/m^3 for order 0; TE-TE only),

    with u the centred coordinate: order 0 is the straight guide, order 1 the coefficient of kappa in a bend
    (h = 1 + kappa u). Side walls (u+-) couple equal n, broad walls (v+-) equal m (order 0) or m + m' odd
    (order 1).
    """
    if order not in (0, 1):
        raise ValueError("order must be 0 or 1")
    a, b = basis.a, basis.b
    te, m, n, c, km, kn, chi2 = _mode_arrays(basis)
    A = np.where(te, -c * kn, c * km)            # e_u = A cos(k_m u') sin(k_n v'): e.n on u-walls
    B = np.where(te, c * km, c * kn)             # e_v = B sin(k_m u') cos(k_n v'): e.n on v-walls
    tete = np.outer(te, te)
    gg = np.outer(chi2 * c, chi2 * c) * tete     # g_p g_q prefactors
    same_n = np.equal.outer(n, n)
    same_m = np.equal.outer(m, m)
    par_m = (-1.0) ** np.add.outer(m, m)
    par_n = (-1.0) ** np.add.outer(n, n)
    eps_n, eps_m = np.where(n == 0, 1.0, 2.0), np.where(m == 0, 1.0, 2.0)
    out = {}
    # side walls u = +-a/2: v-integrals of sin(k_n v')^2 = b/2 (n >= 1) and cos^2 = b/eps_n
    Nu = np.outer(A, A) * same_n * (b / 2)
    Gu = gg * same_n * (b / eps_n)[None, :]
    for wall, sgn, pos in (("u+", par_m, a / 2), ("u-", 1.0, -a / 2)):
        w = 1.0 if order == 0 else pos
        out[wall] = (Nu * sgn * w, Gu * sgn * w)
    # broad walls v = +-b/2: u-integrals with weight (u' - a/2)^order
    if order == 0:
        S = same_m * (a / 2) * (m[None, :] >= 1)
        C = same_m * (a / eps_m)[None, :]
    else:
        mp, mq = np.meshgrid(m, m, indexing="ij")
        odd = (mp + mq) % 2 == 1
        den = np.where(odd, (mp * mp - mq * mq).astype(float) ** 2, 1.0)
        S = np.where(odd, -(a / np.pi) ** 2 * 4.0 * mp * mq / den, 0.0)
        C = np.where(odd, -(a / np.pi) ** 2 * 2.0 * (mp * mp + mq * mq) / den, 0.0)
    Nv = np.outer(B, B) * S
    Gv = gg * C
    for wall, sgn in (("v+", par_n), ("v-", 1.0)):
        out[wall] = (Nv * sgn, Gv * sgn)
    return out


# ------------------------------------------------------------------------------------------ Lambda
def loss_matrices(basis, f, Z, order=0):
    """(Lambda^+, Lambda^-), N x N complex symmetric (1/m for order 0), T1 boxes B2 and B3.

    Parameters
    ----------
    basis : ModeBasis
        Modes of the local frame (a along u). No mode may be exactly at cutoff.
    f : float
        Frequency (Hz).
    Z : complex or dict
        Wall impedances (ohm), see `wall_impedances`.
    order : 0 or 1
        0: straight guide (Lambda0); 1: coefficient of kappa in a bend (Lambda1, dimensionless).

    All entries are returned, for propagating and evanescent modes alike; components apply `coupling_mask`.
    """
    Zw = wall_impedances(Z)
    k = 2 * np.pi * f / cst.c
    N = len(basis)
    Nm = np.zeros((N, N), dtype=complex)
    Gm = np.zeros((N, N), dtype=complex)
    if N == 0:
        return Nm, Gm
    for wall, (Nw, Gw) in wall_integrals(basis, order).items():
        if Zw[wall] != 0:
            Nm += Zw[wall] * Nw
            Gm += Zw[wall] * Gw
    Gm /= (k * ETA0) ** 2
    sY = np.sqrt(basis.admittance(f))            # principal square roots (CONVENTIONS.md section 5)
    YY = np.outer(sY, sY)
    return 0.5 * (YY * Nm + Gm / YY), 0.5 * (-YY * Nm + Gm / YY)


def coupling_mask(basis, f):
    """Boolean N x N mask of the Lambda entries the components keep (T1 section 5).

    True between two propagating modes and on the diagonal: the propagating modes are coupled through Lambda, and
    each evanescent mode only gets its own Lambda_pp (the far off-diagonal entries of the first-order system carry
    spurious O(Z^2) terms).
    """
    p = basis.propagating_mask(f) if len(basis) else np.zeros(0, bool)
    return np.outer(p, p) | np.eye(len(basis), dtype=bool)


def _full(Lp, Lm):
    return np.block([[Lp, Lm], [-Lm, -Lp]])


def loss_matrix_full(basis, f, Z, kappa=0.0, restrict=True):
    """Lambda_full (2N x 2N) of a constant curvature kappa (1/m), for dy/dz = (D + kappa K - Lambda_full) y.

    T1 boxes B4 and B4b: Lambda = Lambda0 + kappa Lambda1. With restrict=True (default) the rule of
    `coupling_mask` is applied: evanescent modes keep only their diagonal Lambda^+_pp (no Lambda^- either).
    """
    Lp, Lm = loss_matrices(basis, f, Z, 0)
    if kappa != 0.0:
        Lp1, Lm1 = loss_matrices(basis, f, Z, 1)
        Lp, Lm = Lp + kappa * Lp1, Lm + kappa * Lm1
    if restrict:
        Lp, Lm = _restrict(basis, f, Lp, Lm)
    return _full(Lp, Lm)


def _restrict(basis, f, Lp, Lm):
    keep = coupling_mask(basis, f)
    prop = basis.propagating_mask(f) if len(basis) else np.zeros(0, bool)
    return np.where(keep, Lp, 0.0), np.where(np.outer(prop, prop), Lm, 0.0)


def bend_loss_matrices(basis, f, Z, restrict=True):
    """(Lambda0_full, Lambda1_full), each 2N x 2N, so that dy/dz = [(D - L0) + kappa(z)(K - L1)] y (T1 box B4b).

    With restrict=True the rule of `coupling_mask` applies to both (evanescent modes: diagonal Lambda^+ only).
    """
    L0p, L0m = loss_matrices(basis, f, Z, 0)
    L1p, L1m = loss_matrices(basis, f, Z, 1)
    if restrict:
        L0p, L0m = _restrict(basis, f, L0p, L0m)
        L1p, L1m = _restrict(basis, f, L1p, L1m)
    return _full(L0p, L0m), _full(L1p, L1m)


# ------------------------------------------------------------------------------------------ attenuation
def attenuation_constant(mode, a, b, f, sigma):
    """Attenuation constant alpha (Np/m) of one mode of an a x b guide whose four walls have conductivity sigma.

    alpha = Re Lambda^+_pp with Z_w = (1 + i) R_s on all walls (T1 B3), which equals Marcuvitz (1951), sec. 2.2,
    eq. (4) (E_mn = TM) and eq. (10) (H_mn = TE); the phase constant grows by the same amount. Below cutoff it is
    the extra decay rate Re Lambda^+_pp of the evanescent amplitude. Broadcasts over f (Hz) and sigma (S/m).
    Multiply by 20 / ln(10) for dB/m. No default material (aluminium 3.5e7 S/m, copper 5.8e7 S/m).
    """
    mode = Mode.parse(mode)
    f = np.asarray(f, dtype=float)
    k = 2 * np.pi * f / cst.c
    m, n = mode.m, mode.n
    km, kn = m * np.pi / a, n * np.pi / b
    chi2 = km * km + kn * kn
    eps_m, eps_n = (1.0 if m == 0 else 2.0), (1.0 if n == 0 else 2.0)
    te = mode.kind == "TE"
    c = (np.sqrt(eps_m * eps_n) if te else 2.0) / (np.sqrt(a * b * chi2))
    A, B = (-c * kn, c * km) if te else (c * km, c * kn)
    Zs = surface_impedance(f, sigma)
    d = k * k - chi2
    if np.any(d == 0):
        raise ValueError(f"{mode.name} is exactly at cutoff: its attenuation is undefined")
    beta = np.where(d > 0, np.sqrt(np.abs(d)) + 0j, -1j * np.sqrt(np.abs(d)))
    Y = beta / (k * ETA0) if te else k / (ETA0 * beta)
    Nd = Zs * (A * A * b + B * B * a)                     # sum over the four walls, p = q
    Gd = Zs * (2 * chi2 * chi2 * c * c * (b / eps_n + a / eps_m) if te else 0.0) / (k * ETA0) ** 2
    lam = 0.5 * (Y * Nd + Gd / Y)
    out = np.real(lam)
    return float(out) if out.ndim == 0 else out


def attenuation_constants(basis, f, sigma):
    """Attenuation constants (Np/m) of all modes of a basis at f (Hz) for walls of conductivity sigma (S/m)."""
    return np.array([attenuation_constant(md, basis.a, basis.b, f, sigma) for md in basis.modes])


# ------------------------------------------------------------------------------------------ straight sections
def straight_gsm_from_loss_matrix(basis, f, length, lam_full, name=""):
    """GSM of a uniform section of length L (m) with dy/dz = (D - Lambda_full) y, y = [A+; A-].

    Propagating modes: the 2Np x 2Np block of D - Lambda_full is integrated exactly, as the GSM of a short slice
    (matrix exponential of a slice converted with `from_transfer_matrix`) doubled by Redheffer star products, so
    that no growing exponential exp(+alpha L) enters (well conditioned for long lossy lines). Evanescent modes must
    be uncoupled (the rule of `coupling_mask`): each gets exp(-(i beta_p + Lambda^+_pp) L), no reflection.
    The section is reciprocal (S = S^T) when Lambda^+ and Lambda^- are symmetric.
    """
    N = len(basis)
    lam_full = np.asarray(lam_full, dtype=complex)
    if lam_full.shape != (2 * N, 2 * N):
        raise ValueError(f"Lambda_full has shape {lam_full.shape}, expected {(2 * N, 2 * N)}")
    length = float(length)
    beta = basis.beta(f)
    prop = basis.propagating_mask(f)
    ip, ie = np.flatnonzero(prop), np.flatnonzero(~prop)
    if len(ie):
        ev = np.zeros(2 * N, dtype=bool)
        ev[ie], ev[ie + N] = True, True
        off = (ev[:, None] | ev[None, :]) & ~np.eye(2 * N, dtype=bool)
        if np.any(lam_full[off]):
            raise ValueError("evanescent modes must be uncoupled in Lambda_full (wrmodes.loss.coupling_mask)")
        if not np.allclose(lam_full[ie, ie], -lam_full[ie + N, ie + N], rtol=1e-14, atol=0):
            raise ValueError("Lambda_full: the A- block of an evanescent mode must be -Lambda^+_pp")
    S11 = np.zeros((N, N), complex)
    S22 = np.zeros((N, N), complex)
    S21 = np.zeros((N, N), complex)
    S12 = np.zeros((N, N), complex)
    if len(ie):
        t = np.exp(-(1j * beta[ie] + lam_full[ie, ie]) * length)
        S21[ie, ie] = t
        S12[ie, ie] = t
    if len(ip):
        sel = np.concatenate([ip, ip + N])
        D = np.concatenate([-1j * beta[ip], 1j * beta[ip]])
        M = np.diag(D) - lam_full[np.ix_(sel, sel)]
        sub = basis.subset(ip)
        # slice length: |Re eig(M)| <= ||Lambda||, keep exp(||Lambda|| h) <= e^0.5 within a slice
        nrm = np.linalg.norm(lam_full[np.ix_(sel, sel)], 1)
        doublings = 0
        if length > 0 and nrm * length > 0.5:
            doublings = int(np.ceil(np.log2(nrm * length / 0.5)))
        h = length / 2 ** doublings
        g = from_transfer_matrix(expm(M * h), sub, f)
        for _ in range(doublings):
            g = g @ g
        S11[np.ix_(ip, ip)], S12[np.ix_(ip, ip)] = g.S11, g.S12
        S21[np.ix_(ip, ip)], S22[np.ix_(ip, ip)] = g.S21, g.S22
    return GSM(S11, S12, S21, S22, basis, basis, f, name or f"lossy straight {length:g} m")


def lossy_straight_gsm(basis, f, length, Z, name=""):
    """GSM of a straight section of length L (m) with wall impedances Z (T1 B4 with kappa = 0).

    The propagating modes are coupled through the full Lambda_full (including the degenerate TE_mn/TM_mn coupling
    and the backward coupling Lambda^-); evanescent modes get only their own Lambda^+_pp (`coupling_mask`).
    For Z = 0 the result is the lossless straight up to round-off; components call `wrmodes.gsm.straight` directly
    for perfect walls.
    """
    return straight_gsm_from_loss_matrix(basis, f, length, loss_matrix_full(basis, f, Z), name)
