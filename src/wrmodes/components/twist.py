"""Twisted rectangular guides (WP6): a straight guide whose cross-section rotates about its axis.

Theory: docs/theory/T3b-twist-coupling.md (boxes 2b and 4; the closed forms used here), derived independently in
docs/theory/T3-twist-coupling.md (box T3-1); the two agree to 8e-18. Conventions: CONVENTIONS.md (fields of
`wrmodes.modes`, amplitudes of section 5, GSM rules of section 6, frames of section 2).

In the co-rotating frame (u, v) of the cross-section, the coupled-mode equations of a twist are exact and linear
in the twist rate tau = theta'(z) (no tau^2 terms, no theta''):

    dV/dz = -i beta I / Y + tau K V,        dI/dz = -i beta Y V - tau K^T I                     (T3b box 2)

with the real, frequency-independent matrix K_pq = iint e_q . d_theta e_p ds (T3b box 2b, closed forms). K is
not antisymmetric (its symmetric part is a wall integral, T3b box 3) and must not be antisymmetrized. In the
power-wave amplitudes y = [a+; a-] of CONVENTIONS.md section 5:

    dy/dz = (D + tau(z) X) y,   D = diag(-i beta, +i beta),   X = [[A, B], [B, A]],
    A = (K rho - K^T / rho) / 2,  B = (K rho + K^T / rho) / 2,  rho_pq = sqrt(Y_p) / sqrt(Y_q)   (T3b box 4)

(principal roots, so that the same formulas hold for evanescent modes).
"""
import numpy as np
import scipy.constants as cst
from scipy.integrate import quad
from scipy.linalg import expm

from ..gsm import GSM
from ..modes import ModeBasis, Mode
from ..loss import loss_matrix_full
from ..transfer_matrix import _slice_exponentials, _slice_edges, _blas_limit
from .base import Component
from .guide import Guide

__all__ = ["Twist", "TwistProfile", "twist_coupling_matrix", "twist_system_matrices",
           "helical_propagation_constants", "zero_length_twist_transfer_matrix", "twist_gsm_blocks"]

PROFILES = ("constant", "cosine", "triangular")


# ============================================================================================ coupling matrix
def _norm(te, m, n, a, b):
    s = np.pi * np.sqrt(m * m * b / a + n * n * a / b)
    eps_m, eps_n = np.where(m == 0, 1.0, 2.0), np.where(n == 0, 1.0, 2.0)
    return np.where(te, np.sqrt(eps_m * eps_n), 2.0) / s


def twist_coupling_matrix(basis):
    """Twist coupling matrix K (N x N, real, dimensionless) of a ModeBasis, rows and columns in basis order.

    K_pq = iint [ div(e_p) (w . e_q) + (r . e_p) div(z x e_q) ] ds = iint e_q . d_theta e_p ds, r = (u, v),
    w = (v, -u) (T3b box 2, T3b box 3), written in the separated 1D integrals of T3b box 2a:

        TE p, TE q: K = -chi_q^2 iint Gamma_q L Gamma_p          TM p, TM q: K = chi_p^2 iint Pi_p L Pi_q
        TM p, TE q: K = (chi_q^2 - chi_p^2) iint Pi_p D Gamma_q + 2 chi_q^2 iint Pi_p Gamma_q
        TE p, TM q: K = 0

    (L = u d_v - v d_u, D = u d_u + v d_v; Gamma, Pi the normalized potentials of CONVENTIONS.md section 4). Zero
    unless m + m' and n + n' are both odd (selection rule: a twist always changes the family n). The cross-section
    rotates about its centre; K is independent of frequency. K_{TE1,0; TE0,1} = 8/pi^2 for every aspect ratio.
    Equal to T3's C^T (T3 box T3-2) to 8e-18.
    """
    a, b = basis.a, basis.b
    te = np.array([md.kind == "TE" for md in basis.modes], dtype=bool)
    m = np.array([md.m for md in basis.modes], dtype=float)
    n = np.array([md.n for md in basis.modes], dtype=float)
    if len(m) == 0:
        return np.zeros((0, 0))
    Nn = _norm(te, m, n, a, b)
    chi2 = np.pi ** 2 * ((m / a) ** 2 + (n / b) ** 2)
    M1, M2, N1, N2 = m[:, None], m[None, :], n[:, None], n[None, :]
    oddm = (M1 + M2) % 2 == 1
    oddn = (N1 + N2) % 2 == 1
    sel = oddm & oddn
    dm = np.where(oddm, M1 * M1 - M2 * M2, 1.0)
    dn = np.where(oddn, N1 * N1 - N2 * N2, 1.0)
    # 1D integrals over [-h/2, h/2] with x' = x + h/2 (T3b section 3a), valid for m + m' odd
    Zm, Zn = a / np.pi * 2 * M1 / dm, b / np.pi * 2 * N1 / dn                      # int s_m c_m'
    Xcm = -2 * a * a / np.pi ** 2 * (M1 * M1 + M2 * M2) / dm ** 2                    # int x c_m c_m'
    Xcn = -2 * b * b / np.pi ** 2 * (N1 * N1 + N2 * N2) / dn ** 2
    Xsm = -4 * a * a / np.pi ** 2 * M1 * M2 / dm ** 2                                # int x s_m s_m'
    Xsn = -4 * b * b / np.pi ** 2 * N1 * N2 / dn ** 2
    NN = Nn[:, None] * Nn[None, :]
    al1, ga1 = M1 * np.pi / a, N1 * np.pi / b
    al2, ga2 = M2 * np.pi / a, N2 * np.pi / b
    c1, c2 = chi2[:, None], chi2[None, :]
    EE = -c2 * NN * (-ga1 * Xcm * Zn + al1 * Zm * Xcn)
    MM = c1 * NN * (ga2 * Xsm * Zn - al2 * Zm * Xsn)
    ME = (c2 - c1) * NN * (-al2 * Xsm * Zn - ga2 * Zm * Xsn) + 2 * c2 * NN * Zm * Zn
    t1, t2 = te[:, None], te[None, :]
    K = np.where(t1 & t2, EE, np.where(~t1 & ~t2, MM, np.where(~t1 & t2, ME, 0.0)))
    return np.where(sel, K, 0.0)


def twist_system_matrices(basis, f, K=None):
    """(d, X): the system dy/dz = (diag(d) + tau(z) X) y of a twist, y = [a+; a-] (T3b box 4).

    d = [-i beta, +i beta] (2N,) and X = [[A, B], [B, A]] (2N x 2N) with A = (K rho - K^T / rho)/2 and
    B = (K rho + K^T / rho)/2, rho_pq = sqrt(Y_p)/sqrt(Y_q) (principal roots, CONVENTIONS.md section 5). For
    propagating modes A is real antisymmetric and B real symmetric (lossless, reciprocal). tau is in rad/m.
    No mode of the basis may be exactly at cutoff (its admittance is undefined, CONVENTIONS.md section 3).
    """
    K = twist_coupling_matrix(basis) if K is None else K
    beta = basis.beta(f)
    sY = np.sqrt(basis.admittance(f).astype(complex))
    rho = sY[:, None] / sY[None, :]
    A = 0.5 * (K * rho - K.T / rho)
    B = 0.5 * (K * rho + K.T / rho)
    X = np.block([[A, B], [B, A]])
    return np.concatenate([-1j * beta, 1j * beta]), X


def helical_propagation_constants(basis, f, rate):
    """Propagation constants gamma (1/m) of the 2N helical modes of a uniformly twisted guide (rate in rad/m).

    The eigenvalues lambda of diag(d) + rate X (`twist_system_matrices`) give fields exp(lambda z) in the
    co-rotating frame; gamma = i lambda (forward propagating modes: Re gamma > 0). Sorted by real part, descending.
    Converges to the exact twisted-guide modes like 1/chi_max of the basis (T3 section 7, T3b section 5).
    """
    d, X = twist_system_matrices(basis, f)
    lam = np.linalg.eigvals(np.diag(d) + rate * X)
    g = 1j * lam
    return g[np.argsort(-g.real)]


def zero_length_twist_transfer_matrix(basis, f, angle_deg):
    """Transfer matrix exp(theta X) of a twist by angle_deg in the limit of zero length (algebraic check only).

    Propagation is negligible in that limit, so only theta enters. Z(-theta) Z(theta) = I exactly (T3 box T3-5,
    T3b box 6). This limit is **not** the physical 90 degree frame relabeling in a truncated basis (T3 section 5,
    T3b section 7): the limits L -> 0 and N -> infinity do not commute. Do not use it as a component.
    """
    _, X = twist_system_matrices(basis, f)
    return expm(np.radians(angle_deg) * X)


# ============================================================================================ rate profiles
class TwistProfile:
    """Twist rate theta'(z) (rad/m) of a twist of total angle `angle` (rad) over `length` (m).

    Named profiles (Doane & Anderson 1989, Fig. 9 and eq. 17; T3 box T3-6, T3b section 8):

    * 'constant':   theta'(z) = theta / L.
    * 'cosine':     theta'(z) = (pi theta / 2L) sin(pi z / L), i.e. theta(z) = theta (1 - cos(pi z / L)) / 2: the
      rate is a half-cosine bump, zero at both ends. Doane & Anderson only name it ("cosine" in Fig. 9); this is
      T3's and T3b's reading, the one that reproduces their curve (the raised cosine does not).
    * 'triangular': Doane & Anderson eq. 17, theta'(z) = (2 theta/L)(2z/L) for 0 < z < L/2 and
      (2 theta/L)(2(L - z)/L) for L/2 < z < L (rises linearly to twice the mean rate at L/2 and falls back).

    A callable r(t), t = z/L in [0, 1] (accepting NumPy arrays) is a rate shape, normalized here so that the
    integral of theta' over [0, L] is theta: theta'(z) = theta r(z/L) / (L int_0^1 r dt). Rate discontinuities
    or kinks of a callable shape should be given as `breakpoints` (t values in (0, 1)) for 4th-order convergence.
    Picklable when the callable is (for `Line.sweep`).
    """

    def __init__(self, profile, angle, length, breakpoints=None):
        self.angle, self.length = float(angle), float(length)
        if isinstance(profile, str):
            if profile not in PROFILES:
                raise ValueError(f"profile must be one of {PROFILES} or a callable rate shape r(t), got {profile!r}")
            self.kind, self.shape, self._scale = profile, None, None
            bps = (0.5,) if profile == "triangular" else ()
        elif callable(profile):
            self.kind, self.shape = "callable", profile
            bps = tuple(sorted(float(t) for t in np.atleast_1d(breakpoints))) if breakpoints is not None else ()
            if any(not 0 < t < 1 for t in bps):
                raise ValueError("breakpoints of a callable rate shape are t = z/L values in (0, 1)")
            integral = quad(lambda t: float(np.asarray(profile(np.array([t])), dtype=float).ravel()[0]), 0.0, 1.0,
                            points=bps or None, limit=500, epsabs=0.0, epsrel=1e-13)[0]
            if not np.isfinite(integral) or abs(integral) < 1e-300:
                raise ValueError("the rate shape r(t) must have a finite, non-zero integral over [0, 1]")
            self._scale = 1.0 / integral
        else:
            raise TypeError("profile must be 'constant', 'cosine', 'triangular' or a callable r(t)")
        self.breakpoints_t = bps

    @property
    def breakpoints(self):
        """Positions (m) of kinks or jumps of theta' inside the twist."""
        return tuple(t * self.length for t in self.breakpoints_t)

    @property
    def is_constant(self):
        return self.kind == "constant"

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        th, L = self.angle, self.length
        if self.kind == "constant":
            return np.full(z.shape, th / L)
        if self.kind == "cosine":
            return (np.pi * th / (2 * L)) * np.sin(np.pi * z / L)
        if self.kind == "triangular":
            return (4 * th / L ** 2) * np.where(z < L / 2, z, L - z)
        return th * self._scale / L * np.broadcast_to(np.asarray(self.shape(z / L), dtype=float), z.shape)

    def __repr__(self):
        return f"TwistProfile({self.kind}, {np.degrees(self.angle):g} deg, {self.length:g} m)"


# ============================================================================================ solver
def _gsm_from_transfer_stack(P, N):
    """GSM blocks (S11, S12, S21, S22) of a stack of transfer matrices (..., 2N, 2N), as `gsm.from_transfer_matrix`."""
    P11, P12, P21, P22 = P[..., :N, :N], P[..., :N, N:], P[..., N:, :N], P[..., N:, N:]
    S12 = np.linalg.inv(P22)
    S11 = -S12 @ P21
    S21 = P11 + P12 @ S11
    S22 = P12 @ S12
    return S11, S12, S21, S22


def _star(A, B):
    """Redheffer star product of GSM blocks, A then B (stacks allowed; no singular case arises in a slice cascade)."""
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = B
    I = np.eye(A22.shape[-1])
    M1 = I - B11 @ A22
    M2 = I - A22 @ B11
    S11 = A11 + A12 @ np.linalg.solve(M1, B11 @ A21)
    S12 = A12 @ np.linalg.solve(M1, B12)
    S21 = B21 @ np.linalg.solve(M2, A21)
    S22 = B22 + B21 @ np.linalg.solve(M2, A22 @ B12)
    return S11, S12, S21, S22


def _star_reduce(S):
    """Star product of a stack of slice GSMs in order (slice 0 first), by pairwise (tree) reduction."""
    while S[0].shape[0] > 1:
        n = S[0].shape[0]
        last = tuple(x[-1:] for x in S) if n % 2 else None
        ev = tuple(x[0:n - n % 2:2] for x in S)
        od = tuple(x[1:n - n % 2:2] for x in S)
        S = _star(ev, od)
        if last is not None:
            S = tuple(np.concatenate([x, y]) for x, y in zip(S, last))
    return tuple(x[0] for x in S)


def _tree_product(P):
    """P[n-1] @ ... @ P[0] by pairwise multiplication."""
    while P.shape[0] > 1:
        n = P.shape[0]
        last = P[-1:] if n % 2 else None
        P = P[1:n - n % 2:2] @ P[0:n - n % 2:2]
        if last is not None:
            P = np.concatenate([P, last])
    return P[0]


def _chunk_size(N):
    """Slices per batched chunk: about 64 MB of complex (2N x 2N) matrices."""
    return int(max(1, min(512, 64e6 / (16 * (2 * N) ** 2 * 4))))


def _sliced_solution(D, X, rate, edges, use_gsm, N):
    """GSM blocks of the twist sliced at `edges` (4th-order commutator-free Magnus with two Gauss points per slice,
    `transfer_matrix._slice_exponentials`). use_gsm: convert every slice to a GSM and star them (evanescent modes or
    loss); otherwise multiply the slice transfer matrices and convert once."""
    acc = None
    c = _chunk_size(N)
    for i0 in range(0, len(edges) - 1, c):
        e = edges[i0:i0 + c + 1]
        P = _slice_exponentials(D, X, rate, e)
        if use_gsm:
            part = _star_reduce(_gsm_from_transfer_stack(P, N))
            acc = part if acc is None else _star(acc, part)
        else:
            part = _tree_product(P)
            acc = part if acc is None else part @ acc
    if use_gsm:
        return acc
    return tuple(x[0] for x in _gsm_from_transfer_stack(acc[None], N))


def twist_gsm_blocks(basis, f, rate, length, loss=None, atol=1e-6, max_slices=16384, slices=None, K=None):
    """GSM blocks (S11, S12, S21, S22) and solver info of a twist in the basis `basis` (all modes coupled).

    Parameters
    ----------
    basis : ModeBasis
        Modes of the (constant) cross-section, co-rotating frame. Evanescent modes allowed (no mode exactly at cutoff).
    f : float
        Frequency (Hz).
    rate : TwistProfile
        theta'(z) (rad/m) on [0, length], with its `breakpoints` and `is_constant`.
    length : float
        Length (m), > 0.
    loss : (2N, 2N) array or None
        Lambda_full of the wall loss (T1 box B4, `wrmodes.loss.loss_matrix_full`); the system becomes
        dy/dz = (D - Lambda_full + tau(z) X) y.
    atol : float
        Target absolute accuracy of every complex entry of S (Richardson estimate of the 4th-order Magnus method,
        adaptive doubling of the slices as in `solve_transfer_matrix(error_control='full')`).
    max_slices : int
        Upper limit of the adaptive doubling.
    slices : int, optional
        Fixed number of slices (no adaptivity; for convergence studies).

    Method (CONVENTIONS.md section 6). Propagating modes only and no loss: the transfer matrix of the twist (one
    `expm` of the whole section for a constant rate, which is exact; a product of 4th-order Magnus slice exponentials
    otherwise), converted with `from_transfer_matrix`. With evanescent modes or loss, the GSM is built directly:
    every slice's transfer matrix is converted to a GSM and the slices are joined by Redheffer star products, with
    slices short enough that |Im beta| h <= 4 and ||Lambda|| h <= 0.5 (no large growing exponential). For a constant
    rate the slices are identical and are doubled by star products (2^k slices; exact up to round-off).
    """
    N = len(basis)
    d, X = twist_system_matrices(basis, f, K)
    D = np.diag(d)
    if loss is not None:
        D = D - np.asarray(loss, dtype=complex)
    beta = basis.beta(f)
    prop = basis.propagating_mask(f)
    use_gsm = bool(loss is not None or not np.all(prop))
    # conditioning cap on the slice length for the GSM route
    hmax = np.inf
    if np.any(~prop):
        hmax = min(hmax, 4.0 / np.max(np.abs(beta[~prop].imag)))
    if loss is not None:
        nrm = np.linalg.norm(loss, 1)
        if nrm > 0:
            hmax = min(hmax, 0.5 / nrm)
    info = dict(route="gsm" if use_gsm else "transfer matrix", n_modes=N)

    if rate.is_constant:
        M = D + rate(np.array([0.0]))[0] * X
        if not use_gsm:
            S = tuple(x[0] for x in _gsm_from_transfer_stack(expm(length * M)[None], N))
            info.update(n_slices=1, error_estimate=0.0, success=True)
            return S, info
        k = 0 if not np.isfinite(hmax) else max(0, int(np.ceil(np.log2(length / hmax))))
        S = tuple(x[0] for x in _gsm_from_transfer_stack(expm(length / 2 ** k * M)[None], N))
        for _ in range(k):
            S = _star(S, S)
        info.update(n_slices=2 ** k, error_estimate=0.0, success=True)
        return S, info

    points = [0.0] + [p for p in rate.breakpoints if 0 < p < length] + [float(length)]
    seg = np.diff(points)
    base = [max(1, int(np.ceil(4 * s / length))) for s in seg]
    if np.isfinite(hmax):
        base = [max(c, int(np.ceil(s / hmax))) for c, s in zip(base, seg)]
    if slices is not None:
        # fixed slicing: distribute over the segments in proportion to their length (at least one each)
        base = [max(1, int(round(slices * s / length))) for s in seg]
        edges = _slice_edges(length, base, points, 0)
        S = _sliced_solution(D, X, rate, edges, use_gsm, N)
        info.update(n_slices=len(edges) - 1, error_estimate=np.nan, success=True)
        return S, info
    # minimum slicing: every pair that can exchange an amplitude above atol slips by at most pi per slice
    zz = np.linspace(0.0, length, 513)
    tmax = float(np.max(np.abs(rate(zz))))
    br = np.where(prop, beta.real, 0.0)
    pp = np.outer(prop, prop)

    def max_relevant(phase, blk):
        size = tmax * np.abs(blk) / np.maximum(phase, 1e-300)
        rel = pp & (size >= atol)
        return float(np.max(phase[rel])) if np.any(rel) else 0.0

    dphase = max(max_relevant(np.abs(br[:, None] - br[None, :]), X[:N, :N]),
                 max_relevant(br[:, None] + br[None, :], X[:N, N:]))
    n_min = int(np.ceil(dphase * length / np.pi))
    level = 0
    while sum(base) * 2 ** level < min(n_min, max_slices // 8) and sum(base) * 2 ** (level + 1) <= max_slices:
        level += 1
    edges = _slice_edges(length, base, points, level)
    S = _sliced_solution(D, X, rate, edges, use_gsm, N)
    n = len(edges) - 1
    status, err, diff_prev, ratio_prev, small_prev, history = 1, np.inf, None, None, False, []
    while 2 * n <= max_slices:
        S_prev = S
        level += 1
        edges = _slice_edges(length, base, points, level)
        n = len(edges) - 1
        S = _sliced_solution(D, X, rate, edges, use_gsm, N)
        diff = max(float(np.max(np.abs(x - y))) for x, y in zip(S, S_prev))
        err = diff / 15.0
        ratio = np.inf if diff == 0 else (diff_prev / diff if diff_prev is not None else None)
        asymptotic = diff == 0 or (ratio is not None and ratio >= 8.0 and ratio_prev is not None
                                   and ratio_prev >= 8.0)
        ratio_prev = ratio
        history.append((n, err))
        small = err <= atol * 1e-3
        if err <= atol and (asymptotic or (small and small_prev)):
            status = 0
            break
        small_prev, diff_prev = small, diff
    info.update(n_slices=n, error_estimate=err, success=status == 0, history=history)
    return S, info


# ============================================================================================ component
class Twist(Component):
    """Twist: a straight a x b guide whose cross-section rotates about its axis by `angle_deg` over `length`.

    Parameters
    ----------
    angle_deg : float
        Total twist angle (degrees). Positive: the cross-section turns from the global +x axis towards +y (from u
        towards v at port 1), i.e. counterclockwise seen with the direction of propagation (port 1 to port 2, +z)
        pointing at the observer; right-handed about +z. This is the theta of T3 and T3b. The rotation axis is the
        centre of the cross-section.
    length : float
        Length (m) along the axis, > 0. Reference planes are the end planes of the twist (CONVENTIONS.md section 6).
    profile : 'constant', 'cosine', 'triangular' or callable
        Rate profile (see `TwistProfile`): 'constant' theta' = theta/L; 'cosine' theta' = (pi theta/2L) sin(pi z/L)
        (Doane & Anderson 1989, Fig. 9, "cosine"; a half-cosine rate, zero at both ends); 'triangular' Doane &
        Anderson eq. 17 (linear rise to 2 theta/L at L/2, linear fall); or a callable rate shape r(t), t = z/L,
        normalized internally so that the integral of theta' is theta.
    guide : Guide, optional
        Global cross-section (w along x, h along y) and wall conductivity; a `Line` attaches its own.
    cutoff_factor : float, optional
        Internal truncation. None (default): solve on the port basis, completed (see Method). A number >= 1: solve
        on all modes of every family with cutoff below cutoff_factor * f (`evanescent_families='all'`), plus the port
        modes, and return the port sub-block.
    atol, max_slices, slices : see `twist_gsm_blocks` (adaptive slicing of non-constant rates).
    breakpoints : t values in (0, 1) of kinks of a callable rate shape.

    Frames (CONVENTIONS.md section 2). The global guide frame rotates with the cross-section, so the GSM has the
    line's global-frame basis at **both** ports for any angle (`basis2 == basis1`), and the guide after the twist is
    the same w x h guide. No `FrameChange` follows a twist: the angle only changes the orientation in space of what
    follows (after a 90 degree twist, an H-plane bend bends in the spatial plane that was the E plane before it).
    `plane = 'H'`: the local frame is the global frame (u = x, v = y). The twist commutes with the H <-> E frame
    change (T3 box T3-5), so an 'E' version would give the same physics; it is not provided.

    Method. T3b box 4 (`twist_system_matrices`), linear in theta', with the coefficients of T3b box 2b; every
    family is coupled (selection rule m + m' and n + n' odd), so the port basis is never split into families.
    `twist_gsm_blocks` integrates it (exact `expm` for a constant rate; 4th-order Magnus slices with automatic
    doubling otherwise) and builds the GSM directly when evanescent modes or loss are present. The port basis is
    completed before solving: with the propagating modes it lacks, and, if it contains evanescent modes, with every
    mode (all families) whose cutoff does not exceed that of its highest mode, since a twist couples all families
    while the line's 'live' rule keeps only some (T3b section 9: evanescent modes of Doane's twist change the
    converted powers by about 6 % relative). The returned GSM is the sub-block of the port basis; the dropped modes
    are taken as matched (a uniform guide continues on both sides), which needs the neighbours to be far enough for
    them to decay (|beta| d >> 1 for every dropped mode e at distance d).

    Accuracy. The modal series converges algebraically (like 1/chi_max) for strong twists (T3 section 7, T3b section
    5); for practical rates (tau a << 1) the truncation effects are of order (tau/k)^2. A mode just above or below
    cutoff couples strongly (the power-wave coupling grows like |beta|^(-1/2)); results within about 1 % of a cutoff
    depend on `cutoff_factor` (docs/reports/WP6.md).

    Wall loss (guide with a conductivity): the T1 loss matrix of the constant cross-section,
    `wrmodes.loss.loss_matrix_full` (propagating modes coupled, evanescent modes their own Lambda^+_pp), is added in
    the co-rotating frame: dy/dz = (D - Lambda_full + theta' X) y. Neglected: the correction of the wall integrals by
    the helicoidal wall, of order (tau rho)^2 for the wall area and of order tau rho in the tangential field; the
    latter is odd under each mirror (like the twist coupling), so it cannot change the attenuation of any mode at
    first order in tau and only perturbs the twist coupling by a relative amount of order R_s / eta0 (about 1e-4).

    Each GSM carries `info` (solver diagnostics: route, n_modes, n_slices, error_estimate, success).
    """

    plane = "H"

    def __init__(self, angle_deg, length, profile="constant", guide=None, cutoff_factor=None, atol=1e-6,
                 max_slices=16384, slices=None, breakpoints=None, name=""):
        length = float(length)
        if not length > 0:
            raise ValueError("length must be positive (a zero-length twist is not a physical frame change; see "
                             "zero_length_twist_transfer_matrix)")
        if guide is not None and not isinstance(guide, Guide):
            raise TypeError("guide must be a wrmodes.components.Guide")
        if cutoff_factor is not None and cutoff_factor < 1:
            raise ValueError("cutoff_factor must be >= 1")
        self.angle_deg, self.length = float(angle_deg), length
        self.profile = profile
        self.breakpoints = breakpoints
        self.rate = TwistProfile(profile, np.radians(self.angle_deg), length, breakpoints)
        self.guide = guide
        self.cutoff_factor = None if cutoff_factor is None else float(cutoff_factor)
        self.atol, self.max_slices = float(atol), int(max_slices)
        self.slices = None if slices is None else int(slices)
        pname = profile if isinstance(profile, str) else "custom"
        self.name = name or f"{self.angle_deg:g} deg {pname} twist {length:g} m"

    def _kwargs(self):
        return dict(profile=self.profile, guide=self.guide, cutoff_factor=self.cutoff_factor, atol=self.atol,
                    max_slices=self.max_slices, slices=self.slices, breakpoints=self.breakpoints, name=self.name)

    def __repr__(self):
        p = self.profile if isinstance(self.profile, str) else "custom"
        return f"Twist({self.angle_deg!r}, {self.length!r}, profile={p!r})"

    def with_guide(self, guide):
        kw = self._kwargs()
        kw["guide"] = guide
        return type(self)(self.angle_deg, self.length, **kw)

    def replace(self, **changes):
        """A copy with some constructor arguments changed (e.g. cutoff_factor, slices, angle_deg)."""
        kw = self._kwargs()
        angle, length = changes.pop("angle_deg", self.angle_deg), changes.pop("length", self.length)
        kw.update(changes)
        return type(self)(angle, length, **kw)

    def default_basis(self, f, cutoff_factor=None):
        """Port bases (equal) from the component's guide, global frame 'H'.

        cutoff_factor None means the component's own cutoff_factor, or 1 (propagating modes) if that is None too.
        Above 1, evanescent modes of **all** families are kept (`evanescent_families='all'`), since a twist couples
        every family."""
        cf = self.cutoff_factor if cutoff_factor is None else cutoff_factor
        cf = 1.0 if cf is None else float(cf)
        if self.guide is None:
            raise ValueError("Twist.default_basis needs a guide (Twist(..., guide=Guide(w, h)))")
        b = self.guide.basis(f, cf, plane="H", evanescent_families="all" if cf > 1 else "live")
        return b, b

    def solve_basis(self, f, basis1):
        """The basis the twist is solved on for port basis `basis1` (see the class docstring, Method)."""
        a, b = basis1.a, basis1.b
        prop = basis1.propagating_mask(f) if len(basis1) else np.zeros(0, bool)
        modes = set(basis1.modes) | set(ModeBasis.below(f, a, b).modes)
        if self.cutoff_factor is not None and self.cutoff_factor > 1:
            modes |= set(ModeBasis.below(self.cutoff_factor * f, a, b).modes)
        if np.any(~prop):
            chi = max(np.pi * np.hypot(m.m / a, m.n / b) for m in basis1.modes)
            fmax = chi * cst.c / (2 * np.pi)
            modes |= {m for m in ModeBasis.below(fmax * (1 + 1e-9), a, b).modes
                      if np.pi * np.hypot(m.m / a, m.n / b) <= chi * (1 + 1e-12)}
        return ModeBasis(a, b, modes, frame=basis1.frame)

    def gsm(self, f, basis1=None, basis2=None):
        """GSM at f (Hz) in the global-frame basis `basis1` (same basis at both ports)."""
        if basis1 is None:
            basis1 = self.default_basis(f)[0]
        if basis2 is not None and basis2 != basis1:
            raise ValueError("a twist has the same basis at both ports (the global frame rotates with the guide)")
        if basis1.frame not in (None, "H"):
            raise ValueError(f"{self.name}: basis has frame {basis1.frame!r}, expected 'H' (insert a FrameChange, "
                             f"or use a Line)")
        if len(basis1) == 0:
            Z = np.zeros((0, 0))
            out = GSM(Z, Z, Z, Z, basis1, basis1, f, self.name)
            out.info = {}
            return out
        if self.guide is not None:
            a, b = self.guide.dims("H")
            if not (np.isclose(basis1.a, a, rtol=1e-12, atol=0) and np.isclose(basis1.b, b, rtol=1e-12, atol=0)):
                raise ValueError(f"basis1 is for a {basis1.a:g} x {basis1.b:g} guide, the twist's guide is {a:g} x {b:g}")
        sb = self.solve_basis(f, basis1)
        Zs = None if self.guide is None else self.guide.wall_impedance(f)
        loss = None if Zs is None else loss_matrix_full(sb, f, Zs)
        with _blas_limit(1):
            S, info = twist_gsm_blocks(sb, f, self.rate, self.length, loss=loss, atol=self.atol,
                                       max_slices=self.max_slices, slices=self.slices)
        if not info.get("success", True):
            import warnings
            warnings.warn(f"{self.name}: tolerance {self.atol:g} not reached with {info['n_slices']} slices at "
                          f"{f / 1e9:.4g} GHz (error estimate {info['error_estimate']:.1e})", RuntimeWarning)
        idx = [sb.index(m) for m in basis1.modes]
        ix = np.ix_(idx, idx)
        out = GSM(S[0][ix], S[1][ix], S[2][ix], S[3][ix], basis1, basis1, f, self.name)
        out.info = info
        return out

    def convergence_study(self, f, input_mode="TE1,0", cutoff_factors=(1.0, 1.5, 2.0, 3.0), slice_factors=(1, 2, 4)):
        """Convergence of the twist at f (Hz) in the mode count and in the slicing.

        Returns a dict with
        * 'modes': list of rows (cutoff_factor, n_modes, n_slices, powers), one per cutoff factor, where powers
          is a dict mode name -> |S21[p, in]|^2 for the propagating output modes, plus 'reflected' (sum of
          |S11[p, in]|^2) and 'spurious' (sum over p != in);
        * 'slices': rows (n_slices, powers) at cutoff_factor = the first of `cutoff_factors`, for fixed slicings of
          slice_factors times the adaptive count (constant rates are exact and give one row).
        The port basis is the propagating basis; evanescent modes enter only inside the solve.
        """
        if self.guide is None:
            raise ValueError("convergence_study needs a guide")
        port = self.guide.basis(f, 1.0)
        mode = Mode.parse(input_mode)
        j = port.index(mode)

        def powers(g):
            col = np.abs(g.S21[:, j]) ** 2
            out = {m.name: float(c) for m, c in zip(port.modes, col)}
            out["reflected"] = float(np.sum(np.abs(g.S11[:, j]) ** 2))
            out["spurious"] = float(np.sum(col) - col[j])
            return out

        rows = []
        for cf in cutoff_factors:
            g = self.replace(cutoff_factor=cf).gsm(f, port)
            rows.append((float(cf), g.info["n_modes"], g.info["n_slices"], powers(g)))
        srows = []
        tw = self.replace(cutoff_factor=cutoff_factors[0])
        n0 = rows[0][2]
        for s in (slice_factors if not self.rate.is_constant else (1,)):
            g = tw.replace(slices=n0 * s).gsm(f, port) if not self.rate.is_constant else tw.gsm(f, port)
            srows.append((g.info["n_slices"], powers(g)))
        return dict(modes=rows, slices=srows)
