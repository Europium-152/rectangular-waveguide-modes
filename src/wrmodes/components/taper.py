"""Components with a varying cross-section: `VariableGuide`, `Taper` and `StrangledBend` (card WP5).

Theory: docs/theory/T2-taper-coupling.md (T2) and docs/theory/T2b-taper-coupling.md (T2b), which agree to 7e-15.
Conventions: CONVENTIONS.md (frames section 2, bases and truncation section 3, amplitudes section 5, GSMs and
reference planes section 6).

Equations (T2 box of the summary, T2b B1-B3). On every cross-section z (z along the reference line of curvature
kappa(z); with u0 = v0 = 0 the reference line is the guide axis), the local-mode voltages and currents
E_t = sum V_q e_q, H_t = sum I_q z x e_q obey

    dV/dz = T V - i Z I,        dI/dz = -i Y V - T^T I,

with T = N^T of T2 (T2b B3 closed forms, linear in a', b', u0', v0', no beta), and Z, Y the T0 curvature matrices of
the local a(z), b(z) (T2 section 3, `V-I coefficients`, with h0 = 1 + kappa u0). Everything is a polynomial in
omega and beta^2: the system is regular through cutoff (T2 section 8.1).

Wall loss (guide with a conductivity): to first order, the T1 matrices of the **local** cross-section enter as a
series impedance and a shunt admittance (T1 box B2): Z -> Z - i N_w, Y -> Y - i G_w, with the wall integrals
weighted by h = 1 + kappa (u0 + u~) (T1 boxes B2a, B4b). Following the rule of T1 section 5, off-diagonal loss
entries are kept only between modes that propagate somewhere in the component (in a uniform guide: the propagating
modes, as `Straight` and `SmoothBend`); every mode keeps its diagonal entries. The tilt of the walls and the
boundary terms of impedance walls on a moving wall (T2 section 13) are neglected (second order).

Method (T2 section 8.2, card WP5 item 3):

1. Mode classes: the geometry couples only modes with the same class key (selection rules of T2 section 5,
   T2b section 4): a' couples Delta m even, u0' and kappa Delta m odd, b' Delta n even, v0' Delta n odd; equal lossy
   walls keep the parities of m and n. Each class of the port modes is solved separately.
2. Internal mode list of a class: every mode of the class with min_z chi_p(z) < mode_factor * k (the cutoff factor
   applied at the widest cross-section), plus the port modes of the class. A mode that propagates anywhere inside is
   always kept. The order is fixed along the component (`ModeBasis(..., sort=False)` semantics).
3. Slices: edges distributed with the density slice_factor * max(max_prop |beta_p| / 2, max_evan |beta_p| / 6,
   k / 2), at least 8 slices per segment; no interior edge within delta_c k^2 of a cutoff (|k^2 - chi^2| >=
   1e-6 k^2, T2 section 8.2 step 2). Segments end at the breakpoints. Adaptive refinement (`atol`, default
   1e-6): the density is raised until the Richardson estimate of the 4th-order error of the entries of the port
   modes is below atol (`_solve_class`). On a `uniform` interval (constant a, b, u0, v0, kappa) one slice of
   length <= 3 / max |Im beta| is computed and doubled by star products (exact).
4. Propagator per slice: 4th-order Magnus with two Gauss points, Omega = h/2 (M1 + M2) + sqrt(3)/12 h^2 [M2, M1].
5. Slice GSM built directly from the propagator with the local admittances at the slice ends (T2 section 8.2
   step 4), no global transfer matrix; slices joined by Redheffer star products (tree reduction).
6. Ports: the bases given (by default the line's rule at the end guides). Internal modes outside a port basis are
   dropped at that port (zero incident wave, outgoing wave discarded).

B5' (T2b section 7, on by default, `tail=False` to switch it off): the modes of the class outside the internal list
are eliminated quasi-statically, Y -> Y - T_out^T Z_out^-1 T_out, which turns the O(1/M) convergence of the plain
truncation into a much faster one. The sum runs explicitly over the outside indices up to (largest internal index +
`TAIL_EXPLICIT`) and the rest is added in closed form from its asymptote sum_r C_r / j^(2r), r = 1..`TAIL_TERMS`
(polygamma functions), fitted to the last explicit terms of each parity.
"""
import math
import warnings

import numpy as np
import scipy.constants as cst
from scipy.linalg import expm
from scipy.special import polygamma

from ..gsm import GSM
from ..modes import Mode, ModeBasis, ETA0
from ..profiles import get_profile
from .base import Component
from .guide import Guide, _check_plane
from .twist import _star, _star_reduce
from ..transfer_matrix import _blas_limit

__all__ = ["VariableGuide", "Taper", "StrangledBend", "GeometryFunction", "vi_system", "taper_coupling_matrix",
           "convergence_study"]

DELTA_C = 1e-6
"""Minimum |k^2 - chi_p^2| / k^2 at an interior slice edge (T2 section 8.2 step 2)."""
GAMMA = 2.0
"""Slice density: max_p |beta_p| h <= GAMMA / slice_factor on every non-uniform slice."""
GAMMA_EVANESCENT = 6.0
"""Slice density for the evanescent modes: max_p |Im beta_p| h <= GAMMA_EVANESCENT / slice_factor."""
GROWTH_MAX = 3.0
"""Largest evanescent growth max |Im beta| h of one slice of a uniform interval."""
MIN_SLICES = 8
"""Minimum number of slices per non-uniform segment."""
TAIL_EXPLICIT = 16
"""B5': outside indices summed explicitly beyond the largest internal index (the rest by its asymptote)."""
TAIL_TERMS = 4
"""B5': number of terms C_r / j^(2r) of the asymptote of the remaining sum, fitted to the last explicit terms."""
_GAUSS = (0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0)
_MAGNUS_C = np.sqrt(3.0) / 12.0


# ============================================================================================ geometry
class GeometryFunction:
    """A function of the centre-line coordinate z (m) with its derivative: value(z) and derivative(z), vectorized.

    Built from a constant, a callable with a ``derivative`` attribute, a pair (f, df) of callables, or a plain
    callable, whose derivative is then taken by 4th-order central differences with step `step` (m) and a
    RuntimeWarning.
    """

    def __init__(self, f, df=None, step=1e-6, name="function"):
        self.const = None
        if df is None and isinstance(f, tuple):
            f, df = f
        if np.isscalar(f) or (isinstance(f, np.ndarray) and f.ndim == 0):
            self.const = float(f)
            self.f, self.df = None, None
        else:
            if not callable(f):
                raise TypeError(f"{name} must be a number or a callable of z")
            if df is None and hasattr(f, "derivative"):
                df = f.derivative
            if df is None:
                warnings.warn(f"{name}: no derivative given; it is taken by finite differences (about 1e-9 "
                              f"relative accuracy). Pass (f, df) for exact coefficients.", RuntimeWarning,
                              stacklevel=3)
            self.f, self.df = f, df
        self.step = float(step)

    @property
    def is_constant(self):
        return self.const is not None

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        if self.const is not None:
            return np.full(z.shape, self.const)
        return np.asarray(self.f(z), dtype=float) * np.ones(z.shape)

    def derivative(self, z):
        z = np.asarray(z, dtype=float)
        if self.const is not None:
            return np.zeros(z.shape)
        if self.df is not None:
            return np.asarray(self.df(z), dtype=float) * np.ones(z.shape)
        h = self.step
        return (self(z - 2 * h) - 8 * self(z - h) + 8 * self(z + h) - self(z + 2 * h)) / (12 * h)

    def __repr__(self):
        return repr(self.const) if self.const is not None else f"GeometryFunction({self.f!r})"


def _fun(x, name):
    if x is None:
        return GeometryFunction(0.0, name=name)
    if isinstance(x, GeometryFunction):
        return x
    return GeometryFunction(x, name=name)


class _Geometry:
    """Local geometry arrays at points z: a, b, da, db, u0, du0, dv0, kappa."""

    __slots__ = ("z", "a", "b", "da", "db", "u0", "du0", "dv0", "kappa")

    def __init__(self, comp, z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        self.z = z
        self.a, self.da = comp.a(z), comp.a.derivative(z)
        self.b, self.db = comp.b(z), comp.b.derivative(z)
        self.u0, self.du0 = comp.u0(z), comp.u0.derivative(z)
        self.dv0 = comp.v0.derivative(z)
        self.kappa = comp.kappa(z)


# ============================================================================================ coefficients
class _Modes:
    """Integer arrays of a mode list: te (bool), m, n, eps_m, eps_n."""

    def __init__(self, modes):
        self.modes = tuple(modes)
        self.te = np.array([md.kind == "TE" for md in modes], dtype=bool)
        self.m = np.array([md.m for md in modes], dtype=float)
        self.n = np.array([md.n for md in modes], dtype=float)
        self.em = np.where(self.m == 0, 1.0, 2.0)
        self.en = np.where(self.n == 0, 1.0, 2.0)

    def __len__(self):
        return len(self.modes)


def _col(x):
    return np.asarray(x, dtype=float)[:, None, None]


def taper_coupling_matrix(rows, cols, a, b, da, db, du0, dv0):
    """T_pq (1/m) of dV_p/dz = ... + sum_q T_pq V_q, T2b box B3 (= N_qp of T2), vectorized over z.

    rows, cols : sequences of Mode (p and q); a, b, da, db, du0, dv0 : arrays of shape (G,) (or scalars).
    Returns an array (G, len(rows), len(cols)). Real, frequency independent, linear in the wall slopes.
    """
    R = rows if isinstance(rows, _Modes) else _Modes(rows)
    C = cols if isinstance(cols, _Modes) else _Modes(cols)
    a, b, da, db, du0, dv0 = (_col(np.atleast_1d(x)) for x in (a, b, da, db, du0, dv0))
    m, n, te_p = R.m[:, None], R.n[:, None], R.te[:, None]
    mq, nq, te_q = C.m[None, :], C.n[None, :], C.te[None, :]
    emp, enp = R.em[:, None], R.en[:, None]
    emq, enq = C.em[None, :], C.en[None, :]
    Np2 = m ** 2 * b ** 2 + n ** 2 * a ** 2
    Nq2 = mq ** 2 * b ** 2 + nq ** 2 * a ** 2
    NpNq = np.sqrt(Np2 * Nq2)
    du = (n == nq)
    dv = (m == mq)
    Ou = np.where(du, np.where((m + mq) % 2 == 0, da, -2.0 * du0), 0.0)
    Ov = np.where(dv, np.where((n + nq) % 2 == 0, db, -2.0 * dv0), 0.0)
    same = du & dv & (te_p == te_q)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.where(same, 1.0, Np2 - Nq2)
        tete = -Nq2 / den * (np.sqrt(emp * emq) * b ** 2 * m ** 2 / a * Ou
                             + np.sqrt(enp * enq) * a ** 2 * n ** 2 / b * Ov) / NpNq
        tmtm = -Np2 / den * 2.0 * (m * mq * b ** 2 / a * Ou + n * nq * a ** 2 / b * Ov) / NpNq
        tmte = a * b * np.sqrt(emq * enq) * m * n / NpNq * (Ou / a - Ov / b)
        dte = -emp * enp / (4.0 * Np2) * (n ** 2 * a * da + m ** 2 * b * db)
        dtm = -(m ** 2 * b ** 2 * da / a + n ** 2 * a ** 2 * db / b) / Np2
    T = np.where(te_p & te_q, tete, 0.0)
    T = np.where(~te_p & ~te_q, tmtm, T)
    T = np.where(~te_p & te_q, tmte, T)
    T = np.where(same & te_p, dte, T)
    T = np.where(same & ~te_p, dtm, T)
    T = np.where(du | dv, T, 0.0)
    return np.where(np.isfinite(T), T, 0.0)


def _curvature_parts(M, a, b):
    """U, U^d, U^g (T2 section 3, centred u~) for the local a, b: arrays (G, N, N)."""
    a, b = _col(a), _col(b)
    m, n, te = M.m, M.n, M.te
    km, kn = m * np.pi / a, n * np.pi / b                     # (G, 1, N)
    chi = np.sqrt(km ** 2 + kn ** 2)
    tu = np.where(te, -kn, km) / chi
    tv = np.where(te, km, kn) / chi
    mp, mq = m[:, None], m[None, :]
    odd = (mp + mq) % 2 == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.where(odd, (mp ** 2 - mq ** 2) ** 2, 1.0)
        Gs = np.where(odd, -8.0 * a * mp * mq / (np.pi ** 2 * den), 0.0)
        Gc = np.where(odd, -2.0 * a * np.sqrt(M.em[:, None] * M.em[None, :]) * (mp ** 2 + mq ** 2)
                      / (np.pi ** 2 * den), 0.0)
    same_n = (n[:, None] == n[None, :])
    tu_p, tu_q = np.swapaxes(tu, -1, -2), tu
    tv_p, tv_q = np.swapaxes(tv, -1, -2), tv
    chi_p, chi_q = np.swapaxes(chi, -1, -2), chi
    U = same_n * (tu_p * tu_q * Gc + tv_p * tv_q * Gs)
    tm2 = np.outer(~te, ~te)
    te2 = np.outer(te, te)
    Ud = same_n * tm2 * chi_p * chi_q * Gs
    Ug = same_n * te2 * chi_p * chi_q * Gc
    return U, Ud, Ug


def _wall_parts(M, a, b):
    """Wall integrals summed over four equal walls (T1 box B2a, `loss.wall_integrals`), vectorized over z:
    (N0, G0, N1, G1), arrays (G, N, N): order 0 (straight) and order 1 (coefficient of kappa, centred u)."""
    a, b = _col(a), _col(b)
    m, n, te = M.m, M.n, M.te
    km, kn = m * np.pi / a, n * np.pi / b
    chi2 = km ** 2 + kn ** 2
    c = np.where(te, np.sqrt(M.em * M.en), 2.0) / (np.sqrt(a * b) * np.sqrt(chi2))
    A = np.where(te, -c * kn, c * km)
    B = np.where(te, c * km, c * kn)
    cc = chi2 * c * te
    T = lambda x: np.swapaxes(x, -1, -2)
    AA, BB, gg = T(A) * A, T(B) * B, T(cc) * cc
    mp, mq = m[:, None], m[None, :]
    same_n = n[:, None] == n[None, :]
    same_m = mp == mq
    par_m = (-1.0) ** (mp + mq)
    par_n = (-1.0) ** (n[:, None] + n[None, :])
    en_q, em_q = M.en[None, :], M.em[None, :]
    # side walls u = +-a/2 (sum of u+ and u-): order 0 weight 1, order 1 weight u = +-a/2
    Nu = AA * same_n * (b / 2)
    Gu = gg * same_n * (b / en_q)
    N0 = Nu * (1 + par_m)
    G0 = Gu * (1 + par_m)
    N1 = Nu * (a / 2) * (par_m - 1)
    G1 = Gu * (a / 2) * (par_m - 1)
    # broad walls v = +-b/2
    S0 = same_m * (a / 2) * (mq >= 1)
    C0 = same_m * (a / em_q)
    N0 = N0 + BB * S0 * (1 + par_n)
    G0 = G0 + gg * C0 * (1 + par_n)
    odd = (mp + mq) % 2 == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.where(odd, (mp ** 2 - mq ** 2) ** 2, 1.0)
        S1 = np.where(odd, -(a / np.pi) ** 2 * 4.0 * mp * mq / den, 0.0)
        C1 = np.where(odd, -(a / np.pi) ** 2 * 2.0 * (mp ** 2 + mq ** 2) / den, 0.0)
    N1 = N1 + BB * S1 * (1 + par_n)
    G1 = G1 + gg * C1 * (1 + par_n)
    return N0, G0, N1, G1


def vi_system(modes, f, geo, Zs=None, loss_set=None, tail_modes=None, tail_remainder=None):
    """Matrices M(z) (G, 2N, 2N) of d[V; I]/dz = M [V; I] at the points of `geo` (a `_Geometry`).

    M = [[T, -i Z_eff], [-i Y_eff, -T^T]] with Z_eff = Z - i N_w, Y_eff = Y - i G_w - tail (see the module
    docstring). Zs: wall surface impedance (ohm) or None. loss_set: boolean (N,) mask of the modes coupled through
    the loss (diagonal entries are always kept). tail_modes: `_Modes` of the outside modes for B5' (or None);
    tail_remainder: list of ([index arrays into tail_modes of the modes at indices J, J - 2, ...], J) for the
    closed-form remainder of the B5' sum over the same-parity indices beyond J (`_tail`).
    """
    M = modes if isinstance(modes, _Modes) else _Modes(modes)
    N = len(M)
    k = 2 * np.pi * f / cst.c
    G = len(geo.z)
    T = taper_coupling_matrix(M, M, geo.a, geo.b, geo.da, geo.db, geo.du0, geo.dv0)
    a3, b3 = _col(geo.a), _col(geo.b)
    chi2 = (M.m * np.pi / a3) ** 2 + (M.n * np.pi / b3) ** 2     # (G, 1, N)
    h0 = 1.0 + geo.kappa * geo.u0
    eye = np.eye(N, dtype=bool)
    zdiag = np.where(M.te, k * ETA0, (k * k - chi2[:, 0, :]) * ETA0 / k) * h0[:, None]
    ydiag = np.where(M.te, (k * k - chi2[:, 0, :]) / (k * ETA0), k / ETA0) * h0[:, None]
    Z = np.zeros((G, N, N), dtype=complex)
    Y = np.zeros((G, N, N), dtype=complex)
    Z[:, eye] = zdiag
    Y[:, eye] = ydiag
    kap = _col(geo.kappa)
    if np.any(geo.kappa != 0):
        U, Ud, Ug = _curvature_parts(M, geo.a, geo.b)
        Z += kap * (k * ETA0 * U - Ud * ETA0 / k)
        Y += kap * (k / ETA0 * U - Ug / (k * ETA0))
    if Zs is not None and Zs != 0:
        N0, G0, N1, G1 = _wall_parts(M, geo.a, geo.b)
        h0c = _col(h0)
        Nw = Zs * (h0c * N0 + kap * N1)
        Gw = Zs / (k * ETA0) ** 2 * (h0c * G0 + kap * G1)
        keep = np.outer(loss_set, loss_set) | eye if loss_set is not None else np.ones((N, N), bool)
        Z = Z - 1j * np.where(keep, Nw, 0.0)
        Y = Y - 1j * np.where(keep, Gw, 0.0)
    if tail_modes is not None and len(tail_modes):
        Y = Y - _tail(tail_modes, M, k, geo, tail_remainder or ())
    out = np.empty((G, 2 * N, 2 * N), dtype=complex)
    out[:, :N, :N] = T
    out[:, :N, N:] = -1j * Z
    out[:, N:, :N] = -1j * Y
    out[:, N:, N:] = -np.swapaxes(T, 1, 2)
    return out


def _tail(out, M, k, geo, remainder=()):
    """B5' correction T_out^T Z_out^-1 T_out (G, N, N), T2b box B5', plus the closed-form remainder.

    remainder: list of (index sets, J). For each, the terms f(j) of the indices j = J, J - 2, ..., J - 2(K - 1)
    (rows `idx` of the outside list) are fitted by f(j) = sum_r C_r / j^(2r), r = 1..K, and the sum over
    j = J + 2, J + 4, ... is added in closed form, sum_{i>=1} (J + 2i)^(-2r) = psi^(2r-1)(J/2 + 1) / ((2r - 1)! 4^r).
    """
    Tout = taper_coupling_matrix(out, M, geo.a, geo.b, geo.da, geo.db, geo.du0, geo.dv0)
    chi2 = (out.m * np.pi / geo.a[:, None]) ** 2 + (out.n * np.pi / geo.b[:, None]) ** 2
    Zq = np.where(out.te, k * ETA0, (k * k - chi2) * ETA0 / k)            # (G, Nout)
    W = Tout / Zq[:, :, None]
    total = np.einsum("gqp,gqr->gpr", W, Tout)
    for sets, J in remainder:
        K = len(sets)
        js = np.array([J - 2.0 * r for r in range(K)])
        V = js[:, None] ** (-2.0 * np.arange(1, K + 1))[None, :]
        F = np.stack([np.einsum("gqp,gqr->gpr", W[:, ix], Tout[:, ix]) for ix in sets])    # (K, G, N, N)
        C = np.tensordot(np.linalg.inv(V), F, axes=(1, 0))
        x = J / 2.0 + 1.0
        w = np.array([float(polygamma(2 * r - 1, x)) / (math.factorial(2 * r - 1) * 4.0 ** r)
                      for r in range(1, K + 1)])
        total = total + np.tensordot(w, C, axes=(0, 0))
    return total


# ============================================================================================ slicing
def _chi2_at(M, a, b):
    return (M.m[None, :] * np.pi / a[:, None]) ** 2 + (M.n[None, :] * np.pi / b[:, None]) ** 2


def _admittance(M, k, chi2):
    """Local wave admittances Y_p (lossless, CONVENTIONS.md section 5), (..., N); beta = -i|beta| below cutoff."""
    d = k * k - chi2
    beta = np.where(d > 0, np.sqrt(np.abs(d)) + 0j, -1j * np.sqrt(np.abs(d)))
    return np.where(M.te, beta / (k * ETA0), k / (ETA0 * np.where(beta == 0, np.nan, beta)))


def _slice_gsms(Phi, d0, d1):
    """GSM blocks of slices from their V-I propagators Phi (S, 2N, 2N) and sqrt(Y) at both ends (S, N)."""
    N = d0.shape[-1]
    s = 1.0 / (2.0 * np.sqrt(2.0))
    S_ = Phi.shape[0]
    X = np.zeros((S_, 2 * N, 2 * N), dtype=complex)
    W = np.zeros((S_, 2 * N, 2 * N), dtype=complex)
    i = np.arange(N)
    X[:, i, i] = s * d0
    X[:, i, N + i] = s / d0
    W[:, i, i] = s * d0
    W[:, i, N + i] = -s / d0
    top, bot = Phi[:, :N, :], Phi[:, N:, :]
    X[:, N:, :] = s * (d1[:, :, None] * top - bot / d1[:, :, None])     # a-(z1)
    W[:, N:, :] = s * (d1[:, :, None] * top + bot / d1[:, :, None])     # a+(z1)
    S = np.swapaxes(np.linalg.solve(np.swapaxes(X, 1, 2), np.swapaxes(W, 1, 2)), 1, 2)
    return S[:, :N, :N], S[:, :N, N:], S[:, N:, :N], S[:, N:, N:]


def _chunk(N):
    return int(max(1, min(512, 48e6 / (16 * (2 * N) ** 2 * 6))))


# ============================================================================================ component
class VariableGuide(Component):
    """Rectangular guide whose cross-section a(z) x b(z), centre offsets and curvature vary along z (T2, T2b).

    Parameters
    ----------
    length : float
        Length L (m) of the reference line; z in [0, L]. With u0 = v0 = 0 (default) it is the guide's centre line,
        as CONVENTIONS.md section 6 requires; with offsets the guide axis is at (u0(z), v0(z)) from it (T2b
        section 10).
    a, b : float, callable, (f, df) or GeometryFunction
        Local sides (m) in the frame of `plane` (a along u, b along v), functions of z. Exact derivatives are
        needed: give a callable with a ``derivative`` method or a pair (f, df); a plain callable is differentiated
        by finite differences with a warning. a and b must be continuous (a discontinuity is a step, `Step`).
    kappa : float or callable, optional
        Curvature (1/m) of the reference line, signed as in `SmoothBend` (kappa > 0 turns towards -u).
    u0, v0 : as a, optional
        Offsets (m) of the guide axis from the reference line; v0 enters only through v0'.
    plane : 'H' or 'E'
        Frame of the component's local dimensions and port bases (CONVENTIONS.md section 2).
    guide : Guide, optional
        The global guide at port 1 (w, h, sigma); its local dimensions must equal a(0), b(0). With a conductivity
        the wall loss is included (module docstring). In a `Line` a component without a guide is attached to the
        line's current guide. `guides` gives (input, output) global guides (CONVENTIONS.md section 10).
    mode_factor : float
        Cutoff factor f_c >= 1 of the internal mode list, applied at the widest cross-section (T2 section 8.2 step
        1). Default 2 (with B5'). See `convergence_study`.
    tail : bool
        Apply the B5' quasi-static tail correction of T2b section 7 (default True).
    slice_factor : float
        Initial slice density multiplier of the adaptive slicing (default 1; module docstring, item 3).
    atol : float or None
        Target accuracy of every complex entry of the port modes (Richardson estimate of the 4th-order Magnus
        method; default 1e-6, as `SmoothBend`). None: fixed slicing at `slice_factor`.
    max_doublings : int
        Upper limit of the adaptive refinements (each multiplies the density by 2 to 4); a RuntimeWarning is
        issued if atol is not reached.
    families : 'separate' or 'coupled'
        Lossy guides only, as in `SmoothBend`: 'separate' (default) keeps the families n of the geometry apart and
        neglects their coupling by the lossy walls v = +-b/2 (n with n +- 2; at most about 6e-6 of the power, T1b);
        'coupled' keeps it (classes by the parity of n), which equals a lossy `Straight` in the uniform limit.
    breakpoints : sequence of float, optional
        Positions (m) where a', b', u0', v0' or kappa jump; slices end there.
    uniform : sequence of (z0, z1), optional
        Intervals on which a, b, u0, v0 and kappa are constant; solved exactly by doubling one slice.
    name : str

    Conventions. Port 1 is at z = 0, port 2 at z = L; the reference planes are the end planes (CONVENTIONS.md
    section 6). The local frame at port 2 is that of port 1 transported along the reference line. The internal
    mode list is restricted to the classes of the port modes (selection rules, T2 section 5); a component with
    b' = v0' = 0 conserves the family n (CONVENTIONS.md section 3), otherwise it couples families. Port modes at
    (within 1e-6 k^2 of) their cutoff raise ValueError.

    Accuracy. The truncated equations converge as O(1/M) in the number of modes per index direction (T2 section
    8.4); B5' restores fast convergence (T2b section 7). Evanescent port modes drop from O(1) at the port; their
    entries are as accurate as the internal list allows near the port (use mode_factor >= the line's
    cutoff_factor). `convergence_study` reports the change with mode_factor, the Richardson extrapolation, the
    slice refinement and B5' on/off.
    """

    def __init__(self, length, a, b, kappa=None, u0=None, v0=None, plane="H", guide=None, mode_factor=2.0,
                 tail=True, slice_factor=1.0, families="separate", breakpoints=None, uniform=None, atol=1e-6,
                 max_doublings=4, name=""):
        length = float(length)
        if not length > 0:
            raise ValueError("length must be positive")
        self.length = length
        self.a, self.b = _fun(a, "a"), _fun(b, "b")
        self.kappa = _fun(kappa, "kappa")
        self.u0, self.v0 = _fun(u0, "u0"), _fun(v0, "v0")
        self.plane = _check_plane(plane)
        if guide is not None and not isinstance(guide, Guide):
            raise TypeError("guide must be a wrmodes.components.Guide")
        self.guide = guide
        self.mode_factor = float(mode_factor)
        if self.mode_factor < 1:
            raise ValueError("mode_factor must be >= 1")
        self.tail = bool(tail)
        self.slice_factor = float(slice_factor)
        self.atol = None if atol is None else float(atol)
        self.max_doublings = int(max_doublings)
        if families not in ("separate", "coupled"):
            raise ValueError("families must be 'separate' or 'coupled'")
        self.families = families
        bp = sorted({float(p) for p in (breakpoints or ()) if 0 < float(p) < length})
        self.breakpoints = tuple(bp)
        self.uniform = tuple((float(x0), float(x1)) for x0, x1 in (uniform or ()))
        for x0, x1 in self.uniform:
            if not 0 <= x0 < x1 <= length:
                raise ValueError("uniform intervals must lie in [0, length]")
        self.name = name or f"variable guide {length:g} m"
        g0 = _Geometry(self, [0.0, length])
        if np.any(g0.a <= 0) or np.any(g0.b <= 0):
            raise ValueError("a and b must be positive")
        if guide is not None:
            a0, b0 = guide.dims(self.plane)
            if not (np.isclose(a0, g0.a[0], rtol=1e-12, atol=0) and np.isclose(b0, g0.b[0], rtol=1e-12, atol=0)):
                raise ValueError(f"guide {guide!r} has local sides {a0:g} x {b0:g} in the {self.plane} frame, but "
                                 f"the component starts with {g0.a[0]:g} x {g0.b[0]:g}")

    # ------------------------------------------------------------------------------------- protocol
    def _kwargs(self):
        return dict(kappa=self.kappa, u0=self.u0, v0=self.v0, plane=self.plane, guide=self.guide,
                    mode_factor=self.mode_factor, tail=self.tail, slice_factor=self.slice_factor,
                    families=self.families, breakpoints=self.breakpoints, uniform=self.uniform, atol=self.atol,
                    max_doublings=self.max_doublings, name=self.name)

    def replace(self, **changes):
        """A copy with some constructor arguments changed (e.g. mode_factor, tail, slice_factor, guide)."""
        kw = self._kwargs()
        kw.update(changes)
        length = kw.pop("length", self.length)
        a, b = kw.pop("a", self.a), kw.pop("b", self.b)
        return VariableGuide(length, a, b, **kw)

    def with_guide(self, guide):
        return self.replace(guide=guide)

    def __repr__(self):
        return f"VariableGuide({self.length:g} m, plane={self.plane!r})"

    def _local_to_guide(self, a, b, sigma):
        return Guide(a, b, sigma) if self.plane == "H" else Guide(b, a, sigma)

    @property
    def guides(self):
        """(input Guide, output Guide) in the global frame, or None if the component has no guide yet (a `Line`
        then attaches its current guide)."""
        if self.guide is None:
            return None
        g = _Geometry(self, [self.length])
        return self.guide, self._local_to_guide(g.a[0], g.b[0], self.guide.sigma)

    def _guides(self):
        if self.guide is not None:
            return self.guides
        g = _Geometry(self, [0.0, self.length])
        return self._local_to_guide(g.a[0], g.b[0], None), self._local_to_guide(g.a[1], g.b[1], None)

    @property
    def centre_line_length(self):
        """Length of the guide axis (m); equal to `length` when u0 = v0 = 0 and the reference line is the axis."""
        if self.u0.is_constant and self.v0.is_constant and (self.kappa.is_constant and self.u0.const == 0):
            return self.length
        z = np.linspace(0, self.length, 4001)
        g = _Geometry(self, z)
        ds = np.sqrt((1 + g.kappa * g.u0) ** 2 + g.du0 ** 2 + g.dv0 ** 2)
        return float(np.trapezoid(ds, z))

    def default_basis(self, f, cutoff_factor=None):
        """Port bases: the line's rule (`Guide.basis`, 'live' families) at the two end guides, in the local frame of
        `plane`; cutoff_factor None means 1 (propagating modes). The internal mode count is `mode_factor`."""
        cf = 1.0 if cutoff_factor is None else float(cutoff_factor)
        g1, g2 = self._guides()
        return g1.basis(f, cf, plane=self.plane), g2.basis(f, cf, plane=self.plane)

    # ------------------------------------------------------------------------------------- structure
    def _segments(self):
        """[(z0, z1, uniform)] covering [0, L]."""
        pts = {0.0, self.length, *self.breakpoints}
        for x0, x1 in self.uniform:
            pts.update((x0, x1))
        pts = sorted(pts)
        out = []
        for z0, z1 in zip(pts[:-1], pts[1:]):
            if z1 - z0 <= 1e-15 * self.length:
                continue
            uni = any(x0 - 1e-12 * self.length <= z0 and z1 <= x1 + 1e-12 * self.length for x0, x1 in self.uniform)
            out.append((z0, z1, uni))
        return out

    def _flags(self):
        """Which couplings are present (sampled on 2001 points): a', u (u0' or kappa), b', v0'. A coupling that
        vanishes at every sample but not in between (not expected for smooth geometries) would be missed."""
        z = np.linspace(0, self.length, 2001)
        g = _Geometry(self, z)
        return dict(a=bool(np.any(g.da != 0)), u=bool(np.any(g.du0 != 0) or np.any(g.kappa != 0)),
                    b=bool(np.any(g.db != 0)), v=bool(np.any(g.dv0 != 0)))

    def _class_key(self, lossy):
        """(key(mode), flags): modes with different keys are not coupled (T2 section 5, T2b section 4). Along u:
        u0' or kappa couple every m; a' (or equal lossy walls) keeps the parity of m; otherwise m itself is kept.
        Along v: v0' couples every n; b' (or lossy walls with families='coupled') keeps the parity of n; otherwise
        n is kept (one family)."""
        fl = self._flags()
        ku = None if fl["u"] else ("par" if (fl["a"] or lossy) else "idx")
        coupled = lossy and self.families == "coupled"
        kv = None if fl["v"] else ("par" if (fl["b"] or coupled) else "idx")

        def key(md):
            u = None if ku is None else (md.m % 2 if ku == "par" else md.m)
            v = None if kv is None else (md.n % 2 if kv == "par" else md.n)
            return (u, v)
        return key, fl

    def _z_samples(self, n=1201):
        zs = [np.linspace(z0, z1, max(3, int(n * (z1 - z0) / self.length))) for z0, z1, _ in self._segments()]
        return np.unique(np.concatenate(zs))

    # ------------------------------------------------------------------------------------- solve
    def _internal_lists(self, f, port_modes, key, flags, mode_factor):
        """{class key: list of Mode} with the internal mode list of every class that has port modes."""
        k = 2 * np.pi * f / cst.c
        z = self._z_samples()
        g = _Geometry(self, z)
        amax, bmax = float(np.max(g.a)), float(np.max(g.b))
        kmax = mode_factor * k
        mmax, nmax = int(kmax * amax / np.pi) + 1, int(kmax * bmax / np.pi) + 1
        cand = []
        for m in range(mmax + 1):
            for n in range(nmax + 1):
                for kind in ("TE", "TM"):
                    if (kind == "TE" and m + n == 0) or (kind == "TM" and m * n == 0):
                        continue
                    if np.pi * np.hypot(m / amax, n / bmax) < kmax:
                        cand.append(Mode(kind, m, n))
        classes = {}
        for md in port_modes:
            classes.setdefault(key(md), set()).add(md)
        if cand:
            CM = _Modes(cand)
            chi_min = np.sqrt(np.min(_chi2_at(CM, g.a, g.b), axis=0))
            for md, c in zip(cand, chi_min):
                kk = key(md)
                if kk in classes and c < kmax:
                    classes[kk].add(md)
        out = {}
        for kk, mods in classes.items():
            mods = list(mods)
            CM = _Modes(mods)
            chi_min = np.sqrt(np.min(_chi2_at(CM, g.a, g.b), axis=0))
            order = np.lexsort((CM.n, CM.m, ~CM.te, chi_min))
            out[kk] = [mods[i] for i in order]
        return out

    def _tail_sets(self, modes, key, flags):
        """Outside modes for B5': explicit set, and the remainder sets [(index-J modes, factor)]."""
        dirs = []
        if flags["a"] or flags["u"]:
            dirs.append("u")
        if flags["b"] or flags["v"]:
            dirs.append("v")
        inside = set(modes)
        klass = key(modes[0])
        explicit, rem = set(), []
        for d in dirs:
            idx = [md.m for md in modes] if d == "u" else [md.n for md in modes]
            other = sorted({md.n for md in modes}) if d == "u" else sorted({md.m for md in modes})
            J = max(idx) + TAIL_EXPLICIT

            def mk(j, o):
                out = []
                for kind in ("TE", "TM"):
                    m, n = (j, o) if d == "u" else (o, j)
                    if (kind == "TE" and m + n > 0) or (kind == "TM" and m > 0 and n > 0):
                        md = Mode(kind, m, n)
                        if md not in inside and key(md) == klass:
                            out.append(md)
                return out
            for j in range(0, J + 1):
                for o in other:
                    explicit.update(mk(j, o))
            for par in (0, 1):
                jl = J if J % 2 == par else J - 1
                sets = [[md for o in other for md in mk(jl - 2 * r, o)] for r in range(TAIL_TERMS)]
                if sets[0] and all(len(x) == len(sets[0]) for x in sets):
                    rem.append((sets, jl))
        explicit = sorted(explicit, key=lambda md: (md.kind, md.m, md.n))
        if not explicit:
            return None, []
        pos = {md: i for i, md in enumerate(explicit)}
        rem = [([np.array([pos[md] for md in x]) for x in sets], J) for sets, J in rem]
        return _Modes(explicit), rem

    def _edges(self, z0, z1, CM, k, sf):
        """Slice edges of a non-uniform segment: density max_p |beta_p| / gamma, cutoff planes avoided."""
        zs = np.linspace(z0, z1, 801)
        g = _Geometry(self, zs)
        chi2 = _chi2_at(CM, g.a, g.b)
        bp = np.sqrt(np.maximum(k * k - chi2, 0.0))
        be = np.sqrt(np.maximum(chi2 - k * k, 0.0))
        q = np.maximum(np.max(bp, axis=1) * sf / GAMMA, np.max(be, axis=1) * sf / GAMMA_EVANESCENT)
        q = np.maximum(q, k * sf / GAMMA)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(zs))])
        n = max(MIN_SLICES, int(np.ceil(cum[-1])))
        edges = np.interp(np.linspace(0, cum[-1], n + 1), cum, zs)
        edges[0], edges[-1] = z0, z1
        # move interior edges away from cutoff planes (T2 section 8.2 step 2)
        def bad(zz):
            gg = _Geometry(self, zz)
            return np.any(np.abs(k * k - _chi2_at(CM, gg.a, gg.b)) < DELTA_C * k * k, axis=1)
        inner = np.arange(1, n)
        todo = inner[bad(edges[1:-1])] if n > 1 else inner
        for j in todo:
            half = 0.5 * (edges[j + 1] - edges[j - 1])
            cands = edges[j] + np.linspace(-0.45, 0.45, 19) * half
            gg = _Geometry(self, cands)
            dist = np.min(np.abs(k * k - _chi2_at(CM, gg.a, gg.b)), axis=1)
            ok = dist >= DELTA_C * k * k
            if np.any(ok):
                edges[j] = cands[ok][np.argmin(np.abs(cands[ok] - edges[j]))]
            elif np.max(dist) > 0:      # a mode stays within delta_c of cutoff over the whole slice: take the best
                edges[j] = cands[np.argmax(dist)]
            else:
                raise RuntimeError("could not move a slice edge away from a cutoff plane")
        return edges

    def _solve_class(self, f, modes, key, flags, Zs, tail, sf, adaptive, visible):
        """GSM blocks (S11, S12, S21, S22) of one class in its internal list, the number of slices, the slice
        factor used and the error estimate.

        Adaptive slicing: S(n) and S(2n) give the Richardson estimate |S(2n) - S(n)| / 15 of the error of S(2n)
        (4th-order method), over the entries of the port modes (`visible`, internal indices). While it exceeds
        `atol`, the density is raised by the predicted factor r = 1.2 (err / atol)^(1/4) (at least 2), with the
        estimate |S(rn) - S(n)| / (r^4 - 1)."""
        k = 2 * np.pi * f / cst.c
        CM = _Modes(modes)
        N = len(CM)
        zs = self._z_samples()
        g = _Geometry(self, zs)
        chi2s = _chi2_at(CM, g.a, g.b)
        loss_set = np.min(chi2s, axis=0) < k * k
        tmodes, trem = (self._tail_sets(modes, key, flags) if tail else (None, []))
        segs = self._segments()
        uni_parts = {}
        for i, (z0, z1, uni) in enumerate(segs):
            if uni:
                uni_parts[i] = self._uniform_part(f, CM, z0, z1, Zs, loss_set, tmodes, trem)

        def total(sfac):
            acc, nsl = None, 0
            for i, (z0, z1, uni) in enumerate(segs):
                if uni:
                    part, n = uni_parts[i]
                else:
                    part, n = self._sliced_part(f, CM, z0, z1, Zs, loss_set, tmodes, trem, sfac)
                nsl += n
                acc = part if acc is None else _star(acc, part)
            return acc, nsl

        S, nsl = total(sf)
        if not adaptive or all(u for *_, u in segs):
            return S, nsl, sf, np.nan
        vis = np.asarray(visible, dtype=int)

        def diff(A, B):
            return max(float(np.max(np.abs(x[np.ix_(vis, vis)] - y[np.ix_(vis, vis)]))) for x, y in zip(A, B))
        # second level: Richardson estimate of the error of S(2n) from |S(2n) - S(n)| / 15 (4th order)
        sf2 = 2 * sf
        S2, nsl = total(sf2)
        err = diff(S2, S) / 15.0
        S, sf = S2, sf2
        for _ in range(self.max_doublings):
            if err <= self.atol:
                break
            # jump to the predicted density (error ~ density^-4), with a 20 % margin
            r = min(4.0, max(2.0, 1.2 * (err / self.atol) ** 0.25))
            S2, nsl = total(sf * r)
            err = diff(S2, S) / (r ** 4 - 1.0)
            S, sf = S2, sf * r
        else:
            warnings.warn(f"{self.name}: slice tolerance {self.atol:g} not reached at {f / 1e9:.4g} GHz "
                          f"(error estimate {err:.1e} with {nsl} slices)", RuntimeWarning, stacklevel=3)
        return S, nsl, sf, err

    def _uniform_part(self, f, CM, z0, z1, Zs, loss_set, tmodes, trem):
        """GSM blocks of a uniform interval: one slice of growth <= GROWTH_MAX, doubled (exact)."""
        k = 2 * np.pi * f / cst.c
        gm = _Geometry(self, [0.5 * (z0 + z1)])
        Mz = vi_system(CM, f, gm, Zs, loss_set, tmodes, trem)[0]
        c2 = _chi2_at(CM, gm.a, gm.b)[0]
        gr = float(np.max(np.sqrt(np.maximum(c2 - k * k, 0.0)))) if len(CM) else 0.0
        L = z1 - z0
        lev = 0 if gr * L <= GROWTH_MAX else int(np.ceil(np.log2(gr * L / GROWTH_MAX)))
        h = L / 2 ** lev
        d = np.sqrt(_admittance(CM, k, c2[None, :]))
        part = tuple(x[0] for x in _slice_gsms(expm(h * Mz)[None], d, d))
        for _ in range(lev):
            part = _star(part, part)
        return part, 2 ** lev

    def _sliced_part(self, f, CM, z0, z1, Zs, loss_set, tmodes, trem, sf):
        """GSM blocks of a non-uniform segment (4th-order Magnus slices, direct slice GSMs, tree star product)."""
        k = 2 * np.pi * f / cst.c
        N = len(CM)
        edges = self._edges(z0, z1, CM, k, sf)
        part = None
        c = _chunk(N)
        for i0 in range(0, len(edges) - 1, c):
            e = edges[i0:i0 + c + 1]
            hh = np.diff(e)
            zg = np.concatenate([e[:-1] + _GAUSS[0] * hh, e[:-1] + _GAUSS[1] * hh])
            Mg = vi_system(CM, f, _Geometry(self, zg), Zs, loss_set, tmodes, trem)
            ns = len(hh)
            M1, M2 = Mg[:ns], Mg[ns:]
            Om = (0.5 * hh)[:, None, None] * (M1 + M2) + (_MAGNUS_C * hh * hh)[:, None, None] * (M2 @ M1 - M1 @ M2)
            Phi = expm(Om)
            ge = _Geometry(self, e)
            d = np.sqrt(_admittance(CM, k, _chi2_at(CM, ge.a, ge.b)))
            sl = _star_reduce(_slice_gsms(Phi, d[:-1], d[1:]))
            part = sl if part is None else _star(part, sl)
        return part, len(edges) - 1

    def gsm(self, f, basis1=None, basis2=None, mode_factor=None, tail=None, slice_factor=None):
        """GSM at f (Hz) between the port bases (default `default_basis(f)`), local frame `plane`.

        mode_factor, tail, slice_factor override the component's settings for this call (convergence studies).
        The GSM carries `info`: dict with 'classes' (number), 'n_modes' (internal modes per class), 'n_slices',
        'mode_factor', 'tail'.
        """
        mf = self.mode_factor if mode_factor is None else float(mode_factor)
        tl = self.tail if tail is None else bool(tail)
        adaptive = slice_factor is None and self.atol is not None
        sf = self.slice_factor if slice_factor is None else float(slice_factor)
        d1, d2 = self.default_basis(f)
        basis1 = d1 if basis1 is None else basis1
        basis2 = d2 if basis2 is None else basis2
        k = 2 * np.pi * f / cst.c
        ge = _Geometry(self, [0.0, self.length])
        for bb, i, port in ((basis1, 0, 1), (basis2, 1, 2)):
            if bb.frame not in (None, self.plane):
                raise ValueError(f"{self.name}: basis{port} has frame {bb.frame!r}, expected {self.plane!r}")
            if not (np.isclose(bb.a, ge.a[i], rtol=1e-12, atol=0) and np.isclose(bb.b, ge.b[i], rtol=1e-12, atol=0)):
                raise ValueError(f"{self.name}: basis{port} is for a {bb.a:g} x {bb.b:g} guide, but port {port} is "
                                 f"{ge.a[i]:g} x {ge.b[i]:g}")
            if len(bb):
                c2 = _chi2_at(_Modes(bb.modes), ge.a[i:i + 1], ge.b[i:i + 1])[0]
                bad = np.abs(k * k - c2) < DELTA_C * k * k
                if np.any(bad):
                    raise ValueError(f"{self.name}: port {port} is at the cutoff of "
                                     f"{', '.join(m.name for m, x in zip(bb.modes, bad) if x)}: its amplitude is "
                                     f"undefined there (CONVENTIONS.md section 3)")
        Zs = None
        g_in = self.guide
        if g_in is not None and g_in.sigma is not None:
            Zs = g_in.wall_impedance(f)
        key, flags = self._class_key(Zs is not None)
        lists = self._internal_lists(f, set(basis1.modes) | set(basis2.modes), key, flags, mf)
        N1, N2 = len(basis1), len(basis2)
        S11 = np.zeros((N1, N1), complex)
        S12 = np.zeros((N1, N2), complex)
        S21 = np.zeros((N2, N1), complex)
        S22 = np.zeros((N2, N2), complex)
        info = dict(classes=len(lists), n_modes={}, n_slices={}, mode_factor=mf, tail=tl, slice_factor={},
                    error_estimate={})
        with _blas_limit(1):
            for kk, modes in lists.items():
                for i, port in ((0, 1), (1, 2)):
                    c2 = _chi2_at(_Modes(modes), ge.a[i:i + 1], ge.b[i:i + 1])[0]
                    if np.any(np.abs(k * k - c2) < DELTA_C * k * k):
                        raise ValueError(f"{self.name}: an internal mode is at its cutoff at port {port}; change "
                                         f"the frequency slightly")
                vis = [i for i, md in enumerate(modes) if md in basis1 or md in basis2]
                blocks, nsl, sfu, err = self._solve_class(f, modes, key, flags, Zs, tl, sf, adaptive, vis)
                info["n_modes"][kk] = len(modes)
                info["n_slices"][kk] = nsl
                info["slice_factor"][kk] = sfu
                info["error_estimate"][kk] = err
                pos = {md: i for i, md in enumerate(modes)}
                i1 = [i for i, md in enumerate(basis1.modes) if md in pos]
                j1 = [pos[basis1.modes[i]] for i in i1]
                i2 = [i for i, md in enumerate(basis2.modes) if md in pos]
                j2 = [pos[basis2.modes[i]] for i in i2]
                A11, A12, A21, A22 = blocks
                S11[np.ix_(i1, i1)] = A11[np.ix_(j1, j1)]
                S12[np.ix_(i1, i2)] = A12[np.ix_(j1, j2)]
                S21[np.ix_(i2, i1)] = A21[np.ix_(j2, j1)]
                S22[np.ix_(i2, i2)] = A22[np.ix_(j2, j2)]
        out = GSM(S11, S12, S21, S22, basis1, basis2, f, self.name)
        out.info = info
        return out

    def convergence_study(self, f, input_mode=None, mode_factors=None, slice_factors=(2.0,),
                          tails=(True, False), basis1=None, basis2=None):
        """See :func:`convergence_study`."""
        return convergence_study(self, f, input_mode, mode_factors, slice_factors, tails, basis1, basis2)


def convergence_study(component, f, input_mode=None, mode_factors=None, slice_factors=(2.0,),
                      tails=(True, False), basis1=None, basis2=None):
    """Convergence of a varying-cross-section component at f (Hz): mode count, B5' on/off, slices.

    For each tail setting in `tails` and each mode factor f_c in `mode_factors` (default (f_c0, 2 f_c0) with the
    component's own f_c0), the propagating block S_p of the GSM (ports 1 then 2, default bases) is computed with the
    adaptive slicing, with the Richardson extrapolation in 1/M of T2 section 8.4, S_inf = (f2 S(f2) - f1 S(f1)) /
    (f2 - f1) (M is proportional to f_c), from the last two mode factors. For the first tail setting and mode factor,
    the slice refinement is reported too: fixed slicings at each of `slice_factors` times the density that the
    adaptive slicing ended with (largest over the classes).

    Returns a dict with 'modes1', 'modes2' (propagating port modes), 'rows': list of dicts (tail, mode_factor,
    slice_factor (None: adaptive), n_modes, n_slices, error_estimate, S), 'richardson': {tail: S_inf}, and, if
    input_mode is given, 'column': the same rows reduced to the column of that mode (S11 of the input mode and
    S21 per output mode).
    """
    comp = component
    if mode_factors is None:
        mode_factors = (comp.mode_factor, 2.0 * comp.mode_factor)
    if basis1 is None or basis2 is None:
        b1, b2 = comp.default_basis(f)
        basis1 = b1 if basis1 is None else basis1
        basis2 = b2 if basis2 is None else basis2
    p1, p2 = basis1.propagating_mask(f), basis2.propagating_mask(f)
    keep = np.concatenate([p1, p2])
    rows = []

    def run(t, mf, sf):
        g = comp.gsm(f, basis1, basis2, mode_factor=mf, tail=t, slice_factor=sf)
        rows.append(dict(tail=t, mode_factor=mf, slice_factor=sf, n_modes=g.info["n_modes"],
                         n_slices=g.info["n_slices"], error_estimate=g.info["error_estimate"],
                         final_slice_factor=g.info["slice_factor"], S=g.S[np.ix_(keep, keep)]))

    for t in tails:
        for mf in mode_factors:
            run(t, mf, None)
    base = max(rows[0]["final_slice_factor"].values()) if rows[0]["final_slice_factor"] else 1.0
    for sf in slice_factors:
        run(tails[0], mode_factors[0], sf * base)
    rich = {}
    for t in tails:
        rr = [r for r in rows if r["tail"] == t and r["slice_factor"] is None]
        if len(rr) >= 2:
            (f1, S1), (f2, S2) = (rr[-2]["mode_factor"], rr[-2]["S"]), (rr[-1]["mode_factor"], rr[-1]["S"])
            rich[t] = (f2 * S2 - f1 * S1) / (f2 - f1)
    out = dict(modes1=[m.name for m, p in zip(basis1.modes, p1) if p],
               modes2=[m.name for m, p in zip(basis2.modes, p2) if p], rows=rows, richardson=rich)
    if input_mode is not None:
        mode = Mode.parse(input_mode)
        j = out["modes1"].index(mode.name)
        n1 = len(out["modes1"])
        out["column"] = [dict(tail=r["tail"], mode_factor=r["mode_factor"], slice_factor=r["slice_factor"],
                              S11=r["S"][j, j], S21={nm: r["S"][n1 + i, j] for i, nm in enumerate(out["modes2"])})
                         for r in rows]
    return out


# ============================================================================================ Taper
class _Ramp:
    """x(z) = x0 + (x1 - x0) p(s(z)), s = (z - z0) / L on [z0, z0 + L] (reversed: s = (z0 + L - z) / L,
    x(z) = x1 + (x0 - x1) p(s)), constant outside; with the analytic derivative."""

    def __init__(self, x0, x1, z0, L, profile, reverse=False):
        self.x0, self.x1, self.z0, self.L, self.p, self.reverse = float(x0), float(x1), float(z0), float(L), \
            profile, reverse

    def _s(self, z):
        s = (np.asarray(z, dtype=float) - self.z0) / self.L
        s = np.clip(s, 0.0, 1.0)
        return 1.0 - s if self.reverse else s

    def __call__(self, z):
        s = self._s(z)
        if self.reverse:
            return self.x1 + (self.x0 - self.x1) * self.p(s)
        return self.x0 + (self.x1 - self.x0) * self.p(s)

    def derivative(self, z):
        z = np.asarray(z, dtype=float)
        s = self._s(z)
        inside = (z >= self.z0) & (z <= self.z0 + self.L)
        d = (self.x1 - self.x0) * self.p.derivative(s) / self.L      # the same for reverse (two sign changes)
        return np.where(inside, d, 0.0)


class _Sum:
    """Sum of geometry functions (with derivatives)."""

    def __init__(self, *parts, const=0.0):
        self.parts, self.const = parts, float(const)

    def __call__(self, z):
        return self.const + sum(p(z) for p in self.parts)

    def derivative(self, z):
        return sum(p.derivative(z) for p in self.parts) + 0.0 * np.asarray(z, dtype=float)


class _Piecewise:
    """Piecewise-constant function with given values on intervals (curvature)."""

    def __init__(self, edges, values):
        self.edges, self.values = np.asarray(edges, float), np.asarray(values, float)

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        i = np.clip(np.searchsorted(self.edges, z, side="right") - 1, 0, len(self.values) - 1)
        return self.values[i]

    def derivative(self, z):
        return np.zeros(np.shape(z))


class Taper(VariableGuide):
    """Taper from the current guide to the local sides (a2, b2), both with the same profile (card WP5).

    a(z) = a1 + (a2 - a1) p(z/L), b(z) = b1 + (b2 - b1) p(z/L), and the guide axis offset by
    u0(z) = du p(z/L), v0(z) = dv p(z/L) from the straight reference line (end offsets du, dv in m; the centre-line
    length then differs from L at second order in the slope, and the GSM length is along the reference line,
    T2b section 10). a1, b1 are the local sides of `guide` in the frame `plane` (default 'H': a along the global
    width w, b along the height h). In a `Line`, a Taper without a guide starts in the line's current guide.

    Parameters
    ----------
    length : float
        Length L (m).
    a2, b2 : float
        Local sides at port 2 (m).
    profile : 'linear', 'raised_cosine', 'bottle', Profile or callable p(t)
        `wrmodes.profiles`.
    guide : Guide, optional
        Global guide at port 1 (and its wall conductivity).
    du, dv : float
        Offsets (m) of the output centre (u0(L), v0(L)).
    plane : 'H' or 'E'
    Other keyword arguments as `VariableGuide` (mode_factor, tail, slice_factor, families, name).
    """

    def __init__(self, length, a2, b2, profile="linear", guide=None, du=0.0, dv=0.0, plane="H", **kw):
        self.a2, self.b2, self.du, self.dv = float(a2), float(b2), float(du), float(dv)
        self.profile = get_profile(profile)
        self.taper_plane = _check_plane(plane)
        self._taper_kw = dict(kw)
        if guide is None:
            # not yet attached: a placeholder geometry with the output sides at both ends (Line calls with_guide)
            a1, b1 = self.a2, self.b2
        else:
            a1, b1 = guide.dims(plane)
        self.a1, self.b1 = a1, b1
        L = float(length)
        p = self.profile
        a = _Ramp(a1, self.a2, 0.0, L, p)
        b = _Ramp(b1, self.b2, 0.0, L, p)
        u0 = _Ramp(0.0, self.du, 0.0, L, p) if self.du else None
        v0 = _Ramp(0.0, self.dv, 0.0, L, p) if self.dv else None
        kw.setdefault("name", f"taper {L:g} m to {self.a2:g} x {self.b2:g} ({p.name})")
        super().__init__(L, a, b, u0=u0, v0=v0, plane=plane, guide=guide, **kw)
        self._attached = guide is not None

    @property
    def guides(self):
        if not self._attached:
            return None
        return super().guides

    def default_basis(self, f, cutoff_factor=None):
        if not self._attached:
            raise ValueError("Taper needs a guide (Taper(..., guide=Guide(w, h))) or a Line")
        return super().default_basis(f, cutoff_factor)

    def gsm(self, f, basis1=None, basis2=None, **kw):
        if not self._attached:
            raise ValueError("Taper needs a guide (Taper(..., guide=Guide(w, h))) or a Line")
        return super().gsm(f, basis1, basis2, **kw)

    def replace(self, **changes):
        kw = dict(self._taper_kw)
        kw.update(mode_factor=self.mode_factor, tail=self.tail, slice_factor=self.slice_factor,
                  families=self.families, atol=self.atol, max_doublings=self.max_doublings, name=self.name)
        kw.update(changes)
        length = kw.pop("length", self.length)
        a2, b2 = kw.pop("a2", self.a2), kw.pop("b2", self.b2)
        profile = kw.pop("profile", self.profile)
        guide = kw.pop("guide", self.guide)
        du, dv = kw.pop("du", self.du), kw.pop("dv", self.dv)
        plane = kw.pop("plane", self.taper_plane)
        return Taper(length, a2, b2, profile, guide, du, dv, plane, **kw)

    def with_guide(self, guide):
        return self.replace(guide=guide)

    def __repr__(self):
        return (f"Taper({self.length:g} m, {self.a1:g} x {self.b1:g} -> {self.a2:g} x {self.b2:g}, "
                f"{self.profile.name}, plane={self.plane!r})")


# ============================================================================================ StrangledBend
class StrangledBend(VariableGuide):
    """The strangled E-plane bend of Butterweck & de Ronde (1968, p. 98, fig. 15): a full-height guide tapered down
    to a flat guide of height `b_flat`, bent in the E plane, and tapered back up.

    Built as one `VariableGuide` in the E-plane frame (u = y, a = h, b = w; CONVENTIONS.md section 2): local side
    a(z) goes from the guide's height h to `b_flat` over `taper_length` with `profile` (a symmetric taper: the guide
    axis stays on the centre line), then a straight flat section of flat_length / 2, a circular arc of centre-line
    radius `radius` and angle `angle_deg` in the flat guide (kappa = 1/radius > 0 turns towards -u = -y; a negative
    angle turns the other way), flat_length / 2 more, and the mirror taper back to h. The width w is constant.

    Port planes: the two full-height ends of the tapers; the local frame of port 2 is that of port 1 transported
    along the centre line (CONVENTIONS.md section 2). `length` = `centre_line_length` = 2 taper_length +
    flat_length + radius |angle| is the total centre-line length (the arm length a line needs for its geometry).

    The geometry conserves the local family n (= global m): one class per local family, the m-index coupled by the
    taper and the curvature. With a lossy guide, the T1 loss of the local cross-section is included (first order).

    Parameters
    ----------
    angle_deg : float
        Bend angle (degrees) in the E plane.
    radius : float
        Centre-line radius (m) of the arc in the flat guide.
    b_flat : float
        Height (m) of the flat guide (the E-plane side, local a).
    taper_length : float
        Length (m) of each taper.
    profile : 'raised_cosine' (default), 'linear', Profile or callable
    flat_length : float
        Total length (m) of straight flat guide, split equally before and after the arc.
    guide : Guide, optional
        Global guide (w, h, sigma) at both ports; attached by a `Line` if None.
    Other keyword arguments as `VariableGuide` (mode_factor, tail, slice_factor, name).
    """

    def __init__(self, angle_deg, radius, b_flat, taper_length, profile="raised_cosine", flat_length=0.0,
                 guide=None, **kw):
        self.angle_deg, self.radius, self.b_flat = float(angle_deg), float(radius), float(b_flat)
        self.taper_length, self.flat_length = float(taper_length), float(flat_length)
        if not (self.radius > 0 and self.b_flat > 0 and self.taper_length > 0 and self.flat_length >= 0):
            raise ValueError("radius, b_flat and taper_length must be positive, flat_length >= 0")
        self.profile = get_profile(profile)
        self._sb_kw = dict(kw)
        self._attached = guide is not None
        if guide is None:
            h, w = 2 * self.b_flat, 1.0               # placeholder until attached
        else:
            h, w = guide.h, guide.w
        if self._attached and not self.b_flat < h:
            raise ValueError("b_flat must be smaller than the guide height")
        Lt, fl = self.taper_length, self.flat_length
        arc = self.radius * abs(np.radians(self.angle_deg))
        z1, z2 = Lt, Lt + fl / 2
        z3 = z2 + arc
        z4 = z3 + fl / 2
        L = z4 + Lt
        p = self.profile
        a_in = _Ramp(h, self.b_flat, 0.0, Lt, p)
        a_out = _Ramp(self.b_flat, h, z4, Lt, p, reverse=True)
        a = _Sum(a_in, a_out, const=-self.b_flat)
        sgn = 1.0 if self.angle_deg >= 0 else -1.0
        kappa = _Piecewise([0.0, z2, z3], [0.0, sgn / self.radius, 0.0]) if arc > 0 else None
        bps = [z1, z2, z3, z4]
        uniform = [(x0, x1) for x0, x1 in ((z1, z2), (z2, z3), (z3, z4)) if x1 > x0]
        kw.setdefault("name", f"strangled E-plane bend {self.angle_deg:g} deg, R = {self.radius:g} m, "
                              f"b1 = {self.b_flat:g} m")
        super().__init__(L, a, w, kappa=kappa, plane="E", guide=guide, breakpoints=bps, uniform=uniform, **kw)
        self.z_marks = (0.0, z1, z2, z3, z4, L)

    @property
    def guides(self):
        return super().guides if self._attached else None

    def default_basis(self, f, cutoff_factor=None):
        if not self._attached:
            raise ValueError("StrangledBend needs a guide or a Line")
        return super().default_basis(f, cutoff_factor)

    def gsm(self, f, basis1=None, basis2=None, **kw):
        if not self._attached:
            raise ValueError("StrangledBend needs a guide or a Line")
        return super().gsm(f, basis1, basis2, **kw)

    def replace(self, **changes):
        kw = dict(self._sb_kw)
        kw.update(mode_factor=self.mode_factor, tail=self.tail, slice_factor=self.slice_factor,
                  families=self.families, atol=self.atol, max_doublings=self.max_doublings, name=self.name)
        kw.update(changes)
        args = {k: kw.pop(k, getattr(self, k)) for k in ("angle_deg", "radius", "b_flat", "taper_length",
                                                          "profile", "flat_length", "guide")}
        return StrangledBend(args["angle_deg"], args["radius"], args["b_flat"], args["taper_length"],
                             args["profile"], args["flat_length"], args["guide"], **kw)

    def with_guide(self, guide):
        return self.replace(guide=guide)

    @property
    def centre_line_length(self):
        return self.length

    def __repr__(self):
        return (f"StrangledBend({self.angle_deg:g} deg, R={self.radius:g}, b_flat={self.b_flat:g}, "
                f"taper={self.taper_length:g} ({self.profile.name}), flat={self.flat_length:g})")
