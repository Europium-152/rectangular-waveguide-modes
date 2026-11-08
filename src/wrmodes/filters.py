"""Butterweck's mode filters as cascadable components, by the exact methods of T1 (impedance walls, resistive sheets).

References: H.-J. Butterweck, "Mode filters for oversized rectangular waveguides", IEEE Trans. MTT-16, 274-281 (1968);
H. J. Butterweck and F. C. de Ronde, Philips Tech. Rev. 29, 86-101 (1968); docs/theory/T1-wall-impedance.md
(boxes B6-B10, sections 4, 7 and 8). Conventions: CONVENTIONS.md (e^{+i omega t}, forward modes e^{-i gamma z} with
Im gamma <= 0, amplitudes normalized with the unconjugated reaction).

Both filters are a section of a loaded guide between two junctions to the empty (perfectly conducting) guide:

    GSM = J  *  diag(exp(-i gamma_i L))  *  J_reversed            (Redheffer star products)

where the loaded-guide modes i are exact solutions of Maxwell's equations with the wall or sheet conditions, and
the junction J follows from E continuity tested with the empty-guide modes and H continuity tested with the loaded
modes (T1 box B9, the reciprocal form; `wrmodes.modematch.junction_gsm`, which accepts complex overlaps):

    a + b = P (c + d),   c - d = P^T (a - b),   P_pi = sqrt(Y_p / 2) int E_i . e_p ds,

with the loaded modes normalized by (1/2) int (E_i x H_i) . z ds = 1 (unconjugated). S is symmetric by
construction. The first-order loss of `wrmodes.loss` is not used: for Butterweck's absorbing walls (|Z| ~ eta0)
it fails completely for every mode with n >= 1 (T1 section 7).

The loaded-guide modes are longitudinal-section modes:

* `LossyWallFilter` (impedance walls at u = +-a/2, T1 box B6): LSE^u (E_u = 0, contains TE_m0) and LSM^u
  (H_u = 0, contains TE_0n) of each family n, transverse wavenumber k_u, gamma^2 = k^2 - (n pi/b)^2 - k_u^2;
* `ResistiveSheetFilter` (sheets parallel to the walls v = +-b/2, T1 boxes B7, B8): LSM^v (H_v = 0, Butterweck's
  EH^(2), contains TE_m0) and LSE^v (E_v = 0, EH^(1), contains TE_0n) of each index m, transverse wavenumber k_v,
  gamma^2 = k^2 - (m pi/a)^2 - k_v^2.

Their transverse wavenumbers s = k_t^2 are the eigenvalues of -X'' = s X with the wall (Robin) or sheet (jump)
conditions, found by Chebyshev collocation (all roots at once, robust for any impedance; continuation in Z from 0
jumps branches for high orders, T1 section 4.2) and polished by Newton's method on the exact dispersion function
(a transfer of (X, X') across the cross-section, entire in s).
"""
import numpy as np
import scipy.constants as cst
from numpy.polynomial.legendre import leggauss

from .gsm import GSM
from .modes import ETA0, Mode, ModeBasis, transverse_field, eplane_frame_signs
from .modematch.junction import junction_gsm
from .components.base import Component
from .components.guide import Guide, map_basis, _check_plane

__all__ = ["LossyWallFilter", "ResistiveSheetFilter", "impedance_wall_modes", "sheet_modes"]

DB = 20 / np.log(10)


# ============================================================================== 1D eigenproblems
def _cheb(N, x0, x1):
    """Chebyshev points on [x0, x1] (increasing) and the first-derivative matrix (Trefethen, cheb.m)."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    dX = x[:, None] - x[None, :]
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return x0 + (x1 - x0) * (1 - x) / 2, -2.0 / (x1 - x0) * D


def _collocation_eigenvalues(breaks, rows, npts):
    """Eigenvalues s of -y'' = s y on [breaks[0], breaks[-1]], piecewise on the sub-intervals between breaks.

    `rows`: 2 * (number of sub-intervals) homogeneous linear conditions, each a list of terms
    (interval, end, coefficient of y, coefficient of y') with end 0 (left) or 1 (right): the wall conditions and the
    interface conditions at the breaks. Each sub-interval i has npts[i] + 1 Chebyshev points; the conditions
    eliminate the end values (as t1lib.lsu_roots_all for one interval).
    """
    nd = len(breaks) - 1
    pts, D1s, off = [], [], [0]
    for i in range(nd):
        x, D1 = _cheb(npts[i], breaks[i], breaks[i + 1])
        pts.append(x)
        D1s.append(D1)
        off.append(off[-1] + npts[i] + 1)
    tot = off[-1]
    Bm = np.zeros((2 * nd, tot), dtype=complex)
    for r, terms in enumerate(rows):
        for i, end, cv, cd in terms:
            loc = 0 if end == 0 else npts[i]
            Bm[r, off[i]:off[i + 1]] += cd * D1s[i][loc]
            Bm[r, off[i] + loc] += cv
    D2 = np.zeros((tot, tot))
    for i in range(nd):
        D2[off[i]:off[i + 1], off[i]:off[i + 1]] = D1s[i] @ D1s[i]
    ends = np.array([off[i] + e for i in range(nd) for e in (0, npts[i])])
    inner = np.setdiff1d(np.arange(tot), ends)
    E = np.linalg.solve(Bm[:, ends], -Bm[:, inner])
    A = -(D2[np.ix_(inner, inner)] + D2[np.ix_(inner, ends)] @ E)
    return np.linalg.eigvals(A)


def _polish(res, s0, scale, maxit=60):
    """Vectorized Newton iteration on res(s) (central differences); returns (s, converged)."""
    s = np.array(s0, dtype=complex)
    ok = np.zeros(s.shape, bool)
    for _ in range(maxit):
        h = 1e-7 * (np.abs(s) + scale)
        r = res(s)
        dr = (res(s + h) - res(s - h)) / (2 * h)
        step = np.where(ok, 0.0, r / np.where(dr == 0, 1.0, dr))
        s = s - step
        ok |= np.abs(step) <= 1e-13 * (np.abs(s) + scale)
        if ok.all():
            break
    return s, ok


def _roots(res, eigs, count, scale):
    """The `count` roots of res with the smallest Re s, from the collocation eigenvalues `eigs` as starting values."""
    eigs = eigs[np.isfinite(eigs)]
    cand = eigs[np.argsort(eigs.real)][: count + max(8, count // 4)]
    s, ok = _polish(res, cand, scale)
    s = s[ok]
    s = s[np.argsort(s.real)]
    keep = []
    for x in s:
        if all(abs(x - y) > 1e-8 * (abs(x) + scale) for y in keep):
            keep.append(x)
    if len(keep) < count:
        raise RuntimeError(f"collocation found {len(keep)} of the {count} roots needed; increase the resolution")
    return np.array(keep[:count])


def _advance(Y, dY, s, d):
    """(Y, Y') after a distance d of -Y'' = s Y; entire in s (no branch of sqrt(s) enters)."""
    q = np.sqrt(s + 0j)
    c = np.cos(q * d)
    sn = d * np.sinc(q * d / np.pi)                  # sin(q d) / q, also at q = 0
    return c * Y + sn * dY, -s * sn * Y + c * dY


def _gamma(g2):
    """Propagation constant with Im gamma <= 0 (CONVENTIONS.md section 3); numerically real roots get Re >= 0."""
    g = np.sqrt(np.asarray(g2, dtype=complex))
    real = np.abs(g.imag) <= 1e-12 * np.abs(g)
    return np.where(real, np.where(g.real < 0, -g, g), np.where(g.imag > 0, -g, g))


def _wavenumber(f):
    return 2 * np.pi * f / cst.c


# ============================================================================== impedance walls (B6)
def _lsu_start(kind, k, Zm):
    """(X(0), X'(0)) satisfying the Leontovich (Robin) condition at u' = 0 (T1 box B6)."""
    if kind == "LSE":                       # Z- X'(0) = i k eta0 X(0)
        return complex(Zm), 1j * k * ETA0
    return complex(ETA0), 1j * k * Zm       # LSM: eta0 X'(0) = i k Z- X(0)


def _lsu_residual(kind, k, a, Zp, Zm):
    X0, dX0 = _lsu_start(kind, k, Zm)

    def res(s):
        X, dX = _advance(X0, dX0, s, a)
        return Zp * dX + 1j * k * ETA0 * X if kind == "LSE" else ETA0 * dX + 1j * k * Zp * X
    return res


def impedance_wall_modes(f, a, Zp, Zm, kind, count, resolution=None):
    """Transverse wavenumbers s = k_u^2 of the LS^u modes of a guide with impedance walls at u = +-a/2.

    T1 box B6: the profile X(u') (u' = u + a/2) solves -X'' = k_u^2 X with the Leontovich conditions
        LSE^u (E_u = 0):  Z- X'(0) = i k eta0 X(0),   Z+ X'(a) = -i k eta0 X(a);
        LSM^u (H_u = 0):  eta0 X'(0) = i k Z- X(0),   eta0 X'(a) = -i k Z+ X(a),
    Z+ (Zp) on the wall u = +a/2, Z- (Zm) on u = -a/2 (ohm, any complex value, 0 = perfect conductor). The roots
    are the zeros of the dispersion function F of B6 (here as the residual of the right-hand condition after a
    transfer of (X, X') across the guide). Returns the `count` values of s with the smallest real part; for
    Z -> 0 they tend to (m pi / a)^2, m = 1, 2, ... (LSE) or m = 0, 1, ... (LSM). The propagation constant of
    family n is gamma = sqrt(k^2 - (n pi/b)^2 - s), Im gamma <= 0.
    """
    k = _wavenumber(f)
    npts = resolution or (2 * count + 48)
    if kind == "LSE":
        rows = [[(0, 0, -1j * k * ETA0, Zm)], [(0, 1, 1j * k * ETA0, Zp)]]
    elif kind == "LSM":
        rows = [[(0, 0, -1j * k * Zm, ETA0)], [(0, 1, 1j * k * Zp, ETA0)]]
    else:
        raise ValueError("kind must be 'LSE' or 'LSM'")
    eigs = _collocation_eigenvalues([0.0, a], rows, [npts])
    return _roots(_lsu_residual(kind, k, a, Zp, Zm), eigs, count, (np.pi / a) ** 2)


class _LoadedModes:
    """Normalized transverse fields of the LS modes of one family of a loaded guide (internal)."""

    def __len__(self):
        return len(self.gamma)

    def admittance(self, f):                  # junction_gsm uses sqrt(Y) = 1: the overlaps carry the admittances
        return np.ones(len(self))


def _lsu_family(f, a, b, n, Zp, Zm, M, quad):
    """LS^u modes of family n (LSE m = 1..M; LSM m = 0..M if n >= 1) with their normalized fields on the grid."""
    k = _wavenumber(f)
    kn = n * np.pi / b
    out = _LoadedModes()
    kinds, s = [], []
    for kind, cnt in (("LSE", M), ("LSM", M + 1 if n >= 1 else 0)):
        if cnt:
            r = impedance_wall_modes(f, a, Zp, Zm, kind, cnt)
            kinds += [kind] * cnt
            s.append(r)
    out.kind = np.array(kinds)
    out.s = np.concatenate(s)
    g2 = k * k - kn * kn - out.s
    if np.any(np.abs(k * k - out.s) < 1e-10 * k * k):
        raise ValueError(f"family n = {n}: a loaded mode has k_u = k (its LS field is self-orthogonal); change the "
                         f"frequency slightly")
    out.gamma = _gamma(g2)
    # profiles X(u'), X'(u') on the quadrature points
    up, wu = quad["u"]
    X = np.empty((len(out.s), len(up)), complex)
    dX = np.empty_like(X)
    for kind in ("LSE", "LSM"):
        sel = out.kind == kind
        if np.any(sel):
            X0, dX0 = _lsu_start(kind, k, Zm)
            X[sel], dX[sel] = _advance(X0, dX0, out.s[sel][:, None], up[None, :])
    IX = (X * X) @ wu                          # unconjugated
    g, sk = out.gamma, k * k - out.s
    lse = out.kind == "LSE"
    # (1/2) int (E x H).z: LSE^u -gamma (k^2 - s)/(2 k eta0) I_X b/eps_n;  LSM^u -gamma (k^2 - s) eta0/(2 k) I_X b/2
    P = np.where(lse, -g * sk / (2 * k * ETA0) * (b / (1.0 if n == 0 else 2.0)), -g * sk * ETA0 / (2 * k) * (b / 2)) * IX
    c = 1 / np.sqrt(P)
    out.c, out.X, out.dX = c, X, dX
    out.kn, out.k = kn, k
    return out


def _lsu_fields(md, vp):
    """(E_u, E_v, H_u, H_v) of the normalized LS^u modes on the grid (u' quadrature points) x vp, unit amplitude."""
    k, kn, c, g = md.k, md.kn, md.c, md.gamma
    cv, sv = np.cos(kn * vp), np.sin(kn * vp)
    lse = (md.kind == "LSE")[:, None, None]
    X, dX = md.X[:, :, None], md.dX[:, :, None]
    sk = (k * k - md.s)[:, None, None]
    cc, gg = c[:, None, None], g[:, None, None]
    # LSE^u: E = (0, i gamma X cos), H = ((k^2 - s) X cos, -kn X' sin) / (i k eta0)
    # LSM^u: E = ((k^2 - s) X sin, kn X' cos) eta0 / (i k), H = (0, -i gamma X sin)
    Eu = np.where(lse, 0.0, cc * sk * X * sv * ETA0 / (1j * k))
    Ev = np.where(lse, cc * 1j * gg * X * cv, cc * kn * dX * cv * ETA0 / (1j * k))
    Hu = np.where(lse, cc * sk * X * cv / (1j * k * ETA0), 0.0)
    Hv = np.where(lse, -cc * kn * dX * sv / (1j * k * ETA0), -cc * 1j * gg * X * sv)
    return Eu, Ev, Hu, Hv


# ============================================================================== resistive sheets (B7, B8)
def _sheet_conditions(kind, k, b, xis, pos):
    """Wall and interface conditions of the LS^v profile Y(v') with sheets at v' = pos (T1 box B8)."""
    nd = len(pos) + 1
    if kind == "LSM":       # Y'(0) = Y'(b) = 0; at a sheet Y' continuous, Y(+) = Y(-) - i Y'(-) / (k xi)
        rows = [[(0, 0, 0.0, 1.0)], [(nd - 1, 1, 0.0, 1.0)]]
        for j, xi in enumerate(xis):
            rows.append([(j, 1, 0.0, 1.0), (j + 1, 0, 0.0, -1.0)])
            rows.append([(j + 1, 0, 1.0, 0.0), (j, 1, -1.0, 1j / (k * xi))])
    elif kind == "LSE":     # Y(0) = Y(b) = 0; at a sheet Y continuous, Y'(+) = Y'(-) + i k Y(-) / xi
        rows = [[(0, 0, 1.0, 0.0)], [(nd - 1, 1, 1.0, 0.0)]]
        for j, xi in enumerate(xis):
            rows.append([(j + 1, 0, 1.0, 0.0), (j, 1, -1.0, 0.0)])
            rows.append([(j + 1, 0, 0.0, 1.0), (j, 1, -1j * k / xi, -1.0)])
    else:
        raise ValueError("kind must be 'LSM' or 'LSE'")
    return rows


def _sheet_states(kind, k, b, xis, pos, s):
    """(Y, Y') at the start of every sub-interval [0, pos_0], [pos_0, pos_1], ... and at v' = b."""
    s = np.asarray(s, dtype=complex)
    Y, dY = (np.ones_like(s), np.zeros_like(s)) if kind == "LSM" else (np.zeros_like(s), np.ones_like(s))
    starts, here = [(Y, dY)], 0.0
    for xi, p in zip(xis, pos):
        Y, dY = _advance(Y, dY, s, p - here)
        here = p
        if kind == "LSM":
            Y = Y - 1j * dY / (k * xi)
        else:
            dY = dY + 1j * k * Y / xi
        starts.append((Y, dY))
    return starts, _advance(Y, dY, s, b - here)


def _sheet_geometry(b, R_sheet, positions):
    pos = np.asarray(positions, dtype=float)
    R = np.broadcast_to(np.asarray(R_sheet, dtype=complex), pos.shape)
    order = np.argsort(pos)
    pos, R = pos[order], R[order]
    if np.any(pos <= 0) or np.any(pos >= b) or np.any(np.diff(pos) <= 0):
        raise ValueError("sheet positions must be distinct and strictly inside the guide")
    if np.any(R == 0) or not np.all(np.isfinite(R)):
        raise ValueError("sheet resistances must be finite and non-zero (ohm per square)")
    return pos, R / ETA0


def sheet_modes(f, b, R_sheet, positions, kind, count, resolution=None):
    """Transverse wavenumbers s = k_v^2 of the LS^v modes of a guide with resistive sheets parallel to v = +-b/2.

    T1 boxes B7 (one sheet, closed form) and B8 (several sheets, transfer form): Y(v') (v' = v + b/2) solves
    -Y'' = k_v^2 Y on each layer, with
        LSM^v (H_v = 0, Butterweck's EH^(2)):  Y'(0) = Y'(b) = 0;  Y' continuous, Y(+) = Y(-) - i Y'/(k xi);
        LSE^v (E_v = 0, Butterweck's EH^(1)):  Y(0) = Y(b) = 0;    Y continuous,  Y'(+) = Y'(-) + i k Y/xi,
    xi = R_sheet / eta0 at each sheet. `positions` are the sheet positions v' in (0, b) (measured from the wall
    v = -b/2); `R_sheet` (ohm per square) is one value or one per sheet. Returns the `count` values of s with the
    smallest real part; LSM^v always has s = 0 (TE_m0, untouched by the sheets). gamma = sqrt(k^2 - (m pi/a)^2 - s).
    """
    k = _wavenumber(f)
    pos, xis = _sheet_geometry(b, R_sheet, positions)
    breaks = np.concatenate([[0.0], pos, [b]])
    npts = resolution or [int(2 * count * (x1 - x0) / b) + 32 for x0, x1 in zip(breaks[:-1], breaks[1:])]
    eigs = _collocation_eigenvalues(breaks, _sheet_conditions(kind, k, b, xis, pos), npts)

    def res(s):
        _, (Y, dY) = _sheet_states(kind, k, b, xis, pos, s)
        return dY if kind == "LSM" else Y
    return _roots(res, eigs, count, (np.pi / b) ** 2)


def _sheet_family(f, a, b, m, xis, pos, Nn, vq):
    """LS^v modes of index m (LSM n = 0..Nn if m >= 1; LSE n = 1..Nn) with normalized fields on the v-grid."""
    k = _wavenumber(f)
    km = m * np.pi / a
    out = _LoadedModes()
    kinds, s = [], []
    for kind, cnt in (("LSM", Nn + 1 if m >= 1 else 0), ("LSE", Nn)):
        if cnt:
            r = sheet_modes(f, b, xis * ETA0, pos, kind, cnt)
            if kind == "LSM":
                r[np.argmin(np.abs(r))] = 0.0      # the exact root s = 0 (TE_m0)
            kinds += [kind] * cnt
            s.append(r)
    out.kind = np.array(kinds)
    out.s = np.concatenate(s)
    if np.any(np.abs(k * k - out.s) < 1e-10 * k * k):
        raise ValueError(f"index m = {m}: a sheet mode has k_v = k (its LS field is self-orthogonal; for the empty "
                         f"guide this is k_t = 0 of CONVENTIONS.md section 7); change the frequency or b slightly")
    out.gamma = _gamma(k * k - km * km - out.s)
    vp, wv, seg = vq
    breaks = np.concatenate([[0.0], pos, [b]])
    Y = np.empty((len(out.s), len(vp)), complex)
    dY = np.empty_like(Y)
    for kind in ("LSM", "LSE"):
        sel = np.flatnonzero(out.kind == kind)
        if len(sel):
            starts, _ = _sheet_states(kind, k, b, xis, pos, out.s[sel])
            for d, (Y0, dY0) in enumerate(starts):
                pts = seg == d
                yy, dd = _advance(Y0[:, None], dY0[:, None], out.s[sel][:, None], (vp[pts] - breaks[d])[None, :])
                Y[np.ix_(sel, np.flatnonzero(pts))] = yy
                dY[np.ix_(sel, np.flatnonzero(pts))] = dd
    IY = (Y * Y) @ wv
    g, sk = out.gamma, k * k - out.s
    lsm = out.kind == "LSM"
    eps_m = 1.0 if m == 0 else 2.0
    # (1/2) int (E x H).z: LSM^v -gamma (k^2 - s) eta0 / (2k) (a/2) I_Y;  LSE^v -gamma (k^2 - s)/(2 k eta0) (a/eps_m) I_Y
    P = np.where(lsm, -g * sk * ETA0 / (2 * k) * (a / 2), -g * sk / (2 * k * ETA0) * (a / eps_m)) * IY
    out.c, out.Y, out.dY = 1 / np.sqrt(P), Y, dY
    out.km, out.k = km, k
    return out


def _sheet_fields(md, up):
    """(E_u, E_v, H_u, H_v) of the normalized LS^v modes on the grid up x (v' quadrature points)."""
    k, km, c, g = md.k, md.km, md.c[:, None, None], md.gamma[:, None, None]
    cu, su = np.cos(km * up)[None, :, None], np.sin(km * up)[None, :, None]
    lsm = (md.kind == "LSM")[:, None, None]
    Y, dY = md.Y[:, None, :], md.dY[:, None, :]
    sk = (k * k - md.s)[:, None, None]
    # LSM^v: E = (km cos Y', (k^2 - s) sin Y) eta0/(i k), H = (i gamma sin Y, 0)
    # LSE^v: E = (-i gamma cos Y, 0), H = (-km sin Y', (k^2 - s) cos Y) / (i k eta0)
    Eu = np.where(lsm, c * km * cu * dY * ETA0 / (1j * k), -c * 1j * g * cu * Y)
    Ev = np.where(lsm, c * sk * su * Y * ETA0 / (1j * k), 0.0)
    Hu = np.where(lsm, c * 1j * g * su * Y, -c * km * su * dY / (1j * k * ETA0))
    Hv = np.where(lsm, 0.0, c * sk * cu * Y / (1j * k * ETA0))
    return Eu, Ev, Hu, Hv


# ============================================================================== filter sections
def _section(f, ideal, loaded, Eu, Ev, eu, ev, W, length, name):
    """GSM (ideal-guide modes at both ports) of junction * loaded section * reversed junction (T1 box B9)."""
    X = (np.einsum("iuv,puv->ip", Eu * W, eu) + np.einsum("iuv,puv->ip", Ev * W, ev)) / np.sqrt(2.0)
    J = junction_gsm(ideal, loaded, X, f, large_port=1, name=name)
    t = np.exp(-1j * loaded.gamma * length)
    Z = np.zeros((len(loaded),) * 2, complex)
    prop = GSM(Z, np.diag(t), np.diag(t), Z.copy(), loaded, loaded, f)
    Jr = GSM(J.S22, J.S21, J.S12, J.S11, loaded, ideal, f)
    return J @ prop @ Jr


def _ideal_fields(basis, U, V):
    eu = np.empty((len(basis),) + U.shape)
    ev = np.empty_like(eu)
    for i, md in enumerate(basis.modes):
        eu[i], ev[i] = transverse_field(md, basis.a, basis.b, U, V)
    return eu, ev


def _assemble(basis, f, blocks, name):
    """Full GSM on `basis` from per-group GSMs on internal bases (requested modes extracted)."""
    N = len(basis)
    S = [np.zeros((N, N), complex) for _ in range(4)]
    for g in blocks:
        internal = g.basis1
        ib = [i for i, md in enumerate(basis.modes) if md in internal]
        ii = [internal.index(basis.modes[i]) for i in ib]
        for blk, src in zip(S, (g.S11, g.S12, g.S21, g.S22)):
            blk[np.ix_(ib, ib)] = src[np.ix_(ii, ii)]
    return GSM(*S, basis, basis, f, name)


class _Filter(Component):
    def _init_common(self, length, plane, guide, name):
        length = float(length)
        if not length >= 0:
            raise ValueError("length must be >= 0")
        if guide is not None and not isinstance(guide, Guide):
            raise TypeError("guide must be a wrmodes.components.Guide")
        self.length = length
        self.plane = _check_plane(plane)
        self.guide = guide
        self.name = name

    def default_basis(self, f, cutoff_factor=None):
        """Port bases from the component's guide; cutoff_factor=None means 1 (propagating modes only)."""
        if self.guide is None:
            raise ValueError(f"{type(self).__name__}.default_basis needs a guide (guide=Guide(w, h))")
        b = self.guide.basis(f, 1.0 if cutoff_factor is None else cutoff_factor, plane=self.plane)
        return b, b

    def _port_basis(self, f, basis1, basis2):
        if basis1 is None:
            basis1 = self.default_basis(f)[0]
        if basis2 is not None and basis2 != basis1:
            raise ValueError(f"{self.name}: the filter has the same basis at both ports")
        if basis1.frame not in (None, self.plane):
            raise ValueError(f"{self.name}: basis has frame {basis1.frame!r}, expected {self.plane!r} "
                             f"(insert a FrameChange, or use a Line)")
        return basis1


class LossyWallFilter(_Filter):
    """Section of guide with impedance (absorbing) walls between two empty guides (Butterweck 1968, eqs. 20-31).

    Parameters
    ----------
    length : float
        Length of the lossy-wall section (m). The reference planes are its two ends (the junctions).
    Zw : complex or dict
        Wall impedance (ohm) of the walls in `walls` (one value for all, or {wall: Z}); Re Zw >= 0 for an
        absorbing wall. Butterweck's filter: 250 ohm (real) on both narrow walls.
    walls : tuple
        The impedance walls, in the local frame of `plane` (labels of `wrmodes.loss`): any of ('u-', 'u+') (the
        walls at u = -+a/2) or any of ('v-', 'v+'), not both pairs. The other walls are perfect conductors.
    plane : 'H' or 'E'
        Local frame of the component (CONVENTIONS.md section 2). The filter is frame dependent: with plane='H' and
        walls ('u-', 'u+') the absorbing walls are the narrow walls x = -+w/2 of the guide (Butterweck's filter,
        which attenuates TE_m0 as m^2 and leaves TE_0n untouched); with plane='E' they are the broad walls.
    modes_per_family : int, optional
        Internal truncation M: per family n, the empty guide keeps TE_mn and TM_mn with m <= M and the lossy guide
        the same number of LS modes. Default: max(80, 2 m_max + 20), m_max the largest m of the port basis. The
        result converges as M grows (TE_m0 to about 1e-3 dB between 80 and 160 modes, T1 section 7).
    guide : Guide, optional
        Cross-section; needed for `default_basis` only. A `Line` attaches its own guide. A conductivity of the
        guide is not applied to the filter's other walls (they are treated as perfect; the metal loss of a 15 cm
        section is of order 0.03 dB, well below the filter's own attenuation).

    Method (T1 box B6 + B9, exact Leontovich problem, not the first-order loss): the walls at u = +-a/2 do not vary
    along v, so the filter conserves the local family n. For each family, the exact LSE^u / LSM^u modes of the
    impedance-walled guide (`impedance_wall_modes`) are matched to the empty-guide modes at both ends. Walls on v
    are handled in the frame rotated by 90 degrees, so the filter then conserves m. Port modes outside the
    internal truncation are not possible (M grows with the basis); evanescent port modes left out are treated as
    matched (neighbouring components must be far enough away for them to decay).
    """

    def __init__(self, length, Zw, walls=("u-", "u+"), plane="H", modes_per_family=None, guide=None, name=""):
        self._init_common(length, plane, guide, name or f"lossy-wall filter {float(length):g} m")
        walls = tuple(walls)
        if not walls or not (set(walls) <= {"u-", "u+"} or set(walls) <= {"v-", "v+"}):
            raise ValueError("walls must be a non-empty subset of ('u-', 'u+') or of ('v-', 'v+')")
        if isinstance(Zw, dict):
            if set(Zw) - set(walls):
                raise ValueError("Zw has walls that are not in `walls`")
            Z = {w: complex(Zw.get(w, 0.0)) for w in walls}
        else:
            Z = {w: complex(Zw) for w in walls}
        self.walls, self.Zw = walls, Z
        self.modes_per_family = None if modes_per_family is None else int(modes_per_family)

    def __repr__(self):
        return f"LossyWallFilter({self.length!r}, {self.Zw!r}, plane={self.plane!r})"

    def with_guide(self, guide):
        return type(self)(self.length, dict(self.Zw), self.walls, self.plane, self.modes_per_family, guide,
                          self.name)

    def _u_walls(self):
        """(Z+, Z-) on the u walls of the frame in which the problem is solved, and whether that frame is rotated."""
        rot = self.walls[0][0] == "v"
        Zp = self.Zw.get("v+" if rot else "u+", 0.0)
        Zm = self.Zw.get("v-" if rot else "u-", 0.0)
        return complex(Zp), complex(Zm), rot

    def gsm(self, f, basis1=None, basis2=None):
        basis = self._port_basis(f, basis1, basis2)
        if len(basis) == 0:
            Z = np.zeros((0, 0))
            return GSM(Z, Z, Z, Z, basis, basis, f, self.name)
        Zp, Zm, rot = self._u_walls()
        if rot:      # local (u, v) -> (u_r, v_r) = (v, -u): v+- walls become u+- walls (as the H -> E map)
            bh = basis.with_frame("H")
            rb = map_basis(bh, "H", "E")
            Fm = eplane_frame_signs(bh, rb)
            g = self._solve(f, rb, Zp, Zm)
            S = [Fm.T @ blk @ Fm for blk in (g.S11, g.S12, g.S21, g.S22)]
            return GSM(*S, basis, basis, f, self.name)
        return self._solve(f, basis, Zp, Zm)

    def _solve(self, f, basis, Zp, Zm):
        a, b = basis.a, basis.b
        m_max = max(md.m for md in basis.modes)
        M = self.modes_per_family or max(80, 2 * m_max + 20)
        if M < m_max:
            raise ValueError(f"modes_per_family = {M} is below the largest m of the port basis ({m_max})")
        x, w = leggauss(int(1.5 * M) + 48)
        up, wu = 0.5 * a * (x + 1), 0.5 * a * w
        blocks = []
        for n in basis.families():
            xv, wvq = leggauss(2 * n + 32)
            vp, wv = 0.5 * b * (xv + 1), 0.5 * b * wvq
            ideal = ModeBasis(a, b, ([Mode("TE", m, n) for m in range(0 if n else 1, M + 1)]
                                     + ([Mode("TM", m, n) for m in range(1, M + 1)] if n else [])))
            loaded = _lsu_family(f, a, b, n, Zp, Zm, M, {"u": (up, wu)})
            Eu, Ev, _, _ = _lsu_fields(loaded, vp[None, None, :])
            U, V = np.meshgrid(up - a / 2, vp - b / 2, indexing="ij")
            eu, ev = _ideal_fields(ideal, U, V)
            blocks.append(_section(f, ideal, loaded, Eu, Ev, eu, ev, np.outer(wu, wv), self.length, self.name))
        return _assemble(basis, f, blocks, self.name)


class ResistiveSheetFilter(_Filter):
    """Section of guide loaded with resistive sheets parallel to the walls v = +-b/2 (Butterweck 1968, eqs. 1-19).

    Parameters
    ----------
    length : float
        Length of the loaded section (m); the reference planes are its ends.
    R_sheet : float or sequence
        Surface resistance of the sheet(s) (ohm per square), one value or one per sheet. Butterweck's optimum for
        one centred sheet in X-band guide at 75 GHz: 56 ohm; his experimental filter: three equidistant 70 ohm
        sheets, 10 cm long (Philips Tech. Rev. 29, p. 95).
    v0 : float
        Position (m) of a single sheet on the local v axis (0 = centred, |v0| < b/2). Only with n_sheets = 1.
    n_sheets : int
        Number of equidistant sheets, at v' = j b / (n_sheets + 1), j = 1..n_sheets (v' = v + b/2), dividing the
        guide into equal layers. Ignored when `positions` is given.
    positions : sequence, optional
        Explicit sheet positions on the local v axis (m, |v| < b/2).
    plane : 'H' or 'E'
        Local frame. With plane='H' the sheets are parallel to the broad walls (Butterweck): TE_m0, whose E field
        is normal to the sheets, passes untouched.
    modes_per_family : int, optional
        Internal truncation Nn: per index m, the empty guide keeps TE_m0, TE_mn and TM_mn with n <= Nn and the
        sheet guide as many LS^v modes. Default: max(40, 2 n_max + 20). Convergence in Nn is slow (algebraic: the
        sheet current makes H jump), see tests/test_loss.py and docs/reports/WP2.md; check it for the case at hand.
    guide : Guide, optional
        Cross-section, for `default_basis`; the guide's wall conductivity is not applied in the filter section.

    Method (T1 boxes B7, B8, B10): the sheets do not vary along u, so the filter conserves the local index m. For
    each m, the exact LSM^v / LSE^v modes of the sheet-loaded guide (`sheet_modes`) are matched to the empty-guide
    modes at both ends (B9 form). The sheets are infinitely thin films (no dielectric carrier).
    """

    def __init__(self, length, R_sheet, v0=0.0, n_sheets=1, positions=None, plane="H", modes_per_family=None,
                 guide=None, name=""):
        self._init_common(length, plane, guide, name or f"resistive-sheet filter {float(length):g} m")
        n_sheets = int(n_sheets)
        if positions is None and n_sheets < 1:
            raise ValueError("n_sheets must be >= 1")
        if positions is None and n_sheets > 1 and v0 != 0.0:
            raise ValueError("v0 is only used with n_sheets = 1; give explicit positions instead")
        self.R_sheet = R_sheet if np.ndim(R_sheet) == 0 else tuple(R_sheet)
        self.v0, self.n_sheets = float(v0), n_sheets
        self.positions = None if positions is None else tuple(float(p) for p in positions)
        self.modes_per_family = None if modes_per_family is None else int(modes_per_family)

    def __repr__(self):
        return f"ResistiveSheetFilter({self.length!r}, {self.R_sheet!r}, plane={self.plane!r})"

    def with_guide(self, guide):
        return type(self)(self.length, self.R_sheet, self.v0, self.n_sheets, self.positions, self.plane,
                          self.modes_per_family, guide, self.name)

    def sheet_positions(self, b):
        """Sheet positions v' in (0, b) measured from the wall v = -b/2."""
        if self.positions is not None:
            return np.array(self.positions) + b / 2
        if self.n_sheets == 1:
            return np.array([self.v0 + b / 2])
        return b * np.arange(1, self.n_sheets + 1) / (self.n_sheets + 1)

    def gsm(self, f, basis1=None, basis2=None):
        basis = self._port_basis(f, basis1, basis2)
        if len(basis) == 0:
            Z = np.zeros((0, 0))
            return GSM(Z, Z, Z, Z, basis, basis, f, self.name)
        a, b = basis.a, basis.b
        pos, xis = _sheet_geometry(b, self.R_sheet, self.sheet_positions(b))
        n_max = max(md.n for md in basis.modes)
        Nn = self.modes_per_family or max(40, 2 * n_max + 20)
        if Nn < n_max:
            raise ValueError(f"modes_per_family = {Nn} is below the largest n of the port basis ({n_max})")
        breaks = np.concatenate([[0.0], pos, [b]])
        vp, wv, seg = [], [], []
        for d, (x0, x1) in enumerate(zip(breaks[:-1], breaks[1:])):
            x, w = leggauss(int(1.5 * Nn * (x1 - x0) / b) + 40)
            vp.append(x0 + (x1 - x0) * (x + 1) / 2)
            wv.append((x1 - x0) * w / 2)
            seg.append(np.full(len(x), d))
        vq = tuple(np.concatenate(z) for z in (vp, wv, seg))
        blocks = []
        for m in sorted({md.m for md in basis.modes}):
            xu, wuq = leggauss(2 * m + 32)
            up, wu = 0.5 * a * (xu + 1), 0.5 * a * wuq
            ideal = ModeBasis(a, b, ([Mode("TE", m, 0)] if m else []) + [Mode("TE", m, n) for n in range(1, Nn + 1)]
                              + ([Mode("TM", m, n) for n in range(1, Nn + 1)] if m else []))
            loaded = _sheet_family(f, a, b, m, xis, pos, Nn, vq)
            Eu, Ev, _, _ = _sheet_fields(loaded, up)
            U, V = np.meshgrid(up - a / 2, vq[0] - b / 2, indexing="ij")
            eu, ev = _ideal_fields(ideal, U, V)
            blocks.append(_section(f, ideal, loaded, Eu, Ev, eu, ev, np.outer(wu, vq[1]), self.length, self.name))
        return _assemble(basis, f, blocks, self.name)
