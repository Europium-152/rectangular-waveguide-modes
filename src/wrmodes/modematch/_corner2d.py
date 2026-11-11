"""Scalar 2D problems of the 90-degree miter: segmentation with Neumann-to-Dirichlet (impedance) cell matrices.

This module is the numerical core of `wrmodes.modematch.MiterMM` (derivation: docs/reports/WP4b.md, "Method").
Everything here is a 2D scalar problem in the bend plane,

    laplacian(psi) + kappa^2 psi = 0,     psi = 0 (wall type 'D') or d psi / dn = 0 (wall type 'N') on the walls,

which is exact for one longitudinal-section family of a component that does not vary along v (CONVENTIONS.md
section 7): psi = E_v for LSM (Dirichlet), psi = H_v for LSE (Neumann), kappa = k_t.

The domain is cut into *cells* (rectangles, and right isosceles triangles obtained from squares by images). On every
interface between cells, or between a cell and a guide arm, the unknown is the normal derivative g = d psi / dn,
expanded in an orthonormal trigonometric basis of the segment (`Basis1D`). Each cell is described by its
Neumann-to-Dirichlet map in Galerkin form,

    Z[k, l] = int_side' w'_k psi[w_l] dt,   psi[w_l] = cell solution for inward normal derivative w_l on one side,

computed in closed form from separable expansions (one expansion per side carrying data, the other sides homogeneous).
For a basis that is the natural eigenbasis of the side (the 1D eigenfunctions with the boundary types of the two
adjacent sides) every block is a closed form. The normal derivative carries no constraint at points where interfaces
meet, unlike the field itself (an expansion of psi in sines on each segment would force psi = 0 at interior junction
points and converge only logarithmically); this is why the impedance form is used for both wall types.

Arms are semi-infinite guides of width `a` with modes phi_m (sin for 'D', cos for 'N'), psi = sum (alpha e^{-i beta
zeta} + omega e^{+i beta zeta}) phi / sqrt(beta), zeta increasing towards the corner (alpha incident). Continuity of
d psi / d zeta (strong, projected on phi_m) and of psi (Galerkin, tested with the interface bases) give

    (Z_cells + i A^T A) G = 2 A^T alpha,    omega = alpha - i A G,    A = diag(beta^-1/2) P,

so S = I - 2 i A (Z_cells + i A^T A)^-1 A^T. Z_cells is real symmetric for real kappa (lossless cells), hence S is
symmetric and its propagating block unitary at every truncation.
"""
import numpy as np
from scipy.special import polygamma

__all__ = ["Basis1D", "rect_ntd", "triangle_t1", "triangle_t2", "Network", "miter_network", "miter_2d"]

_PI = np.pi


# ----------------------------------------------------------------------------------------------- 1D bases
class Basis1D:
    """Orthonormal trigonometric basis on a segment [0, L]: f_k(t) = A_k cos(p_k t + phi_k).

    kind 'cos': cos(k pi t / L), k = 0..K-1 (Neumann at both ends);
    'nd': cos((k + 1/2) pi t / L), k = 0..K-1 (Neumann at t = 0, Dirichlet at t = L);
    'dn': sin((k + 1/2) pi t / L) (Dirichlet at 0, Neumann at L);
    'sin': sin(k pi t / L), k = 1..K (Dirichlet at both ends).
    """

    __slots__ = ("kind", "L", "K", "A", "p", "phi", "end0", "end1")

    def __init__(self, kind, L, K):
        self.kind, self.L, self.K = kind, float(L), int(K)
        k = np.arange(self.K, dtype=float)
        L = self.L
        if kind == "cos":
            self.p = k * _PI / L
            self.phi = np.zeros(self.K)
            self.A = np.where(k == 0, np.sqrt(1 / L), np.sqrt(2 / L))
            self.end0, self.end1 = self.A.copy(), self.A * (-1.0) ** k
        elif kind == "nd":
            self.p = (k + 0.5) * _PI / L
            self.phi = np.zeros(self.K)
            self.A = np.full(self.K, np.sqrt(2 / L))
            self.end0, self.end1 = self.A.copy(), np.zeros(self.K)
        elif kind == "dn":
            self.p = (k + 0.5) * _PI / L
            self.phi = np.full(self.K, -0.5 * _PI)
            self.A = np.full(self.K, np.sqrt(2 / L))
            self.end0, self.end1 = np.zeros(self.K), self.A * (-1.0) ** k
        elif kind == "sin":
            self.p = (k + 1) * _PI / L
            self.phi = np.full(self.K, -0.5 * _PI)
            self.A = np.full(self.K, np.sqrt(2 / L))
            self.end0, self.end1 = np.zeros(self.K), np.zeros(self.K)
        else:
            raise ValueError(f"unknown basis kind {kind!r}")

    def funcs(self):
        return self.A, self.p, self.phi

    def reversed_funcs(self):
        """(A, p, phi) of t -> f(L - t)."""
        return self.A, -self.p, self.p * self.L + self.phi

    def shifted_funcs(self, offset):
        """(A, p, phi) of t -> f(t + offset)."""
        return self.A, self.p, self.phi + self.p * offset

    def __repr__(self):
        return f"Basis1D({self.kind!r}, L={self.L:g}, K={self.K})"


def natural_kind(type0, type1):
    """Eigenbasis kind of a side whose ends meet sides of boundary types type0, type1 ('N' or 'D')."""
    return {("N", "N"): "cos", ("N", "D"): "nd", ("D", "N"): "dn", ("D", "D"): "sin"}[(type0, type1)]


def cc_int(f1, f2, L):
    """Matrix [k, l] = int_0^L A1_k cos(p1_k t + phi1_k) A2_l cos(p2_l t + phi2_l) dt (closed form)."""
    A1, p1, h1 = (np.asarray(x, dtype=float)[:, None] for x in f1)
    A2, p2, h2 = (np.asarray(x, dtype=float)[None, :] for x in f2)

    def I(r, th):        # int_0^L cos(r t + th) dt = L cos(r L / 2 + th) sinc(r L / 2)
        return L * np.cos(0.5 * r * L + th) * np.sinc(0.5 * r * L / _PI)

    return 0.5 * A1 * A2 * (I(p1 - p2, h1 - h2) + I(p1 + p2, h1 + h2))


def _e1(z):
    """(exp(z) - 1) / z, with the limit 1 at z = 0 (complex)."""
    z = np.asarray(z, dtype=complex)
    out = np.ones_like(z)
    nz = z != 0
    out[nz] = np.expm1(z[nz]) / z[nz]
    return out


def ce_int(f, gam, L):
    """Matrix [k, j] = int_0^L A_k cos(p_k t + phi_k) exp(-gam_j t) dt, Re gam >= 0 (closed form)."""
    A, p, h = (np.asarray(x, dtype=float)[:, None] for x in f)
    g = np.asarray(gam, dtype=complex)[None, :]
    ep, em = np.exp(1j * h), np.exp(-1j * h)
    return 0.5 * A * L * (ep * _e1((1j * p - g) * L) + em * _e1((-1j * p - g) * L))


def _gamma(lam, kap2):
    """gamma = sqrt(lam^2 - kappa^2), principal root (Re >= 0)."""
    return np.sqrt(np.asarray(lam, dtype=float) ** 2 - kap2 + 0j)


# ----------------------------------------------------------------------------------------------- rectangle
# side: (length attribute, depth attribute, opposite, side at t = 0, side at t = L)
_SIDES = {"B": ("x", "y", "T", "L", "R"), "T": ("x", "y", "B", "L", "R"),
          "L": ("y", "x", "R", "B", "T"), "R": ("y", "x", "L", "B", "T")}
_DIRECT = {"B", "L"}     # sub-problems of B and L meet their adjacent sides at those sides' t = 0


def _tail_cos_on_nd(K, J, L, kap2):
    """Tail sum_{j >= J} Pi_kj Pi_lj rho_j for 'cos' data functions on a side whose eigenbasis is 'nd'.

    With x = j + 1/2, mu_j = pi x / L, p_k = pi k / L: Pi_kj = <cos_k, nd_j> = nu_k sqrt(2/L) (-1)^(k+j) mu_j /
    (mu_j^2 - p_k^2), and rho_j = -1/mu_j - kappa^2 / (2 mu_j^3) + O(mu^-5) (the opposite side's type only adds
    exponentially small terms). The leading term is summed exactly with polygamma functions,

        sum_{j>=J} x / ((x^2 - k^2)(x^2 - l^2)) = [psi(x0 - l) - psi(x0 - k) + psi(x0 + l) - psi(x0 + k)] / (2 (k^2 - l^2)),

    (k = l: [psi'(x0 - k) - psi'(x0 + k)] / (4 k), and -psi''(x0) / 2 for k = 0), x0 = J + 1/2; the kappa^2 term with
    Pi_kj Pi_lj ~ nu_k nu_l (2/L) (-1)^(k+l) / mu_j^2.
    """
    k = np.arange(K, dtype=float)
    nu = np.where(k == 0, np.sqrt(1 / L), np.sqrt(2 / L))
    x0 = J + 0.5
    kk, ll = k[:, None], k[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        S = (polygamma(0, x0 - ll) - polygamma(0, x0 - kk) + polygamma(0, x0 + ll) - polygamma(0, x0 + kk)) \
            / (2 * (kk ** 2 - ll ** 2))
    kpos = np.where(k == 0, 1.0, k)
    S[np.diag_indices(K)] = np.where(k == 0, -0.5 * polygamma(2, x0),
                                     (polygamma(1, x0 - k) - polygamma(1, x0 + k)) / (4 * kpos))
    pref = -nu[:, None] * nu[None, :] * (2 / L) * (-1.0) ** (kk + ll)
    lead = pref * (L / _PI) ** 3 * S
    nxt = pref * 0.5 * kap2 * (L / _PI) ** 5 * polygamma(4, x0) / 24.0
    return lead + nxt


GAMMA_FLOOR = 1e-7
CLOSED_FORMS = True     # adjacent blocks by Green's identity (False: by the explicit integrals; for checks)
CLOSED_FORM_MARGIN = 1e-4   # ... except within this relative distance of a cell eigenvalue


def rect_ntd(Lx, Ly, kap2, sides, tests=None, J_factor=8, J_min=64, check=False):
    """Neumann-to-Dirichlet blocks of the rectangle [0, Lx] x [0, Ly].

    Parameters
    ----------
    sides : dict 'B', 'T', 'L', 'R' -> 'D', 'N' (walls) or Basis1D (data side: inward normal derivative given in
        this basis; its length must match the side). Data sides are homogeneous Neumann for the sub-problems of the
        other sides. Side parameters: t = x on B (y = 0) and T (y = Ly), t = y on L (x = 0) and R (x = Lx).
        A data basis must be the natural eigenbasis of its side (`natural_kind` of its adjacent sides), except a
        'cos' basis on a side whose natural basis is 'nd' or 'dn' ("mismatched", at most one per rectangle),
        whose own block is summed over J = max(J_min, J_factor (K + kappa L / pi)) eigenfunctions plus an
        analytic tail (`_tail_cos_on_nd`).
    tests : iterable of side names, optional
        Test sides of the blocks to return (default: every data side).
    check : bool
        Diagnostics of the formulas: also evaluate every closed-form adjacent block by the explicit integral
        (rect_ntd.last_check = largest relative difference), and compute every cross block the other way round
        (rect_ntd.last_asymmetry = largest difference; the returned blocks are then the averages).

    Returns
    -------
    dict (s_test, s_data) -> ndarray (K_test, K_data): int_{s_test} w_k psi dt for psi the response to w_l on
    s_data, for s_test in `tests` and every data side s_data. Real for real kappa^2.

    Sub-problem of a data side s (length L, depth H, opposite side of type o): psi = sum_j c_j X_j(t) Y_j(eta),
    X_j the natural eigenbasis of s (eigenvalues lambda_j), eta the distance from s, gamma_j^2 = lambda_j^2 -
    kappa^2, Y_j = cosh(gamma_j (H - eta)) (o = 'N') or sinh(gamma_j (H - eta)) (o = 'D'), c_j = G_j / Y_j'(0).
    Blocks: own side diag(Y(0)/Y'(0)) = -coth(gamma H)/gamma or -tanh(gamma H)/gamma; opposite side (data, o = 'N')
    diag(-1 / (gamma sinh(gamma H))); adjacent side s' sharing the corner e of s (end e' of s'):
    Z[k, j] = X_j(e) int w_k(t') Y_j(eta(t')) dt' / Y_j'(0). When w is the natural basis of s' (eigenvalues p_k,
    Neumann at e', and at its far end the type o of s's opposite side), Green's identity gives the closed form
    Z[k, j] = -w_k(e') X_j(e) / (lambda_j^2 + p_k^2 - kappa^2); otherwise the integral is evaluated (`ce_int`).
    """
    dims = {"x": float(Lx), "y": float(Ly)}
    btype = {s: ("N" if isinstance(v, Basis1D) else v) for s, v in sides.items()}
    data = [s for s in "BTLR" if isinstance(sides[s], Basis1D)]
    tests = data if tests is None else [t for t in "BTLR" if t in tests]
    nat, mismatched = {}, set()
    for s in data:
        _, _, _, e0, e1 = _SIDES[s]
        nk = natural_kind(btype[e0], btype[e1])
        b = sides[s]
        if abs(b.L - dims[_SIDES[s][0]]) > 1e-12 * max(dims.values()):
            raise ValueError(f"side {s}: basis length {b.L} does not match the side ({dims[_SIDES[s][0]]})")
        if b.kind != nk:
            if b.kind == "cos" and nk in ("nd", "dn"):
                mismatched.add(s)
            else:
                raise ValueError(f"side {s}: basis {b.kind!r} on a side whose eigenbasis is {nk!r}")
        nat[s] = nk
    if len(mismatched) > 1:
        raise ValueError("at most one mismatched data side per rectangle is supported")

    def sub(s, X):
        """gamma, own-side ratio, opposite-side ratio, adjacent-kernel denominator and sign, depth H."""
        H_ = dims[_SIDES[s][1]]
        opp = btype[_SIDES[s][2]]
        g = _gamma(X.p, kap2)
        # gamma = 0 (kappa equal to an eigenvalue of the side): removable 0/0 for a Dirichlet opposite side, a pole of
        # the cell matrix (cell resonance) for a Neumann one. Both are evaluated at |gamma| H = GAMMA_FLOOR, a shift
        # of kappa^2 by GAMMA_FLOOR^2 / H^2, far below the solver's accuracy (resonances: `miter_2d`).
        g = np.where(np.abs(g) * H_ < GAMMA_FLOOR, GAMMA_FLOOR / H_, g)
        em2 = np.exp(-2 * g * H_)
        om = -np.expm1(-2 * g * H_)                 # 1 - exp(-2 g H)
        if opp == "N":
            r0 = -(1 + em2) / om / g                 # -coth(g H) / g
            rH = -2 * np.exp(-g * H_) / om / g       # -1 / (g sinh(g H))
            kden, ksgn = g * om, 1.0
        else:
            r0 = -om / (1 + em2) / g                 # -tanh(g H) / g
            rH = None
            kden, ksgn = g * (1 + em2), -1.0
        return g, r0, rH, kden, ksgn, H_

    def adjacent(s, X, g, kden, ksgn, H_, adj, e_idx):
        """Block (adj, s): test side adj, data side s (natural basis X), corner at end e_idx of s."""
        Xend = X.end0 if e_idx == 3 else X.end1
        W = sides[adj]
        direct = s in _DIRECT                      # adj's t' = 0 at the shared corner
        closed = None
        if adj not in mismatched:
            wend = W.end0 if direct else W.end1
            den = X.p[None, :] ** 2 + W.p[:, None] ** 2 - kap2
            closed = -wend[:, None] * Xend[None, :] / den
            # Near a cell eigenvalue (den -> 0) the closed form's pole is rounded independently of the self blocks'
            # (which go through gamma), and the cancellation of the pole in the network fails (error ~ eps /
            # distance^2). The integral form shares gamma with the self blocks and stays consistent there.
            near = np.min(np.abs(den)) < CLOSED_FORM_MARGIN * max(abs(kap2), 1.0 / H_ ** 2) if den.size else False
            if not check and CLOSED_FORMS and not near:
                return closed
        f_dir = W.funcs() if direct else W.reversed_funcs()
        f_rev = W.reversed_funcs() if direct else W.funcs()
        ker = -(ce_int(f_dir, g, H_) + ksgn * np.exp(-g * H_)[None, :] * ce_int(f_rev, g, H_)) / kden[None, :]
        integral = ker * Xend[None, :]
        if closed is None:
            return integral
        if check:
            rect_ntd.last_check = max(rect_ntd.last_check, float(np.max(np.abs(closed - integral))
                                                                / max(np.max(np.abs(closed)), 1e-300)))
        return closed if (CLOSED_FORMS and not near) else integral

    Z = {}
    rect_ntd.last_check = 0.0
    for s in data:
        if s in mismatched:
            continue
        X = sides[s]
        g, r0, rH, kden, ksgn, H_ = sub(s, X)
        opp = _SIDES[s][2]
        want = set(tests) | mismatched
        if check:
            want |= set(data)
        if s in want:
            Z[(s, s)] = np.diag(r0)
        if opp in data and opp in want and rH is not None:
            W = sides[opp]
            Pi = cc_int(W.funcs(), X.funcs(), W.L) if opp in mismatched else np.eye(W.K, X.K)
            Z[(opp, s)] = Pi * rH[None, :]
        for e_idx, adj in ((3, _SIDES[s][3]), (4, _SIDES[s][4])):
            if adj in data and adj in want:
                Z[(adj, s)] = adjacent(s, X, g, kden, ksgn, H_, adj, e_idx)
    for s in mismatched:
        W = sides[s]
        L_ = W.L
        J = int(max(J_min, J_factor * (W.K + np.sqrt(max(kap2, 0.0)) * L_ / _PI)))
        X = Basis1D(nat[s], L_, J)
        g, r0, rH, kden, ksgn, H_ = sub(s, X)
        Pi = cc_int(W.funcs(), X.funcs(), L_)
        tail = _tail_cos_on_nd(W.K, J, L_, kap2)
        if nat[s] == "dn":
            kk = np.arange(W.K)
            tail = tail * (-1.0) ** (kk[:, None] + kk[None, :])
        Z[(s, s)] = (Pi * r0[None, :]) @ Pi.T + tail
        for t in data:
            if t != s:
                Z[(t, s)] = Z[(s, t)].T
    # every block is real for real kappa^2 (lossless cells; gamma real or imaginary); drop the round-off parts
    rect_ntd.last_imag = max((float(np.max(np.abs(np.imag(b)))) for b in Z.values() if b.size), default=0.0)
    out = {key: np.real(blk) for key, blk in Z.items()}
    rect_ntd.last_asymmetry = 0.0
    if check:
        for t in data:
            for s in data:
                if s < t and (s, t) in out and (t, s) in out:
                    diff = np.max(np.abs(out[(t, s)] - out[(s, t)].T)) if out[(t, s)].size else 0.0
                    rect_ntd.last_asymmetry = max(rect_ntd.last_asymmetry, diff)
                    avg = 0.5 * (out[(t, s)] + out[(s, t)].T)
                    out[(t, s)], out[(s, t)] = avg, avg.T
    return {key: blk for key, blk in out.items() if key[0] in tests}


rect_ntd.last_asymmetry = 0.0   # diagnostics of the last call (check=True): max |Z_ts - Z_st^T| before averaging,
rect_ntd.last_check = 0.0       # closed form vs integral of the adjacent blocks,
rect_ntd.last_imag = 0.0        # and the largest imaginary part dropped


# ----------------------------------------------------------------------------------------------- triangles
def _J(K):
    return (-1.0) ** np.arange(K)


def triangle_t2(base, sig1, sig2, kap2):
    """NtD of the right isosceles triangle with data on its hypotenuse (`base`, a 'cos' Basis1D of length e) and
    walls on its legs: the leg through t = 0 of type sig1, the leg through t = e of type sig2 (+1: Neumann, -1:
    Dirichlet). Images across the legs turn it into the square [0, e]^2 with data on all four sides."""
    e, K = base.L, base.K
    sq = rect_ntd(e, e, kap2, {s: base for s in "BTLR"}, tests=("B",))
    J = np.diag(_J(K))
    return sq[("B", "B")] + sig1 * sq[("B", "L")] + sig2 * sq[("B", "R")] @ J + sig1 * sig2 * sq[("B", "T")] @ J


def triangle_t1(leg1, leg2, sig, kap2, hyp="main"):
    """NtD of the right isosceles triangle with data on its legs and a wall of type sig on its hypotenuse.

    hyp='main': triangle {0 <= y <= x <= e}; leg1 = bottom (y = 0, t = x), leg2 = right side (x = e, t = y);
    hypotenuse on y = x. hyp='anti': triangle {x + y <= e}; leg1 = bottom, leg2 = left side (x = 0, t = y);
    hypotenuse on x + y = e. Both legs 'cos' bases of length e. Returns the 2 x 2 block matrix as one array.
    """
    e = leg1.L
    if hyp == "main":
        sq = rect_ntd(e, e, kap2, {"B": leg1, "R": leg2, "L": leg1, "T": leg2}, tests=("B", "R"))
        Z11 = sq[("B", "B")] + sig * sq[("B", "L")]
        Z12 = sq[("B", "R")] + sig * sq[("B", "T")]
        Z21 = sq[("R", "B")] + sig * sq[("R", "L")]
        Z22 = sq[("R", "R")] + sig * sq[("R", "T")]
    else:
        J1, J2 = np.diag(_J(leg1.K)), np.diag(_J(leg2.K))
        sq = rect_ntd(e, e, kap2, {"B": leg1, "L": leg2, "R": leg1, "T": leg2}, tests=("B", "L"))
        Z11 = sq[("B", "B")] + sig * sq[("B", "R")] @ J1
        Z12 = sq[("B", "L")] + sig * sq[("B", "T")] @ J2
        Z21 = sq[("L", "B")] + sig * sq[("L", "R")] @ J1
        Z22 = sq[("L", "L")] + sig * sq[("L", "T")] @ J2
    Z = np.block([[Z11, Z12], [Z21, Z22]])
    return 0.5 * (Z + Z.T)


# ----------------------------------------------------------------------------------------------- network
class Network:
    """Cells joined by interfaces, fed by arms; see the module docstring."""

    def __init__(self):
        self.ifaces = {}
        self.order = []
        self.cells = []
        self.arms = []

    def add_iface(self, name, basis):
        self.ifaces[name] = basis
        self.order.append(name)

    def add_cell(self, Z, sides):
        """Z: matrix over the concatenated data sides; sides: list of (interface name, sign), sign +1 when the
        cell's inward normal is the interface's normal direction."""
        self.cells.append((np.asarray(Z), list(sides)))

    def add_arm(self, segments):
        """segments: list of (interface name, offset of the segment's t = 0 along the arm's transverse coordinate)."""
        self.arms.append(list(segments))

    def _index(self):
        off, n = {}, 0
        for name in self.order:
            off[name] = n
            n += self.ifaces[name].K
        return off, n

    def solve(self, arm_basis, beta):
        """Generalized scattering matrix (n_arms M x n_arms M) of the network for arm modes `arm_basis`
        (Basis1D on [0, a], 'sin' or 'cos') with propagation constants `beta` (CONVENTIONS.md branch)."""
        off, n = self._index()
        Zg = np.zeros((n, n), dtype=complex)
        for Z, sides in self.cells:
            idx = np.concatenate([np.arange(off[nm], off[nm] + self.ifaces[nm].K) for nm, _ in sides])
            sg = np.concatenate([np.full(self.ifaces[nm].K, float(s)) for nm, s in sides])
            Zg[np.ix_(idx, idx)] += sg[:, None] * Z * sg[None, :]
        M = arm_basis.K
        sb = np.sqrt(np.asarray(beta, dtype=complex))
        A = np.zeros((len(self.arms) * M, n), dtype=complex)
        for i, segs in enumerate(self.arms):
            for nm, o in segs:
                w = self.ifaces[nm]
                P = cc_int(arm_basis.shifted_funcs(o), w.funcs(), w.L)
                A[i * M:(i + 1) * M, off[nm]:off[nm] + w.K] = P / sb[:, None]
        Zg += 1j * (A.T @ A)
        X = np.linalg.solve(Zg, A.T)
        S = np.eye(A.shape[0]) - 2j * (A @ X)
        return 0.5 * (S + S.T)


# ----------------------------------------------------------------------------------------------- the miter
def _seg_K(L, a, Ka, K_min=2):
    return int(max(K_min, np.ceil(Ka * L / a - 1e-9)))


def miter_network(a, c, wall, Ka, kap2, parity=None):
    """Network of the 90-degree miter corner in the bend plane (see docs/reports/WP4b.md, Fig. "segmentation").

    Coordinates: inner corner at the origin; arm 1 is x in [0, a], y < 0 (port plane y = 0), arm 2 is y in [0, a],
    x < 0 (port plane x = 0); the outer walls are x = a and y = a; the miter is the line x + y = a + c, i.e.
    c = sqrt(2) d - a with d the distance of the miter from the inner corner (0 <= c <= a; c = 0 the optical mirror,
    c = a the unmitered corner). wall: 'D' or 'N'. Ka: interface functions per width a.

    parity None: the full two-port network (cells C, B, E and the triangle D of Alessandri et al. 1994, Fig. 1).
    parity 'even' / 'odd': the half below the bisector y = x with a Neumann / Dirichlet wall on the bisector (one
    port). The miter is symmetric about the bisector and the reflection maps mode m of arm 1 to mode m of arm 2 with
    sign +1, so S11 = S22 = (G_even + G_odd) / 2 and S21 = S12 = (G_even - G_odd) / 2.
    """
    sw = -1.0 if wall == "D" else 1.0
    nd = "nd" if wall == "D" else "cos"
    e = a - c
    net = Network()
    if parity is None:
        if c <= 0:
            p1, p2 = Basis1D("cos", a, _seg_K(a, a, Ka)), Basis1D("cos", a, _seg_K(a, a, Ka))
            net.add_iface("p1", p1)
            net.add_iface("p2", p2)
            net.add_cell(triangle_t1(p1, p2, sw, kap2, hyp="anti"), [("p1", 1), ("p2", 1)])
        elif e <= 0:
            p1, p2 = Basis1D(nd, a, _seg_K(a, a, Ka)), Basis1D(nd, a, _seg_K(a, a, Ka))
            net.add_iface("p1", p1)
            net.add_iface("p2", p2)
            Z = rect_ntd(a, a, kap2, {"B": p1, "L": p2, "R": wall, "T": wall})
            net.add_cell(np.block([[Z[("B", "B")], Z[("B", "L")]], [Z[("L", "B")], Z[("L", "L")]]]),
                         [("p1", 1), ("p2", 1)])
        else:
            Kc, Ke = _seg_K(c, a, Ka), _seg_K(e, a, Ka)
            g1, g5 = Basis1D("cos", c, Kc), Basis1D("cos", c, Kc)        # port segments on [0, c]
            g3, g7 = Basis1D(nd, e, Ke), Basis1D(nd, e, Ke)              # port segments on [c, a]
            g2, g6 = Basis1D("cos", c, Kc), Basis1D("cos", c, Kc)        # x = c (C|B), y = c (C|E)
            g4, g8 = Basis1D("cos", e, Ke), Basis1D("cos", e, Ke)        # y = c (B|D), x = c (E|D)
            for nm, b in (("g1", g1), ("g3", g3), ("g5", g5), ("g7", g7), ("g2", g2), ("g6", g6), ("g4", g4),
                          ("g8", g8)):
                net.add_iface(nm, b)
            ZC = rect_ntd(c, c, kap2, {"B": g1, "L": g5, "R": g2, "T": g6})
            order = ["B", "L", "R", "T"]
            net.add_cell(np.block([[ZC[(t, s)] for s in order] for t in order]),
                         [("g1", 1), ("g5", 1), ("g2", -1), ("g6", -1)])
            ZB = rect_ntd(e, c, kap2, {"B": g3, "L": g2, "T": g4, "R": wall})
            order = ["B", "L", "T"]
            net.add_cell(np.block([[ZB[(t, s)] for s in order] for t in order]), [("g3", 1), ("g2", 1), ("g4", -1)])
            ZE = rect_ntd(c, e, kap2, {"B": g6, "L": g7, "R": g8, "T": wall})
            order = ["B", "L", "R"]
            net.add_cell(np.block([[ZE[(t, s)] for s in order] for t in order]), [("g6", 1), ("g7", 1), ("g8", -1)])
            net.add_cell(triangle_t1(g4, g8, sw, kap2, hyp="anti"), [("g4", 1), ("g8", 1)])
            net.add_arm([("g1", 0.0), ("g3", c)])
            net.add_arm([("g5", 0.0), ("g7", c)])
            return net
        net.add_arm([("p1", 0.0)])
        net.add_arm([("p2", 0.0)])
        return net
    sb = 1.0 if parity == "even" else -1.0
    if c <= 0:
        p = Basis1D("cos", a, _seg_K(a, a, Ka))
        net.add_iface("p", p)
        net.add_cell(triangle_t2(p, sb, sw, kap2), [("p", 1)])
        net.add_arm([("p", 0.0)])
        return net
    if e <= 0:
        p = Basis1D(nd, a, _seg_K(a, a, Ka))
        net.add_iface("p", p)
        Z = rect_ntd(a, a, kap2, {"B": p, "L": p, "R": wall, "T": wall}, tests=("B",))
        net.add_cell(Z[("B", "B")] + sb * Z[("B", "L")], [("p", 1)])
        net.add_arm([("p", 0.0)])
        return net
    Kc, Ke = _seg_K(c, a, Ka), _seg_K(e, a, Ka)
    g1, g2 = Basis1D("cos", c, Kc), Basis1D("cos", c, Kc)
    g3, g4 = Basis1D(nd, e, Ke), Basis1D("cos", e, Ke)
    for nm, b in (("g1", g1), ("g3", g3), ("g2", g2), ("g4", g4)):
        net.add_iface(nm, b)
    net.add_cell(triangle_t1(g1, g2, sb, kap2, hyp="main"), [("g1", 1), ("g2", -1)])
    ZB = rect_ntd(e, c, kap2, {"B": g3, "L": g2, "T": g4, "R": wall})
    order = ["B", "L", "T"]
    net.add_cell(np.block([[ZB[(t, s)] for s in order] for t in order]), [("g3", 1), ("g2", 1), ("g4", -1)])
    net.add_cell(triangle_t2(g4, sb, sw, kap2), [("g4", 1)])
    net.add_arm([("g1", 0.0), ("g3", c)])
    return net


def arm_modes(a, kap2, M, wall, k2=None, chi2_offset=0.0):
    """Arm basis (Basis1D 'sin' for 'D', 'cos' for 'N', M modes) and beta_m = sqrt(kap2 - (m pi / a)^2) with the
    branch -i|beta| below cutoff (CONVENTIONS.md section 3)."""
    arm = Basis1D("sin" if wall == "D" else "cos", a, M)
    b2 = kap2 - arm.p ** 2
    beta = np.where(b2 > 0, np.sqrt(np.abs(b2)) + 0j, -1j * np.sqrt(np.abs(b2)))
    return arm, beta


def _eig1d(L, t0, t1, kmax):
    """1D eigenvalues (wavenumbers) below kmax of [0, L] with boundary types t0, t1."""
    half = 0.5 if t0 != t1 else 0.0
    j = np.arange(int(kmax * L / _PI) + 2, dtype=float)
    lam = (j + half) * _PI / L
    if t0 == "D" and t1 == "D":
        lam = lam[1:]
    return lam[lam <= kmax]


def cell_resonance_distance(a, c, wall, kap2):
    """min |kappa^2 - kappa_cell^2| / kappa^2 over the eigenvalues of every cell of `miter_network` (data sides
    taken as Neumann; the image squares of the triangles with Neumann on all sides). The cell matrices have poles
    there, and their cancellation in the network loses about 1e-16 / distance^1.3 of accuracy."""
    e = a - c
    k = np.sqrt(abs(kap2)) * 1.5 + 10.0 / a
    rects = []
    if c <= 0:
        rects = [(a, ("N", "N"), a, ("N", "N"))]
    elif e <= 0:
        rects = [(a, ("N", wall), a, ("N", wall))]
    else:
        rects = [(c, ("N", "N"), c, ("N", "N")), (e, ("N", wall), c, ("N", "N")), (e, ("N", "N"), e, ("N", "N"))]
    best = np.inf
    for Lx, tx, Ly, ty in rects:
        lx, ly = _eig1d(Lx, *tx, k), _eig1d(Ly, *ty, k)
        if lx.size and ly.size:
            best = min(best, float(np.min(np.abs(lx[:, None] ** 2 + ly[None, :] ** 2 - kap2))) / abs(kap2))
    return best


RESONANCE_GUARD = 1e-9
RESONANCE_STEP = 1e-8


def miter_2d(a, c, wall, kap2, M, Ka, symmetric=True):
    """(S11, S21) of the scalar 2D miter (M x M each), arm modes of `arm_modes`; S22 = S11, S12 = S21.

    Within RESONANCE_GUARD (relative, in kappa^2) of a cell eigenvalue the result is the mean of the solutions at
    kappa^2 (1 +- RESONANCE_STEP): its error is O(RESONANCE_STEP^2) from the averaging plus about 3e-9 from the
    remaining proximity, instead of an unbounded loss of accuracy. The mean is not exactly unitary; its defect is of
    the order of RESONANCE_STEP^2 times the second derivative (below 1e-12)."""
    if cell_resonance_distance(a, c, wall, kap2) < RESONANCE_GUARD:
        lo = miter_2d(a, c, wall, kap2 * (1 - RESONANCE_STEP), M, Ka, symmetric)
        hi = miter_2d(a, c, wall, kap2 * (1 + RESONANCE_STEP), M, Ka, symmetric)
        return tuple(0.5 * (x + y) for x, y in zip(lo, hi))
    arm, beta = arm_modes(a, kap2, M, wall)
    if np.any(beta == 0):
        raise ValueError("an arm mode is exactly at cutoff")
    if symmetric:
        Ge = miter_network(a, c, wall, Ka, kap2, "even").solve(arm, beta)
        Go = miter_network(a, c, wall, Ka, kap2, "odd").solve(arm, beta)
        return 0.5 * (Ge + Go), 0.5 * (Ge - Go)
    S = miter_network(a, c, wall, Ka, kap2, None).solve(arm, beta)
    return S[:M, :M], S[M:, :M], S[:M, M:], S[M:, M:]
