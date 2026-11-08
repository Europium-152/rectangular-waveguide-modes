"""T3 check 2: the coupled-mode equations against an independent solution of Maxwell's equations.

A uniformly twisted guide (theta' = tau constant) is invariant under the helical motion, so in the co-rotating
coordinates (u, v, z) its fields are exp(-i gamma z) times a cross-section pattern. In those coordinates
Maxwell's equations are those of a straight guide filled with eps0 G, mu0 G, where G = g^{ij} is the inverse
metric of r = R(theta(z)) (u, v) + z z_hat (sqrt(g) = 1):

    g_ij = [[1, 0, w_u], [0, 1, w_v], [w_u, w_v, 1 + |w|^2]],   w = tau (-v, u),   G = g^-1.

Independent method: Galerkin solution of curl(G^-1 curl E) = k^2 G E (weak form, E in a trigonometric basis that
satisfies tangential E = 0 componentwise). No modal expansion, no integration by parts of series: it tests
the treatment of the boundary terms in the coupled-mode derivation (box T3-1), in particular the symmetric
part S of C, which appears with opposite signs in the V and I equations.

For each forward eigenvalue gamma of the coupled-mode system (many modes, evanescent included) the Galerkin
gamma (the gamma at which the Galerkin problem has the eigenvalue k^2, by secant iteration) is computed, and the
difference is followed as the coupled-mode basis grows (3, 5, 8 times f), with a linear extrapolation in
1/cutoff. Three variants of the coupled-mode equations are compared:
  'T3'        dV/dz = ... + tau C^T V,  dI/dz = ... - tau C I   (this note)
  'naive'     both with -tau C        (projection of the rotation generator without boundary terms)
  'antisym'   both with -tau (C - C^T)/2

Run time: about 40 min on two cores.
"""
import os
import sys

import numpy as np
import scipy.constants as cst
from scipy.linalg import eig, eigh

sys.path.insert(0, os.path.dirname(__file__))
from twist_formulas import coupling_matrix  # noqa: E402
from wrmodes.modes import ModeBasis  # noqa: E402


def cm_gammas(basis, f, tau, variant="T3"):
    C = coupling_matrix(basis)
    beta, Y = basis.beta(f), basis.admittance(f)
    N = len(basis)
    if variant == "T3":
        MV, MI = C.T, -C
    elif variant == "naive":
        MV, MI = -C, -C
    else:
        A = 0.5 * (C - C.T)
        MV, MI = -A, -A
    M = np.zeros((2 * N, 2 * N), complex)
    M[:N, :N] = tau * MV
    M[:N, N:] = -1j * np.diag(beta / Y)
    M[N:, :N] = -1j * np.diag(beta * Y)
    M[N:, N:] = tau * MI
    lam, vec = eig(M)
    gam = 1j * lam                       # y ~ exp(lam z) = exp(-i gamma z)
    keep = (np.abs(gam.imag) < 1e-6 * np.abs(gam)) & (gam.real > 0)
    order = np.argsort(-gam[keep].real)
    return gam[keep].real[order], vec[:, keep][:, order]


class Galerkin:
    """Galerkin solution of curl(G^-1 curl E) = k^2 G E on the a x b rectangle, E ~ exp(-i gamma z).

    Basis (unnormalized): E_u = cos(k_m u') sin(l_n v'), E_v = sin cos, E_z = sin sin, so that tangential E
    vanishes on every wall. G and G^-1 are polynomials of degree <= 2 in (u, v), so every matrix element is a
    sum of products of 1D integrals, computed with Gauss-Legendre quadrature.
    """

    def __init__(self, a, b, nm, nn, nq=None):
        self.a, self.b = a, b
        nq = nq or 2 * max(nm, nn) + 30
        x, w = np.polynomial.legendre.leggauss(nq)
        self.u, self.wu = x * a / 2, w * a / 2
        self.v, self.wv = x * b / 2, w * b / 2
        comp, M, N = [], [], []
        for c, (m0, m1, n0, n1) in enumerate([(0, nm - 1, 1, nn), (1, nm, 0, nn - 1), (1, nm, 1, nn)]):
            for m in range(m0, m1 + 1):
                for n in range(n0, n1 + 1):
                    comp.append(c); M.append(m); N.append(n)
        self.comp, self.M, self.N = np.array(comp), np.array(M), np.array(N)
        km, ln = self.M * np.pi / a, self.N * np.pi / b
        up = np.outer(km, self.u + a / 2)
        vp = np.outer(ln, self.v + b / 2)
        self.cu, self.su, self.cv, self.sv = np.cos(up), np.sin(up), np.cos(vp), np.sin(vp)
        self.km, self.ln = km, ln

    def _components(self, gamma):
        """For E and curl E: per component c, (scale s_i, X_i(u) rows, Y_i(v) rows)."""
        nb = len(self.comp)
        Z = np.zeros_like(self.cu)
        zeros = np.zeros(nb, complex)
        c0, c1, c2 = self.comp == 0, self.comp == 1, self.comp == 2
        km, ln = self.km, self.ln

        def pick(mask_rows):
            X = np.where(mask_rows[0][:, None], mask_rows[1], Z)
            return X

        # E components
        E = []
        for c, (X, Y) in enumerate([(self.cu, self.sv), (self.su, self.cv), (self.su, self.sv)]):
            m = self.comp == c
            E.append((np.where(m, 1.0 + 0j, zeros), np.where(m[:, None], X, Z), np.where(m[:, None], Y, Z)))
        # curl components: u: E_v -> i g sin cos ; E_z -> l sin cos
        su_cv_s = np.where(c1, 1j * gamma, 0) + np.where(c2, ln, 0)
        Ru = (su_cv_s, np.where((c1 | c2)[:, None], self.su, Z), np.where((c1 | c2)[:, None], self.cv, Z))
        # v: E_u -> -i g cos sin ; E_z -> -k cos sin
        cu_sv_s = np.where(c0, -1j * gamma, 0) + np.where(c2, -km, 0)
        Rv = (cu_sv_s, np.where((c0 | c2)[:, None], self.cu, Z), np.where((c0 | c2)[:, None], self.sv, Z))
        # z: E_u -> -l cos cos ; E_v -> k cos cos
        cc_s = np.where(c0, -ln, 0) + np.where(c1, km, 0) + 0j
        Rz = (cc_s, np.where((c0 | c1)[:, None], self.cu, Z), np.where((c0 | c1)[:, None], self.cv, Z))
        return E, [Ru, Rv, Rz]

    def _form(self, P, W):
        nb = len(self.comp)
        out = np.zeros((nb, nb), complex)
        upow = [self.wu * self.u ** k for k in range(3)]
        vpow = [self.wv * self.v ** k for k in range(3)]
        for c in range(3):
            for d in range(3):
                for (al, be), coef in W[c][d].items():
                    sc, Xc, Yc = P[c]
                    sd, Xd, Yd = P[d]
                    Iu = (Xc * upow[al]) @ Xd.T
                    Iv = (Yc * vpow[be]) @ Yd.T
                    out += coef * np.outer(sc.conj(), sd) * Iu * Iv
        return out

    def matrices(self, gamma, tau):
        t, t2 = tau, tau ** 2
        Gi = [[{(0, 0): 1}, {}, {(0, 1): -t}],
              [{}, {(0, 0): 1}, {(1, 0): t}],
              [{(0, 1): -t}, {(1, 0): t}, {(0, 0): 1, (2, 0): t2, (0, 2): t2}]]
        G = [[{(0, 0): 1, (0, 2): t2}, {(1, 1): -t2}, {(0, 1): t}],
             [{(1, 1): -t2}, {(0, 0): 1, (2, 0): t2}, {(1, 0): -t}],
             [{(0, 1): t}, {(1, 0): -t}, {(0, 0): 1}]]
        E, R = self._components(gamma)
        return self._form(R, Gi), self._form(E, G)

    def k2_near(self, gamma, tau, k2):
        A, B = self.matrices(gamma, tau)
        ev = eigh(A, B, eigvals_only=True)
        return ev[np.argmin(np.abs(ev - k2))]


def galerkin_gamma(gal, tau, k, gamma0, tol=1e-11):
    """gamma at which the Galerkin problem has the eigenvalue k^2 (secant iteration from gamma0)."""
    def r(g):
        return gal.k2_near(g, tau, k ** 2) - k ** 2
    g0, g1 = gamma0, gamma0 * (1 + 1e-4)
    r0, r1 = r(g0), r(g1)
    for _ in range(12):
        g2 = g1 - r1 * (g1 - g0) / (r1 - r0)
        if abs(g2 - g1) < tol * abs(g1):
            return g2
        g0, r0, g1, r1 = g1, r1, g2, r(g2)
    return g1


def main(a=0.030, b=0.020, f=15e9, tau_over_k=0.15, cutoffs=(3.0, 5.0, 8.0), nb=(20, 28)):
    k = 2 * np.pi * f / cst.c
    tau = tau_over_k * k
    print(f"guide {a*1e3:.0f} x {b*1e3:.0f} mm, f = {f/1e9:g} GHz, k = {k:.2f} 1/m, twist rate tau = {tau:.2f} rad/m "
          f"(90 deg in {np.pi/2/tau*1e3:.1f} mm)")
    prop = ModeBasis.propagating(f, a, b)
    print("propagating modes at tau = 0:", ", ".join(prop.names))
    cm = {}
    for cutoff in cutoffs:
        basis = ModeBasis.below(cutoff * f, a, b)
        for variant in ("T3", "naive", "antisym"):
            cm[(cutoff, variant)] = cm_gammas(basis, f, tau, variant)[0]
        print(f"  coupled-mode basis below {cutoff:4.1f} f: {len(basis):4d} modes")
    ref_cut = cutoffs[-1]
    gam_ref = cm[(ref_cut, "T3")]
    gal = {n: Galerkin(a, b, n, n) for n in nb}
    G = {n: np.array([galerkin_gamma(gal[n], tau, k, g) for g in gam_ref]) for n in nb}
    print("\nGalerkin gamma (1/m):  " + "   ".join(f"Nb={n}: " + ", ".join(f"{x:.5f}" for x in G[n]) for n in nb))
    gref = G[nb[-1]]
    galerr = np.max(np.abs(G[nb[-1]] - G[nb[0]]) / k)
    print(f"Galerkin change between Nb={nb[0]} and Nb={nb[-1]}: {galerr:.1e} of k")
    print("\n(gamma_coupled-mode - gamma_Galerkin) / k, per forward mode:")
    worst, rows = {}, {}
    for variant in ("T3", "naive", "antisym"):
        for cutoff in cutoffs:
            g = cm[(cutoff, variant)]
            if len(g) != len(gref):
                print(f"{variant:8s} below {cutoff:4.1f} f: {len(g)} forward modes (Galerkin: {len(gref)})")
                worst[(variant, cutoff)] = np.inf
                continue
            d = (g - gref) / k
            rows.setdefault(variant, []).append(d)
            worst[(variant, cutoff)] = np.max(np.abs(d))
            print(f"{variant:8s} below {cutoff:4.1f} f: " + " ".join(f"{x:+.2e}" for x in d))
    return worst, galerr, cutoffs, rows


def extrapolate(worst_rows, cutoffs):
    """Least-squares line in 1/cutoff through the signed errors of each mode; value at 1/cutoff = 0."""
    x = 1 / np.asarray(cutoffs)
    return np.polyfit(x, np.asarray(worst_rows), 1)[1]


if __name__ == "__main__":
    worst, galerr, cutoffs, rows = main()
    c = cutoffs[-1]
    print(f"\nworst |dgamma|/k with the largest basis: T3 {worst[('T3', c)]:.1e}, naive {worst[('naive', c)]:.1e}, "
          f"antisym {worst[('antisym', c)]:.1e}")
    ext = {v: extrapolate(rows[v], cutoffs) for v in ("T3", "naive", "antisym")}
    for v, e in ext.items():
        print(f"{v:8s} extrapolated to an infinite basis (linear in 1/cutoff): " + " ".join(f"{x:+.1e}" for x in e))
    # T3 converges to the Galerkin solution: errors shrink monotonically and extrapolate to < 1e-3 k
    r = np.abs(np.asarray(rows["T3"]))
    assert np.all(np.diff(r, axis=0) < 0), "T3 errors do not decrease with the basis"
    assert np.max(np.abs(ext["T3"])) < 1e-3
    assert np.max(np.abs(ext["naive"])) > 1e-2 and np.max(np.abs(ext["antisym"])) > 1e-2
    print("PASS: only the T3 form (C^T in the V equation, -C in the I equation) converges to the Galerkin solution")
