"""Independent check of the twist coupling: exact modes of a uniformly twisted guide.

Run:  cd <worktree> && PYTHONPATH=$PWD/src:$PWD/docs/theory/T3b python3 docs/theory/T3b/check_helical_eigen.py

For a constant twist rate tau the guide is invariant along z in the helical coordinates (u, v, z) of the note, and
Maxwell's equations there are those of a straight guide filled with the anisotropic medium eps_r = mu_r = g^{ij}
(sqrt(g) = 1). Fields ~ exp(-i gamma z) then solve the 2D problem

    iint (curl_g F)^* . g_{ij} (curl_g E) ds = k^2 iint F^* . g^{ij} E ds        (all admissible F)

with curl_g = coordinate curl with d_z -> -i gamma, acting on covariant components, and E_u, E_v, E_z (covariant)
tangential-zero on the walls. It is discretized here with a spectral Nedelec (tensor Legendre) basis that forms
an exact discrete de Rham sequence, so spurious solutions sit at k^2 = 0 exactly. No mode expansion and no
integration by parts on mode fields enter: the boundary terms that make the twist coupling subtle are handled by
the variational form alone.

For the same tau, the coupled-mode system of box 4 (twist_coupling.M_matrix) is exact in tau (no tau^2 terms
are neglected; see section 2 of the note), so its eigenvalues -i gamma must converge to the spectral ones as the
mode basis grows. The check compares, at the frequencies k_j(gamma) of the spectral solution:
  * the coupling of the note, V' = ... + tau K V,  I' = ... - tau K^T I   (K = G^T);
  * the 'naive' coupling obtained by projecting -tau G on both E and H,  V' = ... - tau G V,  I' = ... - tau G I,
    which differs from the note by the symmetric part of G (boundary terms) and is not even power-conserving;
  * the 'swapped' coupling V' = ... - tau K^T V, I' = ... + tau K I (lossless, the roles of E and H exchanged).
"""
import numpy as np
import scipy.constants as cst
from numpy.polynomial import legendre as L
from scipy.linalg import eigh, lu_factor, lu_solve

from wrmodes.modes import ModeBasis
from twist_coupling import K_matrix_fast


# --------------------------------------------------------------------------- 1D spectral bases on [-h, h]
def basis_1d(p, h, x):
    """phi_i (degree <= p, no condition) and psi_j (degree <= p+1, zero at +-h), with derivatives, at points x."""
    t = x / h
    phi, dphi, psi, dpsi = [], [], [], []
    for i in range(p + 1):
        c = np.zeros(i + 1); c[i] = 1
        phi.append(L.legval(t, c)); dphi.append(L.legval(t, L.legder(c)) / h)
    for j in range(1, p + 1):                                   # P_{j+1} - P_{j-1}: zero at t = +-1
        c = np.zeros(j + 2); c[j + 1] = 1; c[j - 1] = -1
        psi.append(L.legval(t, c)); dpsi.append(L.legval(t, L.legder(c)) / h)
    return [np.array(z) for z in (phi, dphi, psi, dpsi)]


def helical_modes(a, b, tau, gamma, p=22, nq=None):
    """k^2 eigenvalues (sorted, spurious zeros removed) of the twisted guide for propagation constant gamma."""
    nq = nq or p + 8
    xq, wq = L.leggauss(nq)
    u, wu = xq * a / 2, wq * a / 2
    v, wv = xq * b / 2, wq * b / 2
    Pu, dPu, Su, dSu = basis_1d(p, a / 2, u)
    Pv, dPv, Sv, dSv = basis_1d(p, b / 2, v)
    U, V = np.meshgrid(u, v, indexing="ij")
    W = np.outer(wu, wv).ravel()
    U, V = U.ravel(), V.ravel()
    kron = lambda fu, fv: (fu[:, None, :, None] * fv[None, :, None, :]).reshape(fu.shape[0] * fv.shape[0], -1)
    # DOF blocks: Eu = phi(u) psi(v), Ev = psi(u) phi(v), Ez = psi(u) psi(v)
    Eu, Eu_v = kron(Pu, Sv), kron(Pu, dSv)
    Ev, Ev_u = kron(Su, Pv), kron(dSu, Pv)
    Ez, Ez_u, Ez_v = kron(Su, Sv), kron(dSu, Sv), kron(Su, dSv)
    nu, nv, nz = Eu.shape[0], Ev.shape[0], Ez.shape[0]
    N, Q = nu + nv + nz, U.size
    F = np.zeros((3, N, Q), dtype=complex)                    # field components (covariant)
    C = np.zeros((3, N, Q), dtype=complex)                    # curl components (contravariant density)
    su, sv, sz = slice(0, nu), slice(nu, nu + nv), slice(nu + nv, N)
    F[0, su], F[1, sv], F[2, sz] = Eu, Ev, Ez
    ig = -1j * gamma                                           # d_z
    # (curl E)^u = d_v E_z - d_z E_v ; (curl E)^v = d_z E_u - d_u E_z ; (curl E)^z = d_u E_v - d_v E_u
    C[0, sz], C[0, sv] = Ez_v, -ig * Ev
    C[1, su], C[1, sz] = ig * Eu, -Ez_u
    C[2, sv], C[2, su] = Ev_u, -Eu_v
    # metric: g_ij (covariant) and g^ij (contravariant), helical coordinates, sqrt(g) = 1
    one, zero = np.ones(Q), np.zeros(Q)
    g_lo = np.array([[one, zero, -tau * V], [zero, one, tau * U], [-tau * V, tau * U, 1 + tau ** 2 * (U ** 2 + V ** 2)]])
    g_up = np.array([[1 + tau ** 2 * V ** 2, -tau ** 2 * U * V, tau * V], [-tau ** 2 * U * V, 1 + tau ** 2 * U ** 2, -tau * U],
                     [tau * V, -tau * U, one]])

    def form(X, G):
        out = np.zeros((X.shape[1], X.shape[1]), dtype=complex)
        for i in range(3):
            for j in range(3):
                out += (X[i].conj() * (G[i, j] * W)) @ X[j].T
        return out
    A = form(C, g_lo)        # curl E is mu^{-1}-weighted with mu^{-1} = (g^{ij})^{-1} = g_ij
    B = form(F, g_up)
    A = (A + A.conj().T) / 2
    B = (B + B.conj().T) / 2
    k2 = eigh(A, B, eigvals_only=True)
    scale = (np.pi / min(a, b)) ** 2
    return np.sort(k2[k2 > 1e-6 * scale])


# --------------------------------------------------------------------------- coupled-mode side
def cm_matrix(basis, K, f, tau, variant):
    """System matrix for [a+; a-] built from V-I twist terms (P on V, Q on I) of the chosen variant."""
    G = K.T                                   # G_pq = iint e_p . d_theta e_q
    P, Qm = {"note": (K, -K.T), "naive": (-G, -G), "swapped": (-K.T, K)}[variant]
    beta, Y = basis.beta(f), basis.admittance(f).astype(complex)
    DV, DI = np.sqrt(2 / Y), np.sqrt(2 * Y)
    PV = P * DV[None, :] / DV[:, None]                          # D_V^-1 P D_V
    QI = Qm * DI[None, :] / DI[:, None]                         # D_I^-1 Q D_I
    N = len(basis)
    M = np.zeros((2 * N, 2 * N), dtype=complex)
    M[:N, :N] = -1j * np.diag(beta) + tau * 0.5 * (PV + QI)
    M[:N, N:] = tau * 0.5 * (PV - QI)
    M[N:, :N] = tau * 0.5 * (PV - QI)
    M[N:, N:] = 1j * np.diag(beta) + tau * 0.5 * (PV + QI)
    return M


def nearest_eig(M, target, iters=6):
    """Eigenvalue of M nearest to target, by shifted inverse iteration (dense LU)."""
    lu = lu_factor(M - target * np.eye(len(M)))
    x = np.ones(len(M), dtype=complex)
    for _ in range(iters):
        x = lu_solve(lu, x)
        x /= np.linalg.norm(x)
    return x.conj() @ M @ x


def cm_gammas(a, b, tau, gamma, ks, chi_max, variant):
    """gamma_CM (signed, from the eigenvalue nearest -i gamma) at each k, basis: all modes with chi < chi_max."""
    basis = ModeBasis.below(chi_max * cst.c / (2 * np.pi), a, b)
    K = K_matrix_fast(basis)
    out = []
    for k in ks:
        lam = nearest_eig(cm_matrix(basis, K, k * cst.c / (2 * np.pi), tau, variant), -1j * gamma)
        out.append((1j * lam).real)
    return np.array(out), len(basis)


def run(a=2.0, b=1.0, tau=0.5, gamma=4.0, n_modes=6, chis=(3.0, 6.0, 12.0)):
    print(f"guide a = {a}, b = {b}, twist rate tau = {tau} rad/unit (tau a = {tau * a}), gamma = {gamma}")
    k2_lo = helical_modes(a, b, tau, gamma, p=16)
    k2 = helical_modes(a, b, tau, gamma, p=22)
    k = np.sqrt(k2[:n_modes])
    print("  spectral k_j:", np.array2string(k, precision=10))
    print(f"  spectral convergence p = 16 -> 22: max |dk/k| = {np.max(np.abs(np.sqrt(k2_lo[:n_modes]) / k - 1)):.1e}")
    res = {}
    for variant in ("note", "naive", "swapped"):
        prev = None
        for c in chis:
            g, nb = cm_gammas(a, b, tau, gamma, k, c * k.max(), variant)
            d = g / gamma - 1
            line = f"  {variant:7s} chi_max = {c:4.0f} k_max ({nb:5d} modes): max |gamma_CM/gamma - 1| = {np.max(np.abs(d)):.2e}"
            if prev is not None:
                rich = (c * d - prev[1] * prev[0]) / (c - prev[1])   # Richardson, error ~ 1/chi_max
                line += f"; Richardson (1/chi) extrapolation: {np.max(np.abs(rich)):.2e}"
                res[variant] = (np.max(np.abs(d)), np.max(np.abs(rich)))
            print(line)
            prev = (d, c)
    return res


if __name__ == "__main__":
    r1 = run()
    r2 = run(a=1.0, b=1.0, tau=1.0, gamma=6.0)
    for r in (r1, r2):
        assert r["note"][1] < 2e-4 and r["naive"][1] > 20 * r["note"][1] and r["swapped"][1] > 20 * r["note"][1]
    print("helical eigenvalue check passed: the coupling of the note converges to the exact twisted-guide modes")
