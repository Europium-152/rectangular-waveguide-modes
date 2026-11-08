"""Surface-integral reference for ohmic wall loss (WP7 item 2, verifies WP2).

For a forward mode p of unit amplitude (CONVENTIONS.md section 5) the fields are

    E_t = sqrt(2/Y) e,   H_t = sqrt(2 Y) z_hat x e,   H_z = (i / (omega mu0)) sqrt(2/Y) (curl e)_z,

the last from Faraday's law, curl E = -i omega mu0 H, whose z component involves only E_t. The transverse field e is
taken from ``wrmodes.modes.transverse_field`` and its curl is computed by fourth-order central differences of that
function (no closed form, no ``wrmodes.modes.divergence``). On the walls,

    P_loss = (1/2) R_s  oint |H_tan|^2 dl,     H_tan = (H_v, H_z) on u = +-a/2,  (H_u, H_z) on v = +-b/2,

with Gauss-Legendre quadrature on each wall, and the power P = (1/2) Re int (E x H*) . z_hat ds by Gauss-Legendre
quadrature over the cross-section (1 for the normalization of CONVENTIONS.md). The attenuation constant of the
field amplitude is alpha = P_loss / (2 P) (Np/m); 8.686 alpha is in dB/m.

The perturbation is applied to every propagating mode, including the degenerate TE_mn/TM_mn pairs (m, n >= 1),
for which the wall integral also couples the pair. ``loss_matrix`` returns the full real symmetric matrix
A_pq = (1/4) R_s oint H_tan,p . conj(H_tan,q) dl (for P_p = P_q = 1), with diag(A) = alpha, so that the amplitudes of
a lossy straight obey da/dz = -(i beta + A) a to first order in R_s (the reactive part of Z_s = (1+i) R_s, which
shifts beta by the same amount, is not part of this reference: it does not change |S|). Per-wall matrices are
returned too.

``textbook_alpha`` gives the classical closed forms (e.g. Collin, *Field Theory of Guided Waves*; Pozar eq. 3.96 for
TE10) for an independent check of the integral.
"""
import numpy as np
import scipy.constants as cst

from wrmodes.modes import Mode, ModeBasis, transverse_field, wave_admittance, cutoff_wavenumber

ETA0 = cst.mu_0 * cst.c
SIGMA_AL = 3.5e7       # SPARC baseline aluminium (CONVENTIONS.md section 1)
SIGMA_CU = 5.8e7       # copper
NP_TO_DB = 20 / np.log(10)


def surface_resistance(f, sigma):
    """R_s = sqrt(omega mu0 / (2 sigma)) (ohm)."""
    return np.sqrt(2 * np.pi * f * cst.mu_0 / (2 * sigma))


def _curl_z(mode, a, b, u, v, h=None):
    """(curl e)_z = d e_v/du - d e_u/dv by 4th-order central differences of transverse_field."""
    h = 1e-4 * min(a, b) if h is None else h

    def ev(uu):
        return transverse_field(mode, a, b, uu, v)[1]

    def eu(vv):
        return transverse_field(mode, a, b, u, vv)[0]

    d_ev = (-ev(u + 2 * h) + 8 * ev(u + h) - 8 * ev(u - h) + ev(u - 2 * h)) / (12 * h)
    d_eu = (-eu(v + 2 * h) + 8 * eu(v + h) - 8 * eu(v - h) + eu(v - 2 * h)) / (12 * h)
    return d_ev - d_eu


def fields(mode, a, b, f, u, v):
    """(E_u, E_v, H_u, H_v, H_z) of the unit-amplitude forward mode at points (u, v)."""
    mode = Mode.parse(mode)
    Y = wave_admittance(mode, a, b, f)
    omega = 2 * np.pi * f
    eu, ev = transverse_field(mode, a, b, u, v)
    sE, sH = np.sqrt(2 / Y), np.sqrt(2 * Y)
    Hz = 1j / (omega * cst.mu_0) * sE * _curl_z(mode, a, b, u, v)
    return sE * eu, sE * ev, -sH * ev, sH * eu, Hz


def _walls(a, b, nq):
    x, w = np.polynomial.legendre.leggauss(nq)
    walls = {}
    for name, (uu, vv, dl, tang) in {
        "u-": (np.full(nq, -a / 2), x * b / 2, b / 2, "v"), "u+": (np.full(nq, a / 2), x * b / 2, b / 2, "v"),
        "v-": (x * a / 2, np.full(nq, -b / 2), a / 2, "u"), "v+": (x * a / 2, np.full(nq, b / 2), a / 2, "u"),
    }.items():
        walls[name] = (uu, vv, w * dl, tang)
    return walls


def power(mode, a, b, f, nq=64):
    """(1/2) Re int (E x H*) . z_hat ds by quadrature (1 for a propagating mode)."""
    x, w = np.polynomial.legendre.leggauss(nq)
    U, V = np.meshgrid(x * a / 2, x * b / 2, indexing="ij")
    W = np.outer(w * a / 2, w * b / 2)
    Eu, Ev, Hu, Hv, _ = fields(mode, a, b, f, U, V)
    return 0.5 * np.real(np.sum(W * (Eu * np.conj(Hv) - Ev * np.conj(Hu))))


def loss_matrix(basis, f, sigma, nq=96, per_wall=False):
    """A_pq = (1/4) R_s oint H_tan,p . conj(H_tan,q) dl / sqrt(P_p P_q) (Np/m) over the propagating modes of basis.

    Returns (A, names) or (A, names, {wall: A_wall}) with per_wall=True. diag(A) is alpha of every mode.
    """
    if not np.all(basis.propagating_mask(f)):
        raise ValueError("loss_matrix: only propagating modes")
    a, b = basis.a, basis.b
    Rs = surface_resistance(f, sigma)
    N = len(basis)
    P = np.array([power(m, a, b, f) for m in basis.modes])
    out = {}
    for name, (uu, vv, wq, tang) in _walls(a, b, nq).items():
        H = []
        for m in basis.modes:
            _, _, Hu, Hv, Hz = fields(m, a, b, f, uu, vv)
            H.append((Hv if tang == "v" else Hu, Hz))
        A = np.zeros((N, N))
        for p in range(N):
            for q in range(p, N):
                val = 0.25 * Rs * np.sum(wq * (H[p][0] * np.conj(H[q][0]) + H[p][1] * np.conj(H[q][1])))
                A[p, q] = A[q, p] = np.real(val) / np.sqrt(P[p] * P[q])
                assert abs(np.imag(val)) <= 1e-9 * max(abs(val), 1e-300) + 1e-300
        out[name] = A
    A = sum(out.values())
    return (A, basis.names, out) if per_wall else (A, basis.names)


def alpha(mode, a, b, f, sigma, nq=96):
    """Attenuation constant (Np/m) of one propagating mode, P_loss / (2 P), from the wall integral."""
    basis = ModeBasis(a, b, [Mode.parse(mode)])
    A, _ = loss_matrix(basis, f, sigma, nq=nq)
    return float(A[0, 0])


def textbook_alpha(mode, a, b, f, sigma):
    """Classical closed forms of alpha (Np/m) for TE_mn and TM_mn (m along a, n along b), for checking."""
    mode = Mode.parse(mode)
    m, n = mode.m, mode.n
    Rs = surface_resistance(f, sigma)
    fc = cutoff_wavenumber(mode, a, b) * cst.c / (2 * np.pi)
    r = (fc / f) ** 2
    s = np.sqrt(1 - r)
    if mode.kind == "TM":
        return 2 * Rs / (ETA0 * b * s) * (m ** 2 * b ** 3 + n ** 2 * a ** 3) / (m ** 2 * b ** 2 * a + n ** 2 * a ** 3)
    if n == 0:
        return Rs / (ETA0 * b * s) * (1 + 2 * b / a * r)
    if m == 0:
        return Rs / (ETA0 * a * s) * (1 + 2 * a / b * r)
    q = b / a
    return 2 * Rs / (ETA0 * b * s) * ((1 + q) * r + (1 - r) * (q * (q * m ** 2 + n ** 2)) / (q ** 2 * m ** 2 + n ** 2))


# ----------------------------------------------------------------------------------------- reference cases
GUIDES = {"WR-90": (0.9 * 0.0254, 0.4 * 0.0254), "square 20 mm": (0.020, 0.020)}
MATERIALS = {"aluminium": SIGMA_AL, "copper": SIGMA_CU}
FREQUENCIES = {"WR-90": [18e9, 30e9, 45e9, 60e9, 90e9], "square 20 mm": [12e9, 20e9, 30e9, 40e9]}


# ========================================================================================= coupled first order
# Derivation (WP7, independent of T1; agrees with T1 boxes B3-B4).
#
# Unconjugated reciprocity for the field (E, H) of the lossy guide and an ideal (PEC) test mode (E_q, H_q):
#     d/dz iint (E x H_q - E_q x H) . z ds = - oint (E x H_q) . n dl        (E_q,tan = 0 on the walls)
# and with Leontovich, E_tan = Z H_tan x n (n from the guide into the wall), (E x H_q) . n = Z H_tan . H_q,tan.
# Expanding E_t, H_t in the ideal modes (CONVENTIONS.md 5) and testing with the backward (forward) mode q gives,
# to first order in Z (wall field H_tan replaced by the modal series),
#     da+_q/dz = -i beta_q a+_q - sum_p (L+_qp a+_p + L-_qp a-_p)
#     da-_q/dz = +i beta_q a-_q + sum_p (L-_qp a+_p + L+_qp a-_p)
# with L+ = P - Q, L- = -(P + Q),
#     P_qp = (Z/4) oint h_t,q h_t,p dl,  h_t = sqrt(2Y) (e . n)                  (transverse wall current)
#     Q_qp = (Z/4) oint h_z,q h_z,p dl,  h_z = (i/(omega mu0)) sqrt(2/Y) (curl e)_z  (longitudinal wall current)
# (unconjugated products, principal square roots). For a propagating mode and Z = (1+i) R_s, L+_pp = (1+i) alpha_p:
# the attenuation of the first part of this module plus the same increase of beta.

def _wall_h(mode, a, b, f, wall, nq):
    """Gauss nodes' weights and (h_t, h_z) of a unit mode on one wall (functions above, no closed forms)."""
    (uu, vv, wq, tang) = _walls(a, b, nq)[wall]
    mode = Mode.parse(mode)
    Y = wave_admittance(mode, a, b, f)
    eu, ev = transverse_field(mode, a, b, uu, vv)
    en = {"u+": eu, "u-": -eu, "v+": ev, "v-": -ev}[wall]
    omega = 2 * np.pi * f
    ht = np.sqrt(2 * Y) * en
    hz = 1j / (omega * cst.mu_0) * np.sqrt(2 / Y) * _curl_z(mode, a, b, uu, vv)
    return wq, ht, hz


def first_order_matrices(basis, f, Z, nq=96):
    """(L+, L-) of the first-order coupled equations above for every mode of basis (Np/m, complex).

    Z: {wall: impedance (ohm)} for the walls 'u-', 'u+', 'v-', 'v+' (missing walls are perfect), or one number for
    all four walls.
    """
    if not isinstance(Z, dict):
        Z = {w: Z for w in ("u-", "u+", "v-", "v+")}
    a, b = basis.a, basis.b
    N = len(basis)
    P = np.zeros((N, N), complex)
    Q = np.zeros((N, N), complex)
    for wall, Zw in Z.items():
        if Zw == 0:
            continue
        hs = [_wall_h(m, a, b, f, wall, nq) for m in basis.modes]
        wq = hs[0][0]
        Ht = np.array([h[1] for h in hs])
        Hz = np.array([h[2] for h in hs])
        P += Zw / 4 * (Ht * wq) @ Ht.T
        Q += Zw / 4 * (Hz * wq) @ Hz.T
    return P - Q, -(P + Q)


def coupled_eigen(basis, f, Lp, Lm):
    """Eigenvalues lambda (fields ~ exp(lambda z)) and eigenvectors of the 2N system [[-iB - L+, -L-], [L-, iB + L+]]."""
    beta = basis.beta(f)
    M = np.block([[np.diag(-1j * beta) - Lp, -Lm], [Lm, np.diag(1j * beta) + Lp]])
    lam, V = np.linalg.eig(M)
    return lam, V


def straight_gsm_blocks(basis, f, Lp, Lm, L):
    """S11, S21 of a uniform lossy straight of length L from the first-order system, by its eigenmodes.

    Forward eigenmodes (Im lambda < 0) are normalized at z = 0, backward ones at z = L, so no exponential exceeds 1
    (well conditioned for any length). a+(0) = I, a-(L) = 0; returns R = a-(0), T = a+(L).
    """
    N = len(basis)
    lam, V = coupled_eigen(basis, f, Lp, Lm)
    fwd = np.argsort(lam.imag)[:N]                  # -i beta: most negative imaginary parts
    bwd = np.setdiff1d(np.arange(2 * N), fwd)
    Vf, Vb = V[:, fwd], V[:, bwd]
    ef, eb = np.exp(lam[fwd] * L), np.exp(-lam[bwd] * L)
    # y(0) = Vf c_f + Vb eb c_b ; y(L) = Vf ef c_f + Vb c_b
    A = np.block([[Vf[:N], Vb[:N] * eb[None, :]], [Vf[N:] * ef[None, :], Vb[N:]]])
    rhs = np.vstack([np.eye(N), np.zeros((N, N))])
    c = np.linalg.solve(A, rhs)
    cf, cb = c[:N], c[N:]
    T = (Vf[:N] * ef[None, :]) @ cf + Vb[:N] @ cb
    R = Vf[N:] @ cf + (Vb[N:] * eb[None, :]) @ cb
    return R, T


# ========================================================================================= exact Leontovich modes
def ls_wall_dispersion(ku, k, a, Zm, Zp, kind):
    """Dispersion function of the longitudinal-section modes with impedance walls Zm (u = -a/2), Zp (u = +a/2).

    Derived (WP7) from E = -curl(u_hat X(u') cos(k_n v') e^{-i gamma z}) (LSE^u, E_u = 0) and
    H = curl(u_hat X(u') sin(k_n v') e^{-i gamma z}) (LSM^u, H_u = 0) with Leontovich on both u walls:
    X'(0) = q_- X(0), X'(a) = -q_+ X(a), q = i k eta0 / Z (LSE^u) or q = i k Z / eta0 (LSM^u). With
    X = cos(k_u u') + (q_-/k_u) sin(k_u u'):  F = (q_- q_+ - k_u^2) sin(k_u a) + k_u (q_- + q_+) cos(k_u a) = 0,
    and gamma^2 = k^2 - k_n^2 - k_u^2. Z = 0 (perfect wall) is handled through 1/q (written as F / (q_- q_+)).
    """
    if kind == "LSE":
        r = [Z / (1j * k * ETA0) for Z in (Zm, Zp)]           # r = 1/q
    else:
        r = [np.inf if Z == 0 else ETA0 / (1j * k * Z) for Z in (Zm, Zp)]
    rm, rp = r
    if np.isinf(rm) or np.isinf(rp):
        raise ValueError("perfect wall in LSM form: use the LSE form (Z = 0 there) or a tiny Z")
    s, c = np.sin(ku * a), np.cos(ku * a)
    return (1 - ku * ku * rm * rp) * s + ku * (rm + rp) * c


def exact_gamma(f, a, b, m, n, kind, Zm, Zp, tol=1e-15):
    """Exact propagation constant gamma (fields ~ exp(-i gamma z), alpha = -Im gamma) of the LS^u mode that tends
    to (m, n) when Z -> 0, with impedance on the u walls only. Newton from k_u = m pi / a (secant derivative)."""
    k = 2 * np.pi * f / cst.c
    ku = m * np.pi / a + 0j
    if m == 0:
        if kind != "LSM":
            raise ValueError("LSE^u modes need m >= 1")
        # small root: (1 - k_u^2 r_- r_+) a + r_- + r_+ ~ 0 with large r = eta0 / (i k Z)
        r = 0.5 * sum(ETA0 / (1j * k * Z) for Z in (Zm, Zp))
        ku = np.sqrt(2 / (r * a) + 0j)
    F = lambda x: ls_wall_dispersion(x, k, a, Zm, Zp, kind) / (x if m == 0 else 1)  # noqa: E731
    for _ in range(100):
        h = 1e-7 * (abs(ku) + 1 / a)
        d = (F(ku + h) - F(ku - h)) / (2 * h)
        step = F(ku) / d
        ku -= step
        if abs(step) < tol * (abs(ku) + 1 / a):
            break
    g2 = k * k - (n * np.pi / b) ** 2 - ku * ku
    g = np.sqrt(g2 + 0j)
    return -g if g.imag > 0 else g, ku


def degenerate_groups(basis, f, rtol=1e-9):
    """Index groups of modes with equal beta (TE_mn/TM_mn; also TE_mn/TE_nm in a square guide)."""
    beta = basis.beta(f).real
    groups, used = [], set()
    for i in range(len(basis)):
        if i in used:
            continue
        g = [j for j in range(len(basis)) if j not in used and abs(beta[j] - beta[i]) <= rtol * beta[i]]
        used.update(g)
        groups.append(g)
    return groups


def one_pair_comparison(a, b, f, sigma, walls="u"):
    """Exact Leontovich attenuation vs three first-order approximations, impedance (1+i) R_s on one pair of walls.

    walls 'u': the walls u = +-a/2; 'v': the walls v = +-b/2 (computed in the frame with a and b swapped, so mode
    names refer to that frame). For every degenerate group G of propagating modes, returns a dict with the exact
    alphas (LSE^u for TE, LSM^u for TM; LSM^u with m = 0 for TE_0n) and the relative errors (max over G) of
    * 'coupled': eigenvalues of the full first-order system (all propagating modes, forward and backward);
    * 'group': eigenvalues of the group's block of L+ only (the WP7 reference before the 2026-11-08 resolution);
    * 'group_back': the group's forward and backward blocks (L+ and L-), no coupling to other groups.
    """
    aa, bb = (a, b) if walls == "u" else (b, a)
    basis = ModeBasis.propagating(f, aa, bb)
    Zs = (1 + 1j) * surface_resistance(f, sigma)
    Lp, Lm = first_order_matrices(basis, f, {"u-": Zs, "u+": Zs})
    lam, _ = coupled_eigen(basis, f, Lp, Lm)
    beta = basis.beta(f).real
    k = 2 * np.pi * f / cst.c
    rows = []
    for G in degenerate_groups(basis, f):
        ex = []
        for i in G:
            m = basis.modes[i]
            kind = "LSM" if (m.kind == "TM" or m.m == 0) else "LSE"
            g, _ = exact_gamma(f, aa, bb, m.m, m.n, kind, Zs, Zs)
            ex.append(-g.imag)
        ex = np.sort(ex)
        near = np.argsort(np.abs(lam + 1j * beta[G[0]]))[:len(G)]
        co = np.sort(-lam[near].real)
        blk, bG = Lp[np.ix_(G, G)], beta[G]
        gr = np.sort(np.linalg.eigvals(blk).real)
        M = np.block([[np.diag(-1j * bG) - blk, -Lm[np.ix_(G, G)]], [Lm[np.ix_(G, G)], np.diag(1j * bG) + blk]])
        l2 = np.linalg.eigvals(M)
        gb = np.sort(-l2[np.argsort(np.abs(l2 + 1j * bG[0]))[:len(G)]].real)
        rows.append(dict(modes=[basis.modes[i].name for i in G], fc_over_f=float(np.sqrt(1 - (beta[G[0]] / k) ** 2)),
                         exact=ex, coupled=float(np.max(np.abs(co / ex - 1))),
                         group=float(np.max(np.abs(gr / ex - 1))), group_back=float(np.max(np.abs(gb / ex - 1)))))
    return rows
