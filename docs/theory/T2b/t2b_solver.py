"""Reference solver of the taper equations that stays regular and stable through cutoff (box B5 of the note).

Method (card T2, item 3):
1. Integrate the V-I system dy/dz = M(z) y, y = [V; I] (box B1), which contains no beta and no sqrt(Y), over
   short slices with the 4th-order Magnus integrator (two Gauss points). Each slice is short enough that
   |beta| dz <= `growth` for every mode, so the slice transfer matrix has no large exponentials.
2. Express every slice in pseudo-waves with a constant, real, positive reference admittance G (here 1/eta0 for every
   mode): V = sqrt2 (al+ + al-)/sqrt G, I = sqrt2 sqrt G (al+ - al-). With a real reference, |al+|^2 - |al-|^2 is
   the exact power, so the slice S matrices of a lossless taper are unitary for propagating and evanescent modes
   alike and the Redheffer cascade is unconditionally stable. Nothing is singular at a cutoff plane.
3. At the two ports, join the pseudo-waves to the modal waves of CONVENTIONS.md with the junction
   S11 = (Y - G)/(Y + G), S21 = S12 = 2 sqrt(Y) sqrt(G)/(Y + G), S22 = -S11 (per mode, Y = modal admittance at the
   port; requires no mode exactly at cutoff at a port).

geom(z) returns dict(a, b, da, db, u0, du0, dv0, kappa). For kappa = 0 the closed-form diagonal Z, Y are used; for
kappa != 0 the curvature integrals are evaluated by quadrature (slow, for checks only).
"""
import numpy as np
import scipy.constants as cst
from scipy.linalg import expm

from wrmodes.modes import Mode, ETA0
from t2b_formulas import T_block, ZY, chi2, beta_Y


def outside_modes(modes, m_tail):
    """Modes coupled to the basis by the taper but not in it: same n (u-walls) or same m (v-walls), index <= m_tail."""
    inside = set(modes)
    out = set()
    for p in modes:
        for j in range(m_tail + 1):
            for kd in ("TE", "TM"):
                for (m, n) in ((j, p.n), (p.m, j)):
                    if (kd == "TE" and m + n > 0) or (kd == "TM" and m * n > 0):
                        q = Mode(kd, m, n)
                        if q not in inside:
                            out.add(q)
    return sorted(out, key=lambda q: (q.kind, q.m, q.n))


def tail_correction(modes, f, g, out):
    """Quasi-static elimination of the modes outside the basis (box B5): Y_eff = Y - T_out^T Z_out^-1 T_out.

    For a strongly evanescent mode q, dV_q/dz is negligible, so I_q = sum_p T_qp V_p / (i Z_qq); fed back into
    dI_p/dz = ... - sum_q T_qp I_q this adds  +i sum_q T_qp T_qp' / Z_qq  V_p'. For TE modes Z_qq = w mu0 does not grow
    with q, which is why the plain truncation converges only as 1/M; the correction restores fast convergence.
    """
    k = 2 * np.pi * f / cst.c
    a, b = g["a"], g["b"]
    if not out:
        return 0.0
    Tout = T_block(out, modes, a, b, g["da"], g["db"], g.get("du0", 0.0), g.get("dv0", 0.0))
    Zq = np.array([k * ETA0 if q.kind == "TE" else (k * k - chi2(q, a, b)) * ETA0 / k for q in out])
    return Tout.T @ (Tout / Zq[:, None])


def system_matrix(modes, f, g, out=None):
    k = 2 * np.pi * f / cst.c
    a, b = g["a"], g["b"]
    T = T_block(modes, modes, a, b, g["da"], g["db"], g.get("du0", 0.0), g.get("dv0", 0.0))
    if g.get("kappa", 0.0) == 0.0:
        c2 = np.array([chi2(p, a, b) for p in modes])
        te = np.array([p.kind == "TE" for p in modes])
        Z = np.diag(np.where(te, k * ETA0, (k * k - c2) * ETA0 / k))
        Y = np.diag(np.where(te, (k * k - c2) / (k * ETA0), k / ETA0))
    else:
        Z, Y = ZY(modes, f, a, b, g.get("u0", 0.0), g["kappa"], nq=max(48, 3 * max(max(p.m, p.n) for p in modes) + 8))
    if out:
        Y = Y - tail_correction(modes, f, g, out)
    return np.block([[T, -1j * Z], [-1j * Y, -T.T]])


def star(A, B):
    """Redheffer star product of S matrices given as 4-tuples (S11, S12, S21, S22)."""
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = B
    I = np.eye(A22.shape[0])
    X1 = np.linalg.solve(I - B11 @ A22, np.hstack([B11 @ A21, B12]))
    X2 = np.linalg.solve(I - A22 @ B11, np.hstack([A21, A22 @ B12]))
    n1, n2 = A21.shape[1], B12.shape[1]
    S11 = A11 + A12 @ X1[:, :n1]
    S12 = A12 @ X1[:, n1:]
    S21 = B21 @ X2[:, :n1]
    S22 = B22 + B21 @ X2[:, n1:]
    return S11, S12, S21, S22


def transfer_to_S(P):
    N = P.shape[0] // 2
    P11, P12, P21, P22 = P[:N, :N], P[:N, N:], P[N:, :N], P[N:, N:]
    S11 = -np.linalg.solve(P22, P21)
    S12 = np.linalg.inv(P22)
    return S11, S12, P11 + P12 @ S11, P12 @ S12


def junction(Y1, Y2):
    s1, s2 = np.sqrt(Y1 + 0j), np.sqrt(Y2 + 0j)
    d = Y1 + Y2
    return np.diag((Y1 - Y2) / d), np.diag(2 * s1 * s2 / d), np.diag(2 * s1 * s2 / d), np.diag((Y2 - Y1) / d)


def solve(modes, f, geom, L, growth=1.0, dz_max=None, return_parts=False, m_tail=0):
    """GSM blocks (S11, S12, S21, S22) of the taper 0 <= z <= L in the modal waves of CONVENTIONS.md."""
    modes = [Mode.parse(p) for p in modes]
    N = len(modes)
    k = 2 * np.pi * f / cst.c
    # slice length from the largest |beta| anywhere (sampled)
    zs = np.linspace(0, L, 41)
    bmax = max(np.max(np.abs(beta_Y(modes, f, geom(z)["a"], geom(z)["b"])[0])) for z in zs)
    dz = min(growth / bmax, dz_max or np.inf, 2 * np.pi / k / 8)
    ns = int(np.ceil(L / dz))
    edges = np.linspace(0, L, ns + 1)
    G = np.full(N, 1 / ETA0)
    gs, gi = np.sqrt(G), 1 / np.sqrt(G)
    Q = np.sqrt(2) * np.block([[np.diag(gi), np.diag(gi)], [np.diag(gs), -np.diag(gs)]])
    Qi = np.linalg.inv(Q)
    c = np.sqrt(3) / 6
    out = outside_modes(modes, m_tail) if m_tail else None
    S = None
    for z0, z1 in zip(edges[:-1], edges[1:]):
        h = z1 - z0
        A1 = system_matrix(modes, f, geom(z0 + (0.5 - c) * h), out)
        A2 = system_matrix(modes, f, geom(z0 + (0.5 + c) * h), out)
        Om = 0.5 * h * (A1 + A2) + (np.sqrt(3) / 12) * h * h * (A2 @ A1 - A1 @ A2)
        Ss = transfer_to_S(Qi @ expm(Om) @ Q)
        S = Ss if S is None else star(S, Ss)
    g0, gL = geom(0.0), geom(L)
    Y0 = beta_Y(modes, f, g0["a"], g0["b"])[1]
    YL = beta_Y(modes, f, gL["a"], gL["b"])[1]
    J1 = junction(Y0, G)
    J2 = junction(G, YL)
    out = star(star(J1, S), J2)
    if return_parts:
        return out, ns
    return out


def full(S):
    return np.block([[S[0], S[1]], [S[2], S[3]]])
