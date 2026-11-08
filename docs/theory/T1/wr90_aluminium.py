"""WR-90 with aluminium walls (sigma = 3.5e7 S/m), 18-90 GHz: the numbers of sections 5 and 7 of the T1 note.

A  attenuation constants (diag Lambda^+) of the low-order modes;
B  degenerate TE_mn / TM_mn pairs: loss eigenmodes, mixing angle, TE -> TM transfer over 1 m and 20 m;
C  non-matched pairs: forward (|Lambda^+_pq| / |beta_p - beta_q|) and backward (|Lambda^-_pq| / (beta_p + beta_q))
   amplitude bounds, conversion of TE10 over a 20 m straight line;
D  validity of the first-order coupled system: TE10 / TE01 eigenvalue of the full propagating system vs the
   diagonal and vs the exact impedance-wall LS modes (two lossy walls);
E  bends (SPARC: 40 cm 90 deg sech, 60 cm 60 deg constant): effect of Lambda0 and of the kappa Z cross term
   kappa Lambda1, H-plane (TE10) and E-plane (TE01 local) orientations.

Run:  PYTHONPATH=<worktree>/src python3 docs/theory/T1/wr90_aluminium.py
"""
import sys
import numpy as np
from scipy.linalg import expm

import t1lib as L
from wrmodes.modes import ModeBasis, Mode
from wrmodes.transfer_matrix import coupled_mode_matrices, _slice_exponentials, _product, _scattering_from_transfer

SIGMA = 3.5e7
A_, B_ = 22.86e-3, 10.16e-3
DB = 20 / np.log(10)
FREQS = (18e9, 30e9, 45e9, 60e9, 75e9, 90e9)
np.set_printoptions(linewidth=150)

# --------------------------------------------------------------------------------------------- A
print("A  attenuation constants, aluminium WR-90 (dB/m)")
modesA = [("TE", 1, 0), ("TE", 2, 0), ("TE", 0, 1), ("TE", 1, 1), ("TM", 1, 1), ("TE", 3, 0), ("TE", 0, 2), ("TE", 1, 2), ("TM", 1, 2)]
print("     f (GHz) " + " ".join(f"{'%s%d,%d' % m:>8s}" for m in modesA) + "   max over all propagating modes")
for f in FREQS:
    B = ModeBasis.propagating(f, A_, B_)
    Lp, _ = L.loss_matrices(B, f, L.all_walls(L.surface_impedance(f, SIGMA)))
    al = Lp.diagonal().real * DB
    row = []
    for m in modesA:
        row.append(f"{al[B.index(m)]:8.4f}" if Mode(*m) in B else f"{'-':>8s}")
    imax = np.argmax(al)
    print(f"     {f / 1e9:6.0f}  " + " ".join(row) + f"   {al[imax]:.3f} ({B.modes[imax].name})")
print("     (Im Lambda^+_pp = Re Lambda^+_pp: the phase constant increases by alpha, in rad/m)")

# --------------------------------------------------------------------------------------------- B
print("\nB  degenerate pairs TE_mn / TM_mn (m, n >= 1), all four walls aluminium")
print("     f  pair     a_TE    a_TM   (dB/m)  Lambda_EM/(a_TM-a_TE)  loss-eigenmode angle  LS^v angle  LS^u angle   TE->TM power: 1 m     20 m")
worst_transfer = (0, None)
for f in FREQS:
    B = ModeBasis.propagating(f, A_, B_)
    k = L.wavenumber(f)
    Zs = L.surface_impedance(f, SIGMA)
    Lp, _ = L.loss_matrices(B, f, L.all_walls(Zs))
    shown = 0
    for i, p in enumerate(B.modes):
        if p.kind != "TE" or p.m == 0 or p.n == 0:
            continue
        j = B.index(Mode("TM", p.m, p.n))
        blk = Lp[np.ix_([i, j], [i, j])]
        bt = (blk / Zs).real                     # real symmetric for Z = (1+i) Rs
        w, V = np.linalg.eigh(bt)
        vec = V[:, 0]                            # lower-loss eigenmode
        ang = np.degrees(np.arctan2(vec[1], vec[0]))
        beta = np.sqrt(k * k - (p.m * np.pi / A_) ** 2 - (p.n * np.pi / B_) ** 2)
        phiv = np.degrees(np.arctan2(p.m * B_ * k, p.n * A_ * beta))    # LSE^v = cos TE - sin TM (CONVENTIONS 7)
        phiu = np.degrees(np.arctan2(p.n * A_ * k, p.m * B_ * beta))    # by symmetry u <-> v
        out = []
        for Lz in (1.0, 20.0):
            T = expm(-blk * Lz)
            out.append(abs(T[1, 0]) ** 2 / (abs(T[0, 0]) ** 2 + abs(T[1, 0]) ** 2))   # fraction of the surviving power
        if out[1] > worst_transfer[0] and beta / k > 0.5:
            worst_transfer = (out[1], (f, p.m, p.n))
        if shown < 3:
            print(f"   {f / 1e9:4.0f}  ({p.m},{p.n})  {blk[0, 0].real * DB:7.4f} {blk[1, 1].real * DB:7.4f}"
                  f"      {abs(blk[0, 1]) / abs(blk[1, 1] - blk[0, 0]):8.3f}              {ang % 180 - 180 if ang % 180 > 90 else ang % 180:7.1f} deg"
                  f"        {-phiv:7.1f}      {phiu:7.1f}      {out[0]:.2e}  {out[1]:.2e}")
            shown += 1
print(f"     largest TE_mn -> TM_mn power fraction after 20 m (pure TE_mn launched, straight guide, beta/k > 0.5): "
      f"{worst_transfer[0]:.2e} at {worst_transfer[1][0] / 1e9:.0f} GHz, (m, n) = {worst_transfer[1][1:]}")

# --------------------------------------------------------------------------------------------- C
print("\nC  non-matched pairs (all propagating modes, all families)")
for f in FREQS:
    B = ModeBasis.propagating(f, A_, B_)
    beta = B.beta(f).real
    Zs = L.surface_impedance(f, SIGMA)
    Lp, Lm = L.loss_matrices(B, f, L.all_walls(Zs))
    n = len(B)
    deg = np.array([[p.m == q.m and p.n == q.n for q in B.modes] for p in B.modes])
    dB = np.abs(beta[:, None] - beta[None, :])
    fwd = np.where(deg, 0, np.abs(Lp) / np.where(deg, 1, dB))
    i, j = np.unravel_index(np.argmax(fwd), fwd.shape)
    bwd = np.abs(Lm) / (beta[:, None] + beta[None, :])
    i2, j2 = np.unravel_index(np.argmax(bwd), bwd.shape)
    cross = np.array([[p.n != q.n for q in B.modes] for p in B.modes])
    fwd_cross = np.max(np.where(cross, fwd, 0))
    # 20 m straight line, TE10 input, forward system (backward coupling bounded separately)
    D = np.diag(-1j * beta)
    T = expm((D - Lp) * 20.0)
    i10 = B.index(Mode("TE", 1, 0))
    col = np.abs(T[:, i10]) ** 2
    conv = np.delete(col, i10)
    jc = np.argmax(conv)
    other = [m for k_, m in enumerate(B.modes) if k_ != i10][jc]
    bwd10 = np.max(bwd[B.index(Mode("TE", 1, 0))])
    fc_i2 = L.cutoff_wavenumber(B.modes[i2], A_, B_) / L.wavenumber(f)
    print(f"   {f / 1e9:4.0f} GHz: TE10 backward max {bwd10:.1e}; worst-backward mode at f_c/f = {fc_i2:.4f}")
    print(f"   {f / 1e9:4.0f} GHz: max |L+_pq|/|dbeta| = {fwd[i, j]:.1e} ({B.modes[i].name}-{B.modes[j].name}); "
          f"across families {fwd_cross:.1e}; max |L-_pq|/(b_p+b_q) = {bwd[i2, j2]:.1e} ({B.modes[i2].name}-{B.modes[j2].name}); "
          f"TE10, 20 m: total converted {conv.sum():.1e}, largest {conv.max():.1e} ({other.name})")

# --------------------------------------------------------------------------------------------- D
print("\nD  first-order coupled system vs diagonal vs exact (relative error of alpha, two lossy walls only)")
for f in (45e9, 90e9):
    k = L.wavenumber(f)
    Zs = L.surface_impedance(f, SIGMA)
    for (m, n, kind, fam, walls) in ((1, 0, "LSE", 0, ("u+", "u-")), (0, 1, "LSM", 1, ("u+", "u-")),
                                     (1, 0, "LSM", 0, ("v+", "v-"))):
        Z = {w: Zs for w in walls}
        kn = n * np.pi / B_
        g0 = np.sqrt(k * k - kn * kn - (m * np.pi / A_) ** 2)
        if walls[0] == "u+":
            ku = L.lsu_roots_all(m + 2, k, A_, Zs, Zs, kind)[m if kind == "LSM" else m - 1]
            lamx = 1j * (L.gamma_from_ku(ku, k, kn) - g0)
            lam1 = L.ls_first_order(m, n, A_, B_, f, Z)[kind + "_u"]
            label = f"{kind}^u({m},{n}) side walls"
        else:
            # broad walls: the same problem with the roles of (u, a, m) and (v, b, n) exchanged
            kv = L.lsu_roots_all(n + 2, k, B_, Zs, Zs, kind)[n if kind == "LSM" else n - 1]
            lamx = 1j * (L.gamma_from_ku(kv, k, m * np.pi / A_) - g0)
            lam1 = L.ls_first_order(m, n, A_, B_, f, Z)[kind + "_v"]
            label = f"{kind}^v({m},{n}) broad walls"
        res = []
        for cf in (1.0, 3.0):
            Bc = ModeBasis.propagating(f, A_, B_, families=None if walls[0] == "v+" else [fam], cutoff_factor=cf)
            if walls[0] == "v+":
                Bc = ModeBasis(A_, B_, [q for q in Bc.modes if q.m == m], frame=None)
            ev = np.linalg.eigvals(L.D_matrix(Bc, f) - L.loss_matrix_full(Bc, f, Z))
            lam = -(ev[np.argmin(abs(ev + 1j * g0))] + 1j * g0)
            res.append((len(Bc), lam.real / lamx.real - 1))
        print(f"   {f / 1e9:4.0f} GHz {label:28s}: diagonal {lam1.real / lamx.real - 1:+.1e}; coupled, propagating "
              f"({res[0][0]} modes) {res[0][1]:+.1e}; coupled, cutoff_factor 3 ({res[1][0]} modes) {res[1][1]:+.1e}")

# --------------------------------------------------------------------------------------------- E
print("\nE  bends with wall loss (one family; Magnus product with fixed slices)")


def sech_curvature(theta_deg, length, T=4.0):
    theta = np.radians(theta_deg)
    gamma = 2 * T / length
    R0 = 4 / theta / gamma * np.arctan(np.tanh(T / 2))
    return (lambda z: 1 / (R0 * np.cosh(gamma * (np.asarray(z) - length / 2)))), R0


def const_curvature(theta_deg, length):
    R0 = length / np.radians(theta_deg)
    return (lambda z: np.ones_like(np.asarray(z, dtype=float)) / R0), R0


def bend_T(B, f, length, kappa, Z, mode, use_l0, use_l1, nsl=1024):
    D, K, _ = coupled_mode_matrices(f, B.a, B.b, B.legacy_modes(f))
    Dm, Km = D.copy(), K.copy()
    if use_l0:
        Dm = Dm - L.loss_matrix_full(B, f, Z)
    if use_l1:
        Lp1, Lm1 = L.loss_matrices(B, f, Z, order=1)
        Km = Km - np.block([[Lp1, Lm1], [-Lm1, -Lp1]])
    edges = np.linspace(0, length, nsl + 1)
    phi = _product(_slice_exponentials(Dm, Km, kappa, edges))
    T, R = _scattering_from_transfer(phi, len(B))
    i = B.index(mode)
    return T[:, i], R[:, i]


cases = (("40 cm 90 deg sech", sech_curvature(90, 0.40), 0.40), ("60 cm 60 deg const", const_curvature(60, 0.60), 0.60))
print("     bend                 orient.   f   kappa_max a/2   |kappa Lambda1|/|Lambda0|  |Lambda1|/|K|   "
      "TE in: loss lossless / +Lambda0 / +kappa Lambda1 (dB);   change by kappa Lambda1: in-mode dB, max converted power")
for name, (kap, R0), length in cases:
    for orient, (a, b, mode, fam) in (("H-plane", (A_, B_, Mode("TE", 1, 0), 0)), ("E-plane", (B_, A_, Mode("TE", 0, 1), 1))):
        for f in (18e9, 45e9, 90e9):
            B = ModeBasis.propagating(f, a, b, families=[fam])
            Z = L.all_walls(L.surface_impedance(f, SIGMA))
            Lp0, _ = L.loss_matrices(B, f, Z, 0)
            Lp1, _ = L.loss_matrices(B, f, Z, 1)
            _, K, _ = coupled_mode_matrices(f, a, b, B.legacy_modes(f))
            T0, _ = bend_T(B, f, length, kap, Z, mode, False, False)
            T1, R1 = bend_T(B, f, length, kap, Z, mode, True, False)
            T2, R2 = bend_T(B, f, length, kap, Z, mode, True, True)
            T2b, _ = bend_T(B, f, length, kap, Z, mode, True, True, nsl=2048)
            i = B.index(mode)
            l0, l1, l2 = (-10 * np.log10(abs(T[i]) ** 2) for T in (T0, T1, T2))
            dconv = np.max(np.abs(np.abs(np.delete(T2, i)) ** 2 - np.abs(np.delete(T1, i)) ** 2))
            assert np.max(np.abs(T2b - T2)) < 1e-8
            print(f"     {name:20s} {orient}  {f / 1e9:3.0f}   {a / 2 / R0:.3f}          {np.max(np.abs(Lp1)) / R0 / np.max(np.abs(Lp0)):.2e}"
                  f"             {np.max(np.abs(Lp1)) / np.max(np.abs(K)):.1e}     {l0:.4f} / {l1:.4f} / {l2:.4f}        "
                  f"{l2 - l1:+.1e}       {dconv:.1e}")
