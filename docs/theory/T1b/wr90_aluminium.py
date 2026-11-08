"""Item 3 and item 4 numbers of T1b: WR-90, aluminium (sigma = 3.5e7 S/m), 18-90 GHz.

Run from the repository root:   PYTHONPATH=src python3 docs/theory/T1b/wr90_aluminium.py

Item 3
  * TE10 attenuation;
  * degenerate pairs (TE_mn, TM_mn): the 2x2 block of Lambda^{++}, its eigenvalues and eigenvectors (mixing angle),
    and the TE_mn -> TM_mn power converted over 1 m and 20 m of straight guide;
  * non-matched pairs: the largest forward (4 |L++_pq|^2 / (beta_p - beta_q)^2) and backward
    (4 |L+-_pq|^2 / (beta_p + beta_q)^2) conversion bounds, over all pairs of propagating modes of all families.
Item 4
  * the SPARC bends (sech profile, T = 4: 60 deg over 60 cm and 90 deg over 40 cm; H plane, family n = 0, and
    E plane, local family n = 1 with a = 10.16 mm): transmission with D + kappa K - Lambda_full, with and without
    the cross term kappa Lambda1_full.
"""
import os
import sys
import itertools
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1b_coefficients as T  # noqa: E402
from wrmodes.transfer_matrix import coupled_mode_matrices, select_coupled_modes  # noqa: E402

SIGMA = 3.5e7
W, H = 22.86e-3, 10.16e-3
DB = 20 * np.log10(np.e)


def prop_modes(f, a, b):
    out = []
    k = T.wavenumber(f)
    for m in range(0, 20):
        for n in range(0, 20):
            for kind in ("TE", "TM"):
                if (kind == "TE" and m + n == 0) or (kind == "TM" and m * n == 0):
                    continue
                p = (kind, m, n)
                if T.cutoff(p, a, b) < k:
                    out.append(p)
    return out


def item3():
    print("ITEM 3 -- WR-90, aluminium")
    a, b = W, H
    freqs = np.linspace(18e9, 90e9, 73)
    print("  TE10 attenuation (dB/m):", ", ".join(
        f"{f/1e9:.0f} GHz {DB*T.lambda_pp(('TE',1,0),('TE',1,0),a,b,f,T.surface_impedance(f,SIGMA)).real:.4f}"
        for f in (18e9, 30e9, 45e9, 60e9, 75e9, 90e9)))
    # range of attenuation of the propagating modes at 90 GHz, excluding those within 2 % of cutoff
    f = 90e9
    Zs = T.surface_impedance(f, SIGMA)
    al = [(DB * T.lambda_pp(p, p, a, b, f, Zs).real, p) for p in prop_modes(f, a, b)
          if T.beta(p, a, b, f).real > 0.2 * T.wavenumber(f)]
    print(f"  90 GHz: {len(prop_modes(f, a, b))} propagating modes; attenuation of those with beta > 0.2 k:"
          f" {min(al)[0]:.3f} dB/m ({min(al)[1]}) to {max(al)[0]:.3f} dB/m ({max(al)[1]})")

    print("  Degenerate pairs (TE_mn, TM_mn), all four walls aluminium:")
    print("    f GHz  pair    a_TE dB/m  a_TM dB/m  |L_TE,TM| dB/m  eig1 dB/m  eig2 dB/m  mix deg  "
          "P(TE->TM) 1 m   P(TE->TM) 20 m  P_TM/P_TE 20 m")
    rows = []
    worst_pair = (0, None)
    for f in freqs:
        Zs = T.surface_impedance(f, SIGMA)
        for p in prop_modes(f, a, b):
            if p[0] != "TE" or p[1] == 0 or p[2] == 0:
                continue
            pE, pM = p, ("TM", p[1], p[2])
            Lb = np.array([[T.lambda_pp(x, y, a, b, f, Zs) for y in (pE, pM)] for x in (pE, pM)])
            w, V = np.linalg.eig(Lb)
            order = np.argsort(w.real)
            w, V = w[order], V[:, order]
            # mixing angle of the less attenuated eigenvector relative to pure TE or pure TM
            v = V[:, 0] / np.linalg.norm(V[:, 0])
            mix = np.degrees(np.arctan2(min(abs(v[0]), abs(v[1])), max(abs(v[0]), abs(v[1]))))
            P1 = abs(expm(-Lb * 1.0)[1, 0]) ** 2
            P20m = expm(-Lb * 20.0)
            P20, ratio20 = abs(P20m[1, 0]) ** 2, abs(P20m[1, 0]) ** 2 / abs(P20m[0, 0]) ** 2
            beta = T.beta(pE, a, b, f).real
            rows.append((f, p, Lb, w, mix, P1, P20, ratio20, beta))
            if abs(Lb[0, 1]) > worst_pair[0]:
                worst_pair = (abs(Lb[0, 1]), (f, p))
    for f in (18e9, 30e9, 45e9, 60e9, 90e9):
        sel = [r for r in rows if abs(r[0] - f) < 1 and r[8] > 0.2 * T.wavenumber(f)][:4]
        for (f_, p, Lb, w, mix, P1, P20, r20, bt) in sel:
            print(f"    {f_/1e9:5.0f}  {p[1]},{p[2]}    {DB*Lb[0,0].real:8.4f}   {DB*Lb[1,1].real:8.4f}"
                  f"   {DB*abs(Lb[0,1]):10.4f}     {DB*w[0].real:8.4f}   {DB*w[1].real:8.4f}  {mix:6.1f}"
                  f"   {P1:.2e}        {P20:.2e}        {r20:.2e}")
    allmix = [r[4] for r in rows if r[8] > 0.2 * T.wavenumber(r[0])]
    sel = [r for r in rows if r[8] > 0.2 * T.wavenumber(r[0])]
    print(f"    over 18-90 GHz (pairs with beta > 0.2 k): mixing angle {min(allmix):.1f} to {max(allmix):.1f} deg;"
          f" max P(TE->TM) after 1 m {max(r[5] for r in sel):.2e}, after 20 m {max(r[6] for r in sel):.2e};"
          f" max P_TM/P_TE after 20 m {max(r[7] for r in sel):.2e}")
    print(f"    largest |Lambda_TE,TM|: {DB*worst_pair[0]:.4f} dB/m for TE/TM{worst_pair[1][1][1]},"
          f"{worst_pair[1][1][2]} at {worst_pair[1][0]/1e9:.0f} GHz")

    print("  Non-matched pairs (all families, all propagating modes, beta > 0.05 k):")
    print("    f GHz  N   max 4|L++|^2/dbeta^2 (pair)                 max 4|L+-|^2/(beta_p+beta_q)^2")
    out = []
    for f in (18e9, 30e9, 45e9, 60e9, 75e9, 90e9):
        Zs = T.surface_impedance(f, SIGMA)
        k = T.wavenumber(f)
        ms = [p for p in prop_modes(f, a, b) if T.beta(p, a, b, f).real > 0.05 * k]
        bt = {p: T.beta(p, a, b, f).real for p in ms}
        fw, bw = (0, None), (0, None)
        for p, q in itertools.combinations(ms, 2):
            if p[1:] == q[1:]:
                continue                           # degenerate pair, treated above
            Lpp = T.lambda_pp(p, q, a, b, f, Zs)
            Lpm = T.lambda_pm(p, q, a, b, f, Zs)
            x = 4 * abs(Lpp) ** 2 / (bt[p] - bt[q]) ** 2
            y = 4 * abs(Lpm) ** 2 / (bt[p] + bt[q]) ** 2
            if x > fw[0]:
                fw = (x, (p, q))
            if y > bw[0]:
                bw = (y, (p, q))
        # self-coupling forward -> backward of a mode (impedance change of the lossy guide)
        sb = max(abs(T.lambda_pm(p, p, a, b, f, Zs)) / (2 * bt[p]) for p in ms)
        out.append((f, fw, bw, sb))
        lab = "none coupled" if fw[1] is None else f"{fw[1][0]}-{fw[1][1]}"
        print(f"    {f/1e9:5.0f} {len(ms):3d}  {fw[0]:.2e} ({lab})"
              f"    {bw[0]:.2e};  max |L+-_pp|/(2 beta_p) = {sb:.1e}")
    # validity near cutoff: |Lambda_pp| / |beta_p|
    print("  First order needs |Lambda_pp| << |beta_p|: smallest beta/k at which |Lambda_pp|/beta = 1e-2, TE10, 18 GHz:")
    f = 18e9
    Zs = T.surface_impedance(f, SIGMA)
    # Lambda_pp ~ C / beta near cutoff; with C = Lambda_pp * beta (evaluated at the actual frequency)
    for p in (("TE", 1, 0), ("TE", 2, 0), ("TM", 1, 1)):
        fc = T.cutoff(p, a, b) * 3e8 / (2 * np.pi)
        ff = fc * 1.5
        C = abs(T.lambda_pp(p, p, a, b, ff, T.surface_impedance(ff, SIGMA))) * abs(T.beta(p, a, b, ff))
        bmin = np.sqrt(C / 1e-2)
        print(f"    {p}: |Lambda_pp| beta ~ {C:.3f} 1/m^2 -> beta > {bmin:.1f} 1/m, i.e. f/f_c - 1 > "
              f"{(bmin / T.cutoff(p, a, b))**2 / 2:.1e}")
    return rows, out


def truncation_artifact():
    """The coupled system in the PEC basis is exact to first order in Z only; its O(Z^2) terms grow with the number
    of modes (item 5). Size for aluminium: eigen-attenuation of the TE10-like and TE01-like modes of D - Lambda_full
    (family n = 0 or m = 0 modes, all four walls aluminium) against Lambda_pp, vs the number of modes N."""
    print("  O(Z^2) truncation artifact of the PEC-basis system, aluminium, 90 GHz:")
    a, b, f = W, H, 90e9
    Zs = T.surface_impedance(f, SIGMA)
    for p0, fam in ((("TE", 1, 0), lambda M: [("TE", m, 0) for m in range(1, M + 1)]),
                    (("TE", 0, 1), lambda M: [("TE", 0, n) for n in range(1, M + 1)])):
        out = []
        for M in (5, 10, 20, 40):
            modes = fam(M)
            beta = np.array([T.beta(p, a, b, f) for p in modes])
            N = len(modes)
            D = np.block([[np.diag(-1j * beta), np.zeros((N, N))], [np.zeros((N, N)), np.diag(1j * beta)]])
            w = np.linalg.eigvals(D - T.lambda_full(modes, a, b, f, Zs))
            j = np.argmin(abs(w + 1j * beta[0]))
            lam = T.lambda_pp(p0, p0, a, b, f, Zs)
            out.append(f"N = {M}: {abs(-w[j] - 1j * beta[0] - lam) / abs(lam):.1e}")
        print(f"    {p0}: |gamma_eig - (beta - i Lambda_pp)| / |Lambda_pp| = " + ", ".join(out))


def sech_curvature(theta_deg, length, Tt=4.0):
    gamma = 2 * Tt / length
    R0 = 4 / np.radians(theta_deg) / gamma * np.arctan(np.tanh(Tt / 2))
    return lambda z: 1 / (R0 * np.cosh(gamma * (np.asarray(z) - length / 2))), 1 / R0


def transmission(D, K, L0, L1, kappa, length, nsl=3000):
    """Midpoint product of slice exponentials; returns (T, R) for unit incident forward waves."""
    N = D.shape[0] // 2
    dz = length / nsl
    Phi = np.eye(2 * N, dtype=complex)
    for i in range(nsl):
        kz = float(kappa((i + 0.5) * dz))
        Phi = expm((D + kz * (K - L1) - L0) * dz) @ Phi
    P11, P12, P21, P22 = Phi[:N, :N], Phi[:N, N:], Phi[N:, :N], Phi[N:, N:]
    R = -np.linalg.solve(P22, P21)
    return P11 + P12 @ R, R


def item4():
    print("ITEM 4 -- SPARC bends with aluminium walls (sech profile, T = 4)")
    print("  plane  bend        f GHz  N   T_main dB (no loss)  T_main dB (loss)  "
          "|dT| from kappa*Lambda1 (dB)  max |d|T_pq|^2| from kappa*Lambda1  peak kappa*a/2")
    res = []
    for plane, (a, b, n) in (("H", (W, H, 0)), ("E", (H, W, 1))):
        for theta, length in ((60, 0.60), (90, 0.40)):
            kappa, kmax = sech_curvature(theta, length)
            for f in (20e9, 45e9, 70e9, 90e9):
                legacy = select_coupled_modes(f, a, b, n=n)
                modes = [(x[0][:2], int(x[1]), int(x[2])) for x in legacy]
                D, K, beta = coupled_mode_matrices(f, a, b, legacy)
                Zs = T.surface_impedance(f, SIGMA)
                L0 = T.lambda_full(modes, a, b, f, Zs)
                L1 = T.lambda1_full(modes, a, b, f, Zs)
                Z0 = np.zeros_like(L0)
                Tn, _ = transmission(D, K, Z0, Z0, kappa, length)
                Tl, _ = transmission(D, K, L0, Z0, kappa, length)
                Tc, _ = transmission(D, K, L0, L1, kappa, length)
                dB = lambda x: 20 * np.log10(abs(x))
                dT = abs(dB(Tc[0, 0]) - dB(Tl[0, 0]))
                dP = np.max(abs(abs(Tc[:, 0]) ** 2 - abs(Tl[:, 0]) ** 2))
                res.append((plane, theta, f, dT, dP))
                print(f"  {plane}      {theta:2d} deg {length*100:.0f} cm  {f/1e9:4.0f}  {len(modes):2d}  "
                      f"{dB(Tn[0,0]):10.5f}           {dB(Tl[0,0]):10.5f}        {dT:.2e}                      "
                      f"{dP:.2e}                       {kmax*a/2:.3f}")
    print(f"  overall max |dT_main| = {max(r[3] for r in res):.2e} dB, max change of any converted power = "
          f"{max(r[4] for r in res):.2e}")
    return res


if __name__ == "__main__":
    item3()
    truncation_artifact()
    item4()
