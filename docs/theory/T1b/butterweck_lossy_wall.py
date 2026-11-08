"""Item 5 of T1b: Butterweck's lossy-wall mode filter, first-order theory vs exact modes + mode matching.

Run from the repository root:   PYTHONPATH=src python3 docs/theory/T1b/butterweck_lossy_wall.py

Butterweck, IEEE MTT-16 (1968) 274, p. 278-279: X-band guide 22.86 x 10.16 mm at 75 GHz, side walls (u = -+a/2,
the walls parallel to the E field of H10) with Z_w = 250 ohm (real), length 15 cm. Theory (his eq. 21): 4 dB/m
(H10) and 36 dB/m (H30), i.e. 0.6 and 5.4 dB; H10 -> H30 conversion between 0.016 and 0.06 (amplitude).
Measured (Butterweck and de Ronde, Philips Tech. Rev. 29 (1968) p. 95): 0.6, 2.5, 5.4 dB for H10, H20, H30.
"""
import os
import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1b_coefficients as T  # noqa: E402

a, b, f, ZW, L = 22.86e-3, 10.16e-3, 75e9, 250.0, 0.15
k = T.wavenumber(f)
DB = 20 * np.log10(np.e)


def first_order():
    print("First order (Lambda with Z_u- = Z_u+ = 250 ohm, v-walls PEC), family n = 0:")
    out = {}
    for m in (1, 2, 3):
        p = ("TE", m, 0)
        lam = T.lambda_pp(p, p, a, b, f, {"u-": ZW, "u+": ZW})
        eq21 = T.butterweck_eq21(m, a, f, ZW)
        out[m] = lam.real
        print(f"   H{m}0: Re Lambda_pp = {lam.real:.6f} Np/m = {DB*lam.real:.3f} dB/m -> {DB*lam.real*L:.3f} dB / 15 cm;"
              f"  Butterweck eq. 21: {DB*eq21:.3f} dB/m (difference {abs(lam.real-eq21)/eq21:.1e})")
    return out


def exact_modes(M):
    return T.impedance_guide_modes(0, a, b, f, ZW, ZW, M)


def filter_gsm(M, length=L):
    pec = [("TE", m, 0) for m in range(1, M + 1)]
    imp = exact_modes(M)
    R = T.lossy_wall_junction(pec, imp, 0, a, b, f)
    J = T.junction_gsm(R)
    P = np.diag(np.exp(-1j * np.array([md["gamma"] for md in imp]) * length))
    Z0 = np.zeros_like(P)
    prop = (Z0, P, P, Z0)
    Jr = (J[3], J[2], J[1], J[0])
    return T.star(T.star(J, prop), Jr), imp, J


def exact():
    print("Exact: LSE^u modes of the impedance-walled guide (k_x tan-type equation) + mode matching")
    imp = exact_modes(12)
    for md in imp[:4]:
        al = -md["gamma"].imag
        print(f"   mode from H{md['m0']}0: kx a/pi = {md['kx']*a/np.pi:.5f}, alpha = {al:.4f} Np/m = {DB*al:.3f} dB/m"
              f" -> {DB*al*L:.3f} dB / 15 cm")
    print("   full filter (PEC | 15 cm lossy | PEC), |S21| of H_m0 -> H_m0 in dB and H10 -> H30 conversion:")
    res = {}
    for M in (6, 12, 20, 30, 40):
        S, imp, J = filter_gsm(M)
        S21 = S[2]
        t = [-20 * np.log10(abs(S21[i, i])) for i in range(3)]
        c13 = abs(S21[2, 0])
        sym = max(np.max(abs(J[1] - J[2].T)), np.max(abs(J[0] - J[0].T)), np.max(abs(J[3] - J[3].T)))
        refl = abs(S[0][0, 0])
        res[M] = (t, c13)
        print(f"   M = {M:2d}: H10 {t[0]:.4f} dB, H20 {t[1]:.4f} dB, H30 {t[2]:.4f} dB; |C3(L)/C1(0)| = {c13:.4f}"
              f" ({20*np.log10(c13):.1f} dB); |S11(H10)| = {refl:.1e}; junction reciprocity {sym:.1e}")
    # passivity of the full filter over the propagating modes
    S, imp, J = filter_gsm(30)
    Sf = np.block([[S[0], S[1]], [S[2], S[3]]])
    nprop = sum(1 for m in range(1, 31) if m * np.pi / a < k)
    idx = list(range(nprop)) + [30 + i for i in range(nprop)]
    sv = np.linalg.svd(Sf[np.ix_(idx, idx)], compute_uv=False)[0]
    print(f"   passivity: largest singular value over the {nprop} propagating modes = {sv:.6f}")
    # conversion vs length (Butterweck's bounds are the min/max over the beat)
    lengths = np.linspace(0.10, 0.20, 101)
    conv = [abs(filter_gsm(20, length=Lx)[0][2][2, 0]) for Lx in lengths]
    print(f"   |C3/C1| over L = 10-20 cm: min {min(conv):.4f}, max {max(conv):.4f}")
    return res


def coupled_mode_conversion():
    """First-order coupled-mode system (propagating H_m0 modes, u-walls 250 ohm) over 15 cm, (a) with all
    propagating modes, (b) without the modes for which |Lambda_pp| > 0.05 beta_p (near cutoff)."""
    allm = [("TE", m, 0) for m in range(1, 12) if m * np.pi / a < k]
    ratio = {p: abs(T.lambda_pp(p, p, a, b, f, {"u-": ZW, "u+": ZW})) / T.beta(p, a, b, f).real for p in allm}
    print("First-order coupled modes over 15 cm (no junctions); |Lambda_pp|/beta_p of the highest modes: "
          + ", ".join(f"H{p[1]}0 {ratio[p]:.3f}" for p in allm[-3:]))
    out = {}
    for label, modes in (("all 11 propagating modes", allm),
                         ("modes with |Lambda_pp| < 0.05 beta_p", [p for p in allm if ratio[p] < 0.05])):
        beta = np.array([T.beta(p, a, b, f) for p in modes])
        N = len(modes)
        D = np.block([[np.diag(-1j * beta), np.zeros((N, N))], [np.zeros((N, N)), np.diag(1j * beta)]])
        Lf = T.lambda_full(modes, a, b, f, {"u-": ZW, "u+": ZW})
        Phi = expm((D - Lf) * L)
        R = -np.linalg.solve(Phi[N:, N:], Phi[N:, :N])
        Tm = Phi[:N, :N] + Phi[:N, N:] @ R
        out[label] = Tm
        print(f"   {label} (N = {N}): "
              + ", ".join(f"H{i+1}0 {-20*np.log10(abs(Tm[i,i])):.4f} dB" for i in range(3))
              + f"; |C3/C1| = {abs(Tm[2,0]):.4f}")
    return out


def breakdown():
    """Perturbation parameter: first-order attenuation (diagonal, or the 2x2 degenerate block, whose eigenvectors are
    exactly LSE^u / LSM^u) against the exact LS^u mode, as a function of Z (both side walls equal, real)."""
    print("Where first order fails: |alpha_1st / alpha_exact - 1| for single LS^u modes (75 GHz, WR-90, Z real)")
    cases = (("LSE", 1, 0), ("LSE", 3, 0), ("LSE", 1, 1), ("LSM", 0, 1), ("LSM", 1, 1))
    print("   Z ohm  " + "  ".join(f"{c[0]}^u_{c[1]},{c[2]}" for c in cases) + "   2Z/(k eta a)   k a Z/eta")
    rows = []
    for Z in (0.1, 1, 10, 50, 250, 1000):
        Zd = {"u-": Z, "u+": Z}
        errs = []
        for kind, m, n in cases:
            kx = T.lsu_kx_roots(kind, [m], a, k, Z, Z, steps=400)[0]
            ge = np.sqrt(k * k - kx * kx - (n * np.pi / b) ** 2 + 0j)
            ge = -ge if ge.imag > 0 else ge
            if m == 0 or n == 0:
                lam = T.lambda_pp(("TE", m, n), ("TE", m, n), a, b, f, Zd)
            else:
                pE, pM = ("TE", m, n), ("TM", m, n)
                Lb = np.array([[T.lambda_pp(x, y, a, b, f, Zd) for y in (pE, pM)] for x in (pE, pM)])
                w, V = np.linalg.eig(Lb)
                sE = np.sqrt(T.admittance(pE, a, b, f)); sM = np.sqrt(T.admittance(pM, a, b, f))
                lse = np.array([T.norm(pM, a, b) * m * np.pi / a / sM, T.norm(pE, a, b) * n * np.pi / b / sE])
                ov = [abs(np.dot(lse, V[:, j])) / np.linalg.norm(lse) / np.linalg.norm(V[:, j]) for j in range(2)]
                j = int(np.argmax(ov)) if kind == "LSE" else int(np.argmin(ov))
                lam = w[j]
            errs.append(abs(lam.real / (-ge.imag) - 1))
        rows.append((Z, errs))
        print(f"   {Z:6g}  " + "  ".join(f"{e:9.2e}" for e in errs)
              + f"      {2*Z/(k*T.ETA0*a):.2e}       {k*a*Z/T.ETA0:.2e}")
    return rows


def check_junction():
    """Junction PEC | impedance walls: Z -> 0 gives a lossless identity-like junction (R R^T = I, S11 = 0), for the
    family n = 0 (LSE^u only) and n = 1 (LSE^u and LSM^u); at 250 ohm the n = 1 filter is reciprocal and passive."""
    print("Junction checks (families n = 0 and n = 1)")
    worst = 0.0
    for n in (0, 1):
        M = 8
        pec = [("TE", m, n) for m in range(0 if n else 1, M + 1)] + ([("TM", m, n) for m in range(1, M + 1)] if n else [])
        imp = T.impedance_guide_modes(n, a, b, f, 1e-9, 1e-9, M)
        R = T.lossy_wall_junction(pec, imp, n, a, b, f)
        S11 = T.junction_gsm(R)[0]
        e1, e2 = np.max(abs(R @ R.T - np.eye(len(pec)))), np.max(abs(S11))
        worst = max(worst, e1, e2)
        print(f"   n = {n}, Z = 1e-9 ohm: |R R^T - I| = {e1:.1e}, |S11| = {e2:.1e}")
    assert worst < 1e-7
    n, M = 1, 12
    pec = [("TE", m, n) for m in range(0, M + 1)] + [("TM", m, n) for m in range(1, M + 1)]
    imp = T.impedance_guide_modes(n, a, b, f, ZW, ZW, M)
    J = T.junction_gsm(T.lossy_wall_junction(pec, imp, n, a, b, f))
    P = np.diag(np.exp(-1j * np.array([md["gamma"] for md in imp]) * L))
    Z0 = np.zeros_like(P)
    S = T.star(T.star(J, (Z0, P, P, Z0)), (J[3], J[2], J[1], J[0]))
    Sf = np.block([[S[0], S[1]], [S[2], S[3]]])
    prop = [i for i, p in enumerate(pec) if T.cutoff(p, a, b) < k]
    idx = prop + [len(pec) + i for i in prop]
    sv = np.linalg.svd(Sf[np.ix_(idx, idx)], compute_uv=False)[0]
    i11 = pec.index(("TE", 1, 1))
    print(f"   n = 1, 250 ohm, 15 cm: reciprocity {np.max(abs(Sf - Sf.T)):.1e}, largest singular value {sv:.4f},"
          f" TE11 -> TE11 {20*np.log10(abs(S[2][i11, i11])):.1f} dB, TE01 -> TE01 "
          f"{20*np.log10(abs(S[2][0, 0])):.1f} dB")


if __name__ == "__main__":
    check_junction()
    fo = first_order()
    ex = exact()
    coupled_mode_conversion()
    breakdown()
