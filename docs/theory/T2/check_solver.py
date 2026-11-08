"""T2 check 3: the cutoff-crossing method (vi_solver.py) and the closed-form special cases.

Run:  PYTHONPATH=src python3 docs/theory/T2/check_solver.py

1. Single mode through cutoff, exact: TE1,0 in a taper with a(z) = pi/sqrt(k^2 - g (z - z_c)), so that
   beta^2 = g (z - z_c) and the single-mode V-I equations reduce to V'' + g (z - z_c) V = 0 (Airy). Port 1 is
   below cutoff, port 2 above. GSM of the solver vs the exact GSM from Ai and Bi.
2. Single mode, linear H-plane taper a = a0 + alpha z, TE1,0: V = sqrt(a) Z_nu(k a/alpha), nu^2 = 1/4 + pi^2/alpha^2.
3. Multimode H-plane wedge (family n = 0) with a mode crossing cutoff, against exact cylindrical-wave fields:
   residual |b - S a| of the solver's GSM applied to the local-mode amplitudes of exact fields.
4. Reciprocity and unitarity of the solver's GSM (curvature + taper + offset, evanescent modes in the list).
5. Total reflection of a mode at its cutoff plane: |S22| and the turning-point phase S22 = i exp(-2 i int beta)
   (TE, N_qq = 0), the Butterweck & de Ronde "reflection at the cutoff plane".
6. Convergence in the number of evanescent modes (cutoff factor) and in the slice count, for a taper with a cutoff
   crossing.
7. Small-slope limit: first-order conversion formula vs the full solution.
8. Lossless structure of the power-wave coupling matrix: K^H G + G K = 0, G = diag(I, -I), for propagating modes.
"""
import os
import sys

import numpy as np
import scipy.constants as cst
from scipy.special import airy, jv, yv, jvp, yvp

sys.path.insert(0, os.path.dirname(__file__))
import t2_coefficients as t2                    # noqa: E402
import vi_solver as vs                          # noqa: E402
import exact_fields as ef                       # noqa: E402
from check_exact_fields import project          # noqa: E402
from wrmodes.modes import Mode, propagation_constant, wave_admittance   # noqa: E402

C0 = cst.c
OK = True


def report(name, err, tol, less=True):
    global OK
    good = err < tol if less else err > tol
    OK &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name}: {err:.3e}  (target {'<' if less else '>'} {tol:g})")


def exact_gsm_single(mode, geometry, f, z0, z1, sols):
    """GSM of one mode between z0 and z1 from two exact solutions sols(z) -> [[V_A, V_B], [I_A, I_B]]."""
    Phi = sols(z1) @ np.linalg.inv(sols(z0))
    g0, g1 = geometry(z0), geometry(z1)
    Pp0, Pm0 = vs.wave_rows([mode], g0, f)
    Pp1, Pm1 = vs.wave_rows([mode], g1, f)
    A = np.vstack([Pp0, Pm1 @ Phi])
    B = np.vstack([Pm0, Pp1 @ Phi])
    return B @ np.linalg.inv(A)


def geo(a, da, b=0.01016, db=0.0, u0=0.0, du0=0.0, kappa=0.0):
    return dict(a=a, b=b, u0=u0, v0=0.0, kappa=kappa, da=da, db=db, du0=du0, dv0=0.0)


def test_airy():
    print("1. Single mode through cutoff (Airy)")
    f = 10e9
    k, w = 2 * np.pi * f / C0, 2 * np.pi * f
    gam, zc, L = k * k / 0.3, 0.03, 0.10
    s = gam ** (1 / 3)
    te10 = Mode("TE", 1, 0)
    geometry = lambda z: geo(np.pi / np.sqrt(k * k - gam * (z - zc)),
                             0.5 * np.pi * gam * (k * k - gam * (z - zc)) ** -1.5)

    def sols(z):
        x = -s * (z - zc)
        ai, aip, bi, bip = airy(x)
        V = np.array([ai, bi])
        dV = -s * np.array([aip, bip])
        return np.array([V, 1j * dV / (w * cst.mu_0)])      # I = (i/(w mu0)) dV/dz

    Sx = exact_gsm_single(te10, geometry, f, 0.0, L, sols)
    for ns in (40, 80, 160):
        Ss = vs.taper_gsm([te10], geometry, f, L, nslices=ns).S
        print(f"     {ns} slices: max|S - S_exact| = {np.max(np.abs(Ss - Sx)):.3e}")
    report("Airy single-mode GSM (160 slices x 2 Magnus steps)", np.max(np.abs(Ss - Sx)), 1e-8)
    print(f"     exact |S11| = {abs(Sx[0, 0]):.4f} (port 1 evanescent: beta = {propagation_constant(te10, geometry(0)['a'], 0.01016, f):.2f}), "
          f"|S22| = {abs(Sx[1, 1]):.6f}, |S21| = {abs(Sx[1, 0]):.4e}")


def test_bessel():
    print("2. Single mode, linear H-plane taper (Bessel)")
    f = 12e9
    k, w = 2 * np.pi * f / C0, 2 * np.pi * f
    a0, alpha, L = 0.014, 0.25, 0.08                    # TE1,0 cutoff at a = 12.49 mm: propagating throughout
    nu = np.sqrt(0.25 + (np.pi / alpha) ** 2)
    K = k / alpha
    te10 = Mode("TE", 1, 0)
    geometry = lambda z: geo(a0 + alpha * z, alpha)

    def sols(z):
        x = a0 + alpha * z
        V = np.array([np.sqrt(x) * jv(nu, K * x), np.sqrt(x) * yv(nu, K * x)])
        dVdx = np.array([jv(nu, K * x) / (2 * np.sqrt(x)) + np.sqrt(x) * K * jvp(nu, K * x),
                         yv(nu, K * x) / (2 * np.sqrt(x)) + np.sqrt(x) * K * yvp(nu, K * x)])
        return np.array([V, 1j * alpha * dVdx / (w * cst.mu_0)])

    Sx = exact_gsm_single(te10, geometry, f, 0.0, L, sols)
    Ss = vs.taper_gsm([te10], geometry, f, L, nslices=160).S
    report("linear H-plane taper single-mode GSM vs sqrt(a) J_nu, Y_nu", np.max(np.abs(Ss - Sx)), 1e-8)
    print(f"     |S11| = {abs(Sx[0, 0]):.4e}; nu = {nu:.4f} (exact wedge mode order pi/phi0 = "
          f"{np.pi / (2 * np.arctan(alpha / 2)):.4f})")


def wave_amplitudes(case, z, modes, f):
    V, I = project(case, z, modes)
    g = case.geometry(z)
    sq = np.sqrt(np.array([wave_admittance(m, g["a"], g["b"], f) for m in modes], dtype=complex))
    return (sq * V + I / sq) / (2 * np.sqrt(2)), (sq * V - I / sq) / (2 * np.sqrt(2))


def test_wedge():
    print("3. Multimode H-plane wedge vs exact cylindrical waves (family 0, TE3,0 crosses cutoff)")
    f = 20e9
    k = 2 * np.pi * f / C0
    phi0, b = 0.4, 0.0102
    za = (3 * np.pi / phi0 / k) * np.cos(phi0 / 2) - 0.015
    z0, z1 = 0.004, 0.034
    fields = [[("TMv", 0, 3, "J", 1.0)], [("TMv", 0, 1, "H2", 1.0)], [("TMv", 0, 1, "H1", 1.0)],
              [("TMv", 0, 2, "J", 1.0)], [("TMv", 0, 4, "J", 1.0)]]
    cases = [ef.HPlaneWedge(f, b, za, phi0, c) for c in fields]
    g0, g1 = cases[0].geometry(z0), cases[0].geometry(z1)
    te = lambda M: [Mode("TE", m, 0) for m in range(1, M + 1)]
    print(f"     a(z0) = {g0['a'] * 1e3:.2f} mm, a(z1) = {g1['a'] * 1e3:.2f} mm; propagating at z0: "
          f"{[m.name for m in te(8) if propagation_constant(m, g0['a'], b, f).imag == 0]}, at z1: "
          f"{[m.name for m in te(8) if propagation_constant(m, g1['a'], b, f).imag == 0]}")
    geometry = cases[0].geometry
    res = []
    for M in (6, 12, 24):
        modes = te(M)
        S = vs.taper_gsm(modes, lambda z: geometry(z + z0), f, z1 - z0, nslices=30).S
        errs = []
        for case in cases:
            ap0, am0 = wave_amplitudes(case, z0, modes, f)
            ap1, am1 = wave_amplitudes(case, z1, modes, f)
            a_in, b_out = np.concatenate([ap0, am1]), np.concatenate([am0, ap1])
            low = np.r_[0:4, M:M + 4]            # outputs TE1..TE4,0 at both ports
            errs.append(np.max(np.abs(b_out[low] - (S @ a_in)[low])) / np.max(np.abs(a_in)))
        res.append(max(errs))
        print(f"     {M:2d} modes, 30 slices: max |b - S a| / max|a| over 5 exact fields = {max(errs):.3e}")
    rate = np.log(res[1] / res[2]) / np.log(2)
    print(f"     convergence rate in the number of modes M: residual ~ M^-{rate:.2f} (wall slope a' = "
          f"{geometry(0)['da']:.3f}); the truncated equations converge as O(1/M), see the T2 note")
    report("exact-field GSM residual (24 modes)", res[2], 5e-2)
    report("convergence rate in M (expected ~1)", abs(rate - 1.0), 0.2)


def test_reciprocity_unitarity():
    print("4. Reciprocity and unitarity (kappa, a', u0', b' together; evanescent modes in the list)")
    f, L = 20e9, 0.12
    def geometry(z):
        s = np.sin(np.pi * z / L)
        return dict(a=0.0229 * (1 + 0.35 * s * s), b=0.0102 * (1 + 0.2 * z / L), u0=0.002 * s, v0=0.0,
                    kappa=4.0 * s, da=0.0229 * 0.35 * np.sin(2 * np.pi * z / L) * np.pi / L,
                    db=0.0102 * 0.2 / L, du0=0.002 * np.cos(np.pi * z / L) * np.pi / L, dv0=0.0)
    modes = [Mode(k, m, n) for m in range(0, 7) for n in (0, 2) for k in ("TE", "TM")
             if not (k == "TE" and m == n == 0) and not (k == "TM" and (m == 0 or n == 0))]
    G = vs.taper_gsm(modes, geometry, f, L, nslices=40)
    S = G.S
    report("reciprocity max|S - S^T|/max|S|", np.max(np.abs(S - S.T)) / np.max(np.abs(S)), 1e-10)
    p1, p2 = G.propagating()
    idx = np.concatenate([np.where(p1)[0], len(p1) + np.where(p2)[0]])
    Sp = S[np.ix_(idx, idx)]
    report(f"unitarity over the {len(idx)} propagating port modes max|Sp^H Sp - I|",
           np.max(np.abs(Sp.conj().T @ Sp - np.eye(len(idx)))), 1e-10)


def test_total_reflection():
    print("5. Total reflection at the cutoff plane (TE3,0 incident from the wide end)")
    f = 20e9
    k = 2 * np.pi * f / C0
    a0, a1, L = 0.0205, 0.0275, 0.25                     # TE3,0 cutoff at a = 22.48 mm
    geometry = lambda z: geo(a0 + (a1 - a0) * z / L, (a1 - a0) / L)
    te30 = Mode("TE", 3, 0)
    zc = (3 * np.pi / k - a0) * L / (a1 - a0)
    zz = np.linspace(zc, L, 20001)
    beta = np.array([propagation_constant(te30, geometry(z)["a"], 0.01016, f).real for z in zz])
    zeta = np.trapezoid(beta, zz)
    modes = [Mode("TE", m, 0) for m in range(1, 8)]
    G = vs.taper_gsm(modes, geometry, f, L, nslices=120)
    i3 = modes.index(te30)
    S22 = G.S22[i3, i3]
    wkb = 1j * np.exp(-2j * zeta)
    print(f"     z_c = {zc * 1e3:.2f} mm, int beta = {zeta:.3f} rad; |S22(TE3,0)| = {abs(S22):.6f}; "
          f"phase error vs WKB i exp(-2i int beta) = {np.angle(S22 / wkb):.4f} rad; slope a' = {(a1 - a0) / L:.3f}")
    report("|S22(TE3,0)| (total reflection)", abs(1 - abs(S22)), 1e-3)
    report("phase of S22 vs WKB turning-point result (rad)", abs(np.angle(S22 / wkb)), 0.05)


def test_convergence():
    print("6. Convergence in the evanescent-mode count and the slice count (TE1,0 -> TE3,0, cutoff crossing)")
    f = 20e9
    a0, a1, L = 0.016, 0.034, 0.10
    geometry = lambda z: geo(a0 + (a1 - a0) * (z / L) ** 2 * (3 - 2 * z / L),
                             (a1 - a0) * 6 * (z / L) * (1 - z / L) / L)
    k = 2 * np.pi * f / C0
    amin = a0
    rows = []
    for fc in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
        mmax = int(np.floor(fc * k * a1 / np.pi - 1e-12))      # modes with min_z chi < fc k  (widest section)
        modes = [Mode("TE", m, 0) for m in range(1, mmax + 1)]
        G = vs.taper_gsm(modes, geometry, f, L, nslices=60)
        i1, i3 = 0, 2
        rows.append((fc, mmax, G.S21[i3, i1], G.S21[i1, i1], G.S11[i1, i1]))
    ref = rows[-1]
    for fc, mmax, s31, s11, r11 in rows:
        print(f"     cutoff factor {fc:4.2f} ({mmax:2d} modes): S21(TE3<-TE1) = {abs(s31):.5e}, "
              f"|dS21(TE1)| = {abs(s11 - ref[3]):.2e}, |dS21(TE3<-TE1)| = {abs(s31 - ref[2]):.2e}, "
              f"|S11(TE1)| = {abs(r11):.3e}")
    report("|S21(TE3<-TE1)(fc=3) - (fc=4)| relative", abs(rows[-2][2] - ref[2]) / abs(ref[2]), 1e-3)
    modes = [Mode("TE", m, 0) for m in range(1, int(np.floor(2 * k * a1 / np.pi)) + 1)]
    vals = [vs.taper_gsm(modes, geometry, f, L, nslices=ns).S21[2, 0] for ns in (15, 30, 60)]
    print(f"     slices 15/30/60: |S21(TE3<-TE1)| = {abs(vals[0]):.6e} {abs(vals[1]):.6e} {abs(vals[2]):.6e}; "
          f"differences {abs(vals[0] - vals[2]):.2e}, {abs(vals[1] - vals[2]):.2e}")
    report("slice convergence |S(30) - S(60)| / |S|", abs(vals[1] - vals[2]) / abs(vals[2]), 1e-4)


def test_first_order():
    print("7. Small-slope limit: first-order conversion vs the full solution")
    f = 20e9
    a0, a1, L = 0.025, 0.035, 0.15
    geometry = lambda z: geo(a0 + (a1 - a0) * z / L, (a1 - a0) / L)
    te10, te30 = Mode("TE", 1, 0), Mode("TE", 3, 0)
    modes = [Mode("TE", m, 0) for m in range(1, 8)]
    G = vs.taper_gsm(modes, geometry, f, L, nslices=60)
    full_t, full_r = G.S21[2, 0], G.S11[0, 0]
    fo_t = t2.first_order_conversion(te10, te30, geometry, f, L, nz=4001)
    fo_r = t2.first_order_conversion(te10, te10, geometry, f, L, nz=4001, reflected=True)
    # phase reference: the full S21 contains the propagation of TE1,0 and TE3,0 along the taper, as does fo_t
    print(f"     S21(TE3<-TE1): full {full_t:.4e}, first order {fo_t:.4e}; S11(TE1): full {full_r:.4e}, "
          f"first order {fo_r:.4e}")
    report("first-order TE1->TE3 conversion vs full (relative)", abs(fo_t - full_t) / abs(full_t), 0.05)
    report("first-order TE1 reflection vs full (relative)", abs(fo_r - full_r) / abs(full_r), 0.05)


def test_power_structure():
    print("8. Lossless structure of the power-wave matrix (propagating modes)")
    f = 30e9
    modes = [Mode(k, m, n) for m in range(0, 5) for n in range(0, 3) for k in ("TE", "TM")
             if not (k == "TE" and m == n == 0) and not (k == "TM" and (m == 0 or n == 0))]
    a, b = 0.0229, 0.0152
    k = 2 * np.pi * f / C0
    modes = [m for m in modes if propagation_constant(m, a, b, f).imag == 0]
    K = t2.power_wave_matrix(modes, a, b, f, da=0.13, db=-0.07, du0=0.21, dv0=0.05, kappa=3.0, u0=0.001)
    n = len(modes)
    G = np.diag(np.r_[np.ones(n), -np.ones(n)])
    report(f"max|K^H G + G K| / max|K| ({n} modes)", np.max(np.abs(K.conj().T @ G + G @ K)) / np.max(np.abs(K)), 1e-12)
    M = t2.vi_matrix(modes + [Mode("TE", 7, 0)], a, b, f, da=0.13, db=-0.07, du0=0.21, dv0=0.05, kappa=3.0, u0=0.001)
    n2 = len(modes) + 1
    J = np.block([[np.zeros((n2, n2)), np.eye(n2)], [-np.eye(n2), np.zeros((n2, n2))]])
    Gh = np.block([[np.zeros((n2, n2)), np.eye(n2)], [np.eye(n2), np.zeros((n2, n2))]])
    report("V-I matrix: reciprocity M^T J + J M = 0 (with an evanescent mode)",
           np.max(np.abs(M.T @ J + J @ M)) / np.max(np.abs(M)), 1e-14)
    report("V-I matrix: power M^H G + G M = 0, G = [[0, I], [I, 0]] (with an evanescent mode)",
           np.max(np.abs(M.conj().T @ Gh + Gh @ M)) / np.max(np.abs(M)), 1e-14)


if __name__ == "__main__":
    test_airy()
    test_bessel()
    test_wedge()
    test_reciprocity_unitarity()
    test_total_reflection()
    test_convergence()
    test_first_order()
    test_power_structure()
    print("ALL PASS" if OK else "SOME CHECKS FAILED")
    sys.exit(0 if OK else 1)
