"""Butterweck's lossy-wall mode filter (IEEE MTT-16, 1968, eqs. 20-31; Philips Tech. Rev. 29, 1968, p. 94-95).

Standard X-band guide (a = 22.86 mm, b = 10.16 mm) at 75 GHz with wall impedance Z_w = 250 ohm (real) on both
narrow walls (u = +-a/2 in the local frame), filter length 15 cm. Three methods:

  M1  first-order attenuation, diag(Lambda^+) of T1 = Butterweck eq. (21);
  M2  first-order coupled-mode equations of T1, dy/dz = (D - Lambda_full) y, in the ideal-guide basis of family
      n = 0 (all propagating TE_m0 modes), no junctions needed;
  M3  exact: LSE^u modes of the impedance-walled guide from the dispersion relation (B6), mode matching to the
      ideal guide at both ends (B9), Redheffer cascade; convergence in the number of modes.

Butterweck's theory values: 4 dB/m (H10) and 36 dB/m (H30), i.e. 0.6 and 5.4 dB over 15 cm; H10 -> H30
conversion between -36 and -24.5 dB (eqs. 30-31). His measurements (Philips Tech. Rev.): 0.6, 2.5 and 5.4 dB for
H10, H20, H30 over 15 cm; H10 -> H30 conversion about -30 dB.

Run:  PYTHONPATH=<worktree>/src python3 docs/theory/T1/butterweck_lossy_wall.py
"""
import sys
import numpy as np
from scipy.linalg import expm

import t1lib as L
from wrmodes.modes import ModeBasis, Mode, ETA0

a, b, f, Zw, length = 22.86e-3, 10.16e-3, 75e9, 250.0, 0.15
k = L.wavenumber(f)
DB = 20 / np.log(10)       # Np -> dB
FAILS = []


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


Zwalls = {"u+": Zw, "u-": Zw}

# ------------------------------------------------------------------------------------------------- M1
print("M1  first-order attenuation (diag Lambda^+, = Butterweck eq. 21)")
alpha1 = {}
for m in (1, 2, 3):
    B = ModeBasis(a, b, [Mode("TE", m, 0)])
    Lp, _ = L.loss_matrices(B, f, Zwalls)
    beta = np.sqrt(k * k - (m * np.pi / a) ** 2)
    eq21 = 2 * np.pi ** 2 * m * m * Zw / (beta * a ** 3 * k * ETA0)
    alpha1[m] = Lp[0, 0].real
    print(f"     H{m}0: alpha = {Lp[0, 0].real * DB:7.3f} dB/m  (eq. 21: {eq21 * DB:7.3f} dB/m), "
          f"{Lp[0, 0].real * DB * length:6.3f} dB over 15 cm;  Im Lambda = {Lp[0, 0].imag:.1e}")
    assert abs(Lp[0, 0].real / eq21 - 1) < 1e-12
check("first-order H10 vs Butterweck theory 4 dB/m (5 %)", abs(alpha1[1] * DB / 4 - 1) < 0.05,
      f"{alpha1[1] * DB:.3f} dB/m, {100 * (alpha1[1] * DB / 4 - 1):+.1f} %")
check("first-order H30 vs Butterweck theory 36 dB/m (5 %)", abs(alpha1[3] * DB / 36 - 1) < 0.05,
      f"{alpha1[3] * DB:.3f} dB/m, {100 * (alpha1[3] * DB / 36 - 1):+.1f} %")
check("first-order H10 vs Butterweck theory 0.6 dB / 15 cm (5 %)", abs(alpha1[1] * DB * length / 0.6 - 1) < 0.05,
      f"{alpha1[1] * DB * length:.3f} dB")
check("first-order H30 vs Butterweck theory 5.4 dB / 15 cm (5 %)", abs(alpha1[3] * DB * length / 5.4 - 1) < 0.05,
      f"{alpha1[3] * DB * length:.3f} dB")

# ------------------------------------------------------------------------------------------------- M2
print("\nM2  first-order coupled-mode equations (all propagating TE_m0 modes, forward and backward)")
B0 = ModeBasis.propagating(f, a, b, families=[0])
n0 = len(B0)
D = L.D_matrix(B0, f)
Lf = L.loss_matrix_full(B0, f, Zwalls)


def tr_from_phi(phi, n):
    Pff, Pfb, Pbf, Pbb = phi[:n, :n], phi[:n, n:], phi[n:, :n], phi[n:, n:]
    R = -np.linalg.solve(Pbb, Pbf)
    return Pff + Pfb @ R, R


i1, i2, i3 = (B0.index(Mode("TE", m, 0)) for m in (1, 2, 3))
for nm in (3, 5, 7, 9, 11):
    Bt = ModeBasis(a, b, [Mode("TE", m, 0) for m in range(1, nm + 1)])
    Tt, Rt = tr_from_phi(expm((L.D_matrix(Bt, f) - L.loss_matrix_full(Bt, f, Zwalls)) * length), nm)
    mm_ = {m: -DB * np.log(abs(Tt[m - 1, m - 1])) for m in (1, 2, 3)}
    cc_ = DB * np.log(abs(Tt[2, 0]))
    print(f"     TE_10..TE_{nm},0 in the basis: H10 {mm_[1]:.4f} dB, H20 {mm_[2]:.4f} dB, H30 {mm_[3]:.4f} dB, "
          f"H10->H30 {cc_:.2f} dB")
    if nm == 3:
        m2, conv2 = mm_, cc_
print("     (the dependence on the basis size is the spurious O(Z^2) drift of the ideal-mode expansion, sec. 6.3)")

# ------------------------------------------------------------------------------------------------- M3
print("\nM3  exact impedance-walled modes + mode matching")


def lossy_modes(M, Zp, Zm):
    """(m, k_u, gamma) of the first M LSE^u modes of family n = 0 (robust root finder, B6)."""
    roots = L.lsu_roots_all(M, k, a, Zp, Zm, "LSE")
    return [(m, ku, L.gamma_from_ku(ku, k, 0.0)) for m, ku in zip(range(1, M + 1), roots)]


ORTHO = {}


def junction(M, Zp, Zm, nq=3000):
    """GSM blocks of the junction ideal (port 1) -> impedance-walled (port 2), M modes on each side (B9).

    E continuity is tested with the ideal modes, H continuity with the impedance-walled modes (unconjugated
    reciprocity), which gives a + b = P (c + d), c - d = P^T (a - b) and a symmetric S.
    """
    x, w = np.polynomial.legendre.leggauss(nq)
    up = 0.5 * a * (x + 1)                     # u' in [0, a]
    wq = 0.5 * a * w
    modes = lossy_modes(M, Zp, Zm)
    Fi, Yi, gi = [], [], []
    for m, ku, g in modes:
        X = np.sin(ku * up) - 1j * ku * Zm / (k * ETA0) * np.cos(ku * up)   # Leontovich at u' = 0 built in
        X = X / np.sqrt(b * np.sum(wq * X * X))                             # unconjugated normalization
        if np.real(np.sum(wq * X * np.sin(m * np.pi * up / a))) < 0:
            X = -X
        Fi.append(X)
        Yi.append(g / (k * ETA0))
        gi.append(g)
    Fi, Yi, gi = np.array(Fi), np.array(Yi), np.array(gi)
    G = b * (Fi * wq) @ Fi.T
    ORTHO[M] = np.max(np.abs(G - np.eye(M)))
    Fp = np.array([np.sqrt(2 / (a * b)) * np.sin(m * np.pi * up / a) for m in range(1, M + 1)])
    Yp = np.array([np.sqrt(k * k - (m * np.pi / a) ** 2 + 0j) for m in range(1, M + 1)])
    Yp = np.where(Yp.imag > 0, -Yp, Yp) / (k * ETA0)
    Mo = b * (Fp * wq) @ Fi.T
    P = np.diag(np.sqrt(Yp)) @ Mo @ np.diag(1 / np.sqrt(Yi))
    Id = np.eye(M)
    W = np.linalg.inv(Id + P.T @ P)
    S21 = 2 * W @ P.T
    S22 = W @ (Id - P.T @ P)
    S11 = P @ S21 - Id
    S12 = 2 * P @ W
    return (S11, S12, S21, S22), gi, Fi


def star(A, Bm):
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = Bm
    Id = np.eye(A22.shape[0])
    X1 = np.linalg.inv(Id - B11 @ A22)
    X2 = np.linalg.inv(Id - A22 @ B11)
    return (A11 + A12 @ X1 @ B11 @ A21, A12 @ X1 @ B12, B21 @ X2 @ A21, B22 + B21 @ X2 @ A22 @ B12)


_JCACHE = {}


def filter_gsm(M, Ltot, Zp=Zw, Zm=Zw):
    if (M, Zp, Zm) not in _JCACHE:
        _JCACHE[(M, Zp, Zm)] = junction(M, Zp, Zm)
    J, g, _ = _JCACHE[(M, Zp, Zm)]
    Jr = (J[3], J[2], J[1], J[0])                 # the reversed junction: ports swapped
    prop = np.diag(np.exp(-1j * g * Ltot))
    Z0 = np.zeros((M, M))
    S = star(star(J, (Z0, prop, prop, Z0)), Jr)
    return S, g


results = {}
for M in (20, 40, 80, 160):
    S, g = filter_gsm(M, length)
    S21 = S[2]
    results[M] = ([-DB * np.log(abs(S21[m - 1, m - 1])) for m in (1, 2, 3)], DB * np.log(abs(S21[2, 0])),
                  np.max(np.abs(S[0][:3, :3])), np.max(np.abs(S[0] - S[0].T)), np.max(np.abs(S[2] - S[1].T)))
    t, c, r, rec1, rec2 = results[M]
    print(f"     M = {M:3d}: H10 {t[0]:.4f} dB, H20 {t[1]:.4f} dB, H30 {t[2]:.4f} dB;  H10->H30 {c:.3f} dB;"
          f"  |S11| <= {r:.1e};  reciprocity {max(rec1, rec2):.1e};  mode orthogonality {ORTHO[M]:.1e}")
t, c = results[160][0], results[160][1]
dt = max(abs(np.array(results[160][0]) - np.array(results[80][0])))
conv_change = abs(results[160][1] - results[80][1])
check("M3 converged in the number of modes (80 -> 160: change < 0.005 dB, conversion < 0.05 dB)", dt < 5e-3 and conv_change < 5e-2,
      f"{dt:.1e} dB, conversion {conv_change:.1e} dB")
check("M3 reciprocity of the filter GSM", max(results[160][3], results[160][4]) < 1e-10, f"{max(results[160][3], results[160][4]):.1e}")
meas = {1: 0.6, 2: 2.5, 3: 5.4}
for m in (1, 2, 3):
    check(f"exact method H{m}0 vs measurement {meas[m]} dB (10 %)", abs(t[m - 1] / meas[m] - 1) < 0.10,
          f"{t[m - 1]:.3f} dB ({100 * (t[m - 1] / meas[m] - 1):+.1f} %)")
for m in (1, 2, 3):
    print(f"     H{m}0: M2 (first-order coupled modes) {m2[m]:.4f} dB vs M3 (exact) {t[m - 1]:.4f} dB: "
          f"{100 * (m2[m] / t[m - 1] - 1):+.2f} %")

# exact propagation constants vs first order
print("\n     exact vs first-order attenuation of the impedance-walled modes, family n = 0:")
for m, ku, g in lossy_modes(6, Zw, Zw)[:5]:
    lam1 = L.ls_first_order(m, 0, a, b, f, Zwalls)["LSE_u"]
    print(f"     m = {m}: exact alpha = {-g.imag * DB:8.3f} dB/m, first order {lam1.real * DB:8.3f} dB/m "
          f"({100 * (lam1.real / -g.imag - 1):+.2f} %); exact delta beta = {g.real - np.sqrt(k*k - (m*np.pi/a)**2):+.3f} rad/m")

# ------------------------------------------------------------------------------------------------- conversion
print("\n     H10 -> H30 conversion vs length (exact M3, M = 80; first-order M2 with TE10-TE30; Butterweck eq. 30)")
beta1, beta3 = (np.sqrt(k * k - (m * np.pi / a) ** 2) for m in (1, 3))
gam1 = 1j * beta1 + alpha1[1]          # Butterweck's gamma (e^{-gamma z})
gam3 = 1j * beta3 + alpha1[3]
print(f"     beat length 2 pi/(beta1 - beta3) = {2 * np.pi / (beta1 - beta3) * 100:.2f} cm")
Ls = np.linspace(0.01, 0.30, 59)
ex, fo, bw = [], [], []
for Lz in Ls:
    S, _ = filter_gsm(80, Lz)
    ex.append(DB * np.log(abs(S[2][2, 0])))
    B3 = ModeBasis(a, b, [Mode("TE", m, 0) for m in (1, 2, 3)])
    T, _ = tr_from_phi(expm((L.D_matrix(B3, f) - L.loss_matrix_full(B3, f, Zwalls)) * Lz), 3)
    fo.append(DB * np.log(abs(T[2, 0])))
    # Butterweck eq. (30), as printed (nu = 3): C3/C1(0) = 4 Zw sin(3 pi/2)/(j Z0 k a) pi^2/(a^2 (g1^2 - g3^2)) (e^-g1 z - e^-g3 z)
    c30 = 4 * Zw * np.sin(1.5 * np.pi) / (1j * ETA0 * k * a) * np.pi ** 2 / (a * a * (gam1 ** 2 - gam3 ** 2)) * (np.exp(-gam1 * Lz) - np.exp(-gam3 * Lz))
    bw.append(DB * np.log(abs(c30) * np.sqrt(beta3 / beta1)))
ex, fo, bw = map(np.array, (ex, fo, bw))
for Lz in (0.05, 0.10, 0.15, 0.20, 0.25):
    j = np.argmin(abs(Ls - Lz))
    print(f"     L = {Ls[j] * 100:5.1f} cm: exact {ex[j]:7.2f} dB, first-order coupled modes {fo[j]:7.2f} dB, "
          f"Butterweck eq. (30) as printed {bw[j]:7.2f} dB, with the factor nu = 3 {bw[j] + DB * np.log(3):7.2f} dB")
S15, _ = filter_gsm(160, length)
conv15 = DB * np.log(abs(S15[2][2, 0]))
pref = 4 * Zw / (ETA0 * k * a * 8)
lo = pref * (np.exp(-alpha1[1] * length) - np.exp(-alpha1[3] * length))
hi = pref * (np.exp(-alpha1[1] * length) + np.exp(-alpha1[3] * length))
print(f"     Butterweck eq. (31) evaluated with his own formula at 15 cm: {DB*np.log(lo):.1f} to {DB*np.log(hi):.1f} dB "
      f"(printed: -36 to -24.5 dB); with the factor nu = 3: {DB*np.log(3*lo):.1f} to {DB*np.log(3*hi):.1f} dB")
print(f"     exact envelope over 1-30 cm: max {ex.max():.2f} dB; exact at 15 cm (M = 160): {conv15:.2f} dB")
check("exact H10->H30 conversion at 15 cm within Butterweck's -36..-24.5 dB band", -36 <= conv15 <= -24.5, f"{conv15:.2f} dB")
check("exact vs first-order coupled-mode conversion at 15 cm (within 1 dB)", abs(conv15 - conv2) < 1.0,
      f"{conv15:.2f} vs {conv2:.2f} dB")

dd = np.abs(ex - np.array([b_ + DB * np.log(3) for b_ in bw]))
print(f"     |exact - Butterweck eq. (30) with nu = 3| over 1-30 cm: max {dd.max():.2f} dB, at 15 cm {dd[np.argmin(abs(Ls - 0.15))]:.2f} dB")

# ------------------------------------------------------------------------------------------------- drift
print("\n     first-order system B4 in the ideal basis: drift of the TE10 eigenvalue with the number of TE_m0 modes")
b1 = np.sqrt(k * k - (np.pi / a) ** 2)
for Zt in (250.0, 2.5):
    ku = L.lsu_roots_all(2, k, a, Zt, Zt, "LSE")[0]
    lam_ex = 1j * (L.gamma_from_ku(ku, k, 0.0) - b1)
    row = []
    for M in (3, 11, 40, 80):
        Bt = ModeBasis(a, b, [Mode("TE", m, 0) for m in range(1, M + 1)], sort=False)
        ev = np.linalg.eigvals(L.D_matrix(Bt, f) - L.loss_matrix_full(Bt, f, {"u+": Zt, "u-": Zt}))
        lam = -(ev[np.argmin(abs(ev + 1j * b1))] + 1j * b1)
        row.append(f"{M} modes: |dLambda|/|Lambda| {abs(lam - lam_ex) / abs(lam_ex):.2e} (Re: {lam.real / lam_ex.real - 1:+.1e})")
    print(f"     Z = {Zt:5.1f} ohm: " + ";  ".join(row))

# ------------------------------------------------------------------------------------------------- n >= 1
print("\n     where the first-order treatment fails: modes with n >= 1 at Z_w = 250 ohm (exact LS^u vs first order)")
for (m, n, kind) in ((1, 1, "LSE"), (1, 1, "LSM"), (0, 1, "LSM"), (2, 1, "LSM")):
    kn = n * np.pi / b
    for Zt in (25.0, 250.0):
        ku = L.lsu_roots_all(m + 1, k, a, Zt, Zt, kind)[m if kind == "LSM" else m - 1]
        g = L.gamma_from_ku(ku, k, kn)
        lam1 = L.ls_first_order(m, n, a, b, f, {"u+": Zt, "u-": Zt})[kind + "_u"]
        print(f"     {kind}^u({m},{n}), Z = {Zt:5.0f} ohm: exact alpha {-g.imag * DB:9.2f} dB/m, first order "
              f"{lam1.real * DB:9.2f} dB/m ({100 * (lam1.real / -g.imag - 1):+.1f} %)")

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
