"""Numerical checks of the coefficients of docs/theory/T1-wall-impedance.md (quadrature and exact roots).

Run:  PYTHONPATH=<worktree>/src python3 docs/theory/T1/check_coefficients.py
Exits with status 1 if a check fails.

Q1  Closed-form wall integrals (B2) vs Gauss-Legendre quadrature of the fields of wrmodes.modes, every pair of
    modes, every wall, straight (order 0) and bend (order 1) parts. WR-90 at 90 GHz (129 modes) and a square
    20 x 20 mm guide at 60 GHz. Target 1e-8 (relative to the diagonal scale sqrt(W_pp W_qq)).
Q2  Lambda^+- from closed forms vs from quadrature. Target 1e-8.
Q3  Diagonal of Lambda^+ vs Marcuvitz 2.2(4), 2.2(10) and the card's TE10 formula. Target 1e-6.
Q4  Symmetry of Lambda^+- (reciprocity).
Q5  One pair of opposite walls: eigenvalues of the 2 x 2 (TE_mn, TM_mn) blocks = LS first-order closed forms
    (B5), eigenvectors = LS rotations (wrmodes.modes.ls_rotation for the broad walls).
Q6  Exact LS^u roots (B6) vs the first-order eigenvalues: the difference is O(Z^2).
Q7  Power balance: -dP/dz from Lambda_full equals 1/2 Re oint h Z (|H_t|^2 + |H_z|^2) dl evaluated by quadrature
    of the reconstructed wall fields, for random amplitudes, unequal walls and a bend. Target 1e-10.
"""
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss

import t1lib as L
from wrmodes.modes import ModeBasis, Mode, ETA0, transverse_field, divergence, ls_rotation

FAILS = []


def check(name, value, tol):
    ok = value <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.2e} (target {tol:g})")
    if not ok:
        FAILS.append(name)


CASES = [("WR-90, 90 GHz", 22.86e-3, 10.16e-3, 90e9), ("square 20 x 20 mm, 60 GHz", 20e-3, 20e-3, 60e9)]

# ---------------------------------------------------------------------------------------------- Q1, Q2
for label, a, b, f in CASES:
    B = ModeBasis.propagating(f, a, b)
    te = np.array([m.kind == "TE" for m in B.modes])
    sN = sum(np.diag(L.wall_matrices(B, w, 0)[0]) for w in L.WALLS)
    sG = sum(np.diag(L.wall_matrices(B, w, 0)[1]) for w in L.WALLS)
    errN = errG = errNrel = 0.0
    for w in L.WALLS:
        for order in (0, 1):
            N1, G1 = L.wall_matrices(B, w, order)
            N2, G2 = L.wall_matrices_quadrature(B, w, order, npts=600)
            scale = (a / 2) ** order
            errN = max(errN, np.max(np.abs(N1 - N2) / (np.sqrt(np.outer(sN, sN)) * scale)))
            mask = np.outer(te, te)
            errG = max(errG, np.max(np.abs(G1 - G2)[mask] / (np.sqrt(np.outer(sG, sG))[mask] * scale)))
            big = np.abs(N2) > 1e-6 * np.sqrt(np.outer(sN, sN)) * scale
            errNrel = max(errNrel, np.max(np.abs(N1 - N2)[big] / np.abs(N2)[big]))
    print(f"-- {label}: {len(B)} modes")
    check(f"Q1  N wall integrals, closed form vs quadrature, {label}", errN, 1e-8)
    check(f"Q1  G wall integrals, closed form vs quadrature, {label}", errG, 1e-8)
    check(f"Q1  N, entrywise relative error on non-negligible entries, {label}", errNrel, 1e-8)
    Z = {"u+": 0.11 + 0.13j, "u-": 0.07 + 0.07j, "v+": 0.2 + 0.1j, "v-": 0.05 + 0.3j}
    for order in (0, 1):
        Lp1, Lm1 = L.loss_matrices(B, f, Z, order)
        Lp2, Lm2 = L.loss_matrices(B, f, Z, order, quadrature=True)
        e = max(np.max(np.abs(Lp1 - Lp2)), np.max(np.abs(Lm1 - Lm2))) / np.max(np.abs(Lp1))
        check(f"Q2  Lambda^+- (order {order}) closed form vs quadrature, unequal walls, {label}", e, 1e-8)
        check(f"Q4  Lambda^+- symmetric (order {order}), {label}",
              max(np.max(np.abs(Lp1 - Lp1.T)), np.max(np.abs(Lm1 - Lm1.T))) / np.max(np.abs(Lp1)), 1e-14)

# ---------------------------------------------------------------------------------------------- Q3
worst = 0.0
for a, b in ((22.86e-3, 10.16e-3), (20e-3, 20e-3), (10.16e-3, 22.86e-3)):
    for f in np.linspace(18e9, 90e9, 13):
        sigma = 3.5e7
        B = ModeBasis.propagating(f, a, b)
        Rs = L.surface_resistance(f, sigma)
        Lp, _ = L.loss_matrices(B, f, L.all_walls(L.surface_impedance(f, sigma)))
        ref = np.array([L.attenuation_marcuvitz(m, a, b, f, Rs) for m in B.modes])
        worst = max(worst, np.max(np.abs(Lp.diagonal().real / ref - 1)), np.max(np.abs(Lp.diagonal().imag / ref - 1)))
check("Q3  Re and Im of diag(Lambda^+) vs Marcuvitz, WR-90 (both orientations) and square, 18-90 GHz, all modes",
      worst, 1e-6)
a, b = 22.86e-3, 10.16e-3
worst = 0.0
for f in np.linspace(7e9, 90e9, 50):
    B = ModeBasis(a, b, [Mode("TE", 1, 0)])
    Lp, _ = L.loss_matrices(B, f, L.all_walls(L.surface_impedance(f, 3.5e7)))
    worst = max(worst, abs(Lp[0, 0].real / L.attenuation_te10_card(a, b, f, L.surface_resistance(f, 3.5e7)) - 1))
check("Q3  TE10 vs the card's formula, WR-90, 7-90 GHz", worst, 1e-6)

# ---------------------------------------------------------------------------------------------- Q5
a, b, f = 22.86e-3, 10.16e-3, 60e9
k = L.wavenumber(f)
worst_val = worst_vec = 0.0
for m in range(0, 7):
    for n in range(0, 4):
        if m + n == 0 or np.hypot(m / a, n / b) * np.pi >= k:
            continue
        modes = [Mode("TE", m, n)] + ([Mode("TM", m, n)] if m * n > 0 else [])
        B = ModeBasis(a, b, modes)
        for walls in (("u+", "u-"), ("v+", "v-")):
            Z = {walls[0]: 0.8 + 0.3j, walls[1]: 0.5 + 0.5j}
            Lp, _ = L.loss_matrices(B, f, Z)
            ls = L.ls_first_order(m, n, a, b, f, Z)
            if walls[0] == "u+":
                cand = ([ls["LSE_u"]] if m >= 1 else []) + ([ls["LSM_u"]] if n >= 1 else [])
            else:
                cand = ([ls["LSE_v"]] if n >= 1 else []) + ([ls["LSM_v"]] if m >= 1 else [])
            ev = np.sort_complex(np.linalg.eigvals(Lp))
            worst_val = max(worst_val, np.max(np.abs(ev - np.sort_complex(np.array(cand)))) / np.max(np.abs(ev)))
            if walls[0] == "v+" and m * n > 0:
                Q, labels = ls_rotation(B, f)       # rows: LSE (TE row), LSM (TM row)
                LQ = Q @ Lp @ Q.T                  # Lambda in the LS^v basis must be diagonal
                worst_vec = max(worst_vec, abs(LQ[0, 1]) / np.max(np.abs(LQ)))
check("Q5  eigenvalues of the (TE_mn, TM_mn) blocks = LS first-order closed forms, WR-90, 60 GHz", worst_val, 1e-12)
check("Q5  broad walls only: Lambda is diagonal in the LS^v basis of ls_rotation (off-diagonal / max)", worst_vec, 1e-12)

# ---------------------------------------------------------------------------------------------- Q6
print("-- Q6: exact LS^u roots vs first order, WR-90 at 40 GHz (relative difference for Z scaled by s)")
a, b, f = 22.86e-3, 10.16e-3, 40e9
k = L.wavenumber(f)
slopes = []
for m, n, kind in ((1, 0, "LSE"), (3, 0, "LSE"), (1, 1, "LSE"), (1, 1, "LSM"), (2, 1, "LSM"), (0, 1, "LSM")):
    kn = n * np.pi / b
    g0 = np.sqrt(k * k - kn * kn - (m * np.pi / a) ** 2)
    errs = []
    for s in (10.0, 1.0, 0.1, 0.01):
        Zp, Zm = (1 + 0.3j) * s, 0.7 * s
        ku = L.lsu_root(m, k, a, Zp, Zm, kind)
        lam_exact = 1j * (L.gamma_from_ku(ku, k, kn) - g0)
        lam1 = L.ls_first_order(m, n, a, b, f, {"u+": Zp, "u-": Zm})[kind + "_u"]
        errs.append(abs(lam_exact - lam1) / abs(lam1))
    slope = np.log10(errs[1] / errs[3]) / 2
    slopes.append(slope)
    print(f"     {kind}^u ({m},{n}): rel. diff {errs[0]:.2e} (|Z|~10 ohm), {errs[1]:.2e}, {errs[2]:.2e}, {errs[3]:.2e} (0.01 ohm); slope {slope:.4f}")
check("Q6  first-order error is O(Z^2): slope of the relative difference vs Z (|slope - 1|)", max(abs(s - 1) for s in slopes), 0.005)

# ---------------------------------------------------------------------------------------------- Q7
rng = np.random.default_rng(1)
a, b, f = 22.86e-3, 10.16e-3, 45e9
k = L.wavenumber(f)
B = ModeBasis.propagating(f, a, b)
Y = B.admittance(f).real
Z = {"u+": 0.11 + 0.13j, "u-": 0.07 + 0.02j, "v+": 0.3 + 0.1j, "v-": 0.05 + 0.3j}
worst = 0.0
for kappa in (0.0, 4.0):
    Lfull = L.loss_matrix_full(B, f, Z, kappa)
    for trial in range(3):
        ap = rng.normal(size=len(B)) + 1j * rng.normal(size=len(B))
        am = rng.normal(size=len(B)) + 1j * rng.normal(size=len(B))
        y = np.concatenate([ap, am])
        dy = -Lfull @ y
        n = len(B)
        dP = 2 * np.real(np.vdot(ap, dy[:n]) - np.vdot(am, dy[n:]))
        Vq = (ap + am) * np.sqrt(2 / Y)
        Iq = (ap - am) * np.sqrt(2 * Y)
        x, wq = leggauss(400)
        diss = 0.0
        for wall, Zw in Z.items():
            if wall in ("u+", "u-"):
                u0 = a / 2 if wall == "u+" else -a / 2
                v = 0.5 * b * x
                u = np.full_like(v, u0)
                dl = 0.5 * b * wq
                nrm = (1.0, 0.0) if wall == "u+" else (-1.0, 0.0)
            else:
                v0 = b / 2 if wall == "v+" else -b / 2
                u = 0.5 * a * x
                v = np.full_like(u, v0)
                dl = 0.5 * a * wq
                nrm = (0.0, 1.0) if wall == "v+" else (0.0, -1.0)
            Ht = sum(I_ * (e[0] * nrm[0] + e[1] * nrm[1]) for I_, e in zip(Iq, (transverse_field(p, a, b, u, v) for p in B.modes)))
            Hz = sum(-1j / (k * ETA0) * V_ * divergence(p, a, b, u, v)[1] for V_, p in zip(Vq, B.modes))
            h = 1 + kappa * u
            diss += 0.5 * Zw.real * np.sum(dl * h * (np.abs(Ht) ** 2 + np.abs(Hz) ** 2))
        worst = max(worst, abs(-dP - diss) / diss)
check("Q7  power balance -dP/dz = wall dissipation (random amplitudes, 4 unequal walls, kappa = 0 and 4 /m)", worst, 1e-10)

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
