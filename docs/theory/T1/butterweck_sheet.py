"""Resistive sheet parallel to the broad walls (Butterweck, IEEE MTT-16, 1968, eqs. 1-19).

S1  Dispersion relations of the LS^v modes (B7, B8): closed form vs the transfer form (any number of sheets) vs
    Butterweck eqs. (8), (9) for a centred sheet.
S2  Butterweck's example (b = 10 mm, lambda = 4 mm): optimum R_sheet and maximum attenuation of EH11^(2) (LSM^v,
    H_v = 0), exact propagation constant; also with his approximation beta ~ k. Theory values: 56 ohm, 1.5 dB/cm.
    EH11^(1) (LSE^v): his values 420 ohm and about 5 dB/cm.
S3  Off-centre sheet: attenuation of EH11^(2) vs the sheet position.
S4  Junction empty guide -> sheet-loaded guide by overlap integrals (B10): R -> infinity reproduces the LS rotation
    of CONVENTIONS.md section 7 (wrmodes.modes.ls_rotation); sheet-mode orthogonality; reciprocity and
    passivity of the junction; a 10 cm filter example.

Run:  PYTHONPATH=<worktree>/src python3 docs/theory/T1/butterweck_sheet.py
"""
import sys
import numpy as np
import scipy.constants as cst
from scipy.optimize import minimize_scalar
from numpy.polynomial.legendre import leggauss

import t1lib as L
from wrmodes.modes import ModeBasis, Mode, ETA0, transverse_field, ls_rotation

DB = 20 / np.log(10)
FAILS = []


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------------ S1
lam0 = 4e-3
f = cst.c / lam0
k = 2 * np.pi / lam0
a, b = 22.86e-3, 10e-3
rng = np.random.default_rng(3)
worst = 0.0
for kind in ("LSM", "LSE"):
    for _ in range(20):
        kv = rng.uniform(100, 2000) + 1j * rng.uniform(-50, 50)
        xi = rng.uniform(0.01, 2.0)
        v0 = rng.uniform(0.05, 0.95) * b
        Ft = L.sheet_dispersion(kv, k, b, [xi], [v0], kind)
        Fc = L.sheet_dispersion_closed(kv, k, b, xi, v0, kind)
        # transfer form: LSM returns Y'(b) = -k_v F/(xi k) ; LSE returns Y(b) = F / (xi k_v^2)
        ratio = Ft / Fc
        expect = -kv / (xi * k) if kind == "LSM" else 1 / (xi * kv * kv)
        worst = max(worst, abs(ratio / expect - 1))
check("S1  transfer form of the sheet dispersion = closed form B7 (random k_v, xi, v0)", worst < 1e-10, f"{worst:.1e}")
worst = 0.0
for kind in ("LSM", "LSE"):
    for _ in range(20):
        kv = rng.uniform(100, 2000) + 1j * rng.uniform(-50, 50)
        xi = rng.uniform(0.01, 2.0)
        Fc = L.sheet_dispersion_closed(kv, k, b, xi, b / 2, kind)
        Fb = L.butterweck_sheet_eq(kv, k, b, xi, kind)
        fac = (k if kind == "LSM" else kv) * np.sin(kv * b / 2) * np.cos(kv * b / 2)
        worst = max(worst, abs(Fc - fac * Fb) / abs(Fc))
check("S1  centred sheet: B7 = sin cos x Butterweck (9) (LSM) and (8) (LSE)", worst < 1e-10, f"{worst:.1e}")


# ------------------------------------------------------------------------------------------------ S2
def track_root(kind, xi_target, v0, n=1, m=1, xis=None):
    """k_v of the LS^v mode (u-index m irrelevant for k_v) that tends to n pi / b for xi -> infinity."""
    if xis is None:
        xis = np.geomspace(1e4, xi_target, 400)
    kv = n * np.pi / b + 1e-9
    for xi in xis:
        kv = L.sheet_root(kv, k, b, [xi], [v0], kind)
    return kv


def alpha_exact(kv, m=1):
    return -L.gamma_from_ku(kv, k, m * np.pi / a).imag


def alpha_butterweck(kv):
    return (kv * kv).imag / (2 * k)        # eq. (14): alpha ~ Im k_y^2 / (2 k), beta ~ k ; sign: e^{+i w t}, see note


def scan(kind, v0, approx=False):
    def neg_alpha(logxi):
        kv = track_root(kind, np.exp(logxi), v0)
        return -(abs(alpha_butterweck(kv)) if approx else alpha_exact(kv))
    res = minimize_scalar(neg_alpha, bounds=(np.log(0.01), np.log(10.0)), method="bounded",
                          options={"xatol": 1e-6})
    return np.exp(res.x) * ETA0, -res.fun


R2, a2 = scan("LSM", b / 2)
R2b, a2b = scan("LSM", b / 2, approx=True)
print(f"     EH11^(2) (LSM^v), centred sheet, exact gamma: optimum R = {R2:.2f} ohm, alpha_max = {a2 * DB / 100:.3f} dB/cm")
print(f"     EH11^(2), Butterweck's approximation alpha = Im k_v^2/(2k): R = {R2b:.2f} ohm, alpha_max = {a2b * DB / 100:.3f} dB/cm")
print(f"     eq. (16): 0.37 Z0 lambda/b = {0.37 * ETA0 * lam0 / b:.1f} ohm; eq. (15): 2.76/(k b^2) = {2.76 / (k * b * b) * DB / 100:.3f} dB/cm")
check("S2  EH11^(2) optimum sheet resistance vs Butterweck 56 ohm (5 %)", abs(R2 / 56 - 1) < 0.05, f"{R2:.2f} ohm ({100 * (R2 / 56 - 1):+.1f} %)")
check("S2  EH11^(2) maximum attenuation vs Butterweck 1.5 dB/cm (5 %)", abs(a2 * DB / 100 / 1.5 - 1) < 0.05,
      f"{a2 * DB / 100:.3f} dB/cm ({100 * (a2 * DB / 100 / 1.5 - 1):+.1f} %)")
kv_opt = track_root("LSM", R2 / ETA0, b / 2)
u_ = kv_opt * b / 2
print(f"     at the optimum: u = k_v b/2 = {u_:.4f}, x y = {u_.real * u_.imag:.3f} (Butterweck: 0.69), 2 xi v = {2 * R2 / ETA0 * k * b / 2:.3f} (2.3)")
R1, a1 = scan("LSE", b / 2)
print(f"     EH11^(1) (LSE^v), centred sheet: optimum R = {R1:.1f} ohm, alpha_max = {a1 * DB / 100:.2f} dB/cm "
      f"(Butterweck: 420 ohm, about 5 dB/cm; ratio to EH11^(2) {a1 / a2:.2f}, his 3.2)")
for Rt in (56.0, 70.0):
    kv = track_root("LSM", Rt / ETA0, b / 2)
    kv1 = track_root("LSE", Rt / ETA0, b / 2)
    print(f"     R = {Rt:.0f} ohm: EH11^(2) {alpha_exact(kv) * DB / 100:.3f} dB/cm, EH11^(1) {alpha_exact(kv1) * DB / 100:.3f} dB/cm, "
          f"TE10 (LSM^v, n = 0) unaffected: k_v = 0 is an exact root")

# ------------------------------------------------------------------------------------------------ S3
print("\nS3  off-centre sheet: EH11^(2)-like mode (LSM^v root from n = 1), R = 56 ohm")
for frac in (0.5, 0.4, 0.3, 0.2, 0.1):
    kv = track_root("LSM", 56 / ETA0, frac * b)
    kv1 = track_root("LSE", 56 / ETA0, frac * b)
    print(f"     v0'/b = {frac:.1f}: LSM^v n=1 {alpha_exact(kv) * DB / 100:.3f} dB/cm;  LSE^v n=1 {alpha_exact(kv1) * DB / 100:.3f} dB/cm")
kv_a = track_root("LSM", 56 / ETA0, 0.3 * b)
kv_b = track_root("LSM", 56 / ETA0, 0.7 * b)
check("S3  mirror symmetry: sheet at v0 and b - v0 give the same k_v", abs(kv_a - kv_b) < 1e-9 * abs(kv_a), f"{abs(kv_a - kv_b):.1e}")


# ------------------------------------------------------------------------------------------------ S4
def sheet_modes(m, nmax, xis, v0s):
    """LS^v modes of u-index m of a guide with sheets: list of (kind, n, k_v, Y-function pieces)."""
    out = []
    for kind in ("LSM", "LSE"):
        for n in range(0 if kind == "LSM" else 1, nmax + 1):
            if kind == "LSM" and n == 0:
                out.append((kind, 0, 0.0 + 0j))    # Y = const: TE_m0, exact root for any sheets
                continue
            kv = n * np.pi / b + 1e-9
            for t in np.geomspace(1e4, 1.0, 300):
                kv = L.sheet_root(kv, k, b, [x * t for x in xis], v0s, kind)
            out.append((kind, n, kv))
    return out


def Yfun(kind, kv, xis, v0s, vp):
    """Y(v') and Y'(v') by the transfer form (piecewise), on the points vp (sorted)."""
    order = np.argsort(v0s)
    xis = np.asarray(xis, dtype=complex)[order]
    v0s = np.asarray(v0s)[order]
    Y0, dY0 = (1.0 + 0j, 0j) if kind == "LSM" else (0j, 1.0 + 0j)
    bounds = [0.0] + list(v0s) + [b]
    starts = []
    Y, dY = Y0, dY0
    for i in range(len(bounds) - 1):
        starts.append((bounds[i], Y, dY))
        if i < len(v0s):
            Y, dY = L._advance(Y, dY, kv, bounds[i + 1] - bounds[i])
            if kind == "LSM":
                Y = Y - 1j * dY / (k * xis[i])
            else:
                dY = dY + 1j * k * Y / xis[i]
    Yv = np.zeros(len(vp), dtype=complex)
    dYv = np.zeros(len(vp), dtype=complex)
    for i, (s, Ys, dYs) in enumerate(starts):
        sel = (vp >= s) & (vp < bounds[i + 1]) if i < len(starts) - 1 else (vp >= s)
        Yv[sel], dYv[sel] = L._advance(Ys, dYs, kv, vp[sel] - s)
    return Yv, dYv


def sheet_fields(kind, kv, m, xis, v0s, up, vp):
    """(E_u, E_v, H_u, H_v) of an LS^v mode on the grid (up, vp), primed coordinates, unnormalized (e^{-i g z})."""
    km = m * np.pi / a
    g = L.gamma_from_ku(kv, k, km)
    Y, dY = Yfun(kind, kv, xis, v0s, vp)
    we, wm = k / ETA0, k * ETA0
    if kind == "LSM":     # H_v = 0
        Eu = km / (1j * we) * np.outer(np.cos(km * up), dY)
        Ev = (g * g + km * km) / (1j * we) * np.outer(np.sin(km * up), Y)
        Hu = 1j * g * np.outer(np.sin(km * up), Y)
        Hv = np.zeros_like(Hu)
    else:                 # E_v = 0
        Eu = -1j * g * np.outer(np.cos(km * up), Y)
        Ev = np.zeros_like(Eu)
        Hu = -km / (1j * wm) * np.outer(np.sin(km * up), dY)
        Hv = (g * g + km * km) / (1j * wm) * np.outer(np.cos(km * up), Y)
    return Eu, Ev, Hu, Hv, g


def piecewise_gauss(breaks, n):
    x, w = leggauss(n)
    pts, wts = [], []
    for s, e in zip(breaks[:-1], breaks[1:]):
        pts.append(0.5 * (e - s) * x + 0.5 * (e + s))
        wts.append(0.5 * (e - s) * w)
    return np.concatenate(pts), np.concatenate(wts)


def junction_sheet(m, nmax, xis, v0s, nq=120):
    up, wu = piecewise_gauss([0, a], 200)
    vp, wv = piecewise_gauss([0.0] + sorted(v0s) + [b], nq)
    W = np.outer(wu, wv)
    modes = sheet_modes(m, nmax, xis, v0s)
    Et, Ht, gs = [], [], []
    for kind, n, kv in modes:
        Eu, Ev, Hu, Hv, g = sheet_fields(kind, kv, m, xis, v0s, up, vp)
        nrm = 0.5 * np.sum(W * (Eu * Hv - Ev * Hu))
        s = np.sqrt(nrm)
        Et.append((Eu / s, Ev / s))
        Ht.append((Hu / s, Hv / s))
        gs.append(g)
    ideal = [Mode("TE", m, 0)] + [Mode(kd, m, n) for n in range(1, nmax + 1) for kd in ("TE", "TM")]
    Bm = ModeBasis(a, b, ideal)
    U, V = np.meshgrid(up - a / 2, vp - b / 2, indexing="ij")
    ep = [transverse_field(p, a, b, U, V) for p in Bm.modes]
    Yp = Bm.admittance(f)
    Pt = np.array([[np.sum(W * (e[0] * E[0] + e[1] * E[1])) for E in Et] for e in ep])
    P = np.diag(np.sqrt(Yp / 2)) @ Pt
    # orthogonality of the sheet modes (unconjugated reaction)
    O = np.array([[0.5 * np.sum(W * (Ei[0] * Hj[1] - Ei[1] * Hj[0])) for Hj in Ht] for Ei in Et])
    # sign convention: make the overlap with the 'parent' ideal combination positive on the diagonal block
    Id = np.eye(len(Bm))
    Wm = np.linalg.inv(Id + P.T @ P)
    S11 = 2 * P @ Wm @ P.T - Id
    S21 = 2 * Wm @ P.T
    S12 = 2 * P @ Wm
    S22 = Wm @ (Id - P.T @ P)
    return Bm, modes, np.array(gs), P, O, (S11, S12, S21, S22)


# Butterweck's example (b = 10 mm, lambda = 4 mm) has k_t = 0 exactly for n = 5, where the LS rotation of the
# empty guide is singular (CONVENTIONS.md section 7); the junction checks use the real X-band height instead.
b = 10.16e-3
print("\nS4  junction empty guide -> sheet-loaded guide (m = 1, n <= 8), b = 10.16 mm, lambda = 4 mm")
# R -> infinity: the sheet modes are the LS^v modes of the empty guide, P must be the LS rotation (up to signs)
Bm, modes, gs, P, O, S = junction_sheet(1, 8, [1e7], [b / 2])
Q, labels = ls_rotation(Bm, f)
# rows of Q: LS amplitudes in terms of TE/TM; P maps sheet amplitudes to ideal ones: a_ideal = P a_sheet -> P ~ Q^T
lab_sheet = [(kd, 1, n) for kd, n, _ in modes]
perm = [lab_sheet.index(lb) for lb in labels]
Pp = P[:, perm]
dev = np.max(np.abs(np.abs(Pp) - np.abs(Q.T)))
check("S4  R -> infinity: |P| = |Q^T| of ls_rotation (sheet modes -> LS^v modes of CONVENTIONS 7)", dev < 1e-6, f"{dev:.1e}")
for Rs_ in (56.0, 300.0):
    Bm, modes, gs, P, O, S = junction_sheet(1, 8, [Rs_ / ETA0], [b / 2])
    Sfull = np.block([[S[0], S[1]], [S[2], S[3]]])
    prop = Bm.propagating_mask(f)
    nsp = np.array([abs(g.imag) < abs(g.real) for g in gs])
    check(f"S4  R = {Rs_:.0f} ohm: sheet modes orthogonal (unconjugated reaction)", np.max(np.abs(O - np.eye(len(O)))) < 1e-9,
          f"{np.max(np.abs(O - np.eye(len(O)))):.1e}")
    check(f"S4  R = {Rs_:.0f} ohm: junction reciprocal", np.max(np.abs(Sfull - Sfull.T)) < 1e-10, f"{np.max(np.abs(Sfull - Sfull.T)):.1e}")
    i10 = Bm.index(Mode("TE", 1, 0))
    print(f"     R = {Rs_:.0f} ohm: |S11| of TE10 = {abs(S[0][i10, i10]):.1e}, |S21| TE10 -> sheet n=0 LSM = {abs(S[2][0, i10]):.6f}")

# 10 cm filter, one centred sheet of 56 ohm: transmission of TE10, TE11, TM11, TE12, TM12
Bm, modes, gs, P, O, J = junction_sheet(1, 12, [56 / ETA0], [b / 2])


def star(A, Bm_):
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = Bm_
    Id = np.eye(A22.shape[0])
    X1 = np.linalg.inv(Id - B11 @ A22)
    X2 = np.linalg.inv(Id - A22 @ B11)
    return (A11 + A12 @ X1 @ B11 @ A21, A12 @ X1 @ B12, B21 @ X2 @ A21, B22 + B21 @ X2 @ A22 @ B12)


Lf = 0.10
pr = np.diag(np.exp(-1j * gs * Lf))
Z0 = np.zeros_like(pr)
Sf = star(star(J, (Z0, pr, pr, Z0)), (J[3], J[2], J[1], J[0]))
print("     10 cm filter, centred 56 ohm sheet, f = 74.95 GHz, a = 22.86 mm, b = 10.16 mm, n <= 12: transmission (dB)")
for md in (("TE", 1, 0), ("TE", 1, 1), ("TM", 1, 1), ("TE", 1, 2), ("TM", 1, 2), ("TE", 1, 3), ("TM", 1, 3)):
    i = Bm.index(Mode(*md))
    print(f"       {Mode(*md).name}: {DB * np.log(abs(Sf[2][i, i]) + 1e-300):9.3f} dB   (total transmitted power {np.sum(np.abs(Sf[2][:, i]) ** 2):.3e})")
sv = np.linalg.svd(np.block([[Sf[0], Sf[1]], [Sf[2], Sf[3]]])[np.ix_(np.r_[Bm.propagating_mask(f), Bm.propagating_mask(f)], np.r_[Bm.propagating_mask(f), Bm.propagating_mask(f)])], compute_uv=False)
check("S4  passivity of the 10 cm sheet filter (largest singular value over propagating modes <= 1)", sv[0] <= 1 + 1e-12, f"{sv[0]:.12f}")

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
