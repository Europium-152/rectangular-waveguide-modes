"""Checks of items 2-4 of T1b: wall integrals, Lambda coefficients, textbook attenuation, bend moments.

Run from the repository root:   PYTHONPATH=src python3 docs/theory/T1b/check_coefficients.py

1. Quadrature: Lambda^{++}, Lambda^{+-}, Lambda^{-+}, Lambda^{--} are computed directly from the reciprocity
   integrals  -(1/4) oint Z h_q^s . h_p^s' dl  with the full PEC mode fields (transverse field from
   wrmodes.modes.transverse_field, H_z from wrmodes.modes.divergence) on the four walls, each wall with its own Z,
   and compared with the closed forms of t1b_coefficients (target 1e-8 relative). The power-loss integral
   (1/2) R_s oint |H_tan|^2 is compared with 2 Re Lambda^{++}_pp. The first moments (weight u) are checked the same
   way.
2. Sympy: the diagonal Lambda^{++}_pp (equal walls, Z = R_s) equals Marcuvitz's TE_mn and TM_mn attenuation
   constants identically in a, b, k, m, n (cases m, n >= 1; n = 0; m = 0).
3. Numbers: diagonal vs Marcuvitz for WR-90 and a square guide, 18-90 GHz (target 1e-6 relative).
"""
import os
import sys
import itertools
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1b_coefficients as T  # noqa: E402
from wrmodes import modes as WM  # noqa: E402

RNG = np.random.default_rng(1)


def mode_obj(p):
    return WM.Mode(*p)


def fields_on_wall(p, a, b, f, wall, s):
    """Forward (+) and backward (-) PEC magnetic fields of mode p (unit amplitude) at the wall points
    parametrized by s; returns tangential components (t1, z) for each sign, where t1 is the transverse tangent."""
    if wall in ("u-", "u+"):
        u = np.full_like(s, -a / 2 if wall == "u-" else a / 2)
        v = s
    else:
        u = s
        v = np.full_like(s, -b / 2 if wall == "v-" else b / 2)
    eu, ev = WM.transverse_field(mode_obj(p), a, b, u, v)
    d, g = WM.divergence(mode_obj(p), a, b, u, v)
    Y = T.admittance(p, a, b, f)
    k = T.wavenumber(f)
    # z x e = (-e_v, e_u); transverse tangent: v_hat on u-walls, u_hat on v-walls
    ht = eu if wall in ("u-", "u+") else -ev
    Hz = -1j * np.sqrt(2 / Y) * g / (k * T.ETA0)          # same for + and - modes
    return {+1: (np.sqrt(2 * Y) * ht, Hz), -1: (-np.sqrt(2 * Y) * ht, Hz)}


def quad_lambda(p, q, a, b, f, Z, weight=None, nq=96):
    """Four blocks from the reciprocity integrals; weight(u, v) multiplies the integrand (h = 1 + kappa u)."""
    Zw = T.wall_impedances(Z)
    x, w = np.polynomial.legendre.leggauss(nq)
    out = {(1, 1): 0j, (1, -1): 0j, (-1, 1): 0j, (-1, -1): 0j}   # (sign of h_p test, sign of h_q)
    for wall in T.WALLS:
        L = b if wall[0] == "u" else a
        s = 0.5 * L * x
        ws = 0.5 * L * w
        if weight is None:
            wt = 1.0
        else:
            uu = np.full_like(s, -a / 2 if wall == "u-" else a / 2) if wall[0] == "u" else s
            wt = weight(uu)
        Fp, Fq = fields_on_wall(p, a, b, f, wall, s), fields_on_wall(q, a, b, f, wall, s)
        for sp_ in (1, -1):
            for sq in (1, -1):
                dot = Fq[sq][0] * Fp[sp_][0] + Fq[sq][1] * Fp[sp_][1]
                out[(sp_, sq)] += Zw[wall] * np.sum(ws * wt * dot)
    # da+_p/dz: test with h_p^- ;  da-_p/dz: test with h_p^+
    Lpp = -0.25 * out[(-1, 1)]
    Lpm = -0.25 * out[(-1, -1)]
    Lmp = -0.25 * out[(1, 1)]
    Lmm = -0.25 * out[(1, -1)]
    return Lpp, Lpm, Lmp, Lmm


def quad_power_loss(p, a, b, f, Rs, nq=96):
    x, w = np.polynomial.legendre.leggauss(nq)
    tot = 0.0
    for wall in T.WALLS:
        L = b if wall[0] == "u" else a
        F = fields_on_wall(p, a, b, f, wall, 0.5 * L * x)[+1]
        tot += np.sum(0.5 * L * w * (abs(F[0]) ** 2 + abs(F[1]) ** 2))
    return 0.5 * Rs * tot


def modes_upto(M, Nn):
    out = []
    for m in range(M + 1):
        for n in range(Nn + 1):
            if m + n > 0:
                out.append(("TE", m, n))
            if m > 0 and n > 0:
                out.append(("TM", m, n))
    return out


def check_quadrature():
    print("1. Closed forms vs reciprocity quadrature (each wall its own complex Z)")
    worst = 0.0
    worst1 = 0.0
    for (a, b, f) in ((22.86e-3, 10.16e-3, 60e9), (15e-3, 15e-3, 45e9), (40e-3, 12e-3, 33e9)):
        Z = {w: complex(RNG.uniform(0.05, 2), RNG.uniform(-1, 2)) for w in T.WALLS}
        ms = modes_upto(4, 3)
        kappa = 3.7
        for p, q in itertools.product(ms, ms):
            Q = quad_lambda(p, q, a, b, f, Z)
            C = (T.lambda_pp(p, q, a, b, f, Z), T.lambda_pm(p, q, a, b, f, Z),
                 T.lambda_pm(p, q, a, b, f, Z), T.lambda_pp(p, q, a, b, f, Z))
            scale = max(abs(T.lambda_pp(p, p, a, b, f, Z)), abs(T.lambda_pp(q, q, a, b, f, Z)))
            worst = max(worst, max(abs(x - y) for x, y in zip(Q, C)) / scale)
            # bend: weight h = 1 + kappa u; closed form Lambda + kappa Lambda1
            Qh = quad_lambda(p, q, a, b, f, Z, weight=lambda u: 1 + kappa * u)
            Ch = (T.lambda_pp(p, q, a, b, f, Z) + kappa * T.lambda1_pp(p, q, a, b, f, Z),
                  T.lambda_pm(p, q, a, b, f, Z) + kappa * T.lambda1_pm(p, q, a, b, f, Z))
            worst1 = max(worst1, max(abs(Qh[0] - Ch[0]), abs(Qh[1] - Ch[1])) / scale)
    print(f"   max relative deviation, straight (all 4 blocks, {len(ms)}^2 pairs x 3 guides): {worst:.2e}")
    print(f"   max relative deviation, bend weight 1 + kappa u:                          {worst1:.2e}")
    assert worst < 1e-8 and worst1 < 1e-8
    # power loss
    worstP = 0.0
    a, b = 22.86e-3, 10.16e-3
    for f in (18e9, 45e9, 90e9):
        Rs = T.surface_resistance(f, 3.5e7)
        for p in modes_upto(8, 4):
            if T.cutoff(p, a, b) >= T.wavenumber(f):
                continue
            Pl = quad_power_loss(p, a, b, f, Rs)
            lam = T.lambda_pp(p, p, a, b, f, Rs * (1 + 1j))
            worstP = max(worstP, abs(Pl - 2 * lam.real) / Pl)
            # the reactance X_s = R_s gives Im Lambda = Re Lambda
            assert abs(lam.imag - lam.real) < 1e-12 * abs(lam)
    print(f"   power-loss integral (1/2)Rs oint|H_tan|^2 vs 2 Re Lambda_pp, WR-90, 18/45/90 GHz: {worstP:.2e}")
    assert worstP < 1e-10
    return worst, worst1, worstP


def check_sympy():
    print("2. Sympy: Lambda_pp (equal walls) == Marcuvitz attenuation, identically")
    a, b, k, R = sp.symbols("a b k R", positive=True)
    m, n = sp.symbols("m n", positive=True, integer=True)
    eta = sp.symbols("eta", positive=True)

    def lam_diag(kind, mm, nn, em, en):
        chi2 = sp.pi ** 2 * (mm ** 2 / a ** 2 + nn ** 2 / b ** 2)
        beta = sp.sqrt(k ** 2 - chi2)
        s = sp.pi * sp.sqrt(mm ** 2 * b / a + nn ** 2 * a / b)
        if kind == "TM":
            N = 2 / s
            cu, cv = N * mm * sp.pi / a, N * nn * sp.pi / b
            Y = k / (eta * beta)
            A = (b / 2) * cu ** 2 * 2 * R + (a / 2) * cv ** 2 * 2 * R
            B = 0
        else:
            N = sp.sqrt(em * en) / s
            cu, cv = -N * nn * sp.pi / b, N * mm * sp.pi / a
            Y = beta / (k * eta)
            A = ((b / 2) * cu ** 2 * 2 * R if nn != 0 else 0) + ((a / 2) * cv ** 2 * 2 * R if mm != 0 else 0)
            B = chi2 ** 2 * N ** 2 * (b / en * 2 * R + a / em * 2 * R)
        return sp.Rational(1, 2) * (Y * A + B / ((k * eta) ** 2 * Y)), beta, chi2

    def marc_te(mm, nn, em, en, chi2):
        x2 = chi2 / k ** 2
        s = sp.sqrt(1 - x2)
        return R / (eta * b) * ((en * mm ** 2 * b / a + em * nn ** 2) / (mm ** 2 * b / a + nn ** 2 * a / b) * s
                                + (en + em * b / a) * x2 / s)

    def marc_tm(mm, nn, chi2):
        x2 = chi2 / k ** 2
        return 2 * R / (eta * a) * (mm ** 2 + nn ** 2 * a ** 3 / b ** 3) / (mm ** 2 + nn ** 2 * a ** 2 / b ** 2) \
            / sp.sqrt(1 - x2)

    results = []
    for label, kind, mm, nn, em, en in (("TE_mn", "TE", m, n, 2, 2), ("TE_m0", "TE", m, 0, 2, 1),
                                         ("TE_0n", "TE", 0, n, 1, 2), ("TM_mn", "TM", m, n, 2, 2)):
        lam, beta, chi2 = lam_diag(kind, mm, nn, em, en)
        ref = marc_te(mm, nn, em, en, chi2) if kind == "TE" else marc_tm(mm, nn, chi2)
        # substitute k^2 = chi2 + beta^2 to make the square roots rational in beta
        bt = sp.symbols("beta_s", positive=True)
        kk = sp.sqrt(chi2 + bt ** 2)
        diff = sp.simplify(sp.together((lam - ref).subs(k, kk)))
        results.append((label, diff))
        print(f"   {label}: Lambda_pp - alpha_Marcuvitz = {diff}")
        assert diff == 0
    # TE10 in the form of the card
    lam, beta, chi2 = lam_diag("TE", 1, 0, 2, 1)
    fc2 = chi2 / k ** 2
    card = R / (eta * b * sp.sqrt(1 - fc2)) * (1 + 2 * b / a * fc2)
    d = sp.simplify(lam - card)
    print(f"   TE10: Lambda_pp - R_s/(eta b sqrt(1-(fc/f)^2)) [1 + (2b/a)(fc/f)^2] = {d}")
    assert d == 0
    return results


def check_numbers():
    print("3. Diagonal vs Marcuvitz, aluminium, 18-90 GHz (relative deviation)")
    worst = 0.0
    for (a, b, lbl) in ((22.86e-3, 10.16e-3, "WR-90"), (20e-3, 20e-3, "square 20 mm")):
        for f in np.linspace(18e9, 90e9, 37):
            Rs = T.surface_resistance(f, 3.5e7)
            for p in modes_upto(10, 6):
                if T.cutoff(p, a, b) >= T.wavenumber(f) * (1 - 1e-6):
                    continue
                lam = T.lambda_pp(p, p, a, b, f, (1 + 1j) * Rs).real
                ref = (T.alpha_te_marcuvitz if p[0] == "TE" else T.alpha_tm_marcuvitz)(p[1], p[2], a, b, f, Rs)
                worst = max(worst, abs(lam - ref) / ref)
        print(f"   {lbl}: running max {worst:.2e}")
    assert worst < 1e-6
    a, b = 22.86e-3, 10.16e-3
    for f in (18e9, 90e9):
        Rs = T.surface_resistance(f, 3.5e7)
        al = T.lambda_pp(("TE", 1, 0), ("TE", 1, 0), a, b, f, (1 + 1j) * Rs).real
        print(f"   WR-90 TE10, Al, {f/1e9:.0f} GHz: alpha = {al:.6e} Np/m = {20*np.log10(np.e)*al:.4f} dB/m")
    return worst


def check_gsm():
    print("4. GSM of a lossy straight (WR-90, aluminium, 1 m, all propagating modes of all families, 90 GHz)")
    from scipy.linalg import expm
    a, b, f, Lg = 22.86e-3, 10.16e-3, 90e9, 1.0
    k = T.wavenumber(f)
    ms = [p for p in modes_upto(12, 6) if T.cutoff(p, a, b) < k]
    N = len(ms)
    beta = np.array([T.beta(p, a, b, f) for p in ms])
    D = np.block([[np.diag(-1j * beta), np.zeros((N, N))], [np.zeros((N, N)), np.diag(1j * beta)]])
    Phi = expm((D - T.lambda_full(ms, a, b, f, T.surface_impedance(f, 3.5e7))) * Lg)
    P11, P12, P21, P22 = Phi[:N, :N], Phi[:N, N:], Phi[N:, :N], Phi[N:, N:]
    iP22 = np.linalg.inv(P22)
    S = np.block([[-iP22 @ P21, iP22], [P11 - P12 @ iP22 @ P21, P12 @ iP22]])
    rec = np.max(abs(S - S.T)) / np.max(abs(S))
    sv = np.linalg.svd(S, compute_uv=False)[0]
    print(f"   N = {N}: reciprocity max|S - S^T|/max|S| = {rec:.1e}; largest singular value = {sv:.8f}")
    assert rec < 1e-10 and sv <= 1
    return rec, sv


if __name__ == "__main__":
    check_quadrature()
    check_sympy()
    check_numbers()
    check_gsm()
    print("all checks passed")
