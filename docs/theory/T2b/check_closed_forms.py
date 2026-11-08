"""Checks of the closed forms of note T2b (boxes B3, B4) and of the structural properties.

1. T_pq: closed form (B3) vs the coordinate-transformation integrals (route of the note, quadrature on the unit
   square) vs the cross-section route: complex-step derivative of the overlap int_{R(z)} e_q(z) . e_p(z') at z' = z,
   integration domain R(z) FIXED (the domain of the expansion), plus a Richardson central difference.
   Also shows that the moving domain R(z') gives -T^T instead (a first-order difference).
2. Exact symbolic check (sympy) of B3 for small indices with symbolic a, b, a', b', u0', v0'.
3. Power-wave form (B4) vs a direct numerical change of variables of the V-I system (including dY/dz).
4. Losslessness / reciprocity structure: T^I = -T^T; A antisymmetric, B symmetric (real above cutoff).
5. TE10 self-coupling vs Katsenelenbaum et al. (1998) eqs. 4.47 / 4.51-4.52: B_11 = -(1/2) d ln(b / beta)/dz.
6. Uniform-guide limit: a' = b' = u0' = v0' = 0 gives T = 0, i.e. the T0 equations.
Run: PYTHONPATH=src python3 docs/theory/T2b/check_closed_forms.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from wrmodes.modes import Mode, ETA0
from t2b_formulas import (T_closed, T_matrix, T_transformed, T_overlap, power_wave_matrices, beta_Y, ZY,
                          system_matrix, dlogY)


def all_modes(M, Nn):
    return [Mode(k, m, n) for m in range(0, M + 1) for n in range(0, Nn + 1) for k in ("TE", "TM")
            if (k == "TE" and m + n > 0) or (k == "TM" and m * n > 0)]


def check_T():
    print("1. T_pq: closed form vs transformation route vs overlap derivative (fixed domain R(z))")
    worst = {"transformed": 0, "overlap (complex step)": 0, "overlap (Richardson)": 0}
    cases = [  # (a, b, a', b', u0', v0', z0, curvature of the geometry functions)
        (0.02286, 0.01016, 0.13, -0.07, 0.05, 0.031),
        (0.030, 0.030, 0.05, 0.11, -0.02, 0.0),      # square guide: degenerate TE_mn / TE_nm pairs
        (0.050, 0.008, -0.2, 0.0, 0.0, -0.04),
    ]
    modes = all_modes(4, 3)
    for (a, b, da, db, du0, dv0) in cases:
        z0 = 0.3

        def geom(z):
            d = z - z0
            return (a + da * d + 0.4 * d * d, b + db * d - 0.3 * d * d, 0.001 + du0 * d + 0.2 * d * d,
                    -0.002 + dv0 * d - 0.1 * d * d)
        T = T_matrix(modes, a, b, da, db, du0, dv0)
        scale = np.max(np.abs(T))
        for i, p in enumerate(modes):
            for j, q in enumerate(modes):
                worst["transformed"] = max(worst["transformed"], abs(T[i, j] - T_transformed(p, q, a, b, da, db, du0, dv0)) / scale)
                worst["overlap (complex step)"] = max(worst["overlap (complex step)"], abs(T[i, j] - T_overlap(p, q, geom, z0)) / scale)
                if (i + j) % 7 == 0:
                    worst["overlap (Richardson)"] = max(worst["overlap (Richardson)"],
                                                        abs(T[i, j] - T_overlap(p, q, geom, z0, method="fd")) / scale)
        # moving domain R(z')
        Tm = np.array([[T_overlap(p, q, geom, z0, moving=True) for q in modes[:12]] for p in modes[:12]])
        dev_moving = np.max(np.abs(Tm + T[:12, :12].T)) / scale
        dev_fixed = np.max(np.abs(Tm - T[:12, :12])) / scale
    for k, v in worst.items():
        print(f"   max |T_closed - T_{k}| / max|T| = {v:.1e}   ({len(modes)} modes, 3 geometries)")
    print(f"   moving domain R(z'): |dO/dz' + T^T| / max|T| = {dev_moving:.1e}; |dO/dz' - T| / max|T| = {dev_fixed:.2f}")
    return max(worst.values())


def check_sympy():
    import sympy as sp
    print("2. Exact symbolic check of B3 (sympy, symbolic a, b, a', b', u0', v0')")
    a, b, z = sp.symbols("a b z", positive=True)
    A_, B_, U_, V_ = [sp.Function(n)(z) for n in ("A", "B", "U0", "V0")]
    u, v = sp.symbols("u v", real=True)
    da, db, du0, dv0 = sp.symbols("da db du0 dv0", real=True)

    def field(kind, m, n, A, B, U0, V0):
        up, vp = m * sp.pi * (u - U0 + A / 2) / A, n * sp.pi * (v - V0 + B / 2) / B
        epsm = 1 if m == 0 else 2
        epsn = 1 if n == 0 else 2
        s = sp.pi * sp.sqrt(m * m * B / A + n * n * A / B)
        if kind == "TM":
            c = 2 / s
            P = c * sp.sin(up) * sp.sin(vp)
            return sp.diff(P, u), sp.diff(P, v)
        c = sp.sqrt(epsm * epsn) / s
        G = c * sp.cos(up) * sp.cos(vp)
        return sp.diff(G, v), -sp.diff(G, u)
    pairs = [(("TE", 1, 0), ("TE", 3, 0)), (("TE", 1, 0), ("TE", 2, 0)), (("TE", 1, 1), ("TM", 1, 1)),
             (("TM", 1, 1), ("TE", 1, 1)), (("TM", 2, 1), ("TE", 1, 1)), (("TM", 1, 2), ("TM", 1, 1)),
             (("TE", 1, 2), ("TE", 1, 0)), (("TE", 0, 1), ("TE", 1, 1)), (("TE", 1, 1), ("TE", 1, 1)),
             (("TM", 1, 1), ("TM", 1, 1)), (("TE", 1, 1), ("TM", 2, 1)), (("TM", 2, 2), ("TE", 2, 1))]
    worst = 0
    for pp, qq in pairs:
        ep = field(*pp, A_, B_, U_, V_)
        eq = field(*qq, A_, B_, U_, V_)
        dep = [sp.diff(c, z) for c in ep]
        integrand = eq[0] * dep[0] + eq[1] * dep[1]
        subs = {sp.Derivative(A_, z): da, sp.Derivative(B_, z): db, sp.Derivative(U_, z): du0,
                sp.Derivative(V_, z): dv0}
        integrand = integrand.subs(subs).subs({A_: a, B_: b, U_: 0, V_: 0})
        val = sp.integrate(sp.integrate(sp.expand(sp.expand_trig(integrand)), (u, -a / 2, a / 2)), (v, -b / 2, b / 2))
        val = sp.simplify(val)
        closed = T_closed(Mode(*pp), Mode(*qq), a, b, da, db, du0, dv0)
        diff = sp.simplify(val - closed)
        rng = np.random.default_rng(1)
        rel = 0.0
        for _ in range(3):
            pt = dict(zip((a, b, da, db, du0, dv0), rng.uniform(0.2, 1.5, 6) * np.array([0.03, 0.02, 0.1, 0.1, 0.1, 0.1])))
            dv_ = abs(complex(sp.N((val - closed).subs(pt), 30)))
            rel = max(rel, dv_ / max(abs(complex(sp.N(val.subs(pt), 30))), 1e-300) if dv_ else 0.0)
        worst = max(worst, rel)
        print(f"   T[{Mode(*pp).name}, {Mode(*qq).name}] = {sp.factor(val)}    (B3 - sympy: simplified {diff}; "
              f"max rel. at 3 random points {rel:.0e})")
    return worst


def check_power_waves():
    print("3. Power-wave form B4 vs numerical change of variables of the V-I system")
    f = 20e9
    modes = all_modes(4, 2)
    z0 = 0.1

    def g(z):
        d = z - z0
        return dict(a=0.04 + 0.08 * d, b=0.02 + 0.05 * d, u0=0.0, da=0.08, db=0.05, du0=0.03, dv0=-0.02)
    gg = g(z0)
    # keep only modes not too close to cutoff for this check
    beta, Y = beta_Y(modes, f, gg["a"], gg["b"])
    modes = [p for p, bt in zip(modes, beta) if abs(bt) > 50]
    N = len(modes)

    def R(z):
        G = g(z)
        _, Yz = beta_Y(modes, f, G["a"], G["b"])
        s = np.sqrt(Yz)
        # a+ = (sV/sqrt2 + I/(sqrt2 s))/2,  a- = (sV/sqrt2 - I/(sqrt2 s))/2
        return 0.5 * np.block([[np.diag(s / np.sqrt(2)), np.diag(1 / (np.sqrt(2) * s))],
                               [np.diag(s / np.sqrt(2)), -np.diag(1 / (np.sqrt(2) * s))]])
    h = 1e-6
    dR = (R(z0 + h) - R(z0 - h)) / (2 * h)
    Mvi = system_matrix(modes, f, gg["a"], gg["b"], gg["da"], gg["db"], 0.0, gg["du0"], gg["dv0"])
    K = R(z0) @ Mvi @ np.linalg.inv(R(z0)) + dR @ np.linalg.inv(R(z0))
    beta, _ = beta_Y(modes, f, gg["a"], gg["b"])
    A, B = power_wave_matrices(modes, f, gg["a"], gg["b"], gg["da"], gg["db"], gg["du0"], gg["dv0"])
    Kf = np.block([[-1j * np.diag(beta) + A, B], [B, 1j * np.diag(beta) + A]])
    err = np.max(np.abs(K - Kf)) / np.max(np.abs(A) + np.abs(B))
    print(f"   max |K_numeric - K_B4| / max|A, B| = {err:.1e}  ({N} modes, propagating and evanescent)")
    prop = np.real(beta) > 0
    Ap, Bp = A[np.ix_(prop, prop)], B[np.ix_(prop, prop)]
    print(f"4. structure (propagating block): |A + A^T| = {np.max(np.abs(Ap + Ap.T)):.1e}, |B - B^T| = "
          f"{np.max(np.abs(Bp - Bp.T)):.1e}, |Im A| = {np.max(np.abs(Ap.imag)):.1e}, |Im B| = {np.max(np.abs(Bp.imag)):.1e}")
    T = T_matrix(modes, gg["a"], gg["b"], gg["da"], gg["db"], gg["du0"], gg["dv0"])
    Mblk = Mvi
    print(f"   V-I system: T^I = -T^T exactly by construction; dI-block + T^T = {np.max(np.abs(Mblk[N:, N:] + T.T)):.1e}; "
          f"Z, Y symmetric: {np.max(np.abs(Mblk[:N, N:] - Mblk[:N, N:].T)):.1e}, {np.max(np.abs(Mblk[N:, :N] - Mblk[N:, :N].T)):.1e}")
    return err


def check_katsenelenbaum():
    print("5. TE10 reflection coupling vs Katsenelenbaum et al. 1998, eq. 4.47 (W = C b / beta, eqs. 4.51-4.52)")
    f = 12e9
    worst = 0
    for (a, b, da, db, du0, dv0) in [(0.02286, 0.01016, 0.1, 0.0, 0.0, 0.0), (0.02286, 0.01016, 0.0, 0.07, 0.0, 0.0),
                                     (0.03, 0.012, 0.06, -0.04, 0.02, 0.05)]:
        p = Mode("TE", 1, 0)
        _, B = power_wave_matrices([p], f, a, b, da, db, du0, dv0)
        k = 2 * np.pi * f / 2.99792458e8
        beta2 = k * k - (np.pi / a) ** 2
        dlnb = db / b
        dlnbeta = (np.pi ** 2 / a ** 3) * da / beta2
        ref = -0.5 * (dlnb - dlnbeta)              # = -(1/2) d ln W / dz,  W = b / beta
        worst = max(worst, abs(B[0, 0] - ref) / abs(ref))
        print(f"   a'={da}, b'={db}, u0'={du0}, v0'={dv0}:  B_11 = {B[0,0].real:+.6e}, -(1/2) d ln(b/beta)/dz = {ref:+.6e}")
    print("   first-order reflection a-(0) = -int B_11 exp(-2i int beta) dz = (1/2) int d ln W exp(...) dz: the voltage\n"
          "   reflection coefficient Delta W / 2W of Katsenelenbaum eq. 4.50 for a small step; offsets do not contribute.")
    return worst


def check_uniform():
    modes = all_modes(4, 3)
    T = T_matrix(modes, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0)
    print(f"6. uniform guide: max|T| = {np.max(np.abs(T)):.1e} (the V-I system is then exactly the T0 one)")
    return np.max(np.abs(T))


if __name__ == "__main__":
    e1 = check_T()
    e2 = check_sympy()
    e3 = check_power_waves()
    e5 = check_katsenelenbaum()
    e6 = check_uniform()
    print(f"\nSUMMARY: T closed form {e1:.1e}; sympy {e2:.1e}; power waves {e3:.1e}; Katsenelenbaum 4.47 {e5:.1e}; uniform {e6:.1e}")
