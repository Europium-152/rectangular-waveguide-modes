"""T2 check 2: the full V-I equations, and the tilted-wall term, against exact Maxwell fields.

Run:  PYTHONPATH=src python3 docs/theory/T2/check_exact_fields.py

For an exact field (exact_fields.py) the projections V_p(z) = int_{S(z)} E_t . e_p, I_p(z) = int_{S(z)} H_t . (z x e_p)
are computed by Gauss-Legendre quadrature on the cross-section at each z, and dV/dz, dI/dz by 4th-order central
differences. The T2 equations

    dV/dz = -i Z I + N^T V,      dI/dz = -i Y V - N I

are infinite systems; with the sums on the right truncated to the modes of a set T, the residual for a fixed set of
low-order test modes must decrease as T grows. The same is done for the variants that a derivation would produce if
it dropped the tilted-wall term (dV/dz = ... + (N^T + W) V) or the moving-domain term (dI/dz = ... - (N + W) I... see
the note), which must not converge.

Tilted-wall term, separately: on the moving walls h E_z + w E_n = 0 pointwise, and the three parts of
dV_p/dz = int dE_t/dz . e_p + int E_t . de_p/dz + oint w E_t . e_p (Leibniz) are evaluated by quadrature; the
tilted-wall boundary term B_p = oint h E_z e_p.n of the integration by parts equals -oint w E_t . e_p exactly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import t2_coefficients as t2                                        # noqa: E402
import exact_fields as ef                                           # noqa: E402
from wrmodes.modes import Mode, transverse_field, propagation_constant   # noqa: E402

NQ = 80
XG, WG = np.polynomial.legendre.leggauss(NQ)


def grid(g):
    u, wu = g["u0"] + XG * g["a"] / 2, WG * g["a"] / 2
    v, wv = g["v0"] + XG * g["b"] / 2, WG * g["b"] / 2
    U, V = np.meshgrid(u, v, indexing="ij")
    return U, V, np.outer(wu, wv)


def efield(mode, g, U, V):
    return np.array(transverse_field(mode, g["a"], g["b"], U - g["u0"], V - g["v0"]))


def project(case, z, modes):
    g = case.geometry(z)
    U, V, W = grid(g)
    F = case.fields(U, V, np.full_like(U, z))
    Vp, Ip = [], []
    for m in modes:
        eu, ev = efield(m, g, U, V)
        Vp.append(np.sum(W * (F[0] * eu + F[1] * ev)))
        Ip.append(np.sum(W * (-F[3] * ev + F[4] * eu)))
    return np.array(Vp), np.array(Ip)


def derivative(fun, z, h):
    vals = [fun(z + s * h) for s in (-2, -1, 1, 2)]
    return [(v0 - 8 * v1 + 8 * v2 - v3) / (12 * h) for v0, v1, v2, v3 in zip(*vals)]


def blocks(modes, g, f):
    geo = {k: g[k] for k in ("da", "db", "du0", "dv0")}
    N = np.array([[t2.overlap_derivative(p, q, g["a"], g["b"], **geo) for q in modes] for p in modes])
    Z = np.array([[t2.impedance(p, q, g["a"], g["b"], f, g["kappa"], g["u0"]) for q in modes] for p in modes])
    Y = np.array([[t2.admittance(p, q, g["a"], g["b"], f, g["kappa"], g["u0"]) for q in modes] for p in modes])
    Wl = np.array([[t2.wall_term(p, q, g["a"], g["b"], **geo) for q in modes] for p in modes])
    return N, Z, Y, Wl


def residuals(case, f, z, test, truncations, h):
    """Residuals of the V and I equations for the test modes, for each truncation set.

    Each mode's residual is divided by the largest of |dX_p/dz| and the magnitudes of the terms of its right-hand
    side (computed with the largest truncation), so that a mode with a small amplitude is not hidden by a large one.
    """
    out = []
    big = truncations[-1]
    Vb, Ib = project(case, z, big)
    dV, dI = derivative(lambda zz: project(case, zz, big), z, h)
    g = case.geometry(z)
    ri = [big.index(m) for m in test]
    N, Z, Y, Wl = blocks(big, g, f)
    sV = np.maximum.reduce([np.abs(dV), np.abs(Z @ Ib), np.abs(N.T @ Vb)])[ri]
    sI = np.maximum.reduce([np.abs(dI), np.abs(Y @ Vb), np.abs(N @ Ib)])[ri]
    # modes that the field does not contain (symmetry) would divide noise by noise: floor at 1e-6 of the largest
    sV, sI = np.maximum(sV, 1e-6 * sV.max()), np.maximum(sI, 1e-6 * sI.max())
    for T in truncations:
        idx = [big.index(m) for m in T]
        N, Z, Y, Wl = blocks(T, g, f)
        V, I = Vb[idx], Ib[idx]
        rhsV = -1j * Z @ I + N.T @ V
        rhsI = -1j * Y @ V - N @ I
        bad_V = rhsV + Wl @ V              # tilted-wall term dropped (moving-domain term kept)
        bad_I = rhsI + Wl @ I              # moving-domain term dropped in the I equation
        ti = [T.index(m) for m in test]
        out.append((len(T),
                    np.max(np.abs(dV[ri] - rhsV[ti]) / sV), np.max(np.abs(dI[ri] - rhsI[ti]) / sI),
                    np.max(np.abs(dV[ri] - bad_V[ti]) / sV), np.max(np.abs(dI[ri] - bad_I[ti]) / sI)))
    return out


def modeset(ms, ns, kinds=("TE", "TM")):
    out = []
    for m in ms:
        for n in ns:
            for k in kinds:
                try:
                    out.append(Mode(k, m, n))
                except ValueError:
                    pass
    return out


def run_case(name, case, f, z, test, truncations, h, tol):
    print(f"{name}  (z = {z} m; geometry {fmt_geo(case.geometry(z))})")
    print("   modes   res V-eq    res I-eq    | V-eq w/o tilted-wall term   I-eq w/o domain term")
    rows = residuals(case, f, z, test, truncations, h)
    for n, rv, ri, bv, bi in rows:
        print(f"   {n:5d}   {rv:.3e}   {ri:.3e}   |   {bv:.3e}                  {bi:.3e}")
    rate_V = np.log(rows[-2][1] / rows[-1][1]) / np.log(rows[-1][0] / rows[-2][0])
    rate_I = np.log(rows[-2][2] / rows[-1][2]) / np.log(rows[-1][0] / rows[-2][0])
    ok = (rows[-1][1] < tol[0] and rate_V > 1.5 and rate_I > 0.7 and rows[-1][2] < rows[-2][2] < rows[-3][2]
          and min(r[3] for r in rows) > 10 * rows[-1][1] and min(r[4] for r in rows) > 3 * rows[-1][2])
    print(f"   observed convergence rates (last doubling): V-eq ~ T^-{rate_V:.2f}, I-eq ~ T^-{rate_I:.2f}")
    print(f"   {'PASS' if ok else 'FAIL'}  V-eq residual < {tol[0]:g}; V-eq and I-eq residuals converge (rates above); the variants "
          f"without the boundary terms do not converge")
    return ok


def fmt_geo(g):
    return ", ".join(f"{k}={g[k]:.5g}" for k in ("a", "b", "u0", "v0", "kappa", "da", "db", "du0", "dv0"))


# ----------------------------------------------------------------------------------------- tilted wall
def tilted_wall_check(case, f, z, modes, truncations, h):
    """hE_z + w E_n = 0 on the moving walls; Leibniz decomposition of dV/dz; B_p = -oint w E_t.e_p."""
    g = case.geometry(z)
    ok = True
    # pointwise wall condition
    worst, scale = 0.0, 0.0
    s = XG
    for axis, pos, sign, w in case.walls(z):
        if axis == 1:
            Uw, Vw = g["u0"] + s * g["a"] / 2, np.full_like(s, pos)
        else:
            Uw, Vw = np.full_like(s, pos), g["v0"] + s * g["b"] / 2
        F = case.fields(Uw, Vw, np.full_like(Uw, z))
        hh = 1 + g["kappa"] * Uw
        En = sign * F[axis]
        worst = max(worst, np.max(np.abs(hh * F[2] + w * En)))
        scale = max(scale, np.max(np.abs(F[axis])), np.max(np.abs(F[2])))
    print(f"   wall condition max|h E_z + w E_n| / max|E| = {worst / scale:.2e}")
    ok &= worst / scale < 1e-12

    # Leibniz decomposition and tilted-wall boundary term, mode by mode
    U, V, W = grid(g)
    Zc = np.full_like(U, z)
    dF = derivative(lambda zz: case.fields(U, V, np.full_like(U, zz)), z, h)
    F0 = case.fields(U, V, Zc)

    def geo_at(zz):
        return case.geometry(zz)

    dV_fd = derivative(lambda zz: project(case, zz, modes), z, h)[0]
    rows = []
    for i, m in enumerate(modes):
        e0 = efield(m, g, U, V)
        de = derivative(lambda zz: list(efield(m, geo_at(zz), U, V)), z, h)
        P1 = np.sum(W * (dF[0] * e0[0] + dF[1] * e0[1]))
        P2 = np.sum(W * (F0[0] * de[0] + F0[1] * de[1]))
        P3, B = 0.0, 0.0
        for axis, pos, sign, w in case.walls(z):
            if axis == 1:
                Uw, Vw, wq = g["u0"] + s * g["a"] / 2, np.full_like(s, pos), WG * g["a"] / 2
            else:
                Uw, Vw, wq = np.full_like(s, pos), g["v0"] + s * g["b"] / 2, WG * g["b"] / 2
            Fw = case.fields(Uw, Vw, np.full_like(Uw, z))
            ew = efield(m, g, Uw, Vw)
            P3 += np.sum(wq * w * (Fw[0] * ew[0] + Fw[1] * ew[1]))
            B += np.sum(wq * (1 + g["kappa"] * Uw) * Fw[2] * sign * ew[axis])
        rows.append((m.name, dV_fd[i], P1 + P2 + P3, P3, B))
    sc = max(abs(r[1]) for r in rows)
    e1 = max(abs(r[1] - r[2]) for r in rows) / sc
    e2 = max(abs(r[3] + r[4]) for r in rows) / max(abs(r[3]) for r in rows)
    print(f"   Leibniz: max|dV/dz - (P1 + P2 + P3)| / max|dV/dz| = {e1:.2e}")
    print(f"   tilted-wall term B = oint h E_z e_n equals -P3 = -oint w E_t.e_p: max|B + P3|/max|P3| = {e2:.2e}"
          f"  (max|P3|/max|dV/dz| = {max(abs(r[3]) for r in rows) / sc:.3f})")
    ok &= e1 < 1e-8 and e2 < 1e-10
    # modal form of the tilted-wall term: B_p = -sum_q W_pq V_q (series over the truncation)
    big = truncations[-1]
    Vb, _ = project(case, z, big)
    for T in truncations:
        idx = [big.index(q) for q in T]
        Wl = blocks(T, g, f)[3]
        series = -(Wl @ Vb[idx])
        err = max(abs(series[T.index(m)] - r[4]) for m, r in zip(modes, rows) if m in T)
        print(f"     {len(T):4d} modes: max|B_p + sum_q W_pq V_q| / max|B| = "
              f"{err / max(abs(r[4]) for r in rows):.3e}")
    print(f"   {'PASS' if ok else 'FAIL'}  tilted-wall term")
    return ok


if __name__ == "__main__":
    ok = True
    # --- E-plane wedge, TE10-like plus TM-to-u and odd-q terms; f = 12 GHz, a = 22.86 mm, b(z) ~ 10-25 mm
    f = 12e9
    cA = ef.EPlaneWedge(f, 0.02286, 0.05, 0.3,
                        [("TEu", 1, 0, "H2", 1.0), ("TEu", 1, 2, "H2", 0.4 - 0.2j), ("TEu", 1, 1, "H1", 0.3),
                         ("TMu", 1, 2, "H2", 0.5j), ("TMu", 1, 1, "J", 0.2)])
    mA = lambda N: modeset([1], range(0, N + 1))
    testA = mA(3)
    ok &= run_case("E-plane wedge (b' = 0.31, tilted walls with E_n != 0)", cA, f, 0.02, testA,
                   [mA(4), mA(8), mA(16), mA(32), mA(64)], 2e-5, (1e-3, None))
    ok &= tilted_wall_check(cA, f, 0.02, mA(4), [mA(4), mA(8), mA(16), mA(32), mA(64)], 2e-5)

    # --- H-plane wedge through cutoff: TE30-like standing wave whose cutoff plane is near z = 0.015
    f = 20e9
    k = 2 * np.pi * f / 299792458.0
    phi0, b = 0.4, 0.0102
    nu3 = 3 * np.pi / phi0
    za = (nu3 / k) * np.cos(phi0 / 2) - 0.015     # ~ cutoff of the q = 3 cylindrical mode at z = 0.015
    cB = ef.HPlaneWedge(f, b, za, phi0,
                        [("TMv", 0, 3, "J", 1.0), ("TMv", 0, 1, "H2", 0.3), ("TMv", 0, 2, "J", 0.2j),
                         ("TEv", 1, 1, "H2", 0.5), ("TMv", 1, 2, "J", 0.4 - 0.1j)])
    mB = lambda M: modeset(range(0, M + 1), [0, 1])
    testB = [m for m in mB(4)]
    for z in (0.010, 0.015, 0.020):
        g = cB.geometry(z)
        bet = propagation_constant(Mode("TE", 3, 0), g["a"], g["b"], f)
        print(f"   [H-plane wedge at z = {z}: TE3,0 beta/k = {bet / k:.4f}; power-wave Lambda_TE30 = "
              f"{t2.log_admittance_derivative(Mode('TE', 3, 0), g['a'], g['b'], f, g['da'], g['db']):.4g} 1/m]")
        ok &= run_case("H-plane wedge, TE3,0 crossing cutoff", cB, f, z, testB,
                       [mB(5), mB(10), mB(20), mB(40)], 2e-5, (1e-3, None))
    # --- straight guide in a curved, offset frame: kappa, a', u0' together
    f = 15e9
    comps = [("TE1,0", 1, 1.0), ("TE2,0", -1, 0.5j), ("TE3,0", 1, 0.3), ("TE4,0", 1, 0.2),     # TE4,0 evanescent
             ("TE1,1", 1, 0.4), ("TM1,1", -1, 0.3 + 0.2j), ("TE0,1", 1, 0.2)]
    cC = ef.CurvedFrame(f, 0.02286, 0.01016, 0.12, 0.003, comps)
    mC = lambda M: modeset(range(0, M + 1), [0, 1])
    ok &= run_case("Straight guide seen in a curved offset frame (R = 0.12 m)", cC, f, 0.03,
                   [m for m in mC(4)], [mC(5), mC(10), mC(20), mC(40)], 2e-5, (1e-3, None))
    ok &= tilted_wall_check(cC, f, 0.03, mC(4), [mC(5), mC(10), mC(20), mC(40)], 2e-5)
    # --- guide tilted in the v-z plane: v0' (and u-z plane: u0')
    cD = ef.TiltedGuide(f, 0.02286, 0.01016, 0.0, 0.25, comps)
    mD = lambda N: modeset([0, 1, 2, 3, 4], range(0, N + 1))
    ok &= run_case("Guide tilted in the v-z plane (v0' = tan 0.25)", cD, f, 0.01,
                   [m for m in mD(2) if m.m <= 3], [mD(3), mD(6), mD(12), mD(24)], 2e-5, (1e-3, None))
    cE = ef.TiltedGuide(f, 0.02286, 0.01016, 0.25, 0.0, comps)
    ok &= run_case("Guide tilted in the u-z plane (u0' = tan 0.25)", cE, f, 0.01,
                   [m for m in mC(4)], [mC(5), mC(10), mC(20), mC(40)], 2e-5, (1e-3, None))
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
