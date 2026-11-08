"""T2 check 1: closed forms of the cross-section (overlap-derivative) term, the wall term, the curvature integrals,
and the uniform-guide limit (T0).

Run:  PYTHONPATH=src python3 docs/theory/T2/check_closed_forms.py

* N_pq = int_{S(z)} e_p(z) . d_z e_q(z) ds is compared with
  (a) Gauss-Legendre quadrature of e_p . (d/dz e_q), the z-derivative taken by a central difference at fixed (u, v);
  (b) a numerical derivative of the overlap O(z1, z2) = int_{S(z1)} e_p(z1) . e_q(z2) ds with respect to z2, with
      the integration domain S(z1) (the cross-section of the projecting mode p), stated explicitly;
  and the other domain choices are shown to give -N^T, which differs from N at first order in the slope.
* W_pq (wall term) against a contour quadrature, and the identity N + N^T + W = 0.
* U, U^d, U^g against quadrature.
* Uniform guide (a' = b' = u0' = v0' = 0): the power-wave matrix equals wrmodes 0.1 D + kappa K (T0) exactly.
"""
import itertools
import sys
import os

import numpy as np
import scipy.constants as cst
import sympy as sp

sys.path.insert(0, os.path.dirname(__file__))
import t2_coefficients as t2                                    # noqa: E402
from wrmodes.modes import Mode, ModeBasis, transverse_field, divergence   # noqa: E402
from wrmodes.transfer_matrix import coupled_mode_matrices       # noqa: E402

NQ = 72
XG, WG = np.polynomial.legendre.leggauss(NQ)


def rect_grid(a, b, u0, v0):
    u, wu = u0 + XG * a / 2, WG * a / 2
    v, wv = v0 + XG * b / 2, WG * b / 2
    U_, V_ = np.meshgrid(u, v, indexing="ij")
    return U_, V_, np.outer(wu, wv)


def field(mode, g, U_, V_):
    """Local mode field at (U, V) for the geometry tuple g = (a, b, u0, v0) (analytic continuation outside)."""
    a, b, u0, v0 = g
    return np.array(transverse_field(mode, a, b, U_ - u0, V_ - v0))


def all_modes(mmax, nmax):
    out = []
    for m in range(mmax + 1):
        for n in range(nmax + 1):
            for kind in ("TE", "TM"):
                try:
                    out.append(Mode(kind, m, n))
                except ValueError:
                    pass
    return out


def report(name, err, tol):
    status = "PASS" if err < tol else "FAIL"
    print(f"  {status}  {name}: {err:.3e}  (target {tol:g})")
    return err < tol


def check_1d_integrals():
    """Symbolic check of the 1D matrices for a sample of index pairs (sympy, exact rationals)."""
    a = sp.symbols("a", positive=True)
    u = sp.symbols("u", real=True)              # u - u0 (centred coordinate)
    ok = True
    for j, l in itertools.product(range(0, 5), repeat=2):
        epsj, epsl = (1 if j == 0 else 2), (1 if l == 0 else 2)
        # d/da at fixed u (centre fixed): xi = u + a/2 moves with a; differentiate, then set a = 1 (the
        # dimensionless matrices are scale-free) and integrate exactly over u in [-1/2, 1/2]
        xi = u + a / 2
        ps = lambda k: sp.sqrt(sp.Rational(2) / a) * sp.sin(k * sp.pi * xi / a)
        pc = lambda k, e: sp.sqrt(sp.Rational(e) / a) * sp.cos(k * sp.pi * xi / a)
        x = sp.symbols("x", real=True)
        at1 = lambda expr: sp.expand(expr.subs(a, 1).subs(u, x - sp.Rational(1, 2)).rewrite(sp.exp))
        I = lambda expr: sp.nsimplify(sp.simplify(sp.integrate(at1(expr), (x, 0, 1))))
        Sa = I(ps(j) * sp.diff(ps(l), a)) if j and l else 0
        Ca = I(pc(j, epsj) * sp.diff(pc(l, epsl), a))
        Su = I(ps(j) * (-sp.diff(ps(l), u))) if j and l else 0
        Cu = I(pc(j, epsj) * (-sp.diff(pc(l, epsl), u)))
        Gs = I(ps(j) * u * ps(l)) if j and l else 0
        Gc = I(pc(j, epsj) * u * pc(l, epsl))
        got = [t2.S_width(j, l), t2.C_width(j, l), t2.S_shift(j, l), t2.C_shift(j, l),
               t2.G_s(j, l, 1.0), t2.G_c(j, l, 1.0)]
        ref = [float(Sa), float(Ca), float(Su), float(Cu), float(Gs), float(Gc)]
        if not np.allclose(got, ref, rtol=1e-13, atol=1e-14):
            print("   1D mismatch", j, l, got, ref)
            ok = False
    print(f"  {'PASS' if ok else 'FAIL'}  1D matrices S_width, C_width, S_shift, C_shift, G_s, G_c vs sympy, "
          f"j, l = 0..4 (exact)")
    return ok


def geometry(z, base, slopes):
    return tuple(base[i] + slopes[i] * z for i in range(4))


def check_overlap_derivative(base, slopes, modes):
    a, b, u0, v0 = base
    da, db, du0, dv0 = slopes
    U_, V_, W_ = rect_grid(a, b, u0, v0)
    n = len(modes)
    Ncf = np.array([[t2.overlap_derivative(p, q, a, b, da, db, du0, dv0) for q in modes] for p in modes])
    scale = np.max(np.abs(Ncf))
    h = 1e-4 * min(a, b) / max(abs(s) for s in slopes)

    # (a) quadrature of e_p . d_z e_q, d_z by a 4th-order central difference at fixed (u, v)
    E0 = [field(m, base, U_, V_) for m in modes]
    def dfield(m):
        fs = [field(m, geometry(s * h, base, slopes), U_, V_) for s in (-2, -1, 1, 2)]
        return (fs[0] - 8 * fs[1] + 8 * fs[2] - fs[3]) / (12 * h)
    dE = [dfield(m) for m in modes]
    Nq = np.array([[np.sum(W_ * (E0[i] * dE[j]).sum(0)) for j in range(n)] for i in range(n)])

    # (b) numerical derivative of the overlap O(z1, z2) = int_{S(z1)} e_p(z1) . e_q(z2), d/dz2 at z2 = z1 = 0
    def overlap(z1, z2, domain_of):
        gz1, gz2 = geometry(z1, base, slopes), geometry(z2, base, slopes)
        gd = gz1 if domain_of == 1 else gz2
        Ug, Vg, Wg = rect_grid(gd[0], gd[1], gd[2], gd[3])
        A = [field(m, gz1, Ug, Vg) for m in modes]
        B = [field(m, gz2, Ug, Vg) for m in modes]
        return np.array([[np.sum(Wg * (A[i] * B[j]).sum(0)) for j in range(n)] for i in range(n)])

    def d_dz2(domain_of):
        Os = [overlap(0.0, s * h, domain_of) for s in (-2, -1, 1, 2)]
        return (Os[0] - 8 * Os[1] + 8 * Os[2] - Os[3]) / (12 * h)

    def d_dz1(domain_of):
        Os = [overlap(s * h, 0.0, domain_of) for s in (-2, -1, 1, 2)]
        return (Os[0] - 8 * Os[1] + 8 * Os[2] - Os[3]) / (12 * h)

    N_dom1 = d_dz2(1)            # domain S(z1): the cross-section of the projecting mode e_p  -> N
    N_dom2 = d_dz2(2)            # domain S(z2): the cross-section of the differentiated mode -> -N^T
    D1_dom1 = d_dz1(1)           # derivative with respect to the projecting mode, domain S(z1) -> -N

    # wall term by contour quadrature: walls u = u0 +- a/2 (w = +-du0 + da/2), v = v0 +- b/2
    Wcf = np.array([[t2.wall_term(p, q, a, b, da, db, du0, dv0) for q in modes] for p in modes])
    vline, wv = v0 + XG * b / 2, WG * b / 2
    uline, wu = u0 + XG * a / 2, WG * a / 2
    Wq = np.zeros((n, n))
    for side in (-1, 1):
        w_u = side * du0 + da / 2
        Fu = [field(m, base, np.full_like(vline, u0 + side * a / 2), vline)[0] for m in modes]
        w_v = side * dv0 + db / 2
        Fv = [field(m, base, uline, np.full_like(uline, v0 + side * b / 2))[1] for m in modes]
        Wq += w_u * np.array([[np.sum(wv * Fu[i] * Fu[j]) for j in range(n)] for i in range(n)])
        Wq += w_v * np.array([[np.sum(wu * Fv[i] * Fv[j]) for j in range(n)] for i in range(n)])

    ok = True
    print(f"  geometry a={a}, b={b}, u0={u0}, v0={v0}; a'={da}, b'={db}, u0'={du0}, v0'={dv0}; {n} modes;"
          f" max|N| = {scale:.4g} 1/m")
    ok &= report("closed-form N vs quadrature of e_p . d_z e_q (relative)", np.max(np.abs(Ncf - Nq)) / scale, 1e-8)
    ok &= report("closed-form N vs d/dz2 of O(z1,z2) on domain S(z1) (relative)",
                 np.max(np.abs(Ncf - N_dom1)) / scale, 1e-8)
    ok &= report("d/dz2 of overlap on domain S(z2) equals -N^T (relative)",
                 np.max(np.abs(-Ncf.T - N_dom2)) / scale, 1e-8)
    ok &= report("d/dz1 of overlap on domain S(z1) equals -N (relative)",
                 np.max(np.abs(-Ncf - D1_dom1)) / scale, 1e-8)
    ok &= report("closed-form W vs contour quadrature (relative)", np.max(np.abs(Wcf - Wq)) / scale, 1e-10)
    ok &= report("identity N + N^T + W = 0 (relative)", np.max(np.abs(Ncf + Ncf.T + Wcf)) / scale, 1e-12)
    diff = np.max(np.abs(N_dom2 - N_dom1)) / scale
    print(f"  INFO  domain choice matters: max|N(S(z2)) - N(S(z1))| / max|N| = {diff:.3f} "
          f"(= max|W|/max|N| = {np.max(np.abs(Wcf)) / scale:.3f})")
    return ok


def check_curvature_integrals(a, b, modes):
    U_, V_, W_ = rect_grid(a, b, 0.0, 0.0)
    n = len(modes)
    E = [np.array(transverse_field(m, a, b, U_, V_)) for m in modes]
    D = [np.array(divergence(m, a, b, U_, V_)) for m in modes]
    Uq = np.array([[np.sum(W_ * U_ * (E[i] * E[j]).sum(0)) for j in range(n)] for i in range(n)])
    Udq = np.array([[np.sum(W_ * U_ * D[i][0] * D[j][0]) for j in range(n)] for i in range(n)])
    Ugq = np.array([[np.sum(W_ * U_ * D[i][1] * D[j][1]) for j in range(n)] for i in range(n)])
    Ucf = np.array([[t2.U(p, q, a, b) for q in modes] for p in modes])
    Udcf = np.array([[t2.Ud(p, q, a, b) for q in modes] for p in modes])
    Ugcf = np.array([[t2.Ug(p, q, a, b) for q in modes] for p in modes])
    ok = report("U vs quadrature (relative)", np.max(np.abs(Ucf - Uq)) / np.max(np.abs(Uq)), 1e-10)
    ok &= report("U^d vs quadrature (relative)", np.max(np.abs(Udcf - Udq)) / np.max(np.abs(Udq)), 1e-10)
    ok &= report("U^g vs quadrature (relative)", np.max(np.abs(Ugcf - Ugq)) / np.max(np.abs(Ugq)), 1e-10)
    return ok


def check_uniform_limit():
    ok = True
    for (a, b, f, fam) in [(0.02286, 0.01016, 40e9, 0), (0.02286, 0.01016, 60e9, 1), (0.044, 0.03, 30e9, 2),
                           (0.01016, 0.02286, 50e9, 1)]:
        basis = ModeBasis.propagating(f, a, b, families=[fam])
        kappa = 3.7
        Kt2 = t2.power_wave_matrix(basis.modes, a, b, f, kappa=kappa)
        D, K, _ = coupled_mode_matrices(f, a, b, basis.legacy_modes())
        ref = D + kappa * K
        err = np.max(np.abs(Kt2 - ref)) / np.max(np.abs(ref))
        ok &= report(f"uniform limit = T0 (D + kappa K), a={a}, b={b}, f={f / 1e9} GHz, family {fam}, "
                     f"{len(basis)} modes", err, 1e-10)
    return ok


if __name__ == "__main__":
    ok = True
    print("1D integrals")
    ok &= check_1d_integrals()
    print("Overlap derivative, wall term (all four slopes at once)")
    modes = all_modes(4, 3)
    ok &= check_overlap_derivative((0.023, 0.0137, 0.0011, -0.0007), (0.21, -0.13, 0.17, 0.09), modes)
    ok &= check_overlap_derivative((0.015, 0.031, 0.0, 0.0), (-0.05, 0.11, 0.0, 0.0), modes)
    print("Curvature integrals")
    ok &= check_curvature_integrals(0.023, 0.0137, all_modes(5, 2))
    print("Uniform-guide limit")
    ok &= check_uniform_limit()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
