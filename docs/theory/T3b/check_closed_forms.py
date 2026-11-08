"""Check the closed forms of twist_coupling.py against quadrature and against exact symbolic integration.

Run:  cd <worktree> && PYTHONPATH=$PWD/src:$PWD/docs/theory/T3b python3 docs/theory/T3b/check_closed_forms.py

Checks (acceptance test "closed forms vs quadrature, 1e-10 relative"):
 1. K_pq from its definition (box 2), Gauss-Legendre quadrature of the wrmodes fields, vs closed form.
 2. G_pq = iint e_p . d_theta e_q, with d_theta e_q computed by complex-step differentiation of the actively
    rotated field R(theta) e_q(R(-theta) r) (no use of the generator formula), and the identity K = G^T.
 3. The generator formula G e = z x e - (u d_v - v d_u) e reproduces the complex-step derivative pointwise,
    and annihilates the rotation-invariant field e = (u, v).
 4. Selection rule: K_pq = 0 unless m + m' and n + n' are both odd.
 5. Exact sympy integrals for a few pairs (symbolic a, b) vs the closed form; c(TE10, TE01) = 8/pi^2.
"""
import itertools
import numpy as np
import sympy as sp

from wrmodes.modes import Mode, ModeBasis, transverse_field, divergence
from twist_coupling import K_element, K_explicit, K_matrix, K_matrix_fast, couples, norm

GUIDES = {"square": (1.0, 1.0), "WR187": (47.549e-3, 22.149e-3), "tall WR187": (22.149e-3, 47.549e-3),
          "WR90": (22.86e-3, 10.16e-3), "3:1": (3.0, 1.0)}


def grid(a, b, N=96):
    x, w = np.polynomial.legendre.leggauss(N)
    U, V = np.meshgrid(x * a / 2, x * b / 2, indexing="ij")
    return U, V, np.outer(w * a / 2, w * b / 2)


def rotated_field(mode, a, b, U, V, th):
    """Active rotation e_th(r) = R(th) e(R(-th) r); works for complex th (complex step)."""
    c, s = np.cos(th), np.sin(th)
    u0, v0 = c * U + s * V, -s * U + c * V                # R(-th) r
    eu, ev = transverse_field(mode, a, b, u0, v0)
    return c * eu - s * ev, s * eu + c * ev


def dtheta_field(mode, a, b, U, V, h=1e-30):
    eu, ev = rotated_field(mode, a, b, U, V, 1j * h)
    return eu.imag / h, ev.imag / h


def K_quad(p, q, a, b, U, V, W):
    pu, pv = transverse_field(p, a, b, U, V)
    qu, qv = transverse_field(q, a, b, U, V)
    dp, _ = divergence(p, a, b, U, V)
    _, gq = divergence(q, a, b, U, V)
    return np.sum(W * (dp * (V * qu - U * qv) + (U * pu + V * pv) * gq))


def G_quad(p, q, a, b, U, V, W):
    pu, pv = transverse_field(p, a, b, U, V)
    du, dv = dtheta_field(q, a, b, U, V)
    return np.sum(W * (pu * du + pv * dv))


def modes_upto(M):
    out = []
    for m in range(M + 1):
        for n in range(M + 1):
            if m + n > 0:
                out.append(Mode("TE", m, n))
            if m > 0 and n > 0:
                out.append(Mode("TM", m, n))
    return out


def check_quadrature():
    worst_K, worst_G, worst_sel, n_pairs = 0.0, 0.0, 0.0, 0
    worst_X = [0.0]
    modes = modes_upto(4)
    for name, (a, b) in GUIDES.items():
        U, V, W = grid(a, b)
        scale = 0.0
        rows = []
        for p, q in itertools.product(modes, modes):
            kc = K_element(p, q, a, b)
            kq = K_quad(p, q, a, b, U, V, W)
            gq = G_quad(q, p, a, b, U, V, W)                 # G_qp, to compare with K_pq
            rows.append((p, q, kc, kq, gq))
            scale = max(scale, abs(kc))
        for p, q, kc, kq, gq in rows:
            n_pairs += 1
            worst_X[0] = max(worst_X[0], abs(K_explicit(p, q, a, b) - kc) / scale)
            if couples(p, q):
                worst_K = max(worst_K, abs(kc - kq) / scale)
                worst_G = max(worst_G, abs(kc - gq) / scale)
            else:
                worst_sel = max(worst_sel, abs(kq) / scale, abs(gq) / scale, abs(kc))
        print(f"{name:11s} max|K| = {scale:.4f}")
    print(f"{n_pairs} ordered pairs (TE/TM, m, n <= 4, 5 guides)")
    print(f"  closed form vs quadrature of box 2:         max rel. error {worst_K:.2e}")
    print(f"  closed form vs G^T (complex-step d/dtheta): max rel. error {worst_G:.2e}")
    print(f"  selection rule (pairs that must vanish):     max |K| / scale {worst_sel:.2e}")
    print(f"  explicit rational form (box 2b) vs box 2a:    max rel. error {worst_X[0]:.2e}")
    big = ModeBasis.below(60e9, 47.549e-3, 22.149e-3)
    ev = np.max(np.abs(K_matrix(big) - K_matrix_fast(big)))
    print(f"  vectorized K_matrix_fast vs K_matrix ({len(big)} modes): max abs. difference {ev:.1e}")
    assert worst_X[0] < 1e-12 and ev < 1e-12
    assert worst_K < 1e-10 and worst_G < 1e-10 and worst_sel < 1e-10
    return worst_K, worst_G, worst_sel


def check_symmetries():
    """K^MM_pq / chi_p^2 antisymmetric; K^EM = 0; A antisymmetric, B symmetric, both real (propagating modes).
    K^EE is NOT antisymmetric in general (printed for information)."""
    from wrmodes.modes import cutoff_wavenumber
    from twist_coupling import A_B_matrices
    worst = 0.0
    for name, (a, b) in GUIDES.items():
        basis = ModeBasis.below(4.2 * 3e8 / (2 * min(a, b)), a, b)
        K = K_matrix_fast(basis)
        sc = np.max(np.abs(K))
        chi2 = np.array([cutoff_wavenumber(m, a, b) ** 2 for m in basis.modes])
        te = np.array([m.kind == "TE" for m in basis.modes])
        EE = K[np.ix_(te, te)]
        MM = K[np.ix_(~te, ~te)] / chi2[~te][:, None]
        worst = max(worst, np.max(np.abs(MM + MM.T)) * chi2.max() / sc, np.max(np.abs(K[np.ix_(te, ~te)])) / sc)
        print(f"  {name:11s}: max|K^EE + K^EE^T| / max|K| = {np.max(np.abs(EE + EE.T)) / sc:.3f} (not antisymmetric)")
        f = 2.5 * 3e8 / (2 * min(a, b))
        A, B = A_B_matrices(basis.subset(np.where(basis.propagating_mask(f))[0]), f)
        worst = max(worst, np.max(np.abs(A + A.T)) / sc, np.max(np.abs(B - B.T)) / sc, np.max(np.abs(A.imag)) / sc)
    print(f"  K^MM/chi_p^2 + transpose, K^EM, A + A^T, B - B^T, Im A: max rel. {worst:.1e}")
    assert worst < 1e-12


def check_boundary_identity():
    """G + G^T = -oint (e_p . e_q)(u n_v - v n_u) dl: the symmetric part of G is a pure wall term (section 3)."""
    worst = 0.0
    x, w = np.polynomial.legendre.leggauss(96)
    for name, (a, b) in GUIDES.items():
        modes = modes_upto(3)
        K = np.array([[K_element(p, q, a, b) for q in modes] for p in modes])
        G = K.T
        Bd = np.zeros_like(G)
        uu, wu = x * a / 2, w * a / 2
        vv, wv = x * b / 2, w * b / 2
        for i, p in enumerate(modes):
            for j, q in enumerate(modes):
                tot = 0.0
                for us, sgn in ((a / 2, 1.0), (-a / 2, -1.0)):           # n = (sgn, 0): u n_v - v n_u = -sgn v
                    pu, pv = transverse_field(p, a, b, us, vv)
                    qu, qv = transverse_field(q, a, b, us, vv)
                    tot += -np.sum(wv * (pu * qu + pv * qv) * (-sgn * vv))
                for vs, sgn in ((b / 2, 1.0), (-b / 2, -1.0)):           # n = (0, sgn): u n_v - v n_u = sgn u
                    pu, pv = transverse_field(p, a, b, uu, vs)
                    qu, qv = transverse_field(q, a, b, uu, vs)
                    tot += -np.sum(wu * (pu * qu + pv * qv) * (sgn * uu))
                Bd[i, j] = tot
        worst = max(worst, np.max(np.abs(G + G.T - Bd)) / np.max(np.abs(K)))
    print(f"  G + G^T = wall integral: max rel. error {worst:.1e}")
    assert worst < 1e-10


def check_generator():
    """G e = z x e - (u d_v - v d_u) e, against the complex-step rotation derivative, pointwise."""
    a, b = 47.549e-3, 22.149e-3
    U, V, _ = grid(a, b, 24)
    h = 1e-30
    worst = 0.0
    for q in modes_upto(3):
        du, dv = dtheta_field(q, a, b, U, V)
        eu, ev = transverse_field(q, a, b, U, V)
        # complex-step partial derivatives of the field
        eu_u = transverse_field(q, a, b, U + 1j * h, V)[0].imag / h
        ev_u = transverse_field(q, a, b, U + 1j * h, V)[1].imag / h
        eu_v = transverse_field(q, a, b, U, V + 1j * h)[0].imag / h
        ev_v = transverse_field(q, a, b, U, V + 1j * h)[1].imag / h
        Gu = -ev - (U * eu_v - V * eu_u)
        Gv = eu - (U * ev_v - V * ev_u)
        sc = np.max(np.hypot(du, dv))
        worst = max(worst, np.max(np.abs(Gu - du)) / sc, np.max(np.abs(Gv - dv)) / sc)
    # rotation-invariant field e = (u, v): R(th) e(R(-th) r) = e(r)
    th = 1j * h
    c, s = np.cos(th), np.sin(th)
    u0, v0 = c * U + s * V, -s * U + c * V
    inv = max(np.max(np.abs((c * u0 - s * v0).imag / h)), np.max(np.abs((s * u0 + c * v0).imag / h)))
    print(f"  generator formula vs complex-step d/dtheta: max rel. error {worst:.2e}; G(u, v) = {inv:.1e}")
    assert worst < 1e-12 and inv == 0.0
    return worst


def check_sympy():
    """Exact integrals with symbolic a, b for a few pairs, against the closed form evaluated at two guides."""
    u, v = sp.symbols("u v", real=True)
    A, Bs = sp.symbols("a b", positive=True)

    def pot(mode):
        eps = lambda l: 1 if l == 0 else 2
        up, vp = mode.m * sp.pi * (u + A / 2) / A, mode.n * sp.pi * (v + Bs / 2) / Bs
        den = sp.pi * sp.sqrt(mode.m ** 2 * Bs / A + mode.n ** 2 * A / Bs)
        if mode.kind == "TE":
            return sp.sqrt(eps(mode.m) * eps(mode.n)) / den * sp.cos(up) * sp.cos(vp)
        return 2 / den * sp.sin(up) * sp.sin(vp)

    def field(mode):
        P = pot(mode)
        if mode.kind == "TM":
            return sp.Matrix([sp.diff(P, u), sp.diff(P, v)])
        return sp.Matrix([sp.diff(P, v), -sp.diff(P, u)])

    def div(e):
        return sp.diff(e[0], u) + sp.diff(e[1], v)

    def zx(e):
        return sp.Matrix([-e[1], e[0]])

    def K_sym(p, q):
        ep, eq = field(p), field(q)
        integrand = div(ep) * (v * eq[0] - u * eq[1]) + (u * ep[0] + v * ep[1]) * div(zx(eq))
        I = sp.integrate(sp.integrate(sp.expand(integrand), (u, -A / 2, A / 2)), (v, -Bs / 2, Bs / 2))
        return sp.simplify(I)

    pairs = [(Mode("TE", 1, 0), Mode("TE", 0, 1)), (Mode("TE", 0, 1), Mode("TE", 1, 0)),
             (Mode("TE", 1, 0), Mode("TE", 2, 1)), (Mode("TM", 2, 1), Mode("TE", 1, 0)),
             (Mode("TM", 1, 2), Mode("TM", 2, 1)), (Mode("TE", 1, 0), Mode("TM", 2, 1)),
             (Mode("TM", 1, 1), Mode("TE", 2, 2))]
    worst = 0.0
    for p, q in pairs:
        Ks = K_sym(p, q)
        print(f"  K[{p.name}, {q.name}] = {sp.simplify(Ks)}")
        for (a, b) in [(1.0, 1.0), (47.549e-3, 22.149e-3)]:
            val = float(Ks.subs({A: a, Bs: b}))
            worst = max(worst, abs(val - K_element(p, q, a, b)) / max(1.0, abs(val)))
    c = K_sym(Mode("TE", 1, 0), Mode("TE", 0, 1))
    print(f"  c = K[TE1,0, TE0,1] = G[TE0,1, TE1,0] = {c}  (8/pi^2 = {8 / np.pi ** 2:.16f})")
    assert sp.simplify(c - 8 / sp.pi ** 2) == 0
    print(f"  sympy vs closed form: max rel. error {worst:.2e}")
    assert worst < 1e-12
    return worst


if __name__ == "__main__":
    print("1-4. quadrature")
    check_quadrature()
    check_symmetries()
    check_boundary_identity()
    check_generator()
    print("5. sympy")
    check_sympy()
    print("all closed-form checks passed")
