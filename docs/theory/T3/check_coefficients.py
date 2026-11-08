"""T3 check 1: closed-form coupling coefficients against quadrature (acceptance test: 1e-10 relative).

* C_pq = iint e_p . G e_q with G e = z x e - (u d_v - v d_u) e, evaluated by Gauss-Legendre quadrature from
  wrmodes.modes.transverse_field (derivatives by complex step, exact to rounding), against coupling_C.
* the generator: G (u, v) = 0 for the rotation-invariant field (u, v); G e equals d/dtheta of the actively
  rotated field R(theta) e(R(-theta) r) at theta = 0 (finite differences);
* selection rules: C_pq = 0 unless m_p + m_q and n_p + n_q are odd;
* symmetric part: C + C^T = -oint (e_p.n)(e_q.n)((z x rho).n) dl (1D quadrature on the walls);
* TE10-TE01: C = 8/pi^2 for any aspect ratio.

Run: PYTHONPATH=src python3 docs/theory/T3/check_coefficients.py
"""
import os
import sys

import numpy as np
import scipy.constants as cst

sys.path.insert(0, os.path.dirname(__file__))
from twist_formulas import coupling_matrix, coupling_C  # noqa: E402
from wrmodes.modes import Mode, ModeBasis, transverse_field  # noqa: E402

H = 1e-30


def grid(a, b, nq):
    x, w = np.polynomial.legendre.leggauss(nq)
    U, V = np.meshgrid(x * a / 2, x * b / 2, indexing="ij")
    return U, V, np.outer(w * a / 2, w * b / 2)


def field(mode, a, b, U, V):
    return np.array(transverse_field(mode, a, b, U, V))


def generator(mode, a, b, U, V):
    """G e = z x e - (u d_v - v d_u) e, derivatives by complex step."""
    e = field(mode, a, b, U, V)
    du = np.imag(np.array(transverse_field(mode, a, b, U + 1j * H, V))) / H
    dv = np.imag(np.array(transverse_field(mode, a, b, U, V + 1j * H))) / H
    zxe = np.array([-e[1], e[0]])
    return zxe - (U * dv - V * du)


def quadrature_C(basis, nq=96):
    a, b = basis.a, basis.b
    U, V, W = grid(a, b, nq)
    E = [field(m, a, b, U, V) for m in basis]
    G = [generator(m, a, b, U, V) for m in basis]
    return np.array([[np.sum(W * (E[p] * G[q]).sum(0)) for q in range(len(basis))] for p in range(len(basis))])


def wall_integral(basis, nq=96):
    """oint (e_p.n)(e_q.n)((z x rho).n) dl."""
    a, b = basis.a, basis.b
    x, w = np.polynomial.legendre.leggauss(nq)
    N = len(basis)
    B = np.zeros((N, N))
    walls = []
    for s in (+1, -1):     # u = s a/2, n = s u_hat, (z x rho).n = s * (-v)
        v, wv = x * b / 2, w * b / 2
        u = np.full_like(v, s * a / 2)
        walls.append((u, v, wv, np.array([s, 0.0]), s * (-v)))
        u2, wu = x * a / 2, w * a / 2     # v = s b/2, n = s v_hat, (z x rho).n = s * u
        v2 = np.full_like(u2, s * b / 2)
        walls.append((u2, v2, wu, np.array([0.0, s]), s * u2))
    for u, v, wl, nvec, rot in walls:
        en = [np.tensordot(nvec, field(m, a, b, u, v), 1) for m in basis]
        for p in range(N):
            for q in range(N):
                B[p, q] += np.sum(wl * en[p] * en[q] * rot)
    return B


def main():
    worst = 0.0
    cases = [(0.02286, 0.01016), (0.01016, 0.02286), (0.02, 0.02), (0.044, 0.030),
             (0.872 * 0.0254, 1.872 * 0.0254)]
    for a, b in cases:
        basis = ModeBasis.below(3.2 * cst.c / (2 * min(a, b)), a, b)
        Cq = quadrature_C(basis)
        Cc = coupling_matrix(basis)
        scale = np.max(np.abs(Cq))
        err = np.max(np.abs(Cq - Cc)) / scale
        worst = max(worst, err)
        # selection rule, checked on the quadrature values
        for i, p in enumerate(basis):
            for j, q in enumerate(basis):
                if (p.m + q.m) % 2 == 0 or (p.n + q.n) % 2 == 0:
                    assert abs(Cq[i, j]) < 1e-11 * scale, (p, q, Cq[i, j])
        # TM -> TE row is zero
        for i, p in enumerate(basis):
            for j, q in enumerate(basis):
                if p.kind == "TM" and q.kind == "TE":
                    assert abs(Cq[i, j]) < 1e-11 * scale
        # symmetric part from the wall integral
        B = wall_integral(basis)
        errS = np.max(np.abs((Cq + Cq.T) + B)) / scale
        worst = max(worst, errS)
        n_sym = int(np.sum(np.abs(Cq + Cq.T) > 1e-6 * scale) // 2)
        print(f"a={a*1e3:7.3f} mm b={b*1e3:7.3f} mm: {len(basis):3d} modes, closed form vs quadrature "
              f"{err:.2e}; C+C^T + wall integral {errS:.2e}; pairs with C+C^T != 0: {n_sym}")
    assert worst < 1e-10, worst

    # generator checks
    a, b = 0.03, 0.02
    U, V, W = grid(a, b, 8)
    # rotation-invariant field (u, v)
    e = np.array([U, V]).astype(complex)
    du = np.array([np.ones_like(U), np.zeros_like(U)])
    dv = np.array([np.zeros_like(U), np.ones_like(U)])
    G = np.array([-e[1], e[0]]) - (U * dv - V * du)
    assert np.max(np.abs(G)) == 0.0
    # active rotation e_theta(r) = R(theta) e(R(-theta) r): d/dtheta at 0 equals G e
    mode = Mode("TM", 2, 1)
    t = 1e-6

    def rotated(th):
        c, s = np.cos(th), np.sin(th)
        ur, vr = c * U + s * V, -s * U + c * V        # R(-theta) r
        eu, ev = transverse_field(mode, a, b, ur, vr)
        return np.array([c * eu - s * ev, s * eu + c * ev])
    fd = (rotated(t) - rotated(-t)) / (2 * t)
    Ge = generator(mode, a, b, U, V)
    rel = np.max(np.abs(fd - Ge)) / np.max(np.abs(Ge))
    print(f"generator vs d/dtheta of the actively rotated field: {rel:.1e}")
    assert rel < 1e-7

    # TE10-TE01: 8/pi^2 for any aspect ratio
    for a, b in [(1.0, 1.0), (0.02286, 0.01016), (0.872, 1.872), (1.0, 0.1), (0.1, 1.0)]:
        c = coupling_C("TE0,1", "TE1,0", a, b)
        basis = ModeBasis(a, b, ["TE1,0", "TE0,1"])
        cq = quadrature_C(basis)[basis.index("TE0,1"), basis.index("TE1,0")]
        print(f"a/b = {a/b:6.3f}: C(TE01, TE10) closed form {c:.15f}, quadrature {cq:.15f}, 8/pi^2 = {8/np.pi**2:.15f}")
        assert abs(c - 8 / np.pi ** 2) < 1e-15 and abs(cq - 8 / np.pi ** 2) < 1e-12
    print(f"worst relative error {worst:.2e} (target 1e-10): PASS")


if __name__ == "__main__":
    main()
