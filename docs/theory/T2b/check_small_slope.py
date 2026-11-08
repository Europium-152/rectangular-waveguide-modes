"""Small-slope (single-scattering) formula of note T2b (box B6) vs the full solution (box B5).

Linear symmetric H-plane taper a(z) = a0 + (a1 - a0) z / L, b constant, f = 15 GHz, family n = 0 (TE_m0 only).
TE30 crosses cutoff inside the taper (a = 30 mm). Incident TE10 at port 1. Compared, as L grows (slope falls):
  * forward TE10 -> TE30 conversion at z = L: direct term + secondary wave generated backwards and totally
    reflected at the cutoff cross-section of TE30 (WKB reflection i exp(-2i int beta)); the secondary term is what
    makes the "bottle" profile of Butterweck & de Ronde (1968) better than a linear taper;
  * TE10 reflection: Gamma = -int B_11 exp(-2i phi_1) dz  (with B_11 = Y'/2Y, Katsenelenbaum 1998 eq. 4.47).
Run: PYTHONPATH=src python3 docs/theory/T2b/check_small_slope.py
"""
import sys, os
import numpy as np
from scipy.integrate import quad, solve_ivp
import scipy.constants as cst
sys.path.insert(0, os.path.dirname(__file__))
from wrmodes.modes import Mode
from t2b_formulas import power_wave_matrices, beta_Y
from t2b_solver import solve

f = 15e9
k = 2 * np.pi * f / cst.c
a0, a1, b0 = 0.02286, 0.045, 0.01016
p, q = Mode("TE", 1, 0), Mode("TE", 3, 0)


def beta(mode, a):
    return np.sqrt(k * k - (mode.m * np.pi / a) ** 2 + 0j)


def first_order(L):
    da = (a1 - a0) / L
    a = lambda z: a0 + da * z
    zc = (3 * np.pi / k - a0) / da                     # TE30 cutoff plane
    # phases phi(z1, z2) = int beta; closed form for a linear taper: int sqrt(k^2 - c^2/a^2) da / a'
    def Phi(mode, z1, z2):
        c = mode.m * np.pi

        def F(aa):
            s = np.sqrt(max(k * k * aa * aa - c * c, 0.0))
            return (s - c * np.arccos(min(c / (k * aa), 1.0))) / da
        return F(a(z2)) - F(a(z1))

    def AB(z):
        A, B = power_wave_matrices([p, q], f, a(z), b0, da, 0.0, 0.0, 0.0)
        return A[1, 0], B[1, 0], B[0, 0]

    def integrand_direct(t):
        z = zc + t * t
        A, B, _ = AB(z)
        return 2 * t * A * np.exp(-1j * (Phi(p, 0, z) + Phi(q, z, L)))

    def integrand_secondary(t):
        z = zc + t * t
        A, B, _ = AB(z)
        return 2 * t * (-1j) * B * np.exp(-1j * (Phi(p, 0, z) + Phi(q, zc, z) + Phi(q, zc, L)))
    T = np.sqrt(L - zc)
    cq = lambda g: quad(lambda t: g(t).real, 0, T, limit=400)[0] + 1j * quad(lambda t: g(t).imag, 0, T, limit=400)[0]
    direct, secondary = cq(integrand_direct), cq(integrand_secondary)
    refl = -(quad(lambda z: (AB(z)[2] * np.exp(-2j * Phi(p, 0, z))).real, 0, L, limit=400)[0]
             + 1j * quad(lambda z: (AB(z)[2] * np.exp(-2j * Phi(p, 0, z))).imag, 0, L, limit=400)[0])
    return direct, secondary, refl, zc


def first_order_regular(L):
    """Box B6, regular form (valid through the cutoff of q), from the reciprocity identity
    d/dz (V I^ - I V^) = s_V I^ - s_I V^ for mode q driven by s_V = T_qp V_p, s_I = -T_pq I_p:

        a_q^+(L) = -(1/4) int_0^L [ T_qp V_p I^_q + T_pq I_p V^_q ] dz,

    with (V_p, I_p) the WKB incident field of p (a_p^+ = exp(-i phi_p)) and (V^_q, I^_q) the single-mode solution of
    q that is incident from port 2 with unit amplitude (a^-_q(L) = 1) and, being totally reflected at its cutoff
    plane, decays towards port 1 (a^+_q(0) = 0). No power-wave normalization of q is needed anywhere."""
    from t2b_formulas import T_closed, chi2
    from wrmodes.modes import ETA0
    da = (a1 - a0) / L
    a = lambda z: a0 + da * z
    Tqp = lambda z: T_closed(q, p, a(z), b0, da, 0, 0, 0)
    Tpq = lambda z: T_closed(p, q, a(z), b0, da, 0, 0, 0)
    Tqq = lambda z: T_closed(q, q, a(z), b0, da, 0, 0, 0)
    Zq, Yq = k * ETA0, lambda z: (k * k - chi2(q, a(z), b0)) / (k * ETA0)      # TE
    bq0 = beta(q, a(0))
    Y0 = bq0 / (k * ETA0)
    y0 = np.array([np.sqrt(2) / np.sqrt(Y0), -np.sqrt(2) * np.sqrt(Y0)], complex)   # a^- = 1, a^+ = 0 at z = 0

    def rhs(z, y):
        V, I = y
        return [-1j * Zq * I + Tqq(z) * V, -1j * Yq(z) * V - Tqq(z) * I]
    sol = solve_ivp(rhs, (0, L), y0, method="DOP853", rtol=1e-11, atol=1e-14, dense_output=True)
    VL, IL = sol.y[:, -1]
    YL = beta(q, a(L)) / (k * ETA0)
    am_L = 0.5 * (VL * np.sqrt(YL) / np.sqrt(2) - IL / (np.sqrt(2) * np.sqrt(YL)))
    def phi_p(z):
        c = np.pi
        F = lambda aa: (np.sqrt(k * k * aa * aa - c * c) - c * np.arccos(c / (k * aa))) / da
        return F(a(z)) - F(a(0))

    def integrand(z):
        Vh, Ih = sol.sol(z) / am_L
        Yp = beta(p, a(z)) / (k * ETA0)
        e = np.exp(-1j * phi_p(z))
        Vp, Ip = np.sqrt(2) * e / np.sqrt(Yp), np.sqrt(2) * np.sqrt(Yp) * e
        return -0.25 * (Tqp(z) * Vp * Ih + Tpq(z) * Ip * Vh)
    pts = np.linspace(0, L, 41)
    tot = 0
    for z1, z2 in zip(pts[:-1], pts[1:]):
        tot += quad(lambda z: integrand(z).real, z1, z2, limit=200)[0] + 1j * quad(lambda z: integrand(z).imag, z1, z2, limit=200)[0]
    return tot


def full(L, M=10, m_tail=500):
    da = (a1 - a0) / L
    modes = [Mode("TE", m, 0) for m in range(1, M + 1)]
    geom = lambda z: dict(a=a0 + da * z, b=b0, da=da, db=0.0, du0=0.0, dv0=0.0, kappa=0.0)
    S11, S12, S21, S22 = solve(modes, f, geom, L, growth=3.0, m_tail=m_tail)
    return S21[2, 0], S11[0, 0]


if __name__ == "__main__":
    print(f"f = {f/1e9} GHz, a: {a0*1e3} -> {a1*1e3} mm, b = {b0*1e3} mm; TE30 cutoff at a = {3*np.pi/k*1e3:.2f} mm")
    print("  L (m)  slope   S21(TE30<-TE10): full (M=10 / change at M=14)       B6 regular   rel.err | WKB direct+secondary "
          "rel.err |  |S11(TE10)| full   B6   rel.err")
    for L in (0.1, 0.2, 0.4, 0.8):
        d, s2, r, zc = first_order(L)
        reg = first_order_regular(L)
        t30, r10 = full(L)
        t30b, _ = full(L, M=14)
        fo = d + s2
        print(f"  {L:5.2f} {(a1-a0)/L:7.4f}  {t30:.4e} / {abs(t30b - t30):.0e}  {reg:.4e}  {abs(reg - t30)/abs(t30):.1e} |"
              f" {abs(fo - t30)/abs(t30):.1e} | {abs(r10):.3e} {abs(r):.3e} {abs(r - r10)/abs(r10):.1e}", flush=True)
