"""Check of the full V-I equations (box B1-B3) against exact Maxwell solutions (card T2b).

For an exact field in a wedge guide (exact_wedge.WedgeField), the local-mode amplitudes
V_p(s) = int E_t . e_p, I_p = int H_t . (z x e_p) are computed by quadrature on the cross-sections of a chosen
centre-line frame (straight or curved), and the residuals

    r^V_p = dV_p/ds - sum_q [ -i Z_pq I_q + T_pq V_q ],     r^I_p = dI_p/ds - sum_q [ -i Y_pq V_q - T_qp I_q ]

are evaluated with a truncated basis of M modes (derivatives d/ds by 8th-order central differences). They must
go to zero as M grows. Cases: symmetric and asymmetric H-plane wedges (a', u0'), families n = 0 and n = 1 (TE/TM
mixing), an E-plane wedge (b', v0', coupling across families), and an asymmetric H-plane wedge seen from a
curved centre line (kappa, a', u0' together).  Run:  PYTHONPATH=src python3 docs/theory/T2b/check_exact_residual.py
"""
import sys, os
import numpy as np
from numpy.polynomial.legendre import leggauss
sys.path.insert(0, os.path.dirname(__file__))
from wrmodes.modes import Mode, transverse_field
from t2b_formulas import T_matrix, ZY, beta_Y
from exact_wedge import WedgeField

FD = np.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])   # 8th-order d/dx


class Frame:
    """Centre-line frame: straight (kappa = 0) along Zc, or a circular arc of curvature kappa in the X-Zc plane
    (centre at u = -1/kappa, CONVENTIONS.md 2), starting at X = x_start, Zc = 0."""
    def __init__(self, kappa=0.0, x_start=0.0):
        self.kappa, self.x0 = kappa, x_start

    def at(self, s):
        if self.kappa == 0:
            return np.array([self.x0, 0, s]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        R = 1 / self.kappa
        th = s / R
        P = np.array([self.x0 - R + R * np.cos(th), 0, R * np.sin(th)])
        return P, np.array([np.cos(th), 0, np.sin(th)]), np.array([0, 1.0, 0])


def wall_coord(frame, s, apex, phi, along):
    """Coordinate (u if along='u', v if along='v') where the cross-section line meets the wall ray at angle phi."""
    P, uh, vh = frame.at(s)
    e = uh if along == "u" else vh
    # wall plane through the apex, spanned by the uniform direction and the in-plane direction d
    if along == "u":
        d = np.array([np.sin(phi), 0, np.cos(phi)])
    else:
        d = np.array([0, np.sin(phi), np.cos(phi)])
    # P + c e = apex + t d + (anything along the uniform direction); solve in the wedge plane
    idx = [0, 2] if along == "u" else [1, 2]
    Mx = np.array([[e[idx[0]], -d[idx[0]]], [e[idx[1]], -d[idx[1]]]])
    rhs = (apex - P)[idx]
    c, _ = np.linalg.solve(Mx, rhs)
    return c


def geometry(frame, s, case):
    if case["along"] == "u":
        um = wall_coord(frame, s, case["apex"], case["phis"][0], "u")
        up = wall_coord(frame, s, case["apex"], case["phis"][1], "u")
        return up - um, case["width"], 0.5 * (up + um), 0.0
    vm = wall_coord(frame, s, case["apex"], case["phis"][0], "v")
    vp = wall_coord(frame, s, case["apex"], case["phis"][1], "v")
    return case["width"], vp - vm, 0.0, 0.5 * (vp + vm)


def amplitudes(field, frame, s, case, modes, nq=None):
    nq = nq or max(48, 2 * max(max(p.m, p.n) for p in modes) + 24)
    a, b, u0, v0 = geometry(frame, s, case)
    P, uh, vh = frame.at(s)
    x, w = leggauss(nq)
    x, w = 0.5 * x, 0.5 * w
    U, Vv = np.meshgrid(a * x, b * x, indexing="ij")
    Wt = np.outer(w, w) * a * b
    pts = P + (u0 + U)[..., None] * uh + (v0 + Vv)[..., None] * vh
    E, H = field.fields(pts)
    Eu, Ev, Hu, Hv = E @ uh, E @ vh, H @ uh, H @ vh
    Vamp, Iamp = [], []
    for p in modes:
        eu, ev = transverse_field(p, a, b, U, Vv)
        Vamp.append(np.sum(Wt * (Eu * eu + Ev * ev)))
        Iamp.append(np.sum(Wt * (-Hu * ev + Hv * eu)))
    return np.array(Vamp), np.array(Iamp)


def residual(case, s, M, f, moving=False):
    frame = case["frame"]
    field = case["field"]
    modes = case["basis"](M)
    h = case["h"]
    Vs, Is, G = [], [], []
    for j in range(-4, 5):
        V, I = amplitudes(field, frame, s + j * h, case, modes)
        Vs.append(V); Is.append(I); G.append(geometry(frame, s + j * h, case))
    Vs, Is, G = np.array(Vs), np.array(Is), np.array(G)
    dV = FD @ Vs / h
    dI = FD @ Is / h
    dG = FD @ G / h
    V, I = Vs[4], Is[4]
    a, b, u0, v0 = G[4]
    da, db, du0, dv0 = dG
    Z, Y = ZY(modes, f, a, b, u0, frame.kappa, nq=max(64, 2 * M + 24))
    T = T_matrix(modes, a, b, da, db, du0, dv0)
    rV = dV - (-1j * Z @ I + (-T.T if moving else T) @ V)
    rI = dI - (-1j * Y @ V - T.T @ I)
    beta, Yw = beta_Y(modes, f, a, b)
    return modes, rV / np.max(np.abs(dV)), rI / np.max(np.abs(dI)), (a, b, u0, v0, da, db, du0, dv0), beta


def family_basis(n, kinds=("TE", "TM")):
    def mk(M):
        out = []
        for m in range(0, M + 1):
            for kd in kinds:
                if (kd == "TE" and m + n > 0) or (kd == "TM" and m * n > 0):
                    out.append(Mode(kd, m, n))
        return out
    return mk


def column_basis(m):
    def mk(M):
        out = []
        for n in range(0, M + 1):
            for kd in ("TE", "TM"):
                if (kd == "TE" and m + n > 0) or (kd == "TM" and m * n > 0):
                    out.append(Mode(kd, m, n))
        return out
    return mk


def cases(f):
    X, Yv, Zv = np.eye(3)
    b0 = 0.0102
    out = []
    # 1. symmetric H-plane wedge, family n = 0 (pure LSM: TE_m0), outgoing j = 1 and 3 + incoming j = 3
    ph = 0.12
    apex = np.array([0.0, 0.0, -0.10])
    fld = WedgeField(f, apex, -ph, ph, b0, Yv, X, [(1.0, "LSM", 1, 0, 2), (0.6, "LSM", 3, 0, 2), (0.3, "LSM", 3, 0, 1)])
    out.append(dict(name="H-plane symmetric wedge, n=0", frame=Frame(), field=fld, along="u", apex=apex,
                    phis=(-ph, ph), width=b0, basis=family_basis(0), h=2e-4))
    # 2. asymmetric H-plane wedge (a' and u0'), family n = 1: LSM + LSE -> TE_m1 and TM_m1 mixed
    apex = np.array([-0.004, 0.0, -0.12])
    phs = (-0.05, 0.17)
    fld = WedgeField(f, apex, *phs, b0, Yv, X, [(1.0, "LSM", 1, 1, 2), (0.7, "LSE", 2, 1, 2), (0.4, "LSE", 1, 1, 1)])
    out.append(dict(name="H-plane asymmetric wedge, n=1", frame=Frame(), field=fld, along="u", apex=apex,
                    phis=phs, width=b0, basis=family_basis(1), h=2e-4))
    # 3. E-plane asymmetric wedge (b' and v0'), column m = 1: couples families n
    a0 = 0.02286
    apex = np.array([0.0, 0.003, -0.08])
    phs = (-0.04, 0.10)
    fld = WedgeField(f, apex, *phs, a0, X, Yv, [(1.0, "LSE", 0, 1, 2), (0.5, "LSM", 1, 1, 2), (0.5, "LSE", 2, 1, 2)])
    out.append(dict(name="E-plane asymmetric wedge, m=1", frame=Frame(), field=fld, along="v", apex=apex,
                    phis=phs, width=a0, basis=column_basis(1), h=2e-4))
    # 4. asymmetric H-plane wedge seen from a curved centre line (kappa = 1/0.25 m), family n = 1
    apex = np.array([-0.004, 0.0, -0.12])
    phs = (-0.05, 0.17)
    fld = WedgeField(f, apex, *phs, b0, Yv, X, [(1.0, "LSM", 1, 1, 2), (0.7, "LSE", 2, 1, 2), (0.4, "LSE", 1, 1, 1)])
    out.append(dict(name="curved frame on asymmetric wedge, n=1", frame=Frame(kappa=4.0, x_start=0.0), field=fld,
                    along="u", apex=apex, phis=phs, width=b0, basis=family_basis(1), h=2e-4))
    return out


def wall_condition(case, s):
    """Max over the side walls of |h E_z + w_n E_n| / max|E|: the tilted-wall PEC condition E_zeta = 0."""
    frame = case["frame"]
    h = case["h"]
    G = np.array([geometry(frame, s + j * h, case) for j in range(-4, 5)])
    dG = FD @ G / h
    a, b, u0, v0 = G[4]
    P, uh, vh = frame.at(s)
    zh = np.cross(uh, vh)
    t = np.linspace(-0.45, 0.45, 7)
    worst, emax = 0.0, 0.0
    for sign in (-1, 1):
        if case["along"] == "u":
            uw = u0 + sign * a / 2
            wn = dG[2] + sign * dG[0] / 2
            pts = P + uw * uh + (v0 + b * t)[:, None] * vh
            E, _ = case["field"].fields(pts)
            hz = 1 + frame.kappa * uw
            res = hz * (E @ zh) + wn * (E @ uh)
        else:
            vw = v0 + sign * b / 2
            wn = dG[3] + sign * dG[1] / 2
            pts = P + (u0 + a * t)[:, None] * uh + vw * vh
            E, _ = case["field"].fields(pts)
            res = (E @ zh) + wn * (E @ vh)
        worst = max(worst, np.max(np.abs(res)))
        emax = max(emax, np.max(np.abs(E)))
    return worst / emax


if __name__ == "__main__":
    f = 30e9
    s_list = [0.02, 0.05]
    print(f"f = {f/1e9} GHz; residuals relative to max_p |dV_p/ds| (resp. |dI_p/ds|), worst of the first 4 modes")
    for case in cases(f):
        print("\n" + case["name"])
        for s in s_list:
            for M in (4, 8, 16, 32):
                modes, rV, rI, g, beta = residual(case, s, M, f)
                k = min(4, len(modes))
                print(f"  s={s:5.3f} M={M:3d} N={len(modes):3d}  |rV|={np.max(np.abs(rV[:k])):.2e}  "
                      f"|rI|={np.max(np.abs(rI[:k])):.2e}")
            _, rVm, _, _, _ = residual(case, s, 16, f, moving=True)
            print(f"    with the moving-domain matrix -T^T in the voltage equation (M=16): |rV| = {np.max(np.abs(rVm[:4])):.1e}")
            print(f"    tilted-wall condition |h E_z + w_n E_n| / max|E| on the side walls: {wall_condition(case, s):.1e}")
            a, b, u0, v0, da, db, du0, dv0 = g
            print(f"    geometry: a={a*1e3:.3f} b={b*1e3:.3f} mm u0={u0*1e3:.3f} v0={v0*1e3:.3f} mm "
                  f"a'={da:.4f} b'={db:.4f} u0'={du0:.4f} v0'={dv0:.4f}; modes and beta/k:")
            k0 = 2 * np.pi * f / 2.99792458e8
            print("    " + ", ".join(f"{p.name}:{(bt/k0).real:+.2f}{(bt/k0).imag:+.2f}j" for p, bt in zip(modes[:6], beta[:6])))
