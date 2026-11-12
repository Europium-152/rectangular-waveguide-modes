"""WP5 acceptance checks with numbers for docs/reports/WP5.md (the same checks as tests/test_taper.py, reported).

* uniform limit: VariableGuide with constant a, b (kappa = 0 or 1/R) vs Straight / SmoothBend, lossless and lossy
  (aluminium), H and E plane, propagating modes (mode_factor 1), max |S - S_ref|;
* single mode through cutoff (Airy) and linear taper (Bessel): max |S - S_exact| vs the slice tolerance;
* total reflection of TE3,0 at its cutoff plane (T2 section 8.5): 1 - |S22| and the phase vs i exp(-2i int beta);
* reciprocity, unitarity (lossless) and passivity (lossy) of a general geometry (a', b', u0', v0', kappa) and of the
  strangled bend, with evanescent port modes.

Run: PYTHONPATH=src python benchmarks/taper/checks.py   (about 1 min)
Output: benchmarks/taper/data/checks.npz
"""
import os
import sys
import warnings

import numpy as np
import scipy.constants as cst
from scipy.special import airy, jv, yv, jvp, yvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

from wrmodes import Straight, SmoothBend  # noqa: E402
from wrmodes.components import Guide, VariableGuide, Taper, StrangledBend, GeometryFunction  # noqa: E402
from wrmodes.modes import Mode, ModeBasis, wave_admittance  # noqa: E402


def exact_single(mode, a0, a1, b, f, Phi):
    def rows(a):
        s = np.sqrt(complex(wave_admittance(mode, a, b, f)))
        return np.array([[s, 1 / s]]) / (2 * np.sqrt(2)), np.array([[s, -1 / s]]) / (2 * np.sqrt(2))
    Pp0, Pm0 = rows(a0)
    Pp1, Pm1 = rows(a1)
    return np.vstack([Pm0, Pp1 @ Phi]) @ np.linalg.inv(np.vstack([Pp0, Pm1 @ Phi]))


def main():
    out = {}
    # uniform limit
    f, L, R = 30e9, 0.1, 0.2
    rows = []
    for sigma in (None, 3.5e7):
        g = Guide(22.86e-3, 10.16e-3, sigma)
        b = g.basis(f, 1.0)
        d = np.max(np.abs(VariableGuide(L, 22.86e-3, 10.16e-3, guide=g, mode_factor=1.0, families="coupled")
                          .gsm(f, b, b).S - Straight(L, g).gsm(f, b).S))
        rows.append(("straight", "-", sigma, d))
        for plane in ("H", "E"):
            a0, b0 = g.dims(plane)
            bb = g.basis(f, 1.0, plane=plane)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = SmoothBend(R * np.pi / 2, 1 / R, plane=plane, guide=g, atol=1e-12).gsm(f, bb)
            vg = VariableGuide(R * np.pi / 2, a0, b0, kappa=1 / R, plane=plane, guide=g, mode_factor=1.0,
                               uniform=[(0, R * np.pi / 2)])
            rows.append(("bend", plane, sigma, np.max(np.abs(vg.gsm(f, bb, bb).S - ref.S))))
    print("uniform limit (max |S - S_ref|, propagating modes):")
    for r in rows:
        print(f"   {r[0]:8s} {r[1]} {'lossless' if r[2] is None else 'aluminium'}: {r[3]:.1e}")
    out["uniform_limit"] = np.array([r[3] for r in rows])
    # Airy
    f = 10e9
    k, w = 2 * np.pi * f / cst.c, 2 * np.pi * f
    gam, zc, L, b = k * k / 0.3, 0.03, 0.10, 0.01016
    s = gam ** (1 / 3)
    a = GeometryFunction(lambda z: np.pi / np.sqrt(k * k - gam * (np.asarray(z) - zc)),
                         lambda z: 0.5 * np.pi * gam * (k * k - gam * (np.asarray(z) - zc)) ** -1.5)

    def sols(z):
        ai, aip, bi, bip = airy(-s * (z - zc))
        return np.array([[ai, bi], 1j * (-s) * np.array([aip, bip]) / (w * cst.mu_0)])
    te10 = Mode("TE", 1, 0)
    Sx = exact_single(te10, a(0.0), a(L), b, f, sols(L) @ np.linalg.inv(sols(0.0)))
    airy_err = []
    for atol in (1e-6, 1e-8, 1e-11):
        G = VariableGuide(L, a, b, mode_factor=1.0, tail=False, atol=atol).gsm(
            f, ModeBasis(a(0.0), b, [te10]), ModeBasis(a(L), b, [te10]))
        airy_err.append((atol, G.info["n_slices"][(1, 0)], np.max(np.abs(G.S - Sx))))
    # Bessel
    f = 12e9
    k, w = 2 * np.pi * f / cst.c, 2 * np.pi * f
    a0, alpha, L = 0.014, 0.25, 0.08
    nu = np.sqrt(0.25 + (np.pi / alpha) ** 2)
    K = k / alpha

    def solb(z):
        x = a0 + alpha * z
        V = np.array([np.sqrt(x) * jv(nu, K * x), np.sqrt(x) * yv(nu, K * x)])
        dV = np.array([jv(nu, K * x) / (2 * np.sqrt(x)) + np.sqrt(x) * K * jvp(nu, K * x),
                       yv(nu, K * x) / (2 * np.sqrt(x)) + np.sqrt(x) * K * yvp(nu, K * x)])
        return np.array([V, 1j * alpha * dV / (w * cst.mu_0)])
    Sx = exact_single(te10, a0, a0 + alpha * L, b, f, solb(L) @ np.linalg.inv(solb(0.0)))
    afun = GeometryFunction(lambda z: a0 + alpha * np.asarray(z), lambda z: alpha + 0 * np.asarray(z))
    bes_err = []
    for atol in (1e-6, 1e-8, 1e-11):
        G = VariableGuide(L, afun, b, mode_factor=1.0, tail=False, atol=atol).gsm(
            f, ModeBasis(a0, b, [te10]), ModeBasis(a0 + alpha * L, b, [te10]))
        bes_err.append((atol, G.info["n_slices"][(1, 0)], np.max(np.abs(G.S - Sx))))
    print("Airy (atol, slices, max |S - S_exact|): " + "; ".join(f"{x[0]:.0e}, {x[1]}, {x[2]:.1e}" for x in airy_err))
    print("Bessel: " + "; ".join(f"{x[0]:.0e}, {x[1]}, {x[2]:.1e}" for x in bes_err))
    out["airy"] = np.array(airy_err)
    out["bessel"] = np.array(bes_err)
    # total reflection
    f = 20e9
    k = 2 * np.pi * f / cst.c
    a0, a1, L = 0.0205, 0.0275, 0.25
    zc = (3 * np.pi / k - a0) * L / (a1 - a0)
    zz = np.linspace(zc, L, 40001)
    beta = np.sqrt(np.maximum(k * k - (3 * np.pi / (a0 + (a1 - a0) * zz / L)) ** 2, 0))
    phase = np.trapezoid(beta, zz)
    tr = []
    for mf, tail in ((2.0, True), (4.0, True), (2.0, False), (4.0, False)):
        G = Taper(L, a1, b, "linear", guide=Guide(a0, b), mode_factor=mf, tail=tail).gsm(f)
        i = G.basis2.index("TE3,0")
        S22 = G.S22[i, i]
        tr.append((mf, tail, 1 - abs(S22), np.angle(S22 / (1j * np.exp(-2j * phase)))))
    print("total reflection TE3,0 (mf, B5', 1 - |S22|, phase error rad): "
          + "; ".join(f"{x[0]:g}, {x[1]}, {x[2]:.1e}, {x[3]:.4f}" for x in tr))
    out["total_reflection"] = np.array([[x[0], x[1], x[2], x[3]] for x in tr], dtype=float)
    # properties
    Lg = 0.12
    sn = lambda z: np.sin(np.pi * np.asarray(z) / Lg)
    cs = lambda z: np.cos(np.pi * np.asarray(z) / Lg)
    geo = dict(a=GeometryFunction(lambda z: 0.0229 * (1 + 0.35 * sn(z) ** 2),
                                  lambda z: 0.0229 * 0.35 * 2 * sn(z) * cs(z) * np.pi / Lg),
               b=GeometryFunction(lambda z: 0.0102 * (1 + 0.2 * np.asarray(z) / Lg), lambda z: 0.0102 * 0.2 / Lg + 0 * z),
               u0=GeometryFunction(lambda z: 0.002 * sn(z), lambda z: 0.002 * cs(z) * np.pi / Lg),
               v0=GeometryFunction(lambda z: 0.001 * sn(z) ** 2, lambda z: 0.001 * 2 * sn(z) * cs(z) * np.pi / Lg),
               kappa=GeometryFunction(lambda z: 4.0 * sn(z), lambda z: 4.0 * cs(z) * np.pi / Lg))
    props = []
    for sigma in (None, 3.5e7):
        g = Guide(0.0229, 0.0102, sigma)
        vg = VariableGuide(Lg, geo["a"], geo["b"], kappa=geo["kappa"], u0=geo["u0"], v0=geo["v0"], guide=g)
        b1, b2 = vg.default_basis(20e9, 1.5)
        G = vg.gsm(20e9, b1, b2)
        props.append(("general (a', b', u0', v0', kappa), 20 GHz, cf 1.5", sigma, G.reciprocity_defect() /
                      np.max(np.abs(G.S)), G.unitarity_defect(), G.passivity_margin()))
        sb = StrangledBend(90, 0.05, 1.6e-3, 0.05, guide=Guide(22.86e-3, 10.16e-3, sigma))
        b1, b2 = sb.default_basis(40e9, 1.5)
        G = sb.gsm(40e9, b1, b2)
        props.append(("strangled bend E3, 40 GHz, cf 1.5", sigma, G.reciprocity_defect() / np.max(np.abs(G.S)),
                      G.unitarity_defect(), G.passivity_margin()))
    print("properties (reciprocity rel., unitarity, passivity margin):")
    for p in props:
        print(f"   {p[0]}, {'lossless' if p[1] is None else 'aluminium'}: {p[2]:.1e}, {p[3]:.1e}, {p[4]:.2e}")
    out["properties"] = np.array([[p[2], p[3], p[4]] for p in props])
    m = meta("benchmarks/taper/checks.py", "WP5 acceptance checks", uniform=dict(f=30e9, L=0.1, R=0.2),
             rows_uniform=[f"{r[0]} {r[1]} {r[2]}" for r in rows], rows_properties=[f"{p[0]} {p[1]}" for p in props],
             airy_columns="atol, slices, max|S - S_exact|", total_reflection_columns="mf, tail, 1-|S22|, phase err")
    save("checks.npz", m, **out)


if __name__ == "__main__":
    main()
