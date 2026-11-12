"""WP6 benchmark: acceptance checks of the Twist component that are reported (not only asserted) in docs/reports/WP6.md.

1. +90 deg then -90 deg constant twists of finite length vs the identity (long-twist limit): square 30 mm at 16.5 GHz,
   tall WR187 at 9 and 16.5 GHz, L = 5, 20, 80, 320 wavelengths.
2. Helical modes of a uniformly twisted guide vs the exact (spectral) solution of T3b section 5 (a = 2, b = 1,
   tau = 0.5, gamma = 4): max |gamma_CM / gamma - 1| for chi_max = 3, 6, 12 k_max, with Richardson extrapolation.
3. Slice convergence of the variable-rate Doane twists (18 GHz, cutoff_factor 1): fixed 128 ... 4096 slices.
4. Wall loss: the 20-inch twists in aluminium (3.5e7 S/m) vs the lossless twists and the lossy straight.
5. Timing of one Doane twist per profile and cutoff factor.

Run: PYTHONPATH=src python benchmarks/twist/checks.py   (about 5 min, mostly the 1443-mode eigenproblem of item 2)
Output: benchmarks/twist/data/checks.npz
"""
import os
import sys
import time

import numpy as np
import scipy.constants as cst

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

from wrmodes.components import Guide, Twist, Straight  # noqa: E402
from wrmodes.components.twist import helical_propagation_constants  # noqa: E402
from wrmodes.modes import ModeBasis  # noqa: E402

INCH = 0.0254
A, B, L20 = 0.872 * INCH, 1.872 * INCH, 20 * INCH
TALL = Guide(A, B)
HELICAL_K = np.array([4.1528675745, 4.7062051643, 5.1629896646, 5.2696835618, 5.5431540615, 5.6122772221])


def finite_pairs():
    rows = []
    for name, g0, f in (("square 30 mm", Guide(0.03, 0.03), 16.5e9), ("WR187 tall", TALL, 9e9),
                        ("WR187 tall", TALL, 16.5e9)):
        lam = cst.c / f
        for n in (5, 20, 80, 320):
            g = Twist(90, n * lam, guide=g0).gsm(f) @ Twist(-90, n * lam, guide=g0).gsm(f)
            j = g.basis1.index("TE1,0")
            dev = np.abs(np.abs(g.S21) - np.eye(len(g.basis1))).max()
            rows.append((f, n, len(g.basis1), 1 - abs(g.S21[j, j]) ** 2, dev, g.unitarity_defect()))
            print(f"  {name:12s} {f / 1e9:5.1f} GHz  L = {n:3d} lambda ({len(g.basis1):2d} modes): TE1,0 lost "
                  f"{rows[-1][3]:.3e}, max ||S21| - I| {dev:.2e}, unitarity {rows[-1][5]:.0e}")
    return np.array(rows)


def helical():
    a, b, tau, gamma = 2.0, 1.0, 0.5, 4.0
    out = []
    prev = None
    for c in (3.0, 6.0, 12.0):
        basis = ModeBasis.below(c * HELICAL_K.max() * cst.c / (2 * np.pi), a, b)
        g = []
        for k in HELICAL_K:
            ev = helical_propagation_constants(basis, k * cst.c / (2 * np.pi), tau)
            g.append(ev[np.argmin(abs(ev - gamma))].real)
        d = np.array(g) / gamma - 1
        rich = np.nan if prev is None else np.max(np.abs((c * d - prev[1] * prev[0]) / (c - prev[1])))
        out.append((c, len(basis), np.max(np.abs(d)), rich))
        print(f"  chi_max = {c:4.0f} k_max ({len(basis):4d} modes): max |gamma_CM/gamma - 1| = {out[-1][2]:.3e}, "
              f"Richardson {rich:.2e}")
        prev = (d, c)
    return np.array(out)


def slices():
    out = []
    for p in ("cosine", "triangular"):
        f = 18e9
        port = TALL.basis(f)
        j = port.index("TE1,0")
        ref = None
        for n in (128, 256, 512, 1024, 2048, 4096):
            g = Twist(90, L20, p, guide=TALL, slices=n).gsm(f, port)
            S = g.S
            d = np.nan if ref is None else np.abs(S - ref).max()
            ref = S
            out.append((n, 1 - abs(g.S21[j, j]) ** 2, d))
            print(f"  {p:10s} {n:5d} slices: spurious {100 * out[-1][1]:.7f} %, max |dS| vs previous {d:.1e}")
        g = Twist(90, L20, p, guide=TALL).gsm(f, port)
        print(f"  {p:10s} adaptive: {g.info['n_slices']} slices, error estimate {g.info['error_estimate']:.1e}")
    return np.array(out)


def lossy():
    al = Guide(A, B, sigma=3.5e7)
    out = []
    for p in ("constant", "cosine", "triangular"):
        for f in (15e9, 18e9):
            g, g0, s = (Twist(90, L20, p, guide=al).gsm(f), Twist(90, L20, p, guide=TALL).gsm(f),
                        Straight(L20, guide=al).gsm(f))
            j = g.basis1.index("TE1,0")
            ohm = 1 - np.sum(np.abs(g.S21[:, j]) ** 2) - np.sum(np.abs(g.S11[:, j]) ** 2)
            ohm_s = 1 - abs(s.S21[j, j]) ** 2
            conv = np.sum(np.abs(g.S21[:, j]) ** 2) - abs(g.S21[j, j]) ** 2
            conv0 = np.sum(np.abs(g0.S21[:, j]) ** 2) - abs(g0.S21[j, j]) ** 2
            out.append((f, ohm, ohm_s, conv, conv0, g.passivity_margin(), g.reciprocity_defect()))
            print(f"  {p:10s} {f / 1e9:4.1f} GHz: ohmic {100 * ohm:.4f} % (straight {100 * ohm_s:.4f} %), converted "
                  f"{100 * conv:.4f} % (lossless {100 * conv0:.4f} %), passivity margin {out[-1][5]:.1e}, "
                  f"reciprocity {out[-1][6]:.0e}")
    return np.array(out)


def timing():
    out = []
    for p in ("constant", "cosine", "triangular"):
        for cf in (1.0, 1.5):
            t = time.perf_counter()
            g = Twist(90, L20, p, guide=TALL, cutoff_factor=cf).gsm(18e9, TALL.basis(18e9))
            dt = time.perf_counter() - t
            out.append((cf, g.info["n_modes"], g.info["n_slices"], dt))
            print(f"  {p:10s} cutoff_factor {cf}: {g.info['n_modes']} modes, {g.info['n_slices']} slices, {dt:.2f} s")
    return np.array(out)


def main():
    print("1. +90 then -90 deg constant twists of finite length")
    fp = finite_pairs()
    print("2. helical modes vs the spectral solution (T3b section 5)")
    he = helical()
    print("3. slice convergence at 18 GHz")
    sl = slices()
    print("4. wall loss (aluminium 3.5e7 S/m)")
    lo = lossy()
    print("5. timing (one twist, 18 GHz)")
    ti = timing()
    m = meta("benchmarks/twist/checks.py", "WP6 reported acceptance checks", a_m=A, b_m=B, length_m=L20,
             columns={"finite_pairs": "f, L/lambda, n_modes, TE10 lost power, max ||S21|-I|, unitarity defect",
                      "helical": "chi_max/k_max, n_modes, max |gamma_CM/gamma-1|, Richardson",
                      "slices": "n_slices, spurious, max |dS| vs previous (cosine rows then triangular)",
                      "lossy": "f, ohmic, ohmic straight, converted, converted lossless, passivity margin, reciprocity",
                      "timing": "cutoff_factor, n_modes, n_slices, seconds"}, sigma=3.5e7)
    print(save("checks.npz", m, finite_pairs=fp, helical=he, slices=sl, lossy=lo, timing=ti))


if __name__ == "__main__":
    main()
