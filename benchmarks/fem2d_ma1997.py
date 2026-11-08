"""2D-solver results for the mitered H-plane bends of Ma, Yamane and Yamashita (1997), for comparison with the
digitized curves of the L1 card (D3).

Reference: Z. Ma, T. Yamane, E. Yamashita, "Analysis and design of mitered H-plane bends in rectangular
waveguides by combining the mode-matching and port reflection coefficient methods", Electronics and
Communications in Japan, Part 2, 80(5), 1-7 (1997).

Geometry (p. 4 and the insets of Figs. 5, 6, 8, 9, 11): a = 15.799 mm, b = 7.899 mm, 90-degree H-plane bend, single-
mode band 9.49-19.0 GHz (f_c = 9.488 GHz). The outer corner is cut by a straight mirror whose ends lie on the two outer
walls at distance c from the outer corner; c/a = 0 is the right-angle corner bend, c/a = 1 the plain miter. In terms
of Alessandri's d (distance from the inner corner to the mirror, units of a): d = (2 - c/a)/sqrt(2).

Results computed:
  * Fig. 5:  c/a = 0,   |S21|^2 for 9.6-18.9 GHz.
  * Fig. 6, 8, 9: c/a = 1, |S21| (dB) and |S11| (dB), 9.6-18.9 GHz.
  * Table 1: c/a = 1 at 16 GHz, |S11| and |S21|.
  * Fig. 11: c/a = 0, 0.6, 0.631, 0.7, 1.0, 1.3, |S11| (dB) vs f/f_c = 1.02-1.98.

Ma's reference planes are not needed: only magnitudes are compared. Usage:
    python benchmarks/fem2d_ma1997.py [--out benchmarks/fem2d_ma1997.npz] [--density 10] [--order 2]
"""
import argparse
import datetime
import json
import subprocess

import numpy as np
import scipy.constants as cst

A, B = 15.799e-3, 7.899e-3
FC = cst.c / (2 * A)


def d_from_c(c_over_a):
    return (2.0 - c_over_a) / np.sqrt(2.0)


def run(c_over_a, freqs, options):
    from wrmodes.fem2d import Miter
    m = Miter(np.pi / 2, chamfer=d_from_c(c_over_a), guide=(A, B), **options)
    s11, s21 = [], []
    for f in freqs:
        g = m.gsm(f)
        s11.append(g.S11[0, 0])
        s21.append(g.S21[0, 0])
    return np.array(s11), np.array(s21)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--density", type=float, default=10.0)
    p.add_argument("--order", type=int, default=2)
    a = p.parse_args()
    options = dict(mesh_density=a.density, order=a.order)
    freqs = np.arange(9.6e9, 18.91e9, 0.1e9)
    results = {"freqs": freqs}
    for ca in (0.0, 1.0):
        s11, s21 = run(ca, freqs, options)
        results[f"S11_c{ca:g}"], results[f"S21_c{ca:g}"] = s11, s21
    fn = np.arange(1.02, 1.981, 0.02)
    results["f_over_fc"] = fn
    for ca in (0.0, 0.6, 0.631, 0.7, 1.0, 1.3):
        s11, s21 = run(ca, fn * FC, options)
        results[f"fig11_S11_c{ca:g}"], results[f"fig11_S21_c{ca:g}"] = s11, s21
    s11, s21 = run(1.0, [16e9], options)
    print(f"Table 1 (c/a = 1, 16 GHz): |S11| = {abs(s11[0]):.4f}, |S21| = {abs(s21[0]):.4f}")
    print("Fig. 5 (c/a = 0): f (GHz), |S21|^2")
    for f, s in zip(freqs[::5], results["S21_c0"][::5]):
        print(f"  {f / 1e9:6.2f}  {abs(s) ** 2:.4f}")
    print("Fig. 11: f/fc, |S11| (dB) for c/a = 0, 0.6, 0.631, 0.7, 1.0, 1.3")
    for i in range(0, len(fn), 5):
        vals = [20 * np.log10(abs(results[f'fig11_S11_c{ca:g}'][i])) for ca in (0.0, 0.6, 0.631, 0.7, 1.0, 1.3)]
        print(f"  {fn[i]:.2f}  " + "  ".join(f"{v:7.2f}" for v in vals))
    if a.out:
        import wrmodes
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            commit = "unknown"
        meta = dict(wrmodes_version=wrmodes.__version__, git_commit=commit, created=datetime.date.today().isoformat(),
                    generator="benchmarks/fem2d_ma1997.py", a=A, b=B, angle_deg=90,
                    reference="Ma, Yamane, Yamashita 1997, Electron. Comm. Japan Pt 2 80(5) 1-7", **options)
        np.savez(a.out, meta=json.dumps(meta), **results)


if __name__ == "__main__":
    main()
