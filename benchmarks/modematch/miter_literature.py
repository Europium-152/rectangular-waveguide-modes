"""MiterMM against the digitized literature (WP4b): Alessandri et al. 1994 Figs. 4-5 (E plane), Alimenti et al. 1994
Figs. 2-6 and Kotik & Sivov 1965 Fig. 3 (H plane), Ma et al. 1997 Fig. 11 (H plane, for the discussion of the
chamfered cases).

Every value is computed with `wrmodes.modematch.MiterMM` and its default truncation, in a guide of the stated
aspect ratio (E plane: TE10 is the local TE0,1; H plane: TE1,0), at the digitized abscissae. For each curve the table
gives z = (ours - digitized) / uncertainty. Writes data/miter_literature.npz and prints the table (markdown).

Geometry conversions (CONVENTIONS.md section 6, d = distance of the miter from the inner corner / in-plane side):
Alessandri d/b as printed (1.41 and 0.707 read as sqrt(2) and 1/sqrt(2)); Alimenti d/a = (1 + t/a)/sqrt(2)
(docs/reports/L2.md); Ma d/a = (2 - c/a)/sqrt(2) (c/a = 1.30 is a mirror cutting into the arms: not supported, left
out); Kotik d/a = 1/sqrt(2), kd = 2 pi a / lambda.
Usage: python benchmarks/modematch/miter_literature.py [--b-over-a 0.5]
"""
import argparse
import os
import sys

import numpy as np
import scipy.constants as cst

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from wrmodes import Guide                                         # noqa: E402
from wrmodes.modematch import MiterMM                             # noqa: E402
from _meta import meta, save                                      # noqa: E402

LIT = os.path.join(os.path.dirname(__file__), "..", "literature")
SQ2 = np.sqrt(2.0)


def load_csv(name):
    rows = [r for r in open(os.path.join(LIT, name)) if not r.startswith("#")][1:]
    out = {}
    for r in rows:
        v = r.strip().split(",")
        out.setdefault(v[0], []).append([float(v[1]), float(v[2]), float(v[3]), float(v[4])])
    return {k: np.array(v) for k, v in out.items()}


def eplane_s11_db(d, x, b_over_a):
    """|S11| (dB) of TE10 of an E-plane miter, f/fc = x, guide w = 1 cm, h = b_over_a w."""
    w = 0.01
    g = Guide(w, b_over_a * w)
    f = x * cst.c / (2 * w)
    G = MiterMM(d, "E", g).gsm(f)
    i = G.basis1.index("TE0,1")
    return 20 * np.log10(abs(G.S11[i, i]))


def hplane(d, x, quantity="s11_db"):
    """TE10 of an H-plane miter, f/fc = x (guide w = 1 cm, h = w/2; below the TE01 cutoff for x < 2)."""
    w = 0.01
    g = Guide(w, w / 2)
    f = x * cst.c / (2 * w)
    G = MiterMM(d, "H", g).gsm(f)
    i = G.basis1.index("TE1,0")
    if quantity == "s11_db":
        return 20 * np.log10(abs(G.S11[i, i]))
    return abs(G.S21[i, i]) ** 2


def cases(b_over_a):
    E = lambda d: (lambda x: eplane_s11_db(d, x, b_over_a))
    H = lambda d: (lambda x: hplane(d, x))
    al = lambda t: (1 + t) / SQ2
    out = [("Alessandri 1994 Fig. 4", "alessandri1994_fig4.csv", "d/b=1.41", E(SQ2)),
           ("Alessandri 1994 Fig. 4", "alessandri1994_fig4.csv", "d/b=0.707", E(1 / SQ2)),
           ("Alessandri 1994 Fig. 4", "alessandri1994_fig4.csv", "d/b=0.87", E(0.87)),
           ("Alessandri 1994 Fig. 4 (ref. [7])", "alessandri1994_fig4.csv", "ref7 d/b=1.41", E(SQ2)),
           ("Alessandri 1994 Fig. 4 (ref. [7])", "alessandri1994_fig4.csv", "ref7 d/b=0.707", E(1 / SQ2)),
           ("Alessandri 1994 Fig. 4 (ref. [7])", "alessandri1994_fig4.csv", "ref7 d/b=0.87", E(0.87))]
    for d in ("0.9", "0.87", "0.85", "0.83"):
        out.append(("Alessandri 1994 Fig. 5", "alessandri1994_fig5.csv", "d/b=" + d, E(float(d))))
    out += [("Alimenti 1994 Fig. 2 (unmitered)", "alimenti1994_fig2_mmfd.csv", "MMFD", H(SQ2)),
            ("Alimenti 1994 Fig. 2 (unmitered)", "alimenti1994_fig2_mm.csv", "MM markers", H(SQ2)),
            ("Alimenti 1994 Fig. 3 (optical mirror)", "alimenti1994_fig3_mm.csv", "MM", H(1 / SQ2)),
            ("Alimenti 1994 Fig. 3 (optical mirror)", "alimenti1994_fig3_mmfd.csv", "MMFD markers", H(1 / SQ2)),
            ("Alimenti 1994 Fig. 4", "alimenti1994_fig4_t030.csv", "t/a=0.30", H(al(0.30))),
            ("Alimenti 1994 Fig. 4", "alimenti1994_fig4_t034.csv", "t/a=0.34", H(al(0.34))),
            ("Alimenti 1994 Fig. 4", "alimenti1994_fig4_t038.csv", "t/a=0.38", H(al(0.38))),
            ("Alimenti 1994 Fig. 6", "alimenti1994_fig6_mmfd.csv", "MMFD", H(al(0.386))),
            ("Alimenti 1994 Fig. 6", "alimenti1994_fig6_bcmm.csv", "BCMM", H(al(0.386))),
            ("Alimenti 1994 Fig. 6", "alimenti1994_fig6_bcmm.csv", "BCMM markers", H(al(0.386))),
            ("Kotik 1965 Fig. 3", "kotik1965_fig3_rigorous.csv", "rigorous",
             lambda kd: hplane(1 / SQ2, kd / np.pi, "t10")),
            ("Ma 1997 Fig. 11", "ma1997_fig11.csv", "c/a=0", H(SQ2)),
            ("Ma 1997 Fig. 11", "ma1997_fig11.csv", "c/a=0.60", H((2 - 0.60) / SQ2)),
            ("Ma 1997 Fig. 11", "ma1997_fig11.csv", "c/a=0.631", H((2 - 0.631) / SQ2)),
            ("Ma 1997 Fig. 11", "ma1997_fig11.csv", "c/a=0.70", H((2 - 0.70) / SQ2)),
            ("Ma 1997 Fig. 11", "ma1997_fig11.csv", "c/a=1.0", H(1 / SQ2))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-over-a", type=float, default=0.5, help="Alessandri's guide (narrow / broad); 0.5 or 0.444")
    ap.add_argument("--out", default="miter_literature.npz")
    args = ap.parse_args()
    arrays, rows, outliers = {}, [], []
    print("| Figure | curve | n | rms z | |z|<=1 | |z|<=2 | max |z| (at x) | mean diff |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for k, (fig, fn, cur, model) in enumerate(cases(args.b_over_a)):
        arr = load_csv(fn)[cur]
        x, y, u, ux = arr.T
        ours = np.array([model(xi) for xi in x])
        if np.any(ux > 0):
            # markers: x reading error converted with the local slope of our curve (central difference)
            h = np.maximum(ux, 1e-4)
            slope = (np.array([model(xi + hi) for xi, hi in zip(x, h)])
                     - np.array([model(xi - hi) for xi, hi in zip(x, h)])) / (2 * h)
            u = np.sqrt(u ** 2 + (slope * ux) ** 2)
        z = (ours - y) / u
        for xi, yi, oi, zi in zip(x, y, ours, z):
            if abs(zi) > 2:
                outliers.append((fig, cur, xi, yi, oi, zi))
        i = int(np.argmax(np.abs(z)))
        rows.append((fig, cur, len(x), np.sqrt(np.mean(z ** 2)), np.mean(np.abs(z) <= 1), np.mean(np.abs(z) <= 2),
                     np.abs(z[i]), x[i], np.mean(ours - y)))
        print("| %s | %s | %d | %.2f | %.0f %% | %.0f %% | %.1f (%.3f) | %+.3f |" % (
            fig, cur, len(x), rows[-1][3], 100 * rows[-1][4], 100 * rows[-1][5], rows[-1][6], rows[-1][7], rows[-1][8]),
            flush=True)
        key = f"c{k:02d}"
        arrays[key + "_x"], arrays[key + "_lit"], arrays[key + "_u"], arrays[key + "_ours"] = x, y, u, ours
    print("\nOutliers |z| > 2 (figure, curve, x, digitized, ours, z):")
    for o in outliers:
        print("  %s | %s | %.4f | %.3f | %.3f | %+.1f" % o)
    m = meta("benchmarks/modematch/miter_literature.py",
             "MiterMM (default truncation) at the digitized abscissae of the miter figures; z = (ours - lit) / u",
             b_over_a_alessandri=args.b_over_a,
             curves=[{"key": f"c{k:02d}", "figure": f, "file": fn, "curve": c} for k, (f, fn, c, _) in
                     enumerate(cases(args.b_over_a))],
             units="Alessandri, Alimenti, Ma: |S11| in dB vs f/fc; Kotik: |S21(TE10)|^2 vs kd",
             uncertainty="u = the file's uncertainty; for markers sqrt(u^2 + (slope x_uncertainty)^2), slope of our curve",
             geometry="see the module docstring")
    save(args.out, m, **arrays)


if __name__ == "__main__":
    main()
