"""FEM miter against Kotik & Sivov (1965), Fig. 3 (WP3b).

TE10 power transmission |B1|^2 of the 90 degree H-plane optical-mirror miter (d/a = 1/sqrt(2), CONVENTIONS.md
section 6) against kd = 2 pi a / lambda = 10-31.4 (WR-90: 21-66 GHz), at the 416 digitized abscissae of
`benchmarks/literature/kotik1965_fig3_rigorous.csv` (L2), with z = (ours - digitized) / uncertainty. Also compares
the FEM with the asymptotic formula 1 - 7.4/(kd)^(3/2) (Katsenelenbaum, as quoted by Kotik & Sivov; the printed
exponent 2/3 is a misprint, docs/reports/L2.md) for kd > 6 pi, where the authors state that it is within 3 %
(relative) of the rigorous solution. Pass/fail reference (card WP3b).

Usage: python benchmarks/fem2d/kotik1965.py [--processes 2]   -> data/kotik1965.npz
"""
import argparse
import numpy as np
import scipy.constants as cst

from _lit import (SQ2, HEADER, compare_curves, outliers, case_meta, meta, save, pmap, te10)

W, H = 22.86e-3, 10.16e-3


def kd_to_f(kd):
    return kd * cst.c / (2 * np.pi * W)


def cases():
    return [dict(fig="Kotik 1965 Fig. 3", file="kotik1965_fig3_rigorous.csv", curve="rigorous",
                 geom=dict(kind="miter", angle=90, chamfer=1 / SQ2, plane="H", w=W, h=H), x_to_f=kd_to_f,
                 quantity="s21_pow", status="pass/fail")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=None)
    args = ap.parse_args()
    cs = cases()
    print(HEADER)
    rows, arrays = compare_curves(cs, args.processes)
    print("\nOutliers |z| > 2 (figure, curve, kd, digitized, ours, z):")
    for o in outliers(rows, arrays):
        print("  %s | %s | %.3f | %.4f | %.4f | %+.1f" % o)
    # asymptotic formula above 6 pi
    kd = np.linspace(6 * np.pi + 0.01, 32.0, 120)
    res = pmap(te10, [dict(cs[0]["geom"], f=kd_to_f(x)) for x in kd], args.processes)
    t = np.array([abs(r[1]) ** 2 for r in res])
    asym = 1 - 7.4 / kd ** 1.5
    rel = (asym - t) / t
    print("\nAsymptotic 1 - 7.4/(kd)^1.5 vs FEM, kd = 6 pi .. 32: max |rel. diff| = %.4f (at kd = %.2f), "
          "mean %.4f; authors: within 0.03" % (np.max(np.abs(rel)), kd[np.argmax(np.abs(rel))], np.mean(rel)))
    away = np.ones(len(kd), bool)
    for nn in range(6, 11):
        away &= ~((kd > nn * np.pi - 0.1) & (kd < nn * np.pi + 0.6))
    print("  away from the cutoff dips (kd outside [n pi - 0.1, n pi + 0.6]): max |rel. diff| = %.4f" %
          np.max(np.abs(rel[away])))
    arrays.update(asym_kd=kd, asym_fem=t, asym_formula=asym)
    m = meta("benchmarks/fem2d/kotik1965.py", "FEM (wrmodes.fem2d.Miter, default FEMOptions) at the digitized "
             "abscissae of Kotik & Sivov 1965 Fig. 3; z = (ours - lit) / u", guide_w=W, guide_h=H,
             curves=case_meta(cs), units="|S21(TE10)|^2 vs kd = 2 pi a / lambda",
             summary=[dict(curve=c["curve"], **s) for c, s in rows],
             asymptotic=dict(max_abs_rel=float(np.max(np.abs(rel))), mean_rel=float(np.mean(rel)),
                             max_abs_rel_away_from_cutoffs=float(np.max(np.abs(rel[away])))))
    print("saved", save("kotik1965.npz", m, **arrays))


if __name__ == "__main__":
    main()
