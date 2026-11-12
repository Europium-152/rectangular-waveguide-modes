"""FEM miter against Alessandri, Mongiardo & Sorrentino (1994), Figs. 4 and 5 (WP3b).

|S11| (dB) of TE10 of 90 degree E-plane miters (the narrow side b in the bend plane) against f/fc, fc the TE10
cutoff, at the digitized abscissae of `benchmarks/literature/alessandri1994_fig4.csv` and `_fig5.csv` (L1).
Alessandri's d/b is the distance from the inner corner to the miter wall over the in-plane side, i.e. the chamfer
of `wrmodes.fem2d.Miter` (CONVENTIONS.md section 6): d/b = 1.41 is the sharp corner (read as sqrt(2)),
0.707 the optical mirror (1/sqrt(2)). The paper does not give the broad side: both b/a = 0.5 (WP4b's finding, the
pass/fail case) and WR-90's 0.444 are computed (card WP3b; WP7 open issue). The crosses of their ref. [7] (Reiter &
Arndt's BCMM) are other authors' results and are reported only.

Usage: python benchmarks/fem2d/alessandri1994.py [--b-over-a 0.5] [--processes 2]  -> data/alessandri1994_b<r>.npz
"""
import argparse
import scipy.constants as cst

from _lit import SQ2, HEADER, compare_curves, outliers, case_meta, meta, save

W = 22.86e-3


def cases(b_over_a, status):
    geom = lambda d: dict(kind="miter", angle=90, chamfer=d, plane="E", w=W, h=b_over_a * W)
    x_to_f = lambda x: x * cst.c / (2 * W)
    out = []
    for fn, cur, d in [("alessandri1994_fig4.csv", "d/b=1.41", SQ2), ("alessandri1994_fig4.csv", "d/b=0.707", 1 / SQ2),
                       ("alessandri1994_fig4.csv", "d/b=0.87", 0.87), ("alessandri1994_fig5.csv", "d/b=0.9", 0.9),
                       ("alessandri1994_fig5.csv", "d/b=0.87", 0.87), ("alessandri1994_fig5.csv", "d/b=0.85", 0.85),
                       ("alessandri1994_fig5.csv", "d/b=0.83", 0.83),
                       ("alessandri1994_fig4.csv", "ref7 d/b=1.41", SQ2),
                       ("alessandri1994_fig4.csv", "ref7 d/b=0.707", 1 / SQ2),
                       ("alessandri1994_fig4.csv", "ref7 d/b=0.87", 0.87)]:
        fig = "Alessandri 1994 Fig. " + fn[-5] + (" (ref. [7])" if cur.startswith("ref7") else "")
        st = "reported (ref. [7])" if cur.startswith("ref7") else status
        out.append(dict(fig=fig, file=fn, curve=cur, geom=geom(d), x_to_f=x_to_f, quantity="s11_db", status=st))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-over-a", type=float, default=0.5)
    ap.add_argument("--processes", type=int, default=None)
    args = ap.parse_args()
    status = "pass/fail" if abs(args.b_over_a - 0.5) < 1e-12 else "reported (b/a test)"
    cs = cases(args.b_over_a, status)
    print(f"b/a = {args.b_over_a}\n" + HEADER)
    rows, arrays = compare_curves(cs, args.processes)
    print("\nOutliers |z| > 2 (figure, curve, f/fc, digitized dB, ours dB, z):")
    for o in outliers(rows, arrays):
        print("  %s | %s | %.4f | %.2f | %.2f | %+.1f" % o)
    m = meta("benchmarks/fem2d/alessandri1994.py", "FEM (wrmodes.fem2d.Miter, plane 'E', default FEMOptions) at the "
             "digitized abscissae of Alessandri et al. 1994 Figs. 4-5; z = (ours - lit) / u", b_over_a=args.b_over_a,
             guide_w=W, curves=case_meta(cs), units="|S11(TE10)| in dB vs f/fc",
             summary=[dict(curve=c["curve"], figure=c["fig"], **s) for c, s in rows])
    print("saved", save("alessandri1994_b%.3f.npz" % args.b_over_a, m, **arrays))


if __name__ == "__main__":
    main()
