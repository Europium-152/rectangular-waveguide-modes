"""FEM miter against Alimenti, Mongiardo & Sorrentino (1994), Figs. 2-6 (WP3b).

|S11| (dB) of TE10 of 90 degree H-plane bends against f/fc (fc the TE10 cutoff), at the digitized abscissae of the
L2 files `benchmarks/literature/alimenti1994_fig*.csv`. Geometry (L2 report, CONVENTIONS.md section 6):
d/a = (1 + t/a)/sqrt(2); Fig. 2 is the unmitered corner (d/a = sqrt(2)), Fig. 3 the fully mitered bend (t/a = 0,
the optical mirror). Figs. 2-3 are pass/fail references; Figs. 4-6 (chamfered) and the design chart of Fig. 5 are
qualitative (Daniel, 11 Nov 2026; docs/discrepancies/2026-11-11-alimenti-chamfered-miters.md): the comparison is
reported.

Fig. 5: for t/a = 0.300 ... 0.400 the matching (reflection-zero) frequencies and the 30 dB bandwidth / fc (width
of the band around the zero(s) where |S11| <= -30 dB) are computed from a scan of f/fc in steps of 0.0025 with
parabolic refinement of the minima of |S11|^2.

Usage: python benchmarks/fem2d/alimenti1994.py [--processes 2] [--skip-chart]  -> data/alimenti1994.npz
"""
import argparse
import numpy as np
import scipy.constants as cst

from _lit import SQ2, HEADER, compare_curves, outliers, case_meta, meta, save, pmap, te10, load_csv

W, H = 22.86e-3, 10.16e-3


def d_of_t(t):
    return (1 + t) / SQ2


def x_to_f(x):
    return x * cst.c / (2 * W)


def geom(d):
    return dict(kind="miter", angle=90, chamfer=d, plane="H", w=W, h=H)


def cases():
    out = [dict(fig="Alimenti 1994 Fig. 2 (unmitered)", file="alimenti1994_fig2_mmfd.csv", curve="MMFD",
                geom=geom(SQ2)),
           dict(fig="Alimenti 1994 Fig. 2 (unmitered)", file="alimenti1994_fig2_mm.csv", curve="MM markers",
                geom=geom(SQ2)),
           dict(fig="Alimenti 1994 Fig. 3 (optical mirror)", file="alimenti1994_fig3_mm.csv", curve="MM",
                geom=geom(1 / SQ2)),
           dict(fig="Alimenti 1994 Fig. 3 (optical mirror)", file="alimenti1994_fig3_mmfd.csv", curve="MMFD markers",
                geom=geom(1 / SQ2))]
    for c in out:
        c["status"] = "pass/fail"
    for t, fn in ((0.30, "alimenti1994_fig4_t030.csv"), (0.34, "alimenti1994_fig4_t034.csv"),
                  (0.38, "alimenti1994_fig4_t038.csv")):
        out.append(dict(fig="Alimenti 1994 Fig. 4", file=fn, curve="t/a=%.2f" % t, geom=geom(d_of_t(t)),
                        status="qualitative"))
    out += [dict(fig="Alimenti 1994 Fig. 6", file="alimenti1994_fig6_mmfd.csv", curve="MMFD", geom=geom(d_of_t(0.386)),
                 status="qualitative"),
            dict(fig="Alimenti 1994 Fig. 6", file="alimenti1994_fig6_bcmm.csv", curve="BCMM", geom=geom(d_of_t(0.386)),
                 status="qualitative"),
            dict(fig="Alimenti 1994 Fig. 6", file="alimenti1994_fig6_bcmm.csv", curve="BCMM markers",
                 geom=geom(d_of_t(0.386)), status="qualitative")]
    for c in out:
        c.update(x_to_f=x_to_f, quantity="s11_db")
    return out


def _parabola_min(x, y):
    """Vertex of the parabola through three points."""
    (x0, x1, x2), (y0, y1, y2) = x, y
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
    return -B / (2 * A)


def design_chart(ts, processes):
    xs = np.arange(1.05, 1.9951, 0.0025)
    jobs = [dict(geom(d_of_t(t)), f=x_to_f(x)) for t in ts for x in xs]
    res = pmap(te10, jobs, processes)
    s11 = np.array([abs(r[0]) for r in res]).reshape(len(ts), len(xs))
    zeros, bands = [], []
    for i, t in enumerate(ts):
        p = s11[i] ** 2
        mins = [j for j in range(1, len(xs) - 1) if p[j] <= p[j - 1] and p[j] <= p[j + 1]]
        zi = [(_parabola_min(xs[j - 1:j + 2], p[j - 1:j + 2]), j) for j in mins
              if 20 * np.log10(s11[i, j]) < -30]
        zeros.append([z for z, _ in zi])
        db = 20 * np.log10(s11[i])
        below = db <= -30
        if not zi:
            bands.append(0.0)
            continue
        j0 = zi[0][1]
        lo = j0
        while lo > 0 and below[lo - 1]:
            lo -= 1
        hi = j0
        while hi < len(xs) - 1 and below[hi + 1]:
            hi += 1
        # linear interpolation of the -30 dB crossings in dB
        xl = xs[lo] if lo == 0 else xs[lo - 1] + (xs[lo] - xs[lo - 1]) * (-30 - db[lo - 1]) / (db[lo] - db[lo - 1])
        xh = xs[hi] if hi == len(xs) - 1 else xs[hi] + (xs[hi + 1] - xs[hi]) * (-30 - db[hi]) / (db[hi + 1] - db[hi])
        bands.append(xh - xl)
    return xs, s11, zeros, np.array(bands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--skip-chart", action="store_true")
    args = ap.parse_args()
    cs = cases()
    print(HEADER)
    rows, arrays = compare_curves(cs, args.processes)
    print("\nOutliers |z| > 2 of the pass/fail curves (figure, curve, f/fc, digitized dB, ours dB, z):")
    for o in outliers(rows[:4], arrays):
        print("  %s | %s | %.4f | %.2f | %.2f | %+.1f" % o)
    extra = {}
    if not args.skip_chart:
        ts = np.round(np.arange(0.300, 0.40001, 0.005), 4)
        ts = np.sort(np.append(ts, [0.386]))
        xs, s11, zeros, bands = design_chart(ts, args.processes)
        mf = load_csv("alimenti1994_fig5_matching_frequency.csv")["matching frequency"]
        bw = load_csv("alimenti1994_fig5_bandwidth.csv")
        print("\nFig. 5 (qualitative): t/a | our zeros f/fc | Alimenti matching f/fc | our 30 dB band/fc | Alimenti band/fc")
        allbw = np.vstack([bw[k] for k in bw])
        for t, z, b in zip(ts, zeros, bands):
            m_ref = np.interp(t, mf[:, 0], mf[:, 1])
            j = np.argmin(np.abs(allbw[:, 0] - t))
            print("  %.3f | %s | %.3f | %.3f | %.3f" % (t, ", ".join("%.4f" % v for v in z) or "-", m_ref, b,
                                                        allbw[j, 1]))
        zz = np.full((len(ts), 2), np.nan)
        for i, z in enumerate(zeros):
            zz[i, :min(2, len(z))] = z[:2]
        extra = dict(chart_t=ts, chart_x=xs, chart_s11=s11, chart_zeros=zz, chart_band=bands)
    arrays.update(extra)
    m = meta("benchmarks/fem2d/alimenti1994.py", "FEM (wrmodes.fem2d.Miter, default FEMOptions) at the digitized "
             "abscissae of Alimenti et al. 1994 Figs. 2-6; z = (ours - lit) / u; design chart of Fig. 5",
             guide_w=W, guide_h=H, curves=case_meta(cs), units="|S11(TE10)| in dB vs f/fc",
             geometry="d/a = (1 + t/a)/sqrt(2) (docs/reports/L2.md)",
             summary=[dict(curve=c["curve"], figure=c["fig"], **s) for c, s in rows])
    print("saved", save("alimenti1994.npz", m, **arrays))


if __name__ == "__main__":
    main()
