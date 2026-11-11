"""FEM miters against Reiter & Arndt (1995), Figs. 3 and 4 (WP3b).

Fig. 3 (WR62 housing, a = 15.799 mm, b = 7.899 mm, |S11| dB vs GHz): (a) sharp H-plane bends, (b) mitered H-plane
bends, (c) sharp E-plane bends, (d) mitered E-plane bends. gamma is the interior angle between the inner walls, so
the deflection angle of `Miter` is 180 - gamma; a_opt and b_2opt are the distances from the inner corner to the
miter wall over the in-plane side, i.e. the chamfer d of `Miter` (L1 header). The branch of Fig. 3(d) that L1 could
not attribute ("right branch (gamma=105 or 135)") is compared with both and reported.

Fig. 4 (WR90, 22.86 x 10.16 mm): a mitered 90 degree H-plane bend (a_opt = 0.976 a) followed by a mitered 90 degree
E-plane bend (b_opt = 0.874 b), joined by a straight of length l = 0 or l = b, as a `wrmodes.Line` (the line inserts
the H -> E frame change). l is taken as the straight between the two miters' reference planes (CONVENTIONS.md
section 6; both miters cut their outer corner only, d > cos 45 deg, so their reference planes lie in uniform guide).
The cascade needs evanescent modes (CONVENTIONS.md section 3; l = 0 is a zero-length cascade): the convergence in the
line's cutoff_factor is shown at three frequencies, and the curves are computed with the converged setting, with
12 elements per wavelength (FIG4_DENSITY: in a zero-length cascade the evanescent entries must be accurate).
Dead families (no propagating mode in the local frame, e.g. global TE2,0 seen by the E-plane miter) are solved
rigorously by the FEM and do matter at l = 0. The
sense of the turns does not matter for |S11| (each bend is symmetric under the mirror that reverses the other).
Pass/fail: every curve within its uncertainty, outliers listed.

Usage: python benchmarks/fem2d/reiter1995.py [--processes 2] [--cutoff-factor 4]  -> data/reiter1995.npz
"""
import argparse
import numpy as np

from _lit import HEADER, compare_curves, case_meta, meta, save, pmap, load_csv, zstats, summary, row_md

A62, B62 = 15.799e-3, 7.899e-3
W90, H90 = 22.86e-3, 10.16e-3
GAMMAS = (90, 105, 120, 135, 150)
H_OPT = {105: 0.949, 120: 0.955, 135: 0.956}
E_OPT = {105: 0.873, 120: 0.898, 135: 0.918}
FIG4_DENSITY = 12      # the zero-length cascade amplifies the error of the evanescent entries: at the default 6
                       # elements per wavelength |S11| at 9 GHz jumps from -18.1 to -23.3 and -11.8 dB at
                       # cutoff_factor 4 and 5; at 12 it stays at -18.07 / -18.08 / -18.09 dB for cf = 2 ... 5


def ghz(x):
    return x * 1e9


def sharp(theta_deg):
    return 1.0 / np.cos(np.radians(theta_deg) / 2)


def cases():
    out = []
    for g in GAMMAS:
        out.append(dict(fig="Reiter 1995 Fig. 3(a)", file="reiter1995_fig3a.csv", curve=f"gamma={g}",
                        geom=dict(kind="miter", angle=180 - g, chamfer=sharp(180 - g), plane="H", w=A62, h=B62)))
    for g, d in H_OPT.items():
        out.append(dict(fig="Reiter 1995 Fig. 3(b)", file="reiter1995_fig3b.csv", curve=f"gamma={g}",
                        geom=dict(kind="miter", angle=180 - g, chamfer=d, plane="H", w=A62, h=B62)))
    for g in GAMMAS:
        out.append(dict(fig="Reiter 1995 Fig. 3(c)", file="reiter1995_fig3c.csv", curve=f"gamma={g}",
                        geom=dict(kind="miter", angle=180 - g, chamfer=sharp(180 - g), plane="E", w=A62, h=B62)))
    for g, d in E_OPT.items():
        out.append(dict(fig="Reiter 1995 Fig. 3(d)", file="reiter1995_fig3d.csv", curve=f"gamma={g}",
                        geom=dict(kind="miter", angle=180 - g, chamfer=d, plane="E", w=A62, h=B62)))
    for c in out:
        c.update(x_to_f=ghz, quantity="s11_db", status="pass/fail")
    for g in (105, 135):
        out.append(dict(fig=f"Reiter 1995 Fig. 3(d) (as gamma={g})", file="reiter1995_fig3d.csv",
                        curve="right branch (gamma=105 or 135)", x_to_f=ghz, quantity="s11_db", status="reported",
                        geom=dict(kind="miter", angle=180 - g, chamfer=E_OPT[g], plane="E", w=A62, h=B62)))
    return out


# ------------------------------------------------------------------------------------------- Fig. 4
def fig4_s11(args):
    """|S11| (dB) of global TE10 of the H-miter / straight l / E-miter line at f (Hz)."""
    f, ell, cf = args
    from wrmodes import Guide, Line, Straight
    from wrmodes.fem2d import Miter
    opts = dict(mesh_density=FIG4_DENSITY)
    comps = [Miter(90, chamfer=0.976, plane="H", **opts)]
    if ell > 0:
        comps.append(Straight(ell))
    comps.append(Miter(90, chamfer=0.874, plane="E", **opts))
    line = Line(Guide(W90, H90), comps)
    g = line.gsm(f, cutoff_factor=cf)
    i = g.basis1.index(("TE", 1, 0))
    return 20 * np.log10(abs(g.S11[i, i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--cutoff-factor", type=float, default=3.0, help="line truncation for the Fig. 4 curves")
    ap.add_argument("--skip-fig3", action="store_true")
    args = ap.parse_args()
    arrays, rows = {}, []
    cs = cases()
    if not args.skip_fig3:
        print(HEADER)
        rows, arrays = compare_curves(cs, args.processes)
        print("\nOutliers |z| > 2 of the pass/fail curves (figure, curve, GHz, digitized dB, ours dB, z):")
        for k, (c, s) in enumerate(rows):
            if c["status"] != "pass/fail":
                continue
            key = f"c{k:02d}"
            for x, y, o, z in zip(arrays[key + "_x"], arrays[key + "_lit"], arrays[key + "_ours"], arrays[key + "_z"]):
                if abs(z) > 2:
                    print("  %s | %s | %.3f | %.2f | %.2f | %+.1f" % (c["fig"], c["curve"], x, y, o, z))
    # Fig. 4: convergence in the cutoff factor
    cfs = [1.0, 2.0, 3.0, 4.0, 5.0]
    fs = [9.0e9, 10.5e9, 11.7e9]
    print("\nFig. 4, convergence of |S11| (dB) in the line's cutoff_factor:")
    print("| l | f (GHz) | " + " | ".join("cf %g" % c for c in cfs) + " |\n| --- | --- |" + " --- |" * len(cfs))
    conv = {}
    for ell, name in ((0.0, "l=0"), (H90, "l=b")):
        res = pmap(fig4_s11, [(f, ell, cf) for f in fs for cf in cfs], args.processes)
        res = np.array(res).reshape(len(fs), len(cfs))
        conv[name] = res
        for f, r in zip(fs, res):
            print("| %s | %.1f | " % (name, f / 1e9) + " | ".join("%.3f" % v for v in r) + " |")
    print("\nFig. 4 curves, cutoff_factor = %g:\n" % args.cutoff_factor + HEADER)
    lit = load_csv("reiter1995_fig4.csv")
    f4 = []
    for ell, name in ((0.0, "l=0"), (H90, "l=b")):
        for cur in [k for k in lit if k.startswith(name + " ") or k == name]:
            x, y, u, ux = lit[cur].T
            vals = np.array(pmap(fig4_s11, [(xi * 1e9, ell, args.cutoff_factor) for xi in x], args.processes))
            if np.any(ux > 0):
                vm = np.array(pmap(fig4_s11, [((xi - d) * 1e9, ell, args.cutoff_factor) for xi, d in zip(x, ux)],
                                   args.processes))
                vp = np.array(pmap(fig4_s11, [((xi + d) * 1e9, ell, args.cutoff_factor) for xi, d in zip(x, ux)],
                                   args.processes))
                # a minimum: the digitized row is the minimum value; compare with our minimum within x -+ ux
                if "minimum" in cur:
                    xs = np.linspace(x[0] - 3 * ux[0], x[0] + 3 * ux[0], 13)
                    vv = np.array(pmap(fig4_s11, [(xx * 1e9, ell, args.cutoff_factor) for xx in xs], args.processes))
                    vals = np.array([vv.min()])
                    print("  %s: digitized %.2f dB at %.3f GHz, ours min %.2f dB at %.3f GHz" % (
                        cur, y[0], x[0], vv.min(), xs[np.argmin(vv)]))
                z, ue = zstats(x, y, u, ux, vals, vm, vp)
            else:
                z, ue = zstats(x, y, u, ux, vals)
            s = summary(z, x, vals, y)
            st = "pass/fail" if "theory" in cur or cur == name else "reported (measurement)"
            if cur == name:
                st = "pass/fail (theory = measurement band)"
            print(row_md("Reiter 1995 Fig. 4", cur, s, st))
            f4.append(dict(curve=cur, status=st, **s))
            key = "fig4_" + cur.replace(" ", "_").replace("=", "")
            arrays.update({key + "_x": x, key + "_lit": y, key + "_ours": vals, key + "_z": z})
            for xi, yi, oi, zi in zip(x, y, vals, z):
                if abs(zi) > 2:
                    print("    outlier %s | %.3f | %.2f | %.2f | %+.1f" % (cur, xi, yi, oi, zi))
    arrays.update(fig4_conv_l0=conv["l=0"], fig4_conv_lb=conv["l=b"], fig4_conv_f=np.array(fs),
                  fig4_conv_cf=np.array(cfs))
    m = meta("benchmarks/fem2d/reiter1995.py", "FEM (wrmodes.fem2d.Miter, default FEMOptions) for Reiter & Arndt "
             "1995 Figs. 3-4; Fig. 4 as a wrmodes.Line with the frame change; z = (ours - lit) / u",
             curves=case_meta(cs), summary=[dict(curve=c["curve"], figure=c["fig"], **s) for c, s in rows],
             fig4=f4, fig4_cutoff_factor=args.cutoff_factor)
    print("saved", save("reiter1995.npz", m, **arrays))


if __name__ == "__main__":
    main()
