"""FEM miters against Ma, Yamane & Yamashita (1997), Figs. 5, 6, 8, 9, 11 and Table 1 (WP3b).

Guide a = 15.799 mm, b = 7.899 mm (p. 4), 90 degree H-plane bends, fc = 9.488 GHz. Ma's c is the cut along each
outer wall from the removed outer corner: d/a = (2 - c/a)/sqrt(2) (CONVENTIONS.md section 6); c/a = 0 is the sharp
corner, c/a = 1 the plain (optical-mirror) miter. Ma model the slanted wall by a staircase (Fig. 3) with three
division methods (Fig. 7: (a) steps through the slanted plane, (b) below it, on the guide side, (c) above it);
"the calculations below are all performed by using division method (a) with 60 division steps" (p. 5). This script
computes both the exact mirror (`Miter`) and the staircases Ma used (`_lit.staircase_miter_outline` in a
`Coplanar2D`), so that Figs. 6, 8, 9 are compared in the published geometry:

* Fig. 5: sharp corner, |S21|^2 (12-term line and BEM circles): pass/fail;
* Fig. 6: |S21| dB, c/a = 1: 60-step (a) staircase for "40/60 steps", 20 and 10 steps (a); BEM circles against the
  exact mirror: pass/fail;
* Fig. 8: |S11| dB, 20 steps, methods (a), (b), (c): pass/fail;
* Fig. 9: |S11| dB, 60 steps (a): pass/fail;
* Fig. 11: |S11| dB vs f/fc: c/a = 0 (sharp) and c/a = 1.0 (60-step (a) staircase) pass/fail; the chamfered curves
  (c/a = 0.60, 0.631, 0.70, 1.30; exact mirrors) qualitative (Daniel, 11 Nov 2026);
* Table 1: |S11|, |S21| at 16 GHz, exact mirror and 50/60-step (a) staircases: qualitative.

Usage: python benchmarks/fem2d/ma1997.py [--processes 2]  -> data/ma1997.npz
"""
import argparse
import numpy as np
import scipy.constants as cst

from _lit import SQ2, HEADER, compare_curves, case_meta, meta, save, pmap, te10

A, B = 15.799e-3, 7.899e-3
FC = cst.c / (2 * A)


def d_of_c(c):
    return (2.0 - c) / SQ2


def mit(c):
    return dict(kind="miter", angle=90, chamfer=d_of_c(c), plane="H", w=A, h=B)


def stair(n, method="a"):
    return dict(kind="stair", n_steps=n, method=method, plane="H", w=A, h=B)


def cases():
    ghz = lambda x: x * 1e9
    ffc = lambda x: x * FC
    P = "pass/fail"
    out = [dict(fig="Ma 1997 Fig. 5", file="ma1997_fig5.csv", curve="3/6/12 terms", geom=mit(0), x_to_f=ghz,
                quantity="s21_pow", status=P),
           dict(fig="Ma 1997 Fig. 5", file="ma1997_fig5.csv", curve="BEM", geom=mit(0), x_to_f=ghz,
                quantity="s21_pow", status=P),
           dict(fig="Ma 1997 Fig. 6 (60 steps a)", file="ma1997_fig6.csv", curve="40/60 steps", geom=stair(60),
                x_to_f=ghz, quantity="s21_db", status=P),
           dict(fig="Ma 1997 Fig. 6 (20 steps a)", file="ma1997_fig6.csv", curve="20 steps", geom=stair(20),
                x_to_f=ghz, quantity="s21_db", status=P),
           dict(fig="Ma 1997 Fig. 6 (10 steps a)", file="ma1997_fig6.csv", curve="10 steps", geom=stair(10),
                x_to_f=ghz, quantity="s21_db", status=P),
           dict(fig="Ma 1997 Fig. 6 (exact mirror)", file="ma1997_fig6.csv", curve="BEM", geom=mit(1.0),
                x_to_f=ghz, quantity="s21_db", status=P)]
    for meth in "abc":
        out.append(dict(fig=f"Ma 1997 Fig. 8 (20 steps {meth})", file="ma1997_fig8.csv", curve=f"({meth})",
                        geom=stair(20, meth), x_to_f=ghz, quantity="s11_db", status=P))
    out.append(dict(fig="Ma 1997 Fig. 9 (60 steps a)", file="ma1997_fig9.csv", curve="(a)=(b)=(c)", geom=stair(60),
                    x_to_f=ghz, quantity="s11_db", status=P))
    out.append(dict(fig="Ma 1997 Fig. 9 (exact mirror)", file="ma1997_fig9.csv", curve="(a)=(b)=(c)", geom=mit(1.0),
                    x_to_f=ghz, quantity="s11_db", status="reported"))
    out += [dict(fig="Ma 1997 Fig. 11", file="ma1997_fig11.csv", curve="c/a=0", geom=mit(0), x_to_f=ffc,
                 quantity="s11_db", status=P),
            dict(fig="Ma 1997 Fig. 11 (60 steps a)", file="ma1997_fig11.csv", curve="c/a=1.0", geom=stair(60),
                 x_to_f=ffc, quantity="s11_db", status=P),
            dict(fig="Ma 1997 Fig. 11 (exact mirror)", file="ma1997_fig11.csv", curve="c/a=1.0", geom=mit(1.0),
                 x_to_f=ffc, quantity="s11_db", status="reported")]
    for c in (0.60, 0.631, 0.70, 1.30):
        out.append(dict(fig="Ma 1997 Fig. 11", file="ma1997_fig11.csv", curve="c/a=%g" % c if c != 0.6 else "c/a=0.60",
                        geom=mit(c), x_to_f=ffc, quantity="s11_db", status="qualitative"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=None)
    args = ap.parse_args()
    cs = cases()
    for c in cs:            # curve names as in the file
        if c["curve"] == "c/a=1.3":
            c["curve"] = "c/a=1.30"
        if c["curve"] == "c/a=0.7":
            c["curve"] = "c/a=0.70"
    print(HEADER)
    rows, arrays = compare_curves(cs, args.processes)
    print("\nOutliers |z| > 2 of the pass/fail curves (figure, curve, x, digitized, ours, z):")
    for k, (c, s) in enumerate(rows):
        if c["status"] != "pass/fail":
            continue
        key = f"c{k:02d}"
        for x, y, o, z in zip(arrays[key + "_x"], arrays[key + "_lit"], arrays[key + "_ours"], arrays[key + "_z"]):
            if abs(z) > 2:
                print("  %s | %s | %.4f | %.3f | %.3f | %+.1f" % (c["fig"], c["curve"], x, y, o, z))
    # Table 1
    tab = {}
    print("\nTable 1 (16 GHz, qualitative): geometry | |S11| | |S21| | Ma Ref. |S11| | |S21|")
    refs = {50: (0.2641, 0.9645), 60: (0.2622, 0.9650)}
    specs = [("exact mirror", mit(1.0), None)] + [(f"{n} steps (a)", stair(n), n) for n in (50, 60)] + \
        [(f"{n} steps ({m})", stair(n, m), None) for n in (60,) for m in "bc"]
    res = pmap(te10, [dict(s, f=16e9) for _, s, _ in specs], args.processes)
    for (name, s, n), r in zip(specs, res):
        ref = refs.get(n, (np.nan, np.nan))
        print("  %s | %.4f | %.4f | %.4f | %.4f" % (name, abs(r[0]), abs(r[1]), *ref))
        tab[name] = (abs(r[0]), abs(r[1]))
    # mesh check of the staircase at 16 GHz
    chk = pmap(te10, [dict(stair(60), f=16e9, options=dict(mesh_density=d)) for d in (6, 10, 14)], args.processes)
    print("  60-step (a) staircase, mesh_density 6 / 10 / 14: |S11| = " + " / ".join("%.5f" % abs(r[0]) for r in chk))
    arrays.update(table1_names=np.array(list(tab)), table1_values=np.array(list(tab.values())),
                  stair_mesh_check=np.array([abs(r[0]) for r in chk]))
    m = meta("benchmarks/fem2d/ma1997.py", "FEM (wrmodes.fem2d, default FEMOptions) for Ma et al. 1997: exact miters "
             "and Ma's staircase miters at the digitized abscissae; z = (ours - lit) / u", a=A, b=B,
             curves=case_meta(cs), summary=[dict(curve=c["curve"], figure=c["fig"], **s) for c, s in rows],
             table1={k: list(v) for k, v in tab.items()})
    print("saved", save("ma1997.npz", m, **arrays))


if __name__ == "__main__":
    main()
