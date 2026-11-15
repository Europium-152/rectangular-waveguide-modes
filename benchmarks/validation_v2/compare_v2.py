"""WP7b comparisons: the wave 2 components against the WP7b references of this folder (run after the references
were committed). Writes markdown tables (stdout, and `--out`), and `data/compare_<part>.json` with the numbers.

    PYTHONPATH=src:benchmarks/validation_v2 python benchmarks/validation_v2/compare_v2.py [tapers staircase twists
        miters filter literature] [--out FILE]

Tolerances (CONVENTIONS.md section 8, two independent methods): 0.01 dB on the main transmission; 1e-4 of the input
power for every converted mode (here: every propagating output mode other than the incident one, transmitted and
reflected; for miters every propagating entry of the whole GSM, as WP4b's card states).
"""
import argparse
import json
import os

import numpy as np

import vcommon as vc

DB_TOL, P_TOL = 0.01, 1e-4


def _blocks(g):
    return np.block([[g.S11, g.S12], [g.S21, g.S22]])


def entry_compare(ref, test, inc):
    """Compare column `inc` (incident mode name at port 1): main transmission in dB and the worst power difference
    over the other output modes (both ports). `ref` and `test` expose S11, S21 and names1, names2."""
    i_r, i_t = ref["names1"].index(inc), test["names1"].index(inc)
    main_r = ref["S21"][ref["names2"].index(inc), i_r] if inc in ref["names2"] else None
    main_t = test["S21"][test["names2"].index(inc), i_t] if inc in test["names2"] else None
    dmax, worst = 0.0, None
    for blk, nr, nt in (("S21", ref["names2"], test["names2"]), ("S11", ref["names1"], test["names1"])):
        for j, n in enumerate(nr):
            if n not in nt or (blk == "S21" and n == inc):
                continue
            d = abs(abs(test[blk][nt.index(n), i_t]) ** 2 - abs(ref[blk][j, i_r]) ** 2)
            if d > dmax:
                dmax, worst = d, f"{blk[1]}:{n}"
    ddb = None if main_r is None else float(vc.db(main_t) - vc.db(main_r))
    return ddb, float(dmax), worst


def as_dict(g):
    return dict(S11=np.asarray(g.S11), S21=np.asarray(g.S21), S12=np.asarray(g.S12), S22=np.asarray(g.S22),
                names1=list(g.basis1.names), names2=list(g.basis2.names))


def stored(d, key):
    s = vc.StoredGSM(d, key)
    return dict(S11=s.S11, S21=s.S21, S12=s.S12, S22=s.S22, names1=s.names1, names2=s.names2)


# --------------------------------------------------------------------------------------------- tapers (FEM)
def tapers(out, res):
    import wrmodes as wr
    from wrmodes.components import Guide
    out.append("\n### WP5 one-family tapers and the strangled bend vs the 2D FEM (WP7b outlines)\n")
    out.append("| case | f (GHz) | FEM main (dB) | FEM error est. (\\|ΔS\\|, mesh / walls) | WP5 Δ main (dB) | WP5 worst ΔP (mode) | "
               "pass | WP5 mode_factor 3: Δ main / worst ΔP | WP5 time (s) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    W, H = vc.W90, vc.H90
    for case in ("H_linear_L100", "H_linear_L200", "E_rc50", "strangled"):
        d = vc.load_npz(f"taper_fem_{case}.npz")
        runs = d["meta"]["runs"]
        for f in sorted(set(r["f"] for r in runs)):
            rr = [r for r in runs if r["f"] == f]
            dmax = max(r["density"] for r in rr)
            nmax = max(r["n_seg"] for r in rr)
            best = [r for r in rr if r["density"] == dmax and r["n_seg"] == nmax][0]
            ref = stored(d, best["key"])
            G = _blocks(vc.StoredGSM(d, best["key"]))
            errs = []
            for r in rr:
                if r is best:
                    continue
                if (r["n_seg"] == nmax and r["density"] == sorted({x["density"] for x in rr})[-2]) or \
                        (r["density"] == dmax and r["n_seg"] != nmax):
                    errs.append(float(np.max(abs(_blocks(vc.StoredGSM(d, r["key"])) - G))))
            plane = "H" if case.startswith("H") else "E"
            inc = "TE1,0" if plane == "H" else "TE0,1"
            if case.startswith("H"):
                L = 0.1 if case.endswith("100") else 0.2
                comp = wr.Taper(L, 45e-3, H, profile="linear", guide=Guide(W, H))
            elif case == "E_rc50":
                comp = wr.Taper(0.05, 1.6e-3, W, profile="raised_cosine", plane="E", guide=Guide(W, H))
            else:
                comp = wr.StrangledBend(90, 0.05, 1.6e-3, 0.05, guide=Guide(W, H))
            b1 = vc_basis(ref["names1"], comp, f, 1)
            b2 = vc_basis(ref["names2"], comp, f, 2)
            t = vc.Timer()
            g = comp.gsm(f, b1, b2)
            dt = t()
            ddb, dp, worst = entry_compare(ref, as_dict(g), inc)
            g3 = comp.gsm(f, b1, b2, mode_factor=3.0)
            ddb3, dp3, _ = entry_compare(ref, as_dict(g3), inc)
            ok = abs(ddb) <= DB_TOL and dp <= P_TOL
            main = vc.db(ref["S21"][ref["names2"].index(inc), ref["names1"].index(inc)])
            out.append(f"| {case} | {f / 1e9:g} | {main:.5f} | {', '.join(f'{e:.1e}' for e in errs)} | {ddb:+.5f} | "
                       f"{dp:.1e} ({worst}) | {'yes' if ok else 'NO'} | {ddb3:+.5f} / {dp3:.1e} | {dt:.1f} |")
            res.append(dict(part="tapers", case=case, f=f, ref_main_db=float(main), fem_err=errs, d_main_db=ddb,
                            worst_dP=dp, worst=worst, ok=bool(ok), d_main_db_mf3=ddb3, worst_dP_mf3=dp3, seconds=dt))
            print(out[-1], flush=True)


def vc_basis(names, comp, f, port):
    """The component's default port basis, reordered/checked to contain `names`."""
    b = comp.default_basis(f)[port - 1]
    missing = [n for n in names if n not in b.names]
    if missing:
        raise RuntimeError(f"port {port} basis lacks {missing}")
    return b


# --------------------------------------------------------------------------------------------- staircase
def staircase(out, res):
    import wrmodes as wr
    from wrmodes.components import Guide
    from wrmodes.profiles import BottleProfile, LinearProfile
    from ref_taper_staircase import A1, A2, B1, B2, F0, LT
    out.append("\n### WP5 tapers of a and b (140 mm, 75 GHz, TE1,0 from the small end) vs the WP4 staircase\n")
    out.append("Staircase reference: Richardson in 1/n from n = 400, 800 at the largest cutoff factor; error estimate "
               "= |extrapolated(cf 4) - extrapolated(cf 3)| + |S(800) - extrapolated|.\n")
    out.append("| profile | ref TE1,0 (dB) | ref P(TE3,0) | ref error est. (max ΔP) | WP5 Δ main (dB) | WP5 worst ΔP (mode) | "
               "pass | WP5 P(TE3,0) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for prof in ("linear", "bottle_a"):
        d = vc.load_npz(f"taper_staircase_{prof}.npz")
        def get(cf, n):
            return stored(d, f"{prof}_cf{cf:g}_n{n}")
        ext = {}
        for cf in (3, 4):
            a, b = get(cf, 400), get(cf, 800)
            ext[cf] = {k: (2 * b[k] - a[k]) if k.startswith("S") else b[k] for k in b}
        ref = ext[4]
        i0 = ref["names1"].index("TE1,0")
        pr = lambda X: abs(X["S21"][:, i0]) ** 2
        err = float(np.max(abs(pr(ext[4]) - pr(ext[3]))) + np.max(abs(pr(get(4, 800)) - pr(ext[4]))))
        if prof == "linear":
            comp = wr.Taper(LT, A2, B2, profile="linear", guide=Guide(A1, B1))
        else:
            bp, lp = BottleProfile(3.5), LinearProfile()
            comp = wr.VariableGuide(LT, (lambda z: A1 + (A2 - A1) * bp(z / LT), lambda z: (A2 - A1) * bp.derivative(z / LT) / LT),
                                    (lambda z: B1 + (B2 - B1) * z / LT, lambda z: (B2 - B1) / LT + 0 * z),
                                    guide=Guide(A1, B1))
        b1, b2 = comp.default_basis(F0)
        g = comp.gsm(F0, b1, b2)
        ddb, dp, worst = entry_compare(ref, as_dict(g), "TE1,0")
        ok = abs(ddb) <= DB_TOL and dp <= P_TOL
        p30 = abs(g.S21[g.basis2.index_name("TE3,0") if hasattr(g.basis2, "index_name") else g.basis2.names.index("TE3,0"),
                        g.basis1.names.index("TE1,0")]) ** 2
        main = vc.db(ref["S21"][ref["names2"].index("TE1,0"), i0])
        out.append(f"| {prof} | {main:.5f} | {pr(ref)[ref['names2'].index('TE3,0')]:.3e} | {err:.1e} | {ddb:+.5f} | "
                   f"{dp:.1e} ({worst}) | {'yes' if ok else 'NO'} | {p30:.3e} |")
        res.append(dict(part="staircase", profile=prof, ref_main_db=float(main), ref_err=err, d_main_db=ddb,
                        worst_dP=dp, worst=worst, ok=bool(ok), wp5_p30=float(p30)))
        print(out[-1], flush=True)


# --------------------------------------------------------------------------------------------- twists
def twists(out, res):
    import wrmodes as wr
    from wrmodes.components import Guide
    import ref_twist as rt
    out.append("\n### WP6 twists vs the helical-coordinate FEM (incident TE1,0, powers in % of the input)\n")
    out.append("| case | f (GHz) | ref spurious | ref error est. | WP6 spurious | WP6 Δ main (dB) | WP6 worst ΔP (mode) | pass | "
               "WP6 cutoff_factor 2: worst ΔP | largest modes (ref → WP6) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    cases = rt.cases()
    for name in ("doane_constant", "doane_cosine", "doane_triangular", "square_const_5lam", "square_const_20lam",
                 "strong_tau0p1", "strong_tau0p5"):
        path = os.path.join(vc.DATA, f"twist_{name}.npz")
        if not os.path.exists(path):
            out.append(f"| {name} | – | not computed | | | | | | | |")
            continue
        d = vc.load_npz(path)
        c = cases[name]
        rows = d["meta"]["rows"]
        for f in c["freqs"]:
            key = f"{name}_f{f / 1e9:g}_main"
            names = [str(s) for s in d[f"{key}_names"]]
            ref = dict(S11=d[f"{key}_S11"], S21=d[f"{key}_S21"], names1=names, names2=names)
            others = [r for r in rows if abs(r["f"] - f) < 1 and r["tag"] != "main"]
            i0 = names.index("TE1,0")
            p_ref = abs(ref["S21"][:, i0]) ** 2
            err = 0.0
            for r in others:
                k2 = f"{name}_f{f / 1e9:g}_{r['tag']}"
                p2 = abs(d[f"{k2}_S21"][:, i0]) ** 2
                err = max(err, float(np.max(abs(p2 - p_ref))))
            main_row = [r for r in rows if abs(r["f"] - f) < 1 and r["tag"] == "main"][0]
            if "extrapolation_change" in main_row:
                err = max(err, main_row["extrapolation_change"] ** 2 + 2 * main_row["extrapolation_change"] * np.sqrt(np.max(p_ref[np.arange(len(p_ref)) != i0])))
            guide = Guide(c["a"], c["b"])
            tw = wr.Twist(np.degrees(c["theta"]), c["L"], profile=c["profile"], guide=guide)
            from wrmodes.modes import ModeBasis
            pb = ModeBasis(c["a"], c["b"], [m for m in tw.default_basis(f)[0].modes], frame="H")
            t = vc.Timer()
            g = tw.gsm(f)
            dt = t()
            test = as_dict(g)
            ddb, dp, worst = entry_compare(ref, test, "TE1,0")
            g2 = tw.replace(cutoff_factor=2.0).gsm(f, g.basis1, g.basis2)
            _, dp2, _ = entry_compare(ref, as_dict(g2), "TE1,0")
            ok = abs(ddb) <= DB_TOL and dp <= P_TOL
            spur_r = 100 * (np.sum(p_ref) - p_ref[i0])
            jt = test["names1"].index("TE1,0")
            pt = abs(test["S21"][:, jt]) ** 2
            spur_t = 100 * (np.sum(pt) - pt[test["names2"].index("TE1,0")])
            top = np.argsort(-p_ref)[1:4]
            tops = ", ".join(f"{names[j]} {100 * p_ref[j]:.3f}→{100 * pt[test['names2'].index(names[j])]:.3f}" for j in top)
            out.append(f"| {name} | {f / 1e9:g} | {spur_r:.4f} | {err:.1e} | {spur_t:.4f} | {ddb:+.5f} | {dp:.1e} ({worst}) | "
                       f"{'yes' if ok else 'NO'} | {dp2:.1e} | {tops} |")
            res.append(dict(part="twists", case=name, f=f, ref_spurious_pct=float(spur_r), ref_err=err,
                            wp6_spurious_pct=float(spur_t), d_main_db=ddb, worst_dP=dp, worst=worst, ok=bool(ok),
                            worst_dP_cf2=dp2, seconds=dt,
                            ref_powers={names[j]: float(p_ref[j]) for j in range(len(names))},
                            wp6_powers={n: float(pt[test["names2"].index(n)]) for n in names if n in test["names2"]}))
            print(out[-1], flush=True)


# --------------------------------------------------------------------------------------------- miters
def miters(out, res):
    import wrmodes as wr
    from wrmodes.components import Guide
    from wrmodes.fem2d import Miter
    import ref_miter_fem as rm
    g = Guide(vc.W90, vc.H90)
    out.append("\n### WP4b mode-matching miters (and the WP3/WP3b FEM miter) vs the FEM on WP7b outlines, 90°, WR-90\n")
    out.append("Per plane and chamfer: worst over the frequencies of |Δ TE1,0 transmission| (dB) and of the power "
               "difference over every propagating entry of the GSM; the frequencies outside the tolerance are listed.\n")
    out.append("| plane | d/a | ref mesh error (max \\|ΔS\\| fine − main) | MM worst Δ main (dB) @ f | MM worst ΔP @ f | MM fails (f, GHz) | "
               "FEM Miter worst Δ main | FEM Miter worst ΔP | MM time (s, 90 GHz) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for plane in ("H", "E"):
        for dd in rm.CHAMFERS:
            path = os.path.join(vc.DATA, f"miter_fem_{plane}_d{dd:.4f}.npz")
            if not os.path.exists(path):
                continue
            d = vc.load_npz(path)
            inc = "TE1,0" if plane == "H" else "TE0,1"
            mm = wr.MiterMM(None if abs(dd - 1 / np.sqrt(2)) < 1e-9 else dd, plane=plane, guide=g)
            fm = Miter(90, chamfer=dd, plane=plane, guide=g)
            worst = dict(mm_db=(0, None), mm_p=(0, None), fm_db=(0, None), fm_p=(0, None))
            fails, mesh_err, t90 = [], 0.0, None
            for f in d["meta"]["freqs"]:
                key = f"{plane}_d{dd:.4f}_f{f:g}_main"
                ref = stored(d, key)
                if f"{plane}_d{dd:.4f}_f{f:g}_fine_S11" in d:
                    mesh_err = max(mesh_err, float(np.max(abs(_blocks(vc.StoredGSM(d, key)) - _blocks(
                        vc.StoredGSM(d, f"{plane}_d{dd:.4f}_f{f:g}_fine"))))))
                basis = g.basis(f * 1e9, 1.0, plane)
                t = vc.Timer()
                G = mm.gsm(f * 1e9, basis, basis)
                if f == 90:
                    t90 = t()
                F = fm.gsm(f * 1e9, basis, basis)
                R = _blocks(vc.StoredGSM(d, key))
                i = ref["names1"].index(inc)
                for tag, X in (("mm", G), ("fm", F)):
                    ddb = float(vc.db(X.S21[i, i]) - vc.db(ref["S21"][i, i]))
                    dp = float(np.max(abs(abs(_blocks(X)) ** 2 - abs(R) ** 2)))
                    if abs(ddb) > abs(worst[f"{tag}_db"][0]):
                        worst[f"{tag}_db"] = (ddb, f)
                    if dp > worst[f"{tag}_p"][0]:
                        worst[f"{tag}_p"] = (dp, f)
                    res.append(dict(part="miters", plane=plane, d=dd, f=f, comp=tag, d_main_db=ddb, worst_dP=dp,
                                    ok=bool(abs(ddb) <= DB_TOL and dp <= P_TOL)))
                    if tag == "mm" and not (abs(ddb) <= DB_TOL and dp <= P_TOL):
                        fails.append(f)
            fl = ", ".join(f"{x:g}" for x in fails) if fails else "none"
            out.append(f"| {plane} | {dd:.4f} | {mesh_err:.1e} | {worst['mm_db'][0]:+.4f} @ {worst['mm_db'][1]} | "
                       f"{worst['mm_p'][0]:.1e} @ {worst['mm_p'][1]} | {fl} | {worst['fm_db'][0]:+.4f} @ {worst['fm_db'][1]} | "
                       f"{worst['fm_p'][0]:.1e} @ {worst['fm_p'][1]} | {t90 if t90 is None else round(t90, 2)} |")
            print(out[-1], flush=True)


# --------------------------------------------------------------------------------------------- filter
def filt(out, res):
    from wrmodes.components import Guide
    from wrmodes.fem2d import Coplanar2D, FEMOptions, straight_outline
    from ref_filter import F, L, ZW
    d = vc.load_npz("filter_lossywall_wp2.npz")
    ref = vc.StoredGSM(d, "M160")
    ref80 = vc.StoredGSM(d, "M80")
    g = Guide(vc.W90, vc.H90)
    basis = g.basis(F, 1.0, "H")
    out.append("\n### WP3b impedance walls vs WP2's exact lossy-wall filter (Butterweck: 75 GHz, 250 Ω, 15 cm)\n")
    out.append("| mode | WP2 M = 160 (dB) | WP2 M = 80 | FEM density 6 (default) | FEM density 10 | Δ rel. (density 10) | pass (5 %) | Butterweck |")
    out.append("|---|---|---|---|---|---|---|---|")
    fem = {}
    for dens in (6, 10):
        c = Coplanar2D(straight_outline(vc.W90, L), b=vc.H90, plane="H", guide=g, wall_impedance=ZW,
                       options=FEMOptions(mesh_density=dens))
        fem[dens] = c.gsm(F, basis, basis)
    butter = {"TE1,0": 0.6, "TE2,0": 2.5, "TE3,0": 5.4}
    for n in basis.names:
        i = basis.names.index(n)
        a_ref = -vc.db(ref.S21[i, i])
        if a_ref < 1e-6:
            continue
        a80 = -vc.db(ref80.S21[i, i])
        a6, a10 = -vc.db(fem[6].S21[i, i]), -vc.db(fem[10].S21[i, i])
        rel = (a10 - a_ref) / a_ref
        ok = abs(rel) <= 0.05
        res.append(dict(part="filter", mode=n, wp2=float(a_ref), fem6=float(a6), fem10=float(a10), rel=float(rel), ok=bool(ok)))
        if n in ("TE1,0", "TE2,0", "TE3,0", "TE0,1", "TE1,1", "TM1,1", "TE4,0", "TE1,2", "TE6,0") or not ok:
            out.append(f"| {n} | {a_ref:.4f} | {a80:.4f} | {a6:.4f} | {a10:.4f} | {rel:+.2e} | {'yes' if ok else 'NO'} | "
                       f"{butter.get(n, '')} |")
    worst = max((r for r in res if r["part"] == "filter"), key=lambda r: abs(r["rel"]))
    out.append(f"\nWorst over all {sum(1 for r in res if r['part'] == 'filter')} attenuated modes: {worst['mode']} "
               f"{worst['rel']:+.2e} relative.")
    print("\n".join(out[-12:]), flush=True)


# --------------------------------------------------------------------------------------------- literature (reference self-checks)
def literature(out, res):
    import csv
    out.append("\n### Reference self-checks against the literature\n")
    # Kotik & Sivov Fig. 3 (optical mirror, H plane): |S21(TE1,0)|^2 vs kd, WP7b FEM sweep
    d = vc.load_npz("miter_fem_H_d0.7071.npz")
    fr = np.array(d["meta"]["freqs"], float)
    p = []
    for f in fr:
        s = vc.StoredGSM(d, f"H_d0.7071_f{f:g}_main")
        p.append(abs(s.s21("TE1,0", "TE1,0")) ** 2)
    kd = 2 * np.pi * fr * 1e9 / 299792458.0 * vc.W90
    rows = [r for r in csv.reader(l for l in open(os.path.join(vc.REPO, "benchmarks/literature/kotik1965_fig3_rigorous.csv"))
                                  if not l.startswith("#"))][1:]
    x = np.array([float(r[1]) for r in rows]); y = np.array([float(r[2]) for r in rows]); u = np.array([float(r[3]) for r in rows])
    within = []
    for k, pp in zip(kd, p):
        if k < x.min() or k > x.max():
            continue
        j = np.argmin(abs(x - k))
        if abs(x[j] - k) > 0.05:
            continue
        within.append((k, pp, y[j], u[j], abs(pp - y[j]) <= u[j]))
    n_ok = sum(w[-1] for w in within)
    out.append(f"* Kotik & Sivov (1965) Fig. 3, optical-mirror H-plane miter: the WP7b FEM sweep (2 GHz grid) lies "
               f"within the digitizing uncertainty at {n_ok} of {len(within)} grid points with a CSV point within 0.05 in kd; "
               "outside: " + (", ".join(f"kd {w[0]:.2f}: {w[1]:.4f} vs {w[2]:.4f} ± {w[3]:.4f}" for w in within if not w[-1]) or "none") + ".")
    res.append(dict(part="literature", source="kotik1965_fig3", n=len(within), n_ok=n_ok))
    # Doane & Anderson Fig. 9: total spurious power at 15, 16.5, 18 GHz
    rows = [r for r in csv.reader(l for l in open(os.path.join(vc.REPO, "benchmarks/literature/doane1989_fig9.csv"))
                                  if not l.startswith("#"))][1:]
    for prof in ("constant", "cosine", "triangular"):
        path = os.path.join(vc.DATA, f"twist_doane_{prof}.npz")
        if not os.path.exists(path):
            continue
        dz = vc.load_npz(path)
        xs = np.array([float(r[1]) for r in rows if r[0] == prof]); ys = np.array([float(r[2]) for r in rows if r[0] == prof])
        us = np.array([float(r[3]) for r in rows if r[0] == prof])
        cells = []
        for f in (15.0, 16.5, 18.0):
            key = f"doane_{prof}_f{f:g}_main"
            names = [str(s) for s in dz[f"{key}_names"]]
            pr = abs(dz[f"{key}_S21"][:, names.index("TE1,0")]) ** 2
            spur = 100 * (pr.sum() - pr[names.index("TE1,0")])
            j = np.argmin(abs(xs - f))
            cells.append(f"{f:g} GHz: {spur:.3f} % vs Fig. 9 {ys[j]:.3f} ± {us[j]:.3f} %")
            res.append(dict(part="literature", source="doane1989_fig9", profile=prof, f=f, ref=float(spur),
                            fig9=float(ys[j]), u=float(us[j])))
        out.append(f"* Doane & Anderson (1989) Fig. 9, {prof}: " + "; ".join(cells) + ".")
    print("\n".join(out[-5:]), flush=True)


PARTS = {"tapers": tapers, "staircase": staircase, "twists": twists, "miters": miters, "filter": filt,
         "literature": literature}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("parts", nargs="*", default=list(PARTS))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = []
    for part in a.parts:
        res = []
        PARTS[part](out, res)
        with open(os.path.join(vc.DATA, f"compare_{part}.json"), "w") as fh:
            json.dump(dict(meta=json.loads(vc.meta(__file__, part=part)), results=res), fh, indent=1, default=float)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write("\n".join(out) + "\n")
