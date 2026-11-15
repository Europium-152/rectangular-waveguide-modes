"""WF2 (tasks/WF2-wave2-fixes.md): the components after the wave 2 fixes against WP7b's committed references.

WP7b's reference scripts and data of this folder are not changed; this script only re-runs the components and
compares them with the stored npz files, in the same way as `compare_v2.py` (whose functions it uses):

* twists (item 1): every WP7b twist case at the new `Twist` defaults, plus tau a = 0.5 with mode_factor 4 and 6,
  and WP6's model (mode_factor 1, no tail) for comparison; run times;
* bends next to a miter (item 2): `Line` [SmoothBend, fem2d.Miter] at cutoff_factor 1 and 3 against the single FEM
  domain of `ref_evanescent_bends.py` (needs wrmodes[fem]);
* strangled-bend port bases (item 3): the 200 flat heights of the adversarial test.

    PYTHONPATH=src:benchmarks/validation_v2 python benchmarks/validation_v2/compare_wf2.py [twists bends ports]
        [--out FILE]

Writes markdown tables (stdout and --out) and data/compare_wf2_<part>.json. Tolerances as in compare_v2.py
(CONVENTIONS.md section 8): 0.01 dB on the main transmission, 1e-4 of the input power per converted mode.
"""
import argparse
import json
import os
import warnings

import numpy as np

import vcommon as vc
from compare_v2 import DB_TOL, P_TOL, as_dict, entry_compare

TWIST_CASES = ("doane_constant", "doane_cosine", "doane_triangular", "square_const_5lam", "square_const_20lam",
               "strong_tau0p1", "strong_tau0p5")


def twists(out, res):
    import wrmodes as wr
    from wrmodes.components import Guide
    import ref_twist as rt
    out.append("\n### After WF2: twists at the new defaults vs the helical-coordinate FEM (incident TE1,0)\n")
    out.append("Settings: `Twist` defaults (internal list to mode_factor 3 f for constant, 2 f for variable rates, "
               "quasi-static tail). 'WP6' = WP6's model (mode_factor 1, tail off), the default before WF2.\n")
    out.append("| case | f (GHz) | setting | modes | WF2 worst ΔP (mode) | Δ main (dB) | pass | WP6 worst ΔP | time (s) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    cases = rt.cases()
    for name in TWIST_CASES:
        d = vc.load_npz(f"twist_{name}.npz")
        c = cases[name]
        for f in c["freqs"]:
            key = f"{name}_f{f / 1e9:g}_main"
            names = [str(s) for s in d[f"{key}_names"]]
            ref = dict(S11=d[f"{key}_S11"], S21=d[f"{key}_S21"], names1=names, names2=names)
            tw = wr.Twist(np.degrees(c["theta"]), c["L"], profile=c["profile"], guide=Guide(c["a"], c["b"]))
            g6 = tw.replace(mode_factor=1.0, tail=False).gsm(f)
            _, dp6, _ = entry_compare(ref, as_dict(g6), "TE1,0")
            settings = [("default", tw)]
            if name == "strong_tau0p5":
                settings += [("mode_factor 4", tw.replace(mode_factor=4.0)), ("mode_factor 6", tw.replace(mode_factor=6.0))]
            for tag, t in settings:
                tm = vc.Timer()
                g = t.gsm(f)
                dt = tm()
                ddb, dp, worst = entry_compare(ref, as_dict(g), "TE1,0")
                ok = abs(ddb) <= DB_TOL and dp <= P_TOL
                out.append(f"| {name} | {f / 1e9:g} | {tag} (mf {t.mode_factor:g}) | {g.info['n_modes']} | {dp:.1e} ({worst}) | "
                           f"{ddb:+.5f} | {'yes' if ok else 'NO'} | {dp6:.1e} | {dt:.1f} |")
                res.append(dict(part="twists", case=name, f=f, setting=tag, mode_factor=t.mode_factor,
                                n_modes=g.info["n_modes"], n_slices=g.info["n_slices"], worst_dP=dp, worst=worst,
                                d_main_db=ddb, ok=bool(ok), wp6_worst_dP=dp6, seconds=dt,
                                unitarity=g.unitarity_defect(), reciprocity=g.reciprocity_defect()))
                print(out[-1], flush=True)
    doane = [r for r in res if r["case"].startswith("doane") and r["setting"] == "default"]
    tot = sum(r["seconds"] for r in doane)
    out.append(f"\nRun time at the defaults, Doane's three twists at 15, 16.5, 18 GHz (9 GSMs, one BLAS thread): "
               f"{tot:.0f} s (constant {sum(r['seconds'] for r in doane if r['case'] == 'doane_constant'):.1f} s, "
               f"cosine {sum(r['seconds'] for r in doane if r['case'] == 'doane_cosine'):.0f} s, "
               f"triangular {sum(r['seconds'] for r in doane if r['case'] == 'doane_triangular'):.0f} s).")
    print(out[-1], flush=True)


def bends(out, res):
    import wrmodes as wr
    from wrmodes.components import ConstantCurvature, Guide, SmoothBend
    from wrmodes.fem2d import Miter
    import ref_evanescent_bends as reb
    d = vc.load_npz(reb.OUT)
    W, H = vc.W90, vc.H90
    g = Guide(W, H)
    miter = Miter(90, plane="H")
    out.append("\n### After WF2: short bend + miter, Line [SmoothBend, fem2d.Miter] vs one FEM domain\n")
    out.append("Max over the propagating GSM of | |S|² difference |. SmoothBend couples the evanescent modes of its basis "
               "(VariableGuide route, internal cutoff factor twice the port basis's); 'WP7b line' = before WF2, "
               "'FEM cascade' = Line [FEM arc, fem2d.Miter] (WP7b).\n")
    out.append("| R/a | f (GHz) | L/a | line cf 1 | line cf 3 | Δ TE1,0 cf 3 (dB) | target | pass | WP7b line cf 3 | FEM cascade cf 3 | time cf 3 (s) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in d["meta"]["rows"]:
        if r["L_over_a"] == 0:
            continue
        key, f, L, R = r["key"], r["f"], r["L_over_a"] * W, r["R_over_a"] * W
        S_fem = vc.StoredGSM(d, key + "_fem")
        names = S_fem.names1
        Sr = np.block([[S_fem.S11, S_fem.S12], [S_fem.S21, S_fem.S22]])
        n, i0 = len(names), names.index("TE1,0")
        row = dict(part="bends", key=key, R_over_a=r["R_over_a"], f=f, L_over_a=r["L_over_a"])
        for cf in (1.0, 3.0):
            tm = vc.Timer()
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                G = wr.Line(g, [SmoothBend(L, ConstantCurvature(1 / R), plane="H"), miter]).gsm(f, cutoff_factor=cf)
            row[f"seconds_cf{cf:g}"] = tm()
            row[f"warned_cf{cf:g}"] = any(w.category.__name__ == "EvanescentJunctionWarning" for w in wl)
            Sl = reb._blk(G, names)
            row[f"max_dP_cf{cf:g}"] = float(np.max(abs(abs(Sl) ** 2 - abs(Sr) ** 2)))
            row[f"dTE10_dB_cf{cf:g}"] = float(vc.db(Sl[n + i0, i0]) - vc.db(Sr[n + i0, i0]))
            row[f"unitarity_cf{cf:g}"] = float(G.unitarity_defect())
        tol = 2e-4 if r["R_over_a"] == 2 else 3e-5
        row["target"], row["ok"] = tol, bool(row["max_dP_cf3"] <= tol)
        out.append(f"| {r['R_over_a']:g} | {f / 1e9:g} | {r['L_over_a']:g} | {row['max_dP_cf1']:.1e} | "
                   f"{row['max_dP_cf3']:.1e} | {row['dTE10_dB_cf3']:+.5f} | {tol:.0e} | {'yes' if row['ok'] else 'NO'} | "
                   f"{r['line_cf3_max_dP']:.1e} | {r['fem_cascade_cf3_max_dP']:.1e} | {row['seconds_cf3']:.1f} |")
        res.append(row)
        print(out[-1], flush=True)
    warned = all(x["warned_cf1"] and not x["warned_cf3"] for x in res)
    out.append(f"\nLine warning (EvanescentJunctionWarning): {'issued at cutoff_factor 1 and not at 3 in every case' if warned else 'see json'}.")


def ports(out, res):
    import wrmodes as wr
    from wrmodes.components import Guide
    f = 60e9
    heights = np.linspace(1.5e-3, 2.6e-3, 200)
    bad = [float(b1) for b1 in heights
           if not (lambda bb: bb[0] == bb[1] and bb[0].names == bb[1].names)(
               wr.StrangledBend(90, 0.05, b1, 0.03, guide=Guide(vc.W90, vc.H90)).default_basis(f))]
    out.append("\n### After WF2: StrangledBend port bases\n")
    out.append(f"WR-90 at 60 GHz, 200 flat heights 1.5-2.6 mm: basis2 != basis1 for {len(bad)} (WP7b: 25).")
    res.append(dict(part="ports", unequal=len(bad), heights=len(heights)))
    print(out[-1], flush=True)


PARTS = {"twists": twists, "bends": bends, "ports": ports}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("parts", nargs="*", default=list(PARTS))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = []
    for part in a.parts:
        res = []
        PARTS[part](out, res)
        with open(os.path.join(vc.DATA, f"compare_wf2_{part}.json"), "w") as fh:
            json.dump(dict(meta=json.loads(vc.meta(__file__, part=part)), results=res), fh, indent=1, default=float)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write("\n".join(out) + "\n")
