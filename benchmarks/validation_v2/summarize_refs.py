"""Convergence tables of the WP7b references (markdown, for REPORT.md). Reads only data/*.npz of this folder.

    PYTHONPATH=src:benchmarks/validation_v2 python benchmarks/validation_v2/summarize_refs.py
"""
import os

import numpy as np

import vcommon as vc


def _blocks(s):
    return np.block([[s.S11, s.S12], [s.S21, s.S22]])


def tapers(out):
    out.append("\n#### Tapers and strangled bend (2D FEM on WP7b outlines)\n")
    out.append("Reference = finest run (mesh density 16 or 18 elements per transverse wavelength, 64 arcs per curved "
               "wall). Changes are max over every entry of the propagating GSM.\n")
    out.append("| case | f (GHz) | ndof (finest) | max \\|ΔS\\| mesh (next coarser) | max \\|Δ\\|S\\|²\\| mesh | max \\|ΔS\\| walls (32 arcs) | "
               "max \\|Δ\\|S\\|²\\| walls | main transmission (dB) | time finest (s) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for case in ("H_linear_L100", "H_linear_L200", "E_rc50", "strangled"):
        d = vc.load_npz(f"taper_fem_{case}.npz")
        runs = d["meta"]["runs"]
        for f in sorted(set(r["f"] for r in runs)):
            rr = [r for r in runs if r["f"] == f]
            dens = sorted({r["density"] for r in rr})
            nseg = max(r["n_seg"] for r in rr)
            best = [r for r in rr if r["density"] == dens[-1] and r["n_seg"] == nseg][0]
            B = _blocks(vc.StoredGSM(d, best["key"]))
            m = [r for r in rr if r["density"] == dens[-2] and r["n_seg"] == nseg][0]
            M = _blocks(vc.StoredGSM(d, m["key"]))
            w = [r for r in rr if r["density"] == dens[-1] and r["n_seg"] != nseg]
            if w:
                Wb = _blocks(vc.StoredGSM(d, w[0]["key"]))
                wcell = f"{np.max(abs(Wb - B)):.1e} | {np.max(abs(abs(Wb) ** 2 - abs(B) ** 2)):.1e}"
            else:
                wcell = "– (straight walls) | –"
            s = vc.StoredGSM(d, best["key"])
            inc = "TE1,0" if case.startswith("H") else "TE0,1"
            out.append(f"| {case} | {f / 1e9:g} | {best['ndof']} | {np.max(abs(M - B)):.1e} | "
                       f"{np.max(abs(abs(M) ** 2 - abs(B) ** 2)):.1e} | {wcell} | {vc.db(s.s21(inc, inc)):.5f} | {best['seconds']} |")


def staircase(out):
    out.append("\n#### 2D tapers (staircase of WP4 steps, class m odd / n even)\n")
    out.append("| profile | cutoff factor | modes (large end) | TE1,0 (dB) n = 100 / 200 / 400 / 800 | P(TE3,0) n = 400 / 800 | "
               "Richardson (1/n) TE1,0 dB, P(TE3,0) | observed order (P(TE3,0)) | time n = 800 (s) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for prof in ("linear", "bottle_a"):
        d = vc.load_npz(f"taper_staircase_{prof}.npz")
        rows = d["meta"]["rows"]
        for cf in sorted({r["cutoff_factor"] for r in rows}):
            rr = {r["n"]: r for r in rows if r["cutoff_factor"] == cf}
            def P(n):
                k = f"{prof}_cf{cf:g}_n{n}"
                names = [str(x) for x in d[f"{k}_names2"]]
                return abs(d[f"{k}_S21"][names.index("TE3,0"), 0]) ** 2, d[f"{k}_S21"][names.index("TE1,0"), 0]
            p = {n: P(n) for n in rr}
            ext_s = 2 * p[800][1] - p[400][1]
            ext_p = 2 * p[800][0] - p[400][0]
            try:
                order = np.log2(abs(p[200][0] - p[400][0]) / abs(p[400][0] - p[800][0]))
            except (ZeroDivisionError, FloatingPointError):
                order = np.nan
            out.append(f"| {prof} | {cf:g} | {rr[800]['n_modes_large']} | "
                       + " / ".join(f"{rr[n]['te10_db']:.5f}" for n in (100, 200, 400, 800))
                       + f" | {p[400][0]:.3e} / {p[800][0]:.3e} | {vc.db(ext_s):.5f}, {ext_p:.3e} | {order:.2f} | {rr[800]['seconds']} |")


def twists(out):
    out.append("\n#### Twists (helical FEM + mode matching)\n")
    out.append("Spurious = power in every propagating mode other than TE1,0 at port 2 (% of the input). 'main' is the "
               "value used; checks: finer mesh (h), more modes (cutoff factor), for variable rates the section count "
               "(Richardson from n = 64, 128; 'sec' = max |ΔS21| column between n = 64 and 128, 'ext' = between n = 128 and "
               "the extrapolation). Unitarity: max |S^H S - I| of the propagating block; 'col' the same for the TE1,0 column.\n")
    out.append("| case | f (GHz) | setting | modes | spurious (%) | TE1,0 (dB) | unitarity / col | sec / ext | time (s) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for name in ("doane_constant", "doane_constant_3mm", "doane_cosine", "doane_triangular", "doane_triangular_cf2",
                 "square_const_5lam",
                 "square_const_20lam", "strong_tau0p1", "strong_tau0p5"):
        p = os.path.join(vc.DATA, f"twist_{name}.npz")
        if not os.path.exists(p):
            out.append(f"| {name} | – | not computed | | | | | | |")
            continue
        d = vc.load_npz(p)
        for r in d["meta"]["rows"]:
            tr = r["transmitted_final"]
            spur = 100 * sum(v for k, v in tr.items() if k != "TE1,0")
            se = f"{r['section_change']:.1e} / {r['extrapolation_change']:.1e}" if "section_change" in r else "–"
            out.append(f"| {name} | {r['f'] / 1e9:g} | {r['tag']} (h {1e3 * r['h']:g} mm, cf {r['cutoff_factor']:g}) | "
                       f"{r['n_basis']} | {spur:.5f} | {10 * np.log10(tr['TE1,0']):.6f} | {r['unitarity']:.1e} / "
                       f"{r['unitarity_incident_column']:.1e} | {se} | {r['seconds']} |")


def square(out):
    out.append("\n#### Square guide (30 mm, 2a/λ = 2.5): polarization of TE1,0 after a 90° twist\n")
    out.append("Local angle = arctan(Re(a01/a10)) of the field at the centre (T3b box 5: −cθ = −72.951° for the "
               "two-mode law), power in all other modes, reflected power.\n")
    out.append("| case | local angle (°) | ellipticity Im(a01/a10)/... | other modes | reflected |")
    out.append("|---|---|---|---|---|")
    for name in ("square_const_5lam", "square_const_20lam"):
        p = os.path.join(vc.DATA, f"twist_{name}.npz")
        if not os.path.exists(p):
            continue
        d = vc.load_npz(p)
        f = d["meta"]["geometry"]["freqs"][0]
        key = f"{name}_f{f / 1e9:g}_main"
        names = [str(s) for s in d[f"{key}_names"]]
        S21, S11 = d[f"{key}_S21"], d[f"{key}_S11"]
        i10, i01 = names.index("TE1,0"), names.index("TE0,1")
        r = S21[i01, i10] / S21[i10, i10]
        ang = np.degrees(np.arctan(r.real))
        p = abs(S21[:, i10]) ** 2
        other = p.sum() - p[i10] - p[i01]
        refl = np.sum(abs(S11[:, i10]) ** 2)
        out.append(f"| {name} | {ang:.3f} | {r.imag:.1e} | {other:.2e} | {refl:.2e} |")


def miters(out):
    out.append("\n#### 90° miters (2D FEM on WP7b outlines, mesh density 10)\n")
    out.append("Mesh check at 10, 30, 50, 70, 90 GHz: density 16 ('fine') and 20 evanescent port modes per polarization "
               "('more_ports'); max over every entry of the propagating GSM.\n")
    out.append("| plane | d/a | max \\|ΔS\\| fine | max \\|Δ\\|S\\|²\\| fine | max \\|ΔS\\| more ports | max \\|Δ main\\| fine (dB) | "
               "reciprocity, S11 = S22 | time 90 GHz (s) |")
    out.append("|---|---|---|---|---|---|---|---|")
    import ref_miter_fem as rm
    for plane in ("H", "E"):
        for dd in rm.CHAMFERS:
            p = os.path.join(vc.DATA, f"miter_fem_{plane}_d{dd:.4f}.npz")
            if not os.path.exists(p):
                continue
            d = vc.load_npz(p)
            inc = "TE1,0" if plane == "H" else "TE0,1"
            e1 = e2 = e3 = e4 = sym = 0.0
            for f in d["meta"]["conv_freqs"]:
                base = vc.StoredGSM(d, f"{plane}_d{dd:.4f}_f{f:g}_main")
                B = _blocks(base)
                fi = vc.StoredGSM(d, f"{plane}_d{dd:.4f}_f{f:g}_fine")
                mp = vc.StoredGSM(d, f"{plane}_d{dd:.4f}_f{f:g}_more_ports")
                e1 = max(e1, np.max(abs(_blocks(fi) - B)))
                e2 = max(e2, np.max(abs(abs(_blocks(fi)) ** 2 - abs(B) ** 2)))
                e3 = max(e3, np.max(abs(_blocks(mp) - B)))
                e4 = max(e4, abs(vc.db(fi.s21(inc, inc)) - vc.db(base.s21(inc, inc))))
            for f in d["meta"]["freqs"]:
                s = vc.StoredGSM(d, f"{plane}_d{dd:.4f}_f{f:g}_main")
                sym = max(sym, np.max(abs(s.S11 - s.S22)), np.max(abs(_blocks(s) - _blocks(s).T)))
            t90 = [r["seconds"] for r in d["meta"]["rows"] if r["f"] == 90 and r["tag"] == "main"]
            out.append(f"| {plane} | {dd:.4f} | {e1:.1e} | {e2:.1e} | {e3:.1e} | {e4:.1e} | {sym:.1e} | {t90[0] if t90 else '–'} |")


def filt(out):
    d = vc.load_npz("filter_lossywall_wp2.npz")
    out.append("\n#### Lossy-wall filter (WP2 exact LS method), attenuation in dB\n")
    out.append("| M | TE1,0 | TE2,0 | TE3,0 | TE0,1 | TE1,1 | TM1,1 | time (s) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in d["meta"]["rows"]:
        out.append(f"| {r['M']} | {r['TE10']:.4f} | {r['TE20']:.4f} | {r['TE30']:.4f} | {r['TE01']:.4f} | {r['TE11']:.4f} | "
                   f"{r['TM11']:.4f} | {r['seconds']} |")


def evanescent(out):
    p = os.path.join(vc.DATA, "evanescent_bend_miter.npz")
    if not os.path.exists(p):
        return
    d = vc.load_npz(p)
    out.append("\n#### Short bend + miter (WP1 leftover): Line [SmoothBend, Miter] against one FEM domain\n")
    out.append("Max over the propagating GSM of | |S|² difference | (power); 'line' = Line [SmoothBend, fem2d.Miter] at "
               "cutoff_factor 1 / 3; 'FEM cascade' = Line [FEM arc, fem2d.Miter] (evanescent modes coupled by the arc) at "
               "cutoff_factor 1 / 3; 'bend' = SmoothBend alone against the FEM arc alone.\n")
    out.append("| R/a | f (GHz) | L/a | line cf 1 | line cf 3 | Δ TE1,0 line cf 3 (dB) | FEM cascade cf 1 | FEM cascade cf 3 | bend alone | FEM mesh |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in d["meta"]["rows"]:
        g = lambda k: f"{r[k]:.1e}" if k in r else "–"
        out.append(f"| {r['R_over_a']:g} | {r['f'] / 1e9:g} | {r['L_over_a']:.2f} | {g('line_cf1_max_dP')} | {g('line_cf3_max_dP')} | "
                   f"{r['line_cf3_dTE10_dB']:+.4f} | {g('fem_cascade_cf1_max_dP')} | {g('fem_cascade_cf3_max_dP')} | "
                   f"{g('bend_max_dP')} | {r['fem_mesh_change']:.0e} |")


def marcuvitz(out):
    d = vc.load_npz("marcuvitz_524a.npz")
    out.append("\n#### Marcuvitz Sec. 5.24a (symmetric H-plane step) against the WP4 mode-matching Step\n")
    out.append("Normalized reactance Xλg/(2aZ0) and impedance ratio Z0'λg a/(Z0λg'a'): relative error of Marcuvitz's "
               "(2a)/(1a) as printed, with N12 corrected (sign of α²R2), and with the corrected N12 and λg' in the "
               "denominator of (2a). N12 'required' = the value the X11 and the X22 equations need (with the printed "
               "N11, N22) to reproduce mode matching.\n")
    out.append("| a/λ | α | MM X | printed | N12 corr. | N12 corr. + λg' | MM Z-ratio | printed | N12 corr. | N12 printed | "
               "N12 required (X11 / X22 eq.) | N12 corrected |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for j in range(len(d["a_lam"])):
        g = lambda k: d[k][j]
        out.append(f"| {g('a_lam'):.2f} | {g('alpha'):.1f} | {g('mm_xn'):.4f} | {g('mz_xn') / g('mm_xn') - 1:+.3f} | "
                   f"{g('mzc_xn') / g('mm_xn') - 1:+.4f} | {g('mzcp_xn') / g('mm_xn') - 1:+.4f} | {g('mm_zn'):.4f} | "
                   f"{g('mz_zn') / g('mm_zn') - 1:+.4f} | {g('mzc_zn') / g('mm_zn') - 1:+.4f} | {g('N12'):.5f} | "
                   f"{g('N12_req_11'):.5f} / {g('N12_req_22'):.5f} | {g('N12_corrected'):.5f} |")


if __name__ == "__main__":
    out = []
    for fn in (tapers, staircase, twists, square, miters, filt, evanescent, marcuvitz):
        try:
            fn(out)
        except FileNotFoundError as e:
            out.append(f"\n(missing data: {e})\n")
    print("\n".join(out))
