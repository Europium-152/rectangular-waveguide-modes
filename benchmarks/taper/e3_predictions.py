"""WP5 benchmark: predictions for COMSOL campaign 2, case E3 (benchmarks/comsol/CAMPAIGN_2.md).

Strangled E-plane bend in WR-90 (22.86 x 10.16 mm): raised-cosine tapers of 50 mm from h = 10.16 mm to 1.6 mm,
a 90 deg E-plane arc of centre-line radius 50 mm in the flat guide, no flat straight, the mirror taper back:
`StrangledBend(90, 0.05, 1.6e-3, 0.05, 'raised_cosine')`. Reference planes: the full-height ends of the tapers
(q = -50 mm and p = 100 mm of the COMSOL model). 18 and 40 GHz, PEC and aluminium (sigma = 3.774e7 S/m, COMSOL's
library value). TE1,0 incident at port 1, in the global frame (a Line inserts the H -> E and E -> H frame changes;
global mode names, E across the 10.16 mm side for TE1,0).

Reported per case: S11(TE1,0), S21(TE1,0) (complex, dB), the reflected power in each propagating mode, the power
transmitted into each converted propagating mode, and the convergence: mode_factor 2 (default) / 3 / 4, B5' on/off,
slice refinement (slice_factor x2 of the adaptive result at mode_factor 2), atol 1e-7.

Sign of the curvature: kappa > 0 (the guide turns towards -y in the local E frame). The magnitudes do not depend on
it. Phases refer to the reference planes; COMSOL's ports are 15 mm beyond them and are de-embedded by WP7b.

Run: PYTHONPATH=src python benchmarks/taper/e3_predictions.py   (about 5 min)
Output: benchmarks/taper/data/e3_predictions.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

from wrmodes import Line  # noqa: E402
from wrmodes.components import Guide, StrangledBend  # noqa: E402

W, H, B_FLAT, LT, R = 22.86e-3, 10.16e-3, 1.6e-3, 0.05, 0.05
SIGMA_AL = 3.774e7
FREQS = (18e9, 40e9)
WALLS = (("PEC", None), ("aluminium", SIGMA_AL))
SETTINGS = (("default (mf 2, B5')", dict(mode_factor=2.0)), ("mf 3, B5'", dict(mode_factor=3.0)),
            ("mf 4, B5'", dict(mode_factor=4.0)), ("mf 2, no B5'", dict(mode_factor=2.0, tail=False)),
            ("mf 4, no B5'", dict(mode_factor=4.0, tail=False)))


def solve(f, sigma, slice_factor=None, **kw):
    g = Guide(W, H, sigma)
    sb = StrangledBend(90, R, B_FLAT, LT, "raised_cosine", guide=g, atol=1e-7, **kw)
    if slice_factor is not None:            # fixed slicing: call the component directly in the E frame
        G = sb.gsm(f, slice_factor=slice_factor)
        return G, sb
    ln = Line(g, [sb])
    return ln.gsm(f), sb


def column(G, f):
    b1, b2 = G.basis1, G.basis2
    j = b1.index("TE1,0")
    p1, p2 = b1.propagating_mask(f), b2.propagating_mask(f)
    refl = {m.name: abs(G.S11[i, j]) ** 2 for i, m in enumerate(b1.modes) if p1[i]}
    trans = {m.name: abs(G.S21[i, j]) ** 2 for i, m in enumerate(b2.modes) if p2[i]}
    return G.S11[j, j], G.S21[b2.index("TE1,0"), j], refl, trans


def main():
    rows, out = [], {}
    names = None
    for f in FREQS:
        for wall, sigma in WALLS:
            res = []
            for label, kw in SETTINGS:
                t0 = time.time()
                G, sb = solve(f, sigma, **kw)
                dt = time.time() - t0
                s11, s21, refl, trans = column(G, f)
                res.append((label, s11, s21, refl, trans, dt, G))
            key = f"{f / 1e9:g}GHz_{wall}"
            best = res[2]
            # slice refinement at the default mode count, in the E frame (same physical quantities)
            Gd = res[0][6]
            sbd = StrangledBend(90, R, B_FLAT, LT, "raised_cosine", guide=Guide(W, H, sigma), atol=1e-7)
            Ge = sbd.gsm(f)
            sf = max(Ge.info["slice_factor"].values())
            Gr = sbd.gsm(f, slice_factor=2 * sf)
            je = Ge.basis1.index("TE0,1")
            dslice = max(np.max(np.abs(Gr.S11[:, je] - Ge.S11[:, je])), np.max(np.abs(Gr.S21[:, je] - Ge.S21[:, je])))
            out[key] = dict(res=res, dslice=dslice)
            label, s11, s21, refl, trans, dt, G = best
            print(f"\nE3 {f / 1e9:g} GHz, {wall}: (best: {label}; {len(G.basis1)} port modes, {dt:.0f} s)")
            print(f"  S11(TE10) = {s11.real:+.5f}{s11.imag:+.5f}i ({20 * np.log10(abs(s11)):.2f} dB), "
                  f"S21(TE10) = {s21.real:+.6f}{s21.imag:+.6f}i ({20 * np.log10(abs(s21)):.4f} dB)")
            print("  reflected: " + ", ".join(f"{m} {p:.3e}" for m, p in sorted(refl.items(), key=lambda x: -x[1])
                                               if p > 1e-7))
            print("  converted: " + ", ".join(f"{m} {p:.3e}" for m, p in sorted(trans.items(), key=lambda x: -x[1])
                                               if m != "TE1,0" and p > 1e-7))
            print("  convergence (dB of S21 TE10, |S11|^2 total refl, total converted; time):")
            for label, s11_, s21_, refl_, trans_, dt_, G_ in res:
                conv = sum(p for m, p in trans_.items() if m != "TE1,0")
                print(f"    {label:22s} {20 * np.log10(abs(s21_)):+.5f} dB  refl {sum(refl_.values()):.4e}  "
                      f"conv {conv:.4e}  |dS21| {abs(s21_ - s21):.1e}  {dt_:.1f} s")
            print(f"    slice refinement (x2 at mf 2): max change of the TE1,0 column {dslice:.1e}")
    # save
    keys = list(out)
    arrays = {}
    for key in keys:
        res = out[key]["res"]
        b1, b2 = res[0][6].basis1, res[0][6].basis2
        modes2 = [m.name for m in b2.modes]
        modes1 = [m.name for m in b1.modes]
        arrays[f"{key}_modes1"] = np.array(modes1)
        arrays[f"{key}_modes2"] = np.array(modes2)
        arrays[f"{key}_settings"] = np.array([r[0] for r in res])
        j = b1.index("TE1,0")
        arrays[f"{key}_S11_col"] = np.array([r[6].S11[:, j] for r in res])
        arrays[f"{key}_S21_col"] = np.array([r[6].S21[:, j] for r in res])
        arrays[f"{key}_time_s"] = np.array([r[5] for r in res])
        arrays[f"{key}_dslice"] = np.array(out[key]["dslice"])
    m = meta("benchmarks/taper/e3_predictions.py", "COMSOL campaign 2, case E3: strangled E-plane bend in WR-90",
             w=W, h=H, b_flat=B_FLAT, taper_length=LT, radius=R, angle_deg=90, profile="raised_cosine",
             flat_length=0.0, sigma_aluminium=SIGMA_AL, freqs=list(FREQS), atol=1e-7,
             reference_planes="full-height ends of the tapers (COMSOL q = -50 mm, p = 100 mm)",
             frame="global H frame of the Line (mode names (M, N): M along w = 22.86 mm)",
             columns="TE1,0 column of S11 and S21 at port bases modes1 / modes2, one row per setting",
             best_setting="mf 4, B5'")
    save("e3_predictions.npz", m, cases=np.array(keys), **arrays)


if __name__ == "__main__":
    main()
