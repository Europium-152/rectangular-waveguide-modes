"""WP6 benchmark: Doane & Anderson (1989) Fig. 9, 20-inch 90 deg twists in tall WR187, 15-18 GHz.

Computes on the grid of the digitized curves (benchmarks/literature/doane1989_fig9.csv), for the three profiles:
  * the full multimode solution with all propagating modes (cutoff_factor 1): total spurious power
    1 - |S21(TE1,0; TE1,0)|^2 (reflection included, as T3), converted power per mode, reflected power;
  * the two-mode solution (basis TE1,0 + TE0,1 only), which is what Fig. 9 resembles;
  * the solution with evanescent modes of all families (cutoff_factor 1.5) on a coarser grid (every 0.25 GHz).
and compares with the digitized curves (z = (ours - Fig. 9) / uncertainty).

Run: PYTHONPATH=src python benchmarks/twist/doane_fig9.py   (about 5 min on a laptop)
Output: benchmarks/twist/data/doane_fig9.npz (+ printed summary used in docs/reports/WP6.md).
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save, REPO  # noqa: E402

from wrmodes.components import Guide, Twist, TwistProfile  # noqa: E402
from wrmodes.components.twist import twist_gsm_blocks  # noqa: E402
from wrmodes.modes import ModeBasis  # noqa: E402

INCH = 0.0254
A, B, L = 0.872 * INCH, 1.872 * INCH, 20 * INCH
GUIDE = Guide(A, B)
PROFILES = ("constant", "cosine", "triangular")
TRACK = ("TE0,1", "TE0,3", "TE2,1", "TM2,1", "TE1,2", "TM1,2", "TE0,5", "TE2,3", "TM2,3")


def fig9():
    path = os.path.join(REPO, "benchmarks", "literature", "doane1989_fig9.csv")
    with open(path) as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    out = {}
    for c in PROFILES:
        r = [x for x in rows if x["curve"] == c]
        out[c] = tuple(np.array([float(x[k]) for x in r]) for k in ("x", "y", "uncertainty"))
    return out


def solve(f, profile, cf):
    port = GUIDE.basis(f, 1.0)
    g = Twist(90, L, profile, guide=GUIDE, cutoff_factor=cf).gsm(f, port)
    j = port.index("TE1,0")
    col = np.abs(g.S21[:, j]) ** 2
    track = [col[port.index(m)] if m in port else np.nan for m in TRACK]
    return 1 - col[j], np.sum(col) - col[j], float(np.sum(np.abs(g.S11[:, j]) ** 2)), track, g.info["n_modes"]


def two_mode(f, profile):
    basis = ModeBasis(A, B, ["TE1,0", "TE0,1"])
    S, _ = twist_gsm_blocks(basis, f, TwistProfile(profile, np.pi / 2, L), L)
    return 1 - abs(S[2][basis.index("TE1,0"), basis.index("TE1,0")]) ** 2


def main():
    data = fig9()
    x = data["constant"][0]
    out = {"freq_ghz": x}
    coarse = np.arange(15.0, 18.0001, 0.25)
    coarse[0], coarse[-1] = 15.025, 17.975
    out["coarse_ghz"] = coarse
    print(f"{'profile':10s} {'f':>6s} | {'full':>7s} {'TE01':>7s} {'2-mode':>7s} {'cf1.5':>7s} | Fig.9 +- u")
    for p in PROFILES:
        tot, conv, refl, trk, two = [], [], [], [], []
        for f in x:
            t, c, r, tr, _ = solve(f * 1e9, p, None)
            tot.append(t); conv.append(c); refl.append(r); trk.append(tr); two.append(two_mode(f * 1e9, p))
        out[f"{p}_total"], out[f"{p}_converted"], out[f"{p}_reflected"] = map(np.array, (tot, conv, refl))
        out[f"{p}_modes"] = np.array(trk)
        out[f"{p}_two_mode"] = np.array(two)
        tc, nc = [], []
        for f in coarse:
            t, _, _, _, n = solve(f * 1e9, p, 1.5)
            tc.append(t); nc.append(n)
        out[f"{p}_total_cf15"], out[f"{p}_n_modes_cf15"] = np.array(tc), np.array(nc)
        xf, yf, uf = data[p]
        out[f"{p}_fig9_x"], out[f"{p}_fig9_y"], out[f"{p}_fig9_u"] = xf, yf, uf
        for fc, tcf in zip(coarse, tc):
            i = int(np.argmin(abs(x - fc)))
            k = int(np.argmin(abs(xf - fc)))
            print(f"{p:10s} {fc:6.3f} | {100 * tot[i]:7.4f} {100 * trk[i][0]:7.4f} {100 * two[i]:7.4f} "
                  f"{100 * tcf:7.4f} | {yf[k]:.4f} +- {uf[k]:.4f}")
    # agreement statistics with the digitized curves (same grid)
    print("\nagreement with Fig. 9 (fraction of the 119 grid points within the digitizing uncertainty; max |z|):")
    for p in PROFILES:
        xf, yf, uf = data[p]
        idx = [int(np.argmin(abs(x - v))) for v in xf]
        for label, key in (("full", f"{p}_total"), ("TE0,1 only", None), ("two-mode", f"{p}_two_mode")):
            ours = 100 * (out[key][idx] if key else out[f"{p}_modes"][idx, 0])
            z = (ours - yf) / uf
            print(f"  {p:10s} {label:11s}: within {np.mean(abs(z) <= 1):5.1%}, max |z| {np.max(abs(z)):6.1f}, "
                  f"rms diff {np.sqrt(np.mean((ours - yf) ** 2)):.4f} %")
            out[f"{p}_z_{label.replace(' ', '_').replace(',', '').replace('-', '_')}"] = z
    m = meta("benchmarks/twist/doane_fig9.py", "Doane & Anderson 1989 Fig. 9: 20-inch 90 deg twists in tall WR187",
             a_m=A, b_m=B, length_m=L, angle_deg=90, input_mode="TE1,0", tracked_modes=list(TRACK),
             cutoff_factor_full=1.0, cutoff_factor_coarse=1.5, evanescent_families="all",
             reference="benchmarks/literature/doane1989_fig9.csv", spurious="1 - |S21(TE1,0; TE1,0)|^2")
    print(save("doane_fig9.npz", m, **out))


if __name__ == "__main__":
    main()
