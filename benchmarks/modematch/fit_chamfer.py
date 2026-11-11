"""Best-fit miter position for the chamfered H-plane literature curves (WP4b, discrepancy report
docs/discrepancies/2026-11-11-alimenti-chamfered-miters.md), and the reflection zeros of MiterMM's 2D solver.

For each curve the cut t/a (d/a = (1 + t/a)/sqrt(2); Ma's c/a = 1 - t/a) minimizing the rms of z = (ours - digitized)
/ uncertainty is found. Usage: python benchmarks/modematch/fit_chamfer.py  (writes data/fit_chamfer.npz)
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from wrmodes.modematch._corner2d import miter_2d                    # noqa: E402
from miter_literature import load_csv                               # noqa: E402
from _meta import meta, save                                        # noqa: E402

CASES = [("alimenti1994_fig4_t030.csv", "t/a=0.30", 0.30), ("alimenti1994_fig4_t034.csv", "t/a=0.34", 0.34),
         ("alimenti1994_fig4_t038.csv", "t/a=0.38", 0.38), ("alimenti1994_fig6_mmfd.csv", "MMFD", 0.386),
         ("alimenti1994_fig6_bcmm.csv", "BCMM", 0.386), ("ma1997_fig11.csv", "c/a=0.70", 0.30),
         ("ma1997_fig11.csv", "c/a=0.631", 0.369), ("ma1997_fig11.csv", "c/a=0.60", 0.40)]
T_ZEROS = [0.30, 0.32, 0.34, 0.36, 0.369, 0.38, 0.386, 0.40]


def s11_db(t, x, M=60):
    S11, _ = miter_2d(1.0, t, "D", (np.pi * x) ** 2, M, int(np.ceil(M / 1.15)))
    return 20 * np.log10(abs(S11[0, 0]))


def zeros(t):
    xs = np.linspace(1.01, 1.99, 197)
    v = np.array([s11_db(t, x) for x in xs])
    out = []
    for i in range(1, len(xs) - 1):
        if v[i] < v[i - 1] and v[i] < v[i + 1] and v[i] < -40:
            r = minimize_scalar(lambda x: s11_db(t, x), bracket=(xs[i - 1], xs[i], xs[i + 1]), tol=1e-7)
            out.append(r.x)
    return out


def main():
    fits = []
    for fn, cur, t in CASES:
        x, y, u = load_csv(fn)[cur][::2, :3].T
        chi = lambda te: np.sqrt(np.mean(((np.array([s11_db(te, xi) for xi in x]) - y) / u) ** 2))
        r = minimize_scalar(chi, bounds=(t - 0.03, t + 0.03), method="bounded", options={"xatol": 2e-4})
        fits.append((t, chi(t), r.x, r.fun))
        print(f"{fn} {cur}: nominal t/a {t:.3f} rms z {fits[-1][1]:.2f}; best t/a {r.x:.4f} rms z {r.fun:.2f}",
              flush=True)
    zs = []
    for t in T_ZEROS:
        z = zeros(t)
        zs.append(z + [np.nan] * (3 - len(z)))
        print(f"t/a = {t:.3f} (d/a = {(1 + t) / np.sqrt(2):.4f}): zeros at f/fc = {', '.join(f'{v:.4f}' for v in z)}",
              flush=True)
    m = meta("benchmarks/modematch/fit_chamfer.py", "best-fit t/a of chamfered H-plane curves; zeros of |S11|",
             cases=[c[:2] for c in CASES], t_zeros=T_ZEROS,
             layout="fits[i] = (nominal t, rms z nominal, best t, rms z best); zeros[j] = f/fc of |S11| minima < -40 dB")
    save("fit_chamfer.npz", m, fits=np.array(fits), zeros=np.array(zs))


if __name__ == "__main__":
    main()
