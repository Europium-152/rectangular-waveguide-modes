"""WP6 benchmark: predictions for COMSOL campaign 2, case E4 (benchmarks/comsol/CAMPAIGN_2.md).

20-inch (508 mm) 90 deg twist in tall WR187 (a = 22.149 mm along u, b = 47.549 mm), incident TE1,0 (E along the long
side), triangular rate at 15 and 18 GHz, cosine rate at 18 GHz (optional control). Output ports of E4 (T3 names):
TE1,0, TE0,1, TE0,3, TE1,2, TM1,2, TE2,1, TM2,1. For each: |S21| (and |S21|^2), with cutoff_factor 1 (propagating
modes) and 1.5 (evanescent modes of all families), the reflected power, and the power in the propagating modes
that E4 does not port.

Magnitudes do not depend on the reference planes, so the 15 mm straight legs of the COMSOL model change nothing.

Run: PYTHONPATH=src python benchmarks/twist/e4_predictions.py   (about 1 min)
Output: benchmarks/twist/data/e4_predictions.npz (+ printed table for docs/reports/WP6.md)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

from wrmodes.components import Guide, Twist  # noqa: E402

INCH = 0.0254
A, B, L = 0.872 * INCH, 1.872 * INCH, 20 * INCH
GUIDE = Guide(A, B)
PORTS = ("TE1,0", "TE0,1", "TE0,3", "TE1,2", "TM1,2", "TE2,1", "TM2,1")
CASES = (("triangular", 15e9), ("triangular", 18e9), ("cosine", 18e9))
CFS = (1.0, 1.5)


def main():
    out = {"ports": np.array(PORTS), "cases": np.array([f"{p} {f / 1e9:g} GHz" for p, f in CASES]),
           "cutoff_factors": np.array(CFS)}
    S = np.zeros((len(CASES), len(CFS), len(PORTS)))
    refl = np.zeros((len(CASES), len(CFS)))
    other = np.zeros((len(CASES), len(CFS)))
    other_names = []
    for ic, (p, f) in enumerate(CASES):
        port = GUIDE.basis(f)
        j = port.index("TE1,0")
        for k, cf in enumerate(CFS):
            g = Twist(90, L, p, guide=GUIDE, cutoff_factor=cf).gsm(f, port)
            col = np.abs(g.S21[:, j])
            S[ic, k] = [col[port.index(m)] for m in PORTS]
            refl[ic, k] = np.sum(np.abs(g.S11[:, j]) ** 2)
            rest = [i for i, m in enumerate(port.modes) if m.name not in PORTS]
            other[ic, k] = np.sum(col[rest] ** 2)
            if k == 0:
                big = sorted(((col[i] ** 2, port.modes[i].name) for i in rest), reverse=True)[:3]
                other_names.append("; ".join(f"{n} {100 * v:.4f} %" for v, n in big))
    out.update(abs_S21=S, power_S21=S ** 2, reflected=refl, unported=other, unported_largest=np.array(other_names))
    print("|S21| (power in %), incident TE1,0; columns: cutoff_factor 1 | 1.5")
    print("| mode | " + " | ".join(f"{p} {f / 1e9:g} GHz" for p, f in CASES) + " |")
    print("| --- |" + " --- |" * len(CASES))
    for im, m in enumerate(PORTS):
        cells = []
        for ic in range(len(CASES)):
            a1, a2 = S[ic, 0, im], S[ic, 1, im]
            cells.append(f"{a1:.5f} ({100 * a1 ** 2:.4f} %) / {a2:.5f} ({100 * a2 ** 2:.4f} %)")
        print(f"| {m} | " + " | ".join(cells) + " |")
    print("| reflected (all modes) | " + " | ".join(f"{100 * refl[ic, 0]:.1e} % / {100 * refl[ic, 1]:.1e} %"
                                                     for ic in range(len(CASES))) + " |")
    print("| not ported (other propagating modes) | " + " | ".join(
        f"{100 * other[ic, 0]:.4f} % / {100 * other[ic, 1]:.4f} %" for ic in range(len(CASES))) + " |")
    for (p, f), names in zip(CASES, other_names):
        print(f"  largest unported, {p} {f / 1e9:g} GHz: {names}")
    m = meta("benchmarks/twist/e4_predictions.py", "WP6 predictions for COMSOL campaign 2 case E4",
             a_m=A, b_m=B, length_m=L, angle_deg=90, input_mode="TE1,0", ports=list(PORTS),
             cases=[[p, f] for p, f in CASES], cutoff_factors=list(CFS), evanescent_families="all")
    print(save("e4_predictions.npz", m, **out))


if __name__ == "__main__":
    main()
