"""WF4 item 7: Doane & Anderson (1989) Fig. 9 with Doane's five modes.

Doane (IEEE Trans. MTT 36, 1033, 1988; Table I) computed the twists of Fig. 9 with the modes TE10, TE01, TE30, TE21
and TM21 in his labels; his first index runs along the broad side, so in wrmodes' tall WR187 (a = 0.872 in along u,
b = 1.872 in along v) they are TE1,0, TE0,1, TE0,3, TE1,2 and TM1,2. This script solves the three 20-inch 90 degree
twists (constant, cosine, triangular rate) on the 119-point grid of benchmarks/literature/doane1989_fig9.csv

  * restricted to these five modes ("five"), and
  * to these five plus wrmodes' TE2,1 (Doane's TE12, the mode of the constant twist's lobes; "five+TE2,1"),

with the plain truncated coupled-mode system on that basis (`twist_gsm_blocks`, no internal list, no tail: the
model of a five-mode calculation), and compares the total spurious power 1 - |S21(TE1,0; TE1,0)|^2 with the
digitized curves, z = (ours - Fig. 9) / uncertainty. For reference it also stores the full solution at the defaults
(benchmarks/twist/doane_fig9.py stores WP6's model).

Run: PYTHONPATH=src python benchmarks/twist/doane_fig9_five_modes.py   (about 2 min, of which the full solution at
the triangular and cosine defaults on a coarse grid is most)
Output: benchmarks/twist/data/doane_fig9_five_modes.npz and the printed summary of docs/reports/WF4.md.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402
from doane_fig9 import fig9, A, B, L, GUIDE, PROFILES  # noqa: E402

from wrmodes.components import Twist, TwistProfile  # noqa: E402
from wrmodes.components.twist import twist_gsm_blocks  # noqa: E402
from wrmodes.modes import ModeBasis  # noqa: E402

FIVE = ["TE1,0", "TE0,1", "TE0,3", "TE1,2", "TM1,2"]
SIX = FIVE + ["TE2,1"]
CHECK_GHZ = (15.0, 16.0, 16.5, 17.0, 18.0)


def spurious(modes, f, profile):
    basis = ModeBasis(A, B, modes)
    S, _ = twist_gsm_blocks(basis, f, TwistProfile(profile, np.pi / 2, L), L)
    i = basis.index("TE1,0")
    return 1 - abs(S[2][i, i]) ** 2


def main():
    data = fig9()
    out = {"five_modes": np.array(FIVE), "six_modes": np.array(SIX)}
    print("agreement with Fig. 9 (119-point grid): fraction within 1 sigma, rms z, max |z|")
    summary = {}
    for p in PROFILES:
        x, y, u = data[p]
        for tag, modes in (("five", FIVE), ("six", SIX)):
            val = 100 * np.array([spurious(modes, fx * 1e9, p) for fx in x])
            z = (val - y) / u
            out[f"{p}_{tag}"], out[f"{p}_{tag}_z"] = val, z
            summary[(p, tag)] = (float(np.mean(abs(z) <= 1)), float(np.sqrt(np.mean(z ** 2))), float(np.max(abs(z))))
            print(f"  {p:10s} {tag:5s}: within {summary[(p, tag)][0]:5.1%}, rms z {summary[(p, tag)][1]:6.2f}, "
                  f"max |z| {summary[(p, tag)][2]:6.1f}   (n = {len(x)})")
        out[f"{p}_fig9_x"], out[f"{p}_fig9_y"], out[f"{p}_fig9_u"] = x, y, u
    print("\nspurious power (%) at the coordinator's check frequencies (Fig. 9 value of the nearest grid point):")
    print(f"{'profile':10s} {'f':>5s} {'five':>7s} {'five+TE21':>9s} {'full':>7s} {'Fig. 9':>7s}")
    full = {}
    for p in PROFILES:
        x, y, u = data[p]
        rows = []
        for fc in CHECK_GHZ:
            k = int(np.argmin(abs(x - fc)))
            port = GUIDE.basis(x[k] * 1e9, 1.0)
            g = Twist(90, L, p, guide=GUIDE).gsm(x[k] * 1e9, port)
            j = port.index("TE1,0")
            fu = 100 * (1 - abs(g.S21[j, j]) ** 2)
            rows.append((x[k], out[f"{p}_five"][k], out[f"{p}_six"][k], fu, y[k]))
            print(f"{p:10s} {x[k]:5.2f} {rows[-1][1]:7.3f} {rows[-1][2]:9.3f} {fu:7.3f} {y[k]:7.3f}")
        full[p] = rows
        out[f"{p}_check"] = np.array(rows)
    m = meta("benchmarks/twist/doane_fig9_five_modes.py",
             "Doane & Anderson 1989 Fig. 9 with Doane's five modes (and + TE2,1), 20-inch 90 deg twists in tall WR187",
             a_m=A, b_m=B, length_m=L, angle_deg=90, input_mode="TE1,0", five_modes=FIVE, six_modes=SIX,
             model="twist_gsm_blocks on the given basis (plain truncated system, no internal list, no tail)",
             reference="benchmarks/literature/doane1989_fig9.csv", spurious="1 - |S21(TE1,0; TE1,0)|^2 in %",
             check_columns=["f_GHz", "five", "five+TE2,1", "full (Twist defaults)", "Fig. 9"],
             summary={f"{p}_{t}": dict(within_1_sigma=v[0], rms_z=v[1], max_abs_z=v[2]) for (p, t), v in summary.items()},
             doane_labels={"TE10": "TE1,0", "TE01": "TE0,1", "TE30": "TE0,3", "TE21": "TE1,2", "TM21": "TM1,2",
                           "TE12": "TE2,1"})
    print(save("doane_fig9_five_modes.npz", m, **out))


if __name__ == "__main__":
    main()
