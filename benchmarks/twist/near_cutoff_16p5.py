"""WP6 benchmark: sensitivity of the Doane twists at 16.5 GHz to the truncation and to the frequency.

TE2,3/TM2,3 of tall WR187 (cutoff 16.5121 GHz) are coupled to TE1,0 at first order and lie 0.07 % above 16.5 GHz.
For the three 20-inch 90 deg twists, at 16.45, 16.5 and 16.55 GHz, and cutoff_factor 1, 1.0001, 1.001, 1.01, 1.1,
1.5, 2, 3 (evanescent modes of all families), the total spurious power and the converted power per mode.

Run: PYTHONPATH=src python benchmarks/twist/near_cutoff_16p5.py   (about 35 min: cutoff_factor 3 with 182 modes
costs about 5 min per variable-rate twist)
Output: benchmarks/twist/data/near_cutoff_16p5.npz (+ printed table for docs/reports/WP6.md)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

from wrmodes.components import Guide, Twist  # noqa: E402

INCH = 0.0254
A, B, L = 0.872 * INCH, 1.872 * INCH, 20 * INCH
GUIDE = Guide(A, B)
FREQS = (16.45e9, 16.5e9, 16.55e9)
CFS = (1.0, 1.0001, 1.001, 1.01, 1.1, 1.5, 2.0, 3.0)
PROFILES = ("constant", "cosine", "triangular")
TRACK = ("TE0,1", "TE0,3", "TE2,1", "TM2,1", "TE1,2", "TM1,2", "TE0,5")


def main():
    tot = np.zeros((len(PROFILES), len(FREQS), len(CFS)))
    modes = np.zeros(tot.shape + (len(TRACK),))
    nmod = np.zeros(tot.shape, int)
    nsl = np.zeros(tot.shape, int)
    for ip, p in enumerate(PROFILES):
        for jf, f in enumerate(FREQS):
            port = GUIDE.basis(f)
            j = port.index("TE1,0")
            for k, cf in enumerate(CFS):
                t0 = time.time()
                g = Twist(90, L, p, guide=GUIDE, cutoff_factor=cf).gsm(f, port)
                col = np.abs(g.S21[:, j]) ** 2
                tot[ip, jf, k] = 1 - col[j]
                modes[ip, jf, k] = [col[port.index(m)] for m in TRACK]
                nmod[ip, jf, k], nsl[ip, jf, k] = g.info["n_modes"], g.info["n_slices"]
                print(f"{p:10s} {f / 1e9:6.2f} GHz cf {cf:<7g} {nmod[ip, jf, k]:4d} modes {nsl[ip, jf, k]:5d} slices: "
                      f"spurious {100 * tot[ip, jf, k]:.4f} %  " +
                      " ".join(f"{m} {100 * v:.4f}" for m, v in zip(TRACK, modes[ip, jf, k])) +
                      f"   ({time.time() - t0:.1f} s)", flush=True)
    m = meta("benchmarks/twist/near_cutoff_16p5.py", "Doane twists near the TE2,3/TM2,3 cutoff (16.512 GHz)",
             a_m=A, b_m=B, length_m=L, angle_deg=90, input_mode="TE1,0", freqs_hz=list(FREQS),
             cutoff_factors=list(CFS), profiles=list(PROFILES), tracked_modes=list(TRACK),
             evanescent_families="all", spurious="1 - |S21(TE1,0; TE1,0)|^2")
    print(save("near_cutoff_16p5.npz", m, total=tot, modes=modes, n_modes=nmod, n_slices=nsl,
               freqs=np.array(FREQS), cutoff_factors=np.array(CFS), profiles=np.array(PROFILES),
               tracked=np.array(TRACK)))


if __name__ == "__main__":
    main()
