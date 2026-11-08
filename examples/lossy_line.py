"""A 20 m aluminium WR-90 line with three bends (wrmodes 1.0, WP2: ohmic wall loss).

Line (global H frame, TE1,0 input): 6.2 m straight, 40 cm 90 deg sech H-plane bend, 6.2 m straight, 40 cm 90 deg
sech E-plane bend, 6.2 m straight, 60 cm 60 deg constant-curvature H-plane bend: 20.0 m of centre line. Walls of
aluminium, sigma = 3.5e7 S/m (SPARC baseline, CONVENTIONS.md section 1; there is no default material).

For each frequency the script prints the TE1,0 transmission loss of the lossless and the lossy line, the ohmic
part, the largest converted powers, the reflected power and the computing time.

Run:  python examples/lossy_line.py [f_GHz ...]        (default 18 54 90)
"""
import sys
import time

import numpy as np

import wrmodes as wr
from wrmodes.components import Guide, Straight, SmoothBend, SechCurvature, ConstantCurvature
from wrmodes.loss import attenuation_constant

W, H = 22.86e-3, 10.16e-3
SIGMA_AL = 3.5e7


def components():
    return [Straight(6.2), SmoothBend(0.4, SechCurvature(90, 0.4), plane="H"), Straight(6.2),
            SmoothBend(0.4, SechCurvature(90, 0.4), plane="E"), Straight(6.2),
            SmoothBend(0.6, ConstantCurvature.from_angle(60, 0.6), plane="H")]


def main(freqs_ghz):
    lossy = wr.Line(Guide(W, H, sigma=SIGMA_AL), components(), name="20 m aluminium")
    ideal = wr.Line(Guide(W, H), components(), name="20 m perfect walls")
    print(lossy.describe())
    for fg in freqs_ghz:
        f = fg * 1e9
        t0 = time.perf_counter()
        g0 = ideal.gsm(f)
        t1 = time.perf_counter()
        g = lossy.gsm(f)
        t2 = time.perf_counter()
        b = g.basis1
        j = b.index("TE1,0")
        col = g.S21[:, j]
        loss0, loss1 = g0.loss_db("TE1,0"), g.loss_db("TE1,0")
        straight_db = 20 / np.log(10) * attenuation_constant("TE1,0", W, H, f, SIGMA_AL) * 20.0
        conv = sorted(((abs(col[i]) ** 2, m.name) for i, m in enumerate(b.modes) if i != j), reverse=True)[:3]
        refl = np.sum(np.abs(g.S11[:, j]) ** 2)
        diss = 1 - np.sum(np.abs(col) ** 2) - refl
        print(f"\n{fg:g} GHz: {len(b)} propagating modes")
        print(f"  TE1,0 loss: perfect walls {loss0:.4f} dB, aluminium {loss1:.4f} dB "
              f"(alpha L of TE1,0 alone over 20 m: {straight_db:.4f} dB)")
        print(f"  dissipated {diss:.4e}, reflected {refl:.2e}; largest converted powers: "
              + ", ".join(f"{n} {p:.2e}" for p, n in conv))
        print(f"  time: perfect walls {t1 - t0:.1f} s, aluminium {t2 - t1:.1f} s")


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or [18, 54, 90])
