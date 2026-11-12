"""WP5: fit of the 'bottle' profile (wrmodes.profiles.BottleProfile) to Butterweck & de Ronde (1968), p. 96, fig. 13.

The paper gives no formula for its bottle-shaped taper (140 mm, 3.10 x 1.55 mm -> 22.86 x 10.16 mm, H10 -> H30
below -40 dB at 4 mm wavelength), only the rule "very small slope at the wide end, greater inclination at the narrow
part". We take the one-parameter family p(t) = (1 - exp(-lam t)) / (1 - exp(-lam)) (slope ratio exp(lam) between
the ends) and compute the TE10 -> TE30 conversion |S21(TE3,0 <- TE1,0)|^2 of the width taper a(z) = a0 + (a1 - a0)
p(z/L) for lam = 0 (linear) ... 6 and 65-85 GHz, family n = 0 (TE_m0, m odd; b = 10.16 mm constant: the TE_m0
modes do not depend on b), mode_factor 1.5, atol 1e-5. The lam with the lowest worst-case conversion over the band is
the fitted value (3.5).

Run: PYTHONPATH=src python benchmarks/taper/fit_bottle.py   (about 3 min)
Output: benchmarks/taper/data/fit_bottle.npz
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

from wrmodes.components import Taper, Guide  # noqa: E402
from wrmodes.modes import ModeBasis  # noqa: E402
from wrmodes.profiles import BottleProfile  # noqa: E402

L, A0, A1, B = 0.14, 3.10e-3, 22.86e-3, 10.16e-3
LAMS = (0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)
FREQS = (65e9, 70e9, 75e9, 80e9, 85e9)


def conversion(lam, f, mode_factor=1.5, atol=1e-5):
    b1 = ModeBasis(A0, B, [m for m in ModeBasis.propagating(f, A0, B).modes if m.n == 0 and m.m % 2])
    b2 = ModeBasis(A1, B, [m for m in ModeBasis.propagating(f, A1, B).modes if m.n == 0 and m.m % 2])
    tp = Taper(L, A1, B, BottleProfile(lam), guide=Guide(A0, B), mode_factor=mode_factor, atol=atol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = tp.gsm(f, b1, b2)
    return abs(g.S21[b2.index("TE3,0"), b1.index("TE1,0")]) ** 2


def main():
    C = np.array([[conversion(lam, f) for f in FREQS] for lam in LAMS])
    db = 10 * np.log10(C)
    print("lam  " + "  ".join(f"{f / 1e9:6.0f}" for f in FREQS) + "   worst (dB)")
    for lam, row in zip(LAMS, db):
        print(f"{lam:4.1f} " + "  ".join(f"{x:6.1f}" for x in row) + f"   {row.max():6.1f}")
    best = LAMS[int(np.argmin(db.max(axis=1)))]
    print(f"lowest worst-case conversion at lam = {best}")
    m = meta("benchmarks/taper/fit_bottle.py", "fit of the bottle profile lam (Butterweck & de Ronde 1968 p. 96)",
             L=L, a0=A0, a1=A1, b=B, mode_factor=1.5, atol=1e-5, family="n = 0, m odd")
    save("fit_bottle.npz", m, lam=np.array(LAMS), f=np.array(FREQS), conversion=C, best_lam=best)


if __name__ == "__main__":
    main()
