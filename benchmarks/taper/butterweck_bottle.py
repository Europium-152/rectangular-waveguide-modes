"""WP5 benchmark: Butterweck & de Ronde (1968) p. 96, items P5 and P6 of benchmarks/literature/butterweck1968_numbers.csv.

A 140 mm taper from standard 4-mm guide (3.10 x 1.55 mm) to standard 3-cm guide (22.86 x 10.16 mm), height b(z)
linear, width a(z) either bottle-shaped (P5: measured H10 -> H30 conversion below -40 dB) or linear (P6: about
-18 dB measured), at 4 mm wavelength (75 GHz). The bottle is our fitted form (wrmodes.profiles.BottleProfile,
lam = 3.5; the paper gives no formula; fit in benchmarks/taper/fit_bottle.py). Both tapers are centred (symmetric),
so TE1,0 only reaches TE_mn with m odd, n even (T2 section 5); the component is solved in that class with every
port mode of it. Reported: |S21(TE3,0 <- TE1,0)|^2 in dB, the other converted modes, the TE1,0 reflection, for
mode_factor 1.5 and 2 with B5' and 1.5 without, and the n = 0-only model (b constant) of the fit.

Run: PYTHONPATH=src python benchmarks/taper/butterweck_bottle.py   (about 20 min)
Output: benchmarks/taper/data/butterweck_bottle.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

from wrmodes.components import VariableGuide, GeometryFunction  # noqa: E402
from wrmodes.modes import ModeBasis  # noqa: E402
from wrmodes.profiles import BottleProfile, LinearProfile  # noqa: E402

F, L = 75e9, 0.14
A0, A1, B0, B1 = 3.10e-3, 22.86e-3, 1.55e-3, 10.16e-3
SETTINGS = ((1.5, True), (2.0, True), (1.5, False))


def ramp(x0, x1, p):
    return GeometryFunction(lambda z: x0 + (x1 - x0) * p(z / L), lambda z: (x1 - x0) * p.derivative(z / L) / L)


def taper(profile, mode_factor, tail, b_linear=True):
    b = ramp(B0, B1, LinearProfile()) if b_linear else B1
    return VariableGuide(L, ramp(A0, A1, profile), b, mode_factor=mode_factor, tail=tail, atol=1e-6)


def port_bases(b_linear=True):
    sel = (lambda m: m.m % 2 == 1 and m.n % 2 == 0) if b_linear else (lambda m: m.m % 2 == 1 and m.n == 0)
    b1 = ModeBasis(A0, B0 if b_linear else B1, [m for m in ModeBasis.propagating(F, A0, B0 if b_linear else B1).modes
                                                if sel(m)])
    b2 = ModeBasis(A1, B1, [m for m in ModeBasis.propagating(F, A1, B1).modes if sel(m)])
    return b1, b2


def main():
    profiles = (("linear", LinearProfile()), ("bottle", BottleProfile()))
    rows = []
    for name, p in profiles:
        for b_lin in (True, False):
            for mf, tail in (SETTINGS if b_lin else ((2.0, True),)):
                b1, b2 = port_bases(b_lin)
                t0 = time.time()
                g = taper(p, mf, tail, b_lin).gsm(F, b1, b2)
                dt = time.time() - t0
                j = b1.index("TE1,0")
                col = np.abs(g.S21[:, j]) ** 2
                c30 = col[b2.index("TE3,0")]
                others = sorted(((c, m.name) for m, c in zip(b2.modes, col) if m.name not in ("TE1,0", "TE3,0")),
                                reverse=True)[:4]
                rows.append(dict(profile=name, b_linear=b_lin, mode_factor=mf, tail=tail, te30=c30,
                                 te10=col[b2.index("TE1,0")], refl=float(np.sum(np.abs(g.S11[:, j]) ** 2)),
                                 others=others, n_modes=list(g.info["n_modes"].values())[0], time=dt,
                                 passivity=g.unitarity_defect()))
                r = rows[-1]
                print(f"{name:7s} b {'linear' if b_lin else 'const '} mf {mf} B5' {'on ' if tail else 'off'}: "
                      f"TE3,0 {10 * np.log10(c30):6.2f} dB, TE1,0 {r['te10']:.5f}, refl {r['refl']:.2e}, "
                      f"others " + ", ".join(f"{m} {10 * np.log10(c):.1f} dB" for c, m in others)
                      + f"  ({r['n_modes']} modes, {dt:.0f} s, unitarity {r['passivity']:.0e})", flush=True)
    m = meta("benchmarks/taper/butterweck_bottle.py", "Butterweck & de Ronde 1968 p. 96 (P5, P6): bottle vs linear",
             f=F, L=L, a0=A0, a1=A1, b0=B0, b1=B1, bottle="BottleProfile(lam=3.5)", atol=1e-6,
             measured="P5: H10->H30 < -40 dB (bottle); P6: about -18 dB (linear); no uncertainty stated")
    save("butterweck_bottle.npz", m, profile=np.array([r["profile"] for r in rows]),
         b_linear=np.array([r["b_linear"] for r in rows]), mode_factor=np.array([r["mode_factor"] for r in rows]),
         tail=np.array([r["tail"] for r in rows]), te30=np.array([r["te30"] for r in rows]),
         te10=np.array([r["te10"] for r in rows]), refl=np.array([r["refl"] for r in rows]),
         n_modes=np.array([r["n_modes"] for r in rows]), time_s=np.array([r["time"] for r in rows]))


if __name__ == "__main__":
    main()
