"""The strangled E-plane bend (Butterweck & de Ronde 1968, fig. 15) and tapers, with wrmodes.components (WP5).

1. A 90 deg E-plane bend of WR-90 strangled to a 1.6 mm flat guide by raised-cosine tapers of 50 mm (the SPARC
   candidate, COMSOL campaign 2 case E3), in a Line with aluminium walls: TE1,0 loss, reflection and the converted
   modes, against a plain 90 deg E-plane SmoothBend of the same centre-line radius in full-height guide.
2. A taper in a Line: WR-90 widened to 30 mm and back (the guide of the line changes in between).
3. The convergence report of a taper (`convergence_study`): mode count with and without B5', slices.

Run: python examples/strangled_bend.py      (about 30 s)
"""
import numpy as np

import wrmodes as wr
from wrmodes.components import Guide, StrangledBend, Taper, SmoothBend, Straight

AL = 3.5e7                                       # aluminium baseline (CONVENTIONS.md section 1)
wr90 = Guide(22.86e-3, 10.16e-3, sigma=AL)

print("1. Strangled 90 deg E-plane bend, R = 50 mm, flat height 1.6 mm, 50 mm raised-cosine tapers, aluminium")
strangled = wr.Line(wr90, [StrangledBend(90, 0.05, 1.6e-3, 0.05)])
plain = wr.Line(wr90, [SmoothBend(0.05 * np.pi / 2, 1 / 0.05, plane="E")])
print(strangled.describe())
for f in (18e9, 40e9, 90e9):
    for name, line in (("strangled", strangled), ("plain E bend", plain)):
        g = line.gsm(f)
        j = g.basis1.index("TE1,0")
        p = np.abs(g.S21[:, j]) ** 2
        r = np.sum(np.abs(g.S11[:, j]) ** 2)
        top = sorted(((p[i], m.name) for i, m in enumerate(g.basis2) if m.name != "TE1,0"), reverse=True)[:2]
        print(f"   {f / 1e9:4.0f} GHz {name:13s}: TE1,0 loss {g.loss_db('TE1,0'):.4f} dB, reflected {r:.2e}, "
              + ", ".join(f"{n} {v:.2e}" for v, n in top))

print("\n2. Taper WR-90 -> 30 mm wide -> WR-90 (raised cosine, 0.1 m each) with 0.2 m of wide guide, 30 GHz")
line = wr.Line(wr90, [Taper(0.1, 30e-3, 10.16e-3, "raised_cosine"), Straight(0.2),
                      Taper(0.1, 22.86e-3, 10.16e-3, "raised_cosine")])
print(line.describe())
g = line.gsm(30e9)
print(f"   TE1,0 loss {g.loss_db('TE1,0'):.4f} dB, passivity margin {g.passivity_margin():.2e}")

print("\n3. Convergence of a linear H-plane taper 22.86 -> 45 mm, L = 0.1 m, 15 GHz (TE3,0 passes through cutoff)")
taper = Taper(0.1, 45e-3, 10.16e-3, "linear", guide=Guide(22.86e-3, 10.16e-3))
rep = taper.convergence_study(15e9, "TE1,0")
for row in rep["column"]:
    label = (f"mode_factor {row['mode_factor']:g}, B5' {'on ' if row['tail'] else 'off'}"
             + ("" if row["slice_factor"] is None else f", slices x{row['slice_factor']:.3g}"))
    print(f"   {label:38s}: S21(TE3,0 <- TE1,0) = {row['S21']['TE3,0']:.5f}, S11 = {row['S11']:.5f}")
