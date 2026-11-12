"""WP5 benchmark: the strangled E-plane bend of Butterweck & de Ronde (1968), p. 98, fig. 15 (items P8, P9 of
benchmarks/literature/butterweck1968_numbers.csv).

The paper gives a x b = 22.86 x 10.13 mm, b1 = 2 mm, r = 420 mm, "a design for use at lambda >= 4 mm", and "the
power loss, which is only of the order of a few tenths of a decibel, is mainly due to the wall losses in the flat
guide" (p. 97). It does not give the taper length, the radius of the bend in the flat guide or the taper shape.
Our reading of fig. 15 (drawing and photograph, 5 cm scale): r is the radius of the curved walls of the tapers, which
are circular arcs tangent to the flat guide (a 420 mm bend radius would not fit the 10 cm block of the photograph);
the bend in the flat guide is compact (inner radius of order 1 cm). Two readings of the taper are computed:

* 'asymmetric': the inner wall of the bend straight (as the photograph suggests), the outer wall an arc of radius
  r, height change 8.13 mm on one wall: taper length sqrt(2 r d - d^2) = 82.2 mm; the guide axis moves by 4.065 mm
  (u0(z) = (a(z) - h)/2 in the E-plane frame, inner wall at u = -a/2);
* 'symmetric': both walls arcs of radius r, 4.065 mm each: taper length 58.3 mm (closer to the ~6 cm of the
  photograph).

The flat bend: 90 deg, centre-line radius R_b = 10, 20 or 40 mm. Walls: aluminium (sigma = 3.5e7 S/m, the SPARC
baseline of CONVENTIONS.md) and copper (5.8e7 S/m), plus perfect walls (the mode-conversion part). Frequencies 40,
60 and 75 GHz (lambda = 7.5, 5, 4 mm). Reported: the total TE1,0 loss -20 log10 |S21(TE1,0)| (dB), the PEC loss
(conversion and reflection), and the ohmic part (their difference). Not pass/fail (card WP5).

Run: PYTHONPATH=src python benchmarks/taper/butterweck_fig15.py   (about 15 min)
Output: benchmarks/taper/data/butterweck_fig15.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

from wrmodes import Line  # noqa: E402
from wrmodes.components import Guide, VariableGuide, GeometryFunction  # noqa: E402

W, H, B1, RW = 22.86e-3, 10.13e-3, 2.0e-3, 0.420
DELTA = H - B1
FREQS = (40e9, 60e9, 75e9)
RBS = (0.010, 0.020, 0.040)
WALLS = (("PEC", None), ("aluminium", 3.5e7), ("copper", 5.8e7))


class Fig15:
    """Local E-frame geometry: a(z) (height h -> b1 -> h) and u0(z) of one reading."""

    def __init__(self, reading, Rb):
        self.reading, self.Rb = reading, Rb
        self.d0 = DELTA if reading == "asymmetric" else DELTA / 2      # wall displacement of one arc
        self.Lt = np.sqrt(2 * RW * self.d0 - self.d0 ** 2)
        self.u0f = (B1 - H) / 2 if reading == "asymmetric" else 0.0     # axis offset in the flat guide
        self.Rref = Rb - self.u0f                                       # reference-line radius
        self.arc = self.Rref * np.pi / 2
        self.z1 = self.Lt
        self.z2 = self.z1 + self.arc
        self.L = self.z2 + self.Lt
        self.nw = 1 if reading == "asymmetric" else 2                   # number of curved walls

    def _x(self, z):
        """Distance from the nearest flat end inside a taper (0 in the flat guide), and the sign of dx/dz."""
        z = np.asarray(z, dtype=float)
        x = np.where(z < self.z1, self.Lt - z, np.where(z > self.z2, z - self.z2, 0.0))
        s = np.where(z < self.z1, -1.0, np.where(z > self.z2, 1.0, 0.0))
        return np.clip(x, 0.0, self.Lt), s

    def a(self, z):
        x, _ = self._x(z)
        return B1 + self.nw * (RW - np.sqrt(RW ** 2 - x ** 2))

    def da(self, z):
        x, s = self._x(z)
        return self.nw * s * x / np.sqrt(RW ** 2 - x ** 2)

    def u0(self, z):
        return (self.a(z) - H) / 2 if self.reading == "asymmetric" else 0.0 * np.asarray(z, dtype=float)

    def du0(self, z):
        return self.da(z) / 2 if self.reading == "asymmetric" else 0.0 * np.asarray(z, dtype=float)

    def kappa(self, z):
        z = np.asarray(z, dtype=float)
        return np.where((z > self.z1) & (z < self.z2), 1.0 / self.Rref, 0.0)

    def component(self, guide, **kw):
        return VariableGuide(self.L, GeometryFunction(self.a, self.da), W,
                             kappa=GeometryFunction(self.kappa, lambda z: 0.0 * np.asarray(z, dtype=float)),
                             u0=GeometryFunction(self.u0, self.du0), plane="E", guide=guide,
                             breakpoints=[self.z1, self.z2], uniform=[(self.z1, self.z2)], **kw)


def main():
    out = np.zeros((2, len(RBS), len(FREQS), len(WALLS)))         # TE1,0 loss (dB)
    refl = np.zeros_like(out)
    t0 = time.time()
    for ir, reading in enumerate(("asymmetric", "symmetric")):
        for ib, Rb in enumerate(RBS):
            geo = Fig15(reading, Rb)
            for jf, f in enumerate(FREQS):
                for jw, (wall, sigma) in enumerate(WALLS):
                    g = Guide(W, H, sigma)
                    G = Line(g, [geo.component(g, atol=1e-6)]).gsm(f)
                    j = G.basis1.index("TE1,0")
                    out[ir, ib, jf, jw] = -20 * np.log10(abs(G.S21[G.basis2.index("TE1,0"), j]))
                    refl[ir, ib, jf, jw] = np.sum(np.abs(G.S11[G.basis1.propagating_mask(f), j]) ** 2)
            print(f"{reading:10s} Lt = {geo.Lt * 1e3:.1f} mm, R_b = {Rb * 1e3:.0f} mm, axis length "
                  f"{(2 * geo.Lt + Rb * np.pi / 2) * 1e3:.1f} mm:", flush=True)
            for jf, f in enumerate(FREQS):
                pec, al, cu = out[ir, ib, jf]
                print(f"    {f / 1e9:4.0f} GHz: PEC {pec:.4f} dB (refl {refl[ir, ib, jf, 0]:.1e}), aluminium "
                      f"{al:.4f} dB (ohmic {al - pec:.4f}), copper {cu:.4f} dB (ohmic {cu - pec:.4f})", flush=True)
    print(f"{time.time() - t0:.0f} s")
    m = meta("benchmarks/taper/butterweck_fig15.py", "Butterweck & de Ronde 1968 fig. 15 strangled E-plane bend",
             w=W, h=H, b1=B1, r_wall=RW, readings=["asymmetric", "symmetric"], bend_radii=list(RBS),
             freqs=list(FREQS), walls=[w for w, _ in WALLS], sigmas=[s for _, s in WALLS], atol=1e-6,
             mode_factor=2.0, quantity="TE1,0 loss -20 log10 |S21| (dB), axes (reading, R_b, f, wall)",
             paper="'a few tenths of a decibel, mainly due to the wall losses in the flat guide' (p. 97)")
    save("butterweck_fig15.npz", m, loss_db=out, reflected=refl, f=np.array(FREQS), Rb=np.array(RBS))


if __name__ == "__main__":
    main()
