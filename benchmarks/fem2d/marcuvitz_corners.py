"""Sharp (unmitered) corners of arbitrary angle: FEM against Marcuvitz (1951), Secs. 5.28b and 5.29b (WP3b).

Used to settle the Reiter & Arndt (1995) Fig. 3(a), (c) outliers (docs/reports/WP3b.md):

* H plane, Sec. 5.29b: measured T-network parameters (series jX_b in each arm, shunt -jX_a) at the reference planes
  through the inner corner, WR-90 (a = 0.900 in, b = 0.400 in), lambda = 3.0, 3.2, 3.4 cm, theta = 30-90 deg
  (transcription: docs/literature/marcuvitz-steps-and-bends.md). S11 = (Gamma_e + Gamma_o) / 2 with
  z_e = j(x_b - 2 x_a), z_o = j x_b.
* E plane, Sec. 5.28b: the static pi network (shunt jB_a at each plane, series -jB_b), valid to a few per cent for
  2b/lambda_g < 0.51, here in Reiter's WR62 at 12.05 and 13.05 GHz, together with Reiter's Fig. 3(c) curves.

The FEM (default settings) is the 90-degree-and-other-angle `Miter` with chamfer >= 1/cos(theta/2) (sharp corner).
Usage: python benchmarks/fem2d/marcuvitz_corners.py  -> data/marcuvitz_corners.npz
"""
import numpy as np
import scipy.constants as cst
from scipy.special import digamma

from _lit import te10, load_csv, meta, save

H_TABLE = {30: {3.0: (0.849, 0.493), 3.2: (0.782, 0.420), 3.4: (0.718, 0.365)}, 45: {3.4: (0.995, 0.567)},
           50: {3.0: (1.102, 0.943), 3.4: (1.068, 0.650)},
           60: {3.0: (1.088, 1.27), 3.2: (1.158, 0.998), 3.4: (1.176, 0.817)}, 75: {3.0: (0.864, 2.14)},
           90: {3.0: (0.324, 6.46), 3.2: (0.739, 2.75), 3.4: (1.055, 1.76)}}


def sharp(theta):
    return 1.0 / np.cos(np.radians(theta) / 2)


def main():
    a, b = 0.9 * 0.0254, 0.4 * 0.0254
    rows_h = []
    print("H plane (Marcuvitz 5.29b, measured): theta | lambda (cm) | Marcuvitz S11 | FEM S11 | |dS11| | dB M / FEM")
    for th, d in H_TABLE.items():
        for lam, (ya, xb) in d.items():
            f = cst.c / (lam * 1e-2)
            xa = 1 / ya
            ze, zo = 1j * (xb - 2 * xa), 1j * xb
            s11 = ((ze - 1) / (ze + 1) + (zo - 1) / (zo + 1)) / 2
            S11, _ = te10(dict(kind="miter", angle=th, chamfer=sharp(th), plane="H", w=a, h=b, f=f))
            rows_h.append([th, lam, s11.real, s11.imag, S11.real, S11.imag])
            print("  %3d | %.1f | %.3f%+.3fj | %.3f%+.3fj | %.4f | %.2f / %.2f" % (
                th, lam, s11.real, s11.imag, S11.real, S11.imag, abs(S11 - s11), 20 * np.log10(abs(s11)),
                20 * np.log10(abs(S11))))
    a62, b62 = 15.799e-3, 7.899e-3
    rr = load_csv("reiter1995_fig3c.csv")
    rows_e = []
    print("\nE plane (Marcuvitz 5.28b, static) in WR62: theta | GHz | 2b/lambda_g | Marcuvitz dB | FEM dB | Reiter Fig. 3(c) dB")
    for th in (30, 45, 60, 75, 90):
        for f in (12.05e9, 13.05e9):
            k = 2 * np.pi * f / cst.c
            lg = 2 * np.pi / np.sqrt(k * k - (np.pi / a62) ** 2)
            Ba = 2 * b62 / lg * (digamma(1 - 0.5 * (1 - th / 180)) - digamma(0.5))
            Bb = lg / (2 * np.pi * b62) / np.tan(np.radians(th) / 2)
            ye, yo = 1j * Ba, 1j * Ba - 2j * Bb
            s11 = ((1 - ye) / (1 + ye) + (1 - yo) / (1 + yo)) / 2
            S11, _ = te10(dict(kind="miter", angle=th, chamfer=sharp(th), plane="E", w=a62, h=b62, f=f))
            c = rr[f"gamma={180 - th}"]
            yr = np.interp(f / 1e9, c[:, 0], c[:, 1])
            rows_e.append([th, f, 2 * b62 / lg, 20 * np.log10(abs(s11)), 20 * np.log10(abs(S11)), yr])
            print("  %3d | %.2f | %.3f | %.2f | %.2f | %.2f" % tuple(rows_e[-1][:1] + [f / 1e9] + rows_e[-1][2:]))
    m = meta("benchmarks/fem2d/marcuvitz_corners.py", "sharp corners of arbitrary angle, FEM vs Marcuvitz 5.29b "
             "(measured, H plane, WR-90) and 5.28b (static, E plane, WR62) and Reiter Fig. 3(c)",
             columns=dict(hplane="theta, lambda cm, Re/Im S11 Marcuvitz, Re/Im S11 FEM",
                          eplane="theta, f, 2b/lambda_g, Marcuvitz dB, FEM dB, Reiter dB"))
    print("saved", save("marcuvitz_corners.npz", m, hplane=np.array(rows_h), eplane=np.array(rows_e)))


if __name__ == "__main__":
    main()
