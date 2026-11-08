"""Strangling section: WR-90 to a 22.86 x 1.6 mm guide and back, by mode matching (wrmodes.modematch).

The two steps are E-plane steps (only b changes), so they couple TE10 to the TE1n/TM1n modes with even n. The
strangled section is a straight of length L between them. The script prints the TE10 transmission and reflection
versus L at 10 GHz and versus frequency for L = 20 mm, and checks reciprocity and unitarity of each result.

Run:  PYTHONPATH=src python3 examples/strangling_step.py
"""
import numpy as np

from wrmodes.modematch import Step
from wrmodes.gsm import straight

W, H = 22.86e-3, 10.16e-3      # WR-90 (m)
HS = 1.6e-3                    # height of the strangled guide (m)

down = Step(W, H, W, HS)       # WR-90 -> 22.86 x 1.6 mm
up = Step(W, HS, W, H)         # and back


def strangled(f, length):
    g_down, g_up = down.gsm(f), up.gsm(f)
    return g_down @ straight(g_down.basis2, f, length) @ g_up


if __name__ == "__main__":
    f = 10e9
    print(f"single step at {f / 1e9:g} GHz: |S11(TE10)| = {abs(down.gsm(f).S11[0, 0]):.5f}, "
          f"{len(down.gsm(f).basis1)} + {len(down.gsm(f).basis2)} modes")
    print("\n  L (mm)   |S21| (dB)   |S11|     reciprocity  unitarity")
    for L in (0.0, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0):
        g = strangled(f, L * 1e-3)
        print(f"  {L:6.1f}   {20 * np.log10(abs(g.S21[0, 0])):9.4f}   {abs(g.S11[0, 0]):.5f}   "
              f"{g.reciprocity_defect():.1e}      {g.unitarity_defect():.1e}")
    L = 20e-3
    print(f"\n  f (GHz)  |S21| (dB)   |S11|     (L = {L * 1e3:g} mm)")
    for f in np.arange(8.5, 12.51, 0.5) * 1e9:
        g = strangled(f, L)
        print(f"  {f / 1e9:6.2f}   {20 * np.log10(abs(g.S21[0, 0])):9.4f}   {abs(g.S11[0, 0]):.5f}")
