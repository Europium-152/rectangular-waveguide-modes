"""Twists: Doane & Anderson's three 20-inch 90 deg twists in tall WR187 (Microwave J. 32(3), 1989, Fig. 9).

The guide is WR187 used "tall": a = 0.872 in along the local u axis, b = 1.872 in along v, and the incident TE1,0 has
its electric field along the long side b. A positive twist angle turns the cross-section from +x towards +y
(right-handed about the direction of propagation). The twist's GSM is in the line's global frame at both ports: the
guide after the twist is the same guide, rotated in space.

Run: python examples/twist.py      (a few seconds; add --sweep for the 15-18 GHz curves, about 2 min)
"""
import sys

import numpy as np

import wrmodes as wr
from wrmodes.components import Guide, Twist, SmoothBend, SechCurvature

INCH = 0.0254
guide = Guide(0.872 * INCH, 1.872 * INCH)        # w along x = 22.15 mm, h along y = 47.55 mm
L = 20 * INCH

print("20-inch 90 deg twists in tall WR187, TE1,0 incident: total spurious power and the largest converted modes")
for profile in ("constant", "cosine", "triangular"):
    twist = Twist(90, L, profile, guide=guide)
    for f in (15e9, 16.5e9, 18e9):
        g = twist.gsm(f)                          # all propagating modes, same (global-frame) basis at both ports
        j = g.basis1.index("TE1,0")
        p = np.abs(g.S21[:, j]) ** 2
        top = sorted(((p[i], m.name) for i, m in enumerate(g.basis1) if i != j), reverse=True)[:3]
        print(f"  {profile:10s} {f / 1e9:5.1f} GHz: {100 * (1 - p[j]):6.3f} %   "
              + ", ".join(f"{n} {100 * v:.3f} %" for v, n in top))

# Convergence in the number of modes (evanescent modes of all families) and in the slicing
study = Twist(90, L, "triangular", guide=guide).convergence_study(15e9, cutoff_factors=(1.0, 1.3))
print("\nconvergence, triangular twist at 15 GHz:")
for cf, n, sl, pw in study["modes"]:
    print(f"  cutoff_factor {cf:4.2f}: {n:3d} modes, {sl:5d} slices, spurious {100 * pw['spurious']:.4f} %")
for sl, pw in study["slices"]:
    print(f"  {sl:5d} slices: spurious {100 * pw['spurious']:.6f} %")

# A twist in a line: +90 deg twist, H-plane bend, -90 deg twist. The bend bends side w, which the first twist has
# turned to lie along y: in space it turns the guide in the input's y-z plane (the E-plane direction of the input),
# while the fields see an H-plane bend; the second twist restores the orientation of the cross-section.
line = wr.Line(guide, [Twist(90, 0.4, "cosine"), SmoothBend(0.3, SechCurvature(90, 0.3), plane="H"),
                       Twist(-90, 0.4, "cosine")])
print("\n" + line.describe())
g = line.gsm(15e9)
j = g.basis1.index("TE1,0")
print(f"  TE1,0 transmission loss {g.loss_db('TE1,0'):.4f} dB, unitarity defect {g.unitarity_defect():.1e}")

if "--sweep" in sys.argv:
    freqs = np.linspace(15e9, 18e9, 31)
    for profile in ("constant", "cosine", "triangular"):
        tw = Twist(90, L, profile, guide=guide)
        tot = [1 - abs(tw.gsm(f).transmission("TE1,0")) ** 2 for f in freqs]
        print(f"{profile}: " + " ".join(f"{100 * t:.3f}" for t in tot))
