"""JINST paper, section 7: a 60 deg bend, 10 m of straight guide, a 90 deg bend, 1 m, and a -90 deg bend (WR-90).

Usage:  python cascade_sequence.py [H|E]     (plane of the bends, default H)

The line includes the reflections between the bends and every propagating family. The paper's ensemble over
straight-section lengths is a loop over Straight(10 + dL) etc.
"""
import sys
import numpy as np
import wrmodes as wr

if __name__ == "__main__":
    plane = sys.argv[1] if len(sys.argv) > 1 else "H"
    bend = lambda theta, L, sign=1: wr.SmoothBend(L, wr.SechCurvature(theta, L, sign=sign), plane=plane)
    line = wr.Line(wr.Guide(w=22.86e-3, h=10.16e-3),
                   [bend(60, 0.6), wr.Straight(10.0), bend(90, 0.4), wr.Straight(1.0), bend(90, 0.4, -1)])
    print(line.describe())
    res = line.sweep(np.arange(18e9, 90.1e9, 6e9), "TE1,0")        # parallel processes, one BLAS thread each
    for f, loss, refl in zip(res.freqs, res.loss_db, res.reflected_power):
        print(f"{f / 1e9:5.1f} GHz   loss {loss:8.4f} dB   reflected power {refl:.2e}")
