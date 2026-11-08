"""T2 study: convergence of the truncated equations in the number of modes M, and Richardson extrapolation.

Run (about 25 minutes, the reference coefficient code is pure Python):
    PYTHONPATH=src python3 docs/theory/T2/study_convergence.py

H-plane wedge (family n = 0), wall slope a' = 0.405, between z0 = 4 mm and z1 = 34 mm of the wedge of
check_solver.test_wedge (TE3,0 crosses cutoff inside). Low-order GSM blocks (TE1..TE4,0 at both ports) for
M = 8, 16, ..., 128 modes; successive differences of the raw results (expected ~1/M) and of the Richardson
extrapolations R(M) = 2 S(M) - S(M/2).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import vi_solver as vs                  # noqa: E402
import exact_fields as ef               # noqa: E402
from wrmodes.modes import Mode          # noqa: E402

f = 20e9
k = 2 * np.pi * f / 299792458.0
phi0 = 0.4
za = (3 * np.pi / phi0 / k) * np.cos(phi0 / 2) - 0.015
z0, z1 = 0.004, 0.034
geometry = ef.HPlaneWedge(f, 0.0102, za, phi0, [("TMv", 0, 1, "J", 1.0)]).geometry
out = {}
for M in (8, 16, 32, 64, 128):
    modes = [Mode("TE", m, 0) for m in range(1, M + 1)]
    G = vs.taper_gsm(modes, lambda z: geometry(z + z0), f, z1 - z0, nslices=30)
    out[M] = np.block([[G.S11[:4, :4], G.S12[:4, :4]], [G.S21[:4, :4], G.S22[:4, :4]]])
    print(f"M = {M} done", flush=True)
Ms = sorted(out)
for m1, m2 in zip(Ms[:-1], Ms[1:]):
    print(f"raw: max|S(M={m2}) - S(M={m1})| = {np.max(np.abs(out[m2] - out[m1])):.2e}")
R = {m2: 2 * out[m2] - out[m1] for m1, m2 in zip(Ms[:-1], Ms[1:])}
Rs = sorted(R)
for m1, m2 in zip(Rs[:-1], Rs[1:]):
    print(f"Richardson: max|R({m2}) - R({m1})| = {np.max(np.abs(R[m2] - R[m1])):.2e}")
