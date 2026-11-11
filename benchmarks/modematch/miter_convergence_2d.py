"""Convergence of the scalar 2D miter problems of `wrmodes.modematch.MiterMM` in the truncation M (WP4b).

For each wall type ('D': LSM, 'N': LSE), chamfer d/a and k_t a / pi, the propagating block of the 2D GSM (S11 and
S21 of the arm modes that propagate) is computed for a grid of M (arm modes; interface functions ceil(M / RATIO)
per width) and compared with a reference: the Aitken extrapolation of three solutions at M_ref, 2 M_ref, 4 M_ref
(the error decays as M^-p with p = 4/3, the exponent of the r^(2/3) field at the inner corner; the fitted p is
stored). Usage: python benchmarks/modematch/miter_convergence_2d.py D|N  (writes data/miter_convergence_2d_<wall>.npz)
"""
import sys
import time

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "src"))
from wrmodes.modematch import MiterMM                                  # noqa: E402
from wrmodes.modematch._corner2d import miter_2d                       # noqa: E402
from _meta import meta, save                                           # noqa: E402

D_VALUES = [1 / np.sqrt(2), 0.80, 0.87, 0.95, 1.0, np.sqrt(2)]
X_VALUES = {"D": [1.3, 2.5, 4.3, 6.7, 9.5, 13.7], "N": [0.5, 1.3, 2.5, 4.3, 6.7, 9.5, 13.7]}
M_GRID = {"D": [40, 60, 80, 100, 120, 160, 200, 240], "N": [60, 80, 100, 130, 160, 200, 260, 320, 400]}
M_REF = (250, 500, 1000)


def block(c, wall, ka, M, P):
    R = MiterMM.RATIO[wall]
    s11, s21 = miter_2d(1.0, c, wall, ka * ka, M, int(np.ceil(M / R)))
    return np.concatenate([s11[:P, :P].ravel(), s21[:P, :P].ravel()])


def main(wall):
    xs, Ms = X_VALUES[wall], M_GRID[wall]
    err = np.full((len(D_VALUES), len(xs), len(Ms)), np.nan)
    p_fit = np.full((len(D_VALUES), len(xs)), np.nan)
    ref_err = np.full((len(D_VALUES), len(xs)), np.nan)
    times = np.full((len(D_VALUES), len(xs), len(Ms)), np.nan)
    for i, d in enumerate(D_VALUES):
        c = min(max(np.sqrt(2) * d - 1, 0.0), 1.0)
        for j, x in enumerate(xs):
            ka = x * np.pi
            m0 = 1 if wall == "D" else 0
            P = int(np.sum(np.arange(m0, 200) * np.pi < ka))
            v = [block(c, wall, ka, M, P) for M in M_REF]
            d1, d2 = v[1] - v[0], v[2] - v[1]
            p = np.log2(np.linalg.norm(d1) / np.linalg.norm(d2))
            ref = v[2] + d2 / (2 ** p - 1)
            p_fit[i, j] = p
            ref_err[i, j] = np.max(np.abs(v[2] - ref))
            for k, M in enumerate(Ms):
                t = time.perf_counter()
                s = block(c, wall, ka, M, P)
                times[i, j, k] = time.perf_counter() - t
                err[i, j, k] = np.max(np.abs(s - ref))
            print(f"{wall} d/a={d:.4f} x={x:5.2f} p={p:.2f} " +
                  " ".join(f"{M}:{e:.1e}" for M, e in zip(Ms, err[i, j])), flush=True)
    m = meta("benchmarks/modematch/miter_convergence_2d.py",
             "Convergence of the scalar 2D miter problems (WP4b): max |S - S_ref| over the propagating block "
             "(S11 and S21 of the propagating arm modes), S_ref = Aitken extrapolation of M_ref, 2 M_ref, 4 M_ref",
             wall=wall, ratio=MiterMM.RATIO[wall], d_over_a=list(D_VALUES), kt_a_over_pi=list(xs), M=list(Ms),
             M_ref=list(M_REF), layout="err[d, x, M], p_fit[d, x] (fitted order), ref_err[d, x] = |S(4 M_ref) - S_ref|")
    save(f"miter_convergence_2d_{wall}.npz", m, err=err, p_fit=p_fit, ref_err=ref_err, times=times,
         d_over_a=np.array(D_VALUES), x=np.array(xs), M=np.array(Ms))


if __name__ == "__main__":
    main(sys.argv[1])
