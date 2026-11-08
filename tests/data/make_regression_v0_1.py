"""Generate tests/data/regression_v0_1.npz: reference outputs of the wrmodes 0.1 transfer-matrix solver.

Run from the repository root at the commit given in the file's metadata (the 0.1 solver code must be unchanged):

    python tests/data/make_regression_v0_1.py

The cases cover the bends of the JINST paper (SPARC section): WR-90 in the H-plane (TE1,0, family n = 0) and
E-plane (TE0,1, family n = 1) orientations, 60 cm 60 deg and 40 cm 90 deg bends, hyperbolic-secant and constant
curvature, at four frequencies across 18-90 GHz, with error control on every entry of T and R.
"""
import datetime
import json
import os
import subprocess
import numpy as np

import wrmodes as wr
from wrmodes.transfer_matrix import coupled_mode_matrices, select_coupled_modes

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 0.9 * 0.0254, 0.4 * 0.0254


def sech_curvature(theta_deg, length, T=4.0):
    gamma = 2 * T / length
    R0 = 4 / np.radians(theta_deg) / gamma * np.arctan(np.tanh(T / 2))
    return lambda z: 1 / (R0 * np.cosh(gamma * (np.asarray(z) - length / 2)))


def constant_curvature(theta_deg, length):
    return lambda z: np.ones_like(np.asarray(z, dtype=float)) * np.radians(theta_deg) / length


CASES = []
for orient, (a, b, n) in {"H": (W, H, 0), "E": (H, W, 1)}.items():
    for theta, L in ((60, 0.60), (90, 0.40)):
        for profile in ("sech", "constant"):
            for f in (20e9, 45e9, 70e9, 90e9):
                CASES.append(dict(orient=orient, a=a, b=b, n=n, theta=theta, L=L, profile=profile, f=f))


def main():
    out = {}
    for i, c in enumerate(CASES):
        kappa = (sech_curvature if c["profile"] == "sech" else constant_curvature)(c["theta"], c["L"])
        modes = select_coupled_modes(c["f"], c["a"], c["b"], n=c["n"])
        D, K, beta = coupled_mode_matrices(c["f"], c["a"], c["b"], modes)
        mode_in = (modes[0][0][:2], int(modes[0][1]), int(modes[0][2]))
        _, res = wr.solve_transfer_matrix(c["f"], c["a"], c["b"], mode_in, c["L"], kappa, error_control="full",
                                          profile=False)
        out[f"c{i}_D"], out[f"c{i}_K"], out[f"c{i}_T"], out[f"c{i}_R"] = D, K, res.T, res.R
        out[f"c{i}_n_slices"] = np.array(res.n_slices)
        out[f"c{i}_names"] = np.array([m[0] for m in modes])
        print(f"case {i:2d}: {c['orient']} {c['theta']} deg {c['L']} m {c['profile']:8s} {c['f']/1e9:4.0f} GHz "
              f"N={len(modes):2d} slices={res.n_slices:5d} loss={-20*np.log10(abs(res.T[0,0])):.4f} dB")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(wrmodes_version=wr.__version__, git_commit=commit, solver_commit="441431e",
                created=datetime.date.today().isoformat(), generator="tests/data/make_regression_v0_1.py",
                cases=CASES, solver_settings=dict(error_control="full", atol=1e-6))
    np.savez_compressed(os.path.join(HERE, "regression_v0_1.npz"), meta=json.dumps(meta), **out)


if __name__ == "__main__":
    main()
