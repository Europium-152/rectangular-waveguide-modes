"""The exp(theta K) study (WP4, tasks/WP4-steps.md item 5): the coupled-mode "zero-length bend".

A bend of angle theta whose length tends to zero at fixed kappa L = theta has the transfer matrix
Phi = exp(theta K), with K the curvature coupling matrix of the coupled-mode equations dy/dz = (D + kappa K) y
(wrmodes.transfer_matrix.coupled_mode_matrices, propagating modes of one family, CONVENTIONS.md section 6).
This script converts Phi with wrmodes.gsm.from_transfer_matrix for a WR-90 90 degree bend in the H plane and in
the E plane, records the unitarity and reciprocity of the GSM and the mode conversion versus frequency, and stores
everything in benchmarks/data/exp_theta_k_wr90.npz (format of CONVENTIONS.md section 9), for comparison with the
miter of the 2D solver (WP3) at Gate A.

Frames (CONVENTIONS.md section 2): H plane, local u = x, a = 22.86 mm, b = 10.16 mm, TE10 in family n = 0. E plane,
local u = y, a = 10.16 mm, b = 22.86 mm, global TE10 = local -TE0,1, family n = 1. Positive curvature.

Run:  PYTHONPATH=src python3 benchmarks/exp_theta_k_study.py
"""
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import scipy.constants as cst
from scipy.linalg import expm

import wrmodes
from wrmodes.modes import ModeBasis
from wrmodes.gsm import from_transfer_matrix
from wrmodes.transfer_matrix import coupled_mode_matrices

W, H = 0.9 * 0.0254, 0.4 * 0.0254          # WR-90 (m)
THETA = np.pi / 2
F_MIN, F_MAX, F_STEP = 7.0e9, 100.0e9, 0.25e9
CASES = {"H": dict(a=W, b=H, family=0, input=("TE", 1, 0)),
         "E": dict(a=H, b=W, family=1, input=("TE", 0, 1))}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "exp_theta_k_wr90.npz")


def zero_length_bend(f, a, b, family, theta, frame=None):
    """GSM of exp(theta K) in the basis of the propagating modes of `family` at f (Hz)."""
    basis = ModeBasis.propagating(f, a, b, families=[family], frame=frame)
    _, K, _ = coupled_mode_matrices(f, a, b, basis.legacy_modes(f))
    phi = expm(theta * K)
    return from_transfer_matrix(phi, basis, f, name=f"exp(theta K), theta = {theta:g}"), phi


def too_close_to_cutoff(f, a, b, family, rel=1e-4):
    fc = ModeBasis.below(2 * f, a, b, families=[family]).cutoff_frequencies()
    return np.any(np.abs(fc - f) < rel * f)


def git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True).stdout
        return c + ("+dirty" if dirty.strip() else "")
    except OSError:
        return "unknown"


def main(out=OUT):
    freqs = np.arange(F_MIN, F_MAX + F_STEP / 2, F_STEP)
    keep = np.array([not any(too_close_to_cutoff(f, c["a"], c["b"], c["family"]) for c in CASES.values())
                     for f in freqs])
    freqs = freqs[keep]
    data = {"f": freqs, "theta": np.array(THETA)}
    for plane, c in CASES.items():
        res = [zero_length_bend(f, c["a"], c["b"], c["family"], THETA, frame=plane) for f in freqs]
        nmax = max(len(g.basis1) for g, _ in res)
        F = len(freqs)
        S = np.full((F, 2 * nmax, 2 * nmax), np.nan + 0j)
        names = np.full((F, nmax), "", dtype="<U8")
        n_modes = np.zeros(F, int)
        uni, rec, t_in, r_in, conv = (np.zeros(F) for _ in range(5))
        for i, (g, _) in enumerate(res):
            N = len(g.basis1)
            n_modes[i] = N
            names[i, :N] = g.basis1.names
            S[i, :N, :N], S[i, :N, nmax:nmax + N] = g.S11, g.S12
            S[i, nmax:nmax + N, :N], S[i, nmax:nmax + N, nmax:nmax + N] = g.S21, g.S22
            uni[i] = g.unitarity_defect()
            rec[i] = g.reciprocity_defect() / np.max(np.abs(g.S))
            j = g.basis1.index(c["input"])
            t_in[i] = abs(g.S21[j, j]) ** 2                     # power kept in the input mode
            r_in[i] = np.sum(np.abs(g.S11[:, j]) ** 2)          # total reflected power
            conv[i] = np.sum(np.abs(g.S21[:, j]) ** 2) - t_in[i]  # power converted to other forward modes
        data.update({f"S_{plane}": S, f"modes_{plane}": names, f"n_modes_{plane}": n_modes,
                     f"unitarity_{plane}": uni, f"reciprocity_{plane}": rec,
                     f"transmitted_{plane}": t_in, f"reflected_{plane}": r_in, f"converted_{plane}": conv})
    meta = {"wrmodes_version": wrmodes.__version__, "git_commit": git_commit(),
            "created": datetime.date.today().isoformat(), "generator": "benchmarks/exp_theta_k_study.py",
            "description": "GSM of Phi = exp(theta K) (coupled-mode zero-length bend), WR-90, 90 degrees, H and E "
                           "plane, propagating modes of the input family, from_transfer_matrix",
            "parameters": {"w": W, "h": H, "theta": THETA, "f_min": F_MIN, "f_max": F_MAX, "f_step": F_STEP,
                           "frames": {"H": "local u = x, a = w, b = h, family n = 0, input TE1,0",
                                      "E": "local u = y, a = h, b = w, family n = 1, input TE0,1 (= -global TE10)"},
                           "curvature_sign": "+", "excluded": "frequencies within 1e-4 relative of a cutoff"},
            "layout": "S_<plane>[i] is the full 2 nmax x 2 nmax matrix [[S11, S12], [S21, S22]] padded with NaN; "
                      "modes_<plane>[i] are the port-basis mode names (same at both ports); transmitted = |S21| "
                      "of the input mode squared, reflected = total reflected power, converted = power in other "
                      "forward modes"}
    data["meta"] = np.array(json.dumps(meta))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **data)
    return data


if __name__ == "__main__":
    d = main(sys.argv[1] if len(sys.argv) > 1 else OUT)
    for plane in CASES:
        print(f"{plane} plane: {len(d['f'])} frequencies, max unitarity defect {d[f'unitarity_{plane}'].max():.2e}, "
              f"max reciprocity defect {d[f'reciprocity_{plane}'].max():.2e}")
        for fg in (8.2, 10, 12.4, 20, 30, 45, 70, 90):
            i = int(np.argmin(np.abs(d["f"] - fg * 1e9)))
            print(f"  {d['f'][i] / 1e9:6.2f} GHz  N={d[f'n_modes_{plane}'][i]:2d}  kept {d[f'transmitted_{plane}'][i]:.6f}"
                  f"  converted {d[f'converted_{plane}'][i]:.6f}  reflected {d[f'reflected_{plane}'][i]:.3e}")
