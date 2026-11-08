"""Generate the WP7 reference data (CONVENTIONS.md section 9 npz files) and check them against wrmodes 0.1.

    python benchmarks/validation_v1/make_references.py [lines] [loss]

* data/line_reference.npz: the three reference lines of ref_line.reference_lines() at the frequencies of
  ref_line.REFERENCE_FREQUENCIES. For each (line, f): S11, S21, S12, S22 of the global-ODE reference over the
  propagating modes (keys '<line>/<f GHz>/S21', mode names in '<line>/<f GHz>/modes'), and the largest power
  difference to the wrmodes 0.1 product of transfer matrices (solver at atol = 1e-10).
* data/loss_reference.npz: the wall-loss matrix A (Np/m, diag = alpha) of every propagating mode for WR-90 and a
  20 mm square guide, aluminium and copper (ref_loss.GUIDES / MATERIALS / FREQUENCIES), with the per-wall parts.

The script prints the acceptance numbers quoted in REPORT.md.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ref_common  # noqa: E402
import ref_line  # noqa: E402
import ref_loss  # noqa: E402
from wrmodes.modes import ModeBasis  # noqa: E402


def fkey(f):
    return f"{f / 1e9:g}"


def make_lines():
    w, h = ref_line.WR90
    arrays, worst = {}, 0.0
    summary = {}
    for name, segs in ref_line.reference_lines().items():
        for f in ref_line.REFERENCE_FREQUENCIES[name]:
            t0 = time.time()
            gb, phi, S, info = ref_line.line_transfer(f, w, h, segs)
            t1 = time.time()
            _, phi01, S01 = ref_line.line_transfer_v01(f, w, h, segs, atol=1e-10)
            t2 = time.time()
            dP = max(np.max(np.abs(np.abs(S[k]) ** 2 - np.abs(S01[k]) ** 2)) for k in ("S11", "S21", "S12", "S22"))
            dA = max(np.max(np.abs(S[k] - S01[k])) for k in ("S11", "S21", "S12", "S22"))
            worst = max(worst, dP)
            key = f"{name}/{fkey(f)}"
            for k in ("S11", "S21", "S12", "S22"):
                arrays[f"{key}/{k}"] = S[k]
            arrays[f"{key}/modes"] = np.array(gb.names)
            arrays[f"{key}/dP_vs_v01"] = np.array(dP)
            j = gb.index(("TE", 1, 0))
            loss = -10 * np.log10(abs(S["S21"][j, j]) ** 2)
            summary[key] = dict(N=len(gb), loss_TE10_dB=loss, dP=dP, dA=dA, t_ref=t1 - t0, t_v01=t2 - t1)
            print(f"{key:24s} N={len(gb):3d}  TE10 loss {loss:9.5f} dB  max|dP| vs 0.1 = {dP:.1e}  "
                  f"max|dS| = {dA:.1e}  ({t1 - t0:.0f} s / {t2 - t1:.0f} s)", flush=True)
    path = ref_common.save_npz(os.path.join(ref_common.DATA, "line_reference.npz"), __file__, arrays,
                               guide="WR-90 0.9 x 0.4 in", lines={k: [s.label for s in v] for k, v in
                                                                  ref_line.reference_lines().items()},
                               frequencies=ref_line.REFERENCE_FREQUENCIES, rtol=1e-11, v01_atol=1e-10,
                               cutoff_factor=1.0)
    print(f"largest power difference to wrmodes 0.1 over all lines: {worst:.2e}  -> {path}")
    return summary


def make_loss():
    arrays = {}
    worst = 0.0
    for g, (a, b) in ref_loss.GUIDES.items():
        for mat, sigma in ref_loss.MATERIALS.items():
            for f in ref_loss.FREQUENCIES[g]:
                basis = ModeBasis.propagating(f, a, b)
                A, names, walls = ref_loss.loss_matrix(basis, f, sigma, per_wall=True)
                tb = np.array([ref_loss.textbook_alpha(m, a, b, f, sigma) for m in basis.modes])
                rel = float(np.max(np.abs(np.diag(A) / tb - 1)))
                worst = max(worst, rel)
                key = f"{g}/{mat}/{fkey(f)}"
                arrays[f"{key}/A"] = A
                arrays[f"{key}/modes"] = np.array(names)
                for wname, Aw in walls.items():
                    arrays[f"{key}/A_{wname}"] = Aw
                print(f"{key:28s} N={len(basis):3d}  TE10 {A[0, 0] * ref_loss.NP_TO_DB:.4f} dB/m  "
                      f"max rel. dev. from textbook {rel:.1e}")
    path = ref_common.save_npz(os.path.join(ref_common.DATA, "loss_reference.npz"), __file__, arrays,
                               guides=ref_loss.GUIDES, sigma=ref_loss.MATERIALS, frequencies=ref_loss.FREQUENCIES,
                               nq=96, definition="A_pq = (1/4) R_s oint H_tan,p . conj(H_tan,q) dl, Np/m")
    print(f"largest relative deviation of alpha from the textbook formulas: {worst:.1e}  -> {path}")


if __name__ == "__main__":
    what = set(sys.argv[1:]) or {"lines", "loss"}
    if "loss" in what:
        make_loss()
    if "lines" in what:
        make_lines()
