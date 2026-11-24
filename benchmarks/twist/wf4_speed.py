"""WF4 item 1: run time and accuracy of the Twist defaults on Doane & Anderson's 20-inch twists (tall WR187).

    WF4_BASELINE_COMMIT=a1943b4 PYTHONPATH=<checkout of a1943b4>/src python benchmarks/twist/wf4_speed.py baseline
    PYTHONPATH=src python benchmarks/twist/wf4_speed.py compare    # on the current code

`baseline` stores the GSMs (propagating port basis, all blocks) of the defaults for the three rate profiles at 15,
16.5 and 18 GHz with their wall times in `data/wf4_twist_baseline.npz`. `compare` recomputes them with the current
code and prints, per case, the wall time of both, the speed-up, and the largest change of any power |S_pq|^2
(transmitted and reflected, every input mode), and of the TE1,0 column alone, and the worst power difference per
mode against WP7b's helical-coordinate FEM (benchmarks/validation_v2/data/twist_doane_*.npz) before and now; the
result is stored in
`data/wf4_twist_compare.npz`. One BLAS thread (as `Twist.gsm` sets it). Card: tasks/WF4-release-fixes.md, item 1.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save, load, REPO  # noqa: E402

from wrmodes.components import Guide, Twist  # noqa: E402

INCH = 0.0254
A, B, L = 0.872 * INCH, 1.872 * INCH, 20 * INCH
GUIDE = Guide(A, B)
CASES = [(p, f) for p in ("constant", "cosine", "triangular") for f in (15e9, 16.5e9, 18e9)]


def key(p, f):
    return f"{p}_f{f / 1e9:g}"


def run(p, f):
    tw = Twist(90, L, p, guide=GUIDE)
    port = GUIDE.basis(f, 1.0)
    t = time.perf_counter()
    g = tw.gsm(f, port)
    dt = time.perf_counter() - t
    return g, dt


def baseline():
    out = {}
    for p, f in CASES:
        g, dt = run(p, f)
        k = key(p, f)
        out[f"{k}_S11"], out[f"{k}_S21"] = g.S11, g.S21
        out[f"{k}_seconds"] = np.array(dt)
        out[f"{k}_n_slices"] = np.array(g.info["n_slices"])
        out[f"{k}_names"] = np.array([m.name for m in g.basis1.modes])
        print(f"{k:22s} {dt:7.2f} s  N={g.info['n_modes']} slices={g.info['n_slices']}", flush=True)
    import wrmodes
    m = meta("benchmarks/twist/wf4_speed.py", "Twist defaults before WF4 (Doane 20-inch twists, tall WR187)",
             a_m=A, b_m=B, length_m=L, angle_deg=90, cases=[key(p, f) for p, f in CASES], blas_threads=1,
             solver_commit=os.environ.get("WF4_BASELINE_COMMIT", "the checkout's"), solver_path=wrmodes.__file__)
    print(save("wf4_twist_baseline.npz", m, **out))


def helical_fem(p, f):
    """WP7b's helical-coordinate FEM GSM (main setting) of Doane's twist with rate profile p at f, or None."""
    path = os.path.join(REPO, "benchmarks", "validation_v2", "data", f"twist_doane_{p}.npz")
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    key = f"doane_{p}_f{f / 1e9:g}_main"
    if f"{key}_S21" not in d:
        return None
    return d[f"{key}_S21"], d[f"{key}_S11"], [str(s) for s in d[f"{key}_names"]]


def worst_vs_fem(S21, S11, names, fem):
    """Worst | |S|^2 difference | over the modes converted from TE1,0 (transmitted and reflected) against the FEM
    (the metric of WF2, tests/test_twist.py::_worst_dP)."""
    F21, F11, fn = fem
    j, jf = names.index("TE1,0"), fn.index("TE1,0")
    out = 0.0
    for S, F, blk in ((S21, F21, "S21"), (S11, F11, "S11")):
        for i, n in enumerate(fn):
            if blk == "S21" and n == "TE1,0":
                continue
            out = max(out, abs(abs(S[names.index(n), j]) ** 2 - abs(F[i, jf]) ** 2))
    return out


def compare():
    ref, rmeta = load("wf4_twist_baseline.npz")
    rows, out = [], {}
    print(f"{'case':22s} {'before':>8s} {'now':>8s} {'speed-up':>8s} {'max dP':>9s} {'max dP TE1,0':>12s} slices")
    for p, f in CASES:
        k = key(p, f)
        g, dt = run(p, f)
        names = [str(s) for s in ref[f"{k}_names"]]
        assert names == [m.name for m in g.basis1.modes]
        dP = max(float(np.max(np.abs(np.abs(getattr(g, b)) ** 2 - np.abs(ref[f"{k}_{b}"]) ** 2))) for b in ("S11", "S21"))
        j = names.index("TE1,0")
        dP10 = max(float(np.max(np.abs(np.abs(getattr(g, b)[:, j]) ** 2 - np.abs(ref[f"{k}_{b}"][:, j]) ** 2)))
                   for b in ("S11", "S21"))
        t0 = float(ref[f"{k}_seconds"])
        fem = helical_fem(p, f)
        fb = fn = np.nan
        if fem is not None:
            fb = worst_vs_fem(ref[f"{k}_S21"], ref[f"{k}_S11"], names, fem)
            fn = worst_vs_fem(g.S21, g.S11, names, fem)
        rows.append((k, t0, dt, t0 / dt, dP, dP10, int(ref[f"{k}_n_slices"]), g.info["n_slices"], fb, fn))
        print(f"{k:22s} {t0:8.2f} {dt:8.2f} {t0 / dt:8.2f} {dP:9.2e} {dP10:12.2e} {rows[-1][6]} -> {rows[-1][7]}"
              f"   vs FEM {fb:.3e} -> {fn:.3e}", flush=True)
        out[f"{k}_S11"], out[f"{k}_S21"] = g.S11, g.S21
    for prof in ("cosine", "triangular"):
        r = [x for x in rows if x[0].startswith(prof)]
        print(f"{prof}: total {sum(x[1] for x in r):.1f} s -> {sum(x[2] for x in r):.1f} s "
              f"({sum(x[1] for x in r) / sum(x[2] for x in r):.2f}x), worst dP {max(x[4] for x in r):.2e}")
    m = meta("benchmarks/twist/wf4_speed.py", "Twist defaults after WF4 against the stored baseline",
             baseline_commit=rmeta["git_commit"], blas_threads=1,
             rows=[dict(case=x[0], seconds_before=x[1], seconds_now=x[2], speedup=x[3], max_dP=x[4],
                        max_dP_TE10=x[5], slices_before=x[6], slices_now=x[7], fem_worst_dP_before=x[8],
                        fem_worst_dP_now=x[9]) for x in rows])
    print(save("wf4_twist_compare.npz", m, **out))


if __name__ == "__main__":
    {"baseline": baseline, "compare": compare}[sys.argv[1] if len(sys.argv) > 1 else "compare"]()
