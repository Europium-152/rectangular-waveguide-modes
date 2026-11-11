"""Speed-up of the parity-reduced Step (WP4b): full solve vs reduced solve (classes of m and n), default truncation.
Cases of the card: WR-90 <-> 22.86 x 1.6 mm at 90 GHz (both directions) and a 2D step (both sides change).
Usage: python benchmarks/modematch/step_parity_timing.py  (writes data/step_parity_timing.npz)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from wrmodes.modematch import Step                         # noqa: E402
from _meta import meta, save                               # noqa: E402

CASES = [("WR-90 -> 22.86 x 1.6 mm, 90 GHz", Step(22.86e-3, 10.16e-3, 22.86e-3, 1.6e-3), 90e9),
         ("22.86 x 1.6 mm -> WR-90, 90 GHz", Step(22.86e-3, 1.6e-3, 22.86e-3, 10.16e-3), 90e9),
         ("2D: WR-90 -> 18 x 8 mm centred, 30 GHz", Step(22.86e-3, 10.16e-3, 18e-3, 8e-3), 30e9),
         ("2D: WR-90 -> 18 x 8 mm centred, 60 GHz", Step(22.86e-3, 10.16e-3, 18e-3, 8e-3), 60e9),
         ("2D: 12 x 6 mm -> WR-90 centred, 25 GHz", Step(12e-3, 6e-3, 22.86e-3, 10.16e-3), 25e9)]


def best(fn, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return min(ts), out


def main():
    rows = []
    print("| case | modes (port 1, port 2) | classes | full solve (s) | reduced (s) | speed-up | max |dS| |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, st, f in CASES:
        b1, b2 = st.default_basis(f)
        t0, g0 = best(lambda: st.solve(f, b1, b2, reduce=False), 2)
        t1, g1 = best(lambda: st.solve(f, b1, b2), 3)
        key = st._class_key()
        ncls = len({key(m) for m in list(b1.modes) + list(b2.modes)})
        d = float(np.max(np.abs(g0.S - g1.S)))
        rows.append((len(b1), len(b2), ncls, t0, t1, t0 / t1, d))
        print(f"| {name} | {len(b1)}, {len(b2)} | {ncls} | {t0:.3f} | {t1:.3f} | {t0 / t1:.1f} | {d:.1e} |", flush=True)
    m = meta("benchmarks/modematch/step_parity_timing.py", "Step.solve full vs parity-reduced (best of runs)",
             cases=[c[0] for c in CASES], cpu_count=os.cpu_count(),
             layout="rows[i] = (N1, N2, classes, t_full, t_reduced, speed-up, max |S_full - S_reduced|)")
    save("step_parity_timing.npz", m, rows=np.array(rows))


if __name__ == "__main__":
    main()
