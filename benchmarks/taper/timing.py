"""WP5 timing (card item 5): WR-90 -> 22.86 x 1.6 mm raised-cosine taper of 50 mm at 90 GHz (target < 5 s per
frequency on a laptop) and the SPARC strangled bend (two such tapers, R = 50-200 mm, 90 deg) at 90 GHz (target
< 30 s), default settings, the line's basis (propagating modes) at the ports. Best of `REPEAT` runs; BLAS limited to
one thread inside the component (as in `Line.sweep`).

Run: PYTHONPATH=src python benchmarks/taper/timing.py
Output: benchmarks/taper/data/timing.npz
"""
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

from wrmodes.components import Guide, Taper, StrangledBend  # noqa: E402

REPEAT = 2
WR90 = Guide(22.86e-3, 10.16e-3)
CASES = (("taper 50 mm to 1.6 mm, 90 GHz", lambda: Taper(0.05, 1.6e-3, 22.86e-3, "raised_cosine", guide=WR90,
                                                            plane="E"), 90e9, 5.0),
         ("strangled bend R = 50 mm, 90 GHz", lambda: StrangledBend(90, 0.05, 1.6e-3, 0.05, guide=WR90), 90e9, 30.0),
         ("strangled bend R = 200 mm, 90 GHz", lambda: StrangledBend(90, 0.2, 1.6e-3, 0.05, guide=WR90), 90e9, 30.0),
         ("strangled bend R = 50 mm, 40 GHz", lambda: StrangledBend(90, 0.05, 1.6e-3, 0.05, guide=WR90), 40e9, 30.0))


def main():
    times, info = [], []
    for name, make, f, target in CASES:
        best = np.inf
        for _ in range(REPEAT):
            c = make()
            t0 = time.time()
            g = c.gsm(f)
            best = min(best, time.time() - t0)
        times.append(best)
        info.append((sum(g.info["n_slices"].values()), g.info["classes"], max(g.info["n_modes"].values())))
        print(f"{name:36s}: {best:6.1f} s (target {target:g} s); {info[-1][1]} classes, up to {info[-1][2]} modes, "
              f"{info[-1][0]} slices", flush=True)
    m = meta("benchmarks/taper/timing.py", "WP5 timing (card item 5)", cases=[c[0] for c in CASES],
             targets_s=[c[3] for c in CASES], machine=platform.platform(), processor=platform.processor(),
             cpu_count=os.cpu_count(), repeat=REPEAT)
    save("timing.npz", m, time_s=np.array(times), slices=np.array([i[0] for i in info]),
         classes=np.array([i[1] for i in info]), max_modes=np.array([i[2] for i in info]))


if __name__ == "__main__":
    main()
