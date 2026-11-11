"""Timings of MiterMM (WP4b): seconds per frequency with the default truncation, WR-90, H and E plane, at 30, 60
and 90 GHz, plain (optical mirror) and chamfered. The card's target: H-plane plain miter at 90 GHz under 1 s.
Usage: python benchmarks/modematch/miter_timing.py  (writes data/miter_timing.npz)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from wrmodes import Guide                                  # noqa: E402
from wrmodes.modematch import MiterMM                      # noqa: E402
from _meta import meta, save                               # noqa: E402

G = Guide(22.86e-3, 10.16e-3)
CH = [None, 0.87, 1.0, np.sqrt(2)]
FS = [30e9, 60e9, 90e9]


def main():
    T = np.zeros((2, len(CH), len(FS)))
    N = np.zeros((2, len(CH), len(FS)), int)
    for p, plane in enumerate("HE"):
        for i, ch in enumerate(CH):
            m = MiterMM(ch, plane, G)
            m.gsm(FS[0])
            for j, f in enumerate(FS):
                ts = []
                for r in range(3):
                    t = time.perf_counter()
                    g = m.gsm(f * (1 + 1e-4 * r))
                    ts.append(time.perf_counter() - t)
                T[p, i, j], N[p, i, j] = np.median(ts), len(g.basis1)
                print(f"{plane} {('mirror' if ch is None else f'{ch:.4f}'):8s} {f / 1e9:4.0f} GHz: "
                      f"{N[p, i, j]:4d} port modes, {T[p, i, j]:.3f} s", flush=True)
    import platform
    m = meta("benchmarks/modematch/miter_timing.py", "MiterMM seconds per frequency (median of 3), default truncation",
             machine=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
             chamfers=["mirror" if c is None else c for c in CH], freqs=FS, layout="time[plane H/E, chamfer, f]")
    save("miter_timing.npz", m, time=T, port_modes=N)


if __name__ == "__main__":
    main()
