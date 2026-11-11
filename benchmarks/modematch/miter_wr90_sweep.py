"""WR-90 90-degree miters by MiterMM (WP4b), H and E plane, d/a = 0.707-1.0, 10-90 GHz: the data for the FEM vs MM
comparison of WP7b (card WP4b: 0.01 dB on TE10 transmission, 1e-4 of the input power per converted mode).

For each plane and chamfer the propagating GSM (both ports, global modes in the local frame of `plane`) is stored at
every frequency, with the default truncation. Frequencies within 0.2 % of a mode cutoff of the basis are skipped
(listed in meta). Usage: python benchmarks/modematch/miter_wr90_sweep.py  (writes data/miter_wr90_sweep.npz)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from wrmodes import Guide                                  # noqa: E402
from wrmodes.modematch import MiterMM                      # noqa: E402
from _meta import meta, save                               # noqa: E402

GUIDE = Guide(22.86e-3, 10.16e-3)
CHAMFERS = [None, 0.80, 0.87, 0.90, 0.95, 1.0]
FREQS = np.arange(10e9, 90.01e9, 2e9)


def near_cutoff(f, rel=2e-3):
    b = GUIDE.basis(f * (1 + 2 * rel), 1.0)
    fc = b.cutoff_frequencies()
    return np.any(np.abs(fc - f) <= rel * f)


def main():
    freqs = np.array([f for f in FREQS if not near_cutoff(f)])
    skipped = [float(f) for f in FREQS if near_cutoff(f)]
    arrays, keys = {}, []
    nmax = max(len(GUIDE.basis(f, 1.0)) for f in freqs)
    for plane in ("H", "E"):
        for ch in CHAMFERS:
            m = MiterMM(ch, plane, GUIDE)
            S = np.full((len(freqs), 2 * nmax, 2 * nmax), np.nan + 0j)
            names = np.full((len(freqs), nmax), "", dtype="<U8")
            t10 = np.full(len(freqs), np.nan)
            times = np.zeros(len(freqs))
            for i, f in enumerate(freqs):
                t = time.perf_counter()
                g = m.gsm(f)
                times[i] = time.perf_counter() - t
                n = len(g.basis1)
                S[i, :2 * n, :2 * n] = g.S
                names[i, :n] = g.basis1.names
                j = g.basis1.index("TE1,0" if plane == "H" else "TE0,1")
                t10[i] = -20 * np.log10(abs(g.S21[j, j]))
            key = f"{plane}_{'mirror' if ch is None else f'{ch:.2f}'}"
            keys.append(key)
            arrays[key + "_S"], arrays[key + "_modes"] = S, names
            arrays[key + "_te10_loss_db"], arrays[key + "_time"] = t10, times
            print(f"{key}: {len(freqs)} frequencies, mean {times.mean():.2f} s, max {times.max():.2f} s", flush=True)
    m = meta("benchmarks/modematch/miter_wr90_sweep.py",
             "MiterMM GSMs of 90-degree WR-90 miters (default truncation), H and E plane, for WP7b",
             w=GUIDE.w, h=GUIDE.h, chamfers=["mirror" if c is None else c for c in CHAMFERS], keys=keys,
             skipped_near_cutoff=skipped,
             layout="<key>_S[i] = [[S11, S12], [S21, S22]] over the propagating modes <key>_modes[i] (local frame "
                    "of the plane: E plane modes are local, global TE10 = -local TE0,1), padded with NaN; "
                    "<key>_te10_loss_db = -20 log10 |S21| of the TE10 mode; reference planes through the inner "
                    "corner (CONVENTIONS.md 6), arm_length = a/2")
    save("miter_wr90_sweep.npz", m, freqs=freqs, **arrays)


if __name__ == "__main__":
    main()
