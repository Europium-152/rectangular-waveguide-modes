"""Error of the default (error-controlled) truncation of MiterMM, WR-90, H and E plane (WP4b).

For each case the propagating block with the default truncation (each scalar problem refined until its estimated
error is below ERR_TARGET = 5e-5) is compared with a reference computed with ERR_TARGET = 1.5e-5. The difference is
the error of the default up to the reference's own error (at most 1.5e-5 by its estimate). Also stored: the largest
error estimate, the largest M used and the time of the default solve.
Usage: python benchmarks/modematch/miter_convergence_3d.py  (writes data/miter_convergence_3d.npz)
"""
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from wrmodes import Guide                                  # noqa: E402
from wrmodes.modematch import MiterMM                      # noqa: E402
from _meta import meta, save                               # noqa: E402

G = Guide(22.86e-3, 10.16e-3)
CH = [None, 0.80, 0.87, 0.95, 1.0, np.sqrt(2)]
FS = [12e9, 30e9, 55e9, 70e9, 90e9]
TIGHT = 1.5e-5


class Tight(MiterMM):
    ERR_TARGET = TIGHT


def main():
    shape = (2, len(CH), len(FS))
    diff, est, mmax, tdef = (np.full(shape, np.nan) for _ in range(4))
    warnings.simplefilter("ignore", RuntimeWarning)
    for p, plane in enumerate("HE"):
        for i, ch in enumerate(CH):
            m, mt = MiterMM(ch, plane, G), Tight(ch, plane, G)
            for j, f in enumerate(FS):
                b, _ = m.default_basis(f)
                prop = np.concatenate([b.propagating_mask(f)] * 2)
                sel = np.ix_(prop, prop)
                t = time.perf_counter()
                g0 = m.gsm(f)
                tdef[p, i, j] = time.perf_counter() - t
                t = time.perf_counter()
                gt = mt.gsm(f)
                diff[p, i, j] = np.max(np.abs(g0.S[sel] - gt.S[sel]))
                est[p, i, j] = max(v[1] for v in g0.truncation.values() if v[1] is not None)
                mmax[p, i, j] = max(v[0] for v in g0.truncation.values())
                print(f"{plane} {('mirror' if ch is None else f'{ch:.4f}'):8s} {f / 1e9:4.0f} GHz: |S - S_tight| "
                      f"{diff[p, i, j]:.1e}, max estimate {est[p, i, j]:.1e}, max M {mmax[p, i, j]:.0f}, "
                      f"default {tdef[p, i, j]:.2f} s, reference {time.perf_counter() - t:.0f} s", flush=True)
    m = meta("benchmarks/modematch/miter_convergence_3d.py",
             "Error of the default (error-controlled) truncation of MiterMM vs a reference with ERR_TARGET = 1.5e-5",
             chamfers=["mirror" if c is None else c for c in CH], freqs=FS, err_target=MiterMM.ERR_TARGET,
             tight_target=TIGHT, layout="diff[plane H/E, chamfer, f] = max |S_default - S_tight| over the propagating "
                                        "block; est = largest error estimate of the default; M_max; t_default (s)")
    save("miter_convergence_3d.npz", m, diff=diff, est=est, M_max=mmax, t_default=tdef)


if __name__ == "__main__":
    main()
