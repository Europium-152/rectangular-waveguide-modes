"""WP5 benchmark: the T2b section 9 table (acceptance) and the COMSOL campaign 2 predictions for case E2.

Linear symmetric H-plane taper 22.86 -> 45 mm (b = 10.16 mm), 15 GHz, L = 0.1, 0.2, 0.4, 0.8 m, TE10 incident at
the narrow end. TE30 passes through cutoff at a = 29.98 mm inside the taper. Reference planes: the taper ends.
For each L: S11(TE1,0), S21(TE1,0 <- TE1,0), S21(TE3,0 <- TE1,0) (E2 calls the last two S21 and S31), for
mode_factor 2 (default), 4, 8 with B5' on and off, and the Richardson extrapolation of the plain (B5' off) values
in 1/M (T2 section 8.4). Bases: family n = 0 (the 2D H-plane problem of E2; the other families do not couple).

T2b reference (docs/theory/T2b-taper-coupling.md section 9, prototype with M = 10 modes per family, B5' with
m_tail = 500): S21(TE30) = 5.0040e-2 - 5.3514e-3i, 2.7271e-3 + 2.5519e-2i, -1.1563e-3 - 1.2944e-2i,
-1.0636e-3 - 6.4437e-3i and |S11| = 2.009e-3, 8.503e-4, 4.394e-4, 2.245e-4. Acceptance: 1e-4 absolute.

Run: PYTHONPATH=src python benchmarks/taper/t2b_table.py   (about 2 min)
Output: benchmarks/taper/data/t2b_table.npz, benchmarks/taper/data/e2_predictions.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

from wrmodes.components import Taper, Guide  # noqa: E402
from wrmodes.modes import ModeBasis  # noqa: E402

F, A0, A1, B = 15e9, 22.86e-3, 45e-3, 10.16e-3
LS = (0.1, 0.2, 0.4, 0.8)
MFS = (2.0, 4.0, 8.0)
T2B_S31 = np.array([5.0040e-02 - 5.3514e-03j, 2.7271e-03 + 2.5519e-02j, -1.1563e-03 - 1.2944e-02j,
                    -1.0636e-03 - 6.4437e-03j])
T2B_ABS_S11 = np.array([2.009e-3, 8.503e-4, 4.394e-4, 2.245e-4])


def bases(f=F):
    b1 = ModeBasis(A0, B, [m for m in ModeBasis.propagating(f, A0, B).modes if m.n == 0], frame="H")
    b2 = ModeBasis(A1, B, [m for m in ModeBasis.propagating(f, A1, B).modes if m.n == 0], frame="H")
    return b1, b2


def column(L, mode_factor=2.0, tail=True, slice_factor=None):
    """(S11, S21, S31, info) of the taper of length L."""
    b1, b2 = bases()
    tp = Taper(L, A1, B, "linear", guide=Guide(A0, B), mode_factor=mode_factor, tail=tail, atol=1e-8)
    g = tp.gsm(F, b1, b2, slice_factor=slice_factor)
    j = b1.index("TE1,0")
    return g.S11[j, j], g.S21[b2.index("TE1,0"), j], g.S21[b2.index("TE3,0"), j], g


def main():
    res = np.zeros((len(LS), 2, len(MFS), 3), complex)      # L, tail (on, off), mode_factor, (S11, S21, S31)
    rich = np.zeros((len(LS), 3), complex)
    unit = np.zeros((len(LS), 2, len(MFS)))
    t0 = time.time()
    for i, L in enumerate(LS):
        for it, tail in enumerate((True, False)):
            for k, mf in enumerate(MFS):
                s11, s21, s31, g = column(L, mf, tail)
                res[i, it, k] = (s11, s21, s31)
                unit[i, it, k] = max(g.unitarity_defect(), g.reciprocity_defect())
        f1, f2 = MFS[-2], MFS[-1]
        rich[i] = (f2 * res[i, 1, -1] - f1 * res[i, 1, -2]) / (f2 - f1)
    elapsed = time.time() - t0
    print("T2b section 9 table: S21(TE3,0 <- TE1,0) and |S11(TE1,0)|, default (mode_factor 2, B5')")
    print("  L (m)  S31 (WP5, default)             |dS31| vs T2b  |S11| WP5   |d|S11|| vs T2b   change to mf 8 (S31, S11)")
    for i, L in enumerate(LS):
        s11, s21, s31 = res[i, 0, 0]
        print(f"  {L:4.1f}  {s31.real:+.5e}{s31.imag:+.5e}i   {abs(s31 - T2B_S31[i]):.1e}       {abs(s11):.4e}  "
              f"{abs(abs(s11) - T2B_ABS_S11[i]):.1e}          {abs(res[i, 0, -1, 2] - s31):.1e}, "
              f"{abs(res[i, 0, -1, 0] - s11):.1e}")
    print("\nConvergence in M (|S - S(mf 8, B5')|, max over S11, S21, S31):")
    for i, L in enumerate(LS):
        ref = res[i, 0, -1]
        on = [np.max(np.abs(res[i, 0, k] - ref)) for k in range(len(MFS))]
        off = [np.max(np.abs(res[i, 1, k] - ref)) for k in range(len(MFS))]
        print(f"  L = {L}: B5' on  mf 2/4/8: " + " ".join(f"{x:.1e}" for x in on)
              + "   B5' off: " + " ".join(f"{x:.1e}" for x in off)
              + f"   Richardson(off, 4-8): {np.max(np.abs(rich[i] - ref)):.1e}")
    print(f"max unitarity/reciprocity defect {unit.max():.1e}; {elapsed:.0f} s")
    m = meta("benchmarks/taper/t2b_table.py", "T2b section 9 table: linear H-plane taper 22.86 -> 45 mm, 15 GHz",
             f=F, a0=A0, a1=A1, b=B, L=list(LS), mode_factors=list(MFS), atol=1e-8,
             axes="L, tail (True, False), mode_factor, (S11 TE10, S21 TE10<-TE10, S21 TE30<-TE10)")
    save("t2b_table.npz", m, L=np.array(LS), mode_factors=np.array(MFS), S=res, richardson_plain=rich,
         t2b_S31=T2B_S31, t2b_abs_S11=T2B_ABS_S11, defects=unit)
    # E2 predictions: L = 0.1 and 0.2 (indices 0, 1)
    m = meta("benchmarks/taper/t2b_table.py", "COMSOL campaign 2, case E2: predictions at the taper end planes",
             f=F, a0=A0, a1=A1, b=B, L=[0.1, 0.2], reference_planes="taper ends (z = 0 and z = L)",
             columns="S11 = TE10 reflection, S21 = TE10 <- TE10, S31 = TE30 <- TE10",
             best="mode_factor 8 with B5'", spread="max over mode_factor 2-8 with B5' and Richardson without")
    best = res[:2, 0, -1]
    spread = np.array([[max(np.max(np.abs(res[i, 0, :, c] - best[i, c])), abs(rich[i, c] - best[i, c]))
                        for c in range(3)] for i in range(2)])
    save("e2_predictions.npz", m, L=np.array([0.1, 0.2]), S=best, spread=spread,
         S_dB=20 * np.log10(np.abs(best)), phase_deg=np.degrees(np.angle(best)))
    print("\nE2 predictions (mode_factor 8, B5'; spread = largest change over the convergence study):")
    for i, L in enumerate((0.1, 0.2)):
        print(f"  L = {L} m: " + ", ".join(f"{nm} = {s.real:+.5f}{s.imag:+.5f}i ({20 * np.log10(abs(s)):.3f} dB, "
                                           f"{np.degrees(np.angle(s)):.2f} deg, +-{d:.0e})"
                                           for nm, s, d in zip(("S11", "S21", "S31"), best[i], spread[i])))


if __name__ == "__main__":
    main()
