"""WP7b reference for the impedance walls of WP3b (card WP7b item 4): Butterweck's lossy-wall mode filter by WP2's
exact longitudinal-section method (`wrmodes.filters.LossyWallFilter`, wave 1, validated by WP7 and T1).

WR-90 (22.86 x 10.16 mm), 75 GHz, 250 ohm (real) on both narrow walls (local u = +-a/2 of the H frame), 15 cm,
between perfect empty guides; reference planes at the two junctions. Port bases: the line's rule (propagating
modes). Convergence in the internal truncation M (modes per family): 80, 120, 160 (WP2 raises a RootFindingError at 320). The comparison (FEM with
`wall_impedance`, WP3b) is in compare_filter.py; the plan's target is 5 % on the attenuation (dB) of every mode.
"""
import json

import numpy as np

from wrmodes.components import Guide
from wrmodes.filters import LossyWallFilter

import vcommon as vc

F, ZW, L = 75e9, 250.0, 0.15

if __name__ == "__main__":
    g = Guide(vc.W90, vc.H90)
    basis = g.basis(F, 1.0, "H")
    arrays, rows = {}, []
    for M in (80, 120, 160):
        t = vc.Timer()
        G = LossyWallFilter(L, ZW, plane="H", modes_per_family=M, guide=g).gsm(F, basis, basis)
        key = f"M{M}"
        arrays.update(vc.gsm_arrays(key, G))
        att = {n: float(-vc.db(G.S21[i, i])) for i, n in enumerate(basis.names)}
        rows.append(dict(M=M, seconds=round(t(), 1), TE10=att["TE1,0"], TE20=att["TE2,0"], TE30=att["TE3,0"],
                         TE01=att["TE0,1"], TE11=att["TE1,1"], TM11=att["TM1,1"]))
        print(json.dumps(rows[-1]), flush=True)
    vc.save_npz("filter_lossywall_wp2.npz", __file__, arrays, rows=rows, f=F, Zw=ZW, length=L, walls=["u-", "u+"],
                method="wrmodes.filters.LossyWallFilter (WP2 exact LS^u modes, T1 B6+B9)")
