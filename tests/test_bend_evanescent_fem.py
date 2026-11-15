"""WF2 item 2 (tasks/WF2-wave2-fixes.md): a short bend directly at a miter, against WP7b's single FEM domain.

Reference: benchmarks/validation_v2/data/evanescent_bend_miter.npz (WP7b, `ref_evanescent_bends.py`: one
`Coplanar2D` domain with the arc and the optical 90 deg miter, WR-90, H plane). Before WF2 the `Line`
[SmoothBend, fem2d.Miter] was 2-4e-2 (R = 2a) and 4-6e-3 (R = 10a) of the input power off, because SmoothBend passed
the evanescent modes as uncoupled straight modes. Acceptance at cutoff_factor 3: 2e-4 (R = 2a) and 3e-5 (R = 10a)
over the propagating block. Full table: benchmarks/validation_v2/compare_wf2.py. Needs wrmodes[fem].
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("skfem")
pytest.importorskip("gmsh")

import wrmodes as wr                                                     # noqa: E402
from wrmodes.components import ConstantCurvature, Guide, SmoothBend    # noqa: E402
from wrmodes.fem2d import Miter                                         # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "validation_v2", "data", "evanescent_bend_miter.npz")
W, H = 22.86e-3, 10.16e-3


def _blk(S11, S12, S21, S22):
    return np.block([[S11, S12], [S21, S22]])


@pytest.mark.parametrize("key, tol", [("R2a_f20_L1a", 2e-4), ("R2a_f20_L0.25a", 2e-4), ("R10a_f20_L1a", 3e-5)])
def test_bend_directly_at_a_miter_against_one_fem_domain(key, tol):
    with np.load(DATA, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    row = [r for r in json.loads(str(d["meta"]))["rows"] if r["key"] == key][0]
    f, L, R = row["f"], row["L_over_a"] * W, row["R_over_a"] * W
    names = [str(s) for s in d[f"{key}_fem_names1"]]
    ref = _blk(*(d[f"{key}_fem_{b}"] for b in ("S11", "S12", "S21", "S22")))
    g = wr.Line(Guide(W, H), [SmoothBend(L, ConstantCurvature(1 / R), plane="H"), Miter(90, plane="H")]).gsm(
        f, cutoff_factor=3)
    idx = [g.basis1.names.index(n) for n in names]
    S = _blk(*(np.asarray(X)[np.ix_(idx, idx)] for X in (g.S11, g.S12, g.S21, g.S22)))
    dP = np.max(np.abs(np.abs(S) ** 2 - np.abs(ref) ** 2))
    assert dP < tol, dP
    assert g.reciprocity_defect() < 1e-10 * np.abs(g.S).max() and g.unitarity_defect() < 1e-9
