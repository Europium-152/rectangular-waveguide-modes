"""Generate tests/data/lines_unburied.npz: GSMs of lines without buried reference planes, frozen before WF3.

WF3 (tasks/WF3-buried-reference-planes.md) changes `Line.gsm` (buried reference planes, passivity check) and
`fem2d.Coplanar2D` (GSM at the uniform-arm planes); lines without buried planes must stay bitwise unchanged. The file
was generated with the code of the commit in its metadata (the WF3 card commit, before any WF3 code), from a
checkout of that commit:

    PYTHONPATH=<checkout of that commit>/src python tests/data/make_lines_unburied.py [--fem]

`tests/test_line.py::test_lines_without_buried_planes_are_bitwise_unchanged` (cases without the FEM) and
`tests/test_fem2d.py::test_fem_lines_without_buried_planes_are_bitwise_unchanged` (`--fem`: FEM miters, stored in
lines_unburied_fem.npz) recompute them. Cases: 90 degree mode-matching miters (H, E and both, with frame changes),
steps, lossy straights with frame changes (compared only with the scipy version that froze them: the lossy
straight's matrix exponential differs by ~1e-11 with scipy 1.13), at cutoff factors 1 to 2 (lossy bends are WF2's
smoothbend_propagating.npz); FEM 90 degree plain and 45 degree partially cut miters (mirror
outside the arms, no buried plane), alone, chained and mixed with MiterMM.
"""
import datetime
import json
import os
import subprocess
import sys

import numpy as np

import wrmodes as wr
from wrmodes.components import Guide
from wrmodes.modematch import Step

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 22.86e-3, 10.16e-3


def cases():
    g = Guide(W, H)
    return [
        ("mm_H_uturn", g, [wr.MiterMM(), wr.Straight(1.5 * W), wr.MiterMM()], ((30e9, 1.0), (30e9, 2.0))),
        ("mm_E_then_H", g, [wr.MiterMM(plane="E"), wr.Straight(W), wr.MiterMM(plane="H", angle_deg=-90)],
         ((12e9, 1.0), (12e9, 1.5))),
        ("steps", g, [Step(W, H, 1.2 * W, H), wr.Straight(0.01), Step(1.2 * W, H, W, H)], ((20e9, 1.0), (20e9, 2.0))),
        ("step_offset_E", g, [Step(H, W, 0.8 * H, W, du=0.05 * H, plane="E"), wr.Straight(0.02),
                              Step(0.8 * H, W, H, W, du=0.05 * H, plane="E")], ((15e9, 1.5),)),
        ("lossy_straights", Guide(W, H, 3.5e7), [wr.Straight(0.2), wr.FrameChange("H", "E"), wr.Straight(0.1),
                                                 wr.FrameChange("E", "H"), wr.Straight(0.05)],
         ((18e9, 1.0), (18e9, 1.5))),
    ]


def fem_cases():
    from wrmodes.fem2d import Miter
    g = Guide(W, H)
    return [
        ("fem_90_uturn", g, [Miter(90), wr.Straight(2 * W), Miter(90)], ((30e9, 1.0), (30e9, 1.5))),
        ("fem_45_cut095", g, [Miter(45, chamfer=0.95), wr.Straight(W), Miter(-45, chamfer=0.95)], ((30e9, 1.5),)),
        ("fem_90E_mmH", g, [Miter(90, plane="E"), wr.Straight(W), wr.MiterMM()], ((12e9, 1.5),)),
    ]


def compute(fem=False):
    out = {}
    for name, g, comps, points in (fem_cases() if fem else cases()):
        for f, cf in points:
            out[f"{name}_{f / 1e9:g}_cf{cf:g}"] = wr.Line(g, comps).gsm(f, cutoff_factor=cf).S
    return out


if __name__ == "__main__":
    fem = "--fem" in sys.argv
    src = os.path.dirname(os.path.dirname(os.path.abspath(wr.__file__)))
    try:
        commit = subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    meta = dict(wrmodes_version=wr.__version__, git_commit=commit, created=datetime.date.today().isoformat(),
                generator="tests/data/make_lines_unburied.py" + (" --fem" if fem else ""),
                cases=[(n, [repr(c) for c in cs], [list(p) for p in pts]) for n, _, cs, pts in
                       (fem_cases() if fem else cases())],
                note="lines without buried reference planes before WF3 (bitwise reference)")
    import scipy
    meta["scipy"], meta["numpy"] = scipy.__version__, np.__version__
    if fem:
        import gmsh
        import skfem
        meta["gmsh"], meta["scikit-fem"] = gmsh.__version__, skfem.__version__
    arrays = compute(fem)
    fn = "lines_unburied_fem.npz" if fem else "lines_unburied.npz"
    np.savez_compressed(os.path.join(HERE, fn), meta=np.array(json.dumps(meta)), **arrays)
    print(len(arrays), "arrays,", commit)
