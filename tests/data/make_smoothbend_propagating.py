"""Generate tests/data/smoothbend_propagating.npz: SmoothBend GSMs with propagating-mode bases, frozen before WF2.

WF2 (tasks/WF2-wave2-fixes.md, item 2) routes SmoothBend through VariableGuide when the bases contain evanescent
modes; the path with propagating modes only must stay bitwise unchanged. The file was generated with the code of
origin/main at the commit in its metadata (before WF2), from a checkout of that commit:

    PYTHONPATH=<checkout of that commit>/src python tests/data/make_smoothbend_propagating.py

`tests/test_components.py::test_propagating_path_is_bitwise_unchanged` recomputes the cases and requires exact
equality. Cases: WR-90 (perfect and aluminium walls), H- and E-plane, sech and constant curvature, families
'separate' and 'coupled', the component alone and inside a Line.
"""
import datetime
import json
import os
import subprocess

import numpy as np

import wrmodes as wr
from wrmodes.components import Guide, SmoothBend, SechCurvature, ConstantCurvature

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 22.86e-3, 10.16e-3


def cases():
    return [
        ("sech_H_40cm", Guide(W, H), SmoothBend(0.4, SechCurvature(90, 0.4), plane="H"), (18e9, 60e9)),
        ("sech_E_40cm", Guide(W, H), SmoothBend(0.4, SechCurvature(90, 0.4), plane="E"), (30e9,)),
        ("const_H_60cm_al", Guide(W, H, 3.5e7), SmoothBend(0.6, ConstantCurvature.from_angle(60, 0.6), plane="H"),
         (30e9,)),
        ("sech_E_40cm_al_coupled", Guide(W, H, 3.5e7),
         SmoothBend(0.4, SechCurvature(90, 0.4), plane="E", families="coupled"), (40e9,)),
        ("short_R2a", Guide(W, H), SmoothBend(W, ConstantCurvature(1 / (2 * W)), plane="H"), (20e9, 40e9)),
    ]


def compute():
    out = {}
    for name, g, b, freqs in cases():
        for f in freqs:
            out[f"{name}_{f / 1e9:g}_line"] = wr.Line(g, [b]).gsm(f).S
            out[f"{name}_{f / 1e9:g}_comp"] = b.with_guide(g).gsm(f).S
    return out


if __name__ == "__main__":
    src = os.path.dirname(os.path.dirname(os.path.abspath(wr.__file__)))
    try:
        commit = subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    meta = dict(wrmodes_version=wr.__version__, git_commit=commit, created=datetime.date.today().isoformat(),
                generator="tests/data/make_smoothbend_propagating.py",
                cases=[(n, repr(b), [float(f) for f in fs]) for n, _, b, fs in cases()],
                note="SmoothBend with propagating-mode bases before WF2 (bitwise reference)")
    arrays = compute()
    np.savez_compressed(os.path.join(HERE, "smoothbend_propagating.npz"), meta=np.array(json.dumps(meta)), **arrays)
    print(len(arrays), "arrays,", commit)
