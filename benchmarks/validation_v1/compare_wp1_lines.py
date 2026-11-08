"""Compare WP1's Line / SmoothBend / Straight / FrameChange with the WP7 global-ODE reference.

    python benchmarks/validation_v1/compare_wp1_lines.py [--fast]

Acceptance (tasks/WP1-cascade.md): for the three reference lines, |S|^2 of every output mode (transmitted and
reflected, every propagating input mode) agrees with the reference to 1e-6 of the input power. WP1's bends are built
with SmoothBend(atol=1e-10) so that their own error is far below that. The FrameChange check compares WP1's matrix
with the relabelling derived in ref_frames (signed permutation, exact).

Exits with status 0 and a message when WP1 is not merged yet. Writes results/wp1_comparison.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ref_common  # noqa: E402
import ref_frames  # noqa: E402
import ref_interfaces as I  # noqa: E402
import ref_line  # noqa: E402
from wrmodes.modes import Mode, ModeBasis  # noqa: E402

TOL_POWER = 1e-6


def _index_map(basis, names):
    """Row/column of each reference mode name in a GSM basis (KeyError if WP1's basis lacks a mode)."""
    return np.array([basis.index(Mode.parse(n)) for n in names])


def compare_gsm(g, ref, names):
    """Largest |dP| between a GSM and reference blocks {S11, S21, S12, S22} over the reference's modes."""
    i1, i2 = _index_map(g.basis1, names), _index_map(g.basis2, names)
    blocks = {"S11": g.S11[np.ix_(i1, i1)], "S21": g.S21[np.ix_(i2, i1)],
              "S12": g.S12[np.ix_(i1, i2)], "S22": g.S22[np.ix_(i2, i2)]}
    out = {}
    for k, B in blocks.items():
        out[k] = float(np.max(np.abs(np.abs(B) ** 2 - np.abs(ref[k]) ** 2)))
    extra = (len(g.basis1) - len(names), len(g.basis2) - len(names))
    return out, extra


def compare_lines(names=None, freqs=None, atol=1e-10):
    data = ref_common.load_npz(os.path.join(ref_common.DATA, "line_reference.npz"))
    w, h = ref_line.WR90
    lines = ref_line.reference_lines()
    results = []
    for name, segs in lines.items():
        if names and name not in names:
            continue
        line = I.build_line(segs, w, h, atol=atol)
        for f in ref_line.REFERENCE_FREQUENCIES[name]:
            if freqs and f not in freqs:
                continue
            key = f"{name}/{f / 1e9:g}"
            ref = {k: data[f"{key}/{k}"] for k in ("S11", "S21", "S12", "S22")}
            modes = [str(m) for m in data[f"{key}/modes"]]
            g = line.gsm(f, cutoff_factor=1.0)
            dP, extra = compare_gsm(g, ref, modes)
            worst = max(dP.values())
            results.append(dict(line=name, f=f, dP=dP, worst=worst, extra_modes=extra, ok=worst <= TOL_POWER))
            print(f"{key:24s} max |dP| = {worst:.2e}  ({'pass' if worst <= TOL_POWER else 'FAIL'})  " +
                  "  ".join(f"{k} {v:.1e}" for k, v in dP.items()))
    return results


def check_frame_change(f=40e9):
    """WP1 FrameChange('H', 'E') against the derived relabelling; H -> E -> H must be the identity.

    Uses the component's default bases (Component.default_basis) and matches modes by label.
    """
    FrameChange = I.get("FrameChange")
    fc = FrameChange("H", "E")
    g = fc.gsm(f)
    b1, b2 = g.basis1, g.basis2
    report = {"basis1": repr(b1), "basis2": repr(b2)}
    gb = ModeBasis(b1.a, b1.b, b1.modes, sort=False)
    F = ref_frames.eplane_matrix(gb)
    lb = ref_frames.eplane_local_basis(gb)
    if np.isclose(b2.a, b1.b) and np.isclose(b2.b, b1.a):
        i2 = _index_map(b2, lb.names)
        report["S21_vs_derived"] = float(np.max(np.abs(g.S21[i2, :] - F)))
        report["S12_vs_derived"] = float(np.max(np.abs(g.S12[:, i2] - F.T)))
        report["S11_S22"] = float(max(np.max(np.abs(g.S11)), np.max(np.abs(g.S22))))
    else:
        report["note"] = (f"FrameChange port 2 basis has a = {b2.a}, b = {b2.b}; expected the E-plane local guide "
                          f"a = h, b = w (CONVENTIONS.md 2). Cannot compare labels: file a discrepancy report.")
    try:
        back = FrameChange("E", "H").gsm(f)
        gg = g @ back
        report["HEH_identity"] = float(max(np.max(np.abs(gg.S21 - np.eye(len(gg.basis1)))),
                                           np.max(np.abs(gg.S11)), np.max(np.abs(gg.S22))))
    except Exception as exc:                       # report, do not hide
        report["HEH_identity_error"] = repr(exc)
    print("FrameChange:", report)
    return report


def main(argv):
    fast = "--fast" in argv
    try:
        I.get("Line")
    except I.InterfaceMissing as exc:
        print(f"WP1 is not available yet, nothing to compare: {exc}")
        return 0
    out = {}
    out["lines"] = compare_lines(freqs=[20e9] if fast else None)
    try:
        out["frame_change"] = check_frame_change()
    except I.InterfaceMissing as exc:
        out["frame_change"] = {"skipped": str(exc)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "wp1_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    bad = [r for r in out["lines"] if not r["ok"]]
    print(f"{len(out['lines']) - len(bad)} of {len(out['lines'])} line cases within {TOL_POWER:g} of input power")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
