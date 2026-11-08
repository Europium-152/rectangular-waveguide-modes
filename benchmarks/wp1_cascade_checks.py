"""WP1 acceptance numbers: cascades of components vs products of transfer matrices (wrmodes 0.1 method).

Usage:  PYTHONPATH=src python3 benchmarks/wp1_cascade_checks.py [--quick]

Prints a JSON summary. The unit tests (tests/test_line.py, tests/test_components.py) check the same quantities
with fewer modes; this script runs the JINST section 7 sequence with the full basis of the line (all families)
as well as in the input family only.
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import wrmodes as wr
from wrmodes import Line, SmoothBend, SechCurvature, ConstantCurvature
from wrmodes.components import map_basis
from wrmodes.gsm import from_transfer_matrix
from wrmodes.modes import eplane_frame_signs
from test_line import reference_phi, components, sequence_items, WR90


def sequence(plane, freqs, family_only):
    items = sequence_items(plane)
    line = Line(WR90, components(items))
    out = dict(max_dS_vs_full_phi=0.0, max_dS21_vs_T_only=0.0, max_S11_line=0.0, max_R_bends=0.0,
               max_reciprocity=0.0, max_unitarity=0.0, n_modes=[])
    for f in freqs:
        full = WR90.basis(f)
        keep = [i for i, m in enumerate(full) if (m.n == 0 if plane == "H" else m.m == 1)] if family_only else \
            list(range(len(full)))
        gb = full.subset(keep)
        g = line.gsm(f, basis1=gb)
        ref = from_transfer_matrix(reference_phi(f, gb, items), gb, f)
        out["max_dS_vs_full_phi"] = max(out["max_dS_vs_full_phi"], float(np.max(np.abs(g.S - ref.S))))
        out["max_reciprocity"] = max(out["max_reciprocity"], g.reciprocity_defect())
        out["max_unitarity"] = max(out["max_unitarity"], g.unitarity_defect())
        out["n_modes"].append(len(gb))
        a, b = WR90.dims(plane)
        nloc = 0 if plane == "H" else 1
        modes, T60, R60, beta = wr.transmission_matrix(f, a, b, nloc, 0.6, SechCurvature(60, 0.6))
        _, T90, R90, _ = wr.transmission_matrix(f, a, b, nloc, 0.4, SechCurvature(90, 0.4))
        _, T90m, R90m, _ = wr.transmission_matrix(f, a, b, nloc, 0.4, -SechCurvature(90, 0.4))
        Tseq = T90m @ wr.straight_section(beta, 1.0) @ T90 @ wr.straight_section(beta, 10.0) @ T60
        lb = map_basis(gb, "H", plane)
        F = eplane_frame_signs(gb, lb) if plane == "E" else np.eye(len(gb))
        S21l = F @ g.S21 @ F.T
        names = [m[0] for m in modes]
        idx = [lb.index(n) for n in names]
        out["max_dS21_vs_T_only"] = max(out["max_dS21_vs_T_only"], float(np.max(np.abs(S21l[np.ix_(idx, idx)] - Tseq))))
        out["max_S11_line"] = max(out["max_S11_line"], float(np.max(np.abs(g.S11))))
        out["max_R_bends"] = max(out["max_R_bends"], float(max(np.abs(R).max() for R in (R60, R90, R90m))))
    return out


def main(quick=False):
    res = {}
    f = 40e9
    items = [("bend", "H", 0.15, ConstantCurvature(1 / 0.25)), ("straight", 0.7),
             ("bend", "H", 0.12, ConstantCurvature(-1 / 0.3))]
    g = Line(WR90, components(items)).gsm(f)
    ref = from_transfer_matrix(reference_phi(f, g.basis1, items), g.basis1, f)
    res["same_plane_cascade_40GHz"] = dict(n_modes=len(g.basis1), max_dS=float(np.max(np.abs(g.S - ref.S))))

    for f in (25e9, 40e9, 60e9):
        L, kappa = 0.4, SechCurvature(90, 0.4)
        g = Line(WR90, [SmoothBend(L, kappa, plane="E")]).gsm(f)
        gb = g.basis1
        lb = map_basis(gb, "H", "E")
        F = eplane_frame_signs(gb, lb)
        S21l, S11l = F @ g.S21 @ F.T, F @ g.S11 @ F.T
        d = 0.0
        for n in lb.families():
            idx = lb.family(n)
            _, T, R, _ = wr.transmission_matrix(f, WR90.h, WR90.w, n, L, kappa)
            d = max(d, np.max(np.abs(S21l[np.ix_(idx, idx)] - T)), np.max(np.abs(S11l[np.ix_(idx, idx)] - R)))
        res[f"E_plane_bend_vs_0.1_{f / 1e9:g}GHz"] = dict(n_modes=len(gb), max_dS=float(d),
                                                         reciprocity=g.reciprocity_defect(),
                                                         unitarity=g.unitarity_defect())

    freqs = np.linspace(18e9, 90e9, 10)
    for plane in ("H", "E"):
        t = time.time()
        res[f"sequence_{plane}_input_family"] = sequence(plane, freqs, True)
        res[f"sequence_{plane}_input_family"]["seconds"] = time.time() - t
        if not quick:
            t = time.time()
            res[f"sequence_{plane}_all_families"] = sequence(plane, freqs, False)
            res[f"sequence_{plane}_all_families"]["seconds"] = time.time() - t
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main("--quick" in sys.argv)
