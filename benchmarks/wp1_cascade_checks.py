"""WP1 acceptance numbers: cascades of components vs products of transfer matrices (wrmodes 0.1 method).

Usage:  PYTHONPATH=src python3 benchmarks/wp1_cascade_checks.py [--quick]

Prints a JSON summary. The unit tests (tests/test_line.py, tests/test_components.py) check the same quantities
with fewer modes; this script runs the JINST section 7 sequence with the full basis of the line (all families)
as well as in the input family only.
"""
import sys, json, time
import numpy as np

import wrmodes as wr
from wrmodes import Line, SmoothBend, SechCurvature, ConstantCurvature
from wrmodes.components import map_basis
from wrmodes.gsm import from_transfer_matrix
from wrmodes.modes import eplane_frame_signs
from scipy.linalg import block_diag as _bd
from wrmodes import Guide, Straight

WR90 = Guide(0.9 * 0.0254, 0.4 * 0.0254)


# reference: products of full transfer matrices (the same construction as tests/test_line.py)
def bend_phi(f, basis, L, kappa, atol=1e-6):
    """Full 2N x 2N transfer matrix of a bend in the order of a local basis of propagating modes, assembled from
    the per-family transfer matrices of solve_transfer_matrix (wrmodes 0.1 method, error_control='full')."""
    N = len(basis)
    phi = np.zeros((2 * N, 2 * N), complex)
    for n in basis.families():
        idx = np.array(basis.family(n))
        sub = basis.subset(idx)
        m0 = sub.modes[0]
        _, r = wr.solve_transfer_matrix(f, basis.a, basis.b, (m0.kind, m0.m, m0.n), L, kappa,
                                        coupled_modes=sub.legacy_modes(f), atol=atol, error_control="full",
                                        profile=False)
        k = len(idx)
        P = r.transfer_matrix
        for bi, rows in enumerate((idx, idx + N)):
            for bj, cols in enumerate((idx, idx + N)):
                phi[np.ix_(rows, cols)] = P[bi * k:(bi + 1) * k, bj * k:(bj + 1) * k]
    return phi


def straight_phi(f, basis, L):
    b = basis.beta(f)
    return np.diag(np.concatenate([np.exp(-1j * b * L), np.exp(1j * b * L)]))


def to_global(phi_local, F):
    """Transfer matrix in the global frame from one in a local frame with a_local = F a_global."""
    FF = _bd(F, F)
    return FF.T @ phi_local @ FF


def reference_phi(f, gbasis, items):
    """Product of full transfer matrices for a list of ('bend', plane, L, kappa) / ('straight', L) in the global basis."""
    phi = np.eye(2 * len(gbasis), dtype=complex)
    for it in items:
        if it[0] == "straight":
            phi = straight_phi(f, gbasis, it[1]) @ phi
            continue
        _, plane, L, kappa = it
        if plane == "H":
            phi = bend_phi(f, gbasis, L, kappa) @ phi
        else:
            lb = map_basis(gbasis, "H", "E")
            F = eplane_frame_signs(gbasis, lb)
            phi = to_global(bend_phi(f, lb, L, kappa), F) @ phi
    return phi


def components(items):
    return [Straight(it[1]) if it[0] == "straight" else SmoothBend(it[2], it[3], plane=it[1]) for it in items]


def sequence_items(plane):
    return [("bend", plane, 0.6, SechCurvature(60, 0.6)), ("straight", 10.0),
            ("bend", plane, 0.4, SechCurvature(90, 0.4)), ("straight", 1.0),
            ("bend", plane, 0.4, -SechCurvature(90, 0.4))]


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
