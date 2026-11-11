"""Chained miters: single-domain outline (`chained_miter_outline`) against the `Line`-style cascade
Miter - Straight(s) - Miter (WP3b).

Two equal miters turning the same way, WR-90, H plane, 30 GHz (TE1,0-TE4,0 propagating), family 0. For each spacer
s (centre-line length between the miters' reference planes) and each truncation of the cascade (cutoff_factor
cf = 1: propagating modes only; 1.5, 2, 3: evanescent modes of the line's rule), the table gives
max |S_single - S_cascade| over the propagating block. The single domain needs no modal truncation between the
miters, so the difference measures how many evanescent modes the cascade needs at that spacer, down to the
discretization floor of the two FEM models (shown by the mesh density). Cases: 90 degree plain miters (a U-turn)
and 45 degree optical mirrors (COMSOL campaign 2, case E5: their mirrors reach a into the spacer, so the uniform
guide between them is s - 2a long, and they overlap below s = 2a, where the cascade is meaningless).

Usage: python benchmarks/fem2d/chained_miters.py [--processes 2]  -> data/chained_miters.npz
"""
import argparse
import numpy as np

from _lit import meta, save, pmap

W, H = 22.86e-3, 10.16e-3
CFS = (1.0, 1.5, 2.0, 3.0)


def run(args):
    f, ang, s, dens = args
    from wrmodes import Guide
    from wrmodes.fem2d import Coplanar2D, Miter, chained_miter_outline
    from wrmodes.gsm import cascade, straight
    g = Guide(W, H)
    o, mirror, L = chained_miter_outline([ang, ang], [s], a=W)
    B = g.basis(f, 1.0, plane="H")
    bp = B.subset(B.family(0))
    g1 = Coplanar2D(o, H, guide=g, mirror=mirror, length=L, mesh_density=dens).gsm(f, bp, bp)
    out = []
    for cf in CFS:
        bc = g.basis(f, cf, plane="H")
        bc = bc.subset(bc.family(0))
        gm = Miter(ang, guide=g, mesh_density=dens).gsm(f, bc, bc)
        try:
            gc = cascade(gm, straight(bc, f, s), gm)
        except np.linalg.LinAlgError:
            out.append(np.nan)
            continue
        ip = [bc.index(m) for m in bp.modes]
        Sc = np.block([[gc.S11[np.ix_(ip, ip)], gc.S12[np.ix_(ip, ip)]],
                       [gc.S21[np.ix_(ip, ip)], gc.S22[np.ix_(ip, ip)]]])
        out.append(np.max(np.abs(Sc - g1.S)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=None)
    args = ap.parse_args()
    f = 30e9
    plan = [(90, (0.25, 0.5, 1.0, 2.0, 3.0, 4.0), (6, 16, 32)), (45, (1.0, 2.0, 2.5, 3.0, 4.0), (6, 16))]
    jobs = [(f, ang, sa * W, dens) for ang, sas, denss in plan for dens in denss for sa in sas]
    res = pmap(run, jobs, args.processes)
    rows = []
    print("| miters | mesh_density | s / a | " + " | ".join("cf = %g" % c for c in CFS) + " |")
    print("| --- | --- | --- |" + " --- |" * len(CFS))
    for (ff, ang, s, dens), r in zip(jobs, res):
        rows.append([ang, dens, s / W] + list(r))
        print("| 2 x %d deg | %g | %.2f | " % (ang, dens, s / W) + " | ".join(
            "%.1e" % v if v == v else "fails (resonance)" for v in r) + " |")
    m = meta("benchmarks/fem2d/chained_miters.py", "single-domain chained miters vs cascade, max |dS| over the "
             "propagating block of family 0, WR-90, H plane", f=f, cutoff_factors=list(CFS),
             columns="angle, mesh_density, s/a, then one column per cutoff factor (nan: cascade singular)")
    print("saved", save("chained_miters.npz", m, table=np.array(rows)))


if __name__ == "__main__":
    main()
