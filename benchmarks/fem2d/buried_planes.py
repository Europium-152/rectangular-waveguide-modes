"""Cascades through reference planes inside a discontinuity (WF3): two 45 degree optical H-plane miters in WR-90.

`Line(Guide(W, H), [Miter(45), Straight(s), Miter(45)])` against the single FEM domain of the same geometry
(`fem2d.chained_miter_outline([45, 45], [s])`, COMSOL campaign 2 case E5), for the table of
docs/discrepancies/2026-11-15-gateb-45deg-miter-cascade-evanescent.md: s = 2a, 3a; 30, 56 GHz; cutoff factor
1, 1.2, 1.5, 2, 3. The reference plane of each miter lies delta = a inside its mirror (CONVENTIONS.md section 6).
Each cascade is computed twice: `buried_planes='absorb'` (WF3: the depth is taken from the straight) and 'keep'
(the cascade before WF3, at the reference planes).

Columns: TE1,0 transmission (dB), max |P_cascade - P_domain| over the TE1,0 column (every propagating output mode,
transmitted and reflected, as a fraction of the input power), the same over the whole propagating block, the largest
singular value of the propagating block, and the warnings `Line` issued.

Usage: python benchmarks/fem2d/buried_planes.py [--processes 2] [--mesh-density 6]  [--modes absorb,keep] [--freqs 30,56] [--cfs ...]
       -> data/buried_planes_d<m>[_<freqs>GHz]_<modes>.npz
"""
import argparse
import time
import warnings

import numpy as np

from _lit import meta, save, pmap

W, H = 22.86e-3, 10.16e-3
SPACERS = (2.0, 3.0)
FREQS = (30e9, 56e9)
CFS = (1.0, 1.2, 1.5, 2.0, 3.0)


def domain(args):
    sa, f, dens = args
    from wrmodes import Guide
    from wrmodes.fem2d import Coplanar2D, chained_miter_outline
    g = Guide(W, H)
    o, mirror, L = chained_miter_outline([45, 45], [sa * W], a=W)
    t = time.time()
    gd = Coplanar2D(o, H, guide=g, mirror=mirror, length=L, mesh_density=dens).gsm(f)
    return gd.S, [m.name for m in gd.basis1.modes], time.time() - t


def cascade(args):
    sa, f, cf, mode, dens = args
    import wrmodes as wr
    from wrmodes import Guide
    from wrmodes.fem2d import Miter
    g = Guide(W, H)
    line = wr.Line(g, [Miter(45, plane="H", mesh_density=dens), wr.Straight(sa * W),
                       Miter(45, plane="H", mesh_density=dens)])
    t = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            gc = line.gsm(f, cutoff_factor=cf, buried_planes=mode)
        except np.linalg.LinAlgError:
            return None, None, None, time.time() - t, ["LinAlgError"]
    p1, p2 = gc.propagating()
    S = np.block([[gc.S11[np.ix_(p1, p1)], gc.S12[np.ix_(p1, p2)]], [gc.S21[np.ix_(p2, p1)], gc.S22[np.ix_(p2, p2)]]])
    names = [m.name for m, p in zip(gc.basis1.modes, p1) if p]
    return S, names, float(np.linalg.norm(S, 2)), time.time() - t, sorted({type(x.message).__name__ for x in w})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--mesh-density", type=float, default=6.0)
    ap.add_argument("--modes", default="absorb,keep")
    ap.add_argument("--freqs", default=",".join(f"{f / 1e9:g}" for f in FREQS), help="GHz, comma separated")
    ap.add_argument("--cfs", default=",".join(f"{c:g}" for c in CFS))
    args = ap.parse_args()
    freqs = tuple(float(x) * 1e9 for x in args.freqs.split(","))
    cfs = tuple(float(x) for x in args.cfs.split(","))
    dens = args.mesh_density
    modes = args.modes.split(",")
    djobs = [(sa, f, dens) for sa in SPACERS for f in freqs]
    cjobs = [(sa, f, cf, mode, dens) for mode in modes for sa in SPACERS for f in freqs for cf in cfs]
    dres = dict(zip(djobs, pmap(domain, djobs, args.processes)))
    cres = dict(zip(cjobs, pmap(cascade, cjobs, args.processes)))
    arrays = {}
    print(f"mesh_density {dens:g}")
    print("| mode | s/a | f (GHz) | cf | TE1,0 (dB) | domain (dB) | max dP TE1,0 column | max dP block | sigma_max "
          "| warnings | time (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (sa, f, cf, mode, _), (S, names, sv, dt, wn) in cres.items():
        Sd, nd, _t = dres[(sa, f, dens)]
        N = len(nd)
        i0 = nd.index("TE1,0")
        dd = 20 * np.log10(abs(Sd[N + i0, i0]))
        key = f"{mode}_s{sa:g}a_{f / 1e9:g}GHz_cf{cf:g}"
        if S is None:
            print(f"| {mode} | {sa:g} | {f / 1e9:g} | {cf:g} | error | {dd:.4f} | – | – | – | {', '.join(wn)} | "
                  f"{dt:.1f} |")
            continue
        assert names == nd, (names, nd)
        dP = np.abs(np.abs(S) ** 2 - np.abs(Sd) ** 2)
        col = max(dP[:, i0].max(), 0.0)
        db = 20 * np.log10(abs(S[N + i0, i0]))
        arrays[key] = S
        print(f"| {mode} | {sa:g} | {f / 1e9:g} | {cf:g} | {db:.4f} | {dd:.4f} | {col:.1e} | {dP.max():.1e} | "
              f"{sv:.6g} | {', '.join(wn) or '–'} | {dt:.1f} |")
    for (sa, f, _), (Sd, nd, dt) in dres.items():
        arrays[f"domain_s{sa:g}a_{f / 1e9:g}GHz"] = Sd
        print(f"domain s = {sa:g}a, {f / 1e9:g} GHz: {dt:.1f} s")
    tag = "" if (freqs, cfs) == (FREQS, CFS) else "_" + "_".join(f"{f / 1e9:g}" for f in freqs) + "GHz"
    save(f"buried_planes_d{dens:g}{tag}_{'_'.join(modes)}", meta("benchmarks/fem2d/buried_planes.py",
                                          "two 45 deg optical H miters, cascade vs single domain (WF3)",
                                          spacers_over_a=SPACERS, freqs=freqs, cutoff_factors=cfs, mesh_density=dens,
                                          modes=modes), **arrays)


if __name__ == "__main__":
    main()
