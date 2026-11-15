"""WP7b, wave 1 leftover (WP1 open issue: evanescent modes of bends are uncoupled): the error of a `Line`
[short SmoothBend, Miter] against one `Coplanar2D` domain containing both (card WP7b item 5).

Geometry (H plane, WR-90): a circular bend of centre-line radius R = 2a or 10a (a = 22.86 mm), length L = 0 ... 2a
(angle L / R), turning towards -u, immediately followed (no straight) by the plain (optical-mirror) 90 deg miter
turning the same way, whose port-1 reference plane is the end plane of the arc. The single-domain reference is the
FEM outline of both (arc walls exact, miter as in ref_miter_fem.py), between the arc's port-1 plane and the miter's
port-2 plane. The line is `wrmodes.Line(Guide, [SmoothBend(L, kappa = 1/R), fem2d.Miter(90)])` with
cutoff_factor 1 (propagating modes only between the two) and 3 (evanescent modes passed through the bend as
uncoupled straight modes, WP1). Quantity: max |S_line - S_FEM| and max | |S_line|^2 - |S_FEM|^2 | over the
propagating block, and the TE1,0 transmission difference in dB, at 20 and 40 GHz (60 GHz was dropped: the
fine-mesh runs exceeded the 7 GB of the container).

The total error has two parts, separated here: the coupled-mode model of the bend itself (SmoothBend against the
FEM of the arc alone, `bend_*`), and the cascade at zero distance (the Line [FEM arc, FEM miter] with the same
cutoff factors against the single domain, `fem_cascade_*`), which is the evanescent-mode effect proper.
"""
import json

import numpy as np

import wrmodes as wr
from wrmodes.components import ConstantCurvature, Guide, SmoothBend
from wrmodes.fem2d import Coplanar2D, FEMOptions, Miter, Outline, PortPlane

import vcommon as vc
from ref_miter_fem import miter90_outline

W, H = vc.W90, vc.H90
RADII = (2 * W, 10 * W)
LENGTHS = [0.0, 0.25 * W, 0.5 * W, W, 2 * W]
FREQS = [20e9, 40e9]


def arc_miter_outline(a, R, alpha, d=1 / np.sqrt(2)):
    C = np.array([-R, 0.0])
    P = lambda r, phi: C + r * np.array([np.cos(phi), np.sin(phi)])
    if alpha == 0:
        return miter90_outline(a, d)
    uhat = np.array([np.cos(alpha), np.sin(alpha)])
    t = np.array([-np.sin(alpha), np.cos(alpha)])
    Q = P(R, alpha)
    m = miter90_outline(a, d)
    G = lambda p: tuple(Q + p[0] * uhat + (p[1] + a / 2) * t)
    mv = [G(p) for p in m.vertices]      # (-h,-h) inner, (h,-h) outer, ..., (-h, h) port-2 outer
    verts = [tuple(P(R - a / 2, 0.0)), tuple(P(R + a / 2, 0.0))] + mv[1:] + [mv[0]]
    arcs = {1: tuple(C), len(verts) - 1: tuple(C)}
    p2 = PortPlane(mv[0], mv[-1])
    return Outline(vertices=tuple(verts), arcs=arcs,
                   ports=(PortPlane(tuple(P(R - a / 2, 0.0)), tuple(P(R + a / 2, 0.0))), p2))


def _blk(G, names):
    idx = [G.basis1.names.index(n) for n in names]
    b = lambda X: np.asarray(X)[np.ix_(idx, idx)]
    return np.block([[b(G.S11), b(G.S12)], [b(G.S21), b(G.S22)]])


def _diff(row, tag, A, B, names):
    i0 = names.index("TE1,0")
    n = len(names)
    row[f"{tag}_max_dS"] = float(np.max(abs(A - B)))
    row[f"{tag}_max_dP"] = float(np.max(abs(abs(A) ** 2 - abs(B) ** 2)))
    row[f"{tag}_dTE10_dB"] = float(vc.db(A[n + i0, i0]) - vc.db(B[n + i0, i0]))


OUT = "evanescent_bend_miter.npz"
SKIP = {"R2a_f40_L2a", "R10a_f40_L2a"}   # OOM-killed (> 6 GB) in the 7 GB container; not computed


def _save(arrays, rows):
    vc.save_npz(OUT, __file__, arrays, rows=rows, radii=RADII, lengths=LENGTHS, freqs=FREQS,
                miter="optical mirror, 90 deg, H plane", method="single Coplanar2D domain vs Line[SmoothBend, Miter]")


def run(density=10.0):
    """Saves after every row and resumes from a partial npz (long runs, 7 GB container)."""
    import os
    from wrmodes.fem2d import arc_bend_outline
    g = Guide(W, H)
    arrays, rows = {}, []
    if os.path.exists(os.path.join(vc.DATA, OUT)):
        old = vc.load_npz(OUT)
        rows = old["meta"]["rows"]
        arrays = {k: v for k, v in old.items() if k != "meta"}
    done = {r["key"] for r in rows}
    for R in RADII:
        for f in FREQS:
            basis = g.basis(f, 1.0, "H")
            names = basis.names
            for L in LENGTHS:
                key = f"R{R / W:g}a_f{f / 1e9:g}_L{L / W:g}a"
                if key in done or key in SKIP:
                    continue
                alpha = L / R
                out = arc_miter_outline(W, R, alpha)
                t = vc.Timer()
                S_ref = Coplanar2D(out, b=H, plane="H", length=L + W, options=FEMOptions(mesh_density=density)).gsm(f, basis, basis)
                t_ref = t()
                S_fine = Coplanar2D(out, b=H, plane="H", length=L + W,
                                    options=FEMOptions(mesh_density=1.5 * density)).gsm(f, basis, basis)
                arrays.update(vc.gsm_arrays(key + "_fem", S_fine))
                Sr = _blk(S_fine, names)
                row = dict(key=key, R_over_a=R / W, f=f, L_over_a=L / W, fem_seconds=round(t_ref, 1),
                           fem_mesh_change=float(np.max(abs(_blk(S_ref, names) - Sr))))
                miter = Miter(90, plane="H")
                for cf in (1.0, 3.0):
                    comps = ([SmoothBend(L, ConstantCurvature(1 / R), plane="H")] if L > 0 else []) + [miter]
                    Sl = _blk(wr.Line(g, comps).gsm(f, cutoff_factor=cf), names)
                    _diff(row, f"line_cf{cf:g}", Sl, Sr, names)
                    arrays[f"{key}_line_cf{cf:g}"] = Sl
                    if L > 0:
                        arc = Coplanar2D(arc_bend_outline(W, R, alpha), b=H, plane="H", length=L,
                                         options=FEMOptions(mesh_density=1.5 * density))
                        Sc = _blk(wr.Line(g, [arc, miter]).gsm(f, cutoff_factor=cf), names)
                        _diff(row, f"fem_cascade_cf{cf:g}", Sc, Sr, names)
                if L > 0:
                    arcf = Coplanar2D(arc_bend_outline(W, R, alpha), b=H, plane="H", length=L,
                                      options=FEMOptions(mesh_density=1.5 * density))
                    Sb = _blk(wr.Line(g, [SmoothBend(L, ConstantCurvature(1 / R), plane="H")]).gsm(f), names)
                    Sa = _blk(wr.Line(g, [arcf]).gsm(f), names)
                    _diff(row, "bend", Sb, Sa, names)
                rows.append(row)
                _save(arrays, rows)
                print(json.dumps({k: (float(f"{v:.3g}") if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    return arrays, rows


if __name__ == "__main__":
    arrays, rows = run()
    _save(arrays, rows)
