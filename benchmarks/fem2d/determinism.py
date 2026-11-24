"""WF4 item 6: how far the 2D FEM (`wrmodes.fem2d`) results depend on the gmsh version, the platform and the mesh
node order.

    python benchmarks/fem2d/determinism.py run [--tag TAG]      -> data/determinism_<gmsh version>[_TAG].npz
    python benchmarks/fem2d/determinism.py compare A.npz B.npz  (prints the largest differences)

`run` computes a fixed set of GSMs (default options) and stores them with the gmsh, scikit-fem, NumPy, SciPy versions
and the platform in `meta`; it also solves each case a second time in the same process (repeatability) and once
with the nodes of the mesh renumbered by a fixed random permutation (the permutation is applied to the gmsh mesh before
assembly), and once at 1.5 times the mesh density (the discretization error scale). Cases:
90 degree H-plane miter (WR-90, 30 GHz), E-plane miter chamfer d/h = 0.87 (24 GHz, COMSOL E1), 45 degree optical
H-plane miter (30 GHz), two 45 degree optical miters with s = 3a as one domain (30 GHz, COMSOL E5).

Run with two gmsh versions (e.g. in two environments) and compare the files. The measured figures are in
docs/reports/WF4.md and the docstring of `wrmodes.fem2d`.
"""
import argparse
import json
import os
import platform
import sys

import numpy as np

from _lit import meta, save

W, H = 22.86e-3, 10.16e-3


def cases():
    from wrmodes.components import Guide
    from wrmodes.fem2d import Miter, Coplanar2D, chained_miter_outline
    g = Guide(W, H)
    out, mirror, length = chained_miter_outline([45, 45], [3 * W], a=W)
    return [("miter90_H_30GHz", Miter(90, plane="H", guide=g), 30e9),
            ("miterE_d087_24GHz", Miter(90, chamfer=0.87, plane="E", guide=g), 24e9),
            ("miter45_H_30GHz", Miter(45, plane="H", guide=g), 30e9),
            ("two45_s3a_30GHz", Coplanar2D(out, b=H, plane="H", guide=g, mirror=mirror, length=length), 30e9)]


def run(tag):
    import gmsh
    import scipy
    import skfem
    import wrmodes
    import wrmodes.fem2d.solver as solver
    arrays, rows = {}, []
    for name, comp, f in cases():
        g1 = comp.gsm(f)
        g2 = comp.gsm(f)
        fine = comp.with_options(mesh_density=1.5 * comp.options.mesh_density).gsm(f)
        perm = _permuted_gsm(comp, f, solver)
        arrays[f"{name}_S"] = g1.S
        arrays[f"{name}_S_fine"] = fine.S
        rows.append(dict(case=name, f=f, ndof=int(sum(n for _, n in g1.fem_info["ndof"])),
                         repeat=float(np.max(np.abs(g1.S - g2.S))),
                         node_order=None if perm is None else float(np.max(np.abs(g1.S - perm.S))),
                         node_order_power=None if perm is None else float(np.max(np.abs(np.abs(g1.S) ** 2 - np.abs(perm.S) ** 2))),
                         mesh_x1p5=float(np.max(np.abs(g1.S - fine.S))),
                         mesh_x1p5_power=float(np.max(np.abs(np.abs(g1.S) ** 2 - np.abs(fine.S) ** 2)))))
        print(json.dumps(rows[-1]), flush=True)
    versions = dict(gmsh=gmsh.__version__, scikit_fem=getattr(skfem, "__version__", "?"), numpy=np.__version__,
                    scipy=scipy.__version__, python=sys.version.split()[0], platform=platform.platform(),
                    machine=platform.machine())
    m = meta("benchmarks/fem2d/determinism.py", "FEM determinism: gmsh version, platform, node order", rows=rows,
             versions=versions)
    name = f"determinism_{gmsh.__version__}" + (f"_{tag}" if tag else "") + ".npz"
    print(save(name, m, **arrays))


def _permuted_gsm(comp, f, solver):
    """GSM with the gmsh mesh nodes renumbered by a fixed random permutation (the same mesh in another node order,
    as another platform or gmsh build may return it): `geometry._gmsh_mesh` is wrapped for this one call."""
    import wrmodes.fem2d.geometry as geo
    orig = geo._gmsh_mesh
    rng = np.random.default_rng(12345)

    def wrapped(*args, **kwargs):
        p, t, region = orig(*args, **kwargs)
        perm = rng.permutation(p.shape[1])
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        return p[:, perm], inv[t], region
    geo._gmsh_mesh = wrapped
    try:
        return comp.gsm(f)
    finally:
        geo._gmsh_mesh = orig


def compare(a, b):
    za, zb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    ma, mb = json.loads(str(za["meta"])), json.loads(str(zb["meta"]))
    print(f"A: {ma['parameters']['versions']}\nB: {mb['parameters']['versions']}")
    for k in sorted(za.files):
        if k == "meta" or k not in zb.files:
            continue
        Sa, Sb = za[k], zb[k]
        if Sa.shape != Sb.shape:
            print(f"{k:28s} shapes differ {Sa.shape} {Sb.shape}")
            continue
        print(f"{k:28s} max |dS| {np.max(np.abs(Sa - Sb)):.2e}   max |d|S|^2| {np.max(np.abs(abs(Sa) ** 2 - abs(Sb) ** 2)):.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "compare"])
    p.add_argument("files", nargs="*")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    run(a.tag) if a.mode == "run" else compare(*a.files)
