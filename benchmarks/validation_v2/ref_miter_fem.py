"""WP7b reference for WP4b (mode-matching miters) and the FEM miters of WP3/WP3b: the merged 2D modal-port FEM
(`wrmodes.fem2d.Coplanar2D`, wave 1) on 90 deg miter outlines built here from CONVENTIONS.md section 6.

Geometry (in-plane side a; H plane: a = w = 22.86 mm, b = h = 10.16 mm; E plane: a = h, b = w): arm 1 along +y with
u = +x, arm 2 along -x with u = +y (port 1 frame rotated about v by 90 deg); the arm axes meet at the origin, the
inner corner is O = (-a/2, -a/2), the reference planes are y = -a/2 (port 1) and x = -a/2 (port 2), both through O.
The mirror is the line perpendicular to the bisector (1, 1)/sqrt(2) at distance d a from O (Alessandri's d);
d = 1/sqrt(2) is the optical mirror (through the outer walls at the reference planes), d = sqrt(2) the sharp
corner. The mesh is built symmetric about the bisector, so S11 = S22 to rounding.

Card WP7b item 3: d/a = 0.707, 0.8, 0.9, 1.0, H and E plane, WR-90 at 10-90 GHz every 2 GHz plus the near-cutoff
points of Gate A (26.3, 52.5, 53, 53.5 GHz); port bases = the line's rule (`Guide.basis(f, 1, plane)`, propagating
modes). Convergence: mesh density 10 (main) against 16 and more evanescent port modes at 10, 30, 50, 70, 90 GHz.

    PYTHONPATH=src:benchmarks/validation_v2 python benchmarks/validation_v2/ref_miter_fem.py [--plane H E] [--d ...]
"""
import argparse
import json

import numpy as np

from wrmodes.components import Guide
from wrmodes.fem2d import Coplanar2D, FEMOptions, Outline, PortPlane

import vcommon as vc

FREQS = sorted(set(np.arange(10, 91, 2).tolist() + [26.3, 52.5, 53.0, 53.5]))
CONV_FREQS = [10, 30, 50, 70, 90]
CHAMFERS = (1 / np.sqrt(2), 0.8, 0.9, 1.0)


def miter90_outline(a, d):
    """Core outline of the 90 deg miter between its reference planes (see module docstring)."""
    h = a / 2
    s = d * a * np.sqrt(2) - a          # mirror: x + y = s
    pts = [(-h, -h), (h, -h)]
    ym = s - h                          # mirror meets the outer wall x = +a/2 at y = ym
    if ym > -h + 1e-12 * a:
        pts.append((h, min(ym, h)))
    xm = s - h                          # and the outer wall y = +a/2 at x = xm
    if xm > -h + 1e-12 * a and ym < h - 1e-12 * a:
        pts.append((xm, h))
    elif ym >= h - 1e-12 * a:
        pts.append((h, h))              # sharp corner
    pts.append((-h, h))
    clean = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 1e-12 * a:
            clean.append(p)
    return Outline(vertices=tuple(clean), ports=(PortPlane((-h, -h), (h, -h)), PortPlane((-h, -h), (-h, h))))


def mirror_line(a):
    return ((-a / 2, -a / 2), (1 / np.sqrt(2), 1 / np.sqrt(2)))


def component(plane, d, density=10.0, n_evanescent=10):
    g = Guide(vc.W90, vc.H90)
    a, b = g.dims(plane)
    return Coplanar2D(miter90_outline(a, d), b=b, plane=plane, mirror=mirror_line(a), length=a,
                      options=FEMOptions(mesh_density=density, n_evanescent=n_evanescent))


def run(plane, d, freqs, conv_freqs):
    arrays, rows = {}, []
    g = Guide(vc.W90, vc.H90)
    for f in freqs:
        basis = g.basis(f * 1e9, 1.0, plane)
        settings = [("main", 10.0, 10)]
        if f in conv_freqs:
            settings += [("fine", 16.0, 10), ("more_ports", 10.0, 20)]
        for tag, dens, ne in settings:
            c = component(plane, d, dens, ne)
            t = vc.Timer()
            G = c.gsm(f * 1e9, basis, basis)
            dt = t()
            key = f"{plane}_d{d:.4f}_f{f:g}_{tag}"
            arrays.update(vc.gsm_arrays(key, G))
            i0 = basis.names.index("TE1,0" if plane == "H" else "TE0,1")
            row = dict(key=key, f=f, tag=tag, density=dens, n_evanescent=ne, seconds=round(dt, 2),
                       ndof=int(sum(x[1] for x in G.fem_info["ndof"])),
                       s21_main_db=float(vc.db(G.S21[i0, i0])), n_modes=len(basis.modes))
            rows.append(row)
            print(json.dumps(row), flush=True)
    return arrays, rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plane", nargs="*", default=["H", "E"])
    p.add_argument("--d", nargs="*", type=float, default=list(CHAMFERS))
    p.add_argument("--freqs", nargs="*", type=float, default=FREQS)
    a = p.parse_args()
    for plane in a.plane:
        for d in a.d:
            arrays, rows = run(plane, d, a.freqs, CONV_FREQS)
            vc.save_npz(f"miter_fem_{plane}_d{d:.4f}.npz", __file__, arrays, plane=plane, chamfer=d, freqs=a.freqs,
                        rows=rows, conv_freqs=CONV_FREQS, fem_order=3,
                        method="wrmodes.fem2d.Coplanar2D on WP7b outlines (symmetric mesh)", guide=[vc.W90, vc.H90])
