"""WP7b reference for WP5 (one-family tapers and the strangled E-plane bend): the merged 2D modal-port FEM
(`wrmodes.fem2d.Coplanar2D`, wave 1) on polygon outlines built here.

A component that does not vary along v (an H-plane taper of a only, or an E-plane taper of the E-plane side with
an E-plane arc) splits exactly into one scalar 2D problem per family (CONVENTIONS.md section 7), so the 2D FEM is an
exact, independent method (no cross-section expansion, no coupled-mode coefficients). Cases (card WP7b item 1):

* `H_linear_L100`, `H_linear_L200`: the T2b section 9 linear H-plane taper 22.86 -> 45 mm (b = 10.16 mm),
  symmetric about the axis, L = 0.1 and 0.2 m, 15 GHz.
* `E_rc50`: raised-cosine E-plane taper WR-90 -> 22.86 x 1.6 mm over 50 mm (symmetric about the axis), 18, 30, 60
  and 90 GHz; local E frame (a = h along u, b = w along v).
* `strangled`: the COMSOL campaign 2 E3 geometry: that taper, a 90 deg E-plane arc of centre-line radius 50 mm in the
  1.6 mm guide (kappa > 0, turning towards -u), and the mirror taper; 18, 30, 40, 60, 90 GHz.

Port planes are the end planes of the tapers (CONVENTIONS.md section 6); the port bases are the line's rule at the
end guides (`Guide.basis(f, 1, plane)`, propagating modes). Curved walls of the tapers are polylines with
`n_seg` circular arcs per wall (each through three samples of the profile; geometric error O(s^4)); the arc walls are exact circular arcs of the
outline. Convergence: mesh density (elements per transverse wavelength) and the number of chords; the reference
value is the finest run and its error estimate the change from the next finer run.

    PYTHONPATH=src python benchmarks/validation_v2/ref_taper_fem.py [--cases ...] [--quick]
"""
import argparse
import json

import numpy as np

from wrmodes.components import Guide
from wrmodes.fem2d import Coplanar2D, FEMOptions, Outline, PortPlane

import vcommon as vc

W, H = vc.W90, vc.H90
B_FLAT, L_TAPER, R_ARC = 1.6e-3, 50e-3, 50e-3


def rc(t):
    return 0.5 * (1 - np.cos(np.pi * t))


def linear_h_outline(a1, a2, L):
    return Outline(vertices=((-a1 / 2, 0.0), (a1 / 2, 0.0), (a2 / 2, L), (-a2 / 2, L)),
                   ports=(PortPlane((-a1 / 2, 0.0), (a1 / 2, 0.0)), PortPlane((-a2 / 2, L), (a2 / 2, L))))


def circumcentre(p, q, r):
    """Centre of the circle through three points, or None when they are (nearly) collinear."""
    (ax, ay), (bx, by), (cx, cy) = p, q, r
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    span = np.hypot(cx - ax, cy - ay)
    if abs(d) < 1e-12 * span ** 2:
        return None
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    if np.hypot(ux - ax, uy - ay) > 1e4 * span:   # radius 1e4 times the arc span: a straight edge
        return None
    return (ux, uy)


def arc_chain(points):
    """Vertices and arcs (offsets into the chain) of a curve sampled at an odd number of points: one circular arc
    through every three consecutive samples (geometric error O(s^4) instead of the O(s^2) of chords)."""
    assert len(points) % 2 == 1
    verts, arcs = [], {}
    for j in range(0, len(points) - 1, 2):
        c = circumcentre(points[j], points[j + 1], points[j + 2])
        if c is not None:
            arcs[len(verts)] = c
        verts.append(points[j])
    verts.append(points[-1])
    return verts, arcs


class Chain:
    """Builds an outline from consecutive pieces (each piece starts where the previous ended)."""

    def __init__(self):
        self.verts, self.arcs = [], {}

    def add(self, verts, arcs=None):
        start = len(self.verts)
        if self.verts:
            assert np.allclose(self.verts[-1], verts[0], atol=1e-12), (self.verts[-1], verts[0])
            verts, start = verts[1:], start - 1
            arcs = {k - 0: v for k, v in (arcs or {}).items()}
        for k, v in (arcs or {}).items():
            self.arcs[start + k] = v
        self.verts.extend(tuple(map(float, p)) for p in verts)

    def close(self):
        if np.allclose(self.verts[-1], self.verts[0], atol=1e-12):
            self.verts.pop()
        return tuple(self.verts), dict(self.arcs)


def taper_walls(a1, a2, L, n_seg, profile=rc):
    """Half-widths of a symmetric taper at n_seg + 1 points z_k (port 1 at z = 0)."""
    z = np.linspace(0.0, L, n_seg + 1)
    return z, 0.5 * (a1 + (a2 - a1) * profile(z / L))


def rc_taper_outline(a1, a2, L, n_seg):
    """Symmetric raised-cosine taper, n_seg circular arcs per wall."""
    z, h = taper_walls(a1, a2, L, 2 * n_seg)
    ch = Chain()
    ch.add([(-h[0], 0.0), (h[0], 0.0)])                               # port 1 (u = -a/2 -> +a/2)
    ch.add(*arc_chain([(h[k], z[k]) for k in range(len(z))]))        # u = +a/2 wall
    ch.add([(h[-1], L), (-h[-1], L)])                                 # port 2
    ch.add(*arc_chain([(-h[k], z[k]) for k in range(len(z))][::-1]))  # u = -a/2 wall back to port 1
    verts, arcs = ch.close()
    return Outline(vertices=verts, arcs=arcs,
                   ports=(PortPlane((-h[0], 0.0), (h[0], 0.0)), PortPlane((-h[-1], L), (h[-1], L))))


def strangled_outline(a1, af, Lt, R, n_seg):
    """Input taper along +y (u = +x at port 1), arc of centre (-R, Lt) turning towards -x by 90 deg, output taper
    along -x. Port 2 frame: port 1 transported, u on the outer side (larger y). n_seg circular arcs per taper wall;
    the bend walls are exact arcs."""
    z, h = taper_walls(a1, af, Lt, 2 * n_seg)
    cx, cy = -R, Lt
    ch = Chain()
    ch.add([(-h[0], 0.0), (h[0], 0.0)])                                           # port 1
    ch.add(*arc_chain([(h[k], z[k]) for k in range(len(z))]))                    # outer wall, input taper
    ch.add([(af / 2, Lt), (cx, cy + R + af / 2)], {0: (cx, cy)})                 # outer wall of the bend
    hr = h[::-1]                                                                 # output taper: af -> a1
    ch.add(*arc_chain([(cx - z[k], cy + R + hr[k]) for k in range(len(z))]))     # outer wall, output taper
    p2_outer, p2_inner = (cx - Lt, cy + R + a1 / 2), (cx - Lt, cy + R - a1 / 2)
    ch.add([p2_outer, p2_inner])                                                 # port 2
    ch.add(*arc_chain([(cx - z[k], cy + R - hr[k]) for k in range(len(z))][::-1]))  # inner wall, output taper
    ch.add([(cx, cy + R - af / 2), (-af / 2, Lt)], {0: (cx, cy)})                # inner wall of the bend
    ch.add(*arc_chain([(-h[k], z[k]) for k in range(len(z))][::-1]))             # inner wall, input taper
    verts, arcs = ch.close()
    return Outline(vertices=verts, arcs=arcs,
                   ports=(PortPlane((-h[0], 0.0), (h[0], 0.0)), PortPlane(p2_inner, p2_outer)))


def case_setup(case):
    """(outline builder of n_seg, b, plane, guide1, guide2, freqs)."""
    if case.startswith("H_linear"):
        L = 0.1 if case.endswith("100") else 0.2
        return (lambda n: linear_h_outline(W, 45e-3, L)), H, "H", Guide(W, H), Guide(45e-3, H), [15e9]
    if case == "E_rc50":
        return (lambda n: rc_taper_outline(H, B_FLAT, L_TAPER, n)), W, "E", Guide(W, H), Guide(W, B_FLAT), \
            [18e9, 30e9, 60e9, 90e9]
    if case == "strangled":
        return (lambda n: strangled_outline(H, B_FLAT, L_TAPER, R_ARC, n)), W, "E", Guide(W, H), Guide(W, H), \
            [18e9, 30e9, 40e9, 60e9, 90e9]
    raise ValueError(case)


def run(case, densities, segs, out):
    build, b, plane, g1, g2, freqs = case_setup(case)
    arrays, runs = {}, []
    for f in freqs:
        b1, b2 = g1.basis(f, 1.0, plane), g2.basis(f, 1.0, plane)
        settings = [(d, segs[-1]) for d in densities] + [(densities[-1], s) for s in segs[:-1]]
        if case.startswith("H_linear"):
            settings = [(d, 0) for d in densities]
        for d, n in settings:
            c = Coplanar2D(build(n), b=b, plane=plane, options=FEMOptions(mesh_density=d))
            t = vc.Timer()
            g = c.gsm(f, b1, b2)
            dt = t()
            key = f"{case}_f{f / 1e9:g}_d{d:g}_n{n}"
            arrays.update(vc.gsm_arrays(key, g))
            ndof = int(sum(x[1] for x in g.fem_info["ndof"]))
            runs.append(dict(key=key, f=f, density=d, n_seg=n, ndof=ndof, seconds=round(dt, 2)))
            print(json.dumps(runs[-1]), flush=True)
    return arrays, runs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="*", default=["H_linear_L100", "H_linear_L200", "E_rc50", "strangled"])
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    dens = (6, 10) if a.quick else (8, 12, 16)
    segs = (16, 32) if a.quick else (32, 64)
    for case in a.cases:
        arrays, runs = run(case, dens, segs, None)
        vc.save_npz(f"taper_fem_{case}.npz", __file__, arrays, case=case, runs=runs, densities=dens, n_seg=segs,
                    fem_order=3, method="wrmodes.fem2d.Coplanar2D on WP7b polygon outlines",
                    b_flat=B_FLAT, taper_length=L_TAPER, radius=R_ARC)
