"""Transmission loss of a plain 90-degree H-plane miter in WR-90, 18-90 GHz (wrmodes 1.0, WP3).

Usage:
    python examples/h_plane_miter.py [--step GHz] [--near-cutoffs] [--order 2|3] [--density N] [--processes P]
                                     [--out file.npz]

The miter couples TE10 only to the TE_m0 modes (family n = 0, LSM), so only that family is solved. The loss is
-20 log10 |S21| of TE10 -> TE10: power converted to the other TE_m0 modes plus reflected power (the walls are
perfect conductors). For comparison: Butterweck & de Ronde (Philips Tech. Rev. 29, 1968, p. 97) quote Bouwkamp's
result that the mode-conversion loss of the optical H-plane bend is below 0.4 dB above eight times the cut-off
frequency (52.5 GHz for WR-90).

Requires the optional extra: pip install "wrmodes[fem]".
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

W, H = 0.9 * 0.0254, 0.4 * 0.0254          # WR-90 (m)


def point(args):
    f, options = args
    from wrmodes.fem2d import Miter
    from wrmodes.modes import ModeBasis
    miter = Miter(90, guide=(W, H), **options)
    basis = ModeBasis.propagating(f, W, H, families=[0], frame="H")
    t0 = time.perf_counter()
    g = miter.gsm(f, basis, basis)
    dt = time.perf_counter() - t0
    return (f, g.loss_db("TE1,0"), float(np.abs(g.S11[0, 0]) ** 2), g.unitarity_defect(), g.reciprocity_defect(),
            float(np.max(np.abs(g.S11 - g.S22))), dt, sum(n for _, n in g.fem_info["ndof"]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fmin", type=float, default=18.0)
    p.add_argument("--fmax", type=float, default=90.0)
    p.add_argument("--step", type=float, default=2.0, help="frequency step (GHz)")
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--density", type=float, default=6.0, help="elements per transverse wavelength")
    p.add_argument("--port-distance", type=float, default=None, help="arm length (m); default a/4")
    p.add_argument("--near-cutoffs", action="store_true",
                   help="add points just below and above every TE_m0 cut-off in the range, where the loss peaks")
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--out", default=None, help="save the curve to this .npz file")
    a = p.parse_args()
    options = dict(order=a.order, mesh_density=a.density)
    if a.port_distance is not None:
        options["port_distance"] = a.port_distance
    freqs = np.arange(a.fmin, a.fmax + 1e-9, a.step) * 1e9
    fc = 299792458.0 / (2 * W)
    if a.near_cutoffs:
        extra = [m * fc * (1 + d) for m in range(1, 40) for d in (-0.01, -0.002, -1e-4, 1e-6, 1e-4, 5e-4, 0.002, 0.005, 0.01, 0.02, 0.04)]
        freqs = np.unique(np.concatenate([freqs, [f for f in extra if a.fmin * 1e9 <= f <= a.fmax * 1e9]]))
    jobs = [(f, options) for f in freqs]
    if a.processes > 1:
        with ProcessPoolExecutor(a.processes) as ex:
            rows = list(ex.map(point, jobs))
    else:
        rows = [point(j) for j in jobs]
    print(f"WR-90 plain 90-degree H-plane miter, options {options}")
    print(" f (GHz)   f/fc    loss (dB)   |S11|^2     unitarity  reciprocity  |S11-S22|   time (s)  ndof")
    for f, loss, r, u, rec, sym, dt, ndof in rows:
        print(f"{f / 1e9:8.3f} {f / fc:7.4f} {loss:10.5f} {r:11.3e} {u:11.1e} {rec:11.1e} {sym:11.1e} {dt:9.2f} {ndof:6d}")
    if a.out:
        import json, datetime, subprocess
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            commit = "unknown"
        import wrmodes
        meta = dict(wrmodes_version=wrmodes.__version__, git_commit=commit, created=datetime.date.today().isoformat(),
                    generator="examples/h_plane_miter.py", guide=[W, H], angle_deg=90, chamfer=None, **options)
        np.savez(a.out, rows=np.array(rows), meta=json.dumps(meta))


if __name__ == "__main__":
    main()
