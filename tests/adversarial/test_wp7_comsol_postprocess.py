"""The COMSOL post-processing (benchmarks/comsol/wp7_postprocess.py) recovers a planted GSM column.

Synthetic E and H are built from known modal amplitudes (CONVENTIONS.md section 5) on the sampling planes of the C1
and C3 geometries, with evanescent modes and longitudinal components added (which the projection must ignore),
written in the CSV format of the Java models, and processed. This checks the frames, the projection, the
de-embedding and the file format without COMSOL.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "benchmarks", "comsol")))

import wp7_postprocess as PP  # noqa: E402
from wrmodes.modes import Mode, ModeBasis, transverse_field  # noqa: E402

A, B = 0.9 * 0.0254, 0.4 * 0.0254


def _fields(plane, f, ap, am, basis, rng):
    a, b = plane["a"], plane["b"]
    o, u, v, z = (np.array(plane[k], float) for k in ("origin", "u", "v", "z"))
    nu, nv = np.array(plane["nodes_u"]), np.array(plane["nodes_v"])
    U, V = np.meshgrid(nu * a / 2, nv * b / 2, indexing="ij")
    U, V = U.ravel(), V.ravel()
    Y = basis.admittance(f)
    Eu = np.zeros(U.shape, complex); Ev = np.zeros_like(Eu); Hu = np.zeros_like(Eu); Hv = np.zeros_like(Eu)
    for m, y, p, q in zip(basis.modes, Y, ap, am):
        eu, ev = transverse_field(m, a, b, U, V)
        Eu += (p + q) * np.sqrt(2 / y) * eu
        Ev += (p + q) * np.sqrt(2 / y) * ev
        Hu += (p - q) * np.sqrt(2 * y) * (-ev)
        Hv += (p - q) * np.sqrt(2 * y) * eu
    Ez = rng.normal(size=U.shape) + 1j * rng.normal(size=U.shape)      # longitudinal parts: ignored
    Hz = rng.normal(size=U.shape) + 1j * rng.normal(size=U.shape)
    xyz = o[None] + U[:, None] * u[None] + V[:, None] * v[None]
    E = Eu[:, None] * u[None] + Ev[:, None] * v[None] + Ez[:, None] * z[None]
    H = Hu[:, None] * u[None] + Hv[:, None] * v[None] + Hz[:, None] * z[None]
    return xyz, E, H


def _write(folder, cid, case, rows):
    with open(os.path.join(folder, f"WP7_{cid}_case.json"), "w") as fh:
        json.dump(case, fh)
    with open(os.path.join(folder, f"WP7_{cid}_fields.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["excitation", "freq_Hz", "plane", "point", "x", "y", "z", "Ex_re", "Ex_im", "Ey_re", "Ey_im",
                    "Ez_re", "Ez_im", "Hx_re", "Hx_im", "Hy_re", "Hy_im", "Hz_re", "Hz_im"])
        for exc, f, name, xyz, E, H in rows:
            for k in range(len(xyz)):
                vals = []
                for c in range(3):
                    vals += [E[k, c].real, E[k, c].imag]
                for c in range(3):
                    vals += [H[k, c].real, H[k, c].imag]
                w.writerow([exc, repr(float(f)), name, k, *[repr(float(x)) for x in xyz[k]], *[repr(float(x)) for x in vals]])


def _plane(name, port, origin, u, v, z, a, b, offset, nu, nv):
    return {"name": name, "port": port, "origin": origin, "u": u, "v": v, "z": z, "a": a, "b": b,
            "offset": offset, "loss": None, "nodes_u": list(np.polynomial.legendre.leggauss(nu)[0]),
            "nodes_v": list(np.polynomial.legendre.leggauss(nv)[0])}


def _run_case(tmp_path, cid, planes, excitations, f, rng):
    case = {"case": cid, "planes": planes}
    bases = {p["port"]: ModeBasis.propagating(f, p["a"], p["b"]) for p in planes}
    full = {p["port"]: ModeBasis.propagating(f, p["a"], p["b"], cutoff_factor=1.6, evanescent_families="all")
            for p in planes}
    planted, rows = {}, []
    for exc in excitations:
        amp = {}
        for p in planes:
            fb = full[p["port"]]
            ap = np.zeros(len(fb), complex)
            am = np.zeros(len(fb), complex)
            prop = fb.propagating_mask(f)
            # outgoing waves: random; incident: only TE10 at the excited port
            out = rng.normal(size=len(fb)) + 1j * rng.normal(size=len(fb))
            if (p["port"] == 1) == (exc == 1):
                inc = 0.7 - 0.3j
                if p["port"] == 1:
                    am[prop] = out[prop] * 0.3
                    ap[fb.index(Mode("TE", 1, 0))] = inc
                else:
                    ap[prop] = out[prop] * 0.3
                    am[fb.index(Mode("TE", 1, 0))] = inc
            else:
                (ap if p["port"] == 2 else am)[prop] = out[prop] * 0.3
            # evanescent content at the reference plane, decaying away from the component
            ev = ~prop
            (am if p["port"] == 1 else ap)[ev] = 0.5 * out[ev]
            amp[p["port"]] = (ap.copy(), am.copy())
            beta = fb.beta(f)
            zs = p["offset"]
            rows.append((exc, f, p["name"], *_fields(p, f, ap * np.exp(-1j * beta * zs), am * np.exp(1j * beta * zs),
                                                     fb, rng)))
        planted[exc] = amp
    _write(str(tmp_path), cid, case, rows)
    _, res = PP.process(str(tmp_path), cid)
    r = res[f"{f / 1e9:g}"]
    for exc in excitations:
        ap1, am1 = planted[exc][1]
        ap2, am2 = planted[exc][2]
        f1, f2 = full[1], full[2]
        i1 = [f1.index(m) for m in bases[1].modes]
        i2 = [f2.index(m) for m in bases[2].modes]
        if exc == 1:
            inc = ap1[f1.index(Mode("TE", 1, 0))]
            assert np.max(np.abs(r["S11"] - am1[i1] / inc)) < 1e-9      # quadrature aliasing of the planted evanescent modes
            assert np.max(np.abs(r["S21"] - ap2[i2] / inc)) < 1e-9      # quadrature aliasing of the planted evanescent modes
        else:
            inc = am2[f2.index(Mode("TE", 1, 0))]
            assert np.max(np.abs(r["S22"] - ap2[i2] / inc)) < 1e-9      # quadrature aliasing of the planted evanescent modes
            assert np.max(np.abs(r["S12"] - am1[i1] / inc)) < 1e-9      # quadrature aliasing of the planted evanescent modes
        assert r[f"spurious_incident_exc{exc}"] < 1e-9      # quadrature aliasing of the planted evanescent modes
    return res


def test_c1_geometry_projection(tmp_path):
    """C1 frames: port 1 u = +x, v = -z, z = +y; port 2 u = +y, v = -z, z = -x; offsets -+ 15 mm."""
    rng = np.random.default_rng(3)
    planes = [_plane("port1", 1, [0, -A / 2 - 0.015, B / 2], [1, 0, 0], [0, 0, -1], [0, 1, 0], A, B, -0.015, 48, 16),
              _plane("port2", 2, [-A / 2 - 0.015, 0, B / 2], [0, 1, 0], [0, 0, -1], [-1, 0, 0], A, B, 0.015, 48, 16)]
    _run_case(tmp_path, "C1", planes, [1], 40e9, rng)


def test_c3_geometry_projection_both_excitations(tmp_path):
    rng = np.random.default_rng(4)
    b2 = 1.6e-3
    planes = [_plane("port1", 1, [-0.015, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], A, B, -0.015, 24, 32),
              _plane("port2", 2, [0.010, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], A, b2, 0.010, 24, 12)]
    _run_case(tmp_path, "C3", planes, [1, 2], 55e9, rng)


def test_misplaced_points_are_detected(tmp_path):
    rng = np.random.default_rng(5)
    planes = [_plane("port1", 1, [0, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], A, B, 0.0, 16, 4),
              _plane("port2", 2, [0.5, 0.3, 0], [0.866, -0.5, 0], [0, 0, -1], [0.5, 0.866, 0], A, B, 0.0, 16, 4)]
    f = 20e9
    case = {"case": "C2", "planes": planes}
    basis = ModeBasis.propagating(f, A, B)
    rows = []
    for p in planes:
        xyz, E, H = _fields(p, f, np.ones(len(basis)), np.zeros(len(basis)), basis, rng)
        rows.append((1, f, p["name"], xyz + (1e-6 if p["port"] == 2 else 0), E, H))
    _write(str(tmp_path), "C2", case, rows)
    try:
        PP.process(str(tmp_path), "C2")
    except ValueError as exc:
        assert "do not match" in str(exc)
    else:
        raise AssertionError("misplaced sampling points not detected")
