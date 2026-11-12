"""Shared helpers of the WP3b literature benchmarks (benchmarks/fem2d): CSV reading, z statistics, the `meta` entry
of CONVENTIONS.md section 9, parallel evaluation in separate processes (gmsh keeps global state), and the outlines
that the literature uses beyond `Miter` (Ma et al.'s staircase miters)."""
import datetime
import json
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
LIT = os.path.join(REPO, "benchmarks", "literature")
sys.path.insert(0, os.path.join(REPO, "src"))
SQ2 = np.sqrt(2.0)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:          # pragma: no cover
        return "unknown"


def meta(generator, description, **parameters):
    import wrmodes
    return {"wrmodes_version": wrmodes.__version__, "git_commit": git_commit(),
            "created": datetime.date.today().isoformat(), "generator": generator, "description": description,
            "parameters": parameters}


def save(name, meta_dict, **arrays):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    np.savez_compressed(path, meta=np.array(json.dumps(meta_dict)), **arrays)
    return path


def load(name):
    d = np.load(os.path.join(DATA, name), allow_pickle=False)
    return {k: d[k] for k in d.files}, json.loads(str(d["meta"]))


def load_csv(name):
    """{curve: array (rows: x, y, uncertainty, x_uncertainty)} of a merged literature file."""
    rows = [r for r in open(os.path.join(LIT, name)) if not r.startswith("#")][1:]
    out = {}
    for r in rows:
        v = r.strip().split(",")
        out.setdefault(v[0], []).append([float(v[1]), float(v[2]), float(v[3]), float(v[4])])
    return {k: np.array(v) for k, v in out.items()}


def pmap(fn, items, processes=None):
    """map in worker processes (one gmsh per process); order preserved."""
    items = list(items)
    if not items:
        return []
    processes = processes or min(os.cpu_count() or 1, 4)
    if processes == 1:
        return [fn(x) for x in items]
    sys.stdout.flush()                     # forked workers must not inherit (and lose) unflushed output
    ctx = mp.get_context("fork")
    with ctx.Pool(processes, maxtasksperchild=50) as pool:
        return pool.map(fn, items, chunksize=max(1, len(items) // (8 * processes)))


def zstats(x, y, u, ux, ours, ours_minus=None, ours_plus=None):
    """z = (ours - y) / u_eff, u_eff including the x reading error of markers through the slope of our curve
    (ours_minus/ours_plus: our values at x -+ ux). Returns (z, u_eff)."""
    u = np.array(u, float)
    if ours_minus is not None:
        h = np.where(ux > 0, ux, 1.0)
        slope = (np.asarray(ours_plus) - np.asarray(ours_minus)) / (2 * h)
        u = np.where(ux > 0, np.sqrt(u ** 2 + (slope * ux) ** 2), u)
    return (np.asarray(ours) - y) / u, u


def summary(z, x, ours, y):
    i = int(np.argmax(np.abs(z)))
    return dict(n=len(z), rms_z=float(np.sqrt(np.mean(z ** 2))), within1=float(np.mean(np.abs(z) <= 1)),
                within2=float(np.mean(np.abs(z) <= 2)), max_z=float(np.abs(z[i])), x_max=float(x[i]),
                mean_diff=float(np.mean(np.asarray(ours) - y)))


def row_md(fig, curve, s, status=""):
    return ("| %s | %s | %d | %.2f | %.0f %% | %.0f %% | %.1f (%.4g) | %+.3f | %s |" % (
        fig, curve, s["n"], s["rms_z"], 100 * s["within1"], 100 * s["within2"], s["max_z"], s["x_max"],
        s["mean_diff"], status))


HEADER = ("| Figure | curve | n | rms z | share \\|z\\| <= 1 | share \\|z\\| <= 2 | max \\|z\\| (at x) | mean diff | "
          "status |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def staircase_miter_outline(a, n_steps, method="a", lead=None):
    """Ma et al. (1997, Figs. 3, 7) staircase model of the plain 90 degree H-plane miter (c/a = 1).

    The 45 degree mirror from the outer wall of arm 1 (at the reference plane through the inner corner) to the
    outer wall of arm 2 is replaced by n_steps equal steps of a/n_steps: method 'a' (steps through the slanted
    plane: tread and riser midpoints on it), 'b' (steps below it, on the guide side), 'c' (above it). Coordinates
    of `wrmodes.fem2d.miter_outline` (arm 1 along +y, u = +x, turning towards -x). Straight lead-ins of length
    `lead` (default a/4) are added before both reference planes (needed for method 'b', whose first riser lies
    on the reference plane); the port planes are de-embedded by `lead`.
    """
    from wrmodes.fem2d import Outline, PortPlane
    lead = a / 4 if lead is None else lead
    w = a / n_steps
    O = np.array([-a / 2, -a / 2])
    P1 = np.array([a / 2, -a / 2])
    pts = [P1]
    x, y = P1
    if method == "a":
        y += w / 2
        pts.append((x, y))
        for i in range(n_steps):
            x -= w
            pts.append((x, y))
            y += w if i < n_steps - 1 else w / 2
            pts.append((x, y))
    elif method == "b":
        for i in range(n_steps):
            x -= w
            pts.append((x, y))
            y += w
            pts.append((x, y))
    elif method == "c":
        for i in range(n_steps):
            y += w
            pts.append((x, y))
            x -= w
            pts.append((x, y))
    else:
        raise ValueError("method must be 'a', 'b' or 'c'")
    pts = [np.array(p, float) for p in pts]
    assert np.allclose(pts[-1], (-a / 2, a / 2))
    E1i, E1o = O - (0, lead), P1 - (0, lead)                 # port 1, a lead below the reference plane
    E2o, E2i = np.array([-a / 2 - lead, a / 2]), np.array([-a / 2 - lead, -a / 2])
    verts = [E1i, E1o] + pts + [E2o, E2i, O]
    # remove collinear interior points (P1 for methods a and c, the end point for b and c)
    clean = []
    n = len(verts)
    for i, v in enumerate(verts):
        p0, p2 = verts[i - 1], verts[(i + 1) % n]
        cr = (v[0] - p0[0]) * (p2[1] - p0[1]) - (v[1] - p0[1]) * (p2[0] - p0[0])
        if abs(cr) < 1e-14 * a * a and i not in (0, 1, len(pts) + 2, len(pts) + 3):
            continue
        clean.append(tuple(v))
    ports = (PortPlane(tuple(E1i), tuple(E1o), deembed=lead), PortPlane(tuple(E2i), tuple(E2o), deembed=lead))
    return Outline(vertices=clean, ports=ports)


# ------------------------------------------------------------------------------------------- evaluation
def te10(spec):
    """Complex (S11, S21) of the global TE10 mode for a geometry spec (dict), with the FEM (wrmodes.fem2d).

    spec keys: 'kind' ('miter' or 'stair'), 'f' (Hz), 'w', 'h' (global guide, m), 'plane' ('H' or 'E'),
    'angle' (deg) and 'chamfer' (d/a or None) for a miter, 'n_steps' and 'method' for a staircase, and
    'options' (FEMOptions keywords). Only the family of TE10 is solved (H plane: n = 0; E plane: local n = 1),
    with its propagating modes as port bases: a miter couples nothing else.
    """
    from wrmodes import Guide
    from wrmodes.fem2d import Coplanar2D, Miter
    g = Guide(spec["w"], spec["h"])
    plane = spec.get("plane", "H")
    f = spec["f"]
    opts = spec.get("options", {})
    basis = g.basis(f, 1.0, plane=plane)
    fam = 0 if plane == "H" else 1
    sub = basis.subset(basis.family(fam))
    if spec["kind"] == "miter":
        c = Miter(spec["angle"], chamfer=spec.get("chamfer"), plane=plane, guide=g, **opts)
    elif spec["kind"] == "stair":
        a, b = g.dims(plane)
        c = Coplanar2D(staircase_miter_outline(a, spec["n_steps"], spec["method"]), b, plane=plane, guide=g,
                       **opts)
    else:
        raise ValueError(spec["kind"])
    G = c.gsm(f, sub, sub)
    i = sub.index(("TE", 1, 0) if plane == "H" else ("TE", 0, 1))
    return complex(G.S11[i, i]), complex(G.S21[i, i])


def quantity(q, s11, s21):
    if q == "s11_db":
        return 20 * np.log10(abs(s11))
    if q == "s21_db":
        return 20 * np.log10(abs(s21))
    if q == "s21_pow":
        return abs(s21) ** 2
    raise ValueError(q)


def compare_curves(cases, processes=None, log=print):
    """Evaluate and compare a list of cases.

    Each case: dict(fig, file, curve, geom (spec without 'f'), x_to_f (callable), quantity, status). Markers
    (x_uncertainty > 0) are also evaluated at x -+ x_uncertainty. Returns (rows, arrays).
    """
    jobs, where = [], []
    loaded = []
    for k, c in enumerate(cases):
        arr = load_csv(c["file"])[c["curve"]]
        x, y, u, ux = arr.T
        loaded.append((x, y, u, ux))
        for j, xi in enumerate(x):
            for side, dx in ((0, 0.0), (-1, -ux[j]), (1, ux[j])):
                if side and ux[j] == 0:
                    continue
                spec = dict(c["geom"], f=c["x_to_f"](xi + dx))
                jobs.append(spec)
                where.append((k, j, side))
    res = pmap(te10, jobs, processes)
    rows, arrays = [], {}
    for k, c in enumerate(cases):
        x, y, u, ux = loaded[k]
        vals = {side: np.full(len(x), np.nan) for side in (-1, 0, 1)}
        s11 = np.zeros(len(x), complex)
        s21 = np.zeros(len(x), complex)
        for (kk, j, side), r in zip(where, res):
            if kk != k:
                continue
            vals[side][j] = quantity(c["quantity"], *r)
            if side == 0:
                s11[j], s21[j] = r
        markers = np.any(ux > 0)
        z, ueff = zstats(x, y, u, ux, vals[0], vals[-1] if markers else None, vals[1] if markers else None)
        s = summary(z, x, vals[0], y)
        rows.append((c, s))
        log(row_md(c["fig"], c["curve"], s, c.get("status", "")))
        key = f"c{k:02d}"
        arrays.update({key + "_x": x, key + "_lit": y, key + "_u": ueff, key + "_ours": vals[0],
                       key + "_S11": s11, key + "_S21": s21, key + "_z": z})
    return rows, arrays


def outliers(rows, arrays, zmax=2.0):
    out = []
    for k, (c, s) in enumerate(rows):
        key = f"c{k:02d}"
        for x, y, o, z in zip(arrays[key + "_x"], arrays[key + "_lit"], arrays[key + "_ours"], arrays[key + "_z"]):
            if abs(z) > zmax:
                out.append((c["fig"], c["curve"], x, y, o, z))
    return out


def case_meta(cases):
    return [{"key": f"c{k:02d}", "figure": c["fig"], "file": c["file"], "curve": c["curve"],
             "geometry": {kk: v for kk, v in c["geom"].items() if kk != "options"}, "quantity": c["quantity"],
             "status": c.get("status", "")} for k, c in enumerate(cases)]
