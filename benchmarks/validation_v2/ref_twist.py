"""WP7b reference for WP6 (twists): helical-coordinate FEM eigenmodes, mode matching at the end planes, and a
cascade of constant-rate sections for variable rates (`helical_fem.py`; no T3/T3b coupled-mode matrix is used).

Cases (card WP7b item 2):

* `doane_<profile>`: Doane & Anderson's 20-inch 90 deg twists in WR187 used tall (a = 0.872 in along u, b = 1.872
  in, TE1,0 with E along the long side), constant, cosine (theta' = (pi Theta / 2L) sin(pi z / L)) and triangular
  (Doane eq. 17) rates, at 15, 16.5 and 18 GHz (15 and 18 GHz are COMSOL campaign 2 E4).
* `square_*`: a 30 mm square guide (T3b section 6 cases): 90 deg constant twists at 2a/lambda = 2.5 with
  L = 5 and 20 lambda, and a triangular twist at 2a/lambda = 4.5 with L = 20 lambda.
* `strong_*`: a strong constant-rate twist, tau a = 0.1 in WR-90 (a = 22.86 mm along u), 90 deg (L = 0.359 m), at 15
  and 30 GHz; and tau a = 0.5 (L = 71.8 mm) at 30 GHz.

Settings and convergence (reported per case): FEM order 3 on squares of side h,
straight basis of every mode with cutoff below cutoff_factor * f, and for
variable rates n = 8, 16, 32 sections with Richardson extrapolation in 1/n^2. The incident mode is TE1,0.

    PYTHONPATH=src:benchmarks/validation_v2 python benchmarks/validation_v2/ref_twist.py [--cases ...]
"""
import argparse
import json

import numpy as np

import helical_fem as hf
import vcommon as vc

INCH = 0.0254
A187, B187 = 0.872 * INCH, 1.872 * INCH
L_DOANE = 20 * INCH
THETA = np.pi / 2


def rate_fn(profile, theta, L):
    if profile == "constant":
        return lambda z: theta / L
    if profile == "cosine":
        return lambda z: np.pi * theta / (2 * L) * np.sin(np.pi * z / L)
    if profile == "triangular":
        return lambda z: (4 * theta / L ** 2) * (z if z < L / 2 else L - z)
    raise ValueError(profile)


def cases():
    c = {}
    for prof in ("constant", "cosine", "triangular"):
        c[f"doane_{prof}"] = dict(a=A187, b=B187, theta=THETA, L=L_DOANE, profile=prof, freqs=[15e9, 16.5e9, 18e9])
    c["doane_constant_3mm"] = dict(c["doane_constant"], settings=[("main", 3.0e-3, 1.6)])
    for prof in ("cosine", "triangular"):     # mode-count check of the variable rates (many matching planes)
        c[f"doane_{prof}_cf2"] = dict(c[f"doane_{prof}"], freqs=[18e9], settings=[("main", 3.0e-3, 2.0)],
                                      sections=(64, 128))
        c[f"doane_{prof}_cf2p5"] = dict(c[f"doane_{prof}"], freqs=[18e9], settings=[("main", 3.0e-3, 2.5)],
                                        sections=(64, 128))
    a = 30e-3
    f25, f45 = 2.5 * hf.C0 / (2 * a), 4.5 * hf.C0 / (2 * a)
    c["square_const_5lam"] = dict(a=a, b=a, theta=THETA, L=5 * hf.C0 / f25, profile="constant", freqs=[f25])
    c["square_const_20lam"] = dict(a=a, b=a, theta=THETA, L=20 * hf.C0 / f25, profile="constant", freqs=[f25])
    c["square_tri_20lam_4p5"] = dict(a=a, b=a, theta=THETA, L=20 * hf.C0 / f45, profile="triangular", freqs=[f45])
    w, h = vc.W90, vc.H90
    c["strong_tau0p1"] = dict(a=w, b=h, theta=THETA, L=THETA * w / 0.1, profile="constant", freqs=[15e9, 30e9])
    c["strong_tau0p5"] = dict(a=w, b=h, theta=THETA, L=THETA * w / 0.5, profile="constant", freqs=[30e9])
    return c


def summary(g, basis, pbasis, inc="TE1,0"):
    """Powers (fractions of the input) per propagating output mode for the incident mode, transmitted and
    reflected, and the defects of the propagating block."""
    prop = [basis.names.index(n) for n in pbasis.names]
    i0 = basis.names.index(inc)
    S = np.block([[g.S11, g.S12], [g.S21, g.S22]])
    ip = prop + [len(basis.modes) + i for i in prop]
    Sp = S[np.ix_(ip, ip)]
    U = Sp.conj().T @ Sp - np.eye(len(ip))
    col = prop.index(i0)
    return dict(
        transmitted={pbasis.names[j]: float(abs(g.S21[prop[j], i0]) ** 2) for j in range(len(prop))},
        reflected={pbasis.names[j]: float(abs(g.S11[prop[j], i0]) ** 2) for j in range(len(prop))},
        main_db=float(vc.db(g.S21[i0, i0])),
        unitarity=float(np.max(abs(U))), unitarity_incident_column=float(np.max(abs(U[:, col]))),
        reciprocity=float(np.max(abs(Sp - Sp.T))))


def propagating_block(g, basis, pbasis):
    prop = [basis.names.index(n) for n in pbasis.names]
    return [np.asarray(B)[np.ix_(prop, prop)] for B in (g.S11, g.S21, g.S12, g.S22)]


def run_case(name, c, sections=(32, 64, 128)):
    """Constant rate: main h = 2.5 mm, cutoff_factor 2.0; checks h = 2 mm and cutoff_factor 2.6.
    Variable rate: h = 3 mm, cutoff_factor 1.6, n = 32, 64, 128 sections (Richardson from the last two; the
    sections must be short enough that their 'grating' wavenumber 2 pi n / L exceeds every beat Delta beta, else
    the piecewise rate converts resonantly: n >= 32 for Doane's twists). The mesh and mode-count sensitivity of the
    same ingredients is measured on the constant-rate twists (h 2.5 -> 2 mm, cutoff_factor 2 -> 2.6), where it is
    at the 1e-6 level in power; h = 3 mm and cutoff_factor 1.6 were checked against h = 2.5 mm and cutoff_factor 2
    on Doane's constant twist (`doane_constant_3mm` rows)."""
    arrays, rows = {}, []
    for f in c["freqs"]:
        pbasis = hf.straight_basis(c["a"], c["b"], f, 1.0)
        if "settings" in c:
            settings = c["settings"]
        elif c["profile"] == "constant":
            settings = [("main", 2.5e-3, 2.0), ("h_check", 2.0e-3, 2.0), ("cf_check", 2.5e-3, 2.6)]
        else:
            settings = [("main", 3.0e-3, 1.6)]
        guides = {}
        out = {}
        for tag, h, cf in settings:
            basis = hf.straight_basis(c["a"], c["b"], f, cf)
            if h not in guides:
                guides[h] = hf.HelicalGuide(c["a"], c["b"], h, 3)
            G = guides[h]
            t = vc.Timer()
            if c["profile"] == "constant":
                g, sec = hf.constant_twist(G, c["theta"], c["L"], f, basis)
                runs = {"exact": g}
            else:
                rate = rate_fn(c["profile"], c["theta"], c["L"])
                cache = {}
                ns = sections if tag == "main" else sections[-2:]
                runs = {n: hf.piecewise_twist(G, rate, c["L"], f, basis, n, cache) for n in ns}
            dt = t()
            blocks = {k: propagating_block(g, basis, pbasis) for k, g in runs.items()}
            if c["profile"] != "constant":
                n1, n2 = sorted(blocks)[-2:]
                ext = [(4 * B2 - B1) / 3 for B1, B2 in zip(blocks[n1], blocks[n2])]
                blocks["extrapolated"] = ext
            final = blocks["exact" if c["profile"] == "constant" else "extrapolated"]
            key = f"{name}_f{f / 1e9:g}_{tag}"
            for lab, B in zip(("S11", "S21", "S12", "S22"), final):
                arrays[f"{key}_{lab}"] = B
            arrays[f"{key}_names"] = np.array(pbasis.names)
            for k, B in blocks.items():
                if k not in ("exact", "extrapolated"):
                    arrays[f"{key}_n{k}_S21"] = B[1]
                    arrays[f"{key}_n{k}_S11"] = B[0]
            gfin = runs[max(runs)] if c["profile"] != "constant" else runs["exact"]
            info = summary(gfin, basis, pbasis)
            i0 = pbasis.names.index("TE1,0")
            info.update(tag=tag, f=f, h=h, cutoff_factor=cf, n_basis=len(basis.modes), seconds=round(dt, 1),
                        transmitted_final={n: float(abs(final[1][j, i0]) ** 2) for j, n in enumerate(pbasis.names)},
                        reflected_final={n: float(abs(final[0][j, i0]) ** 2) for j, n in enumerate(pbasis.names)})
            if c["profile"] != "constant":
                n1, n2 = sorted(k for k in blocks if k != "extrapolated")[-2:]
                info["section_change"] = float(np.max(abs(blocks[n2][1][:, i0] - blocks[n1][1][:, i0])))
                info["extrapolation_change"] = float(np.max(abs(blocks["extrapolated"][1][:, i0] - blocks[n2][1][:, i0])))
            out[tag] = final
            rows.append(info)
            spur = sum(v for k, v in info["transmitted_final"].items() if k != "TE1,0")
            print(json.dumps(dict(case=name, f=f / 1e9, tag=tag, n_basis=len(basis.modes), seconds=info["seconds"],
                                  spurious_pct=round(100 * spur, 5), main_db=round(float(vc.db(final[1][i0, i0])), 6),
                                  unitarity=info["unitarity"], col=info["unitarity_incident_column"],
                                  sec=info.get("section_change"), ext=info.get("extrapolation_change"))), flush=True)
    return arrays, rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="*", default=list(cases()))
    a = p.parse_args()
    cs = cases()
    for name in a.cases:
        arrays, rows = run_case(name, cs[name], **({"sections": cs[name]["sections"]} if "sections" in cs[name] else {}))
        c = {k: v for k, v in cs[name].items() if k not in ("settings", "sections")}
        vc.save_npz(f"twist_{name}.npz", __file__, arrays, case=name, geometry=c, rows=rows,
                    method="helical-coordinate FEM eigenmodes (Nedelec/Lagrange order 3) + mode matching; "
                           "variable rate: cascade of constant-rate sections, Richardson in 1/n^2",
                    incident="TE1,0")
