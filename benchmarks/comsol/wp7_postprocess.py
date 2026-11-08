"""Convert the outputs of the WP7 COMSOL models (C1-C3) to npz files in the format of CONVENTIONS.md section 9.

    python benchmarks/comsol/wp7_postprocess.py <folder with WP7_C<k>_case.json, _fields.csv, _sparams.csv> [C1 C2 C3]

Writes benchmarks/comsol/data/WP7_C<k>.npz (or --out DIR).

Method: COMSOL's own port modes are not used for the GSM (their sign and phase conventions are not those of
CONVENTIONS.md and could not be checked without COMSOL). The Java models write E and H at Gauss-Legendre points of
one sampling plane per port; this script projects them on the normalized mode fields of wrmodes.modes
(CONVENTIONS.md sections 4-5):

    V_p = int E_t . e_p ds = (a+ + a-) sqrt(2/Y_p),     I_p = int H_t . (z_hat x e_p) ds = (a+ - a-) sqrt(2 Y_p),

in the local frame (u, v, z) of the port as written in the case file (z along the direction of propagation from
port 1 to port 2). The amplitudes are moved from the sampling plane (local z = offset) to the reference plane
(z = 0) with exp(-+ i beta offset) (the leads are straight and lossless where offset != 0). For excitation 1 (TE10
incident at port 1) the column of the GSM is S11[:, TE10] = a-(port 1) / a+_TE10(port 1) and
S21[:, TE10] = a+(port 2) / a+_TE10(port 1); for excitation 2 (TE10 at port 2) S12 and S22 likewise. The phases are
those of CONVENTIONS.md, at the reference planes of the component.

Diagnostics stored per frequency: the amplitudes of waves travelling towards the component other than the
excitation ("spurious incident", from imperfect absorption at the ports or PML; they should be < 1e-3 of the
incident amplitude), and COMSOL's own TE10 |S| for comparison with the projected |S| (sign-free check).
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "validation_v1"))

from wrmodes.modes import Mode, ModeBasis, transverse_field  # noqa: E402

TE10 = Mode("TE", 1, 0)


def read_case(folder, case):
    with open(os.path.join(folder, f"WP7_{case}_case.json")) as fh:
        return json.load(fh)


def read_fields(folder, case):
    """{(excitation, f): {plane: (xyz (N, 3), E (N, 3) complex, H (N, 3) complex)}}, points in file order."""
    out = {}
    with open(os.path.join(folder, f"WP7_{case}_fields.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    groups = {}
    for r in rows:
        key = (int(r["excitation"]), float(r["freq_Hz"]), r["plane"])
        groups.setdefault(key, []).append(r)
    for (exc, f, plane), rs in groups.items():
        rs.sort(key=lambda r: int(r["point"]))
        xyz = np.array([[float(r[c]) for c in ("x", "y", "z")] for r in rs])
        E = np.array([[float(r[f"E{c}_re"]) + 1j * float(r[f"E{c}_im"]) for c in "xyz"] for r in rs])
        H = np.array([[float(r[f"H{c}_re"]) + 1j * float(r[f"H{c}_im"]) for c in "xyz"] for r in rs])
        out.setdefault((exc, f), {})[plane] = (xyz, E, H)
    return out


def read_sparams(folder, case):
    path = os.path.join(folder, f"WP7_{case}_sparams.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d = {}
            for k in ("S11", "S21", "S12", "S22"):
                if r.get(f"{k}_re"):
                    d[k] = float(r[f"{k}_re"]) + 1j * float(r[f"{k}_im"])
            out[(int(r["excitation"]), float(r["freq_Hz"]))] = d
    return out


def plane_nodes(case, plane):
    nu = np.array(plane.get("nodes_u", case.get("nodes_u")))
    nv = np.array(plane.get("nodes_v", case.get("nodes_v")))
    xu, wu = np.polynomial.legendre.leggauss(len(nu))
    xv, wv = np.polynomial.legendre.leggauss(len(nv))
    if np.max(np.abs(xu - nu)) > 1e-12 or np.max(np.abs(xv - nv)) > 1e-12:
        raise ValueError("Gauss nodes of the case file differ from numpy's")
    return xu, wu, xv, wv


def project(plane, case, f, xyz, E, H):
    """Forward and backward amplitudes (at the sampling plane) of every propagating mode of the plane's guide."""
    a, b = plane["a"], plane["b"]
    o, u, v = (np.array(plane[k], float) for k in ("origin", "u", "v"))
    xu, wu, xv, wv = plane_nodes(case, plane)
    U, V = np.meshgrid(xu * a / 2, xv * b / 2, indexing="ij")
    W = np.outer(wu * a / 2, wv * b / 2).ravel()
    U, V = U.ravel(), V.ravel()
    # the points must be where the case file says
    expect = o[None, :] + U[:, None] * u[None, :] + V[:, None] * v[None, :]
    if np.max(np.abs(expect - xyz)) > 1e-9:
        raise ValueError(f"plane {plane['name']}: sampling points do not match the case geometry")
    Eu, Ev = E @ u, E @ v
    Hu, Hv = H @ u, H @ v
    basis = ModeBasis.propagating(f, a, b)
    Y = basis.admittance(f)
    ap, am = [], []
    for m, y in zip(basis.modes, Y):
        eu, ev = transverse_field(m, a, b, U, V)
        Vp = np.sum(W * (Eu * eu + Ev * ev))
        Ip = np.sum(W * (Hu * (-ev) + Hv * eu))              # H_t . (z_hat x e), z_hat x e = (-e_v, e_u)
        x, yy = Vp * np.sqrt(y / 2), Ip / np.sqrt(2 * y)
        ap.append(0.5 * (x + yy))
        am.append(0.5 * (x - yy))
    return basis, np.array(ap), np.array(am)


def to_reference(basis, f, ap, am, offset):
    """Move amplitudes from local z = offset to z = 0 (lossless straight lead)."""
    beta = basis.beta(f)
    return ap * np.exp(1j * beta * offset), am * np.exp(-1j * beta * offset)


def process(folder, case_id):
    case = read_case(folder, case_id)
    fields = read_fields(folder, case_id)
    spar = read_sparams(folder, case_id)
    planes = {p["name"]: p for p in case["planes"]}
    res = {}
    for (exc, f), pl in sorted(fields.items()):
        amp = {}
        for name, (xyz, E, H) in pl.items():
            p = planes[name]
            if p.get("loss"):
                raise NotImplementedError("lossy leads are not de-embedded (the models sample at offset 0 there)")
            basis, ap, am = project(p, case, f, xyz, E, H)
            ap, am = to_reference(basis, f, ap, am, p["offset"])
            amp[p["port"]] = (basis, ap, am)
        b1, ap1, am1 = amp[1]
        b2, ap2, am2 = amp[2]
        if exc == 1:
            inc = ap1[b1.index(TE10)]
            refl, trans = am1 / inc, ap2 / inc
            spur = np.concatenate([np.delete(ap1, b1.index(TE10)), am2]) / inc
            keys = ("S11", "S21")
        else:
            inc = am2[b2.index(TE10)]
            refl, trans = ap2 / inc, am1 / inc          # S22 column at port 2, S12 column at port 1
            spur = np.concatenate([np.delete(am2, b2.index(TE10)), ap1]) / inc
            keys = ("S22", "S12")
        fk = f"{f / 1e9:g}"
        r = res.setdefault(fk, {"f": f, "modes1": b1.names, "modes2": b2.names})
        r[keys[0]] = refl
        r[keys[1]] = trans
        r[f"spurious_incident_exc{exc}"] = float(np.max(np.abs(spur))) if spur.size else 0.0
        r[f"incident_amplitude_exc{exc}"] = complex(inc)
        cs = spar.get((exc, f), {})
        for k, val in cs.items():
            r[f"comsol_{k}"] = complex(val)
    return case, res


def save(case, res, path, folder):
    sys.path.insert(0, os.path.join(REPO, "benchmarks", "validation_v1"))
    import ref_common
    arrays = {}
    freqs = []
    for fk, r in res.items():
        freqs.append(r["f"])
        arrays[f"{fk}/modes1"] = np.array(r["modes1"])
        arrays[f"{fk}/modes2"] = np.array(r["modes2"])
        for k, v in r.items():
            if k.startswith("S") or k.startswith("comsol_") or k.startswith("spurious") or k.startswith("incident"):
                arrays[f"{fk}/{k}"] = np.array(v)
    arrays["freqs"] = np.array(sorted(freqs))
    ref_common.save_npz(path, __file__, arrays, case=case, source_folder=os.path.abspath(folder),
                        method="projection of COMSOL E, H on wrmodes.modes.transverse_field (CONVENTIONS.md 4-5)",
                        columns="S11/S21: TE10 incident at port 1 (column of the GSM); S12/S22: TE10 at port 2")
    return path


def summary(case, res):
    lines = []
    for fk, r in sorted(res.items(), key=lambda kv: kv[1]["f"]):
        i1, i2 = r["modes1"].index("TE1,0"), r["modes2"].index("TE1,0")
        s = f"{case['case']} {float(fk):6.2f} GHz"
        if "S21" in r:
            s += f"  TE10->TE10 |S21| {20 * np.log10(abs(r['S21'][i2])):8.4f} dB |S11| {20 * np.log10(abs(r['S11'][i1])):8.2f} dB"
            if "comsol_S21" in r:
                s += f"  (COMSOL ports: {20 * np.log10(abs(r['comsol_S21'])):8.4f} dB)"
            s += f"  spurious {r['spurious_incident_exc1']:.1e}"
        if "S12" in r:
            s += f"  |S12| {20 * np.log10(abs(r['S12'][i1])):8.4f} dB spurious(2) {r['spurious_incident_exc2']:.1e}"
        lines.append(s)
    return "\n".join(lines)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    folder = argv[0]
    out = os.path.join(HERE, "data")
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    cases = [c for c in argv[1:] if c in ("C1", "C2", "C3")] or ["C1", "C2", "C3"]
    for c in cases:
        if not os.path.exists(os.path.join(folder, f"WP7_{c}_case.json")):
            print(f"{c}: no WP7_{c}_case.json in {folder}, skipped")
            continue
        case, res = process(folder, c)
        path = save(case, res, os.path.join(out, f"WP7_{c}.npz"), folder)
        print(summary(case, res))
        print("->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
