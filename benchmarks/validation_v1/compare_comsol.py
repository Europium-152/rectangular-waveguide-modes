"""Compare wave-1 components with the COMSOL Gate A campaign (C1-C3), once Daniel's results are post-processed.

    python benchmarks/validation_v1/compare_comsol.py [--data benchmarks/comsol/data]

Inputs: benchmarks/comsol/data/WP7_C<k>.npz from benchmarks/comsol/wp7_postprocess.py. Tolerance
(CONVENTIONS.md section 8): 0.02 dB on the main transmission (TE10 -> TE10). Also reported, not pass/fail: the
complex difference of every TE10 column entry (phases and signs of CONVENTIONS.md at the reference planes), the
converted powers, and the consistency of the projected |S| with COMSOL's own TE10 port.

* C1 vs WP3: Line(Guide(WR-90), [Miter(90, plane='H')]) (plain miter, chamfer None).
* C2 vs WP1 + WP2: Line(Guide(WR-90, sigma=3.5e7), [SmoothBend(0.6, sech 60 deg, plane='H')]).
* C3 vs WP4: Step(22.86 mm, 10.16 mm, 22.86 mm, 1.6 mm).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ref_common  # noqa: E402
import ref_interfaces as I  # noqa: E402
import ref_line  # noqa: E402
from wrmodes.modes import Mode  # noqa: E402

DATA = os.path.abspath(os.path.join(HERE, "..", "comsol", "data"))
WR90 = ref_line.WR90
TE10 = Mode("TE", 1, 0)
TOL_DB = 0.02


def model_gsm(case, f):
    if case == "C1":
        Miter, Guide, Line = I.get("Miter"), I.get("Guide"), I.get("Line")
        return Line(Guide(*WR90), [Miter(90, chamfer=None, plane="H")]).gsm(f, cutoff_factor=1.0)
    if case == "C2":
        Guide, SmoothBend, Line = I.get("Guide"), I.get("SmoothBend"), I.get("Line")
        bend = SmoothBend(0.6, ref_line.sech_curvature(60, 0.6), plane="H", atol=1e-10)
        return Line(Guide(*WR90, sigma=3.5e7), [bend]).gsm(f, cutoff_factor=1.0)
    if case == "C3":
        Step = I.get("Step")
        return Step(WR90[0], WR90[1], WR90[0], 1.6e-3).gsm(f)
    raise KeyError(case)


def compare_case(case, path):
    d = ref_common.load_npz(path)
    rows = []
    for f in d["freqs"]:
        fk = f"{f / 1e9:g}"
        m1 = [Mode.parse(str(x)) for x in d[f"{fk}/modes1"]]
        m2 = [Mode.parse(str(x)) for x in d[f"{fk}/modes2"]]
        g = model_gsm(case, float(f))
        j1, j2 = g.basis1.index(TE10), g.basis2.index(TE10)
        r = {"f": float(f)}
        if f"{fk}/S21" in d:
            S21c, S11c = d[f"{fk}/S21"], d[f"{fk}/S11"]
            S21m = np.array([g.S21[g.basis2.index(m), j1] for m in m2])
            S11m = np.array([g.S11[g.basis1.index(m), j1] for m in m1])
            i2 = m2.index(TE10)
            r["dB_S21_TE10"] = float(20 * np.log10(abs(S21m[i2]) / abs(S21c[i2])))
            r["max_dS_col1"] = float(max(np.max(np.abs(S21m - S21c)), np.max(np.abs(S11m - S11c))))
            r["max_dP_col1"] = float(max(np.max(np.abs(np.abs(S21m) ** 2 - np.abs(S21c) ** 2)),
                                         np.max(np.abs(np.abs(S11m) ** 2 - np.abs(S11c) ** 2))))
            r["spurious"] = float(d[f"{fk}/spurious_incident_exc1"])
            if f"{fk}/comsol_S21" in d:
                r["projection_vs_comsol_port_dB"] = float(20 * np.log10(abs(S21c[i2]) / abs(d[f"{fk}/comsol_S21"])))
        if f"{fk}/S12" in d:
            S12c, S22c = d[f"{fk}/S12"], d[f"{fk}/S22"]
            S12m = np.array([g.S12[g.basis1.index(m), j2] for m in m1])
            S22m = np.array([g.S22[g.basis2.index(m), j2] for m in m2])
            r["max_dS_col2"] = float(max(np.max(np.abs(S12m - S12c)), np.max(np.abs(S22m - S22c))))
        r["ok"] = abs(r.get("dB_S21_TE10", 0.0)) <= TOL_DB
        rows.append(r)
        print(f"{case} {f / 1e9:6.2f} GHz: dB(TE10) {r.get('dB_S21_TE10', float('nan')):+.4f}  "
              f"max|dS| {r.get('max_dS_col1', float('nan')):.1e}  max|dP| {r.get('max_dP_col1', float('nan')):.1e}  "
              f"spurious {r.get('spurious', float('nan')):.1e}")
    return rows


def main(argv):
    data = DATA
    if "--data" in argv:
        data = argv[argv.index("--data") + 1]
    out = {}
    for case in ("C1", "C2", "C3"):
        path = os.path.join(data, f"WP7_{case}.npz")
        if not os.path.exists(path):
            print(f"{case}: no COMSOL data yet ({path})")
            out[case] = {"skipped": "no data"}
            continue
        try:
            out[case] = compare_case(case, path)
        except I.InterfaceMissing as exc:
            print(f"{case}: {exc}")
            out[case] = {"skipped": str(exc)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "comsol_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
