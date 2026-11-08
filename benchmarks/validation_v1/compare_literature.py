"""Compare WP3's Miter and WP4's Step with the literature (L1 data D1, D3; Marcuvitz D6; Ma 1997 Table 1).

    python benchmarks/validation_v1/compare_literature.py [--literature-dir DIR] [--b-over-a R]

* D1/D3: every ``benchmarks/literature/alessandri1994_fig*.csv`` and ``ma1997_fig*.csv`` present is compared with
  WP3's miter, built as Line(Guide(a, b), [Miter(90, chamfer=d, plane)]) so that the input is the global TE10
  (tasks/WP3, tasks/WP1). Tolerance: the uncertainty column of each point (CONVENTIONS.md section 8). Alessandri
  (1994) does not state the guide; --b-over-a (narrow/broad) must be given for fig4/fig5 (default: skipped).
* Ma et al. (1997) Table 1: |S11| and |S21| at 16 GHz, c/a = 1, within the spread the authors report.
* D6: WP4's Step(a, b, a, b') against Marcuvitz sec. 5.26a, eq. (2a), for b/lambda_g < 1, tolerance from the
  stated 1 % on B (propagated to |S11|). Cases: WR-90 -> 22.86 x b' for b' = 1.6, 3, 5, 8 mm, 8-29 GHz.

Missing interfaces or data files are reported and skipped. Writes results/literature_comparison.json.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ref_interfaces as I  # noqa: E402
import ref_literature as L  # noqa: E402
from wrmodes.modes import Mode  # noqa: E402

LIT = os.path.abspath(os.path.join(HERE, "..", "literature"))
TE10 = Mode("TE", 1, 0)


def wp3_miter_s():
    """Adapter: (f, chamfer, plane, a, b) -> (S11, S21) of WP3's 90 deg miter for the global TE10."""
    Miter, Guide, Line = I.get("Miter"), I.get("Guide"), I.get("Line")
    cache = {}

    def miter_s(f, chamfer, plane, a, b, angle=90):
        key = (round(chamfer, 12), plane, a, b, angle)
        if key not in cache:
            cache[key] = Line(Guide(a, b), [Miter(angle, chamfer=chamfer, plane=plane)])
        g = cache[key].gsm(f)
        return g.S11[g.basis1.index(TE10), g.basis1.index(TE10)], g.S21[g.basis2.index(TE10), g.basis1.index(TE10)]
    return miter_s


def wp4_step_s11():
    Step = I.get("Step")

    def s11(f, a, b, b2):
        g = Step(a, b, a, b2).gsm(f)
        i = g.basis1.index(TE10)
        return g.S11[i, i]
    return s11


def marcuvitz_cases():
    a, b = 22.86e-3, 10.16e-3
    for b2 in (1.6e-3, 3e-3, 5e-3, 8e-3):
        for f in np.arange(8e9, 29.01e9, 3e9):
            yield a, b, b2, f


def compare_marcuvitz(step_s11):
    rows = []
    for a, b, b2, f in marcuvitz_cases():
        B, y0, S = L.marcuvitz_eplane_step(f, a, b, b2)
        lo, hi = sorted(abs((1 - (y0 + 1j * B * s)) / (1 + (y0 + 1j * B * s))) for s in (0.99, 1.01))
        got = abs(step_s11(f, a, b, b2))
        ok = lo - 1e-12 <= got <= hi + 1e-12
        rows.append(dict(a=a, b=b, b2=b2, f=f, marcuvitz=abs(S), band=(lo, hi), wp4=got, ok=bool(ok)))
        print(f"E-step {b * 1e3:.2f}->{b2 * 1e3:.2f} mm {f / 1e9:5.1f} GHz |S11| WP4 {got:.4f} Marcuvitz {abs(S):.4f} "
              f"[{lo:.4f}, {hi:.4f}] {'pass' if ok else 'FAIL'}")
    return rows


def compare_ma_table(miter_s):
    t = L.MA1997_TABLE1
    S11, S21 = miter_s(t["f"], L.ma_chamfer(t["c_over_a"]), "H", t["a"], t["b"])
    r11 = abs(abs(S11) / t["S11"] - 1)
    r21 = abs(abs(S21) / t["S21"] - 1)
    res = dict(S11=abs(S11), S21=abs(S21), rel_dev_S11=r11, rel_dev_S21=r21,
               ok=bool(r11 <= t["rel_unc_S11"] and r21 <= t["rel_unc_S21"]))
    print("Ma 1997 Table 1:", res)
    return res


def main(argv):
    lit_dir = LIT
    ratio = None
    if "--literature-dir" in argv:
        lit_dir = argv[argv.index("--literature-dir") + 1]
    if "--b-over-a" in argv:
        ratio = float(argv[argv.index("--b-over-a") + 1])
    out = {}
    try:
        out["marcuvitz_eplane_step"] = compare_marcuvitz(wp4_step_s11())
    except I.InterfaceMissing as exc:
        out["marcuvitz_eplane_step"] = {"skipped": str(exc)}
        print("WP4 skipped:", exc)
    try:
        miter_s = wp3_miter_s()
    except I.InterfaceMissing as exc:
        miter_s = None
        out["miter"] = {"skipped": str(exc)}
        print("WP3 skipped:", exc)
    if miter_s is not None:
        out["ma1997_table1"] = compare_ma_table(miter_s)
        files = sorted(glob.glob(os.path.join(lit_dir, "alessandri1994_fig*.csv")) +
                       glob.glob(os.path.join(lit_dir, "ma1997_fig*.csv")))
        if not files:
            print(f"no L1 miter files in {lit_dir} yet")
        out["miter_files"] = []
        for p in files:
            try:
                rows = L.compare_miter_file(p, miter_s, b_over_a=ratio)
            except (ValueError, KeyError) as exc:
                print(f"{os.path.basename(p)} skipped: {exc}")
                out["miter_files"].append({"file": os.path.basename(p), "skipped": str(exc)})
                continue
            for r in rows:
                r = {k: v for k, v in r.items() if k != "residuals"}
                out["miter_files"].append(r)
                print(f"{r['file']} {r['curve']}: {r['n_within']}/{r['n']} within uncertainty, worst "
                      f"{r['worst_ratio']:.2f} x unc at x = {r['worst_x']:g}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "literature_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
