"""Gate A: compare wrmodes with COMSOL campaign 1 (benchmarks/comsol/CAMPAIGN_1.md).

    python benchmarks/comsol/campaign_1/compare_campaign_1.py [C1 C1b C2 C3] [--out results.md]

COMSOL exports (raw/) are global S-parameters of TE10 rectangular ports; the Java models are in java/.
Conventions: COMSOL and wrmodes both use exp(+j omega t), so complex values compare directly once the port
distances are de-embedded. The S21 phase is compared modulo 180 degrees (COMSOL picks the sign of each port mode).
C1/C1b need scikit-fem and gmsh (the 'fem' extra).
"""
import argparse
import pathlib
import re

import numpy as np

import wrmodes as wr
from wrmodes.components import ConstantCurvature, Guide, SechCurvature, SmoothBend, Straight
from wrmodes.modematch import Step
from wrmodes.modes import Mode, propagation_constant

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
FIG6 = HERE.parent  # the PEC Fig. 6 exports of the paper
W, H = 22.86e-3, 10.16e-3
TE10 = Mode("TE", 1, 0)
SIGMA_AL = 3.774e7  # COMSOL's library aluminium, used in the C2 models
LEG = 15e-3


def read(name):
    """{freq_GHz: {'S11': complex, 'S21': complex}} from a COMSOL global-value export (first data row)."""
    lines = (RAW / name).read_text().splitlines()
    hdr = next(l for l in lines if l.startswith("% X"))
    cols = re.findall(r"emw\.(S\d\d) \(1\) @ freq=([\d.]+)", hdr)
    row = next(l for l in lines if l.strip() and not l.startswith("%")).split()
    vals = [complex(s.replace("i", "j")) for s in row[len(row) - len(cols):]]
    out = {}
    for (q, f), v in zip(cols, vals):
        out.setdefault(float(f), {})[q] = v
    return dict(sorted(out.items()))


def read_fig6_db(name):
    lines = (FIG6 / name).read_text().splitlines()
    hdr = next(l for l in lines if l.startswith("% X"))
    f = [float(s.split("freq=")[1].split()[0]) for s in hdr.split("emw.S21dB")[1:]]
    v = next(l for l in lines if not l.startswith("%")).split()[3:]
    return dict(zip(f, map(float, v)))


def db(x):
    return 20 * np.log10(abs(x))


def te10(g):
    i1, i2 = g.basis1.index(TE10), g.basis2.index(TE10)
    return g.S11[i1, i1], g.S21[i2, i1]


def phase_mod180(z):
    return (np.degrees(np.angle(z)) + 90) % 180 - 90


def c1(out):
    from wrmodes.fem2d import Miter
    out.append("\n### C1 / C1b: plain (optical-mirror) H-plane miters, 2D\n")
    out.append("| case | f (GHz) | wrmodes \\|S21\\| (dB) | COMSOL | COMSOL finest (fine; 46–86 off-grid: fine + 15 cm PML; 26.3: ports TE10–TE40) | Δ vs finest (dB) | wrmodes \\|S11\\| | COMSOL \\|S11\\| | Δ∠S21 (°, mod 180) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    cases = [("90°", 90, 15e-3, "C1_miter90"), ("60°", 60, 28.1982e-3, "C1b_60miter"), ("120°", 120, 15e-3, "C1b_120miter")]
    for tag, ang, leg, fn in cases:
        c, cf = read(fn + ".txt"), read(fn + "_fine.txt")
        if (RAW / (fn + "_fine_long_pml.txt")).exists():  # fine mesh and 15 cm PMLs at the near-cutoff points
            cf.update(read(fn + "_fine_long_pml.txt"))
        if (RAW / (fn + "_more_ports.txt")).exists():  # ports for TE10-TE40 on each arm (plus the PMLs)
            # complete set of propagating modes only below the TE50 cutoff
            cf.update({f: v for f, v in read(fn + "_more_ports.txt").items() if f * 1e9 < 5 * 299792458 / (2 * W)})
        line = wr.Line(Guide(W, H), [Miter(ang, plane="H")])
        for f, v in c.items():
            s11, s21 = te10(line.gsm(f * 1e9))
            b = propagation_constant(TE10, W, H, f * 1e9)
            s11, s21 = s11 * np.exp(-2j * b * leg), s21 * np.exp(-2j * b * leg)
            ref = cf.get(f, v)
            fine = f"{db(cf[f]['S21']):.4f}" if f in cf else "–"
            out.append(f"| {tag} | {f:g} | {db(s21):.4f} | {db(v['S21']):.4f} | {fine} | {db(s21) - db(ref['S21']):+.4f} "
                       f"| {abs(s11):.4f} | {abs(ref['S11']):.4f} | {phase_mod180(s21 / ref['S21']):+.2f} |")
            print(out[-1], flush=True)


def c2(out):
    th, r0 = np.pi / 3, 0.055
    L_sech = 2 * th * r0 / np.arctan(np.tanh(2.0))  # the COMSOL model gives r0 and S = 4 and derives the length
    cases = [("sech", L_sech, SechCurvature(60, L_sech), "C2_sech_aluminum.txt", None,
              "15cm_60deg_hyperbolic_curve_side_a_short_e_field_horizontal"),
             ("constant", 0.15, ConstantCurvature.from_angle(60, 0.15), "C2_constant_aluminum_firstrow.txt",
              "C2_constant_PEC_tol1eminus4.txt", "15cm_60deg_constant_curve_side_a_short_e_field_horizontal"),
             ("constant, user mesh", 0.15, ConstantCurvature.from_angle(60, 0.15), "C2_constant_aluminum_user_mesh.txt",
              "C2_constant_PEC_tol1eminus4_user_mesh.txt", "15cm_60deg_constant_curve_side_a_short_e_field_horizontal")]
    out.append("\n### C2: 15 cm 60° E-plane bends, aluminium (σ = 3.774e7 S/m)\n")
    out.append("Ohmic part = |S21| aluminium − |S21| PEC (dB). PEC COMSOL: paper Fig. 6 export (sech, constant) and the "
               "same-model PEC run (constant, 18/40/60 GHz). 'constant' uses COMSOL's physics-controlled mesh, 'constant, user mesh' "
               "the paper's recipe (free tetrahedra, hmax = c/60 GHz/5 on all domains).\n")
    out.append("| bend | f | COMSOL Al | wrmodes Al | Δ | COMSOL PEC (paper) | COMSOL PEC (same model) | wrmodes PEC | ohmic COMSOL (same model) | ohmic COMSOL (vs paper) | ohmic wrmodes | Δ∠S21 (°) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, L, kap, fal, fpec, fig6 in cases:
        al, pec_paper = read(fal), read_fig6_db(fig6)
        pec_same = read(fpec) if fpec else {}
        lossy = wr.Line(Guide(W, H, sigma=SIGMA_AL), [SmoothBend(L, kap, plane="E")])
        ideal = wr.Line(Guide(W, H), [SmoothBend(L, kap, plane="E")])
        for f, v in al.items():
            _, s21 = te10(lossy.gsm(f * 1e9))
            _, s21p = te10(ideal.gsm(f * 1e9))
            cA, wA, wP, cPp = db(v["S21"]), db(s21), db(s21p), pec_paper.get(f, np.nan)
            cPs = db(pec_same[f]["S21"]) if f in pec_same else np.nan
            ohm_same = f"{cA - cPs:.4f}" if f in pec_same else "–"
            out.append(f"| {name} | {f:g} | {cA:.4f} | {wA:.4f} | {wA - cA:+.4f} | {cPp:.4f} | "
                       + (f"{cPs:.4f}" if f in pec_same else "–")
                       + f" | {wP:.4f} | {ohm_same} | {cA - cPp:.4f} | {wA - wP:.4f} | {phase_mod180(s21 / v['S21']):+.2f} |")
            print(out[-1], flush=True)


def c3(out):
    out.append("\n### C3: centred E-plane height steps WR-90 → h, full model, legs 15 mm\n")
    out.append("| case | f | wrmodes \\|S11\\| | COMSOL | COMSOL fine | Δ vs finest (dB) | wrmodes \\|S21\\| | COMSOL finest \\|S21\\| | Δ (dB) | COMSOL \\|S11\\|²+\\|S21\\|² | Δ∠S11 (°) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")

    def row(tag, f, s11, s21, c, cf):
        b = propagation_constant(TE10, W, H, f * 1e9)
        s11 = s11 * np.exp(-2j * b * LEG)
        ref = cf.get(f, c[f]) if cf else c[f]
        fine = f"{abs(cf[f]['S11']):.5f}" if cf and f in cf else "–"
        psum = abs(ref["S11"]) ** 2 + abs(ref["S21"]) ** 2
        out.append(f"| {tag} | {f:g} | {abs(s11):.5f} | {abs(c[f]['S11']):.5f} | {fine} | {db(s11) - db(ref['S11']):+.4f} "
                   f"| {abs(s21):.5f} | {abs(ref['S21']):.5f} | {db(s21) - db(ref['S21']):+.4f} | {psum:.5f} "
                   f"| {np.degrees(np.angle(s11 / ref['S11'])):+.2f} |")
        print(out[-1], flush=True)

    for h, tag in ((2.0e-3, "2_0"), (1.6e-3, "1_6")):
        c, cf = read(f"C3a_step_h{tag}.txt"), read(f"C3a_step_h{tag}_fine.txt")
        if tag == "2_0":
            cf = read("C3a_step_h2_0_fine_tol1eminus4.txt")
        st = Step(W, H, W, h, plane="H")
        for f in c:
            row(f"C3a h = {h * 1e3:.1f} mm", f, *te10(st.gsm(f * 1e9)), c, cf)
    c = read("C3b_strangled_100mm.txt")
    line = wr.Line(Guide(W, H), [Step(W, H, W, 1.6e-3), Straight(0.1), Step(W, 1.6e-3, W, H)])
    for f in c:
        row("C3b 1.6 mm × 100 mm", f, *te10(line.gsm(f * 1e9)), c, None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cases", nargs="*", default=["C1", "C2", "C3"])
    p.add_argument("--out", default=None, help="write the markdown tables here")
    a = p.parse_args()
    out = []
    for k in a.cases:
        {"C1": c1, "C1b": c1, "C2": c2, "C3": c3}[k](out)
    if a.out:
        pathlib.Path(a.out).write_text("\n".join(out) + "\n")
