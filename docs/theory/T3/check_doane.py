"""T3 check 4: rate profiles, first-order conversion, Doane & Anderson (1989) eq. 18 and Fig. 9.

Fig. 9 of Doane & Anderson (Microwave J. 32(3), p. 159): mode conversion ("total spurious mode power") in 20-inch
90-degree twists of WR187 used as "tall guide" (1.872" x 0.872" ID, TE10 with a = 0.872", b = 1.872"), 15-18 GHz,
for constant, cosine and triangular rates of twist.

Computes, at 15, 16.5 and 18 GHz and on a frequency grid:
  * eq. 18, P_c = (8 c theta)^2 / (dbeta L)^4 with c = 8/pi^2 and the exact TE10-TE01 dbeta (and with Doane's
    far-from-cutoff dbeta);
  * the first-order integral |K+|^2 |F(dbeta)|^2 for TE10 -> TE01 and summed over all propagating modes;
  * the full coupled-mode solution (all propagating modes, forward and backward) for each profile:
    total spurious power = 1 - |S21(TE10, TE10)|^2 (GSM from the transfer matrix);
  * reciprocity and unitarity of those GSMs.
Fig. 9 values quoted below are eyeball readings of the scan (+-0.05 % absolute), for orientation only; the
digitized curves are L1's job.
"""
import os
import sys

import numpy as np
import scipy.constants as cst

sys.path.insert(0, os.path.dirname(__file__))
from twist_formulas import (coupling_C, doane_eq18, first_order_power, profile_transform, rate_profile,  # noqa
                            transfer_matrix)
from wrmodes.gsm import from_transfer_matrix  # noqa: E402
from wrmodes.modes import ModeBasis  # noqa: E402

INCH = 0.0254
A, B, L, THETA = 0.872 * INCH, 1.872 * INCH, 20 * INCH, np.pi / 2
FIG9 = {  # eyeball readings of Fig. 9 (percent), +-0.05 absolute
    "constant": {15.0: 1.35, 16.5: 0.03, 18.0: 1.7},
    "cosine": {15.0: 0.35, 16.5: 0.85, 18.0: 1.03},
    "triangular": {15.0: 0.05, 16.5: 0.08, 18.0: 0.08},
}


def dbeta_te10_te01(f):
    basis = ModeBasis(A, B, ["TE1,0", "TE0,1"])
    be = basis.beta(f).real
    return abs(be[basis.index("TE0,1")] - be[basis.index("TE1,0")])


def full_solution(f, kind, slices=600):
    basis = ModeBasis.propagating(f, A, B)
    Phi = transfer_matrix(basis, f, rate_profile(kind, THETA, L), L, slices=slices)
    g = from_transfer_matrix(Phi, basis, f)
    i = basis.index("TE1,0")
    spur = 1 - abs(g.S21[i, i]) ** 2
    s21 = np.abs(g.S21[:, i]) ** 2
    s11 = np.abs(g.S11[:, i]) ** 2
    return basis, spur, s21, s11, g


def main():
    c = coupling_C("TE0,1", "TE1,0", A, B)
    print(f"WR187 tall guide a = {A*1e3:.3f} mm, b = {B*1e3:.3f} mm, L = {L*1e3:.0f} mm, theta = 90 deg, c = {c:.6f}")
    for f in (15e9, 16.5e9, 18e9):
        print(f"  propagating modes at {f/1e9:g} GHz:", ", ".join(ModeBasis.propagating(f, A, B).names))
    # profile transforms: closed form vs numerical integration
    for kind in ("constant", "triangular", "cosine", "cos2"):
        z = np.linspace(0, L, 200001)
        tp = rate_profile(kind, THETA, L)(z)
        for db in (0.0, 7.0, 26.8, 61.8, np.pi / L, 2 * np.pi / L):
            num = abs(np.trapezoid(tp * np.exp(1j * db * z), z))
            cf = profile_transform(kind, THETA, L, db)
            assert abs(num - cf) < 1e-8 * THETA, (kind, db, num, cf)
    print("profile transforms: closed forms agree with numerical integration (1e-8)")

    print("\n f/GHz  dbeta*L  |eq.18 exact dbeta  eq.18 Doane dbeta | first order TE10->TE01 (%):"
          " const   tri    cos   cos2 | max_x of tri: (16 c theta)^2/(dbL)^4")
    for f in (15e9, 16.5e9, 18e9):
        db = dbeta_te10_te01(f)
        lam = cst.c / f
        dbL_doane = (L * lam / A ** 2) * abs(A ** 2 / B ** 2 - 1) * np.pi / 4
        basis = ModeBasis(A, B, ["TE1,0", "TE0,1"])
        fo = {k: 100 * first_order_power(basis, f, k, THETA, L, "TE1,0")[basis.index("TE0,1")]
              for k in ("constant", "triangular", "cosine", "cos2")}
        print(f" {f/1e9:5.1f}  {db*L:7.3f}  |   {100*doane_eq18(c, THETA, db*L):6.3f}           "
              f"{100*doane_eq18(c, THETA, dbL_doane):6.3f}       |   "
              f"{fo['constant']:6.4f} {fo['triangular']:6.4f} {fo['cosine']:6.4f} {fo['cos2']:6.4f} | "
              f"{100*(16*c*THETA)**2/(db*L)**4:6.3f}")

    print("\nFirst order summed over all propagating modes, and full coupled-mode solution (all propagating modes,"
          " forward + backward), total spurious power in %:")
    print(" f/GHz | profile     first-order-sum  full-solution  (Fig. 9 eyeball) | largest contributions (full)"
          " | recip.  unit.")
    for f in (15e9, 16.5e9, 18e9):
        for kind in ("constant", "cosine", "triangular", "cos2"):
            basis, spur, s21, s11, g = full_solution(f, kind)
            fo = 100 * np.sum(first_order_power(basis, f, kind, THETA, L, "TE1,0"))
            i = basis.index("TE1,0")
            contrib = sorted([(s21[j], basis[j].name) for j in range(len(basis)) if j != i], reverse=True)[:3]
            fig = FIG9.get(kind, {}).get(f / 1e9)
            figs = f"{fig:5.2f}" if fig is not None else "  -  "
            print(f" {f/1e9:5.1f} | {kind:10s}  {fo:10.4f}       {100*spur:10.4f}     ({figs})          | "
                  + ", ".join(f"{n} {100*p:.4f}" for p, n in contrib)
                  + f" | {g.reciprocity_defect():.0e} {g.unitarity_defect():.0e}")
            assert g.reciprocity_defect() < 1e-10 and g.unitarity_defect() < 1e-10
    # the acceptance test: triangular, first order (TE10-TE01 and all modes) below 0.1 % at 15-18 GHz
    fr = np.linspace(15e9, 18e9, 61)
    worst_fo, worst_full = 0.0, 0.0
    for f in fr:
        if any(abs(fc - f) < 0.02e9 for fc in ModeBasis.below(20e9, A, B).cutoff_frequencies()):
            continue
        basis = ModeBasis.propagating(f, A, B)
        worst_fo = max(worst_fo, np.sum(first_order_power(basis, f, "triangular", THETA, L, "TE1,0")))
    for f in np.linspace(15e9, 18e9, 13):
        worst_full = max(worst_full, full_solution(f, "triangular", slices=300)[1])
    print(f"\ntriangular twist, 15-18 GHz: max first-order (all modes) {100*worst_fo:.3f} %, "
          f"max full solution {100*worst_full:.3f} %  (Fig. 9: about 0.05-0.08 %)")
    assert worst_fo < 1e-3
    print("PASS")


if __name__ == "__main__":
    main()
