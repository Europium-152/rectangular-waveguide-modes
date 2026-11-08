"""Extra full coupled-mode runs for the 20-inch 90-degree WR187 triangular twist (section 8 of the note).

Run:  cd <worktree> && PYTHONPATH=$PWD/src:$PWD/docs/theory/T3b python3 docs/theory/T3b/doane_fig9_extras.py

 1. Evanescent modes (and slice convergence): GSM cascade of short slices (each slice through its transfer matrix, well conditioned for
    one slice), cutoff_factor 1, 1.5, 2, 3 (all families), tall WR187 at 16.5 GHz.
 2. Ordinary orientation (a = 1.872 in along u, incident TE1,0 with E along the short side).
 3. The coupling variants of check_helical_eigen.py ('naive', 'swapped') on the same twist, to show how much the
    result depends on the boundary terms (and that 'naive' is not power-conserving).
"""
import numpy as np
from scipy.linalg import expm

from wrmodes.modes import Mode, ModeBasis
from wrmodes.gsm import from_transfer_matrix
from twist_coupling import M_matrix, K_matrix_fast
from check_helical_eigen import cm_matrix
from rate_profiles import rate, L20, TH, INCH

TALL = (0.872 * INCH, 1.872 * INCH)
TE10, TE01, TE12 = Mode("TE", 1, 0), Mode("TE", 0, 1), Mode("TE", 1, 2)


def sliced_gsm(basis, f, M0, M1, prof="triangular", ns=150):
    dz = L20 / ns
    G = None
    for s in range(ns):
        g = from_transfer_matrix(expm((M0 + rate(prof, (s + 0.5) * dz, L20, TH) * M1) * dz), basis, f)
        G = g if G is None else G @ g
    return G


def report(basis, f, G, label):
    i = basis.index(TE10)
    p = basis.propagating_mask(f)
    t = np.abs(G.S21[:, i]) ** 2
    t12 = 100 * t[basis.index(TE12)] if TE12 in basis else float("nan")
    print(f"  {label}: TE0,1 {100 * t[basis.index(TE01)]:.4f} %, TE1,2 {t12:.4f} %, all spurious (propagating) "
          f"{100 * (t[p].sum() - t[i]):.4f} %")


def main():
    a, b = TALL
    f = 16.5e9
    print("1. tall WR187, 16.5 GHz, triangular: effect of evanescent modes")
    for cf in (1.0, 1.5, 2.0, 3.0):
        basis = ModeBasis.propagating(f, a, b, cutoff_factor=cf, evanescent_families="all")
        M0 = M_matrix(basis, f, 0.0)
        G = sliced_gsm(basis, f, M0, M_matrix(basis, f, 1.0) - M0)
        report(basis, f, G, f"cutoff_factor {cf} ({len(basis):3d} modes)")
    print("   slice convergence (propagating modes only)")
    basis = ModeBasis.propagating(f, a, b)
    M0 = M_matrix(basis, f, 0.0)
    for ns in (150, 600):
        report(basis, f, sliced_gsm(basis, f, M0, M_matrix(basis, f, 1.0) - M0, ns=ns), f"{ns} slices")
    print("2. ordinary WR187 (a = 1.872 in, b = 0.872 in), triangular, propagating modes")
    for f in (15e9, 16.5e9, 18e9):
        basis = ModeBasis.propagating(f, b, a)
        M0 = M_matrix(basis, f, 0.0)
        G = sliced_gsm(basis, f, M0, M_matrix(basis, f, 1.0) - M0)
        report(basis, f, G, f"{f / 1e9:4.1f} GHz")
    print("3. tall WR187, triangular, propagating modes: coupling variants")
    for f in (15e9, 16.5e9, 18e9):
        basis = ModeBasis.propagating(f, a, b)
        K = K_matrix_fast(basis)
        for var in ("note", "naive", "swapped"):
            M0 = cm_matrix(basis, K, f, 0.0, var)
            G = sliced_gsm(basis, f, M0, cm_matrix(basis, K, f, 1.0, var) - M0)
            report(basis, f, G, f"{f / 1e9:4.1f} GHz {var:7s} (unitarity defect {G.unitarity_defect():.1e})")


if __name__ == "__main__":
    main()
