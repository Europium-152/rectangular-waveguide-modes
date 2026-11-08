"""Twist-rate profiles, first-order conversion integrals and Doane's eq. 18 (section 6 of the note).

Run:  cd <worktree> && PYTHONPATH=$PWD/src:$PWD/docs/theory/T3b python3 docs/theory/T3b/rate_profiles.py

Profiles of the twist angle theta(z), 0 <= z <= L, total angle Th (box 7):
  constant    theta'(z) = Th / L
  cosine      theta(z) = (Th/2)(1 - cos(pi z / L)),  theta'(z) = (pi Th / 2L) sin(pi z / L)
  triangular  theta'(z) = (2Th/L)(2z/L) for z < L/2, (2Th/L)(2(L - z)/L) for z > L/2   (Doane eq. 17)

First-order (Born) conversion from the incident forward mode p into the forward mode q and backward mode q:
  a+_q(L) = A_qp F(beta_q - beta_p) e^{-i beta_q L},   a-_q(0) = -B_qp F(-(beta_p + beta_q)),
  F(D) = int_0^L theta'(z) e^{i D z} dz, with |F| =
      Th |sinc(D L / 2)|                              (constant)
      Th pi^2 |cos(D L / 2)| / |pi^2 - (D L)^2|        (cosine)
      Th [sin(D L / 4) / (D L / 4)]^2                  (triangular)

Doane & Anderson (1989) eq. 18 (p. 159), from Unger (1957) eq. 13:  P_c ~ (8 c Th)^2 / (D L)^4, with
|D L| ~ (L lambda / a^2) |a^2 / b^2 - 1| pi / 4 and c ~ 8/pi^2. The exact first-order triangular result is
P = 256 c^2 Th^2 sin^4(D L/4) / (D L)^4: its maximum over frequency is 4 times eq. 18 and its average 1.5 times.

The guide is the 20-inch WR187 twist of Doane's Fig. 9, 'tall' orientation (a = 0.872 in along u, b = 1.872 in
along v, TE1,0 has E along the long side), and also the ordinary orientation for comparison.
"""
import numpy as np
import scipy.constants as cst
from scipy.linalg import expm

from wrmodes.modes import Mode, ModeBasis
from twist_coupling import A_B_matrices, M_matrix, C_TE10_TE01

INCH = 0.0254
L20 = 20 * INCH
TH = np.pi / 2


def rate(profile, z, L, Th):
    z = np.asarray(z, dtype=float)
    if profile == "constant":
        return np.full_like(z, Th / L)
    if profile == "cosine":
        return np.pi * Th / (2 * L) * np.sin(np.pi * z / L)
    if profile == "triangular":
        return np.where(z < L / 2, 2 * Th / L * 2 * z / L, 2 * Th / L * 2 * (L - z) / L)
    raise ValueError(profile)


def F_abs(profile, D, L, Th):
    x = D * L
    if profile == "constant":
        return Th * np.abs(np.sinc(x / (2 * np.pi)))
    if profile == "cosine":
        return Th * np.pi ** 2 * np.abs(np.cos(x / 2)) / np.abs(np.pi ** 2 - x ** 2)
    if profile == "triangular":
        return Th * np.sinc(x / (4 * np.pi)) ** 2
    raise ValueError(profile)


def F_quad(profile, D, L, Th, n=4000):
    z, w = np.polynomial.legendre.leggauss(n)
    if profile == "triangular":          # split at the kink
        z1, w1 = (z + 1) * L / 4, w * L / 4
        zz, ww = np.concatenate([z1, z1 + L / 2]), np.concatenate([w1, w1])
    else:
        zz, ww = (z + 1) * L / 2, w * L / 2
    return abs(np.sum(ww * rate(profile, zz, L, Th) * np.exp(1j * D * zz)))


def first_order(basis, f, p, profile, L=L20, Th=TH):
    """Forward and backward first-order conversion powers from mode p into each mode of the basis."""
    A, B = A_B_matrices(basis, f)
    beta = basis.beta(f).real
    i = basis.index(p)
    fw = np.array([abs(A[j, i]) ** 2 * F_abs(profile, beta[j] - beta[i], L, Th) ** 2 for j in range(len(basis))])
    bw = np.array([abs(B[j, i]) ** 2 * F_abs(profile, -(beta[j] + beta[i]), L, Th) ** 2 for j in range(len(basis))])
    return fw, bw


def eq18(f, a, b, L=L20, Th=TH, exact_dbeta=False):
    lam = cst.c / f
    if exact_dbeta:
        k = 2 * np.pi / lam
        D = abs(np.sqrt(k ** 2 - (np.pi / a) ** 2) - np.sqrt(k ** 2 - (np.pi / b) ** 2))
        DL = D * L
    else:
        DL = L * lam / a ** 2 * abs(a ** 2 / b ** 2 - 1) * np.pi / 4
    return (8 * C_TE10_TE01 * Th) ** 2 / DL ** 4


def full_cm(basis, f, p, profile, L=L20, Th=TH, nslice=400):
    """Full coupled-mode solution (propagating modes, forward + backward) by midpoint slices: power of p and of
    the converted modes at port 2, and reflected power at port 1."""
    from wrmodes.gsm import from_transfer_matrix
    dz = L / nslice
    Phi = np.eye(2 * len(basis), dtype=complex)
    M0 = M_matrix(basis, f, 0.0)
    M1 = M_matrix(basis, f, 1.0) - M0
    for s in range(nslice):
        tau = rate(profile, (s + 0.5) * dz, L, Th)
        Phi = expm((M0 + tau * M1) * dz) @ Phi
    S = from_transfer_matrix(Phi, basis, f)
    i = basis.index(p)
    t = np.abs(S.S21[:, i]) ** 2
    r = np.abs(S.S11[:, i]) ** 2
    return t, r, S


def main():
    a, b = 0.872 * INCH, 1.872 * INCH          # tall WR187: TE1,0 has E along the long side b
    print("check of the closed forms of |F| against quadrature")
    worst = 0.0
    for prof in ("constant", "cosine", "triangular"):
        for D in (3.0, 17.0, 29.0, 61.0, 400.0):
            worst = max(worst, abs(F_abs(prof, D, L20, TH) - F_quad(prof, D, L20, TH)) / TH)
    print(f"  max |F_closed - F_quad| / Th = {worst:.1e}")
    assert worst < 1e-10

    print("\neq. 18 (Doane) for the 20-inch 90-degree WR187 twist, TE1,0 - TE0,1:")
    for f in (15e9, 16.5e9, 18e9):
        print(f"  {f / 1e9:5.1f} GHz: approx. Delta beta {100 * eq18(f, a, b):.3f} %, exact Delta beta "
              f"{100 * eq18(f, a, b, exact_dbeta=True):.3f} %")

    for label, (aa, bb) in (("tall WR187 (a = 0.872 in, b = 1.872 in), incident TE1,0 (E along 1.872 in)", (a, b)),
                            ("ordinary WR187 (a = 1.872 in, b = 0.872 in), incident TE1,0 (E along 0.872 in)", (b, a))):
        print(f"\n{label}")
        print("  first order, power in % of the incident power: TE0,1 only / all forward / all backward")
        for prof in ("constant", "cosine", "triangular"):
            vals = []
            for f in (15e9, 16.5e9, 18e9):
                basis = ModeBasis.propagating(f, aa, bb)
                fw, bw = first_order(basis, f, Mode("TE", 1, 0), prof)
                vals.append((100 * fw[basis.index(Mode("TE", 0, 1))], 100 * fw.sum(), 100 * bw.sum()))
            print(f"  {prof:10s} " + "  ".join(f"{f:4.1f} GHz: {x[0]:.4f} / {x[1]:.4f} / {x[2]:.1e}"
                                               for f, x in zip((15, 16.5, 18), vals)))
        f = 16.5e9
        basis = ModeBasis.propagating(f, aa, bb)
        fw, _ = first_order(basis, f, Mode("TE", 1, 0), "triangular")
        top = np.argsort(fw)[::-1][:4]
        print("  triangular, 16.5 GHz, largest first-order products: "
              + ", ".join(f"{basis.modes[j].name} {100 * fw[j]:.2e} %" for j in top))

    print("\nsweep 15-18 GHz, tall WR187, incident TE1,0: full coupled modes (all propagating modes, forward and")
    print("backward, 150 midpoint slices), power in %: TE0,1 / all spurious [largest other mode]; first order in ( )")
    for prof in ("constant", "cosine", "triangular"):
        print(f"  {prof}")
        for f in np.arange(15e9, 18.01e9, 0.5e9):
            basis = ModeBasis.propagating(f, a, b)
            fw, bw = first_order(basis, f, Mode("TE", 1, 0), prof)
            t, r, S = full_cm(basis, f, Mode("TE", 1, 0), prof, nslice=150)
            i, j = basis.index(Mode("TE", 1, 0)), basis.index(Mode("TE", 0, 1))
            other = [k for k in np.argsort(t)[::-1] if k not in (i, j)][0]
            print(f"    {f / 1e9:5.2f} GHz ({len(basis):2d} modes): {100 * t[j]:.4f} / {100 * (t.sum() - t[i]):.4f} "
                  f"[{basis.modes[other].name} {100 * t[other]:.4f}]   ({100 * fw[j]:.4f} / {100 * fw.sum():.4f})   "
                  f"refl. {r.sum():.0e}, unitarity {S.unitarity_defect():.0e}, reciprocity {S.reciprocity_defect():.0e}")


if __name__ == "__main__":
    main()
