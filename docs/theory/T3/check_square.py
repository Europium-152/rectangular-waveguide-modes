"""T3 check 3: square guide (a = b), polarization rotation (box T3-4).

1. Two-mode truncation TE10/TE01 (degenerate, r = 1, S = 0, K- = 0): a10 = cos(c theta) e^{-i beta z},
   a01 = -sin(c theta) e^{-i beta z} for ANY profile theta(z), c = 8/pi^2. The polarization angle psi
   (counterclockwise from v_hat; TE10 is along +v_hat, TE01 along -u_hat) is -c theta in the local frame and
   (1 - c) theta in the fixed frame. Checked with the transfer-matrix integrator for a non-uniform profile.
2. Full basis, uniform twist tau: the TE10/TE01 pair becomes two helical eigenmodes with gamma = beta -+ c_eff tau.
   c_eff from the eigenvalues of the full system (evanescent modes included) as a function of tau/k and of the
   basis size.
3. Finite twists in an overmoded square guide, all propagating modes, raised-cosine profile: output polarization
   angle and power left in TE10 + TE01, against the two-mode result.
"""
import os
import sys

import numpy as np
import scipy.constants as cst
from scipy.linalg import eig

sys.path.insert(0, os.path.dirname(__file__))
from twist_formulas import coupling_C, rate_profile, system_matrix, transfer_matrix  # noqa: E402
from wrmodes.gsm import from_transfer_matrix  # noqa: E402
from wrmodes.modes import ModeBasis  # noqa: E402

C8 = 8 / np.pi ** 2


def pol_angle(a10, a01):
    """Polarization angle psi (rad) of a10 e10 + a01 e01 in the local frame, counterclockwise from +v_hat.

    e10 points along +v_hat and e01 along -u_hat (CONVENTIONS.md fields), so (a10, a01) ~ (cos psi, sin psi) is
    the field cos(psi) v_hat - sin(psi) u_hat, i.e. v_hat rotated by +psi. Also returns the ellipticity
    |Im| / |Re| after removing the common phase."""
    ph = np.angle(a10) if abs(a10) > abs(a01) else np.angle(a01)
    x, y = a10 * np.exp(-1j * ph), a01 * np.exp(-1j * ph)
    psi = np.arctan2(y.real, x.real)
    psi = (psi + np.pi / 2) % np.pi - np.pi / 2         # a polarization direction is defined modulo pi
    return psi, np.hypot(x.imag, y.imag) / np.hypot(x.real, y.real)


def main():
    a = b = 0.02
    assert abs(coupling_C("TE0,1", "TE1,0", a, b) - C8) < 1e-15
    # 1. two-mode truncation, exact
    f = 10e9                                          # only TE10/TE01 propagate (7.49 < f < 10.6 GHz)
    two = ModeBasis(a, b, ["TE1,0", "TE0,1"])
    L = 0.3
    worst = 0.0
    for theta in (np.pi / 2, np.pi, 2.7):
        for kind in ("constant", "cos2", "triangular"):
            Phi = transfer_matrix(two, f, rate_profile(kind, theta, L), L, slices=200)
            beta = two.beta(f)[0].real
            y0 = np.zeros(4, complex)
            y0[two.index("TE1,0")] = 1.0                   # TE10 incident
            y = Phi @ y0
            a10, a01 = y[two.index("TE1,0")], y[two.index("TE0,1")]
            exact = np.exp(-1j * beta * L) * np.array([np.cos(C8 * theta), -np.sin(C8 * theta)])
            err = np.max(np.abs(np.array([a10, a01]) - exact))
            worst = max(worst, err, np.max(np.abs(y[2:])))      # no backward waves
            psi, ell = pol_angle(a10, a01)
            wrap = lambda x: (x + np.pi / 2) % np.pi - np.pi / 2       # noqa: E731
            assert abs(wrap(psi + C8 * theta)) < 1e-12
            if kind == "cos2":
                print(f"two-mode, theta = {np.degrees(theta):6.1f} deg: psi_local = {np.degrees(psi):8.3f} deg "
                      f"(-c theta mod 180 = {np.degrees(wrap(-C8 * theta)):8.3f}), fixed frame "
                      f"{np.degrees(wrap(psi + theta)):7.3f} ((1-c) theta = {np.degrees((1 - C8) * theta):7.3f}); "
                      f"max amplitude error over three profiles {err:.1e}")
    assert worst < 1e-10
    # the 4x4 system matrix of the pair: K- = 0 and K+ = [[0, c], [-c, 0]] exactly
    M1 = system_matrix(two, f, 1.0) - system_matrix(two, f, 0.0)
    i, j = two.index("TE1,0"), two.index("TE0,1")
    Kp = np.zeros((2, 2))
    Kp[i, j], Kp[j, i] = C8, -C8                     # da10/dz = ... + c theta' a01, da01/dz = ... - c theta' a10
    assert np.allclose(M1, np.kron(np.eye(2), Kp), atol=1e-15, rtol=0)
    print("two-mode truncation: K+ = [[0, c], [-c, 0]], K- = 0, rotation -c theta exact: PASS")

    # 2. full basis, uniform twist: c_eff = (gamma_+ - gamma_-) / (2 tau)
    print("\nuniform twist, square 20 x 20 mm at 10 GHz (only TE10/TE01 propagate), c_eff / c - 1:")
    k = 2 * np.pi * f / cst.c
    print("  basis (cutoff/f, modes) " + " ".join(f"tau/k={t:5.3f}" for t in (0.01, 0.03, 0.1, 0.2)))
    for cut in (2, 4, 8, 16):
        basis = ModeBasis.below(cut * f, a, b)
        row = []
        for t in (0.01, 0.03, 0.1, 0.2):
            tau = t * k
            lam = eig(system_matrix(basis, f, tau), right=False)
            gam = 1j * lam
            beta = basis.beta(f)[0].real
            near = np.sort(gam.real[(np.abs(gam.imag) < 1e-6 * k) & (np.abs(gam.real - beta) < 2 * tau)])
            near = near[near > 0]
            ceff = (near[-1] - near[0]) / (2 * tau)
            row.append(ceff / C8 - 1)
        print(f"  {cut:3d} f ({len(basis):5d} modes)        " + " ".join(f"{r:+11.2e}" for r in row))

    # 3. finite twists, overmoded square guide, all propagating modes
    a = b = 0.0508
    print(f"\nsquare {a*1e3:.1f} mm, raised-cosine 90-deg twists, all propagating modes, TE10 incident:")
    print("  f/GHz modes  L/m   psi_local (deg)  -c theta   power in TE10+TE01   reflected")
    for f in (8e9, 12e9):
        basis = ModeBasis.propagating(f, a, b)
        for L in (0.1, 0.3, 1.0):
            Phi = transfer_matrix(basis, f, rate_profile("cos2", np.pi / 2, L), L, slices=400)
            g = from_transfer_matrix(Phi, basis, f)
            i, j = basis.index("TE1,0"), basis.index("TE0,1")
            a10, a01 = g.S21[i, i], g.S21[j, i]
            psi, ell = pol_angle(a10, a01)
            pw = abs(a10) ** 2 + abs(a01) ** 2
            refl = np.sum(np.abs(g.S11[:, i]) ** 2)
            print(f"  {f/1e9:5.1f} {len(basis):4d} {L:5.2f}   {np.degrees(psi):8.3f} (ell {ell:.1e})  "
                  f"{np.degrees(-C8*np.pi/2):8.3f}    {pw:.6f}            {refl:.1e}")
            assert g.reciprocity_defect() < 1e-10 and g.unitarity_defect() < 1e-10


if __name__ == "__main__":
    main()
