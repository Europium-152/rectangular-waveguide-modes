"""Item 6 of T1b: resistive sheet(s) parallel to the broad walls (Butterweck 1968, eqs. 1-19).

Run from the repository root:   PYTHONPATH=src python3 docs/theory/T1b/butterweck_sheet.py

1. The transverse-resonance dispersion functions (any number of sheets) agree with the closed forms for one
   off-centre sheet (note, eqs. S1-S2) and, for a centred sheet, with Butterweck's eqs. (8) and (9).
2. Limit R -> infinity: the junction empty guide | sheet guide is the identity (S11 = 0, |S21| = 1), and the
   overlap matrix R is complex orthogonal (R R^T = I). This ties the sheet-mode fields to CONVENTIONS.md.
3. Butterweck's example (b = 10 mm, lambda = 4 mm, a = 22.86 mm, one centred sheet): optimum R and maximum
   attenuation of EH11^(2) (LSM^v, H_v = 0) and EH11^(1) (LSE^v, E_v = 0). Butterweck: 56 ohm, 1.5 dB/cm
   (eqs. 15-16, approximations for beta ~ k); EH11^(1): about 420 ohm and about 5 dB/cm (p. 277).
4. A filter section (empty | 10 cm sheet | empty): TE10 passes untouched; reciprocity and passivity.
"""
import os
import sys
import numpy as np
import scipy.constants as cst
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1b_coefficients as T  # noqa: E402

DB = 20 * np.log10(np.e)


def closed_lsm(ky, b, s, k, R):
    return np.sin(ky * b) + 1j * T.ETA0 * ky / (k * R) * np.sin(ky * s) * np.sin(ky * (b - s))


def closed_lse(ky, b, s, k, R):
    return ky * np.sin(ky * b) + 1j * k * T.ETA0 / R * np.sin(ky * s) * np.sin(ky * (b - s))


def butterweck8(ky, b, k, R):      # EH^(1), E_y = 0
    return 2 * R / T.ETA0 + 1j * k / ky * np.tan(ky * b / 2)


def butterweck9(ky, b, k, R):      # EH^(2), H_y = 0
    return 2 * R / T.ETA0 + 1j * ky / k * np.tan(ky * b / 2)


def check_dispersion():
    print("1. Dispersion functions")
    b, f = 10e-3, cst.c / 4e-3
    k = T.wavenumber(f)
    worst = 0.0
    for s_frac in (0.5, 0.3, 0.77):
        s = s_frac * b
        for R in (20.0, 56.0, 420.0):
            for kind in ("LSM", "LSE"):
                ky2 = T.sheet_ky2_roots(kind, [1, 2, 3], b, [(s, R)], k)
                for x in ky2:
                    ky = np.sqrt(x)
                    if kind == "LSM":
                        r = abs(closed_lsm(ky, b, s, k, R)) / (1 + abs(T.ETA0 * ky / (k * R)))
                    else:
                        r = abs(closed_lse(ky, b, s, k, R)) / (abs(ky) + k * T.ETA0 / R)
                    worst = max(worst, r)
                    if s_frac == 0.5 and abs(np.sin(ky * b / 2)) > 1e-6:
                        bw = butterweck9 if kind == "LSM" else butterweck8
                        worst = max(worst, abs(bw(ky, b, k, R)) / (2 * R / T.ETA0))
    print(f"   residual of the closed forms (S1, S2; Butterweck 8, 9 for s = b/2) at the roots: {worst:.1e}")
    assert worst < 1e-9
    return worst


def check_limit():
    print("2. R -> infinity: junction = LS^v rotation of CONVENTIONS.md section 7 (wrmodes.modes.ls_rotation)")
    from wrmodes.modes import ModeBasis, ls_rotation
    a, b, f = 22.86e-3, 10.16e-3, 75e9
    worst = 0.0
    for m, v0 in ((1, 0.0), (2, 1.3e-3), (0, -2.0e-3)):
        Nn = 6
        pec = ([("TE", m, 0)] if m >= 1 else []) + [("TE", m, n) for n in range(1, Nn + 1)] \
            + ([("TM", m, n) for n in range(1, Nn + 1)] if m >= 1 else [])
        sm = T.sheet_modes(m, a, b, [(v0, np.inf)], f, Nn)
        R = T.sheet_junction(pec, sm, m, a, b, f)
        S11, S12, S21, S22 = T.junction_gsm(R)
        basis = ModeBasis(a, b, pec, sort=False)
        Q, labels = ls_rotation(basis, f)
        # row of Q for each sheet mode: LS label (kind, m, n0); LSM n0 = 0 is TE_m0
        rows = [labels.index((md["kind"], m, md["n0"])) for md in sm]
        Qs = Q[rows, :]
        sign = np.sign(np.real(np.sum(S21 * Qs, axis=1)))
        e1 = np.max(abs(R @ R.T - np.eye(len(pec))))
        e2 = np.max(abs(S11))
        e3 = np.max(abs(S21 - sign[:, None] * Qs))
        worst = max(worst, e1, e2, e3)
        print(f"   m = {m}, v0 = {v0*1e3:+.1f} mm: |R R^T - I| = {e1:.1e}, |S11| = {e2:.1e},"
              f" |S21 - (+-) Q_LS| = {e3:.1e}")
    assert worst < 1e-10
    return worst


def butterweck_example():
    print("3. Butterweck's example: b = 10 mm, lambda = 4 mm, a = 22.86 mm, one centred sheet, m = 1")
    a, b = 22.86e-3, 10e-3
    f = cst.c / 4e-3
    k = T.wavenumber(f)
    km = np.pi / a
    res = {}
    for kind, label in (("LSM", "EH11^(2) (LSM^v, H_v = 0)"), ("LSE", "EH11^(1) (LSE^v, E_v = 0)")):
        def alpha(logR):
            ky2 = T.sheet_ky2_roots(kind, [1], b, [(b / 2, np.exp(logR))], k)[0]
            g = np.sqrt(k * k - km * km - ky2 + 0j)
            return -abs(g.imag)
        opt = minimize_scalar(alpha, bounds=(np.log(5), np.log(5000)), method="bounded",
                              options=dict(xatol=1e-6))
        Ropt, amax = np.exp(opt.x), -opt.fun
        # Butterweck's approximation alpha ~ Im(ky^2)/(2k) at the same R
        ky2 = T.sheet_ky2_roots(kind, [1], b, [(b / 2, Ropt)], k)[0]
        approx = ky2.imag / (2 * k)
        res[kind] = (Ropt, DB * amax / 100)
        print(f"   {label}: optimum R = {Ropt:.2f} ohm, max attenuation {DB*amax/100:.4f} dB/cm"
              f" (with beta ~ k as Butterweck: {DB*approx/100:.4f} dB/cm)")
    lam = 4e-3
    print(f"   Butterweck eqs. (15)-(16): R_opt = 0.372 Z0 lambda/b = {0.372*T.ETA0*lam/b:.1f} ohm,"
          f" alpha_max = 2.76/(k b^2) = {DB*2.76/(k*b*b)/100:.3f} dB/cm")
    Ro, ao = res["LSM"]
    print(f"   deviation from Butterweck's 56 ohm / 1.5 dB/cm: {Ro/56-1:+.1%} / {ao/1.5-1:+.1%}")
    # same with b = 10.16 mm (the actual X-band height; Philips Tech. Rev. eq. 4-5)
    b2 = 10.16e-3

    def alpha2(logR):
        ky2 = T.sheet_ky2_roots("LSM", [1], b2, [(b2 / 2, np.exp(logR))], k)[0]
        return -abs(np.sqrt(k * k - km * km - ky2 + 0j).imag)
    o2 = minimize_scalar(alpha2, bounds=(np.log(5), np.log(5000)), method="bounded", options=dict(xatol=1e-6))
    print(f"   with b = 10.16 mm: optimum {np.exp(o2.x):.2f} ohm, {-DB*o2.fun/100:.4f} dB/cm")
    return res


def filter_example():
    print("4. Filter: empty | 10 cm centred sheet, R = 56 ohm | empty, WR-90 at 75 GHz, family m = 1")
    a, b, f = 22.86e-3, 10.16e-3, 75e9
    for Nn in (6, 10, 14):
        S, pec, sm = T.sheet_filter_gsm(1, a, b, [(0.0, 56.0)], f, 0.10, Nn)
        S21 = S[2]
        i10, i11, j11 = pec.index(("TE", 1, 0)), pec.index(("TE", 1, 1)), pec.index(("TM", 1, 1))
        Sf = np.block([[S[0], S[1]], [S[2], S[3]]])
        rec = np.max(abs(Sf - Sf.T))
        k = T.wavenumber(f)
        prop = [i for i, p in enumerate(pec) if T.cutoff(p, a, b) < k]
        idx = prop + [len(pec) + i for i in prop]
        sv = np.linalg.svd(Sf[np.ix_(idx, idx)], compute_uv=False)[0]
        print(f"   Nn = {Nn:2d}: |S21(TE10)| - 1 = {abs(S21[i10,i10])-1:.1e}; TE11 -> TE11 {20*np.log10(abs(S21[i11,i11])):.2f} dB,"
              f" TM11 -> TM11 {20*np.log10(abs(S21[j11,j11])):.2f} dB, TE11 -> TM11 {20*np.log10(abs(S21[j11,i11])):.2f} dB;"
              f" reciprocity {rec:.1e}; largest singular value (propagating) {sv:.6f}")


def three_sheets():
    print("5. Three equidistant sheets (v0 = -b/4, 0, b/4), 70 ohm, WR-90 at 75 GHz (Philips Tech. Rev. experiment):")
    a, b, f = 22.86e-3, 10.16e-3, 75e9
    sheets = [(-b / 4, 70.0), (0.0, 70.0), (b / 4, 70.0)]
    k = T.wavenumber(f)
    for m in (0, 1, 2, 3):
        sm = T.sheet_modes(m, a, b, sheets, f, 6)
        prop = [md for md in sm if (m * np.pi / a) ** 2 + (md["n0"] * np.pi / b) ** 2 < k * k]
        txt = ", ".join(f"{md['kind']}{m},{md['n0']} {DB*(-md['gamma'].imag)/100:.3f}" for md in prop)
        print(f"   m = {m}: attenuation (dB/cm) of the modes from propagating empty-guide modes: {txt}")


if __name__ == "__main__":
    check_dispersion()
    check_limit()
    butterweck_example()
    filter_example()
    three_sheets()
    print("all checks passed")
