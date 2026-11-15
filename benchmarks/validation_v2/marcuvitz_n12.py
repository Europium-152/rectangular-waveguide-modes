"""WP7b, wave 1 leftover (WP4 open issue 2, card WP7b item 5): where do Marcuvitz's frequency-dependent formulas for
the symmetric H-plane step (Waveguide Handbook 1951, Sec. 5.24a, eqs. 1a-3a, p. 296-298) go wrong?

Reference: the merged mode-matching `Step` (WP4, converged default truncation), converted to Marcuvitz's equivalent
circuit: shunt reactance jX at the junction plane T between the line (Z0, lambda_g) of the larger guide (width a,
reference plane T) and the line (Z0', lambda_g') of the smaller guide (width a' = alpha a), whose reference plane T'
lies a distance l from T on the side of the larger guide (Fig. 5.24-1). From S11 (large side, plane T):
y = (1 - S11)/(1 + S11) = Z0/Z0' - j Z0/X; from the phase of S22 at the junction plane, l (S22 at T' equals S22 at
the junction times exp(2 i beta' l)).

The exact circuit then gives, through (3a), (1a) and (2a) inverted, the impedance elements X0, X11, X22 that the
static-method formulas should reproduce; X11 and X22 contain the frequency terms N11, N22, N12. Solving the X11 and
the X22 equation for N12 with the printed N11 and N22 gives two 'required' N12 values; if they agree with each other
and differ from the printed N12, the error is in N12 (and not in N11, N22 or the outer formulas). The script also
evaluates (2a) with lambda_g' in place of lambda_g in its denominator (the other suspected misprint).

Elliptic integrals with modulus alpha (F(alpha) = K(m = alpha^2)), as settled by L1 against Fig. 5.24-2.
"""
import json

import numpy as np
from scipy.special import ellipe, ellipk

from wrmodes.modematch import Step
from wrmodes.modes import Mode

import vcommon as vc

C0 = 299792458.0


def static(al):
    q = (1 - al) / (1 + al)
    R1 = -q ** al
    R2 = q ** (1 / al)
    T = 4 * al / (1 - al ** 2) * q ** (0.5 * (al + 1 / al))
    den = (1 - R1) * (1 - R2) - T ** 2
    A = ((1 + R1) * (1 - R2) + T ** 2) / den
    Ap = ((1 - R1) * (1 + R2) + T ** 2) / den
    C = 2 * T / den
    return dict(q=q, R1=R1, R2=R2, T=T, A=A, Ap=Ap, C=C)


def n_terms(al, a_lam):
    s = static(al)
    q, R1, R2, T = s["q"], s["R1"], s["R2"], s["T"]
    alp = np.sqrt(1 - al ** 2)
    F, E = ellipk(al ** 2), ellipe(al ** 2)
    Fp, Ep = ellipk(alp ** 2), ellipe(alp ** 2)
    ap_lam = al * a_lam
    Q = 1 - np.sqrt(1 - (2 * a_lam / 3) ** 2)
    Qp = 1 - np.sqrt(1 - (2 * ap_lam / 3) ** 2)
    hq = Q - 0.5 * (2 * a_lam / 3) ** 2
    hqp = Qp - 0.5 * (2 * ap_lam / 3) ** 2
    pre = 2 * a_lam ** 2
    N11_1 = pre * (1 + 16 * R1 / (np.pi * (1 - al ** 2)) * (E - alp ** 2 * F) * (Ep - al ** 2 * Fp) - R1 ** 2 - al ** 2 * T ** 2)
    N11 = N11_1 + 12 * al ** 4 / (1 - al ** 2) ** 2 * q ** (4 * al) * hq + 48 * al ** 2 / (1 - al ** 2) ** 2 * q ** (al + 3 / al) * hqp
    N22_1 = pre * (al ** 2 + 16 * al ** 2 * R2 * E / (np.pi * (1 - al ** 2)) * (Fp - Ep) - al ** 2 * R2 ** 2 - T ** 2)
    N22 = N22_1 + 48 * al ** 2 / (1 - al ** 2) ** 2 * q ** (3 * al + 1 / al) * hq + 12 / (1 - al ** 2) ** 2 * q ** (4 / al) * hqp
    N12_brace = alp ** 2 - 4 * E / np.pi * (Ep - al ** 2 * Fp) - R1 + al ** 2 * R2
    N12_1 = pre * N12_brace * T
    N12_q = 24 * al ** 3 / (1 - al ** 2) ** 2 * q ** (0.5 * (7 * al + 1 / al)) * hq + 24 * al / (1 - al ** 2) ** 2 * q ** (0.5 * (al + 7 / al)) * hqp
    # candidate correction (WP7b): -R1 - alpha^2 R2 inside the brace, i.e. the cross term of the bilinear structure
    # of N11 (1 - R1^2 - alpha^2 T^2) and N22 (alpha^2 - alpha^2 R2^2 - T^2): -(R1 + alpha^2 R2) T
    N12_corr = pre * (N12_brace - 2 * al ** 2 * R2) * T + N12_q
    return dict(N11=N11, N22=N22, N12=N12_1 + N12_q, N12_first=N12_1, N12_q=N12_q, N12_brace=N12_brace,
                N12_corrected=N12_corr, **s)


def marcuvitz(al, a_lam, N12=None, lg_prime_in_2a=False):
    """Circuit parameters of (1a)-(3a) (X/Z0, Z0'/Z0, l/a), with an optional replacement value of N12."""
    n = n_terms(al, a_lam)
    A, Ap, C = n["A"], n["Ap"], n["C"]
    N11, N22 = n["N11"], n["N22"]
    N12 = n["N12"] if N12 is None else N12
    X11 = A / (1 - ((A + 1) ** 2 * N11 + 2 * (A + 1) * C * N12 + C ** 2 * N22) / (4 * A))
    X22 = Ap / (1 - ((Ap + 1) ** 2 * N22 + 2 * (Ap + 1) * C * N12 + C ** 2 * N11) / (4 * Ap))
    X0 = Ap - C ** 2 / A
    lam = 1.0
    a = a_lam
    ap = al * a
    lg = lam / np.sqrt(1 - (lam / (2 * a)) ** 2)
    lgp = lam / np.sqrt(1 - (lam / (2 * ap)) ** 2)
    K = 1 + (2 * ap * X0 / lgp) ** 2
    zr = (lgp * ap) / (lg * a) * X11 / (al ** 2 * (X22 - X0)) * K
    lden = lgp if lg_prime_in_2a else lg
    x = 2 * a / lg * X11 * K / (1 + (2 * ap / lden) ** 2 * X0 * X22)
    l_a = al * X0 / np.pi
    return dict(x=x, zr=zr, l_a=l_a, X11=X11, X22=X22, X0=X0, lg=lg, lgp=lgp, **n)


def mode_matching(al, a_lam, a=22.86e-3):
    """X/Z0, Z0'/Z0, l/a from the merged Step (large guide = port 1)."""
    lam = a / a_lam
    f = C0 / lam
    b = 0.2 * lam                              # only the family n = 0 propagates
    st = Step(a, b, al * a, b, plane="H")
    g = st.gsm(f)
    te = Mode("TE", 1, 0)
    i1, i2 = g.basis1.index(te), g.basis2.index(te)
    S11, S22 = g.S11[i1, i1], g.S22[i2, i2]
    y = (1 - S11) / (1 + S11)
    zr = 1 / y.real
    x = -1 / y.imag
    yp = zr * (1 - 1j / x)
    G2 = (1 - yp) / (1 + yp)
    lgp = lam / np.sqrt(1 - (lam / (2 * al * a)) ** 2)
    bp = 2 * np.pi / lgp
    phase = np.angle(G2 / S22)                  # = 2 beta' l (mod 2 pi)
    l = phase / (2 * bp)
    return dict(x=x, zr=zr, l_a=l / a, S11=S11, S22=S22, mag_check=abs(abs(G2) - abs(S22)),
                n_modes=len(g.basis1.modes))


def invert(mm, al, a_lam):
    """X0, X11, X22 that (1a)-(3a) need to reproduce the mode-matching circuit."""
    lam, a = 1.0, a_lam
    ap = al * a
    lg = lam / np.sqrt(1 - (lam / (2 * a)) ** 2)
    lgp = lam / np.sqrt(1 - (lam / (2 * ap)) ** 2)
    X0 = np.pi * mm["l_a"] / al
    K = 1 + (2 * ap * X0 / lgp) ** 2
    zeta = mm["zr"] * (lg * a) / (lgp * ap)
    c = (2 * ap / lg) ** 2
    g = 2 * a / lg * zeta * al ** 2
    X22 = (g * X0 + mm["x"]) / (g - mm["x"] * c * X0)
    X11 = zeta * al ** 2 * (X22 - X0) / K
    return dict(X0=X0, X11=X11, X22=X22)


def required_n12(req, n):
    """N12 from the X11 equation and from the X22 equation, with the printed N11 and N22."""
    A, Ap, C, N11, N22 = n["A"], n["Ap"], n["C"], n["N11"], n["N22"]
    # X11 = A / (1 - B1/(4A)) -> B1 = 4A (1 - A/X11)
    B1 = 4 * A * (1 - A / req["X11"])
    B2 = 4 * Ap * (1 - Ap / req["X22"])
    n12_from_11 = (B1 - (A + 1) ** 2 * N11 - C ** 2 * N22) / (2 * (A + 1) * C)
    n12_from_22 = (B2 - (Ap + 1) ** 2 * N22 - C ** 2 * N11) / (2 * (Ap + 1) * C)
    return n12_from_11, n12_from_22


GRID = [(a_lam, al) for a_lam in (0.6, 0.7, 0.85, 0.99, 1.2, 1.39) for al in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        if 2 * al * a_lam > 1.02]

if __name__ == "__main__":
    rows = []
    for a_lam, al in GRID:
        mm = mode_matching(al, a_lam)
        mz = marcuvitz(al, a_lam)
        mz_lgp = marcuvitz(al, a_lam, lg_prime_in_2a=True)
        req = invert(mm, al, a_lam)
        r11, r22 = required_n12(req, mz)
        mz_req = marcuvitz(al, a_lam, N12=0.5 * (r11 + r22))
        mz_c = marcuvitz(al, a_lam, N12=mz["N12_corrected"])
        mz_cp = marcuvitz(al, a_lam, N12=mz["N12_corrected"], lg_prime_in_2a=True)
        mz_0 = marcuvitz(al, a_lam, N12=0.0)
        lg = mz["lg"]
        row = dict(a_lam=a_lam, alpha=al,
                   mm_xn=mm["x"] * lg / (2 * a_lam), mz_xn=mz["x"] * lg / (2 * a_lam),
                   mz_lgp_xn=mz_lgp["x"] * lg / (2 * a_lam), mm_zn=mm["zr"] * lg * a_lam / (mz["lgp"] * al * a_lam),
                   mz_zn=mz["zr"] * lg / (mz["lgp"] * al), mm_l_a=mm["l_a"], mz_l_a=mz["l_a"],
                   X0_req=req["X0"], X0_static=mz["X0"], X11_req=req["X11"], X11=mz["X11"], X22_req=req["X22"],
                   X22=mz["X22"], N11=mz["N11"], N22=mz["N22"], N12=mz["N12"], N12_first=mz["N12_first"],
                   N12_q=mz["N12_q"], N12_brace=mz["N12_brace"], T=mz["T"], R1=mz["R1"], R2=mz["R2"],
                   N12_req_11=r11, N12_req_22=r22, mz_req_xn=mz_req["x"] * lg / (2 * a_lam),
                   N12_corrected=mz["N12_corrected"], mzc_xn=mz_c["x"] * lg / (2 * a_lam),
                   mzcp_xn=mz_cp["x"] * lg / (2 * a_lam), mz0_xn=mz_0["x"] * lg / (2 * a_lam),
                   mzc_zn=mz_c["zr"] * lg / (mz["lgp"] * al),
                   mm_mag_check=mm["mag_check"], mm_modes=mm["n_modes"])
        rows.append(row)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    vc.save_npz("marcuvitz_524a.npz", __file__, arr, grid=GRID,
                method="merged WP4 Step (mode matching) converted to Marcuvitz Sec. 5.24a equivalent circuit")
