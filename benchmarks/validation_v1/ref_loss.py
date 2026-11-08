"""Surface-integral reference for ohmic wall loss (WP7 item 2, verifies WP2).

For a forward mode p of unit amplitude (CONVENTIONS.md section 5) the fields are

    E_t = sqrt(2/Y) e,   H_t = sqrt(2 Y) z_hat x e,   H_z = (i / (omega mu0)) sqrt(2/Y) (curl e)_z,

the last from Faraday's law, curl E = -i omega mu0 H, whose z component involves only E_t. The transverse field e is
taken from ``wrmodes.modes.transverse_field`` and its curl is computed by fourth-order central differences of that
function (no closed form, no ``wrmodes.modes.divergence``). On the walls,

    P_loss = (1/2) R_s  oint |H_tan|^2 dl,     H_tan = (H_v, H_z) on u = +-a/2,  (H_u, H_z) on v = +-b/2,

with Gauss-Legendre quadrature on each wall, and the power P = (1/2) Re int (E x H*) . z_hat ds by Gauss-Legendre
quadrature over the cross-section (1 for the normalization of CONVENTIONS.md). The attenuation constant of the
field amplitude is alpha = P_loss / (2 P) (Np/m); 8.686 alpha is in dB/m.

The perturbation is applied to every propagating mode, including the degenerate TE_mn/TM_mn pairs (m, n >= 1),
for which the wall integral also couples the pair. ``loss_matrix`` returns the full real symmetric matrix
A_pq = (1/4) R_s oint H_tan,p . conj(H_tan,q) dl (for P_p = P_q = 1), with diag(A) = alpha, so that the amplitudes of
a lossy straight obey da/dz = -(i beta + A) a to first order in R_s (the reactive part of Z_s = (1+i) R_s, which
shifts beta by the same amount, is not part of this reference: it does not change |S|). Per-wall matrices are
returned too.

``textbook_alpha`` gives the classical closed forms (e.g. Collin, *Field Theory of Guided Waves*; Pozar eq. 3.96 for
TE10) for an independent check of the integral.
"""
import numpy as np
import scipy.constants as cst

from wrmodes.modes import Mode, ModeBasis, transverse_field, wave_admittance, cutoff_wavenumber

ETA0 = cst.mu_0 * cst.c
SIGMA_AL = 3.5e7       # SPARC baseline aluminium (CONVENTIONS.md section 1)
SIGMA_CU = 5.8e7       # copper
NP_TO_DB = 20 / np.log(10)


def surface_resistance(f, sigma):
    """R_s = sqrt(omega mu0 / (2 sigma)) (ohm)."""
    return np.sqrt(2 * np.pi * f * cst.mu_0 / (2 * sigma))


def _curl_z(mode, a, b, u, v, h=None):
    """(curl e)_z = d e_v/du - d e_u/dv by 4th-order central differences of transverse_field."""
    h = 1e-4 * min(a, b) if h is None else h

    def ev(uu):
        return transverse_field(mode, a, b, uu, v)[1]

    def eu(vv):
        return transverse_field(mode, a, b, u, vv)[0]

    d_ev = (-ev(u + 2 * h) + 8 * ev(u + h) - 8 * ev(u - h) + ev(u - 2 * h)) / (12 * h)
    d_eu = (-eu(v + 2 * h) + 8 * eu(v + h) - 8 * eu(v - h) + eu(v - 2 * h)) / (12 * h)
    return d_ev - d_eu


def fields(mode, a, b, f, u, v):
    """(E_u, E_v, H_u, H_v, H_z) of the unit-amplitude forward mode at points (u, v)."""
    mode = Mode.parse(mode)
    Y = wave_admittance(mode, a, b, f)
    omega = 2 * np.pi * f
    eu, ev = transverse_field(mode, a, b, u, v)
    sE, sH = np.sqrt(2 / Y), np.sqrt(2 * Y)
    Hz = 1j / (omega * cst.mu_0) * sE * _curl_z(mode, a, b, u, v)
    return sE * eu, sE * ev, -sH * ev, sH * eu, Hz


def _walls(a, b, nq):
    x, w = np.polynomial.legendre.leggauss(nq)
    walls = {}
    for name, (uu, vv, dl, tang) in {
        "u-": (np.full(nq, -a / 2), x * b / 2, b / 2, "v"), "u+": (np.full(nq, a / 2), x * b / 2, b / 2, "v"),
        "v-": (x * a / 2, np.full(nq, -b / 2), a / 2, "u"), "v+": (x * a / 2, np.full(nq, b / 2), a / 2, "u"),
    }.items():
        walls[name] = (uu, vv, w * dl, tang)
    return walls


def power(mode, a, b, f, nq=64):
    """(1/2) Re int (E x H*) . z_hat ds by quadrature (1 for a propagating mode)."""
    x, w = np.polynomial.legendre.leggauss(nq)
    U, V = np.meshgrid(x * a / 2, x * b / 2, indexing="ij")
    W = np.outer(w * a / 2, w * b / 2)
    Eu, Ev, Hu, Hv, _ = fields(mode, a, b, f, U, V)
    return 0.5 * np.real(np.sum(W * (Eu * np.conj(Hv) - Ev * np.conj(Hu))))


def loss_matrix(basis, f, sigma, nq=96, per_wall=False):
    """A_pq = (1/4) R_s oint H_tan,p . conj(H_tan,q) dl / sqrt(P_p P_q) (Np/m) over the propagating modes of basis.

    Returns (A, names) or (A, names, {wall: A_wall}) with per_wall=True. diag(A) is alpha of every mode.
    """
    if not np.all(basis.propagating_mask(f)):
        raise ValueError("loss_matrix: only propagating modes")
    a, b = basis.a, basis.b
    Rs = surface_resistance(f, sigma)
    N = len(basis)
    P = np.array([power(m, a, b, f) for m in basis.modes])
    out = {}
    for name, (uu, vv, wq, tang) in _walls(a, b, nq).items():
        H = []
        for m in basis.modes:
            _, _, Hu, Hv, Hz = fields(m, a, b, f, uu, vv)
            H.append((Hv if tang == "v" else Hu, Hz))
        A = np.zeros((N, N))
        for p in range(N):
            for q in range(p, N):
                val = 0.25 * Rs * np.sum(wq * (H[p][0] * np.conj(H[q][0]) + H[p][1] * np.conj(H[q][1])))
                A[p, q] = A[q, p] = np.real(val) / np.sqrt(P[p] * P[q])
                assert abs(np.imag(val)) <= 1e-9 * max(abs(val), 1e-300) + 1e-300
        out[name] = A
    A = sum(out.values())
    return (A, basis.names, out) if per_wall else (A, basis.names)


def alpha(mode, a, b, f, sigma, nq=96):
    """Attenuation constant (Np/m) of one propagating mode, P_loss / (2 P), from the wall integral."""
    basis = ModeBasis(a, b, [Mode.parse(mode)])
    A, _ = loss_matrix(basis, f, sigma, nq=nq)
    return float(A[0, 0])


def textbook_alpha(mode, a, b, f, sigma):
    """Classical closed forms of alpha (Np/m) for TE_mn and TM_mn (m along a, n along b), for checking."""
    mode = Mode.parse(mode)
    m, n = mode.m, mode.n
    Rs = surface_resistance(f, sigma)
    fc = cutoff_wavenumber(mode, a, b) * cst.c / (2 * np.pi)
    r = (fc / f) ** 2
    s = np.sqrt(1 - r)
    if mode.kind == "TM":
        return 2 * Rs / (ETA0 * b * s) * (m ** 2 * b ** 3 + n ** 2 * a ** 3) / (m ** 2 * b ** 2 * a + n ** 2 * a ** 3)
    if n == 0:
        return Rs / (ETA0 * b * s) * (1 + 2 * b / a * r)
    if m == 0:
        return Rs / (ETA0 * a * s) * (1 + 2 * a / b * r)
    q = b / a
    return 2 * Rs / (ETA0 * b * s) * ((1 + q) * r + (1 - r) * (q * (q * m ** 2 + n ** 2)) / (q ** 2 * m ** 2 + n ** 2))


# ----------------------------------------------------------------------------------------- reference cases
GUIDES = {"WR-90": (0.9 * 0.0254, 0.4 * 0.0254), "square 20 mm": (0.020, 0.020)}
MATERIALS = {"aluminium": SIGMA_AL, "copper": SIGMA_CU}
FREQUENCIES = {"WR-90": [18e9, 30e9, 45e9, 60e9, 90e9], "square 20 mm": [12e9, 20e9, 30e9, 40e9]}
