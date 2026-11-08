"""Final formulas of theory note T1b (docs/theory/T1b-wall-impedance.md), as plain functions.

Every boxed formula of the note has one function here, with the physical inputs of card T1: mode labels
('TE' | 'TM', m, n), guide sides a and b (m), frequency f (Hz), and wall impedances in ohm. The functions use only
numpy and the constants of CONVENTIONS.md (eta0 = mu0 c); they do not call the coupled-mode code of wrmodes, so
that the check scripts can compare them with field quadratures built on ``wrmodes.modes``.

Conventions (CONVENTIONS.md): exp(+i omega t); forward modes exp(-i beta z), Im beta <= 0; amplitudes
E_t = sum (a+ + a-) sqrt(2/Y) e,  H_t = sum (a+ - a-) sqrt(2Y) z x e;  e_TM = grad Pi, e_TE = grad Gamma x z.

Walls are labelled 'u-' (u = -a/2), 'u+' (u = +a/2), 'v-' (v = -b/2) and 'v+' (v = +b/2). A wall impedance argument
``Z`` is either a scalar (all four walls) or a dict with some of these keys (missing walls are perfect conductors).
The Leontovich condition is E_tan = Z_s H_tan x n_hat, n_hat pointing from the guide into the wall.

Contents
--------
item 1   surface_resistance, surface_impedance
item 2   wall_A, wall_B                      boundary integrals (closed forms, per wall)
         lambda_pp, lambda_pm                Lambda^{++} = Lambda^{--},  Lambda^{+-} = Lambda^{-+}
         lambda_full                          the 2N x 2N matrix entering dy/dz = (D + kappa K - Lambda_full) y
         dV_dI_boundary                       the same result in the T0 voltage/current form
item 3   alpha_te_marcuvitz, alpha_tm_marcuvitz   textbook attenuation constants (Marcuvitz 1951, sec. 2.2)
item 4   wall_A1, wall_B1, lambda1_full        first moments (weight u) of the wall integrals, for bends
item 5   lsu_dispersion, lsu_kx_roots, impedance_guide_modes, lossy_wall_filter_gsm, butterweck_eq21
item 6   sheet_dispersion, sheet_ky_root, sheet_modes, sheet_junction_gsm, sheet_filter_gsm
"""
import numpy as np
import scipy.constants as cst

ETA0 = cst.mu_0 * cst.c
WALLS = ("u-", "u+", "v-", "v+")


# ============================================================================== helpers (CONVENTIONS.md)
def _eps(l):
    return 1.0 if l == 0 else 2.0


def wavenumber(f):
    return 2 * np.pi * f / cst.c


def cutoff(mode, a, b):
    kind, m, n = mode
    return np.pi * np.hypot(m / a, n / b)


def beta(mode, a, b, f):
    """Propagation constant with the branch of CONVENTIONS.md section 3 (-i|beta| below cutoff)."""
    k, chi = wavenumber(f), cutoff(mode, a, b)
    d = k * k - chi * chi
    return np.sqrt(d) + 0j if d >= 0 else -1j * np.sqrt(-d)


def admittance(mode, a, b, f):
    """Wave admittance Y: beta/(k eta0) for TE, k/(eta0 beta) for TM."""
    k, bt = wavenumber(f), beta(mode, a, b, f)
    return bt / (k * ETA0) if mode[0] == "TE" else k / (ETA0 * bt)


def norm(mode, a, b):
    """Normalization constant N of Pi (TM) or Gamma (TE), JINST eq. 5.4 / CONVENTIONS.md section 4."""
    kind, m, n = mode
    s = np.pi * np.sqrt(m * m * b / a + n * n * a / b)
    return (np.sqrt(_eps(m) * _eps(n)) if kind == "TE" else 2.0) / s


def wall_impedances(Z):
    """Dict of the four wall impedances (ohm) from a scalar or a partial dict (missing walls: 0, i.e. PEC)."""
    if isinstance(Z, dict):
        bad = set(Z) - set(WALLS)
        if bad:
            raise ValueError(f"unknown walls {bad}; use {WALLS}")
        return {w: complex(Z.get(w, 0.0)) for w in WALLS}
    return {w: complex(Z) for w in WALLS}


# ============================================================================== item 1
def surface_resistance(f, sigma):
    """R_s = sqrt(omega mu0 / (2 sigma)) (ohm)."""
    return np.sqrt(2 * np.pi * f * cst.mu_0 / (2 * sigma))


def surface_impedance(f, sigma):
    """Z_s = (1 + i) R_s for exp(+i omega t) (CONVENTIONS.md section 1)."""
    return (1 + 1j) * surface_resistance(f, sigma)


# ============================================================================== item 2: wall integrals
def _cu(mode, a, b):
    """u-wall amplitude: e_p . n_hat on the wall u' = 0 equals c_u sin(n pi v'/b) (times (-1)^m on u' = a, with
    the outward normal of that wall); c_u = N k_m (TM), -N k_n (TE)."""
    kind, m, n = mode
    N = norm(mode, a, b)
    return N * m * np.pi / a if kind == "TM" else -N * n * np.pi / b


def _cv(mode, a, b):
    """v-wall amplitude: |e_p . n_hat| on the v-walls equals c_v sin(m pi u'/a) (sign (-1) on v' = 0 and (-1)^n on
    v' = b, which cancel in products); c_v = N k_n (TM), N k_m (TE)."""
    kind, m, n = mode
    N = norm(mode, a, b)
    return N * n * np.pi / b if kind == "TM" else N * m * np.pi / a


def _zu(Zw, mp, mq):
    return Zw["u-"] + (-1) ** (mp + mq) * Zw["u+"]


def _zv(Zw, np_, nq):
    return Zw["v-"] + (-1) ** (np_ + nq) * Zw["v+"]


def wall_A(p, q, a, b, Z):
    """A_pq = sum_walls Z_w  int (e_p . n_hat)(e_q . n_hat) dl   (ohm/m; e has units 1/m).

    Closed form (note, eq. A):
      u-walls (need n_p = n_q = n >= 1):  (b/2) c_u(p) c_u(q) [Z_u- + (-1)^(m_p+m_q) Z_u+]
      v-walls (need m_p = m_q = m >= 1):  (a/2) c_v(p) c_v(q) [Z_v- + (-1)^(n_p+n_q) Z_v+]
    """
    Zw = wall_impedances(Z)
    (kp, mp, np_), (kq, mq, nq) = p, q
    A = 0j
    if np_ == nq and np_ >= 1:
        A += 0.5 * b * _cu(p, a, b) * _cu(q, a, b) * _zu(Zw, mp, mq)
    if mp == mq and mp >= 1:
        A += 0.5 * a * _cv(p, a, b) * _cv(q, a, b) * _zv(Zw, np_, nq)
    return A


def wall_B(p, q, a, b, Z):
    """B_pq = sum_walls Z_w int g_p g_q dl (ohm/m^5), g = div(z x e) = -chi^2 Gamma (TE), 0 (TM).

    Closed form (TE-TE only):
      u-walls (n_p = n_q = n):  chi_p^2 chi_q^2 N_p N_q (b/eps_n) [Z_u- + (-1)^(m_p+m_q) Z_u+]
      v-walls (m_p = m_q = m):  chi_p^2 chi_q^2 N_p N_q (a/eps_m) [Z_v- + (-1)^(n_p+n_q) Z_v+]
    """
    Zw = wall_impedances(Z)
    (kp, mp, np_), (kq, mq, nq) = p, q
    if kp != "TE" or kq != "TE":
        return 0j
    c = cutoff(p, a, b) ** 2 * cutoff(q, a, b) ** 2 * norm(p, a, b) * norm(q, a, b)
    B = 0j
    if np_ == nq:
        B += c * b / _eps(np_) * _zu(Zw, mp, mq)
    if mp == mq:
        B += c * a / _eps(mp) * _zv(Zw, np_, nq)
    return B


def _lam(A, B, p, q, a, b, f, sign):
    k = wavenumber(f)
    s = np.sqrt(admittance(p, a, b, f)) * np.sqrt(admittance(q, a, b, f))
    return 0.5 * (sign * s * A + B / ((k * ETA0) ** 2 * s))


def lambda_pp(p, q, a, b, f, Z):
    """Lambda^{++}_pq = Lambda^{--}_pq = (1/2) [ sqrtY_p sqrtY_q A_pq + B_pq / ((k eta0)^2 sqrtY_p sqrtY_q) ]  (1/m)."""
    return _lam(wall_A(p, q, a, b, Z), wall_B(p, q, a, b, Z), p, q, a, b, f, +1)


def lambda_pm(p, q, a, b, f, Z):
    """Lambda^{+-}_pq = Lambda^{-+}_pq = (1/2) [ -sqrtY_p sqrtY_q A_pq + B_pq / ((k eta0)^2 sqrtY_p sqrtY_q) ]  (1/m)."""
    return _lam(wall_A(p, q, a, b, Z), wall_B(p, q, a, b, Z), p, q, a, b, f, -1)


def lambda_full(modes, a, b, f, Z):
    """Lambda_full = [[L++, L+-], [-L-+, -L--]] (2N x 2N), so that dy/dz = (D + kappa K - Lambda_full) y with
    y = [A+; A-] and D = diag(-i beta, +i beta) as in wrmodes.transfer_matrix.coupled_mode_matrices."""
    N = len(modes)
    Lpp = np.array([[lambda_pp(p, q, a, b, f, Z) for q in modes] for p in modes])
    Lpm = np.array([[lambda_pm(p, q, a, b, f, Z) for q in modes] for p in modes])
    return np.block([[Lpp, Lpm], [-Lpm, -Lpp]])


def dV_dI_boundary(p, q, a, b, f, Z):
    """Boundary terms of the T0 transmission-line equations (note, eq. VI):
        dV_p/dz = -i (beta_p/Y_p) I_p - sum_q A_pq I_q,
        dI_p/dz = -i beta_p Y_p V_p - sum_q B_pq V_q / (k eta0)^2.
    Returns (A_pq, B_pq / (k eta0)^2)."""
    k = wavenumber(f)
    return wall_A(p, q, a, b, Z), wall_B(p, q, a, b, Z) / (k * ETA0) ** 2


# ============================================================================== item 3: textbook constants
def alpha_te_marcuvitz(m, n, a, b, f, Rs):
    """Attenuation constant (Np/m) of TE_mn (H_mn), Marcuvitz 1951, sec. 2.2, eq. (10) (p. 60), with zeta = eta0,
    Marcuvitz's characteristic resistance = R_s, x = a, y = b, epsilon_l = 1 (l = 0), 2 (l > 0)."""
    lam_ratio2 = (cutoff(("TE", m, n), a, b) / wavenumber(f)) ** 2
    s = np.sqrt(1 - lam_ratio2)
    em, en = _eps(m), _eps(n)
    t1 = (en * m * m * b / a + em * n * n) / (m * m * b / a + n * n * a / b) * s
    t2 = (en + em * b / a) * lam_ratio2 / s
    return Rs / (ETA0 * b) * (t1 + t2)


def alpha_tm_marcuvitz(m, n, a, b, f, Rs):
    """Attenuation constant (Np/m) of TM_mn (E_mn), Marcuvitz 1951, sec. 2.2, eq. (4) (p. 57)."""
    lam_ratio2 = (cutoff(("TM", m, n), a, b) / wavenumber(f)) ** 2
    return 2 * Rs / (ETA0 * a) * (m * m + n * n * a ** 3 / b ** 3) / (m * m + n * n * a * a / (b * b)) \
        / np.sqrt(1 - lam_ratio2)


# ============================================================================== item 4: bends (weight u)
def _S1(mp, mq, a):
    """int_0^a (u' - a/2) sin(mp pi u'/a) sin(mq pi u'/a) du'."""
    if (mp + mq) % 2 == 0:
        return 0.0
    return -(a / np.pi) ** 2 * 4 * mp * mq / (mp * mp - mq * mq) ** 2


def _C1(mp, mq, a):
    """int_0^a (u' - a/2) cos(mp pi u'/a) cos(mq pi u'/a) du'."""
    if (mp + mq) % 2 == 0:
        return 0.0
    return -(a / np.pi) ** 2 * 2 * (mp * mp + mq * mq) / (mp * mp - mq * mq) ** 2


def wall_A1(p, q, a, b, Z):
    """A1_pq = sum_walls Z_w int u (e_p . n_hat)(e_q . n_hat) dl  (first moment, for h = 1 + kappa u)."""
    Zw = wall_impedances(Z)
    (kp, mp, np_), (kq, mq, nq) = p, q
    A = 0j
    if np_ == nq and np_ >= 1:
        A += 0.5 * b * _cu(p, a, b) * _cu(q, a, b) * 0.5 * a * ((-1) ** (mp + mq) * Zw["u+"] - Zw["u-"])
    if mp >= 1 and mq >= 1:
        A += _cv(p, a, b) * _cv(q, a, b) * _zv(Zw, np_, nq) * _S1(mp, mq, a)
    return A


def wall_B1(p, q, a, b, Z):
    """B1_pq = sum_walls Z_w int u g_p g_q dl (TE-TE only)."""
    Zw = wall_impedances(Z)
    (kp, mp, np_), (kq, mq, nq) = p, q
    if kp != "TE" or kq != "TE":
        return 0j
    c = cutoff(p, a, b) ** 2 * cutoff(q, a, b) ** 2 * norm(p, a, b) * norm(q, a, b)
    B = 0j
    if np_ == nq:
        B += c * b / _eps(np_) * 0.5 * a * ((-1) ** (mp + mq) * Zw["u+"] - Zw["u-"])
    B += c * _zv(Zw, np_, nq) * _C1(mp, mq, a)
    return B


def lambda1_pp(p, q, a, b, f, Z):
    return _lam(wall_A1(p, q, a, b, Z), wall_B1(p, q, a, b, Z), p, q, a, b, f, +1)


def lambda1_pm(p, q, a, b, f, Z):
    return _lam(wall_A1(p, q, a, b, Z), wall_B1(p, q, a, b, Z), p, q, a, b, f, -1)


def lambda1_full(modes, a, b, f, Z):
    """First-moment loss matrix: in a bend, dy/dz = (D + kappa (K - Lambda1_full) - Lambda_full) y."""
    Lpp = np.array([[lambda1_pp(p, q, a, b, f, Z) for q in modes] for p in modes])
    Lpm = np.array([[lambda1_pm(p, q, a, b, f, Z) for q in modes] for p in modes])
    return np.block([[Lpp, Lpm], [-Lpm, -Lpp]])


# ============================================================================== item 5: impedance walls at u = +-a/2
def lsu_dispersion(kx, kind, a, k, Zm, Zp):
    """Dispersion function of the LS^u modes of a guide with impedance walls Z_u- = Zm, Z_u+ = Zp (other walls PEC).

    LSE^u (E_u = 0, kind 'LSE'): with p = Z/(k eta0),  (1 + p- p+ kx^2) sin(kx a) - i (p- + p+) kx cos(kx a) = 0;
    LSM^u (H_u = 0, kind 'LSM'): with q = k Z/eta0,    (kx^2 + q- q+) sin(kx a) - i (q- + q+) kx cos(kx a) = 0.
    """
    if kind == "LSE":
        pm, pp = Zm / (k * ETA0), Zp / (k * ETA0)
        return (1 + pm * pp * kx * kx) * np.sin(kx * a) - 1j * (pm + pp) * kx * np.cos(kx * a)
    qm, qp = k * Zm / ETA0, k * Zp / ETA0
    return (kx * kx + qm * qp) * np.sin(kx * a) - 1j * (qm + qp) * kx * np.cos(kx * a)


def _newton(fun, x0, tol=1e-13, maxit=60):
    x = complex(x0)
    for _ in range(maxit):
        h = 1e-7 * max(1.0, abs(x))
        d = (fun(x + h) - fun(x - h)) / (2 * h)
        dx = fun(x) / d
        x -= dx
        if abs(dx) < tol * max(1.0, abs(x)):
            return x
    raise RuntimeError("Newton did not converge")


def lsu_kx_roots(kind, m_list, a, k, Zm, Zp, steps=200):
    """Roots kx of lsu_dispersion continued from the PEC values (LSE: kx = m pi/a, m >= 1, Dirichlet;
    LSM: kx = m pi/a, m >= 0, Neumann) by continuation in t, Z -> t Z, t = 0 ... 1."""
    out = []
    for m in m_list:
        ts = np.linspace(0, 1, steps + 1)[1:]
        x = m * np.pi / a
        if m == 0:   # LSM^u_0n (TE_0n): kx^2 ~ i (q- + q+) / a for small Z
            x = np.sqrt(1j * k * (Zm + Zp) / ETA0 * ts[0] / a + 0j)
        for t in ts:
            x = _newton(lambda z: lsu_dispersion(z, kind, a, k, t * Zm, t * Zp) / (1 if m else z), x)
        out.append(x)
    return np.array(out)


def _gl(n, lo, hi):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def impedance_guide_modes(n, a, b, f, Zm, Zp, M, quad=200):
    """Modes of family n of the guide with impedance walls at u = -+a/2 (Zm, Zp) and PEC walls at v = +-b/2.

    Returns a list of dicts (kind, kx, gamma, X(u') callable for the u-dependence, normalization c) such that
    the transverse fields, normalized with (1/2) int (e x h).z ds = 1, are
      LSE^u:  psi = X(u') cos(n pi v'/b):  e = c * (0, i gamma X cos),
              h = c/(i omega mu0) * ((k^2 - kx^2) X cos, -k_n X' sin)
      LSM^u:  psi = Y(u') sin(n pi v'/b):  e = c/(i omega eps0) * ((k^2 - kx^2) Y sin, k_n Y' cos),
              h = c * (0, -i gamma Y sin)
    with X = p- kx cos(kx u') + i sin(kx u') (LSE, p = Z/(k eta0)) and Y = kx cos(kx u') + i q- sin(kx u')
    (LSM, q = k Z / eta0). gamma = sqrt(k^2 - kx^2 - (n pi/b)^2) with Im gamma <= 0.
    """
    k = wavenumber(f)
    kn = n * np.pi / b
    wmu, weps = k * ETA0, k / ETA0
    u, w = _gl(quad, 0.0, a)
    modes = []
    kinds = [("LSE", range(1, M + 1))] + ([("LSM", range(0, M + 1))] if n >= 1 else [])
    for kind, ms in kinds:
        roots = lsu_kx_roots(kind, list(ms), a, k, Zm, Zp)
        for m0, kx in zip(ms, roots):
            g2 = k * k - kx * kx - kn * kn
            gam = np.sqrt(g2 + 0j)
            if gam.imag > 0:
                gam = -gam
            if kind == "LSE":
                pm = Zm / (k * ETA0)
                X = (lambda kx_, pm_: (lambda x: pm_ * kx_ * np.cos(kx_ * x) + 1j * np.sin(kx_ * x)))(kx, pm)
                dX = (lambda kx_, pm_: (lambda x: -pm_ * kx_ * kx_ * np.sin(kx_ * x) + 1j * kx_ * np.cos(kx_ * x)))(kx, pm)
                # (1/2) int (e x h).z = (1/2) * c^2 * (-i gamma)(k^2-kx^2)/(i omega mu) * int X^2 * int cos^2
                Ivv = b / _eps(n)
                P = 0.5 * (-gam * (k * k - kx * kx) / wmu) * np.sum(w * X(u) ** 2) * Ivv
            else:
                qm = k * Zm / ETA0
                X = (lambda kx_, qm_: (lambda x: kx_ * np.cos(kx_ * x) + 1j * qm_ * np.sin(kx_ * x)))(kx, qm)
                dX = (lambda kx_, qm_: (lambda x: -kx_ * kx_ * np.sin(kx_ * x) + 1j * qm_ * kx_ * np.cos(kx_ * x)))(kx, qm)
                Ivv = b / 2
                P = 0.5 * (-gam * (k * k - kx * kx) / weps) * np.sum(w * X(u) ** 2) * Ivv
            c = 1 / np.sqrt(P + 0j)
            modes.append(dict(kind=kind, m0=m0, kx=kx, gamma=gam, X=X, dX=dX, c=c))
    return modes


def _pec_e_on_line(mode, a, b, u):
    """u-dependence of the PEC mode field: e_u = Eu * cos(k_m u') sin(k_n v'), e_v = Ev * sin(k_m u') cos(k_n v')."""
    kind, m, n = mode
    N = norm(mode, a, b)
    km, kn = m * np.pi / a, n * np.pi / b
    if kind == "TM":
        return N * km, N * kn
    return -N * kn, N * km


def lossy_wall_junction(pec_modes, imp_modes, n, a, b, f, quad=200):
    """Mode-matching matrix R_pi = (1/2) int (e_i x h_p).z ds (PEC mode p, impedance-guide mode i)."""
    k = wavenumber(f)
    kn = n * np.pi / b
    u, w = _gl(quad, 0.0, a)
    R = np.zeros((len(pec_modes), len(imp_modes)), dtype=complex)
    for ip, p in enumerate(pec_modes):
        Eu, Ev = _pec_e_on_line(p, a, b, u)
        km = p[1] * np.pi / a
        Yp = admittance(p, a, b, f)
        for ii, md in enumerate(imp_modes):
            X, c, kx, gam = md["X"], md["c"], md["kx"], md["gamma"]
            if md["kind"] == "LSE":
                # e_i = c (0, i gamma X(u') cos(kn v'))
                val = c * 1j * gam * np.sum(w * Ev * np.sin(km * u) * X(u)) * (b / _eps(n))
            else:
                # e_i = c/(i omega eps0) ((k^2 - kx^2) Y sin(kn v'), kn Y' cos(kn v')), n >= 1
                val = c / (1j * k / ETA0) * ((k * k - kx * kx) * Eu * np.sum(w * np.cos(km * u) * X(u))
                                             + kn * Ev * np.sum(w * np.sin(km * u) * md["dX"](u))) * (b / 2)
            R[ip, ii] = 0.5 * np.sqrt(2 * Yp) * val
    return R


def junction_gsm(R):
    """GSM of a junction from the mode-matching matrix R (a+ + a- = R (c+ + c-), R^T (a+ - a-) = c+ - c-):
    S11 = (I + R R^T)^-1 (R R^T - I), S12 = 2 (I + R R^T)^-1 R, S21 = S12^T, S22 = I - R^T S12."""
    N1 = R.shape[0]
    P = R @ R.T
    inv = np.linalg.inv(np.eye(N1) + P)
    S11 = inv @ (P - np.eye(N1))
    S12 = 2 * inv @ R
    S21 = S12.T
    S22 = np.eye(R.shape[1]) - R.T @ S12
    return S11, S12, S21, S22


def star(A, B):
    """Redheffer star product of two GSMs given as 4-tuples (S11, S12, S21, S22)."""
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = B
    I = np.eye(A22.shape[0])
    X = np.linalg.inv(I - B11 @ A22)
    Y = np.linalg.inv(I - A22 @ B11)
    return (A11 + A12 @ X @ B11 @ A21, A12 @ X @ B12, B21 @ Y @ A21, B22 + B21 @ Y @ A22 @ B12)


def butterweck_eq21(m, a, f, Zw):
    """Butterweck 1968, eq. (21): alpha_m = 2 pi^2 m^2 Re(Zw) / (beta_m a^3 k Z0)  (Np/m), H_m0 modes, lossy side walls."""
    k = wavenumber(f)
    bm = np.sqrt(k * k - (m * np.pi / a) ** 2)
    return 2 * np.pi ** 2 * m * m * np.real(Zw) / (bm * a ** 3 * k * ETA0)


# ============================================================================== item 6: resistive sheets at v = v_j
# A sheet of resistance R (ohm/square) at v = v0 (v' = s = v0 + b/2) carries J = E_tan / R:
#   n x (H(v0+) - H(v0-)) = E_tan / R, E_tan continuous (Butterweck eq. 2).
# The loaded guide (sheets parallel to the walls v = -+b/2, uniform in u and z) has exact LS modes w.r.t. v:
#   LSM^v (H_v = 0, Butterweck EH^(2)):  psi_e = sin(k_m u') P(v'),  P'' = -ky^2 P,  P'(0) = P'(b) = 0,
#        at a sheet: P' continuous, P(s+) - P(s-) = -i eta0/(k R) P'(s);
#   LSE^v (E_v = 0, Butterweck EH^(1)):  psi_h = cos(k_m u') P(v'),  P(0) = P(b) = 0,
#        at a sheet: P continuous, P'(s+) - P'(s-) = +i k eta0 / R P(s);
# gamma^2 = k^2 - (m pi/a)^2 - ky^2.  For one sheet at s the dispersion functions are (note, eq. S1-S2)
#   LSM^v:  sin(ky b) + i (eta0 ky / (k R)) sin(ky s) sin(ky (b - s)) = 0      (s = b/2: Butterweck eq. 9)
#   LSE^v:  ky sin(ky b) + i (k eta0 / R) sin(ky s) sin(ky (b - s)) = 0        (s = b/2: Butterweck eq. 8)
def _sheet_state(ky2, kind, b, sheets, k, v_eval=None):
    """Propagate (P, P') from v' = 0 to b through the sheets [(s_j, R_j)], s_j in (0, b) measured from v = -b/2.
    Returns the end state, and, if v_eval is given (sorted, not on a sheet), the state at those points."""
    ky = np.sqrt(ky2 + 0j)

    def M(d):
        c = np.cos(ky * d)
        sk = d if ky == 0 else np.sin(ky * d) / ky
        return np.array([[c, sk], [-ky2 * sk, c]])

    x = np.array([1.0 + 0j, 0.0]) if kind == "LSM" else np.array([0.0 + 0j, 1.0])
    pos = 0.0
    out = []
    pts = [] if v_eval is None else list(v_eval)
    i = 0
    for s, R in sorted(sheets) + [(b, None)]:
        while i < len(pts) and pts[i] < s:
            out.append(M(pts[i] - pos) @ x)
            i += 1
        x = M(s - pos) @ x
        pos = s
        if R is not None:
            G = 0.0 if np.isinf(R) else 1.0 / R
            if kind == "LSM":
                x = np.array([x[0] - 1j * ETA0 * G / k * x[1], x[1]])
            else:
                x = np.array([x[0], x[1] + 1j * k * ETA0 * G * x[0]])
    return x, (np.array(out) if v_eval is not None else None)


def sheet_dispersion(ky2, kind, b, sheets, k):
    """P'(b) (LSM^v) or P(b) (LSE^v) for the start state P = 1, P' = 0 (LSM) or P = 0, P' = 1 (LSE) at v' = 0;
    an entire function of ky^2 whose zeros are the modes. The LSM^v root ky = 0 (the TE_m0 modes, untouched by the
    sheets) is removed by dividing by ky^2."""
    end, _ = _sheet_state(ky2, kind, b, sheets, k)
    if kind == "LSM":
        return end[1] / ky2 if ky2 != 0 else -b
    return end[0]


def sheet_ky2_roots(kind, n_list, b, sheets, k, steps=200):
    """ky^2 of the sheet-loaded modes continued from the empty-guide values (n pi/b)^2 by scaling all sheet
    conductances 1/R from 0 to their values."""
    out = []
    for n in n_list:
        x = (n * np.pi / b) ** 2 + 0j
        for t in np.linspace(0, 1, steps + 1)[1:]:
            sh = [(s, (np.inf if (np.isinf(R) or t == 0) else R / t)) for s, R in sheets]
            x = _newton(lambda z: sheet_dispersion(z, kind, b, sh, k), x)
        out.append(x)
    return np.array(out)


def sheet_modes(m, a, b, sheets, f, Nn, quad=64):
    """Modes of family m (fixed u-dependence) of the guide loaded with resistive sheets [(v0_j, R_j)] (v0 measured
    from the axis, -b/2 < v0 < b/2). Returns dicts with kind, n0 (empty-guide index), ky2, gamma, normalization c and
    the profile on the quadrature points. Fields (with psi as in the module comment, normalized
    (1/2) int (e x h).z = 1):
      LSM^v: e = c/(i omega eps0) (k_m cos(k_m u') P', (k^2 - ky^2) sin(k_m u') P),  h = c (i gamma sin(k_m u') P, 0)
      LSE^v: e = c (-i gamma cos(k_m u') P, 0),  h = c/(i omega mu0) (-k_m sin(k_m u') P', (k^2 - ky^2) cos(k_m u') P)
    """
    k = wavenumber(f)
    km = m * np.pi / a
    sh = [(v0 + b / 2, R) for v0, R in sheets]
    edges = [0.0] + sorted(s for s, _ in sh) + [b]
    vq, wq = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        x, w = _gl(quad, lo, hi)
        vq.append(x)
        wq.append(w)
    vq, wq = np.concatenate(vq), np.concatenate(wq)
    out = []
    kinds = ([("LSM", range(0, Nn + 1))] if m >= 1 else []) + [("LSE", range(1, Nn + 1))]
    for kind, ns in kinds:
        nlist = list(ns)
        roots = [0j if (kind == "LSM" and n == 0) else None for n in nlist]
        todo = [n for n, r in zip(nlist, roots) if r is None]
        rr = iter(sheet_ky2_roots(kind, todo, b, sh, k)) if todo else iter([])
        roots = [r if r is not None else next(rr) for r in roots]
        for n0, ky2 in zip(nlist, roots):
            _, st = _sheet_state(ky2, kind, b, sh, k, vq)
            P, dP = st[:, 0], st[:, 1]
            g2 = k * k - km * km - ky2
            gam = np.sqrt(g2 + 0j)
            gam = -gam if gam.imag > 0 else gam
            IP2 = np.sum(wq * P * P)
            if kind == "LSM":
                Nrm = -0.5 * gam * (k * k - ky2) / (k / ETA0) * (a / 2) * IP2
            else:
                Nrm = -0.5 * gam * (k * k - ky2) / (k * ETA0) * (a / _eps(m)) * IP2
            out.append(dict(kind=kind, n0=n0, ky2=ky2, gamma=gam, c=1 / np.sqrt(Nrm + 0j), P=P, dP=dP, v=vq, w=wq))
    return out


def sheet_junction(pec_modes, smodes, m, a, b, f):
    """Mode-matching matrix R_pi = (1/2) sqrt(2 Y_p) int e_i . e_p ds between the PEC modes p (all with the same m)
    and the sheet-guide modes i of family m."""
    k = wavenumber(f)
    km = m * np.pi / a
    R = np.zeros((len(pec_modes), len(smodes)), dtype=complex)
    for ip, p in enumerate(pec_modes):
        assert p[1] == m
        Eu, Ev = _pec_e_on_line(p, a, b, None)
        kn = p[2] * np.pi / b
        Yp = admittance(p, a, b, f)
        for ii, md in enumerate(smodes):
            v, w, P, dP, c = md["v"], md["w"], md["P"], md["dP"], md["c"]
            sv, cv = np.sin(kn * v), np.cos(kn * v)
            if md["kind"] == "LSM":
                val = c / (1j * k / ETA0) * (a / 2) * (km * Eu * np.sum(w * sv * dP)
                                                         + (k * k - md["ky2"]) * Ev * np.sum(w * cv * P))
            else:
                val = c * (-1j * md["gamma"]) * Eu * (a / _eps(m)) * np.sum(w * sv * P)
            R[ip, ii] = 0.5 * np.sqrt(2 * Yp) * val
    return R


def sheet_filter_gsm(m, a, b, sheets, f, length, Nn):
    """GSM (4-tuple) of empty guide | sheet-loaded section of the given length | empty guide, family m, with the
    PEC modes TE_m0 (m >= 1), TE_mn and TM_mn (n = 1..Nn) in the order returned as the second value."""
    pec = ([("TE", m, 0)] if m >= 1 else []) + [("TE", m, n) for n in range(1, Nn + 1)] \
        + ([("TM", m, n) for n in range(1, Nn + 1)] if m >= 1 else [])
    sm = sheet_modes(m, a, b, sheets, f, Nn)
    J = junction_gsm(sheet_junction(pec, sm, m, a, b, f))
    Pm = np.diag(np.exp(-1j * np.array([md["gamma"] for md in sm]) * length))
    Z0 = np.zeros_like(Pm)
    return star(star(J, (Z0, Pm, Pm, Z0)), (J[3], J[2], J[1], J[0])), pec, sm
