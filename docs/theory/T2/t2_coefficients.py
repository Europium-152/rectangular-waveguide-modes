"""Coupling coefficients of T2 (tapers with curvature), one function per coefficient.

This is the machine-checkable version of the boxed formulas of docs/theory/T2-taper-coupling.md. It is a theory
check module, not library code: it depends only on numpy, scipy and wrmodes.modes (for Mode and the constants).

Geometry (CONVENTIONS.md sections 2-4). At each z the cross-section is the rectangle
    u in [u0 - a/2, u0 + a/2],  v in [v0 - b/2, v0 + b/2],
in the frame (u, v, z) whose z scale factor is h = 1 + kappa(z) u. a, b, u0, v0, kappa are functions of z; the
coefficients below take their local values and the derivatives da = a'(z), db = b'(z), du0 = u0'(z), dv0 = v0'(z).
The local modes are the modes of CONVENTIONS.md section 4 with the local a, b, centred on (u0, v0).

Equations (V-I form, regular at cutoff), for the transverse fields E_t = sum V_q e_q, H_t = sum I_q z x e_q:

    dV/dz = -i Z I + N^T V,      dI/dz = -i Y V - N I,

with N_pq = overlap_derivative(p, q, ...) = int_{S(z)} e_p . d/dz e_q ds   (domain: the cross-section at z),
Z_pq = impedance(p, q, ...), Y_pq = admittance(p, q, ...).

Power waves (CONVENTIONS.md section 5): V = (a+ + a-) sqrt(2/Y_p), I = (a+ - a-) sqrt(2 Y_p), and
    da+/dz = K_ff a+ + K_fb a-,   da-/dz = K_bf a+ + K_bb a-   (functions K_ff, K_fb, K_bf, K_bb).

Mode arguments p, q: wrmodes.modes.Mode, ('TE', m, n) tuples or names 'TE1,0'. Units: SI (m, Hz, 1/m).
"""
import numpy as np
import scipy.constants as cst

from wrmodes.modes import Mode, ETA0, propagation_constant, wave_admittance

MU0 = cst.mu_0
EPS0 = 1.0 / (MU0 * cst.c ** 2)          # consistent with ETA0 = mu0 c (CONVENTIONS.md section 1)


def _eps(j):
    return 1.0 if j == 0 else 2.0


def _mode(p):
    return Mode.parse(p)


# ----------------------------------------------------------------------------------------- polarization vector
def t_vector(p, a, b):
    """(t_u, t_v) of mode p: e_p = (t_u phi^c_m psi^s_n, t_v phi^s_m psi^c_n), with the orthonormal 1D functions
    phi^s_m = sqrt(2/a) sin(m pi xi/a), phi^c_m = sqrt(eps_m/a) cos(m pi xi/a) (xi = u - u0 + a/2) and the same in
    v (psi, side b, index n). TM: (k_m, k_n)/chi; TE: (-k_n, k_m)/chi; k_m = m pi/a, k_n = n pi/b."""
    p = _mode(p)
    km, kn = p.m * np.pi / a, p.n * np.pi / b
    chi = np.hypot(km, kn)
    return (km / chi, kn / chi) if p.kind == "TM" else (-kn / chi, km / chi)


def dtheta_da(p, a, b):
    """d theta/da at fixed b, theta = atan2(k_n, k_m): sin(theta) cos(theta)/a = k_m k_n/(a chi^2)."""
    p = _mode(p)
    km, kn = p.m * np.pi / a, p.n * np.pi / b
    return km * kn / (a * (km * km + kn * kn))


def dtheta_db(p, a, b):
    """d theta/db at fixed a: -k_m k_n/(b chi^2)."""
    p = _mode(p)
    km, kn = p.m * np.pi / a, p.n * np.pi / b
    return -km * kn / (b * (km * km + kn * kn))


# ----------------------------------------------------------------------------------------- 1D matrices
# All four are dimensionless; the 1D overlap derivative is (1/a) times them (side a) or (1/b) (side b).
def S_width(j, l):
    """a <phi^s_j, d/da phi^s_l> (symmetric widening, centre fixed): 2 j l/(j^2 - l^2) for j + l even, j != l."""
    if j == l or (j + l) % 2 == 1 or j == 0 or l == 0:
        return 0.0
    return 2.0 * j * l / (j * j - l * l)


def C_width(j, l):
    """a <phi^c_j, d/da phi^c_l>: sqrt(eps_j eps_l) l^2/(j^2 - l^2) for j + l even, j != l; -eps_j/2 for j = l."""
    if j == l:
        return -_eps(j) / 2.0
    if (j + l) % 2 == 1:
        return 0.0
    return np.sqrt(_eps(j) * _eps(l)) * l * l / (j * j - l * l)


def S_shift(j, l):
    """a <phi^s_j, d/du0 phi^s_l> (rigid shift of the side-a walls): -4 j l/(j^2 - l^2) for j + l odd."""
    if (j + l) % 2 == 0 or j == 0 or l == 0:
        return 0.0
    return -4.0 * j * l / (j * j - l * l)


def C_shift(j, l):
    """a <phi^c_j, d/du0 phi^c_l>: -2 sqrt(eps_j eps_l) l^2/(j^2 - l^2) for j + l odd."""
    if (j + l) % 2 == 0:
        return 0.0
    return -2.0 * np.sqrt(_eps(j) * _eps(l)) * l * l / (j * j - l * l)


def G_s(j, l, a):
    """<phi^s_j, u~ phi^s_l> (u~ = u - u0): -8 a j l/(pi^2 (j^2 - l^2)^2) for j + l odd, else 0."""
    if (j + l) % 2 == 0 or j == 0 or l == 0:
        return 0.0
    return -8.0 * a * j * l / (np.pi ** 2 * (j * j - l * l) ** 2)


def G_c(j, l, a):
    """<phi^c_j, u~ phi^c_l>: -2 a sqrt(eps_j eps_l) (j^2 + l^2)/(pi^2 (j^2 - l^2)^2) for j + l odd, else 0."""
    if (j + l) % 2 == 0:
        return 0.0
    return -2.0 * a * np.sqrt(_eps(j) * _eps(l)) * (j * j + l * l) / (np.pi ** 2 * (j * j - l * l) ** 2)


# ----------------------------------------------------------------------------------------- overlap derivative N
def N_a(p, q, a, b):
    """dN_pq/da': coefficient of a'(z) in N_pq = int_S(z) e_p . d_z e_q ds (symmetric width change of side a)."""
    p, q = _mode(p), _mode(q)
    if p.n != q.n:
        return 0.0
    tp, tq = t_vector(p, a, b), t_vector(q, a, b)
    val = (tp[0] * tq[0] * C_width(p.m, q.m) + tp[1] * tq[1] * S_width(p.m, q.m)) / a
    if p.m == q.m and p.kind != q.kind:            # rotation of the TE/TM polarization vectors
        val += dtheta_da(q, a, b) * (1.0 if p.kind == "TE" else -1.0)
    return val


def N_b(p, q, a, b):
    """Coefficient of b'(z) in N_pq (symmetric height change of side b)."""
    p, q = _mode(p), _mode(q)
    if p.m != q.m:
        return 0.0
    tp, tq = t_vector(p, a, b), t_vector(q, a, b)
    val = (tp[0] * tq[0] * S_width(p.n, q.n) + tp[1] * tq[1] * C_width(p.n, q.n)) / b
    if p.n == q.n and p.kind != q.kind:
        val += dtheta_db(q, a, b) * (1.0 if p.kind == "TE" else -1.0)
    return val


def N_u(p, q, a, b):
    """Coefficient of u0'(z) in N_pq (offset of the centre along u)."""
    p, q = _mode(p), _mode(q)
    if p.n != q.n:
        return 0.0
    tp, tq = t_vector(p, a, b), t_vector(q, a, b)
    return (tp[0] * tq[0] * C_shift(p.m, q.m) + tp[1] * tq[1] * S_shift(p.m, q.m)) / a


def N_v(p, q, a, b):
    """Coefficient of v0'(z) in N_pq (offset of the centre along v)."""
    p, q = _mode(p), _mode(q)
    if p.m != q.m:
        return 0.0
    tp, tq = t_vector(p, a, b), t_vector(q, a, b)
    return (tp[0] * tq[0] * S_shift(p.n, q.n) + tp[1] * tq[1] * C_shift(p.n, q.n)) / b


def overlap_derivative(p, q, a, b, da=0.0, db=0.0, du0=0.0, dv0=0.0):
    """N_pq = int_{S(z)} e_p(z) . (d/dz e_q)(z) ds, d/dz at fixed (u, v), over the cross-section S(z) at z (1/m)."""
    return da * N_a(p, q, a, b) + db * N_b(p, q, a, b) + du0 * N_u(p, q, a, b) + dv0 * N_v(p, q, a, b)


def wall_term(p, q, a, b, da=0.0, db=0.0, du0=0.0, dv0=0.0):
    """W_pq = oint w e_p,n e_q,n dl over the contour of S(z), w = outward normal velocity of the wall (dimensionless).
    Identity: N + N^T + W = 0 (z-derivative of the orthonormality over the moving domain)."""
    p, q = _mode(p), _mode(q)
    tp, tq = t_vector(p, a, b), t_vector(q, a, b)
    W = 0.0
    if p.n == q.n:          # walls u = u0 -+ a/2: e_n = e_u = t_u phi^c_m psi^s_n; w = -+(du0) + da/2
        par = (-1.0) ** (p.m + q.m)
        c = tp[0] * tq[0] * np.sqrt(_eps(p.m) * _eps(q.m)) / a
        W += c * (da / 2 * (1 + par) + du0 * (par - 1))
    if p.m == q.m:          # walls v = v0 -+ b/2: e_n = e_v = t_v phi^s_m psi^c_n
        par = (-1.0) ** (p.n + q.n)
        c = tp[1] * tq[1] * np.sqrt(_eps(p.n) * _eps(q.n)) / b
        W += c * (db / 2 * (1 + par) + dv0 * (par - 1))
    return W


# ----------------------------------------------------------------------------------------- curvature integrals
def U(p, q, a, b):
    """U_pq = int u~ e_p . e_q ds, u~ = u - u0 (m). Couples m + m' odd within a family."""
    p, q = _mode(p), _mode(q)
    if p.n != q.n:
        return 0.0
    tp, tq = t_vector(p, a, b), t_vector(q, a, b)
    return tp[0] * tq[0] * G_c(p.m, q.m, a) + tp[1] * tq[1] * G_s(p.m, q.m, a)


def _chi(p, a, b):
    return np.pi * np.hypot(p.m / a, p.n / b)


def Ud(p, q, a, b):
    """U^d_pq = int u~ d_p d_q ds, d = div e = -chi^2 Pi (TM), 0 (TE) (1/m^3)."""
    p, q = _mode(p), _mode(q)
    if p.kind != "TM" or q.kind != "TM" or p.n != q.n:
        return 0.0
    return _chi(p, a, b) * _chi(q, a, b) * G_s(p.m, q.m, a)


def Ug(p, q, a, b):
    """U^g_pq = int u~ g_p g_q ds, g = div(z x e) = -chi^2 Gamma (TE), 0 (TM) (1/m^3)."""
    p, q = _mode(p), _mode(q)
    if p.kind != "TE" or q.kind != "TE" or p.n != q.n:
        return 0.0
    return _chi(p, a, b) * _chi(q, a, b) * G_c(p.m, q.m, a)


# ----------------------------------------------------------------------------------------- V-I coefficients
def impedance(p, q, a, b, f, kappa=0.0, u0=0.0):
    """Z_pq = omega mu0 int h e_p.e_q - (1/(omega eps0)) int h d_p d_q, h = 1 + kappa (u0 + u~) (ohm/m).
    Regular at cutoff: Z_pp = h0 omega mu0 (TE), h0 beta^2/(omega eps0) (TM), h0 = 1 + kappa u0."""
    p, q = _mode(p), _mode(q)
    w = 2 * np.pi * f
    h0 = 1.0 + kappa * u0
    val = kappa * (w * MU0 * U(p, q, a, b) - Ud(p, q, a, b) / (w * EPS0))
    if p == q:
        val += h0 * (w * MU0 - (_chi(p, a, b) ** 2 / (w * EPS0) if p.kind == "TM" else 0.0))
    return val


def admittance(p, q, a, b, f, kappa=0.0, u0=0.0):
    """Y_pq = omega eps0 int h e_p.e_q - (1/(omega mu0)) int h g_p g_q (S/m).
    Regular at cutoff: Y_pp = h0 beta^2/(omega mu0) (TE), h0 omega eps0 (TM)."""
    p, q = _mode(p), _mode(q)
    w = 2 * np.pi * f
    h0 = 1.0 + kappa * u0
    val = kappa * (w * EPS0 * U(p, q, a, b) - Ug(p, q, a, b) / (w * MU0))
    if p == q:
        val += h0 * (w * EPS0 - (_chi(p, a, b) ** 2 / (w * MU0) if p.kind == "TE" else 0.0))
    return val


def vi_matrix(modes, a, b, f, da=0.0, db=0.0, du0=0.0, dv0=0.0, kappa=0.0, u0=0.0):
    """M (2N x 2N) of d[V; I]/dz = M [V; I]: M = [[N^T, -i Z], [-i Y, -N]]."""
    modes = [_mode(m) for m in modes]
    n = len(modes)
    N = np.array([[overlap_derivative(p, q, a, b, da, db, du0, dv0) for q in modes] for p in modes])
    Z = np.array([[impedance(p, q, a, b, f, kappa, u0) for q in modes] for p in modes])
    Y = np.array([[admittance(p, q, a, b, f, kappa, u0) for q in modes] for p in modes])
    M = np.zeros((2 * n, 2 * n), dtype=complex)
    M[:n, :n], M[:n, n:], M[n:, :n], M[n:, n:] = N.T, -1j * Z, -1j * Y, -N
    return M


# ----------------------------------------------------------------------------------------- power-wave form
def log_admittance_derivative(p, a, b, f, da=0.0, db=0.0):
    """Lambda_p = Y_p'/(2 Y_p) = -(chi_p^2)'/(4 beta_p^2) (TE), +(chi_p^2)'/(4 beta_p^2) (TM);
    (chi^2)' = -2 (k_m^2 a'/a + k_n^2 b'/b). Diverges at cutoff."""
    p = _mode(p)
    km, kn = p.m * np.pi / a, p.n * np.pi / b
    dchi2 = -2.0 * (km * km * da / a + kn * kn * db / b)
    beta = propagation_constant(p, a, b, f)
    s = -1.0 if p.kind == "TE" else 1.0
    return s * dchi2 / (4.0 * beta * beta)


def _sqY(p, a, b, f):
    return np.sqrt(complex(wave_admittance(p, a, b, f)))


def _PQ(p, q, a, b, f, da, db, du0, dv0):
    """(P_pq, Q_pq) = (sqrt(Y_p) N_qp / sqrt(Y_q), N_pq sqrt(Y_q) / sqrt(Y_p))."""
    rp, rq = _sqY(p, a, b, f), _sqY(q, a, b, f)
    Nqp = overlap_derivative(q, p, a, b, da, db, du0, dv0)
    Npq = overlap_derivative(p, q, a, b, da, db, du0, dv0)
    return rp * Nqp / rq, Npq * rq / rp


def _YZ(p, q, a, b, f, kappa, u0):
    """(script-Y_pq, script-Z_pq) = (Y_pq/(sqrt(Y_p) sqrt(Y_q)), sqrt(Y_p) Z_pq sqrt(Y_q))."""
    rp, rq = _sqY(p, a, b, f), _sqY(q, a, b, f)
    return admittance(p, q, a, b, f, kappa, u0) / (rp * rq), rp * impedance(p, q, a, b, f, kappa, u0) * rq


def K_ff(p, q, a, b, f, da=0.0, db=0.0, du0=0.0, dv0=0.0, kappa=0.0, u0=0.0):
    """Coefficient of a+_q in da+_p/dz: (P - Q)/2 - i (sY + sZ)/2. Uniform guide: -i beta_p delta_pq."""
    P, Q = _PQ(p, q, a, b, f, da, db, du0, dv0)
    sY, sZ = _YZ(p, q, a, b, f, kappa, u0)
    return 0.5 * (P - Q) - 0.5j * (sY + sZ)


def K_fb(p, q, a, b, f, da=0.0, db=0.0, du0=0.0, dv0=0.0, kappa=0.0, u0=0.0):
    """Coefficient of a-_q in da+_p/dz: Lambda_p delta_pq + (P + Q)/2 - i (sY - sZ)/2."""
    P, Q = _PQ(p, q, a, b, f, da, db, du0, dv0)
    sY, sZ = _YZ(p, q, a, b, f, kappa, u0)
    lam = log_admittance_derivative(p, a, b, f, da, db) if _mode(p) == _mode(q) else 0.0
    return lam + 0.5 * (P + Q) - 0.5j * (sY - sZ)


def K_bf(p, q, a, b, f, da=0.0, db=0.0, du0=0.0, dv0=0.0, kappa=0.0, u0=0.0):
    """Coefficient of a+_q in da-_p/dz: Lambda_p delta_pq + (P + Q)/2 + i (sY - sZ)/2."""
    P, Q = _PQ(p, q, a, b, f, da, db, du0, dv0)
    sY, sZ = _YZ(p, q, a, b, f, kappa, u0)
    lam = log_admittance_derivative(p, a, b, f, da, db) if _mode(p) == _mode(q) else 0.0
    return lam + 0.5 * (P + Q) + 0.5j * (sY - sZ)


def K_bb(p, q, a, b, f, da=0.0, db=0.0, du0=0.0, dv0=0.0, kappa=0.0, u0=0.0):
    """Coefficient of a-_q in da-_p/dz: (P - Q)/2 + i (sY + sZ)/2. Uniform guide: +i beta_p delta_pq."""
    P, Q = _PQ(p, q, a, b, f, da, db, du0, dv0)
    sY, sZ = _YZ(p, q, a, b, f, kappa, u0)
    return 0.5 * (P - Q) + 0.5j * (sY + sZ)


def power_wave_matrix(modes, a, b, f, **geo):
    """K (2N x 2N) of d[a+; a-]/dz = K [a+; a-]."""
    modes = [_mode(m) for m in modes]
    blocks = [[np.array([[fn(p, q, a, b, f, **geo) for q in modes] for p in modes]) for fn in row]
              for row in ((K_ff, K_fb), (K_bf, K_bb))]
    return np.block(blocks)


# ----------------------------------------------------------------------------------------- small-slope limit
def first_order_conversion(p_in, q_out, geometry, f, L, nz=2001, reflected=False):
    """Single-scattering amplitude of mode q_out produced by a unit forward mode p_in (q_out != p_in):

        a+_q(L) = int_0^L K_ff[q,p](z) exp(-i int_0^z beta_p - i int_z^L beta_q) dz          (transmitted)
        a-_q(0) = -int_0^L K_bf[q,p](z) exp(-i int_0^z (beta_p + beta_q)) dz                  (reflected)

    with the T2 coefficients without the diagonal terms. Valid while both modes propagate on [0, L] (no cutoff
    plane inside; see the T2 note for the turning-point correction). geometry(z) returns a dict with keys
    a, b, da, db, du0, dv0, kappa, u0. Trapezoidal rule on nz points.
    """
    z = np.linspace(0.0, L, nz)
    g = [geometry(zz) for zz in z]
    bp = np.array([propagation_constant(p_in, gg["a"], gg["b"], f) for gg in g])
    bq = np.array([propagation_constant(q_out, gg["a"], gg["b"], f) for gg in g])

    def cumtrapz(y):
        return np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(z))))

    Bp, Bq = cumtrapz(bp), cumtrapz(bq)
    geo = lambda gg: {k: gg[k] for k in ("da", "db", "du0", "dv0", "kappa", "u0")}
    if reflected:
        c = np.array([K_bf(q_out, p_in, gg["a"], gg["b"], f, **geo(gg)) for gg in g])
        integrand = -c * np.exp(-1j * (Bp + Bq))
    else:
        c = np.array([K_ff(q_out, p_in, gg["a"], gg["b"], f, **geo(gg)) for gg in g])
        integrand = c * np.exp(-1j * Bp - 1j * (Bq[-1] - Bq))
    return np.trapezoid(integrand, z)
