"""Symbolic checks of the algebra of docs/theory/T1-wall-impedance.md (sympy).

Run:  PYTHONPATH=<worktree>/src python3 docs/theory/T1/check_sympy.py
Every check prints PASS/FAIL; the script exits with status 1 if any check fails.

S1  Leontovich condition and Z_s = (1+i) R_s for e^{+i omega t}, n pointing into the wall.
S2  The projected equations with the exact boundary terms (eqs. 3-4 of the note), for generic Maxwell fields
    that satisfy no wall condition at all (LSE^u and LSM^u fields with arbitrary k_u and phase).
S3  Leontovich substitution and conversion to power waves: Lambda^+-, Lambda_full.
S4  Closed forms of the wall integrals with symbolic mode numbers.
S5  Diagonal of Lambda^+ = Marcuvitz eqs. 2.2(4) and 2.2(10), symbolically.
S6  LS^u dispersion relation from the field ansatz + Leontovich (B6), its symmetric factorization, and the
    small-Z expansion (B5) which reproduces Butterweck eqs. (29) and (21).
S7  Resistive-sheet dispersion relations (B7) from the field ansatz + sheet condition; centred sheet gives
    Butterweck eqs. (8) and (9).
S8  Butterweck eq. (28): the coupling coefficient of the projected wave equation carries a factor nu/mu.
"""
import sys
import sympy as sp

FAILS = []


def check(name, expr_or_bool):
    if isinstance(expr_or_bool, (bool, sp.logic.boolalg.BooleanAtom)):
        ok = bool(expr_or_bool)
    else:
        ok = sp.simplify(expr_or_bool) == 0
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


I = sp.I
u, v, z = sp.symbols("u v z", real=True)            # here u, v are the primed coordinates u', v' in [0,a],[0,b]
a, b, k, eta = sp.symbols("a b k eta", positive=True)
wmu, weps = k * eta, k / eta                       # omega mu0 = k eta0, omega eps0 = k / eta0


def curl(F):
    Fu, Fv, Fz = F
    return sp.Matrix([sp.diff(Fz, v) - sp.diff(Fv, z), sp.diff(Fu, z) - sp.diff(Fz, u), sp.diff(Fv, u) - sp.diff(Fu, v)])


# ---------------------------------------------------------------------------------------------- S1
x = sp.symbols("x", real=True)
sigma, mu, w = sp.symbols("sigma mu omega", positive=True)
gam = sp.sqrt(I * w * mu * sigma)                  # e^{-gam x} into the conductor (x along n)
E0 = sp.symbols("E0")
E = sp.Matrix([0, E0 * sp.exp(-gam * x), 0])       # tangential E (y), wave travelling along +x = n
Hc = curl_x = sp.Matrix([0, 0, sp.diff(E[1], x)]) / (-I * w * mu)   # curl E = -i w mu H, only d/dx
nvec = sp.Matrix([1, 0, 0])
Zs = (I * w * mu / gam)
lhs = E.subs(x, 0)
rhs = Zs * Hc.subs(x, 0).cross(nvec)
check("S1a  E_tan = Z_s H_tan x n (n into the wall), Z_s = i omega mu / gamma", (lhs - rhs).applyfunc(sp.simplify) == sp.zeros(3, 1))
Rs = sp.sqrt(w * mu / (2 * sigma))
check("S1b  Z_s = (1+i) R_s", sp.simplify(sp.expand_complex(Zs - (1 + I) * Rs)))
Sn = sp.simplify(sp.re(sp.expand_complex((rhs.cross(Hc.subs(x, 0).conjugate()))[0])))
# 1/2 Re(E x H*) . n with E = Z H x n equals 1/2 Re(Z) |H_t|^2 >= 0
s1c = (Sn - sp.re(sp.expand_complex(Zs)) * sp.Abs(Hc.subs(x, 0)[2]) ** 2)
s1c_vals = [abs(sp.N(s1c.subs({E0: e0, w: w_, mu: m_, sigma: s_}), 30)) for e0, w_, m_, s_ in
            ((1 + I, 2, 3, 5), (sp.Rational(13, 10) - I / 5, 7, 1, 11), (3, 1, 2, 1))]
check("S1c  power flows into the wall: Re(E x H*).n = Re(Z_s)|H_t|^2 (three numeric cases, 30 digits)",
      bool(max(s1c_vals) < 1e-25))


# ---------------------------------------------------------------------------------------------- S2
def ideal_fields(kind, m, n):
    """Ideal mode e (2-vector), d = div e, g = div(z x e), chi^2 (fields of wrmodes.modes, primed coords)."""
    km, kn = m * sp.pi / a, n * sp.pi / b
    chi2 = km ** 2 + kn ** 2
    s = sp.pi * sp.sqrt(m ** 2 * b / a + n ** 2 * a / b)
    if kind == "TM":
        P = 2 / s * sp.sin(km * u) * sp.sin(kn * v)
        e = sp.Matrix([sp.diff(P, u), sp.diff(P, v)])
    else:
        epsm, epsn = (1 if m == 0 else 2), (1 if n == 0 else 2)
        G = sp.sqrt(epsm * epsn) / s * sp.cos(km * u) * sp.cos(kn * v)
        e = sp.Matrix([sp.diff(G, v), -sp.diff(G, u)])
    d = sp.diff(e[0], u) + sp.diff(e[1], v)
    g = sp.diff(-e[1], u) + sp.diff(e[0], v)        # z x e = (-e_v, e_u)
    return e, sp.simplify(d), sp.simplify(g), chi2


def integrate2(f):
    return sp.integrate(sp.integrate(sp.expand(f), (u, 0, a), conds="none"), (v, 0, b), conds="none")


ku, ph, gz = sp.symbols("k_u phi gamma")

# Generic fields: the LS^u solutions of Maxwell's equations with arbitrary k_u and phase (no wall condition).
n_f = 1
kn_f = n_f * sp.pi / b
X = sp.cos(ku * u + ph)
prop = sp.exp(-I * gz * z)
gam2 = k ** 2 - kn_f ** 2 - ku ** 2                # gamma^2 of a Maxwell solution
fields = {}
# LSE^u: E = -curl(u_hat Phi), H = -curl E / (i w mu)
Phi = X * sp.cos(kn_f * v) * prop
E_lse = -curl(sp.Matrix([Phi, 0, 0]))
H_lse = -curl(E_lse) / (I * wmu)
fields["LSE^u"] = (E_lse, H_lse)
# LSM^u: H = curl(u_hat Psi), E = curl H / (i w eps)
Psi = X * sp.sin(kn_f * v) * prop
H_lsm = curl(sp.Matrix([Psi, 0, 0]))
E_lsm = curl(H_lsm) / (I * weps)
fields["LSM^u"] = (E_lsm, H_lsm)

for label, (Ef, Hf) in fields.items():
    # Maxwell residual (curl E + i w mu H must vanish once gamma^2 = k^2 - k_n^2 - k_u^2)
    res = (curl(Ef) + I * wmu * Hf).applyfunc(lambda t: sp.simplify((t / prop).subs(gz, sp.sqrt(gam2))))
    check(f"S2.0 {label} test field satisfies Maxwell's equations", res == sp.zeros(3, 1))

walls = [  # (fixed coordinate, value, normal (nu, nv), tangent (tu, tv) = z x n, line variable, range)
    (u, a, (1, 0), (0, 1), v, b),
    (u, 0, (-1, 0), (0, -1), v, b),
    (v, b, (0, 1), (-1, 0), u, a),
    (v, 0, (0, -1), (1, 0), u, a),
]

for label, (Ef, Hf) in fields.items():
    for kind, m, n in (("TE", 1, 1), ("TM", 1, 1), ("TE", 2, 1), ("TM", 3, 1), ("TE", 0, 1)):
        e, d, g, chi2 = ideal_fields(kind, m, n)
        zxe = sp.Matrix([-e[1], e[0]])
        Et, Ht = sp.Matrix([Ef[0], Ef[1]]), sp.Matrix([Hf[0], Hf[1]])
        Vp = integrate2((Et.T * e)[0])
        Ip = integrate2((Ht.T * zxe)[0])
        bt_V = 0
        bt_I = 0
        for (cvar, cval, nn, tt, lvar, L) in walls:
            en = (e[0] * nn[0] + e[1] * nn[1]).subs(cvar, cval)
            Ez = Ef[2].subs(cvar, cval)
            Etan = (Ef[0] * tt[0] + Ef[1] * tt[1]).subs(cvar, cval)
            bt_V += sp.integrate(sp.expand(Ez * en), (lvar, 0, L), conds="none")
            bt_I += sp.integrate(sp.expand(g.subs(cvar, cval) * Etan), (lvar, 0, L), conds="none")
        tm = 1 if kind == "TM" else 0
        te = 1 - tm
        rV = sp.diff(Vp, z) - (-I * wmu * Ip + tm * I * chi2 / weps * Ip + bt_V)
        rI = sp.diff(Ip, z) - (-I * weps * Vp + te * I * chi2 / wmu * Vp - I / wmu * bt_I)
        rV = sp.simplify((rV / prop).subs(gz, sp.sqrt(gam2)))
        rI = sp.simplify((rI / prop).subs(gz, sp.sqrt(gam2)))
        check(f"S2   {label} field, {kind}{m},{n}: dV/dz = -i w mu I + [TM] i chi^2/(w eps) I + oint E_z (e.n)", rV)
        check(f"S2   {label} field, {kind}{m},{n}: dI/dz = -i w eps V + [TE] i chi^2/(w mu) V - i/(w mu) oint g E_t", rI)

# ---------------------------------------------------------------------------------------------- S3
# Leontovich on the wall: E_z = -Z H_t (t = z x n),  E_t = Z H_z.  Check from E_tan = Z H x n.
Z = sp.symbols("Z")
Hn, Ht_, Hz_ = sp.symbols("H_n H_t H_z")
# local orthonormal frame (n, t, z) is right-handed: n x t = z
Hvec = sp.Matrix([Hn, Ht_, Hz_])
Etan = Z * Hvec.cross(sp.Matrix([1, 0, 0]))
check("S3a  Leontovich: E_t = Z H_z and E_z = -Z H_t (t = z x n)", sp.simplify(Etan[1] - Z * Hz_) == 0 and sp.simplify(Etan[2] + Z * Ht_) == 0)
# H_t on the wall from the series: H_t = sum I_q (z x e_q).t = sum I_q e_q.n ; H_z = (i/(w mu)) div(E x z) = -(i/(w mu)) sum V_q g_q
# wall terms:  oint E_z (e_p.n) = -sum_q I_q oint Z (e_p.n)(e_q.n)  => -N I
#             -(i/(w mu)) oint g_p E_t = -(i/(w mu)) oint g_p Z (-(i/(w mu)) sum V_q g_q) => -G V,  G = oint Z g g / (w mu)^2
gp, gq = sp.symbols("g_p g_q")
Vq = sp.symbols("V_q")
termI = -I / wmu * gp * Z * (-I / wmu * Vq * gq)
check("S3b  wall term of dI/dz equals -(Z g_p g_q/(k eta)^2) V_q", sp.simplify(termI + Z * gp * gq / (k * eta) ** 2 * Vq))
# power waves (2 modes p, q suffice; the algebra is per pair)
Yp, Yq = sp.symbols("Y_p Y_q", positive=True)
Npq, Gpq = sp.symbols("N_pq G_pq")
apl, ami = sp.symbols("a^+_q a^-_q")
Vq_ = (apl + ami) * sp.sqrt(2 / Yq)
Iq_ = (apl - ami) * sp.sqrt(2 * Yq)
dV = -Npq * Iq_
dI = -Gpq * Vq_
dap = sp.Rational(1, 2) * (dV / sp.sqrt(2 / Yp) + dI / sp.sqrt(2 * Yp))
dam = sp.Rational(1, 2) * (dV / sp.sqrt(2 / Yp) - dI / sp.sqrt(2 * Yp))
Lp = sp.Rational(1, 2) * (sp.sqrt(Yp) * sp.sqrt(Yq) * Npq + Gpq / (sp.sqrt(Yp) * sp.sqrt(Yq)))
Lm = sp.Rational(1, 2) * (-sp.sqrt(Yp) * sp.sqrt(Yq) * Npq + Gpq / (sp.sqrt(Yp) * sp.sqrt(Yq)))
check("S3c  da+/dz = -Lambda^+ a+ - Lambda^- a-", sp.expand(dap - (-Lp * apl - Lm * ami)))
check("S3d  da-/dz = +Lambda^- a+ + Lambda^+ a-", sp.expand(dam - (Lm * apl + Lp * ami)))
# A transmission-line check: single TE mode, gamma^2 = Z'Y' with Z' = i beta/Y + N, Y' = i beta Y + G.
beta = sp.symbols("beta", positive=True)
eps_ = sp.symbols("epsilon")
gexact = sp.sqrt((I * beta / Yp + eps_ * Npq) * (I * beta * Yp + eps_ * Gpq))
ser = sp.series(gexact, eps_, 0, 2).removeO()
check("S3e  first-order propagation constant i beta + Lambda^+_pp (transmission-line limit)",
      sp.simplify(ser - (I * beta + eps_ * Lp.subs(Yq, Yp))))

# ---------------------------------------------------------------------------------------------- S4
mm, mq, nn_, nq = sp.symbols("m m_q n n_q", integer=True, positive=True)
xx = sp.symbols("x", real=True)
L = sp.symbols("L", positive=True)
ss1 = sp.integrate((xx - L / 2) * sp.sin(mm * sp.pi * xx / L) * sp.sin(mq * sp.pi * xx / L), (xx, 0, L))
cc1 = sp.integrate((xx - L / 2) * sp.cos(mm * sp.pi * xx / L) * sp.cos(mq * sp.pi * xx / L), (xx, 0, L))
# evaluate generically for m != m_q
def generic_branch(pw):
    """The branch of a Piecewise valid for m != m_q."""
    if not isinstance(pw, sp.Piecewise):
        return pw
    for expr, cond in pw.args:
        if cond != True and cond.has(sp.Ne) or (isinstance(cond, sp.And) and any(isinstance(c, sp.Ne) for c in cond.args)):
            return expr
    return pw.args[0][0]


def trig_int(e):
    return sp.simplify(e.subs({sp.sin(sp.pi * mm): 0, sp.sin(sp.pi * mq): 0, sp.cos(sp.pi * mm): (-1) ** mm,
                               sp.cos(sp.pi * mq): (-1) ** mq}))


ss1g = trig_int(generic_branch(ss1))
cc1g = trig_int(generic_branch(cc1))
ok = True
for p_ in range(0, 6):
    for q_ in range(0, 6):
        if p_ == q_:
            continue
        odd = (p_ + q_) % 2
        s_cf = (-(L / sp.pi) ** 2 * 4 * p_ * q_ / (p_ ** 2 - q_ ** 2) ** 2) if odd else 0
        c_cf = (-(L / sp.pi) ** 2 * 2 * (p_ ** 2 + q_ ** 2) / (p_ ** 2 - q_ ** 2) ** 2) if odd else 0
        s_ex = sp.integrate((xx - L / 2) * sp.sin(p_ * sp.pi * xx / L) * sp.sin(q_ * sp.pi * xx / L), (xx, 0, L))
        c_ex = sp.integrate((xx - L / 2) * sp.cos(p_ * sp.pi * xx / L) * sp.cos(q_ * sp.pi * xx / L), (xx, 0, L))
        ok &= sp.simplify(s_ex - s_cf) == 0 and sp.simplify(c_ex - c_cf) == 0
    s_eq = sp.integrate((xx - L / 2) * sp.sin(p_ * sp.pi * xx / L) ** 2, (xx, 0, L))
    c_eq = sp.integrate((xx - L / 2) * sp.cos(p_ * sp.pi * xx / L) ** 2, (xx, 0, L))
    ok &= sp.simplify(s_eq) == 0 and sp.simplify(c_eq) == 0
check("S4a  u-weighted broad-wall integrals S1, C1 (closed forms, 0 <= m, m' <= 5)", ok)
# Symbolic (any integers) versions of the generic expressions
s_cf_g = -(L / sp.pi) ** 2 * (1 - (-1) ** (mm + mq)) * 2 * mm * mq / (mm ** 2 - mq ** 2) ** 2
c_cf_g = -(L / sp.pi) ** 2 * (1 - (-1) ** (mm + mq)) * (mm ** 2 + mq ** 2) / (mm ** 2 - mq ** 2) ** 2
okg = all(sp.simplify((ss1g - s_cf_g).subs({mm: p_, mq: q_})) == 0 and sp.simplify((cc1g - c_cf_g).subs({mm: p_, mq: q_})) == 0
          for p_ in range(1, 9) for q_ in range(1, 9) if p_ != q_)
check("S4b  sympy's generic (symbolic m != m') S1 and C1 equal the closed forms (evaluated for 1 <= m, m' <= 8)", okg)
# Side-wall normal fields of the ideal modes (A coefficients) and broad-wall ones (B coefficients)
ok = True
for kind, m, n in (("TE", 1, 1), ("TM", 2, 3), ("TE", 3, 0), ("TE", 0, 2), ("TM", 1, 1)):
    e, d, g, chi2 = ideal_fields(kind, m, n)
    km, kn = m * sp.pi / a, n * sp.pi / b
    chi = sp.sqrt(chi2)
    c = (2 if kind == "TM" else sp.sqrt((1 if m == 0 else 2) * (1 if n == 0 else 2))) / (sp.sqrt(a * b) * chi)
    A = c * km if kind == "TM" else -c * kn
    B = c * kn if kind == "TM" else c * km
    ok &= sp.simplify(e[0].subs(u, 0) - A * sp.sin(kn * v)) == 0
    ok &= sp.simplify(e[0].subs(u, a) - (-1) ** m * A * sp.sin(kn * v)) == 0
    ok &= sp.simplify(e[1].subs(v, 0) - B * sp.sin(km * u)) == 0
    ok &= sp.simplify(e[1].subs(v, b) - (-1) ** n * B * sp.sin(km * u)) == 0
    if kind == "TE":
        Gam = c * sp.cos(km * u) * sp.cos(kn * v)
        ok &= sp.simplify(g - (-chi2 * Gam)) == 0
check("S4d  wall values of e.n (A, B coefficients) and g = -chi^2 Gamma", ok)

# ---------------------------------------------------------------------------------------------- S5
m_, n_ = sp.symbols("m n", positive=True)
Rs_ = sp.symbols("R_s", positive=True)
km, kn = m_ * sp.pi / a, n_ * sp.pi / b
chi2 = km ** 2 + kn ** 2
bet = sp.sqrt(k ** 2 - chi2)
YE, YM = bet / (k * eta), k / (eta * bet)


def diag_lambda(kind, epsm, epsn):
    cE = sp.sqrt(epsm * epsn) / (sp.sqrt(a * b) * sp.sqrt(chi2))
    cM = 2 / (sp.sqrt(a * b) * sp.sqrt(chi2))
    if kind == "TM":
        A, B = cM * km, cM * kn
        N = 2 * A ** 2 * b / 2 + 2 * B ** 2 * a / 2
        return sp.Rational(1, 2) * YM * N * Rs_
    A, B = -cE * kn, cE * km
    Nside = 2 * A ** 2 * (b / 2 if epsn == 2 else 0)
    Nbroad = 2 * B ** 2 * (a / 2 if epsm == 2 else 0)
    Gside = 2 * chi2 ** 2 * cE ** 2 * (b / 2 if epsn == 2 else b)
    Gbroad = 2 * chi2 ** 2 * cE ** 2 * (a / 2 if epsm == 2 else a)
    return sp.Rational(1, 2) * (YE * (Nside + Nbroad) + (Gside + Gbroad) / (k * eta) ** 2 / YE) * Rs_


x2 = chi2 / k ** 2
marc_TM = 2 * Rs_ / (eta * a) * (m_ ** 2 + n_ ** 2 * a ** 3 / b ** 3) / (m_ ** 2 + n_ ** 2 * a ** 2 / b ** 2) / sp.sqrt(1 - x2)


def marc_TE(epsm, epsn):
    return Rs_ / (eta * b) * ((epsn * m_ ** 2 * b / a + epsm * n_ ** 2) / (m_ ** 2 * b / a + n_ ** 2 * a / b) * sp.sqrt(1 - x2)
                             + (epsn + epsm * b / a) * x2 / sp.sqrt(1 - x2))


def same(e1, e2):
    num = sp.N((e1 - e2).subs({a: sp.Rational(2286, 100000), b: sp.Rational(1016, 100000), k: 1500, eta: 377, Rs_: 1,
                               m_: sp.Rational(3), n_: sp.Rational(2)}), 30)
    return abs(num) < 1e-25 and sp.simplify(sp.radsimp(e1 - e2)) == 0


check("S5a  Re Lambda^+_pp (TM_mn) = Marcuvitz 2.2(4)", same(diag_lambda("TM", 2, 2), marc_TM))
check("S5b  Re Lambda^+_pp (TE_mn, m, n >= 1) = Marcuvitz 2.2(10)", same(diag_lambda("TE", 2, 2), marc_TE(2, 2)))
lam_m0 = diag_lambda("TE", 2, 1).subs(n_, 0)
check("S5c  TE_m0 = Marcuvitz 2.2(10) with n = 0", sp.simplify(lam_m0 - marc_TE(2, 1).subs(n_, 0)))
lam_10 = lam_m0.subs(m_, 1)
fc2 = (sp.pi / a) ** 2 / k ** 2
check("S5d  TE10 = card formula Rs/(eta b sqrt(1-(fc/f)^2)) [1 + (2b/a)(fc/f)^2]",
      sp.simplify(lam_10 - Rs_ / (eta * b * sp.sqrt(1 - fc2)) * (1 + 2 * b / a * fc2)))
lam_0n = diag_lambda("TE", 1, 2).subs(m_, 0)
check("S5e  TE_0n = Marcuvitz 2.2(10) with m = 0", sp.simplify(lam_0n - marc_TE(1, 2).subs(m_, 0)))

# ---------------------------------------------------------------------------------------------- S6
uu = sp.symbols("uu", real=True)        # centred u
Zp, Zm = sp.symbols("Z_p Z_m")
A_, B_ = sp.symbols("A B")
Xc = A_ * sp.cos(ku * uu) + B_ * sp.sin(ku * uu)
# Wall conditions derived from the fields (S2 field ansatz) with Leontovich, recomputed here in centred coords:
# LSE^u: E_v = -dPhi/dz,  H_z = (1/(i w mu)) d2Phi/(du dz)   ->  at u=+a/2: E_v = Z H_z ; at u=-a/2: E_v = -Z H_z
Phi_c = Xc * sp.exp(-I * gz * z)
Ev = -sp.diff(Phi_c, z)
Hz = sp.diff(sp.diff(Phi_c, uu), z) / (I * wmu)
bc_p = (Ev - Zp * Hz).subs(uu, a / 2)
bc_m = (Ev + Zm * Hz).subs(uu, -a / 2)
M = sp.Matrix([[sp.diff(bc_p, A_), sp.diff(bc_p, B_)], [sp.diff(bc_m, A_), sp.diff(bc_m, B_)]]) / prop
detE = sp.simplify(M.det())
pp, pm = eta / Zp, eta / Zm
F = (ku ** 2 + k ** 2 * pp * pm) * sp.sin(ku * a) - I * k * ku * (pp + pm) * sp.cos(ku * a)
def const_ratio(num, den, extra=()):
    vals = {a: sp.Rational(23, 1000), k: sp.Rational(1500), eta: sp.Rational(377), Zp: sp.Rational(30) + 7 * I,
            Zm: sp.Rational(11), gz: sp.Rational(1300), b: sp.Rational(1, 100)}
    r = [sp.N((num / den).subs(vals).subs(ku, kv_), 30) for kv_ in (sp.Rational(91), sp.Rational(207), sp.Rational(333) + 5 * I)]
    return abs(r[0] - r[1]) < 1e-20 * abs(r[0]) and abs(r[0] - r[2]) < 1e-20 * abs(r[0]) and abs(r[0]) > 0


check("S6a  LSE^u: det of the Leontovich conditions is proportional to F(k_u) with p = eta/Z", const_ratio(detE, F))
# LSM^u: Psi = X sin(k_n v), H = curl(u Psi): H_z = -dPsi/dv, E_v = (1/(i w eps)) d2Psi/(du dv)
Psi_c = Xc * sp.sin(kn_f * v) * sp.exp(-I * gz * z)
Hz2 = -sp.diff(Psi_c, v)
Ev2 = sp.diff(sp.diff(Psi_c, uu), v) / (I * weps)
bc_p = ((Ev2 - Zp * Hz2) / sp.cos(kn_f * v)).subs(uu, a / 2)
bc_m = ((Ev2 + Zm * Hz2) / sp.cos(kn_f * v)).subs(uu, -a / 2)
M = sp.Matrix([[sp.diff(bc_p, A_), sp.diff(bc_p, B_)], [sp.diff(bc_m, A_), sp.diff(bc_m, B_)]]) / prop
detM = sp.simplify(M.det())
F2 = (ku ** 2 + k ** 2 * (Zp / eta) * (Zm / eta)) * sp.sin(ku * a) - I * k * ku * (Zp / eta + Zm / eta) * sp.cos(ku * a)
check("S6b  LSM^u: det proportional to F(k_u) with p = Z/eta", const_ratio(detM, F2))
# symmetric factorization
p_ = sp.symbols("p")
Fs = (ku ** 2 + k ** 2 * p_ ** 2) * sp.sin(ku * a) - 2 * I * k * ku * p_ * sp.cos(ku * a)
fact = 2 * (ku * sp.sin(ku * a / 2) - I * k * p_ * sp.cos(ku * a / 2)) * (ku * sp.cos(ku * a / 2) + I * k * p_ * sp.sin(ku * a / 2))
check("S6c  symmetric walls: F = 2 (k_u sin - i k p cos)(k_u cos + i k p sin) (even/odd modes)",
      sp.simplify(sp.expand(sp.expand_trig(Fs - fact))))
# small-Z expansion of the LSE^u root: k_u = m pi/a + eps
t = sp.symbols("t")       # Z = t * Z0
mI = sp.symbols("m", integer=True, positive=True)
epsk = sp.symbols("epsilon")
Fz = ((t * Zp / eta) * (t * Zm / eta) * ku ** 2 + k ** 2) * sp.sin(ku * a) - I * k * ku * (t * Zm / eta + t * Zp / eta) * sp.cos(ku * a)
Fz = Fz.subs(ku, mI * sp.pi / a + epsk)
sol = sp.solve(sp.series(Fz.subs(epsk, t * sp.Symbol("c1")), t, 0, 2).removeO().subs(sp.cos(sp.pi * mI), (-1) ** mI).subs(sp.sin(sp.pi * mI), 0), sp.Symbol("c1"))
c1 = sp.simplify(sol[0])
check("S6d  LSE^u first order: k_u = m pi/a + i (m pi/a)(Z+ + Z-)/(eta k a)", sp.simplify(c1 - I * (mI * sp.pi / a) * (Zp + Zm) / (eta * k * a)))
# Butterweck eq. (29): gamma_B^2 = k_u^2 - k^2 with e^{-gamma_B z} convention, symmetric real Z_w, m = 1
Zw = sp.symbols("Z_w", positive=True)
kuB = sp.pi / a + I * (sp.pi / a) * 2 * Zw / (eta * k * a)
g2 = sp.expand(kuB ** 2 - k ** 2)
g2_first = sp.series(g2.subs(Zw, t * Zw), t, 0, 2).removeO().subs(t, 1)
butterweck29 = sp.pi ** 2 / a ** 2 - k ** 2 - sp.pi ** 2 / a ** 2 * 4 * Zw / (I * eta * k * a)
check("S6e  low-impedance limit reproduces Butterweck eq. (29)", sp.simplify(g2_first - butterweck29))
# alpha from (29) for |Z| << Z0: alpha = Re sqrt(gamma_B^2) ~ (pi/a)^2 4 Z/(eta k a) / (2 beta) -> eq. (21)
betm = sp.symbols("beta_m", positive=True)
alpha21 = 2 * sp.pi ** 2 * mI ** 2 * Zw / (betm * a ** 3 * k * eta)
dgam2 = (mI * sp.pi / a) ** 2 * 4 * I * Zw / (eta * k * a)      # i * (...) for general m from S6d
check("S6f  alpha = Re(d gamma_B^2)/(2 j beta)... = Butterweck eq. (21) for all m",
      sp.simplify(sp.im(dgam2) / (2 * betm) - alpha21))

# ---------------------------------------------------------------------------------------------- S7
xi, kv, d1, d2 = sp.symbols("xi k_v d_1 d_2", positive=True)
vv = sp.symbols("vv", real=True)       # v' measured from the lower wall
km_s = sp.symbols("k_m", positive=True)
A1, A2 = sp.symbols("A_1 A_2")
R = xi * eta
# LSM^v (H_v = 0): H = curl(v_hat Psi), Psi = sin(k_m u) Y(v) e^{-i g z};  E = curl H/(i w eps)
Ylo = A1 * sp.cos(kv * vv)
Yhi = A2 * sp.cos(kv * (d1 + d2 - vv))


def fields_lsm(Y):
    P = sp.sin(km_s * u) * Y * sp.exp(-I * gz * z)
    H = sp.Matrix([-sp.diff(P, z), 0, sp.diff(P, u)])
    Ecur = sp.Matrix([sp.diff(H[2], vv) - sp.diff(H[1], z), sp.diff(H[0], z) - sp.diff(H[2], u), sp.diff(H[1], u) - sp.diff(H[0], vv)]) / (I * weps)
    return Ecur, H


def fields_lse(Y):
    P = sp.cos(km_s * u) * Y * sp.exp(-I * gz * z)
    Ecur = -sp.Matrix([-sp.diff(P, z), 0, sp.diff(P, u)])
    H = -sp.Matrix([sp.diff(Ecur[2], vv) - sp.diff(Ecur[1], z), sp.diff(Ecur[0], z) - sp.diff(Ecur[2], u), sp.diff(Ecur[1], u) - sp.diff(Ecur[0], vv)]) / (I * wmu)
    return Ecur, H


def sheet_conditions(fields_fn, Ylo, Yhi):
    Elo, Hlo = fields_fn(Ylo)
    Ehi, Hhi = fields_fn(Yhi)
    eqs = []
    for comp in (0, 2):     # tangential E continuous (u and z components)
        eqs.append(((Elo[comp] - Ehi[comp]).subs(vv, d1)))
    # H(-0) - H(+0) = (1/R) v_hat x E_tan, v_hat x E = (E_z, 0, -E_u)
    eqs.append(((Hlo[0] - Hhi[0]) - Elo[2] / R).subs(vv, d1))
    eqs.append(((Hlo[2] - Hhi[2]) + Elo[0] / R).subs(vv, d1))
    return [sp.simplify(e / sp.exp(-I * gz * z)) for e in eqs]


def det_of(eqs):
    rows = []
    for e in eqs:
        e = sp.expand(e)
        # strip the u-dependence (each equation is a multiple of sin or cos of k_m u)
        for fu in (sp.sin(km_s * u), sp.cos(km_s * u)):
            cf = e.coeff(fu)
            if cf != 0:
                e = cf
                break
        rows.append([sp.diff(e, A1), sp.diff(e, A2)])
    Mx = sp.Matrix(rows)
    # all 2x2 minors must vanish; return the nonzero ones
    minors = set()
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            mnr = sp.simplify(Mx[[i, j], :].det())
            if mnr != 0:
                minors.add(mnr)
    return minors


Fl = xi * k * sp.sin(kv * (d1 + d2)) + I * kv * sp.sin(kv * d1) * sp.sin(kv * d2)
mins = det_of(sheet_conditions(fields_lsm, Ylo, Yhi))
# robust numeric version of S7a/S7b: every minor vanishes at a root of the closed form and not elsewhere
import mpmath as mp
mp.mp.dps = 50


def minors_vanish_at_roots(mins, Fclosed, kind):
    vals = {d1: sp.Rational(3, 10), d2: sp.Rational(7, 10), xi: sp.Rational(1, 7), k: sp.Rational(9, 2), km_s: 1, eta: 377,
            gz: sp.Rational(13, 10)}
    f = sp.lambdify(kv, Fclosed.subs(vals), "mpmath")
    root = mp.findroot(f, mp.mpc(2.8, 0.4))
    ok = True
    for mn in mins:
        g = sp.lambdify(kv, mn.subs(vals), "mpmath")
        ok &= abs(g(root)) < 1e-35 * max(1, abs(g(root + 0.3)))
        ok &= abs(g(root + 0.3)) > 1e-8
    return ok


check("S7a  LSM^v sheet minors vanish exactly at the roots of the closed form (B7)", minors_vanish_at_roots(mins, Fl, "LSM"))
Ylo = A1 * sp.sin(kv * vv)
Yhi = A2 * sp.sin(kv * (d1 + d2 - vv))
Fe = xi * kv * sp.sin(kv * (d1 + d2)) + I * k * sp.sin(kv * d1) * sp.sin(kv * d2)
mins_e = det_of(sheet_conditions(fields_lse, Ylo, Yhi))
check("S7b  LSE^v (E_v = 0) sheet minors vanish exactly at the roots of the closed form (B7)", minors_vanish_at_roots(mins_e, Fe, "LSE"))
# centred sheet -> Butterweck (8), (9)
bb = sp.symbols("b_", positive=True)
F9 = Fl.subs({d1: bb / 2, d2: bb / 2})
B9 = 2 * xi + I * (kv / k) * sp.tan(kv * bb / 2)
check("S7c  centred LSM^v: closed form = 2 k sin(k_v b/2) cos(k_v b/2) x Butterweck (9)",
      sp.simplify(sp.expand_trig(F9 - k * sp.sin(kv * bb / 2) * sp.cos(kv * bb / 2) * B9)))
F8 = Fe.subs({d1: bb / 2, d2: bb / 2})
B8 = 2 * xi + I * (k / kv) * sp.tan(kv * bb / 2)
check("S7d  centred LSE^v: closed form = k_v sin cos x Butterweck (8)",
      sp.simplify(sp.expand_trig(F8 - kv * sp.sin(kv * bb / 2) * sp.cos(kv * bb / 2) * B8)))
# R -> infinity: LS^v of the empty guide (k_v = n pi / b) ; R -> 0: the sheet is a short (two half guides)
check("S7e  xi -> oo: LSM^v reduces to sin(k_v b) = 0", sp.simplify(sp.limit(Fl / xi, xi, sp.oo) - k * sp.sin(kv * (d1 + d2))))

# ---------------------------------------------------------------------------------------------- S8
# Butterweck eqs. (24)-(27): E_y = sum C_nu cos(nu pi x/a), x in [0, a/2] (symmetric half), wall at x = a/2.
nu, muI = sp.symbols("nu mu", integer=True, positive=True)
Zw_, kk = sp.symbols("Z_w k", positive=True)
# E_y(a/2) = -(Z_w/(j Z0 k)) dE/dx(a/2), dE/dx = sum D_mu sin(mu pi x/a), D_mu = -(a/(mu pi)) (d2/dz2 + k^2) C_mu
Wmu = sp.symbols("W_mu")    # stands for (d^2/dz^2 + k^2) C_mu
Ey_wall_term = -(Zw_ / (I * eta * kk)) * (-(a / (muI * sp.pi)) * Wmu) * sp.sin(muI * sp.pi / 2)
eq24_wall = (nu * sp.pi / a) * Ey_wall_term * sp.sin(nu * sp.pi / 2)
coef = sp.simplify(eq24_wall / ((Zw_ / (I * eta * kk)) * Wmu * sp.sin(muI * sp.pi / 2) * sp.sin(nu * sp.pi / 2)))
check("S8   the coupling term of Butterweck eq. (28) carries the factor nu/mu (missing in the printed eq.)",
      sp.simplify(coef - nu / muI))

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
