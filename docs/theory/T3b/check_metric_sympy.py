"""Symbolic checks of section 1-2 of the note (helical coordinates, exact twisted-guide equations).

Run:  cd <worktree> && PYTHONPATH=$PWD/src:$PWD/docs/theory/T3b python3 docs/theory/T3b/check_metric_sympy.py

 1. Metric of x = u cos(th) - v sin(th), y = u sin(th) + v cos(th), z = z, th = th(z): g_ij, det g = 1,
    g^ij = L L^T with L = [[I, tau w], [0, 1]], w = (v, -u).
 2. Covariant Maxwell: for test Cartesian fields, the coordinate curl of the covariant components equals
    sqrt(g) times the contravariant components of the Cartesian curl (so curl E = -i w mu0 g^ij H_j in helical
    coordinates, i.e. the medium eps_r = mu_r = g^ij).
 3. Covariant transverse components are the rotating-frame Cartesian components (E.u_hat, E.v_hat) and the
    contravariant z component is the Cartesian E_z.
 4. The exact transverse equation  d_z E_t = grad_t E^z + i w mu0 z x H_t - tau G E_t  (E^z contravariant)
    follows from the covariant Faraday law for arbitrary fields (no tau^2 term), with
    G E = z x E - (u d_v - v d_u) E = grad(w . E) + r div(E x z).
 5. PEC wall u = a/2: tangential covariant components E_v = E_z = 0 imply E^z = tau v E_u on the wall.
"""
import sympy as sp

u, v, z, x, y = sp.symbols("u v z x y", real=True)
th = sp.Function("theta")(z)
tau = sp.diff(th, z)
X = sp.Matrix([u * sp.cos(th) - v * sp.sin(th), u * sp.sin(th) + v * sp.cos(th), z])
q = [u, v, z]
J = X.jacobian(q)                                     # columns: dr/du, dr/dv, dr/dz
g_lo = sp.simplify(J.T * J)
g_up = sp.simplify(g_lo.inv())
detg = sp.simplify(g_lo.det())
w = sp.Matrix([v, -u])
Lm = sp.Matrix([[1, 0, tau * v], [0, 1, -tau * u], [0, 0, 1]])
print("1. g_ij =", g_lo.tolist())
print("   det g =", detg)
print("   g^ij =", sp.simplify(g_up).tolist())
assert detg == 1
assert sp.simplify(g_up - Lm * Lm.T) == sp.zeros(3, 3)

# 2. covariant curl identity on test fields
grad_q = sp.simplify(J.inv())                          # rows: grad u, grad v, grad z (Cartesian components)
tests = [sp.Matrix([sp.sin(2 * y) * z, x * x * sp.cos(z), x * y]),
         sp.Matrix([y * z ** 2, sp.exp(x) * z, sp.sin(x + y)])]
th_test = {th: sp.Rational(3, 7) * z ** 2 + z / 5}        # a non-uniform twist profile
for F in tests:
    curlF = sp.Matrix([sp.diff(F[2], y) - sp.diff(F[1], z), sp.diff(F[0], z) - sp.diff(F[2], x),
                       sp.diff(F[1], x) - sp.diff(F[0], y)])
    sub = {x: X[0], y: X[1]}
    Fq = F.subs(sub)
    E_cov = (J.T * Fq)                                 # E_i = E . dr/dq^i
    curl_cov = sp.Matrix([sp.diff(E_cov[2], v) - sp.diff(E_cov[1], z), sp.diff(E_cov[0], z) - sp.diff(E_cov[2], u),
                          sp.diff(E_cov[1], u) - sp.diff(E_cov[0], v)])
    curl_contra = grad_q * curlF.subs(sub)             # (curl F) . grad q^i
    diff = sp.simplify((curl_cov - curl_contra).subs(th_test).doit())
    assert diff == sp.zeros(3, 1), diff
print("2. coordinate curl of covariant components = contravariant Cartesian curl (sqrt g = 1): ok")

# 3. meaning of the components
uh = sp.Matrix([sp.cos(th), sp.sin(th), 0])
vh = sp.Matrix([-sp.sin(th), sp.cos(th), 0])
assert sp.simplify(J[:, 0] - uh) == sp.zeros(3, 1) and sp.simplify(J[:, 1] - vh) == sp.zeros(3, 1)
assert sp.simplify(grad_q[2, :] - sp.Matrix([[0, 0, 1]])) == sp.zeros(1, 3)
print("3. dr/du = u_hat, dr/dv = v_hat (covariant E_t = rotating-frame components); grad z = z_hat (E^z = E.z_hat): ok")

# 4. exact transverse equation from the covariant Faraday law, arbitrary fields
Eu, Ev, Ez, Hu, Hv, Hz = [sp.Function(n)(u, v, z) for n in ("Eu", "Ev", "Ez", "Hu", "Hv", "Hz")]
om_mu = sp.Symbol("omega_mu0")
t = sp.Symbol("tau")                                   # tau as a free symbol (the identity is pointwise in z)
gup = Lm.subs(tau, t) * Lm.subs(tau, t).T
E = sp.Matrix([Eu, Ev, Ez])
H = sp.Matrix([Hu, Hv, Hz])
curlE = sp.Matrix([sp.diff(Ez, v) - sp.diff(Ev, z), sp.diff(Eu, z) - sp.diff(Ez, u), sp.diff(Ev, u) - sp.diff(Eu, v)])
Fres = curlE + sp.I * om_mu * gup * H                  # Faraday residual (zero for a solution)
E_up = gup * E                                         # contravariant E
Hz_up = (gup * H)[2]
Et = sp.Matrix([Eu, Ev])
zx = lambda A: sp.Matrix([-A[1], A[0]])
r = sp.Matrix([u, v])
GE = (sp.Matrix([sp.diff(w.dot(Et), u), sp.diff(w.dot(Et), v)])
      + r * (sp.diff(Ev, u) - sp.diff(Eu, v)))           # grad(w.E) + r div(E x z)
GE2 = zx(Et) - (u * sp.diff(Et, v) - v * sp.diff(Et, u))
assert sp.simplify(GE - GE2) == sp.zeros(2, 1)
lhs = (sp.diff(Et, z) - sp.Matrix([sp.diff(E_up[2], u), sp.diff(E_up[2], v)])
       - sp.I * om_mu * zx(sp.Matrix([Hu, Hv])) + t * GE)
# lhs must be a combination of the Faraday residuals: z x F_t and r F_z
comb = sp.expand(lhs + zx(sp.Matrix([Fres[0], Fres[1]])) - t * r * Fres[2])
assert sp.simplify(comb) == sp.zeros(2, 1), comb
print("4. d_z E_t = grad E^z + i w mu0 z x H_t - tau G E_t  is exact (= -z x F_t + tau r F_z): ok")
print("   with G E = grad(w.E) + r div(E x z) = z x E - (u d_v - v d_u) E: ok")
# the magnetic equation has the same form (duality E -> H, mu0 -> -eps0), checked the same way:
om_eps = sp.Symbol("omega_eps0")
curlH = sp.Matrix([sp.diff(Hz, v) - sp.diff(Hv, z), sp.diff(Hu, z) - sp.diff(Hz, u), sp.diff(Hv, u) - sp.diff(Hu, v)])
Ares = curlH - sp.I * om_eps * gup * E
H_up = gup * H
Ht = sp.Matrix([Hu, Hv])
GH = sp.Matrix([sp.diff(w.dot(Ht), u), sp.diff(w.dot(Ht), v)]) + r * (sp.diff(Hv, u) - sp.diff(Hu, v))
lhsH = (sp.diff(Ht, z) - sp.Matrix([sp.diff(H_up[2], u), sp.diff(H_up[2], v)])
        + sp.I * om_eps * zx(Et) + t * GH)
assert sp.simplify(sp.expand(lhsH + zx(sp.Matrix([Ares[0], Ares[1]])) - t * r * Ares[2])) == sp.zeros(2, 1)
print("   d_z H_t = grad H^z - i w eps0 z x E_t - tau G H_t  is exact: ok")

# 5. wall condition
a = sp.Symbol("a", positive=True)
Ez_up_wall = sp.simplify(E_up[2].subs({Ev: 0, Ez: 0}).subs(u, a / 2))
print("5. on u = a/2 with E_v = E_z(cov) = 0:  E^z =", Ez_up_wall)
assert sp.simplify(Ez_up_wall - t * v * Eu.subs(u, a / 2)) == 0
print("all symbolic checks passed")
