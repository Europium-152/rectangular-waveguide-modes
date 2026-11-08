"""Exact Maxwell fields in four tapered / offset / curved rectangular guides, for the T2 checks.

Each test case gives the field (E, H) at frame points (u, v, z) in frame components (E_u, E_v, E_z, H_u, H_v, H_z)
and the local geometry of the cross-section at z (a, b, u0, v0, kappa and the z-derivatives of a, b, u0, v0).
Fields are exact solutions of Maxwell's equations with perfectly conducting walls (time dependence exp(+i w t)):

* EPlaneWedge  -- straight E-plane wedge (b(z) linear, symmetric): walls v = +-(z + z_a) tan(phi0/2), u = +-a/2.
  Fields from TE-to-u and TM-to-u potentials (Hankel functions of order q pi/phi0). The tilted walls carry E_n != 0:
  this is the case that exercises the tilted-wall term.
* HPlaneWedge  -- straight H-plane wedge (a(z) linear): walls u = +-(z + z_a) tan(phi0/2), v = +-b/2. TE-to-v and
  TM-to-v potentials with Bessel J (standing waves that cross cutoff inside the range of z).
* CurvedFrame  -- a straight guide (a_g x b_g, axis along physical Z, centre offset X_c) seen in a curved frame of
  radius R: the cross-section at frame position z is a rectangle with a(z) = a_g/cos(z/R), u0(z) = (R + X_c)/cos(z/R)
  - R, and the scale factor h = 1 + u/R. Exercises kappa, a' and u0' together.
* TiltedGuide  -- a straight guide tilted in the v-z plane by gamma (and/or in the u-z plane by alpha) in a straight
  frame: b = b_g/cos(gamma), v0 = z tan(gamma) (u: a = a_g/cos(alpha), u0 = z tan(alpha)).

The straight-guide fields are superpositions of forward and backward modes (propagating and evanescent) built from
wrmodes.modes.transverse_field and divergence: E_t = e exp(-+i beta Z), H_t = +-Y z x e exp(..),
E_Z = +-Y d/(i w eps0) exp(..), H_Z = g/(i w mu0) exp(..).
"""
import numpy as np
import scipy.constants as cst
import sympy as sp

from wrmodes.modes import Mode, transverse_field, divergence, propagation_constant, wave_admittance

MU0 = cst.mu_0
EPS0 = 1.0 / (MU0 * cst.c ** 2)


# ----------------------------------------------------------------------------------------- potential fields
def _potential_fields(terms, k, coords):
    """Lambdified (E, H) in Cartesian frame components (u, v, z) for a sum of potential terms.

    terms: list of (kind, direction, expr) with kind 'E' (E = curl(d F), H = (i/(w mu0)) curl E) or 'H'
    (H = curl(d G), E = -(i/(w eps0)) curl H); direction 0, 1, 2 = u, v, z; expr a sympy expression in coords.
    """
    u, v, z = coords
    w = k * cst.c

    def curl(F):
        Fu, Fv, Fz = F
        return (sp.diff(Fz, v) - sp.diff(Fv, z), sp.diff(Fu, z) - sp.diff(Fz, u), sp.diff(Fv, u) - sp.diff(Fu, v))

    E = [sp.Integer(0)] * 3
    H = [sp.Integer(0)] * 3
    for kind, d, expr in terms:
        vec = [sp.Integer(0)] * 3
        vec[d] = expr
        if kind == "E":
            e = curl(vec)
            h = [sp.I / (w * MU0) * c for c in curl(e)]
        else:
            h = curl(vec)
            e = [-sp.I / (w * EPS0) * c for c in curl(h)]
        E = [E[i] + e[i] for i in range(3)]
        H = [H[i] + h[i] for i in range(3)]
    fn = sp.lambdify((u, v, z), E + H, modules=["scipy", "numpy"])

    def fields(U, V, Z):
        out = fn(U, V, Z)
        return [np.broadcast_to(np.asarray(o, dtype=complex), np.broadcast(U, V, Z).shape) for o in out]
    return fields


class EPlaneWedge:
    def __init__(self, f, a, za, phi0, comps):
        """comps: list of (kind, m, q, which, coeff): kind 'TEu' (E-potential sin(m pi u'/a), Neumann in phi) or
        'TMu' (H-potential cos(m pi u'/a), Dirichlet in phi); which 'H1', 'H2', 'J' or 'Y'."""
        self.f, self.a, self.za, self.t = f, a, za, np.tan(phi0 / 2)
        k = 2 * np.pi * f / cst.c
        u, v, z = sp.symbols("u v z", real=True)
        Z = z + za
        rho, phi = sp.sqrt(v ** 2 + Z ** 2), sp.atan2(v, Z)
        terms = []
        for kind, m, q, which, coeff in comps:
            kt = float(np.sqrt(k * k - (m * np.pi / a) ** 2))
            nu = sp.Rational(q) * sp.pi / sp.Float(phi0, 30) if q else sp.Integer(0)
            radial = _radial(which, nu, kt * rho)
            if kind == "TEu":
                terms.append(("E", 0, coeff * sp.sin(m * sp.pi * (u + a / 2) / a) * radial
                              * sp.cos(nu * (phi + phi0 / 2))))
            else:
                terms.append(("H", 0, coeff * sp.cos(m * sp.pi * (u + a / 2) / a) * radial
                              * sp.sin(nu * (phi + phi0 / 2))))
        self._fields = _potential_fields(terms, k, (u, v, z))

    def fields(self, U, V, Z):
        return self._fields(U, V, Z)

    def geometry(self, z):
        return dict(a=self.a, b=2 * self.t * (z + self.za), u0=0.0, v0=0.0, kappa=0.0,
                    da=0.0, db=2 * self.t, du0=0.0, dv0=0.0)

    def walls(self, z):
        """Moving walls: list of (normal axis, position, outward sign, normal velocity w)."""
        return [(1, s * self.t * (z + self.za), s, self.t) for s in (-1, 1)]


class HPlaneWedge:
    def __init__(self, f, b, za, phi0, comps):
        """comps: list of (kind, n, q, which, coeff): kind 'TEv' (E-potential sin(n pi v'/b), Neumann in phi) or
        'TMv' (H-potential cos(n pi v'/b), Dirichlet in phi)."""
        self.f, self.b, self.za, self.t = f, b, za, np.tan(phi0 / 2)
        k = 2 * np.pi * f / cst.c
        u, v, z = sp.symbols("u v z", real=True)
        Z = z + za
        rho, phi = sp.sqrt(u ** 2 + Z ** 2), sp.atan2(u, Z)
        terms = []
        for kind, n, q, which, coeff in comps:
            kt = float(np.sqrt(k * k - (n * np.pi / b) ** 2))
            nu = sp.Rational(q) * sp.pi / sp.Float(phi0, 30) if q else sp.Integer(0)
            radial = _radial(which, nu, kt * rho)
            if kind == "TEv":
                terms.append(("E", 1, coeff * sp.sin(n * sp.pi * (v + b / 2) / b) * radial
                              * sp.cos(nu * (phi + phi0 / 2))))
            else:
                terms.append(("H", 1, coeff * sp.cos(n * sp.pi * (v + b / 2) / b) * radial
                              * sp.sin(nu * (phi + phi0 / 2))))
        self._fields = _potential_fields(terms, k, (u, v, z))

    def fields(self, U, V, Z):
        return self._fields(U, V, Z)

    def geometry(self, z):
        return dict(a=2 * self.t * (z + self.za), b=self.b, u0=0.0, v0=0.0, kappa=0.0,
                    da=2 * self.t, db=0.0, du0=0.0, dv0=0.0)

    def walls(self, z):
        return [(0, s * self.t * (z + self.za), s, self.t) for s in (-1, 1)]


def _radial(which, nu, x):
    if which == "J":
        return sp.besselj(nu, x)
    if which == "Y":
        return sp.bessely(nu, x)
    if which == "H1":
        return sp.hankel1(nu, x)
    return sp.hankel2(nu, x)


# ----------------------------------------------------------------------------------------- straight guide fields
class StraightGuideField:
    """Superposition of modes of a straight a_g x b_g guide in its own Cartesian frame (X', Y', Z'), centred."""

    def __init__(self, f, a_g, b_g, comps):
        """comps: list of (mode, direction +1/-1, coefficient)."""
        self.f, self.a_g, self.b_g = f, a_g, b_g
        self.comps = [(Mode.parse(m), d, c) for m, d, c in comps]
        self.w = 2 * np.pi * f

    def __call__(self, X, Y, Z):
        a, b, w = self.a_g, self.b_g, self.w
        out = [np.zeros(np.broadcast(X, Y, Z).shape, dtype=complex) for _ in range(6)]
        for mode, d, c in self.comps:
            beta = propagation_constant(mode, a, b, self.f)
            Ym = wave_admittance(mode, a, b, self.f)
            ph = c * np.exp(-1j * d * beta * Z)
            eu, ev = transverse_field(mode, a, b, X, Y)
            dd, gg = divergence(mode, a, b, X, Y)
            out[0] += eu * ph
            out[1] += ev * ph
            out[2] += d * Ym * dd / (1j * w * EPS0) * ph
            # H_t = d Y z x e = d Y (-e_v, e_u)
            out[3] += -d * Ym * ev * ph
            out[4] += d * Ym * eu * ph
            out[5] += gg / (1j * w * MU0) * ph
        return out


class CurvedFrame:
    def __init__(self, f, a_g, b_g, R, Xc, comps):
        self.R, self.Xc, self.a_g, self.b_g = R, Xc, a_g, b_g
        self.guide = StraightGuideField(f, a_g, b_g, comps)

    def fields(self, U, V, Z):
        th = Z / self.R
        c, s = np.cos(th), np.sin(th)
        X = -self.R + (self.R + U) * c
        Zp = (self.R + U) * s
        Ex, Ey, Ez, Hx, Hy, Hz = self.guide(X - self.Xc, V, Zp)
        # frame unit vectors: u = (c, 0, s), v = (0, 1, 0), z = (-s, 0, c)
        return [Ex * c + Ez * s, Ey, -Ex * s + Ez * c, Hx * c + Hz * s, Hy, -Hx * s + Hz * c]

    def geometry(self, z):
        R, th = self.R, z / self.R
        sec, tan = 1 / np.cos(th), np.tan(th)
        return dict(a=self.a_g * sec, b=self.b_g, u0=(R + self.Xc) * sec - R, v0=0.0, kappa=1 / R,
                    da=self.a_g * sec * tan / R, db=0.0, du0=(R + self.Xc) * sec * tan / R, dv0=0.0)

    def walls(self, z):
        g = self.geometry(z)
        return [(0, g["u0"] + s * g["a"] / 2, s, s * g["du0"] + g["da"] / 2) for s in (-1, 1)]


class TiltedGuide:
    def __init__(self, f, a_g, b_g, alpha, gamma, comps):
        """Guide rotated by gamma about u (tilt in v-z), or by alpha about v (tilt in u-z); use one at a time."""
        assert alpha == 0 or gamma == 0
        self.alpha, self.gamma, self.a_g, self.b_g = alpha, gamma, a_g, b_g
        self.guide = StraightGuideField(f, a_g, b_g, comps)

    def fields(self, U, V, Z):
        ca, sa, cg, sg = np.cos(self.alpha), np.sin(self.alpha), np.cos(self.gamma), np.sin(self.gamma)
        if self.gamma:
            Yp, Zp = V * cg - Z * sg, V * sg + Z * cg
            Ex, Ey, Ez, Hx, Hy, Hz = self.guide(U, Yp, Zp)
            return [Ex, Ey * cg + Ez * sg, -Ey * sg + Ez * cg, Hx, Hy * cg + Hz * sg, -Hy * sg + Hz * cg]
        Xp, Zp = U * ca - Z * sa, U * sa + Z * ca
        Ex, Ey, Ez, Hx, Hy, Hz = self.guide(Xp, V, Zp)
        return [Ex * ca + Ez * sa, Ey, -Ex * sa + Ez * ca, Hx * ca + Hz * sa, Hy, -Hx * sa + Hz * ca]

    def geometry(self, z):
        ta, tg = np.tan(self.alpha), np.tan(self.gamma)
        return dict(a=self.a_g / np.cos(self.alpha), b=self.b_g / np.cos(self.gamma), u0=z * ta, v0=z * tg,
                    kappa=0.0, da=0.0, db=0.0, du0=ta, dv0=tg)

    def walls(self, z):
        g = self.geometry(z)
        if self.gamma:
            return [(1, g["v0"] + s * g["b"] / 2, s, s * g["dv0"]) for s in (-1, 1)]
        return [(0, g["u0"] + s * g["a"] / 2, s, s * g["du0"]) for s in (-1, 1)]
