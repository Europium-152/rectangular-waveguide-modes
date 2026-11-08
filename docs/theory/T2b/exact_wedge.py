"""Exact Maxwell fields in a PEC wedge guide, used to check the taper equations (card T2b).

A wedge guide has two flat PEC walls through an apex line and two parallel PEC walls perpendicular to that line.
Its fields are exact superpositions of cylindrical waves (e.g. the H-plane sectoral horn), with time dependence
exp(+i w t):

* the guide is uniform along a Cartesian direction t (unit vector `tvec`, width `w` between the parallel walls,
  centred on t = 0); the wedge lies in the plane spanned by the other transverse axis `pvec` and the axis Zc;
* in that plane, polar coordinates around the apex A: r, phi with phi measured from +Zc towards +pvec; the walls
  are at phi_minus < phi_plus;
* LSM-to-t solutions (no H_t):   Phi = cos(l pi (t + w/2)/w) psi_D,  E = grad(d_t Phi) + k^2 t Phi,
                                 H = i w eps0 grad(Phi) x t,  psi_D = H2_nu(k_t r) sin(nu (phi - phi_minus));
* LSE-to-t solutions (no E_t):   Psi = sin(l pi (t + w/2)/w) psi_N,  H = grad(d_t Psi) + k^2 t Psi,
                                 E = -i w mu0 grad(Psi) x t,  psi_N = H2_nu(k_t r) cos(nu (phi - phi_minus));
  with nu = j pi / (phi_plus - phi_minus) and k_t^2 = k^2 - (l pi / w)^2.  H2 (second kind) is outgoing (+r)
  for exp(+i w t); `kind=1` gives the incoming wave and `kind=0` the standing wave J_nu (incoming + outgoing),
  regular at the apex: a mode incident from the wide end and totally reflected at its cutoff cross-section.
"""
import numpy as np
import scipy.constants as cst
from scipy.special import hankel1, hankel2, h1vp, h2vp, jv, jvp

from wrmodes.modes import ETA0


class WedgeField:
    def __init__(self, f, apex, phi_minus, phi_plus, width, tvec, pvec, terms):
        """terms: list of (coef, 'LSM'|'LSE', j, l, kind) with kind 2 (outgoing) or 1 (incoming)."""
        self.k = 2 * np.pi * f / cst.c
        self.w = 2 * np.pi * f
        self.apex = np.asarray(apex, float)       # 3-vector (global X, Y, Zc)
        self.pm, self.pp = phi_minus, phi_plus
        self.width = width
        self.t = np.asarray(tvec, float)
        self.p = np.asarray(pvec, float)
        self.zc = np.array([0.0, 0.0, 1.0])
        self.terms = terms

    def fields(self, X):
        """E, H (complex, shape (..., 3)) at global points X (shape (..., 3))."""
        X = np.asarray(X, float)
        R = X - self.apex
        T = R @ self.t
        P = R @ self.p
        Zc = R @ self.zc
        r = np.hypot(P, Zc)
        phi = np.arctan2(P, Zc)
        er = (P[..., None] * self.p + Zc[..., None] * self.zc) / r[..., None]
        ephi = (Zc[..., None] * self.p - P[..., None] * self.zc) / r[..., None]
        E = np.zeros(X.shape, complex)
        H = np.zeros(X.shape, complex)
        k, w, mu, ep = self.k, self.w, cst.mu_0, 1 / (ETA0 * cst.c)   # eps0 consistent with ETA0 = mu0 c
        for coef, typ, j, l, kind in self.terms:
            kl = l * np.pi / self.width
            kt = np.sqrt(k * k - kl * kl + 0j)
            nu = j * np.pi / (self.pp - self.pm)
            Hf, Hd = {2: (hankel2, h2vp), 1: (hankel1, h1vp), 0: (jv, jvp)}[kind]
            Z, dZ = Hf(nu, kt * r), kt * Hd(nu, kt * r)
            ang = nu * (phi - self.pm)
            if typ == "LSM":
                A, dA = np.sin(ang), nu * np.cos(ang)
                tau, dtau, ddtau = (np.cos(kl * (T + self.width / 2)), -kl * np.sin(kl * (T + self.width / 2)),
                                    -kl * kl * np.cos(kl * (T + self.width / 2)))
            else:
                A, dA = np.cos(ang), -nu * np.sin(ang)
                tau, dtau, ddtau = (np.sin(kl * (T + self.width / 2)), kl * np.cos(kl * (T + self.width / 2)),
                                    -kl * kl * np.sin(kl * (T + self.width / 2)))
            psi = Z * A
            gpsi = (dZ * A)[..., None] * er + (Z * dA / r)[..., None] * ephi     # in-plane gradient
            gradPhi = (dtau * psi)[..., None] * self.t + tau[..., None] * gpsi
            grad_dtPhi = (ddtau * psi)[..., None] * self.t + dtau[..., None] * gpsi
            F1 = grad_dtPhi + k * k * (tau * psi)[..., None] * self.t
            F2 = np.cross(gradPhi, self.t)
            if typ == "LSM":
                E += coef * F1
                H += coef * 1j * w * ep * F2
            else:
                H += coef * F1
                E += coef * (-1j * w * mu) * F2
        return E, H
