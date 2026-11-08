"""Reference solver for the T2 equations: GSM slices with V-I propagators (prototype for WP5, not library code).

Method (docs/theory/T2-taper-coupling.md, section "Cutoff crossings"):

1. One mode list for the whole taper (the union over z of the modes kept), in a fixed order.
2. Slice boundaries z_0 = 0 < ... < z_S = L, none of them close to a cutoff plane: |beta_p(z_j)^2| >= DELTA_C k^2
   for every mode (boundaries that violate this are moved), and |Im beta_p| dz <= GROWTH_MAX on every slice.
3. On each slice, the V-I propagator Phi (d[V; I]/dz = M(z) [V; I], M regular at cutoff) by the 4th-order Magnus
   method with two Gauss points per sub-step.
4. The slice GSM is built directly from Phi by solving the two-point problem in power waves at the two slice ends
   (local admittances Y_p(z_j), principal square roots, CONVENTIONS.md section 5).
5. The slice GSMs are cascaded with the Redheffer star product (wrmodes.gsm), each port carrying the local basis.
"""
import os
import sys

import numpy as np
import scipy.constants as cst
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(__file__))
import t2_coefficients as t2                     # noqa: E402
from wrmodes.modes import Mode, ModeBasis, wave_admittance, cutoff_wavenumber   # noqa: E402
from wrmodes.gsm import GSM, cascade             # noqa: E402

DELTA_C = 1e-6          # minimum |beta^2|/k^2 at a slice boundary (cond(D) ~ DELTA_C**-0.25)
GROWTH_MAX = 3.0        # maximum |Im beta| dz on one slice


def vi_matrix(modes, g, f):
    return t2.vi_matrix(modes, g["a"], g["b"], f, g["da"], g["db"], g["du0"], g["dv0"], g["kappa"], g["u0"])


def propagator(modes, geometry, f, z0, z1, nsub):
    """Phi with [V; I](z1) = Phi [V; I](z0): 4th-order Magnus (two Gauss points) on nsub sub-steps."""
    n2 = 2 * len(modes)
    Phi = np.eye(n2, dtype=complex)
    edges = np.linspace(z0, z1, nsub + 1)
    c = np.sqrt(3) / 6
    for za, zb in zip(edges[:-1], edges[1:]):
        h = zb - za
        A1 = vi_matrix(modes, geometry(za + (0.5 - c) * h), f)
        A2 = vi_matrix(modes, geometry(za + (0.5 + c) * h), f)
        Om = 0.5 * h * (A1 + A2) + (np.sqrt(3) / 12) * h * h * (A2 @ A1 - A1 @ A2)
        Phi = expm(Om) @ Phi
    return Phi


def wave_rows(modes, g, f):
    """(P+, P-): a+ = P+ [V; I], a- = P- [V; I] with V = sqrt(2)(a+ + a-)/sqrt(Y), I = sqrt(2)(a+ - a-) sqrt(Y)."""
    sq = np.sqrt(np.array([wave_admittance(m, g["a"], g["b"], f) for m in modes], dtype=complex))
    Pp = np.hstack([np.diag(sq), np.diag(1 / sq)]) / (2 * np.sqrt(2))
    Pm = np.hstack([np.diag(sq), -np.diag(1 / sq)]) / (2 * np.sqrt(2))
    return Pp, Pm


def slice_gsm(modes, geometry, f, z0, z1, nsub):
    """GSM of one slice built directly from its V-I propagator (two-point problem in power waves)."""
    Phi = propagator(modes, geometry, f, z0, z1, nsub)
    g0, g1 = geometry(z0), geometry(z1)
    Pp0, Pm0 = wave_rows(modes, g0, f)
    Pp1, Pm1 = wave_rows(modes, g1, f)
    A = np.vstack([Pp0, Pm1 @ Phi])          # incident: a+(z0), a-(z1)
    B = np.vstack([Pm0, Pp1 @ Phi])          # scattered: a-(z0), a+(z1)
    S = B @ np.linalg.inv(A)
    b0 = ModeBasis(g0["a"], g0["b"], modes, sort=False)
    b1 = ModeBasis(g1["a"], g1["b"], modes, sort=False)
    return GSM.from_matrix(S, b0, b1, f)


def slice_edges(modes, geometry, f, L, nslices):
    k = 2 * np.pi * f / cst.c
    edges = list(np.linspace(0.0, L, nslices + 1))
    dz = L / nslices

    def ok(z):
        g = geometry(z)
        return all(abs(k * k - cutoff_wavenumber(m, g["a"], g["b"]) ** 2) >= DELTA_C * k * k for m in modes)

    for j in range(1, nslices):
        if not ok(edges[j]):
            for shift in (0.25, -0.25, 0.4, -0.4):
                if ok(edges[j] + shift * dz):
                    edges[j] += shift * dz
                    break
            else:
                raise RuntimeError("could not move a slice boundary away from a cutoff plane")
    return edges


def taper_gsm(modes, geometry, f, L, nslices=40, nsub=2):
    """GSM of the section [0, L] in the given (fixed) mode list, local bases at the two ends."""
    modes = [Mode.parse(m) for m in modes]
    for end in (0.0, L):
        g = geometry(end)
        k = 2 * np.pi * f / cst.c
        if any(abs(k * k - cutoff_wavenumber(m, g["a"], g["b"]) ** 2) < DELTA_C * k * k for m in modes):
            raise ValueError("a port plane is at the cutoff of a mode: its power-wave amplitude is undefined")
    edges = slice_edges(modes, geometry, f, L, nslices)
    k = 2 * np.pi * f / cst.c
    gsms = []
    for z0, z1 in zip(edges[:-1], edges[1:]):
        # conditioning of the slice GSM: split the slice if the largest evanescent growth on it exceeds GROWTH_MAX
        zz = np.linspace(z0, z1, 5)
        kappa_max = max(np.sqrt(max(cutoff_wavenumber(m, geometry(z)["a"], geometry(z)["b"]) ** 2 - k * k, 0.0))
                        for m in modes for z in zz)
        nsplit = max(1, int(np.ceil(kappa_max * (z1 - z0) / GROWTH_MAX)))
        sub = np.linspace(z0, z1, nsplit + 1)
        for za, zb in zip(sub[:-1], sub[1:]):
            gsms.append(slice_gsm(modes, geometry, f, za, zb, nsub))
    return cascade(*gsms)
