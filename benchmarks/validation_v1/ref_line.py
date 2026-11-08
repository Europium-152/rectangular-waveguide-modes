"""Global-ODE reference for transmission lines of straights and smooth bends (WP7 item 1, verifies WP1).

The whole line is one coupled-mode system in the global guide frame (CONVENTIONS.md section 2):

    dy/dz = (D + kappa(z) K_g(z)) y,      y = [A+; A-]  (all propagating modes of the w x h guide),

where K_g is the curvature matrix of the segment at z in the global frame: the closed-form matrix of wrmodes 0.1
(``coupled_mode_matrices``, family by family) for H-plane segments, and F^T K_loc F for E-plane segments, with the
frame relabelling F derived independently in ``ref_frames``. D = diag(-i beta, +i beta) is the same in both frames.

The system is integrated in the interaction picture, y(z) = exp(D z) c(z) with the absolute line coordinate z:

    dc/dz = kappa(z) exp(-D z) K_g exp(D z) c,     c(0) = I  (fundamental matrix),

with scipy's DOP853 at rtol = 1e-11, piecewise between the curvature breakpoints. Straights have kappa = 0, so c is
constant there and arbitrarily long straights carry no integration error; the phase exp(D L) is applied once at the
end. The transfer matrix of the full line is Phi = exp(D L) c(L); T and R follow from the algebra of the JINST paper,
R = -Phi_bb^-1 Phi_bf, T = Phi_ff + Phi_fb R. Nothing here uses wrmodes.gsm (no Redheffer product, no GSM).

``line_transfer_v01`` builds the same line with the wrmodes 0.1 transfer-matrix solver (product of the transfer
matrices of the segments): the acceptance check of WP7 compares the two to 1e-9 in power.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp

from wrmodes.modes import ModeBasis
from wrmodes.transfer_matrix import coupled_mode_matrices, solve_transfer_matrix

import ref_frames

INCH = 0.0254
WR90 = (0.9 * INCH, 0.4 * INCH)


@dataclass
class Segment:
    """One piece of a line. kind: 'straight' or 'bend'. curvature: kappa(z) in 1/m for z in [0, length] (local)."""
    kind: str
    length: float
    curvature: object = None
    plane: str = "H"
    breakpoints: tuple = field(default_factory=tuple)
    label: str = ""


def straight(length, label=""):
    return Segment("straight", float(length), None, "H", (), label or f"straight {length:g} m")


def bend(length, curvature, plane="H", breakpoints=(), label=""):
    return Segment("bend", float(length), curvature, plane, tuple(breakpoints), label)


def sech_curvature(theta_deg, length, T=4.0, sign=1.0):
    """Hyperbolic-secant profile of the JINST paper (eqs. 4.6-4.7), total angle theta over `length`."""
    th = np.radians(theta_deg)
    g = 2 * T / length
    R0 = 4 / th / g * np.arctan(np.tanh(T / 2))
    return lambda z: sign / (R0 * np.cosh(g * (np.asarray(z, dtype=float) - length / 2)))


def constant_curvature(theta_deg, length, sign=1.0):
    k0 = sign * np.radians(theta_deg) / length
    return lambda z: np.full(np.shape(z), k0)


# ----------------------------------------------------------------------------------------- matrices
def global_basis(f, w, h):
    """All propagating modes of the w x h guide at f (cutoff_factor = 1), in the order of CONVENTIONS.md."""
    return ModeBasis.propagating(f, w, h)


def _local_K(f, basis):
    """(D, K) of the 0.1 closed form on a local basis, assembled family by family (families do not couple)."""
    N = len(basis)
    D = np.zeros((2 * N, 2 * N), complex)
    K = np.zeros((2 * N, 2 * N), complex)
    for n in basis.families():
        idx = basis.family(n)
        sub = basis.subset(idx)
        Df, Kf, beta = coupled_mode_matrices(f, basis.a, basis.b, sub.legacy_modes(f))
        Nf = len(idx)
        gi = np.array(idx + [N + i for i in idx])
        D[np.ix_(gi, gi)] = Df
        K[np.ix_(gi, gi)] = Kf
        k = 2 * np.pi * f / 299792458.0          # two formulas for beta: equal up to rounding of k^2 - chi^2
        assert np.allclose(beta, basis.beta(f)[idx], rtol=0, atol=1e-10 * k)
    return D, K


def plane_matrices(f, gbasis):
    """D and {'H': K_H, 'E': K_E} in the global frame for the global basis."""
    D, KH = _local_K(f, gbasis)
    lb = ref_frames.eplane_local_basis(gbasis)
    DE, KE_loc = _local_K(f, lb)
    Fd = ref_frames.doubled(ref_frames.eplane_matrix(gbasis, lb))
    KE = Fd.T @ KE_loc @ Fd                                   # a_loc = F a_glob, F^-1 = F^T
    assert np.allclose(Fd.T @ DE @ Fd, D, rtol=0, atol=1e-10 * np.max(np.abs(D)))
    return D, {"H": KH, "E": KE}


def scattering_from_transfer(phi):
    """S11 = R, S21 = T (port-1 incidence) and S12, S22 (port-2 incidence) from the 2N x 2N transfer matrix.

    y = [A+; A-]: A-(L) = 0 for port-1 incidence gives R = -Pbb^-1 Pbf, T = Pff + Pfb R (JINST paper). For incidence
    A-(L) = a2 from port 2 with A+(0) = 0: A-(0) = Pbb^-1 a2, A+(L) = Pfb Pbb^-1 a2.
    """
    N = phi.shape[0] // 2
    Pff, Pfb, Pbf, Pbb = phi[:N, :N], phi[:N, N:], phi[N:, :N], phi[N:, N:]
    R = -np.linalg.solve(Pbb, Pbf)
    T = Pff + Pfb @ R
    S12 = np.linalg.inv(Pbb)
    S22 = Pfb @ S12
    return {"S11": R, "S21": T, "S12": S12, "S22": S22}


# ----------------------------------------------------------------------------------------- reference
def coupled_components(Ks, N, tol=0.0):
    """Groups of modes that the curvature matrices connect (union-find on the N x N coupling pattern).

    Each group is an invariant subspace of the whole line's system, so integrating the groups separately is exact
    (for a line in one plane the groups are the families n).
    """
    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pattern = np.zeros((N, N), bool)
    for K in Ks:
        A = np.abs(K)
        pattern |= (A[:N, :N] + A[:N, N:] + A[N:, :N] + A[N:, N:]) > tol
    for i, j in zip(*np.nonzero(pattern)):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


def line_transfer(f, w, h, segments, rtol=1e-11, atol=1e-13):
    """Transfer matrix of a whole line by one interaction-picture DOP853 integration (see module docstring).

    The modes are split into the groups that the line's curvature matrices connect (``coupled_components``); each
    group is integrated over the whole line. Returns (basis, phi, S, info): S is the dict of
    ``scattering_from_transfer``, info holds the number of right-hand-side evaluations and the line length.
    """
    gb = global_basis(f, w, h)
    N = len(gb)
    D, Ks = plane_matrices(f, gb)
    d_all = np.diag(D).copy()
    used = sorted({s.plane for s in segments if s.kind != "straight" and s.curvature is not None})
    groups = coupled_components([Ks[p] for p in used], N) if used else [[i] for i in range(N)]
    total = sum(s.length for s in segments)
    phi = np.zeros((2 * N, 2 * N), complex)
    nfev = 0
    for g in groups:
        gi = np.array(g + [N + i for i in g])
        d = d_all[gi]
        n2 = len(gi)
        C = np.eye(n2, dtype=complex)
        z = 0.0
        for seg in segments:
            if seg.kind == "straight" or seg.curvature is None:
                z += seg.length                              # kappa = 0: c is constant
                continue
            K = Ks[seg.plane][np.ix_(gi, gi)]
            if not np.any(K):
                z += seg.length
                continue
            z0 = z
            kap = seg.curvature
            pts = [0.0] + sorted(p for p in seg.breakpoints if 0 < p < seg.length) + [seg.length]

            def rhs(zz, c, K=K, z0=z0, kap=kap, d=d, n2=n2):
                e = np.exp(d * zz)
                M = (K * e[None, :]) / e[:, None]          # exp(-D z) K exp(D z)
                return (float(kap(zz - z0)) * (M @ c.reshape(n2, n2))).ravel()

            for p0, p1 in zip(pts[:-1], pts[1:]):
                sol = solve_ivp(rhs, (z0 + p0, z0 + p1), C.ravel(), method="DOP853", rtol=rtol, atol=atol)
                if not sol.success:
                    raise RuntimeError(sol.message)
                nfev += sol.nfev
                C = sol.y[:, -1].reshape(n2, n2)
            z += seg.length
        phi[np.ix_(gi, gi)] = np.exp(d * total)[:, None] * C
    return gb, phi, scattering_from_transfer(phi), {"nfev": nfev, "length": total, "groups": len(groups)}


def line_transfer_v01(f, w, h, segments, atol=1e-10):
    """The same line from the wrmodes 0.1 transfer-matrix solver: product of segment transfer matrices.

    Each bend family is solved with solve_transfer_matrix(error_control='full', atol=atol); E-plane bends are solved
    in their local frame (a = h, b = w) and relabelled with ref_frames.eplane_matrix. Straights are exp(D L).
    """
    gb = global_basis(f, w, h)
    N = len(gb)
    beta = gb.beta(f)
    d = np.concatenate([-1j * beta, 1j * beta])
    phi = np.eye(2 * N, dtype=complex)
    lb = ref_frames.eplane_local_basis(gb)
    Fd = ref_frames.doubled(ref_frames.eplane_matrix(gb, lb))
    for seg in segments:
        if seg.kind == "straight" or seg.curvature is None:
            P = np.diag(np.exp(d * seg.length))
        else:
            basis = gb if seg.plane == "H" else lb
            P = np.zeros((2 * N, 2 * N), complex)
            for n in basis.families():
                idx = basis.family(n)
                sub = basis.subset(idx)
                legacy = sub.legacy_modes(f)
                m0 = sub.modes[0]
                _, res = solve_transfer_matrix(f, basis.a, basis.b, (m0.kind, m0.m, m0.n), seg.length, seg.curvature,
                                               coupled_modes=legacy, atol=atol, error_control="full", profile=False,
                                               breakpoints=list(seg.breakpoints) or None, max_slices=2 ** 16)
                gi = np.array(idx + [N + i for i in idx])
                P[np.ix_(gi, gi)] = res.transfer_matrix
            if seg.plane == "E":
                P = Fd.T @ P @ Fd
        phi = P @ phi
    return gb, phi, scattering_from_transfer(phi)


# ----------------------------------------------------------------------------------------- the three lines
def reference_lines():
    """The three reference lines of WP7 (WR-90, global frame with TE10's E along y).

    * 'same-plane': straight 0.3 m, 60 deg sech H bend 0.6 m, straight 1 m, 90 deg sech H bend 0.4 m turning the
      other way (negative kappa), straight 0.2 m.
    * 'mixed-planes': 60 deg sech H bend 0.6 m, straight 0.5 m, 90 deg sech E bend 0.4 m, straight 0.3 m, 90 deg
      sech H bend 0.4 m, then a 45 deg constant-curvature E bend 0.2 m with negative kappa.
    * 'long-straights': the paper's section 7 sequence with 10 m straights, H plane: 60 deg sech 0.6 m, 10 m,
      90 deg sech 0.4 m, 10 m, -90 deg sech 0.4 m (21.4 m in total).
    """
    return {
        "same-plane": [straight(0.3), bend(0.6, sech_curvature(60, 0.6), "H", label="H 60 sech 0.6 m"),
                       straight(1.0), bend(0.4, sech_curvature(90, 0.4, sign=-1), "H", label="H -90 sech 0.4 m"),
                       straight(0.2)],
        "mixed-planes": [bend(0.6, sech_curvature(60, 0.6), "H", label="H 60 sech 0.6 m"), straight(0.5),
                         bend(0.4, sech_curvature(90, 0.4), "E", label="E 90 sech 0.4 m"), straight(0.3),
                         bend(0.4, sech_curvature(90, 0.4), "H", label="H 90 sech 0.4 m"),
                         bend(0.2, constant_curvature(45, 0.2, sign=-1), "E", label="E -45 const 0.2 m")],
        "long-straights": [bend(0.6, sech_curvature(60, 0.6), "H", label="H 60 sech 0.6 m"), straight(10.0),
                           bend(0.4, sech_curvature(90, 0.4), "H", label="H 90 sech 0.4 m"), straight(10.0),
                           bend(0.4, sech_curvature(90, 0.4, sign=-1), "H", label="H -90 sech 0.4 m")],
    }


REFERENCE_FREQUENCIES = {
    "same-plane": [20e9, 40e9, 60e9, 90e9],
    "mixed-planes": [20e9, 30e9, 40e9],
    "long-straights": [20e9, 50e9, 90e9],
}
