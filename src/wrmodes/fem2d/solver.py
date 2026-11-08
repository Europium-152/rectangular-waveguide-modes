"""Finite-element solution of the scalar 2D problems of a coplanar component (CONVENTIONS.md section 7).

For each family n and LS polarization, the scalar psi (E_v profile for LSM, H_v profile for LSE) solves

    laplace(psi) + k_t^2 psi = 0,   k_t^2 = k^2 - (n pi / b)^2,

with psi = 0 (LSM) or d psi / dn = 0 (LSE) on the walls. Each port is the far end of a straight arm; there the
field is expanded in the port modes phi_m (``normalization.scalar_mode``),

    psi = sum_m (alpha_m^in exp(-i beta_m s) + alpha_m^out exp(+i beta_m s)) phi_m(u),

with s the distance into the domain, so that the outward normal derivative is
d psi / dn = sum_m i beta_m (2 alpha_m^in - c_m) phi_m, c_m = int_port psi phi_m. The weak form with Lagrange elements
(P2 by default) is therefore

    int grad(w).grad(psi) - k_t^2 w psi + sum_ports sum_m i beta_m (int w phi_m)(int psi phi_m)
        = sum_ports sum_m 2 i beta_m alpha_m^in int w phi_m,

the modal (Dirichlet-to-Neumann) port condition truncated to the first M port modes. The matrix A is complex
symmetric (unconjugated). With the port vectors b_m = int w phi_m, the outgoing amplitudes are
alpha^out = c - alpha^in, so that S_ij = 2 i beta_j b_i^T A^-1 b_j - delta_ij: beta_i S_ij = beta_j S_ji holds to
rounding (reciprocity), and since K and M are real while i beta_m is imaginary for propagating and real positive
for evanescent port modes, the discrete S conserves power over the propagating port modes to rounding.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .normalization import scalar_mode

__all__ = ["Discretization", "solve_scalar"]


class Discretization:
    """Mesh, Lagrange basis and the frequency-independent matrices of a domain (MeshData from geometry)."""

    def __init__(self, mesh_data, order=2):
        from skfem import MeshTri, MeshTri2, Basis, FacetBasis, ElementTriP2, ElementTriP3, BilinearForm, asm
        from skfem.helpers import dot, grad
        self.md = md = mesh_data
        if order not in (2, 3):
            raise ValueError("order must be 2 or 3")
        elem = {2: ElementTriP2, 3: ElementTriP3}[order]()
        mesh = MeshTri(md.p, md.t)
        if md.arcs:
            # isoparametric quadratic geometry: move the mid-edge nodes of curved boundary edges onto the arcs
            b0 = Basis(mesh, ElementTriP2())
            doflocs = b0.doflocs.copy()
            bf = mesh.boundary_facets()
            for cx, cy, r in md.arcs:
                c = np.array([[cx], [cy]])
                ends = mesh.p[:, mesh.facets[:, bf]]            # (2, 2, nb)
                on = ((np.abs(np.hypot(*(ends[:, 0] - c)) - r) < 1e-9 * r)
                      & (np.abs(np.hypot(*(ends[:, 1] - c)) - r) < 1e-9 * r))
                dofs = b0.facet_dofs[0, bf[on]]
                x = doflocs[:, dofs] - c
                doflocs[:, dofs] = c + r * x / np.hypot(*x)
            mesh = MeshTri2(doflocs, mesh.t)
        self.mesh = mesh
        self.elem = elem
        self.basis = Basis(mesh, elem, intorder=2 * order + 1)
        self.n = self.basis.N

        @BilinearForm
        def stiff(u, v, w):
            return dot(grad(u), grad(v))

        @BilinearForm
        def mass(u, v, w):
            return u * v

        self.K = asm(stiff, self.basis).tocsr()
        self.M = asm(mass, self.basis).tocsr()
        # boundary facets on the modal ports (far ends of the arms); all other boundary facets are walls
        bf = mesh.boundary_facets()
        mid = 0.5 * (mesh.p[:, mesh.facets[0, bf]] + mesh.p[:, mesh.facets[1, bf]])
        self.port_facets, self.port_frames = [], []
        tol = 1e-7 * md.h
        for port, nrm in zip(md.ports, md.normals):
            start = np.array(port.inner) - md.port_distance * nrm      # u = -a/2 end of the modal port
            u = port.u_hat
            s = u @ (mid - start[:, None])
            dn = nrm @ (mid - start[:, None])
            on = (np.abs(dn) < tol) & (s > -tol) & (s < port.a + tol)
            if not on.any():
                raise RuntimeError("no boundary facet found on a modal port")
            self.port_facets.append(bf[on])
            self.port_frames.append((start, u, port.a))
        wall = np.setdiff1d(bf, np.concatenate(self.port_facets))
        self.wall_dofs = np.unique(self.basis.get_dofs(facets=wall).flatten())
        self.port_bases = [FacetBasis(mesh, elem, facets=f, intorder=16) for f in self.port_facets]

    def port_matrix(self, k, pol, modes):
        """B (n x len(modes)): column m is int_port phi_m(u) w ds over the modal port of arm k."""
        from skfem import LinearForm, asm
        fb = self.port_bases[k]
        start, u, a = self.port_frames[k]
        cols = []
        for m in modes:
            @LinearForm
            def lf(v, w, m=m):
                s = u[0] * (w.x[0] - start[0]) + u[1] * (w.x[1] - start[1])
                return scalar_mode(pol, m, a, s) * v
            cols.append(asm(lf, fb))
        return sp.csr_matrix(np.array(cols).T)


def solve_scalar(disc, kt2, pol, port_modes, port_betas, excite):
    """Solve one scalar 2D problem and return its generalized scattering matrix in 2D amplitudes.

    Parameters
    ----------
    disc : Discretization
    kt2 : float
        k_t^2 (1/m^2).
    pol : 'LSM' (Dirichlet) or 'LSE' (Neumann).
    port_modes, port_betas : list of two sequences
        Mode indices m and propagation constants (branch of CONVENTIONS.md section 3) of the modal condition at each
        port.
    excite : list of two sequences
        Positions (in port_modes[k]) of the modes whose scattering is wanted at each port.

    Returns
    -------
    S : (N, N) complex array
        Outgoing per unit incident 2D amplitude at the modal ports (port 1 modes first), with
        beta_i S_ij = beta_j S_ji to rounding.
    info : dict
        'ndof': number of unknowns.
    """
    A = (disc.K - kt2 * disc.M).astype(complex)
    Bs = []
    for k in range(2):
        B = disc.port_matrix(k, pol, port_modes[k])
        Bs.append(B)
        A = A + B @ sp.diags(1j * np.asarray(port_betas[k], dtype=complex)) @ B.T
    free = np.ones(disc.n, dtype=bool)
    if pol == "LSM":
        free[disc.wall_dofs] = False
    fidx = np.flatnonzero(free)
    Af = A.tocsr()[fidx][:, fidx].tocsc()
    P = np.hstack([Bs[k].tocsr()[fidx][:, list(excite[k])].toarray() for k in range(2)])
    beta = np.concatenate([np.asarray(port_betas[k], dtype=complex)[list(excite[k])] for k in range(2)])
    lu = spla.splu(Af, permc_spec="COLAMD")
    W = lu.solve(P * (2j * beta)[None, :])
    S = P.T @ W - np.eye(len(beta))
    return S, {"ndof": len(fidx)}
