"""Frame change between the global (H-plane) guide frame and the E-plane local frame, derived independently.

WP7 reference (tasks/WP7-verification-wave1.md, item 1). This module does not use ``wrmodes.modes.eplane_frame_signs``;
the test suite compares the two.

Derivation (CONVENTIONS.md sections 2 and 4)
--------------------------------------------
Global frame (x, y, z): guide w x h, x' = x + w/2, y' = y + h/2. E-plane local frame: u = y, v = -x (a rotation by
-90 degrees about z, so z is shared and u_hat x v_hat = y_hat x (-x_hat) = z_hat), sides a = h, b = w, and
u' = u + h/2 = y', v' = v + w/2 = w - x'.

The transverse fields are coordinate-free vector expressions of the potentials, e_TE = grad Gamma x z_hat and
e_TM = grad Pi, and z_hat is common to both frames. A local mode is therefore +-1 times a global mode exactly when
its potential is +-1 times the global potential at every point.

* TE, local (m, n) = (N, M):  Gamma_loc = c' cos(N pi y'/h) cos(M pi (w - x')/w) = (-1)^M c' cos(M pi x'/w) cos(N pi y'/h).
  The normalization c' = sqrt(eps_m eps_n) / (pi sqrt(m^2 b/a + n^2 a/b)) with a = h, b = w, m = N, n = M equals the
  global c = sqrt(eps_M eps_N) / (pi sqrt(M^2 h/w + N^2 w/h)). Hence e_loc = (-1)^M e_glob.
* TM: Pi_loc = c sin(N pi y'/h) sin(M pi (w - x')/w), and sin(M pi - t) = -(-1)^M sin t, so
  Pi_loc = (-1)^(M+1) Pi_glob and e_loc = (-1)^(M+1) e_glob.

So a_loc = F a_glob with F[loc(N, M), glob(M, N)] = (-1)^M (TE) or (-1)^(M+1) (TM). The forward and backward
amplitudes transform alike: both share e (CONVENTIONS.md section 5), and z_hat x (R e) = R (z_hat x e) for a rotation
R about z.

``overlap_matrix`` evaluates the same matrix by Gauss-Legendre quadrature of ``transverse_field`` (CONVENTIONS.md
requires field values to come from it), as a numerical check of the derivation.
"""
import numpy as np

from wrmodes.modes import Mode, ModeBasis, transverse_field


def local_mode(global_mode):
    """Local E-plane mode corresponding to a global mode (M, N): (N, M), same kind."""
    g = Mode.parse(global_mode)
    return Mode(g.kind, g.n, g.m)


def eplane_sign(global_mode):
    """Sign s with e_loc(N, M) = s e_glob(M, N): (-1)^M for TE, (-1)^(M+1) for TM (derivation above)."""
    g = Mode.parse(global_mode)
    return (-1.0) ** g.m if g.kind == "TE" else (-1.0) ** (g.m + 1)


def eplane_local_basis(global_basis):
    """The E-plane local basis (h x w guide) that spans the same modes as a global (w x h) basis."""
    return ModeBasis(global_basis.b, global_basis.a, [local_mode(m) for m in global_basis.modes])


def eplane_matrix(global_basis, local_basis=None):
    """F with a_local = F a_global, from the analytic derivation (no quadrature, no wrmodes.modes.eplane_frame_signs)."""
    local_basis = eplane_local_basis(global_basis) if local_basis is None else local_basis
    F = np.zeros((len(local_basis), len(global_basis)))
    for j, g in enumerate(global_basis.modes):
        F[local_basis.index(local_mode(g)), j] = eplane_sign(g)
    return F


def overlap_matrix(global_basis, local_basis=None, nq=48):
    """Overlap  int e_loc_i . e_glob_j dx dy  by Gauss-Legendre quadrature of wrmodes.modes.transverse_field.

    The local field components (e_u, e_v) are converted to global ones with u_hat = y_hat, v_hat = -x_hat:
    e_x = -e_v, e_y = e_u. Exact (to rounding) for nq larger than the highest mode index.
    """
    local_basis = eplane_local_basis(global_basis) if local_basis is None else local_basis
    w, h = global_basis.a, global_basis.b
    x, wt = np.polynomial.legendre.leggauss(nq)
    X, Y = np.meshgrid(x * w / 2, x * h / 2, indexing="ij")
    W = np.outer(wt * w / 2, wt * h / 2)
    G = np.zeros((len(local_basis), len(global_basis)))
    glob = [transverse_field(g, w, h, X, Y) for g in global_basis.modes]
    for i, loc in enumerate(local_basis.modes):
        eu, ev = transverse_field(loc, h, w, Y, -X)       # local coordinates of the point (x, y): u = y, v = -x
        lx, ly = -ev, eu
        for j, (ex, ey) in enumerate(glob):
            G[i, j] = np.sum(W * (ex * lx + ey * ly))
    return G


def doubled(F):
    """The transformation of y = [A+; A-]: the same F on both halves."""
    Z = np.zeros_like(F)
    return np.block([[F, Z], [Z, F]])
