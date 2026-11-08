"""Explicit changes of the local frame between H-plane and E-plane components (CONVENTIONS.md section 2)."""
import numpy as np

from ..gsm import GSM
from ..modes import transverse_field, eplane_frame_signs
from .base import Component
from .guide import Guide, map_basis, _check_plane

__all__ = ["FrameChange", "frame_overlap"]


def _global_fields(basis, plane, X, Y):
    """Normalized transverse fields (e_x, e_y) of the modes of a local basis, on the global grid (X, Y).

    H frame: u = x, v = y. E frame: u = y, v = -x, so u_hat = y_hat, v_hat = -x_hat and
    (e_x, e_y) = (-e_v, e_u) evaluated at (u, v) = (y, -x).
    """
    a, b = basis.a, basis.b
    ex = np.empty((len(basis),) + X.shape)
    ey = np.empty_like(ex)
    for i, mode in enumerate(basis.modes):
        if plane == "H":
            ex[i], ey[i] = transverse_field(mode, a, b, X, Y)
        else:
            eu, ev = transverse_field(mode, a, b, Y, -X)
            ex[i], ey[i] = -ev, eu
    return ex, ey


def frame_overlap(from_basis, from_plane, to_basis, to_plane, nq=None):
    """Overlap matrix M[p, q] = int e_p^to . e_q^from ds of the normalized mode fields (Gauss-Legendre quadrature).

    Both bases describe the same cross-section in their own local frames. For frames that differ by a multiple
    of 90 degrees M is a signed permutation, and a_to = M a_from (the wave admittances of corresponding modes
    are equal, so field overlap and amplitude map coincide, CONVENTIONS.md section 5).
    """
    w, h = (from_basis.a, from_basis.b) if from_plane == "H" else (from_basis.b, from_basis.a)
    wt, ht = (to_basis.a, to_basis.b) if to_plane == "H" else (to_basis.b, to_basis.a)
    if not (np.isclose(w, wt, rtol=1e-12, atol=0) and np.isclose(h, ht, rtol=1e-12, atol=0)):
        raise ValueError("frame_overlap: the two bases describe different cross-sections")
    if nq is None:
        idx = max([max(m.m, m.n) for m in from_basis.modes + to_basis.modes] + [0])
        nq = 2 * idx + 32          # Gauss-Legendre: exact for the trigonometric products to far below 1e-12
    x, wq = np.polynomial.legendre.leggauss(nq)
    X, Y = np.meshgrid(x * w / 2, x * h / 2, indexing="ij")
    W = np.outer(wq * w / 2, wq * h / 2)
    fx, fy = _global_fields(from_basis, from_plane, X, Y)
    tx, ty = _global_fields(to_basis, to_plane, X, Y)
    return (np.einsum("pij,qij->pq", tx * W, fx) + np.einsum("pij,qij->pq", ty * W, fy))


def _as_signed_permutation(M, tol=1e-12):
    """Round M to the exact signed permutation after checking that it is one to `tol`."""
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"frame change between bases of different sizes {M.shape}")
    R = np.round(M)
    if (np.max(np.abs(M - R)) > tol or not np.all(np.sum(np.abs(R), axis=0) == 1)
            or not np.all(np.sum(np.abs(R), axis=1) == 1)):
        raise ValueError(f"frame overlap is not a signed permutation to {tol:g} "
                         f"(max deviation {np.max(np.abs(M - R)):.2e}); the two bases do not hold the same modes")
    return R


class FrameChange(Component):
    """Change of local frame between the H-plane and E-plane frames of one guide (CONVENTIONS.md section 2).

    A frame change is not a physical object: it is a lossless, reflectionless junction of zero length that
    relabels the modes, a_to = F a_from, with F the overlap of the normalized transverse fields
    (`wrmodes.modes.transverse_field`) computed by quadrature and rounded to the exact signed permutation after
    checking that it is one to 1e-12. From the global (H) frame to the E frame, F is the matrix of
    `wrmodes.modes.eplane_frame_signs`: global (M, N) -> local (N, M) times (-1)^M for TE and (-1)^(M+1) for TM.
    S11 = S22 = 0, S21 = F, S12 = F^T (F is orthogonal, so the GSM is symmetric and unitary).

    Parameters
    ----------
    from_plane, to_plane : 'H' or 'E'
        Frames of port 1 and port 2.
    guide : Guide, optional
        Only needed for `default_basis`; `gsm` takes the cross-section from the basis.
    """

    def __init__(self, from_plane, to_plane, guide=None, name=""):
        self.from_plane = _check_plane(from_plane)
        self.to_plane = _check_plane(to_plane)
        if guide is not None and not isinstance(guide, Guide):
            raise TypeError("guide must be a wrmodes.components.Guide")
        self.guide = guide
        self.name = name or f"frame change {from_plane}->{to_plane}"

    #: None: the two ports have different frames (from_plane, to_plane); `Line` treats frame changes explicitly
    plane = None

    def __repr__(self):
        return f"FrameChange({self.from_plane!r}, {self.to_plane!r})"

    def with_guide(self, guide):
        return type(self)(self.from_plane, self.to_plane, guide, self.name)

    def default_basis(self, f, cutoff_factor=None):
        """Port bases from the component's guide; cutoff_factor=None means 1 (propagating modes only)."""
        cutoff_factor = 1.0 if cutoff_factor is None else cutoff_factor
        if self.guide is None:
            raise ValueError("FrameChange.default_basis needs a guide (FrameChange(..., guide=Guide(w, h)))")
        b1 = self.guide.basis(f, cutoff_factor, plane=self.from_plane)
        return b1, map_basis(b1, self.from_plane, self.to_plane)

    def matrix(self, basis1, basis2=None):
        """The signed permutation F with a(port 2) = F a(port 1); basis2 defaults to the mapped basis1."""
        basis2 = map_basis(basis1, self.from_plane, self.to_plane) if basis2 is None else basis2
        if basis2.frame not in (None, self.to_plane):
            raise ValueError(f"basis2 has frame {basis2.frame!r}, expected {self.to_plane!r}")
        if len(basis1) == 0 or len(basis2) == 0:
            if len(basis1) != len(basis2):
                raise ValueError("frame change between bases of different sizes")
            return np.zeros((0, 0)), basis2
        if self.from_plane == self.to_plane:
            F = np.zeros((len(basis2), len(basis1)))
            for j, m in enumerate(basis1.modes):
                F[basis2.index(m), j] = 1.0
            return F, basis2
        F = _as_signed_permutation(frame_overlap(basis1, self.from_plane, basis2, self.to_plane))
        # cross-check against the closed form of CONVENTIONS.md section 2
        Fc = (eplane_frame_signs(basis1, basis2) if self.from_plane == "H"
              else eplane_frame_signs(basis2, basis1).T)
        if not np.array_equal(F, Fc):
            raise AssertionError("FrameChange: quadrature overlap disagrees with modes.eplane_frame_signs")
        return F, basis2

    def gsm(self, f, basis1=None, basis2=None):
        if basis1 is None:
            basis1, basis2 = self.default_basis(f)
        if basis1.frame not in (None, self.from_plane):
            raise ValueError(f"basis1 has frame {basis1.frame!r}, expected {self.from_plane!r}")
        # a frame change has zero length: it stays lossless in a guide with a wall conductivity
        basis1 = basis1 if basis1.frame is not None else basis1.with_frame(self.from_plane)
        F, basis2 = self.matrix(basis1, basis2)
        Z1, Z2 = np.zeros((len(basis1),) * 2), np.zeros((len(basis2),) * 2)
        return GSM(Z1, F.T, F, Z2, basis1, basis2, f, self.name)
