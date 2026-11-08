"""Smooth bends of constant cross-section, computed with the coupled-mode transfer-matrix solver of wrmodes 0.1."""
import numpy as np

from ..gsm import straight, from_transfer_matrix, block_diag
from ..transfer_matrix import solve_transfer_matrix, coupled_mode_matrices
from .base import Component
from .guide import Guide, _check_plane

__all__ = ["SmoothBend", "SechCurvature", "ConstantCurvature"]


class SechCurvature:
    """Hyperbolic-secant curvature profile of total angle `theta_deg` over `length` (m), truncated at +-T.

    kappa(z) = 1 / (R0 cosh(gamma (z - L/2))), gamma = 2T/L, R0 = 4 arctan(tanh(T/2)) / (theta gamma)
    (JINST paper, eqs. 4.6-4.7; the same formula as `benchmarks/paper_figures.py`). The integral of kappa over
    [0, L] is theta. `sign=-1` gives the bend turning the other way. Picklable (for `Line.sweep`).
    """

    def __init__(self, theta_deg, length, T=4.0, sign=1.0):
        self.theta_deg, self.length, self.T, self.sign = float(theta_deg), float(length), float(T), float(sign)
        theta = np.radians(self.theta_deg)
        self.gamma = 2 * self.T / self.length
        self.R0 = 4 / theta / self.gamma * np.arctan(np.tanh(self.T / 2))

    def __call__(self, z):
        return self.sign / (self.R0 * np.cosh(self.gamma * (np.asarray(z) - self.length / 2)))

    def __neg__(self):
        return SechCurvature(self.theta_deg, self.length, self.T, -self.sign)

    def __repr__(self):
        s = "-" if self.sign < 0 else ""
        return f"{s}SechCurvature({self.theta_deg:g} deg, {self.length:g} m, T={self.T:g})"


class ConstantCurvature:
    """Constant curvature kappa (1/m). Picklable (for `Line.sweep`)."""

    def __init__(self, kappa):
        self.kappa = float(kappa)

    @classmethod
    def from_angle(cls, theta_deg, length):
        return cls(np.radians(theta_deg) / length)

    def __call__(self, z):
        return np.full(np.shape(z), self.kappa)

    def __neg__(self):
        return ConstantCurvature(-self.kappa)

    def __repr__(self):
        return f"ConstantCurvature({self.kappa:g} 1/m)"


class SmoothBend(Component):
    """Smooth coplanar bend of constant cross-section, by the coupled-mode equations of the JINST paper.

    Parameters
    ----------
    length : float
        Length of the centre line (m).
    curvature : float or callable
        Curvature kappa(z) (1/m) for z in [0, length], a constant or a function that accepts NumPy arrays. A
        positive kappa turns the guide towards -u (CONVENTIONS.md section 2); a bend turning the other way in the
        same plane has a negative kappa.
    plane : 'H' or 'E'
        Plane of the bend with respect to the global frame of the line: 'H' bends side w (u = x, a = w, b = h),
        'E' bends side h (u = y, v = -x, a = h, b = w). The component's bases are local bases of this frame
        (frame tag `plane`, or no tag); `Line` inserts the frame changes.
    breakpoints : array_like, optional
        Positions (m) of curvature discontinuities, passed to `solve_transfer_matrix`.
    atol : float
        Accuracy of the complex entries of T and R of every family (`solve_transfer_matrix` with
        error_control='full').
    guide : Guide, optional
        Global cross-section; needed for `default_basis` only. A `Line` attaches its own guide.
    max_slices : int
        Upper limit of the adaptive slicing of `solve_transfer_matrix`.

    Method
    ------
    The bend does not vary along v, so it only couples modes of the same local family n (CONVENTIONS.md section 3).
    For each family, the propagating modes of the basis are coupled by the curvature: their 2N x 2N transfer matrix
    from `solve_transfer_matrix(error_control='full')` is converted with `from_transfer_matrix`. A family whose
    coupling matrix vanishes (e.g. TE1,0 alone, below the cutoff of TE2,0) is a straight section.

    **Evanescent modes** in the basis (needed next to miters and steps) are not coupled: they are propagated as
    uncoupled straight modes, exp(-i beta L) = exp(-|beta| L), over the centre-line length, with no reflection.
    This is an approximation. It is exact in the straight limit (kappa = 0); in a bend it neglects the coupling of
    evanescent modes to the propagating ones and to each other. Over a smooth bend the coupling rate (of order
    kappa) is small compared with the decay rate |beta| of an evanescent mode, so an evanescent mode excited at
    one end (by a neighbouring miter or step) has decayed long before the curvature can convert it. The result
    for the propagating modes does not depend on whether evanescent modes are included (they are separate blocks).

    The families are combined with `block_diag` in exactly the order of the given basis.
    """

    def __init__(self, length, curvature, plane="H", breakpoints=None, atol=1e-6, guide=None, max_slices=4096,
                 name=""):
        length = float(length)
        if not length > 0:
            raise ValueError("length must be positive")
        if guide is not None and not isinstance(guide, Guide):
            raise TypeError("guide must be a wrmodes.components.Guide")
        self.length = length
        self.curvature = curvature
        self.plane = _check_plane(plane)
        self.breakpoints = None if breakpoints is None else tuple(float(p) for p in np.atleast_1d(breakpoints))
        self.atol = float(atol)
        self.guide = guide
        self.max_slices = int(max_slices)
        self.name = name or f"{plane}-plane bend {length:g} m"

    def __repr__(self):
        return f"SmoothBend({self.length!r}, {self.curvature!r}, plane={self.plane!r})"

    def with_guide(self, guide):
        return SmoothBend(self.length, self.curvature, self.plane, self.breakpoints, self.atol, guide,
                          self.max_slices, self.name)

    @property
    def curvature_function(self):
        """kappa(z) as a callable."""
        if callable(self.curvature):
            return self.curvature
        return ConstantCurvature(self.curvature)

    def default_basis(self, f, cutoff_factor=1.0):
        if self.guide is None:
            raise ValueError("SmoothBend.default_basis needs a guide (SmoothBend(..., guide=Guide(w, h)))")
        b = self.guide.basis(f, cutoff_factor, plane=self.plane)
        return b, b

    def _wall_loss(self, basis, f):
        """Hook for WP2 (ohmic loss in bends). Lossless guides only, for now."""
        if self.guide is not None:
            self.guide.require_lossless(self.name)

    def gsm(self, f, basis1=None, basis2=None):
        """GSM at f (Hz) in the local basis of the bend's plane (same basis at both ports).

        Each GSM carries `info`, a dict family n -> solver diagnostics (n_slices, error_estimate, success), with
        None for families that are straight sections.
        """
        if basis1 is None:
            basis1 = self.default_basis(f)[0]
        if basis2 is not None and basis2 != basis1:
            raise ValueError("a smooth bend has the same basis at both ports")
        if basis1.frame not in (None, self.plane):
            raise ValueError(f"{self.name}: basis has frame {basis1.frame!r}, expected {self.plane!r} "
                             f"(insert a FrameChange, or use a Line)")
        self._wall_loss(basis1, f)
        a, b, L = basis1.a, basis1.b, self.length
        kappa = self.curvature_function
        prop = basis1.propagating_mask(f)
        parts, info = [], {}
        for n in basis1.families():
            idx = basis1.family(n)
            pidx = [i for i in idx if prop[i]]
            eidx = [i for i in idx if not prop[i]]
            info[n] = None
            if pidx:
                sub = basis1.subset(pidx)
                modes = sub.legacy_modes(f)
                _, K, _ = coupled_mode_matrices(f, a, b, modes)
                if not np.any(K):
                    parts.append(straight(sub, f, L))
                else:
                    m0 = sub.modes[0]
                    _, res = solve_transfer_matrix(f, a, b, (m0.kind, m0.m, m0.n), L, kappa, coupled_modes=modes,
                                                   atol=self.atol, error_control="full", breakpoints=self.breakpoints,
                                                   max_slices=self.max_slices, profile=False, blas_threads=1)
                    info[n] = dict(n_slices=res.n_slices, error_estimate=float(res.error_estimate),
                                   success=bool(res.success))
                    parts.append(from_transfer_matrix(res.transfer_matrix, sub, f))
            if eidx:
                parts.append(straight(basis1.subset(eidx), f, L))     # evanescent: uncoupled straight modes
        out = block_diag(*parts, basis1=basis1, basis2=basis1, name=self.name)
        out.info = info
        return out
