"""Smooth bends of constant cross-section, computed with the coupled-mode transfer-matrix solver of wrmodes 0.1."""
import numpy as np

from ..gsm import GSM, straight, from_transfer_matrix, block_diag
from ..transfer_matrix import solve_transfer_matrix, coupled_mode_matrices
from .base import Component
from .guide import Guide, _check_plane
from ..loss import bend_loss_matrices, straight_gsm_from_loss_matrix

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
    families : 'separate' or 'coupled'
        Wall loss only (lossless bends always treat the families separately, which is exact for them).
        'separate' (default): one coupled-mode system per local family n, as for a lossless bend; 'coupled': one
        system with all propagating modes of even n and one with those of odd n (the walls are equal, so Lambda
        does not couple n of different parity), which keeps the coupling of the families by the lossy walls
        v = +-b/2 (see Wall loss). 'coupled' costs 5-15 times more at 90 GHz (34-37 s per SPARC bend). Use it
        for multimode method-vs-method validation (CONVENTIONS.md section 8).

    Method
    ------
    The bend does not vary along v, so it only couples modes of the same local family n (CONVENTIONS.md section 3).
    For each family, the propagating modes of the basis are coupled by the curvature: their 2N x 2N transfer matrix
    from `solve_transfer_matrix(error_control='full')` is converted with `from_transfer_matrix`. A family whose
    coupling matrices vanish (e.g. TE1,0 alone, below the cutoff of TE2,0) is a straight section, lossy if the
    guide is.

    Wall loss (guide with a conductivity sigma; docs/theory/T1-wall-impedance.md, box B4b): the system of every
    family is dy/dz = [(D - Lambda0_full) + kappa(z) (K - Lambda1_full)] y, with the surface impedance (1 + i) R_s
    on all four walls (`wrmodes.loss.bend_loss_matrices`, passed to `solve_transfer_matrix(loss_matrix=...)`).
    The kappa Lambda1 term (the walls of a bend are not equally long) is kept: it changes the converted powers of
    the SPARC bends by up to 1.3e-4. The degenerate TE_mn/TM_mn coupling is kept (same family). Evanescent modes
    get only their own Lambda^+_pp (`wrmodes.loss.coupling_mask`). With families='separate' the coupling of
    different local families by the lossy walls v = +-b/2 (n with n +- 2; through kappa Lambda1 also m with m +- odd)
    is neglected, while a lossy `Straight` keeps it. This coupling is non-synchronous, but summed over the many
    propagating modes it carries a genuine O(ka Z/eta0) correction of the attenuation (T1 section 5; e.g. +1.4e-3
    of alpha for TE1,0 from the broad walls at 90 GHz, which brings it within 3e-4 of the exact Leontovich value).
    Measured for the SPARC bends in aluminium WR-90 (40 cm 90 deg sech, 60 cm 60 deg constant, H and E plane),
    'separate' vs 'coupled', against the targets of CONVENTIONS.md section 8 (1e-4 of the input power for every
    converted mode): for TE1,0 input the default is within 1e-4 (40 cm 90 deg sech H-plane bend at 90 GHz: largest
    change in the TE1,0 column 1.8e-5, i.e. 1.3e-4 dB on its own transmission, and at most 2.4e-5 for the 60 cm
    bend; below 1e-7 in the E plane). The largest
    change of any |S_ij|^2 of the propagating block, 1.0-1.3e-4 at 90 GHz (6.3e-5 at 60 GHz, 4e-6 at 30 GHz), occurs
    for near-cutoff modes, where it is below the error of the first-order model itself (about 2.8e-4, T1 section
    5). In an E-plane bend the local walls v = +-b/2 are the narrow walls of the guide.

    **Evanescent modes** in the basis (needed next to miters and steps) are not coupled: they are propagated as
    uncoupled straight modes, exp(-i beta L) = exp(-|beta| L), over the centre-line length, with no reflection.
    This is an approximation. It is exact in the straight limit (kappa = 0); in a bend it neglects the coupling of
    evanescent modes to the propagating ones and to each other. It is good when, for every evanescent mode e,
    |beta_e| is large compared with the curvature coupling (of order kappa |K_pe|) and |beta_e| L >> 1, so that an
    evanescent mode excited at one end (by a neighbouring miter or step) has decayed before the curvature can
    convert it. Both conditions fail for modes just below cutoff (|beta_e| -> 0). The error of this approximation
    next to a miter or a step has not been quantified. The result for the propagating modes does not depend on
    whether evanescent modes are included (they are separate blocks). Local families without any propagating mode
    (which the line's basis contains for E-plane components) are treated in the same way.

    The families are combined with `block_diag` in exactly the order of the given basis.
    """

    def __init__(self, length, curvature, plane="H", breakpoints=None, atol=1e-6, guide=None, max_slices=4096,
                 name="", families="separate"):
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
        if families not in ("separate", "coupled"):
            raise ValueError("families must be 'separate' or 'coupled'")
        self.families = families
        self.name = name or f"{plane}-plane bend {length:g} m"

    def __repr__(self):
        return f"SmoothBend({self.length!r}, {self.curvature!r}, plane={self.plane!r})"

    def with_guide(self, guide):
        return type(self)(self.length, self.curvature, self.plane, self.breakpoints, self.atol, guide,
                          self.max_slices, self.name, self.families)

    @property
    def curvature_function(self):
        """kappa(z) as a callable."""
        if callable(self.curvature):
            return self.curvature
        return ConstantCurvature(self.curvature)

    def default_basis(self, f, cutoff_factor=None):
        """Port bases from the component's guide; cutoff_factor=None means 1 (propagating modes only)."""
        cutoff_factor = 1.0 if cutoff_factor is None else cutoff_factor
        if self.guide is None:
            raise ValueError("SmoothBend.default_basis needs a guide (SmoothBend(..., guide=Guide(w, h)))")
        b = self.guide.basis(f, cutoff_factor, plane=self.plane)
        return b, b

    def _loss_matrix(self, basis, f):
        """(Lambda0_full, Lambda1_full), 2N x 2N each (T1 box B4b), or None for perfect walls."""
        Zs = None if self.guide is None else self.guide.wall_impedance(f)
        if Zs is None or len(basis) == 0:
            return None
        return bend_loss_matrices(basis, f, Zs)

    def _solve(self, sub, f, lam):
        """GSM of the propagating modes `sub` (one family, or all with families='coupled'); lam = sub-blocks or None."""
        a, b, L = sub.a, sub.b, self.length
        modes = sub.legacy_modes(f)
        _, K, _ = coupled_mode_matrices(f, a, b, modes)
        if not np.any(K) and (lam is None or not np.any(lam[1])):
            g = straight(sub, f, L) if lam is None else straight_gsm_from_loss_matrix(sub, f, L, lam[0])
            return g, None
        m0 = sub.modes[0]
        _, res = solve_transfer_matrix(f, a, b, (m0.kind, m0.m, m0.n), L, self.curvature_function,
                                       coupled_modes=modes, atol=self.atol, error_control="full",
                                       breakpoints=self.breakpoints, max_slices=self.max_slices, profile=False,
                                       blas_threads=1, loss_matrix=lam)
        info = dict(n_slices=res.n_slices, error_estimate=float(res.error_estimate), success=bool(res.success))
        return from_transfer_matrix(res.transfer_matrix, sub, f), info

    def gsm(self, f, basis1=None, basis2=None):
        """GSM at f (Hz) in the local basis of the bend's plane (same basis at both ports).

        Each GSM carries `info`, a dict family n -> solver diagnostics (n_slices, error_estimate, success), with
        None for families that are straight sections (keys 'even' and 'odd', the parity of n, with
        families='coupled' in a lossy guide).
        """
        if basis1 is None:
            basis1 = self.default_basis(f)[0]
        if basis2 is not None and basis2 != basis1:
            raise ValueError("a smooth bend has the same basis at both ports")
        if basis1.frame not in (None, self.plane):
            raise ValueError(f"{self.name}: basis has frame {basis1.frame!r}, expected {self.plane!r} "
                             f"(insert a FrameChange, or use a Line)")
        if len(basis1) == 0:                    # below the first cutoff: nothing propagates, nothing to compute
            Z = np.zeros((0, 0))
            out = GSM(Z, Z, Z, Z, basis1, basis1, f, self.name)
            out.info = {}
            return out
        lossm = self._loss_matrix(basis1, f)
        N = len(basis1)

        def blocks(idx):
            if lossm is None:
                return None
            sel = np.concatenate([idx, np.asarray(idx) + N])
            return lossm[0][np.ix_(sel, sel)], lossm[1][np.ix_(sel, sel)]

        prop = basis1.propagating_mask(f)
        parts, info = [], {}
        if lossm is not None and self.families == "coupled":
            # equal walls: Lambda0 and Lambda1 vanish exactly between even and odd n (and K couples equal n only),
            # so the coupled system splits into the even-n and the odd-n modes
            groups = [(key, [i for i, md in enumerate(basis1.modes) if md.n % 2 == par])
                      for key, par in (("even", 0), ("odd", 1))]
            groups = [g for g in groups if g[1]]
        else:
            groups = [(n, basis1.family(n)) for n in basis1.families()]
        for key, idx in groups:
            pidx = [i for i in idx if prop[i]]
            eidx = [i for i in idx if not prop[i]]
            info[key] = None
            if pidx:
                g, info[key] = self._solve(basis1.subset(pidx), f, blocks(pidx))
                parts.append(g)
            if eidx:                               # evanescent: uncoupled straight modes (+ their own Lambda_pp)
                sub = basis1.subset(eidx)
                parts.append(straight(sub, f, self.length) if lossm is None
                             else straight_gsm_from_loss_matrix(sub, f, self.length, blocks(eidx)[0]))
        out = block_diag(*parts, basis1=basis1, basis2=basis1, name=self.name)
        out.info = info
        return out
