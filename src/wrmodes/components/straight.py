"""Straight section of a guide."""
import numpy as np

from .. import gsm as _gsm
from .base import Component
from .guide import Guide
from ..loss import loss_matrix_full, straight_gsm_from_loss_matrix

__all__ = ["Straight"]


class Straight(Component):
    """Straight section of length `length` (m): S21 = S12 = diag(exp(-i beta L)), no reflection (`wrmodes.gsm.straight`).

    A straight section is the same in every local frame (`plane = None`): in a `Line` it takes the frame of the
    component before it, so that no frame change is needed around it. Evanescent modes in the basis decay as
    exp(-|beta| L).

    Parameters
    ----------
    length : float
        Length (m), >= 0.
    guide : Guide, optional
        Cross-section; needed for `default_basis` only. A `Line` attaches its own guide.

    Wall loss: in a guide with a conductivity (Guide.sigma, e.g. aluminium 3.5e7 S/m) the section follows
    docs/theory/T1-wall-impedance.md, box B4 with kappa = 0: dy/dz = (D - Lambda_full) y for y = [A+; A-], with
    `_loss_matrix(basis, f)` = Lambda_full = [[Lambda^+, Lambda^-], [-Lambda^-, -Lambda^+]] (2N x 2N, the same
    surface impedance (1 + i) R_s on all four walls). The propagating modes are coupled through the full matrix
    (this keeps the resonant coupling of the degenerate TE_mn/TM_mn pairs, which converts 3-34 % of a TE_mn into
    TM_mn over 20 m of aluminium WR-90, and the tiny reflection through Lambda^-); each evanescent mode only gets
    its own attenuation Lambda^+_pp (`wrmodes.loss.coupling_mask`). The GSM is computed without growing
    exponentials (`wrmodes.loss.straight_gsm_from_loss_matrix`). Lossless guides (sigma=None) give exactly
    `wrmodes.gsm.straight`.

    With equal walls the section is the same in both local frames, so `plane` stays None (checked in
    tests/test_loss.py, degenerate coupling included). Broad-wall loss couples families n and n +- 2 (non
    synchronously); the section keeps that coupling, since it uses the whole basis.
    """

    plane = None

    def __init__(self, length, guide=None, name=""):
        length = float(length)
        if length < 0:
            raise ValueError("length must be >= 0")
        if guide is not None and not isinstance(guide, Guide):
            raise TypeError("guide must be a wrmodes.components.Guide")
        self.length, self.guide = length, guide
        self.name = name or f"straight {length:g} m"

    def __repr__(self):
        return f"Straight({self.length!r})"

    def with_guide(self, guide):
        return type(self)(self.length, guide, self.name)

    def default_basis(self, f, cutoff_factor=None):
        """Port bases from the component's guide; cutoff_factor=None means 1 (propagating modes only)."""
        cutoff_factor = 1.0 if cutoff_factor is None else cutoff_factor
        if self.guide is None:
            raise ValueError("Straight.default_basis needs a guide (Straight(L, guide=Guide(w, h)))")
        b = self.guide.basis(f, cutoff_factor)
        return b, b

    def _loss_matrix(self, basis, f):
        """Lambda_full (2N x 2N, 1/m) of the modes of `basis` at f: zero for perfect walls (T1 box B4)."""
        n = len(basis)
        Zs = None if self.guide is None else self.guide.wall_impedance(f)
        if Zs is None or n == 0:
            return np.zeros((2 * n, 2 * n))
        return loss_matrix_full(basis, f, Zs)

    def gsm(self, f, basis1=None, basis2=None):
        if basis1 is None:
            basis1 = self.default_basis(f)[0]
        if basis2 is not None and basis2 != basis1:
            raise ValueError("a straight section has the same basis at both ports")
        lam = self._loss_matrix(basis1, f)
        if not np.any(lam):
            return _gsm.straight(basis1, f, self.length, self.name)
        return straight_gsm_from_loss_matrix(basis1, f, self.length, lam, self.name)
