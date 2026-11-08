"""Straight section of a guide."""
import numpy as np
from scipy.linalg import expm

from .. import gsm as _gsm
from .base import Component
from .guide import Guide

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

    Wall loss (hook for WP2): `_loss_matrix(basis, f)` returns the full N x N attenuation matrix Lambda of the
    forward amplitudes, dA+/dz = -(i diag(beta) + Lambda) A+, and the GSM is then
    S21 = expm(-(i diag(beta) + Lambda) L), S12 = S21^T, no reflection (Lambda must be symmetric for a reciprocal
    section). Lambda may couple modes, e.g. the degenerate TE/TM pairs. Until WP2 implements it, Lambda = 0 and a
    guide with a conductivity (Guide.sigma not None) raises NotImplementedError.
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
        """Hook for WP2: the N x N attenuation matrix Lambda (1/m) in `basis` at f. Lossless guides only, for now."""
        if self.guide is not None:
            self.guide.require_lossless(self.name)
        return np.zeros((len(basis), len(basis)))

    def gsm(self, f, basis1=None, basis2=None):
        if basis1 is None:
            basis1 = self.default_basis(f)[0]
        if basis2 is not None and basis2 != basis1:
            raise ValueError("a straight section has the same basis at both ports")
        lam = self._loss_matrix(basis1, f)
        if not np.any(lam):
            return _gsm.straight(basis1, f, self.length, self.name)
        P = expm(-(np.diag(1j * basis1.beta(f)) + lam) * self.length)
        Z = np.zeros((len(basis1),) * 2, dtype=complex)
        return _gsm.GSM(Z, P.T, P, Z.copy(), basis1, basis1, f, self.name)
