"""Straight section of a guide."""
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

    Wall loss: a guide with a conductivity (Guide.sigma not None) raises NotImplementedError until WP2 adds the
    ohmic attenuation; the hook is `_attenuation`.
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
        return Straight(self.length, guide, self.name)

    def default_basis(self, f, cutoff_factor=1.0):
        if self.guide is None:
            raise ValueError("Straight.default_basis needs a guide (Straight(L, guide=Guide(w, h)))")
        b = self.guide.basis(f, cutoff_factor)
        return b, b

    def _attenuation(self, basis, f):
        """Hook for WP2: extra attenuation per mode. Lossless guides only, for now."""
        if self.guide is not None:
            self.guide.require_lossless(self.name)
        return None

    def gsm(self, f, basis1=None, basis2=None):
        if basis1 is None:
            basis1 = self.default_basis(f)[0]
        if basis2 is not None and basis2 != basis1:
            raise ValueError("a straight section has the same basis at both ports")
        self._attenuation(basis1, f)
        return _gsm.straight(basis1, f, self.length, self.name)
