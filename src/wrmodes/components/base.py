"""Component protocol of wrmodes 1.0 (CONVENTIONS.md section 10)."""
from abc import ABC, abstractmethod


class Component(ABC):
    """A two-port waveguide component: immutable, no frequency-dependent state."""

    name = ""

    @abstractmethod
    def default_basis(self, f, cutoff_factor=1.0):
        """(basis1, basis2): default port bases (wrmodes.modes.ModeBasis) at frequency f (Hz)."""

    @abstractmethod
    def gsm(self, f, basis1=None, basis2=None):
        """wrmodes.gsm.GSM at frequency f (Hz); None bases mean default_basis(f)."""
