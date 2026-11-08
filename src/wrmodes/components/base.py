"""Component protocol of wrmodes 1.0 (CONVENTIONS.md section 10)."""
from abc import ABC, abstractmethod


class _Undeclared:
    """Marker for a `plane` that a component class has not declared."""

    def __repr__(self):
        return "<undeclared>"


UNDECLARED = _Undeclared()


class Component(ABC):
    """A two-port waveguide component: immutable, no frequency-dependent state.

    Attributes of the protocol (CONVENTIONS.md section 2):

    * ``plane``: the local frame of the component's port bases, 'H' or 'E'; or None for a component that the
      H <-> E frame change leaves unchanged (a straight section). Every component must declare it, as a class or
      instance attribute: `wrmodes.Line` raises TypeError for a component that leaves it `UNDECLARED`.
    * ``guides`` (optional): (input Guide, output Guide) in the global frame, for components that change the
      cross-section. `Line` then calls ``gsm(f, basis1, basis2)`` with basis2 chosen by the line's truncation rule
      in the output guide, mapped to the component's plane.
    """

    name = ""
    plane = UNDECLARED

    @abstractmethod
    def default_basis(self, f, cutoff_factor=None):
        """(basis1, basis2): default port bases (wrmodes.modes.ModeBasis) at frequency f (Hz).

        cutoff_factor=None means the component's own default truncation (e.g. 1 for coupled-mode components, many
        evanescent modes for mode-matching junctions). `Line` does not call default_basis: it passes the line's
        bases to `gsm`."""

    @abstractmethod
    def gsm(self, f, basis1=None, basis2=None):
        """wrmodes.gsm.GSM at frequency f (Hz); None bases mean default_basis(f)."""
