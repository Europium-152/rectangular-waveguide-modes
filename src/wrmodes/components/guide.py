"""The global cross-section of a transmission line, and the mapping of mode bases between local frames.

CONVENTIONS.md section 2: a line has a global guide frame (x, y, z) with the guide's width w along x and height h
along y. A component in the H plane of the global TE1,0 mode uses the local frame u = x, v = y (a = w, b = h); a
component in the E plane uses u = y, v = -x (a = h, b = w), where global mode (M, N) is local mode (N, M).
"""
import numpy as np

from ..modes import Mode, ModeBasis

__all__ = ["Guide", "PLANES", "map_basis"]

PLANES = ("H", "E")


def _check_plane(plane):
    if plane not in PLANES:
        raise ValueError(f"plane must be one of {PLANES}, got {plane!r}")
    return plane


def map_basis(basis, from_plane, to_plane):
    """The basis of the same physical modes in another local frame (CONVENTIONS.md section 2).

    `basis` is a local basis of the `from_plane` frame ('H' or 'E'); its frame tag must be `from_plane` or None.
    Between the H and E frames the sides swap (a <-> b) and mode (m, n) becomes (n, m); the result is in the
    standard order of the new frame and carries the frame tag `to_plane`. Only the mode set is mapped: the signs
    of the amplitudes are those of `FrameChange`.
    """
    _check_plane(from_plane)
    _check_plane(to_plane)
    if basis.frame not in (None, from_plane):
        raise ValueError(f"basis has frame {basis.frame!r}, expected {from_plane!r}")
    if from_plane == to_plane:
        return basis if basis.frame == to_plane else basis.with_frame(to_plane)
    return ModeBasis(basis.b, basis.a, [Mode(m.kind, m.n, m.m) for m in basis.modes], frame=to_plane)


class Guide:
    """Global cross-section of a line: width w along x and height h along y (m), and the wall conductivity.

    Parameters
    ----------
    w, h : float
        Inner width and height (m). For SPARC, WR-90: w = 22.86 mm, h = 10.16 mm; TE1,0 has E along y.
    sigma : float or None
        Wall conductivity (S/m) of all four walls, or None for perfectly conducting walls. There is no default
        material: the SPARC baseline is aluminium, sigma = 3.5e7 S/m (CONVENTIONS.md section 1); copper is 5.8e7.
        Straight sections and smooth bends include the ohmic loss (wrmodes.loss, WP2). Components that do not
        support wall loss (steps and miters in wrmodes 1.0 wave 1) refuse a lossy guide with NotImplementedError
        (`require_lossless`) instead of silently returning the lossless result.

    A Guide is immutable and hashable.
    """

    __slots__ = ("_w", "_h", "_sigma")

    def __init__(self, w, h, sigma=None):
        w, h = float(w), float(h)
        if not (w > 0 and h > 0):
            raise ValueError("w and h must be positive")
        if sigma is not None:
            sigma = float(sigma)
            if not sigma > 0:
                raise ValueError("sigma must be positive (S/m) or None for perfectly conducting walls")
        object.__setattr__(self, "_w", w)
        object.__setattr__(self, "_h", h)
        object.__setattr__(self, "_sigma", sigma)

    def __setattr__(self, key, value):
        raise AttributeError("Guide is immutable")

    def __reduce__(self):
        return (Guide, (self._w, self._h, self._sigma))

    w = property(lambda self: self._w, doc="width along x (m)")
    h = property(lambda self: self._h, doc="height along y (m)")
    sigma = property(lambda self: self._sigma, doc="wall conductivity (S/m), None for perfect walls")

    def __eq__(self, other):
        return (isinstance(other, Guide) and np.isclose(self.w, other.w, rtol=1e-12, atol=0)
                and np.isclose(self.h, other.h, rtol=1e-12, atol=0) and self.sigma == other.sigma)

    def __hash__(self):
        return hash(self.sigma)

    def __repr__(self):
        s = "" if self.sigma is None else f", sigma={self.sigma:g}"
        return f"Guide(w={self.w:g}, h={self.h:g}{s})"

    @property
    def lossless(self):
        return self.sigma is None

    def require_lossless(self, what="this component"):
        """Raise NotImplementedError if the guide has a wall conductivity (for components without wall loss)."""
        if self.sigma is not None:
            raise NotImplementedError(f"{what}: ohmic wall loss (Guide.sigma = {self.sigma:g} S/m) is not supported "
                                      f"by this component in wrmodes 1.0 (lossless walls only); use a guide with "
                                      f"sigma=None for it")

    def wall_impedance(self, f):
        """Surface impedance Z_s = (1 + i) R_s (ohm) of the walls at f (Hz), or None for perfect walls."""
        if self.sigma is None:
            return None
        from ..loss import surface_impedance
        return complex(surface_impedance(f, self.sigma))

    def dims(self, plane="H"):
        """Local sides (a, b) of the cross-section in the frame of `plane` (CONVENTIONS.md section 2)."""
        return (self.w, self.h) if _check_plane(plane) == "H" else (self.h, self.w)

    def basis(self, f, cutoff_factor=1.0, plane="H", evanescent_families="live"):
        """The line's basis at f (Hz), expressed in the local frame of `plane`.

        The basis is chosen once in the global frame (frame tag 'H'): the propagating modes of all families, plus
        the evanescent modes with cutoff below cutoff_factor * f selected by `evanescent_families`, and then mapped
        with `map_basis`, so that every frame describes the same physical modes.

        evanescent_families
            'live' (default): keep an evanescent global mode (M, N) if a propagating mode has the same N (a live
            family of the H frame) or the same M (a live family of the E frame). Components that conserve the local
            family index, in either plane, then see every evanescent mode that can couple to a propagating mode,
            and the rule is symmetric under the H <-> E exchange (in a square guide, rotated lines get the same
            modes).
            'propagating': only the live families N of the global (H) frame (`ModeBasis.propagating`).
            'all': every mode below cutoff_factor * f.
        """
        if evanescent_families not in ("live", "propagating", "all"):
            raise ValueError("evanescent_families must be 'live', 'propagating' or 'all'")
        if cutoff_factor < 1:
            raise ValueError("cutoff_factor must be >= 1 (the basis must contain the propagating modes)")
        if evanescent_families == "live":
            full = ModeBasis.below(cutoff_factor * f, self.w, self.h, frame="H")
            prop = full.propagating_mask(f) if len(full) else np.zeros(0, bool)
            live_n = {m.n for m, p in zip(full.modes, prop) if p}
            live_m = {m.m for m, p in zip(full.modes, prop) if p}
            g = full.subset([i for i, (m, p) in enumerate(zip(full.modes, prop))
                             if p or m.n in live_n or m.m in live_m])
        else:
            g = ModeBasis.propagating(f, self.w, self.h, cutoff_factor=cutoff_factor, frame="H",
                                      evanescent_families=evanescent_families)
        return map_basis(g, "H", plane)
