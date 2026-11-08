"""Step junction between two rectangular guides, one cross-section inside the other (mode matching)."""
import numpy as np
import scipy.constants as cst

from ..components import Component
from ..modes import Mode, ModeBasis
from .overlaps import overlap_matrix
from .junction import junction_gsm

__all__ = ["Step", "convergence_study"]

_REL = 1e-12


def _same(x, y):
    return abs(x - y) <= _REL * max(abs(x), abs(y))


def _modes_below(f_max, a, b, ms=None, ns=None):
    """Modes of an a x b guide with cutoff below f_max (the test of ModeBasis.below), optionally restricted to the
    indices m in `ms` and n in `ns`; vectorized, for large truncations."""
    chi_max = 2 * np.pi * f_max / cst.c
    m = np.arange(int(chi_max * a / np.pi) + 2) if ms is None else np.asarray(ms, dtype=int)
    n = np.arange(int(chi_max * b / np.pi) + 2) if ns is None else np.asarray(ns, dtype=int)
    M, N = np.meshgrid(m, n, indexing="ij")
    below = np.pi * np.hypot(M / a, N / b) < chi_max
    out = []
    for mm, nn in zip(M[below], N[below]):
        if mm + nn > 0:
            out.append(Mode("TE", int(mm), int(nn)))
        if mm > 0 and nn > 0:
            out.append(Mode("TM", int(mm), int(nn)))
    return out


class Step(Component):
    """Zero-length junction from guide 1 (a1 x b1) to guide 2 (a2 x b2), by mode matching.

    One cross-section must contain the other: either a1 >= a2 and b1 >= b2 (reduction, large to small) or
    a1 <= a2 and b1 <= b2 (expansion, small to large). The centre of the smaller guide is at (du, dv) in the local
    frame (u, v) of the larger one (CONVENTIONS.md section 2; side a along u, side b along v).

    Both reference planes are the junction plane (CONVENTIONS.md section 6). Each port has the ModeBasis of its own
    cross-section; both carry the same frame tag `frame`. The GSM is symmetric by construction and its propagating
    block is unitary at every truncation (see ``wrmodes.modematch.junction``).

    Mode coupling: a step that keeps b (then dv = 0) only couples modes of the same family n; a step that keeps a
    with du = 0 only couples modes of the same m. Steps in b, or with dv != 0, couple families, and their bases
    keep every family.

    Parameters
    ----------
    a1, b1, a2, b2 : float
        Cross-sections (m) of guide 1 (port 1) and guide 2 (port 2).
    du, dv : float
        Offset (m) of the centre of the smaller guide in the larger guide's frame.
    frame : optional
        Frame tag of both port bases (CONVENTIONS.md section 2), e.g. 'H' or 'E'.
    cutoff_factor : float, optional
        Truncation used by `default_basis` when it is called without one. Both ports keep the modes with cutoff
        frequency below cutoff_factor * f (the same cutoff on both sides, so that the numbers of modes along each
        side are in proportion to the dimensions). If None, the rule of `default_cutoff_frequency` is used.
    """

    name = "step"
    #: default truncation of steps in one dimension: half-periods across the smallest feature (see
    #: `default_cutoff_frequency`)
    MIN_INDEX_1D = 20
    #: default truncation of steps in both dimensions
    MIN_INDEX_2D = 16
    #: default lower bound on cutoff_factor
    MIN_CUTOFF_FACTOR = 3.0

    def __init__(self, a1, b1, a2, b2, du=0.0, dv=0.0, frame=None, cutoff_factor=None, name=""):
        self.a1, self.b1, self.a2, self.b2 = float(a1), float(b1), float(a2), float(b2)
        self.du, self.dv = float(du), float(dv)
        self.frame = frame
        self.cutoff_factor = cutoff_factor
        if name:
            self.name = name
        if min(self.a1, self.b1, self.a2, self.b2) <= 0:
            raise ValueError("guide dimensions must be positive")
        grow_a = self.a2 >= self.a1 or _same(self.a1, self.a2)
        grow_b = self.b2 >= self.b1 or _same(self.b1, self.b2)
        shrink_a = self.a2 <= self.a1 or _same(self.a1, self.a2)
        shrink_b = self.b2 <= self.b1 or _same(self.b1, self.b2)
        if grow_a and grow_b:
            self.large_port = 2
        elif shrink_a and shrink_b:
            self.large_port = 1
        else:
            raise ValueError(f"neither cross-section contains the other ({a1:g} x {b1:g} vs {a2:g} x {b2:g})")
        (aL, bL), (aS, bS) = self._large(), self._small()
        tol = 1e-12 * max(aL, bL)
        if abs(self.du) > 0.5 * (aL - aS) + tol or abs(self.dv) > 0.5 * (bL - bS) + tol:
            raise ValueError(f"offset ({du:g}, {dv:g}) puts the {aS:g} x {bS:g} guide outside the {aL:g} x {bL:g} one")

    # ------------------------------------------------------------------------------------- geometry
    def _large(self):
        return (self.a1, self.b1) if self.large_port == 1 else (self.a2, self.b2)

    def _small(self):
        return (self.a2, self.b2) if self.large_port == 1 else (self.a1, self.b1)

    @property
    def conserves_n(self):
        """True when b is unchanged (then dv = 0): only modes of the same family n couple."""
        return _same(self.b1, self.b2)

    @property
    def conserves_m(self):
        """True when a is unchanged and du = 0: only modes with the same m couple."""
        return _same(self.a1, self.a2)

    def __repr__(self):
        off = f", du={self.du:g}, dv={self.dv:g}" if (self.du or self.dv) else ""
        return f"Step({self.a1:g} x {self.b1:g} -> {self.a2:g} x {self.b2:g}{off})"

    # ------------------------------------------------------------------------------------- bases
    def default_cutoff_frequency(self, f):
        """Default truncation (Hz): max(MIN_CUTOFF_FACTOR f, M c / (2 d)), i.e. M half-periods across d.

        * Step in one dimension (a or b): M = MIN_INDEX_1D and d is the smallest feature along that dimension:
          the side of the smaller guide or the smallest non-zero gap between its walls and the larger guide's
          (bounded below by a quarter of the side), so that the edge fields of nearly flush and of strong steps
          are both resolved.
        * Step in both dimensions: M = MIN_INDEX_2D and d is the smaller side of the smaller guide (the number of
          modes grows with the square of M).

        The resulting convergence is documented in docs/reports/WP4.md; use `convergence_study` to check a case.
        """
        (aL, bL), (aS, bS) = self._large(), self._small()
        changed = [(s, l, d) for s, l, d in ((aS, aL, self.du), (bS, bL, self.dv)) if not _same(s, l)]
        if not changed:
            return self.MIN_CUTOFF_FACTOR * f
        if len(changed) == 1:
            s, l, d = changed[0]
            gaps = [g for g in (0.5 * (l - s) + d, 0.5 * (l - s) - d) if g > _REL * l]
            feature = max(min([s] + gaps), 0.25 * s)
            f_index = self.MIN_INDEX_1D * cst.c / (2 * feature)
        else:
            f_index = self.MIN_INDEX_2D * cst.c / (2 * min(aS, bS))
        return max(self.MIN_CUTOFF_FACTOR * f, f_index)

    def default_basis(self, f, cutoff_factor=None):
        """(basis1, basis2) at f (Hz): the modes of each guide with cutoff below cutoff_factor * f.

        cutoff_factor=None uses the Step's own `cutoff_factor`, or `default_cutoff_frequency(f)` if that is None.
        Modes that cannot couple to any propagating mode are left out: when the step conserves n (or m), only the
        families n (or the indices m) that have a propagating mode in either guide are kept. Steps that do not
        conserve n keep every family.
        """
        if cutoff_factor is None:
            cutoff_factor = self.cutoff_factor
        f_max = self.default_cutoff_frequency(f) if cutoff_factor is None else cutoff_factor * f
        if f_max < f:
            raise ValueError("cutoff_factor must be >= 1 (the bases must contain the propagating modes)")
        prop = [_modes_below(f, a, b) for a, b in ((self.a1, self.b1), (self.a2, self.b2))]
        keep_m = sorted({md.m for pr in prop for md in pr}) if self.conserves_m else None
        keep_n = sorted({md.n for pr in prop for md in pr}) if self.conserves_n else None
        return tuple(ModeBasis(a, b, _modes_below(f_max, a, b, keep_m, keep_n), frame=self.frame)
                     for a, b in ((self.a1, self.b1), (self.a2, self.b2)))

    # ------------------------------------------------------------------------------------- GSM
    def overlaps(self, basis1, basis2):
        """Overlap matrix X (small x large) for the given port bases."""
        bL, bS = (basis1, basis2) if self.large_port == 1 else (basis2, basis1)
        return overlap_matrix(bS, bL, self.du, self.dv)

    def gsm(self, f, basis1=None, basis2=None):
        """GSM (wrmodes.gsm.GSM) of the step at f (Hz); None bases mean `default_basis(f)`.

        The bases must have the cross-sections of guides 1 and 2 and the same frame tag.
        """
        if basis1 is None or basis2 is None:
            d1, d2 = self.default_basis(f)
            basis1 = d1 if basis1 is None else basis1
            basis2 = d2 if basis2 is None else basis2
        for bb, a, b, port in ((basis1, self.a1, self.b1, 1), (basis2, self.a2, self.b2, 2)):
            if not (_same(bb.a, a) and _same(bb.b, b)):
                raise ValueError(f"basis{port} is for a {bb.a:g} x {bb.b:g} guide, port {port} is {a:g} x {b:g}")
        if basis1.frame != basis2.frame:
            raise ValueError(f"both ports of a step use the same frame tag (got {basis1.frame!r}, {basis2.frame!r})")
        bL, bS = (basis1, basis2) if self.large_port == 1 else (basis2, basis1)
        X = overlap_matrix(bS, bL, self.du, self.dv)
        return junction_gsm(bL, bS, X, f, large_port=self.large_port, name=self.name if self.name != "step" else repr(self))

    def convergence_study(self, f, cutoff_factors):
        """See :func:`convergence_study`."""
        return convergence_study(self, f, cutoff_factors)


def convergence_study(component, f, cutoff_factors):
    """Convergence of the propagating block of a component's GSM with the number of evanescent modes.

    Parameters
    ----------
    component : Component
        Any component whose `default_basis(f, cutoff_factor)` adds evanescent modes as cutoff_factor grows.
    f : float
        Frequency (Hz).
    cutoff_factors : sequence of float
        Truncations to compute, in increasing order.

    Returns
    -------
    dict with
        'cutoff_factor' : (K,) array
        'n_modes' : (K, 2) int array, numbers of modes of port 1 and port 2
        'modes1', 'modes2' : names of the propagating modes of port 1 and port 2
        'S' : (K, P, P) complex array, propagating block of S (ports 1 then 2) at each truncation
        'change' : (K,) array, max |(|S_k| - |S_{k-1}|)| over the propagating block (nan for k = 0)
        'unitarity', 'reciprocity' : (K,) arrays, defects of each GSM
    """
    cfs = np.asarray(cutoff_factors, dtype=float)
    S_list, n_modes, uni, rec = [], [], [], []
    names = None
    for cf in cfs:
        b1, b2 = component.default_basis(f, cutoff_factor=cf)
        g = component.gsm(f, b1, b2)
        m1, m2 = g.propagating()
        nm = ([md.name for md, p in zip(b1.modes, m1) if p], [md.name for md, p in zip(b2.modes, m2) if p])
        if names is not None and nm != names:
            raise RuntimeError("the propagating modes changed between truncations")
        names = nm
        keep = np.concatenate([m1, m2])
        S_list.append(g.S[np.ix_(keep, keep)])
        n_modes.append((len(b1), len(b2)))
        uni.append(g.unitarity_defect())
        rec.append(g.reciprocity_defect() / np.max(np.abs(g.S)))
    S = np.array(S_list)
    change = np.full(len(cfs), np.nan)
    for k in range(1, len(cfs)):
        change[k] = np.max(np.abs(np.abs(S[k]) - np.abs(S[k - 1])))
    return {"cutoff_factor": cfs, "n_modes": np.array(n_modes), "modes1": names[0], "modes2": names[1], "S": S,
            "change": change, "unitarity": np.array(uni), "reciprocity": np.array(rec)}
