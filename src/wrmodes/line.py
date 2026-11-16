"""Transmission lines: cascades of components, with explicit frame changes, and frequency sweeps.

A `Line` is a sequence of two-port components (`wrmodes.components`) starting in a guide (`Guide`). At each
frequency it chooses one basis in the global frame of each guide (CONVENTIONS.md section 3, `Guide.basis`), maps it
to the local frame of every component, and cascades the GSMs with the Redheffer star product (CONVENTIONS.md
section 6).

Protocol (see `wrmodes.components.base.Component`):

* ``plane`` (required): 'H' or 'E' for components whose local frame is the H-plane or E-plane frame; None for
  components that the H <-> E frame change leaves unchanged (straight sections). A component that does not declare
  it is refused with TypeError. Where the plane changes, the line inserts a `FrameChange`; the inserted frame
  changes are part of `line.elements` and of `line.describe()`, so that they are never hidden.
* ``guides`` (optional): (input Guide, output Guide) for components that change the cross-section (steps). The
  input guide must be the line's current guide; the line then calls ``gsm(f, basis1, basis2)`` with basis2 chosen
  by the line's truncation rule in the output guide, in the component's frame.
* ``guide`` / ``with_guide(guide)`` (optional): a component without ``guides`` and without a guide of its own is
  attached to the current guide of the line; a component whose guide differs from it is refused. A component whose
  ``guides`` is None until it is attached (`Taper`, `StrangledBend` without a guide) gets its ``guides`` from the
  attached copy, so it may change the cross-section too.

Nested lines are flattened into the outer line.

Evanescent near fields (WF2). Two non-uniform components (anything but `Straight` and `FrameChange`) joined
**directly** exchange their evanescent near fields: the cascade is right only if the basis at the junction carries
the evanescent modes (CONVENTIONS.md section 8, zero-length cascades). "Directly" means that the straights between
them (frame changes count as zero length) are shorter than `NEAR_FIELD_DECAY_LENGTHS` = 3 decay lengths 1/|beta| of
the least-attenuated evanescent mode of the guide there, so that this mode keeps more than e^-3 = 5 % of its
amplitude from one component to the other. When such a junction has a basis without evanescent modes (the default
cutoff_factor = 1), `Line.gsm` issues one `EvanescentJunctionWarning` per line (WP7b: 0.4-4 % of the power off for a
short bend directly at a miter; with cutoff_factor 3 and evanescent-coupling bends within 2e-4).

Reference planes inside a discontinuity (WF3). The CONVENTIONS.md section 6 reference plane of some components lies
a depth delta > 0 inside the discontinuity (the optical mirror below 90 degrees: delta = a at 45 degrees). Their GSM
continues the uniform arm analytically to that plane, so the evanescent entries carry exp(|beta| delta) per port
(about 1e13 at cutoff_factor 3 for a 45 degree miter in WR-90), and a star product with such entries loses
everything to rounding. Protocol:

* The component returns its GSM at the reference planes, as documented, and attaches
  ``g.buried = wrmodes.gsm.BuriedPlanes(depth=(delta1, delta2), gsm=g_u)``: g_u is its GSM in the same port bases at
  the planes delta_p further out along each arm, where the arm is uniform, so that
  g = shift_planes(g_u, delta1, delta2) in exact arithmetic. `fem2d.Coplanar2D` and `fem2d.Miter` do so whenever
  a port plane of the outline is de-embedded into the core (`PortPlane.deembed` > 0). Components whose reference
  planes are uniform cross-sections (`MiterMM`, `Step`, bends, tapers, twists) leave ``buried`` None and need
  nothing else; a later component takes part by attaching `BuriedPlanes` to its GSM.
* `Line.gsm` takes each depth from the run of uniform elements (`Straight`, `FrameChange`) next to that port: the
  component enters the cascade as g_u, with the propagating modes shifted by delta at the component (a phase,
  exact) and the evanescent modes moved into the straights of the run, nearest first, which then decay as
  exp(-|beta| (s - delta)) instead of exp(+|beta| delta) followed by exp(-|beta| s). This is exact: in a uniform
  section (lossless or lossy, CONVENTIONS.md section 3: evanescent modes are uncoupled there) and through a frame
  change, an evanescent mode is only multiplied by its own factor, which commutes with the shift. Two buried ports
  share a straight: s >= delta2 + delta1' is needed (s = 2a for two 45 degree optical miters, whose mirrors then
  touch).
* If the run is shorter than the depth and ends at another non-uniform component, the discontinuities overlap (the
  cascade does not describe the physical geometry): that port keeps its GSM at the reference plane, as before WF3,
  and `Line.gsm` issues one `BuriedPlaneWarning` per line when the basis there has evanescent modes. If the run
  reaches the line's own port, the line's port lies inside the discontinuity too: the line's GSM is given at that
  plane (as for the component alone), with the shift applied last, and carries `buried` itself.
* The near-field warning above measures the uniform guide actually left, s - delta2 - delta1'.
* ``Line.gsm(..., buried_planes='keep')`` cascades the GSMs at the reference planes, as before WF3 (for checks).

Passivity check (WF3). `Line.gsm` computes the largest singular value of the propagating block of its result (the
2-norm, one SVD of a 2 Np x 2 Np matrix, small next to the cascade) and issues a `NonPassiveCascadeWarning` when it
exceeds 1 + `PASSIVITY_TOL` = 1 + 1e-8, for lossless and lossy lines alike. The warning names the element whose GSM
is not passive, or the junction at which the partial cascade first stops being passive (found by replaying the
cascade, only when the check fails). It is a warning, not an error: the GSM is still returned for inspection, a
sweep is not interrupted, and `warnings.simplefilter('error', NonPassiveCascadeWarning)` makes it an error (the
test suite does so).
"""
import warnings
import os
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.constants as cst

from .gsm import GSM, BuriedPlanes, straight, shift_planes
from .modes import Mode
from .transfer_matrix import _blas_limit
from .components.base import Component, UNDECLARED
from .components.guide import Guide, map_basis, _check_plane
from .components.frames import FrameChange
from .components import Straight, SmoothBend, SechCurvature, ConstantCurvature   # noqa: F401 (re-exported by wrmodes)

__all__ = ["Line", "SweepResult", "EvanescentJunctionWarning", "BuriedPlaneWarning", "NonPassiveCascadeWarning",
           "NEAR_FIELD_DECAY_LENGTHS", "PASSIVITY_TOL"]

NEAR_FIELD_DECAY_LENGTHS = 3.0
"""Two non-uniform components closer than this many decay lengths of the least-attenuated evanescent mode are joined
"directly" (module docstring)."""


PASSIVITY_TOL = 1e-8
"""`Line.gsm` warns when the largest singular value of the propagating block exceeds 1 + PASSIVITY_TOL."""

_DEPTH_RTOL = 1e-9
"""Relative tolerance on 'straight length >= depth' (e.g. s = 2a between two 45 degree optical miters)."""


class EvanescentJunctionWarning(UserWarning):
    """Two non-uniform components are joined directly while the line's basis there has no evanescent modes."""


class BuriedPlaneWarning(UserWarning):
    """A reference plane inside a discontinuity has less uniform guide next to it than its depth (WF3)."""


class NonPassiveCascadeWarning(UserWarning):
    """The cascade of a line is not passive: the largest singular value of its propagating block exceeds 1."""


def _nonuniform(e):
    return not isinstance(e, (Straight, FrameChange))


def _depths(g):
    """(depth1, depth2) of the reference planes of g inside its discontinuity (`GSM.buried`), or (0, 0)."""
    b = getattr(g, "buried", None)
    return (0.0, 0.0) if b is None else tuple(b.depth)


def _largest_singular_value(g, f):
    """Largest singular value of the propagating block of g (0 if it has no propagating mode)."""
    m1, m2 = g.basis1.propagating_mask(f), g.basis2.propagating_mask(f)
    if not (m1.any() or m2.any()):
        return 0.0
    Sp = np.block([[g.S11[np.ix_(m1, m1)], g.S12[np.ix_(m1, m2)]], [g.S21[np.ix_(m2, m1)], g.S22[np.ix_(m2, m2)]]])
    return float(np.linalg.norm(Sp, 2))


def _least_attenuated(guide, f):
    """(|beta|, Mode) of the evanescent mode of `guide` with the lowest cutoff above f (any family), or None."""
    from .modes import ModeBasis
    k = 2 * np.pi * f / cst.c
    full = ModeBasis.below(2.0 * f, guide.w, guide.h)
    chi = np.array([np.pi * np.hypot(m.m / guide.w, m.n / guide.h) for m in full.modes])
    ev = chi > k
    if not np.any(ev):
        return None
    i = int(np.flatnonzero(ev)[np.argmin(chi[ev])])
    return float(np.sqrt(chi[i] ** 2 - k * k)), full.modes[i]


def _attach(component, guide):
    own = getattr(component, "guide", None)
    if own is None:
        attach = getattr(component, "with_guide", None)
        return attach(guide) if attach is not None else component
    if own != guide:
        raise ValueError(f"{component!r} belongs to {own!r}, not to the line's current {guide!r}")
    return component


def _plane_of(component):
    p = getattr(component, "plane", UNDECLARED)
    if p is UNDECLARED:
        raise TypeError(f"{component!r} does not declare its plane (Component.plane must be 'H', 'E' or None)")
    if p not in ("H", "E", None):
        raise TypeError(f"{component!r} has plane {p!r}; expected 'H', 'E' or None")
    return p


class Line(Component):
    """A line of components, in order of propagation.

    Parameters
    ----------
    guide : Guide
        Global cross-section (w along x, h along y) at the input, and wall conductivity.
    components : sequence of Component
        In order of propagation: straights, smooth bends, frame changes, lines, and components that change the
        cross-section through the ``guides`` attribute (see the module docstring).
    plane : 'H' or 'E'
        Frame of the line's two ports. 'H' (default) is the global frame of CONVENTIONS.md section 2: local mode
        numbers are the global ones. A Line is itself a component of this plane and can be nested (it is then
        flattened into the outer line).

    Attributes
    ----------
    elements : tuple of Component
        The components, attached to their guides, with the inserted frame changes.
    inserted : tuple of bool
        True for the frame changes the line inserted.
    guides : (Guide, Guide)
        Guides at the input and at the output of the line.
    """

    def __init__(self, guide, components, plane="H", name=""):
        if not isinstance(guide, Guide):
            raise TypeError("guide must be a wrmodes.components.Guide")
        self.guide = guide
        self.plane = _check_plane(plane)
        self.name = name or "line"
        elements, inserted, guides_out, keys = [], [], [], []

        def add(e, ins, key, g_out):
            elements.append(e); inserted.append(ins); keys.append(key); guides_out.append(g_out)

        def frame_change(frm, to, g):
            add(FrameChange(frm, to, g), True, ("FrameChange", frm, to), g)

        plane_now, guide_now = plane, guide
        for c in components:
            if not isinstance(c, Component):
                raise TypeError(f"{c!r} is not a wrmodes component")
            if isinstance(c, Line):
                if c.guide != guide_now:
                    raise ValueError(f"{c!r} starts in {c.guide!r}, but the line is in {guide_now!r} at this point")
                if c.plane != plane_now:
                    frame_change(plane_now, c.plane, guide_now)
                for e, ins, key, g_out in zip(c.elements, c.inserted, c._keys, c._guides_out):
                    add(e, ins, key, g_out)
                plane_now, guide_now = c.plane, c.guides[1]
                continue
            if isinstance(c, FrameChange):
                if c.from_plane != plane_now:
                    raise ValueError(f"{c!r} starts in the {c.from_plane} frame, but the line is in the {plane_now} "
                                     f"frame at this point")
                add(_attach(c, guide_now), False, ("FrameChange", c.from_plane, c.to_plane), guide_now)
                plane_now = c.to_plane
                continue
            p = _plane_of(c)
            if p is not None and p != plane_now:
                frame_change(plane_now, p, guide_now)
                plane_now = p
            g = getattr(c, "guides", None)
            if g is not None:
                g_in, g_out = g
                if not (isinstance(g_in, Guide) and isinstance(g_out, Guide)):
                    raise TypeError(f"{c!r}.guides must be two wrmodes.components.Guide")
                if g_in != guide_now:
                    if (guide_now.sigma is not None and g_in.sigma is None
                            and g_in == Guide(guide_now.w, guide_now.h)):
                        raise NotImplementedError(
                            f"{c!r} has perfectly conducting walls only (wrmodes 1.0: no ohmic loss in steps and "
                            f"miters), but the line's guide {guide_now!r} has a wall conductivity; build the line "
                            f"with sigma=None, or leave this component out")
                    raise ValueError(f"{c!r} starts in {g_in!r}, but the line is in {guide_now!r} at this point")
                add(c, False, c, g_out)
                guide_now = g_out
            else:
                e = _attach(c, guide_now)
                g2 = getattr(e, "guides", None)          # attached components that change the cross-section (Taper)
                if g2 is not None:
                    if g2[0] != guide_now:
                        raise ValueError(f"{e!r} starts in {g2[0]!r}, but the line is in {guide_now!r} at this point")
                    add(e, False, e, g2[1])
                    guide_now = g2[1]
                else:
                    add(e, False, c, guide_now)
        if plane_now != plane:
            frame_change(plane_now, plane, guide_now)
        self.elements = tuple(elements)
        self.inserted = tuple(inserted)
        self._keys = tuple(keys)
        self._guides_out = tuple(guides_out)
        self.guides = (guide, guide_now)

    def __repr__(self):
        return f"<Line {self.name!r}: {len(self.elements)} elements in {self.guide!r}, ports in the {self.plane} frame>"

    def describe(self):
        """One line per element, marking the frame changes the line inserted and the changes of guide."""
        rows = [f"{self.name}: {self.guide!r}, ports in the {self.plane} frame"]
        g_now = self.guide
        for i, (e, ins, g_out) in enumerate(zip(self.elements, self.inserted, self._guides_out)):
            p = getattr(e, "plane", None)
            fr = f"[{e.from_plane}->{e.to_plane}]" if isinstance(e, FrameChange) else f"[{p or '-'}]"
            extra = "   (inserted by Line)" if ins else ""
            if g_out != g_now:
                extra += f"   (to {g_out!r})"
                g_now = g_out
            rows.append(f"  {i:3d} {fr:7s} {e.name or repr(e)}{extra}")
        return "\n".join(rows)

    def __str__(self):
        return self.describe()

    def with_guide(self, guide):
        if guide != self.guide:
            raise ValueError("a Line cannot be moved to another guide")
        return self

    def default_basis(self, f, cutoff_factor=None):
        """Bases of the line's rule in the input and output guides; cutoff_factor=None means 1."""
        cutoff_factor = 1.0 if cutoff_factor is None else cutoff_factor
        return (self.guides[0].basis(f, cutoff_factor, plane=self.plane),
                self.guides[1].basis(f, cutoff_factor, plane=self.plane))

    def gsm(self, f, basis1=None, basis2=None, cutoff_factor=1.0, evanescent_families="live", cache=None,
            buried_planes="absorb"):
        """Cascade of the line at f (Hz), with both ports in the line's frame (the global frame for plane='H').

        Parameters
        ----------
        basis1 : ModeBasis, optional
            Basis of port 1, a local basis of the line's plane for the input guide (its sides are checked against
            `guide.dims(plane)`). Default: `Guide.basis(f, cutoff_factor, plane, evanescent_families)`. A
            sub-basis (e.g. one family) is only exact for coplanar lines of constant cross-section, where the
            families do not mix; with H- and E-plane components, the families of one frame mix in the other.
            After a change of guide, the basis is always the line's rule in the new guide.
        basis2 : ModeBasis, optional
            Only checked: it must equal the basis the line arrives at.
        cutoff_factor, evanescent_families
            Truncation rule of `Guide.basis` (default 'live': evanescent modes of the live families of both frames).
        cache : dict, optional
            Memo of component GSMs, keyed by (component, guide, f, port bases). Pass the same dict to several calls
            (e.g. an ensemble of lines in which only straight lengths change, built from the same component
            objects) to compute each bend once per frequency. Straight sections are not cached (they are cheap).
            Components are immutable; a cache must not outlive changes to them. A memo is always used within one
            call, so that a component that appears several times is computed once.
        buried_planes : 'absorb' or 'keep'
            Reference planes inside a discontinuity (module docstring, WF3). 'absorb' (default): the depth of such
            a plane is taken from the uniform guide next to the component, so that no growing evanescent factor
            enters the star products. 'keep': cascade the GSMs at the reference planes as they are (the cascade
            before WF3; ill-conditioned with evanescent modes, for checks only).

        Every component receives the same physical modes, mapped to its local frame. The result is checked for
        passivity (module docstring): a `NonPassiveCascadeWarning` names the element or junction where the
        partial cascade first exceeds a largest singular value of 1 + PASSIVITY_TOL over the propagating modes.
        If the line's own ports lie inside a discontinuity (a buried component at an end, with less uniform guide
        before the line's port than its depth), the result is the GSM at those planes, as for the component alone,
        and carries `buried` (`wrmodes.gsm.BuriedPlanes`) like a component.
        """
        if buried_planes not in ("absorb", "keep"):
            raise ValueError("buried_planes must be 'absorb' or 'keep'")

        def rule(guide, plane):
            return guide.basis(f, cutoff_factor, plane=plane, evanescent_families=evanescent_families)

        if basis1 is None:
            basis1 = rule(self.guide, self.plane)
        else:
            if basis1.frame is None:
                basis1 = basis1.with_frame(self.plane)
            elif basis1.frame != self.plane:
                raise ValueError(f"basis1 has frame {basis1.frame!r}, expected the line's plane {self.plane!r}")
            a, b = self.guide.dims(self.plane)
            if not (np.isclose(basis1.a, a, rtol=1e-12, atol=0) and np.isclose(basis1.b, b, rtol=1e-12, atol=0)):
                raise ValueError(f"basis1 is for a {basis1.a:g} x {basis1.b:g} cross-section, but the line's guide "
                                 f"is {a:g} x {b:g} in the {self.plane} frame")
        memo = {} if cache is None else cache
        # 1. the GSM of every element, in the bases of the line
        gsms, bases_in, guides_in = [], [], []
        current, guide_now = basis1, self.guide
        for e, key, g_out in zip(self.elements, self._keys, self._guides_out):
            if isinstance(e, FrameChange):
                nxt = map_basis(current, e.from_plane, e.to_plane)
            elif g_out != guide_now:
                nxt = rule(g_out, current.frame)
            else:
                nxt = current
            ck = None if isinstance(e, Straight) else (key, guide_now, float(f), current, nxt)
            g = memo.get(ck) if ck is not None else None
            if g is None:
                g = e.gsm(f, current, nxt)
                if g.basis1 != current or g.basis2 != nxt:
                    raise ValueError(f"{e!r} returned a GSM in other bases than the ones the line gave it")
                if ck is not None:
                    memo[ck] = g
            gsms.append(g)
            bases_in.append(current)
            guides_in.append(guide_now)
            current, guide_now = nxt, g_out
        if basis2 is not None and basis2 != current:
            raise ValueError(f"basis2 {basis2!r} is not the line's output basis {current!r}")
        # 2. reference planes inside discontinuities
        depths = [_depths(g) for g in gsms]
        if buried_planes == "absorb" and any(d != (0.0, 0.0) for d in depths):
            gsms, defer = self._absorb(f, gsms, depths, basis1, current)
        else:
            defer = None
        # 3. near-field warning (WF2), on the uniform guide actually left between non-uniform components
        last, gap = None, 0.0
        for i, e in enumerate(self.elements):
            if _nonuniform(e):
                if last is not None and not getattr(self, "_junction_warned", False):
                    left = gap - depths[last][1] - depths[i][0]
                    self._check_junction(f, self.elements[last], e, 0.0 if abs(left) < 1e-12 else left,
                                         bases_in[i], guides_in[i])
                last, gap = i, 0.0
            elif isinstance(e, Straight):
                gap += float(e.length)
        # 4. cascade
        out = None
        for g in gsms:
            out = g if out is None else out @ g
        if out is None:
            out = straight(basis1, f, 0.0)
        res = GSM(out.S11, out.S12, out.S21, out.S22, out.basis1, out.basis2, f, self.name)
        if defer is not None and (defer[2] > 0 or defer[3] > 0):
            # the line's own ports lie inside a discontinuity: shift the evanescent modes (deferred through the
            # uniform end runs) to them; the propagating modes are already at the reference planes
            res = shift_planes(res, defer[0], defer[1])
            p1, p2 = res.basis1.propagating_mask(f), res.basis2.propagating_mask(f)
            core = shift_planes(out, -defer[2] * p1, -defer[3] * p2, name=self.name)
            res.buried = BuriedPlanes((defer[2], defer[3]), core)
        self._check_passivity(f, res, gsms)
        return res

    def _absorb(self, f, gsms, depths, basis1, basis2):
        """Take the depth of buried reference planes from the uniform guide next to them (module docstring).

        Returns the GSMs to cascade and the deferred shifts at the line's ports (per-mode arrays for port 1 and 2,
        and the two depths)."""
        n = len(gsms)
        uniform = [not _nonuniform(e) for e in self.elements]
        remaining = [float(e.length) if isinstance(e, Straight) else 0.0 for e in self.elements]
        moved = [[0.0, 0.0] for _ in range(n)]      # evanescent shift moved into straight i at its port 1 / 2 side
        out = list(gsms)
        defer = [np.zeros(len(basis1)), np.zeros(len(basis2)), 0.0, 0.0]
        for i, g in enumerate(gsms):
            if depths[i] == (0.0, 0.0):
                continue
            resid = list(depths[i])
            for port, step in ((0, -1), (1, 1)):
                d = depths[i][port]
                if d <= 0:
                    continue
                run, j = [], i + step
                while 0 <= j < n and uniform[j]:
                    run.append(j)
                    j += step
                boundary = not 0 <= j < n
                avail = sum(remaining[j] for j in run)
                if avail < d * (1 - _DEPTH_RTOL) and not boundary:
                    self._warn_buried(f, i, port, d, avail, self.elements[j], g)
                    continue                                            # present behaviour: shift at the component
                take = min(d, avail)
                left = take
                for j in run:
                    c = min(remaining[j], left)
                    if c > 0:
                        remaining[j] -= c
                        moved[j][1 if step < 0 else 0] += c          # the straight's end next to the component
                        left -= c
                resid[port] = d - take + left
                if boundary:                            # the line's own port lies resid[port] inside: defer
                    k = 0 if step < 0 else 1
                    defer[k] = defer[k] + resid[port]
                    defer[k + 2] += resid[port]
                    resid[port] = 0.0
            if tuple(resid) == depths[i]:
                continue                                                # nothing absorbed: the GSM as given
            # propagating modes: the whole shift, at the component; evanescent modes: only the residual depth
            u = g.buried.gsm
            p1, p2 = u.basis1.propagating_mask(f), u.basis2.propagating_mask(f)
            out[i] = shift_planes(u, np.where(p1, depths[i][0], resid[0]), np.where(p2, depths[i][1], resid[1]))
        for j in range(n):
            if moved[j] != [0.0, 0.0]:
                g = out[j]
                e1, e2 = ~g.basis1.propagating_mask(f), ~g.basis2.propagating_mask(f)
                out[j] = shift_planes(g, moved[j][0] * e1, moved[j][1] * e2)
        # deferred shifts at the line's ports: evanescent modes only, in the line's port bases (frame changes in
        # the end runs map each mode to a mode with the same beta)
        defer[0] = defer[0] * ~basis1.propagating_mask(f)
        defer[1] = defer[1] * ~basis2.propagating_mask(f)
        return out, defer

    def _warn_buried(self, f, i, port, depth, avail, neighbour, g):
        """Warn once per line: a buried plane next to a non-uniform neighbour with too little straight between."""
        basis = g.basis1 if port == 0 else g.basis2
        prop = basis.propagating_mask(f)
        if getattr(self, "_buried_warned", False) or np.all(prop):
            return
        self._buried_warned = True
        e = self.elements[i]
        gain = float(np.max(np.abs(basis.beta(f)[~prop].imag))) * depth
        warnings.warn(
            f"{self.name}: port {port + 1} of {e.name or type(e).__name__} lies {depth * 1e3:.4g} mm inside its "
            f"discontinuity, but only {avail * 1e3:.4g} mm of uniform guide separate it from "
            f"{neighbour.name or type(neighbour).__name__}: the two discontinuities overlap, so the cascade does not "
            f"describe the physical geometry, and the evanescent entries at that plane carry factors up to "
            f"exp({gain:.3g}). Use a longer straight, or model both together (e.g. fem2d.chained_miter_outline). "
            f"(Warned once per line.)", BuriedPlaneWarning, stacklevel=3)

    def _check_passivity(self, f, res, gsms):
        """Warn if the largest singular value of the propagating block exceeds 1 + PASSIVITY_TOL (module docstring)."""
        sv = _largest_singular_value(res, f)
        if sv <= 1 + PASSIVITY_TOL:
            return
        label = lambda k: f"element {k} ({self.elements[k].name or type(self.elements[k]).__name__})"
        where = "the whole line"
        out = None
        for k, g in enumerate(gsms):                    # locate the first partial cascade that is not passive
            out = g if out is None else out @ g
            if _largest_singular_value(out, f) > 1 + PASSIVITY_TOL:
                if _largest_singular_value(g, f) > 1 + PASSIVITY_TOL:
                    where = f"{label(k)} itself"
                else:                                   # the junction with the last non-uniform element before k
                    p = max([q for q in range(k) if _nonuniform(self.elements[q])], default=k - 1)
                    where = f"the junction of {label(p)} and {label(k)}"
                break
        warnings.warn(
            f"{self.name}: the cascade at {f / 1e9:.6g} GHz is not passive (largest singular value {sv:.12g} of the "
            f"propagating block, > 1 + {PASSIVITY_TOL:g}); first at {where}. The result is not physical: usually an "
            f"ill-conditioned star product (evanescent entries at reference planes inside a discontinuity, or "
            f"components too close together for their bases).", NonPassiveCascadeWarning, stacklevel=3)

    def _check_junction(self, f, left, right, gap, basis, guide):
        """Warn once if `left` and `right` are joined directly while `basis` has no evanescent modes."""
        if len(basis) == 0 or not np.all(basis.propagating_mask(f)):
            return
        la = _least_attenuated(guide, f)
        if la is None:
            return
        alpha, mode = la
        if alpha * gap >= NEAR_FIELD_DECAY_LENGTHS:
            return
        self._junction_warned = True
        warnings.warn(
            f"{self.name}: {left.name or type(left).__name__!s} and {right.name or type(right).__name__!s} are joined "
            f"directly ({gap * 1e3:.3g} mm apart, less than {NEAR_FIELD_DECAY_LENGTHS:g} decay lengths "
            f"{NEAR_FIELD_DECAY_LENGTHS / alpha * 1e3:.3g} mm of {mode.name} at {f / 1e9:.4g} GHz) while the line's "
            f"basis there has no evanescent modes, so their near fields do not interact in the cascade (a few % of "
            f"the power for a bend directly at a miter). Use cutoff_factor >= 3 or a longer straight between them. "
            f"(Warned once per line.)", EvanescentJunctionWarning, stacklevel=3)

    # ------------------------------------------------------------------------------------------ sweeps
    def sweep(self, freqs, input_mode, processes=None, cutoff_factor=1.0, keep_gsm=False,
              evanescent_families="live"):
        """Frequency sweep, one frequency per task, in parallel processes.

        Parameters
        ----------
        freqs : array_like
            Frequencies (Hz).
        input_mode : Mode, tuple or name
            Mode incident at port 1, in the frame of the line's ports (the global frame for plane='H').
        processes : int or None
            Number of worker processes; None means os.cpu_count() (capped at the number of frequencies), 1 runs in
            this process. Every process, this one included, limits BLAS/LAPACK to one thread (as `blas_threads=1`
            in wrmodes 0.1), so that results do not depend on the number of processes. The line is sent to the
            workers by pickling (start method 'spawn'): curvatures must be picklable (module-level functions, or
            `SechCurvature` / `ConstantCurvature`, not lambdas). If the line cannot be pickled and the platform
            has the 'fork' start method, 'fork' is used instead. Scripts that use processes > 1 need the usual
            `if __name__ == "__main__":` guard.
        cutoff_factor, evanescent_families
            Truncation rule, passed to `Line.gsm`.
        keep_gsm : bool
            Keep the GSM of every frequency in the result.

        Returns
        -------
        SweepResult
        """
        freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        mode = Mode.parse(input_mode)
        args = [(f, mode, cutoff_factor, keep_gsm, evanescent_families) for f in freqs]
        if processes is None:
            processes = os.cpu_count() or 1
        processes = max(1, min(int(processes), len(freqs)))
        if processes == 1:
            points = [_sweep_point(self, *a) for a in args]
        else:
            try:
                pickle.dumps(self)
                ctx = mp.get_context("spawn")
            except (pickle.PicklingError, AttributeError, TypeError) as exc:
                if "fork" not in mp.get_all_start_methods():
                    raise TypeError(f"Line.sweep with processes > 1 needs a picklable line ({exc}); use picklable "
                                    f"curvatures (e.g. SechCurvature) or processes=1") from exc
                ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=processes, mp_context=ctx, initializer=_init_worker,
                                     initargs=(self,)) as ex:
                points = list(ex.map(_sweep_worker, args))
        return SweepResult(freqs, mode, points, keep_gsm)


_WORKER_LINE = None


def _init_worker(line):
    global _WORKER_LINE
    _WORKER_LINE = line


def _sweep_worker(args):
    return _sweep_point(_WORKER_LINE, *args)


def _sweep_point(line, f, mode, cutoff_factor, keep_gsm, evanescent_families="live"):
    with _blas_limit(1):
        g = line.gsm(f, cutoff_factor=cutoff_factor, evanescent_families=evanescent_families)
    b1, b2 = g.basis1, g.basis2
    p1, p2 = b1.propagating_mask(f), b2.propagating_mask(f)
    out = dict(gsm=g if keep_gsm else None, loss_db=np.nan, converted={}, reflected=np.nan)
    if mode not in b1 or not p1[b1.index(mode)]:
        return out
    j = b1.index(mode)
    col = g.S21[:, j]
    out["converted"] = {m: float(abs(col[i]) ** 2) for i, m in enumerate(b2.modes) if p2[i] and m != mode}
    out["loss_db"] = float(-20 * np.log10(abs(col[b2.index(mode)]))) if mode in b2 else np.inf
    out["reflected"] = float(np.sum(np.abs(g.S11[p1, j]) ** 2))
    return out


class _ModeDict(dict):
    """dict keyed by Mode that also accepts names ('TE2,0') and tuples (('TE', 2, 0))."""

    def __getitem__(self, key):
        return dict.__getitem__(self, Mode.parse(key))

    def __contains__(self, key):
        try:
            return dict.__contains__(self, Mode.parse(key))
        except (ValueError, TypeError, IndexError):
            return False


class SweepResult:
    """Result of `Line.sweep`.

    Attributes
    ----------
    freqs : ndarray
        Frequencies (Hz).
    input_mode : Mode
    loss_db : ndarray
        -20 log10 |S21| from the input mode to the same mode (dB): mode conversion and reflection loss. NaN where
        the input mode does not propagate.
    converted_power : dict Mode -> ndarray
        Power transmitted to each other propagating mode at port 2, as a fraction of the input power (also
        indexed by name, e.g. result.converted_power['TE2,0']). NaN where the mode does not propagate.
    reflected_power : ndarray
        Total power reflected into the propagating modes at port 1, as a fraction of the input power.
    gsms : list of GSM or None
        The GSMs, if the sweep was run with keep_gsm=True.
    """

    def __init__(self, freqs, input_mode, points, keep_gsm):
        self.freqs = freqs
        self.input_mode = input_mode
        self.loss_db = np.array([p["loss_db"] for p in points])
        self.reflected_power = np.array([p["reflected"] for p in points])
        modes = sorted({m for p in points for m in p["converted"]}, key=lambda m: (m.kind, m.m, m.n))
        self.converted_power = _ModeDict()
        for m in modes:
            self.converted_power[m] = np.array([p["converted"].get(m, np.nan) for p in points])
        self.gsms = [p["gsm"] for p in points] if keep_gsm else None

    def __repr__(self):
        return (f"<SweepResult: {len(self.freqs)} frequencies, input {self.input_mode}, "
                f"{len(self.converted_power)} converted modes>")
