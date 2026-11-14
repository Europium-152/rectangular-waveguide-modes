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
"""
import os
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .gsm import GSM, straight
from .modes import Mode
from .transfer_matrix import _blas_limit
from .components.base import Component, UNDECLARED
from .components.guide import Guide, map_basis, _check_plane
from .components.frames import FrameChange
from .components import Straight, SmoothBend, SechCurvature, ConstantCurvature   # noqa: F401 (re-exported by wrmodes)

__all__ = ["Line", "SweepResult"]


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

    def gsm(self, f, basis1=None, basis2=None, cutoff_factor=1.0, evanescent_families="live", cache=None):
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

        Every component receives the same physical modes, mapped to its local frame.
        """
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
        current, guide_now, out = basis1, self.guide, None
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
            current, guide_now = nxt, g_out
            out = g if out is None else out @ g
        if out is None:
            out = straight(basis1, f, 0.0)
        if basis2 is not None and basis2 != current:
            raise ValueError(f"basis2 {basis2!r} is not the line's output basis {current!r}")
        return GSM(out.S11, out.S12, out.S21, out.S22, out.basis1, out.basis2, f, self.name)

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
