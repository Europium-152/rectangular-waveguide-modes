"""Transmission lines: cascades of components in one guide, with explicit frame changes, and frequency sweeps.

A `Line` is a sequence of two-port components (`wrmodes.components`) in a guide of constant cross-section
(`Guide`). At each frequency it chooses one basis in the global frame (CONVENTIONS.md section 3), maps it to the
local frame of every component, and cascades the GSMs with the Redheffer star product (CONVENTIONS.md section 6).

Frames (CONVENTIONS.md section 2). Every component may declare a `plane` attribute: 'H' or 'E' for components
whose local frame is the H-plane or E-plane frame, None for components that are the same in every frame (straight
sections). Where the plane changes, the line inserts a `FrameChange`; the inserted frame changes are part of
`line.elements` and of `line.describe()`, so that they are never hidden. A component without a `plane`
attribute is treated as frame-independent.

Guides. Components may hold a guide (for their `default_basis`); a component with a `with_guide(guide)` method and
no guide of its own is attached to the line's guide. A component whose guide differs from the line's is refused.
"""
import os
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .gsm import straight
from .modes import Mode
from .transfer_matrix import _blas_limit
from .components.base import Component
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
        raise ValueError(f"{component!r} belongs to {own!r}, not to the line's {guide!r}")
    return component


class Line(Component):
    """A line of components in one guide, in order of propagation.

    Parameters
    ----------
    guide : Guide
        Global cross-section (w along x, h along y) and wall conductivity.
    components : sequence of Component
        In order of propagation. Each component must keep the cross-section (straights, smooth bends, frame
        changes, and lines in the same guide); components that change it (steps, tapers) are not supported yet.
    plane : 'H' or 'E'
        Frame of the line's two ports. 'H' (default) is the global frame of CONVENTIONS.md section 2: local mode
        numbers are the global ones. A Line is itself a component of this plane and can be nested.

    Attributes
    ----------
    elements : tuple of Component
        The components, attached to the guide, with the inserted frame changes.
    inserted : tuple of bool
        True for the frame changes the line inserted.
    """

    def __init__(self, guide, components, plane="H", name=""):
        if not isinstance(guide, Guide):
            raise TypeError("guide must be a wrmodes.components.Guide")
        self.guide = guide
        self.plane = _check_plane(plane)
        self.name = name or "line"
        elements, inserted = [], []
        current = plane
        for c in components:
            if not isinstance(c, Component):
                raise TypeError(f"{c!r} is not a wrmodes component")
            if isinstance(c, FrameChange):
                if c.from_plane != current:
                    raise ValueError(f"{c!r} starts in the {c.from_plane} frame, but the line is in the {current} "
                                     f"frame at this point")
                elements.append(_attach(c, guide)); inserted.append(False)
                current = c.to_plane
                continue
            p = getattr(c, "plane", None)
            if p is not None and p != current:
                elements.append(FrameChange(current, p, guide)); inserted.append(True)
                current = p
            elements.append(_attach(c, guide)); inserted.append(False)
        if current != plane:
            elements.append(FrameChange(current, plane, guide)); inserted.append(True)
        self.elements = tuple(elements)
        self.inserted = tuple(inserted)

    def __repr__(self):
        return f"<Line {self.name!r}: {len(self.elements)} elements in {self.guide!r}, ports in the {self.plane} frame>"

    def describe(self):
        """One line per element, marking the frame changes the line inserted."""
        rows = [f"{self.name}: {self.guide!r}, ports in the {self.plane} frame"]
        for i, (e, ins) in enumerate(zip(self.elements, self.inserted)):
            p = getattr(e, "plane", None)
            fr = f"[{e.from_plane}->{e.to_plane}]" if isinstance(e, FrameChange) else f"[{p or '-'}]"
            rows.append(f"  {i:3d} {fr:7s} {e.name or repr(e)}" + ("   (inserted by Line)" if ins else ""))
        return "\n".join(rows)

    def __str__(self):
        return self.describe()

    def with_guide(self, guide):
        if guide != self.guide:
            raise ValueError("a Line cannot be moved to another guide")
        return self

    def default_basis(self, f, cutoff_factor=1.0):
        b = self.guide.basis(f, cutoff_factor, plane=self.plane)
        return b, b

    def gsm(self, f, basis1=None, basis2=None, cutoff_factor=1.0, evanescent_families="propagating"):
        """Cascade of the line at f (Hz), with both ports in the line's frame (the global frame for plane='H').

        The basis is `Guide.basis(f, cutoff_factor)` unless `basis1` is given (a local basis of the line's plane).
        Every component receives the same physical modes, mapped to its local frame.
        """
        if basis1 is None:
            basis1 = self.guide.basis(f, cutoff_factor, plane=self.plane, evanescent_families=evanescent_families)
        elif basis1.frame is None:
            basis1 = basis1.with_frame(self.plane)
        elif basis1.frame != self.plane:
            raise ValueError(f"basis1 has frame {basis1.frame!r}, expected the line's plane {self.plane!r}")
        if basis2 is not None and basis2 != basis1:
            raise ValueError("a line in one guide has the same basis at both ports")
        if not self.elements:
            return straight(basis1, f, 0.0, self.name)
        current, out = basis1, None
        for e in self.elements:
            if isinstance(e, FrameChange):
                nxt = map_basis(current, e.from_plane, e.to_plane)
                g = e.gsm(f, current, nxt)
            else:
                g = e.gsm(f, current)
                if g.basis1 != current or g.basis2 != current:
                    raise ValueError(f"{e!r} returned a GSM in other bases than the line's; components that "
                                     f"change the cross-section are not supported by Line yet")
            current = g.basis2
            out = g if out is None else out @ g
        out.name = self.name
        return out

    # ------------------------------------------------------------------------------------------ sweeps
    def sweep(self, freqs, input_mode, processes=None, cutoff_factor=1.0, keep_gsm=False):
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
        cutoff_factor : float
            Passed to `Line.gsm`.
        keep_gsm : bool
            Keep the GSM of every frequency in the result.

        Returns
        -------
        SweepResult
        """
        freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        mode = Mode.parse(input_mode)
        args = [(f, mode, cutoff_factor, keep_gsm) for f in freqs]
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


def _sweep_point(line, f, mode, cutoff_factor, keep_gsm):
    with _blas_limit(1):
        g = line.gsm(f, cutoff_factor=cutoff_factor)
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
