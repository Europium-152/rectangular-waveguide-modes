"""One record for every convergence study of wrmodes (WF4).

Every `convergence_study` (`Twist`, `Taper` / `VariableGuide` / `StrangledBend`, `SmoothBend`, `MiterMM`, `Step`
and `wrmodes.modematch.convergence_study`, `wrmodes.fem2d.convergence_study`) returns a `ConvergenceRecord`: a dict
with the fixed keys below, one list entry per computed row (one GSM), in the order computed:

* ``component`` (str): the component's name or repr; ``f`` (Hz); ``input_mode`` (name of the incident mode);
* ``parameter`` (list of str): the parameter varied in each row: 'mode_factor', 'cutoff_factor', 'n_modes',
  'mesh_density', 'n_evanescent' or 'slices';
* ``values`` (list): its value in each row;
* ``variants`` (list of dict): the other settings that distinguish rows of the same parameter (e.g. {'tail': True});
* ``n_modes`` (list): the size of the solved problem per row (internal mode count; a pair (port 1, port 2) for
  steps; the arm modes per scalar problem for MiterMM);
* ``powers`` (list of dict): {'transmitted': {mode: |S21[mode, in]|^2}, 'reflected': {mode: |S11[mode, in]|^2}}
  over the propagating modes of port 2 and port 1, for the input mode;
* ``max_change`` (list of float): max | P - P_finest | over all those powers, against the finest row (the last
  one) of the same parameter and variant (0 for the finest row itself);
* ``richardson`` (dict or None): an extrapolation where the series supports it: {'parameter', 'variant', 'order'
  (the assumed order of convergence in 1/value), 'powers'} (and component-specific extra keys);
* ``wall_time`` (list of float): seconds per row;
* ``reference`` (str): what the rows are compared with (the finest row of each sweep).

The keys of the earlier return values of each study are kept in the same dict as aliases for one release (wrmodes
1.0); they are listed in the docstring of each study. New code should use the keys above.
"""
import time

import numpy as np

from .modes import Mode

__all__ = ["ConvergenceRecord", "column_powers", "build_record", "timed"]

PARAMETERS = ("mode_factor", "cutoff_factor", "n_modes", "mesh_density", "n_evanescent", "slices")
RECORD_KEYS = ("component", "f", "input_mode", "parameter", "values", "variants", "n_modes", "powers", "max_change",
               "richardson", "wall_time", "reference")


class ConvergenceRecord(dict):
    """The common return value of every `convergence_study` (module docstring). A dict; `rows()` iterates over
    the per-row entries as dicts."""

    def rows(self):
        keys = ("parameter", "values", "variants", "n_modes", "powers", "max_change", "wall_time")
        return [dict(zip(keys, vals)) for vals in zip(*(self[k] for k in keys))]

    def table(self):
        """A plain-text table of the rows (parameter, value, variant, n_modes, max_change, wall time)."""
        out = [f"{self['component']} at {self['f'] / 1e9:.6g} GHz, input {self['input_mode']} ({self['reference']})"]
        for r in self.rows():
            var = ", ".join(f"{k}={v}" for k, v in r["variants"].items())
            out.append(f"  {r['parameter']:>13s} = {r['values']!s:>8s}  [{var}]  n_modes {r['n_modes']!s:>10s}  "
                       f"max change {r['max_change']:.2e}  {r['wall_time']:.2f} s")
        return "\n".join(out)


def default_input_mode(basis, f, input_mode=None):
    """The input mode: the given one, else TE1,0 if it is in `basis` and propagates, else the first propagating
    mode of `basis`."""
    if input_mode is not None:
        return Mode.parse(input_mode)
    prop = basis.propagating_mask(f)
    te10 = Mode("TE", 1, 0)
    if te10 in basis and prop[basis.index(te10)]:
        return te10
    return basis.modes[int(np.flatnonzero(prop)[0])]


def column_powers(g, input_mode, f=None):
    """{'transmitted': {mode: |S21[mode, in]|^2}, 'reflected': {mode: |S11[mode, in]|^2}} of the GSM g for the
    input mode, over the propagating modes of each port."""
    f = g.f if f is None else f
    j = g.basis1.index(Mode.parse(input_mode))
    p1, p2 = g.basis1.propagating_mask(f), g.basis2.propagating_mask(f)
    return {"transmitted": {m.name: float(abs(g.S21[i, j]) ** 2) for i, m in enumerate(g.basis2.modes) if p2[i]},
            "reflected": {m.name: float(abs(g.S11[i, j]) ** 2) for i, m in enumerate(g.basis1.modes) if p1[i]}}


def powers_from_column(t, r, names2, names1):
    """The `powers` entry from complex columns t (port 2) and r (port 1) with their mode names."""
    return {"transmitted": {n: float(abs(x) ** 2) for n, x in zip(names2, t)},
            "reflected": {n: float(abs(x) ** 2) for n, x in zip(names1, r)}}


def _max_change(p, q):
    out = 0.0
    for side in ("transmitted", "reflected"):
        for k, v in p[side].items():
            if k in q[side]:
                out = max(out, abs(v - q[side][k]))
    return out


def timed(fn, *args, **kwargs):
    """(result, wall time in s) of fn(*args, **kwargs)."""
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t


def build_record(component, f, input_mode, rows, richardson=None, reference=None, aliases=None):
    """A `ConvergenceRecord` from rows dict(parameter, value, variant, n_modes, powers, wall_time).

    max_change is computed against the last row with the same parameter and variant. `aliases`: the earlier keys of
    the study, merged into the record (they must not shadow the record's keys, except where documented)."""
    rec = ConvergenceRecord()
    finest = {}
    for r in rows:
        finest[(r["parameter"], tuple(sorted(r["variant"].items(), key=lambda kv: kv[0])))] = r["powers"]
    rec.update(component=str(getattr(component, "name", "") or repr(component)), f=float(f),
               input_mode=Mode.parse(input_mode).name,
               parameter=[r["parameter"] for r in rows], values=[r["value"] for r in rows],
               variants=[dict(r["variant"]) for r in rows], n_modes=[r["n_modes"] for r in rows],
               powers=[r["powers"] for r in rows],
               max_change=[_max_change(r["powers"], finest[(r["parameter"], tuple(sorted(r["variant"].items(),
                                                                                    key=lambda kv: kv[0])))])
                           for r in rows],
               richardson=richardson, wall_time=[float(r["wall_time"]) for r in rows],
               reference=reference or "the finest row (last value) of each parameter and variant")
    for k, v in (aliases or {}).items():
        if k in RECORD_KEYS:
            continue
        rec[k] = v
    return rec
