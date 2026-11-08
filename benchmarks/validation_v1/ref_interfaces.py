"""Locate the wave-1 interfaces that the WP7 comparisons and adversarial tests are written against.

The interfaces are those specified by the task cards (tasks/WP1..WP4), not by any implementation. Until a card's
code is merged, ``get(...)`` raises ``InterfaceMissing`` with a message that says what is missing; the comparison
scripts print it and exit, and the tests skip with it (``require``).

Card interfaces used here:

* WP1 (tasks/WP1-cascade.md): ``wrmodes.components`` with ``Component``, ``Guide(w, h, sigma=None)``,
  ``Straight(length)``, ``SmoothBend(length, curvature, plane='H'|'E', breakpoints=None, atol=1e-6)``,
  ``FrameChange(from_plane, to_plane)``; ``wrmodes.line.Line(guide, components, plane='H')`` with
  ``.gsm(f, cutoff_factor=1.0)`` and ``.sweep(freqs, input_mode, processes=None)``.
* WP2 (tasks/WP2-loss-and-filters.md): ``wrmodes.loss`` with ``surface_resistance(f, sigma)`` and per-mode
  attenuation constants (name not fixed by the card; see ``attenuation_function``).
* WP3 (tasks/WP3-2d-solver.md): ``wrmodes.fem2d.Miter(angle, chamfer=None, plane='H')`` (needs the ``fem`` extra).
* WP4 (tasks/WP4-steps.md): ``wrmodes.modematch.Step(a1, b1, a2, b2, du=0.0, dv=0.0)``.
"""
import importlib


class InterfaceMissing(RuntimeError):
    """A card interface is not available (not merged yet, or optional extra missing)."""


def _import(module):
    try:
        return importlib.import_module(module)
    except ImportError as exc:                    # includes ModuleNotFoundError and missing optional extras
        raise InterfaceMissing(f"{module} cannot be imported ({exc}); the card that provides it is not merged yet "
                               f"or its optional dependencies are missing") from exc


def _find(name, modules):
    errors = []
    for mod in modules:
        try:
            m = _import(mod)
        except InterfaceMissing as exc:
            errors.append(str(exc))
            continue
        if hasattr(m, name):
            return getattr(m, name)
        errors.append(f"{mod} has no attribute {name!r}")
    raise InterfaceMissing(f"{name} not found: " + "; ".join(errors))


def get(name):
    """One card interface by name: Guide, Straight, SmoothBend, FrameChange, Line, Component, Miter, Step, loss."""
    table = {
        "Component": ["wrmodes.components"],
        "Guide": ["wrmodes.components", "wrmodes.line", "wrmodes"],
        "Straight": ["wrmodes.components", "wrmodes"],
        "SmoothBend": ["wrmodes.components", "wrmodes"],
        "FrameChange": ["wrmodes.components", "wrmodes"],
        "Line": ["wrmodes.line", "wrmodes.components", "wrmodes"],
        "Miter": ["wrmodes.fem2d"],
        "Step": ["wrmodes.modematch"],
    }
    if name == "loss":
        return _import("wrmodes.loss")
    if name not in table:
        raise KeyError(name)
    return _find(name, table[name])


def attenuation_function():
    """The WP2 per-mode attenuation function, as f(mode, a, b, f, sigma) -> alpha (Np/m).

    WP2's card does not fix its name or signature. The names tried are listed below; if none exists, the comparison
    raises InterfaceMissing and the coordinator wires the right call here (one line).
    """
    loss = get("loss")
    for name in ("attenuation_constant", "attenuation", "mode_attenuation", "alpha", "attenuation_constants"):
        fn = getattr(loss, name, None)
        if callable(fn):
            return name, fn
    raise InterfaceMissing("wrmodes.loss has none of attenuation_constant / attenuation / mode_attenuation / alpha / "
                           "attenuation_constants; wire WP2's per-mode attenuation in ref_interfaces.attenuation_function")


def require(*names):
    """For pytest: return the interfaces or skip the test with the reason."""
    import pytest
    out = []
    for n in names:
        try:
            out.append(get(n))
        except InterfaceMissing as exc:
            pytest.skip(f"WP7: interface {n} not available yet: {exc}")
    return out[0] if len(out) == 1 else tuple(out)


def build_line(segments, w, h, sigma=None, atol=1e-10):
    """A WP1 Line from ref_line Segments, with SmoothBend(atol=atol) so that its error is well below 1e-6 in power."""
    Guide, Straight, SmoothBend, Line = (get(n) for n in ("Guide", "Straight", "SmoothBend", "Line"))
    comps = []
    for s in segments:
        if s.kind == "straight":
            comps.append(Straight(s.length))
        else:
            comps.append(SmoothBend(s.length, s.curvature, plane=s.plane,
                                    breakpoints=list(s.breakpoints) or None, atol=atol))
    guide = Guide(w, h) if sigma is None else Guide(w, h, sigma=sigma)
    return Line(guide, comps)
