"""Taper profiles p(t) on t in [0, 1] with p(0) = 0 and p(1) = 1, and their derivatives p'(t).

A taper from a dimension x0 to x1 over a length L uses x(z) = x0 + (x1 - x0) p(z / L), so that
x'(z) = (x1 - x0) p'(z / L) / L. Profiles are used by `wrmodes.components.Taper` and `StrangledBend`.

Named profiles (`get_profile(name)`):

* ``'linear'``: p = t, p' = 1.
* ``'raised_cosine'``: p = (1 - cos(pi t)) / 2, p' = (pi / 2) sin(pi t) (zero slope at both ends).
* ``'bottle'``: p = (1 - exp(-3.5 t)) / (1 - exp(-3.5)), our fitted form of the bottle-shaped taper of Butterweck &
  de Ronde (1968, p. 96, fig. 13), which gives no formula (`BottleProfile`: source, fit and exact formula).

Any callable p(t) is accepted too (`CallableProfile`); its derivative is taken by central finite differences, with
a warning, unless a derivative function is given.
"""
import warnings

import numpy as np

__all__ = ["Profile", "LinearProfile", "RaisedCosineProfile", "CallableProfile", "BottleProfile", "get_profile",
           "PROFILES"]


class Profile:
    """Base class: p(t) and p'(t) on [0, 1], vectorized; p(0) = 0, p(1) = 1."""

    name = "profile"

    def __call__(self, t):
        raise NotImplementedError

    def derivative(self, t):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(type(self).__name__)


class LinearProfile(Profile):
    """p(t) = t."""

    name = "linear"

    def __call__(self, t):
        return np.asarray(t, dtype=float) * 1.0

    def derivative(self, t):
        return np.ones_like(np.asarray(t, dtype=float))


class RaisedCosineProfile(Profile):
    """p(t) = (1 - cos(pi t)) / 2, p'(t) = (pi / 2) sin(pi t): zero slope at both ends (C1 joins to straights)."""

    name = "raised_cosine"

    def __call__(self, t):
        return 0.5 * (1.0 - np.cos(np.pi * np.asarray(t, dtype=float)))

    def derivative(self, t):
        return 0.5 * np.pi * np.sin(np.pi * np.asarray(t, dtype=float))


class CallableProfile(Profile):
    """A user profile p(t) (vectorized callable on [0, 1], p(0) = 0, p(1) = 1).

    `derivative` is the given callable dp, or else a fourth-order central finite difference of p with step
    `step` (one-sided near the ends), with a RuntimeWarning at construction: finite differences limit the accuracy
    of the taper coefficients to about 1e-9 relative. The end values are checked to 1e-9.
    """

    name = "callable"

    def __init__(self, p, dp=None, step=1e-4):
        if not callable(p):
            raise TypeError("profile must be callable")
        self.p, self.dp, self.step = p, dp, float(step)
        p0, p1 = float(np.asarray(p(np.array([0.0])))[0]), float(np.asarray(p(np.array([1.0])))[0])
        if abs(p0) > 1e-9 or abs(p1 - 1.0) > 1e-9:
            raise ValueError(f"a profile must have p(0) = 0 and p(1) = 1 (got {p0:g}, {p1:g})")
        if dp is None:
            warnings.warn("CallableProfile: no derivative given; p'(t) is taken by finite differences "
                          "(about 1e-9 relative accuracy)", RuntimeWarning, stacklevel=3)

    def __call__(self, t):
        return np.asarray(self.p(np.asarray(t, dtype=float)), dtype=float)

    def derivative(self, t):
        t = np.asarray(t, dtype=float)
        if self.dp is not None:
            return np.asarray(self.dp(t), dtype=float) * np.ones_like(t)
        h = self.step
        # shift the 5-point stencil inside [0, 1] near the ends (p is only defined there)
        c = np.clip(t, 2 * h, 1 - 2 * h)
        s = t - c                                   # offset of t from the stencil centre (|s| <= 2h)
        f = [self(c + j * h) for j in (-2, -1, 0, 1, 2)]
        d1 = (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * h)
        d2 = (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * h * h)
        return d1 + s * d2

    def __repr__(self):
        return f"CallableProfile({self.p!r})"

    def __eq__(self, other):
        return isinstance(other, CallableProfile) and self.p is other.p and self.dp is other.dp

    def __hash__(self):
        return hash(id(self.p))


class BottleProfile(Profile):
    """Bottle-shaped profile: exponentially decreasing slope, p(t) = (1 - exp(-lam t)) / (1 - exp(-lam)),
    p'(t) = lam exp(-lam t) / (1 - exp(-lam)); fitted form of the taper of Butterweck & de Ronde (1968).

    **Source.** H. J. Butterweck and F. C. de Ronde, "Oversized rectangular waveguide components for millimetre
    waves", Philips Tech. Rev. 29 (1968) 86-101, p. 96 and fig. 13: a 140 mm taper from standard 4-mm guide
    (3.10 x 1.55 mm) to standard 3-cm guide (22.86 x 10.16 mm) whose width a(z) "has a bottle-like shape" and whose
    height b(z) is linear, "the theoretical design providing a minimum H10 -> H30 mode conversion" for its length;
    measured H10 -> H30 below -40 dB, against about -18 dB for a linear a(z) of the same length (4 mm wavelength).
    **The paper gives no formula and no dimensions of the shape**: fig. 13 is an unscaled perspective sketch. Its
    text gives the design rule: the wide end, where the beat wavelength of H10 and H30 is longest, "has to be
    designed with a very small slope ..., whereas the narrower part may have a greater inclination" (p. 96).

    **Fitted form (ours).** The simplest smooth monotone profile with that property: the slope decreases
    exponentially from the narrow end (t = 0) to the wide end (t = 1), by the factor exp(lam). lam = 3.5 was fitted
    with this module's solver (`benchmarks/taper/fit_bottle.py`): over lam = 0, 1, ..., 6 (steps of 0.5 near the
    optimum), the paper's taper (3.10 -> 22.86 mm in width, 140 mm, family n = 0) has the lowest worst-case
    TE10 -> TE30 conversion over 65-85 GHz at lam = 3.5: -40.3 dB (-40.7 dB at 75 GHz), against -20.5 dB for the
    linear taper. lam is fitted to a step of 0.5 only; the profile is a reconstruction of the design rule, not
    Butterweck's (unpublished) shape, and the comparison with the -40 dB measurement is therefore not an
    independent check. Other bottle shapes are possible: a profile with constant TE10 -> TE30 conversion per beat
    length beyond the TE30 cutoff plane (a' = C a (beta_1 - beta_3)) gives only -23 dB.

    Parameters
    ----------
    lam : float
        Logarithm of the ratio of the end slopes, p'(0) / p'(1) = exp(lam) (default 3.5); lam -> 0 is linear.
    """

    name = "bottle"

    def __init__(self, lam=3.5):
        self.lam = float(lam)

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        if abs(self.lam) < 1e-12:
            return t * 1.0
        return -np.expm1(-self.lam * t) / -np.expm1(-self.lam)

    def derivative(self, t):
        t = np.asarray(t, dtype=float)
        if abs(self.lam) < 1e-12:
            return np.ones_like(t)
        return self.lam * np.exp(-self.lam * t) / -np.expm1(-self.lam)

    def __repr__(self):
        return f"BottleProfile(lam={self.lam:g})"


PROFILES = {"linear": LinearProfile, "raised_cosine": RaisedCosineProfile, "bottle": BottleProfile}


def get_profile(profile):
    """A Profile from a name ('linear', 'raised_cosine', 'bottle'), a Profile, or a callable p(t)."""
    if isinstance(profile, Profile):
        return profile
    if isinstance(profile, str):
        try:
            return PROFILES[profile]()
        except KeyError:
            raise ValueError(f"unknown profile {profile!r}; named profiles are {sorted(PROFILES)}") from None
    if callable(profile):
        return CallableProfile(profile)
    raise TypeError(f"profile must be a name, a Profile or a callable, got {profile!r}")
