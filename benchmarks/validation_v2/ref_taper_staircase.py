"""WP7b reference for WP5 tapers of both a and b: a staircase of merged WP4 mode-matching `Step`s (wave 1) joined by
straight sections, extrapolated in the step count and checked in the mode count.

Cases (card WP7b item 1, Butterweck & de Ronde 1968 p. 96): a 140 mm taper from 3.10 x 1.55 mm (standard 4-mm guide)
to 22.86 x 10.16 mm, centred (u0 = v0 = 0), at 75 GHz, incident TE1,0 from the small end:

* `linear`: a(z) and b(z) linear;
* `bottle_a`: a(z) with wrmodes' bottle profile p(t) = (1 - exp(-3.5 t)) / (1 - exp(-3.5)) (the formula stated in the
  `wrmodes.profiles` docstring, our fit of the paper's sketch) and b(z) linear, as the paper describes its taper.

The staircase has n sections of length L/n with the cross-section of the taper at their midpoints, n + 1 centred
steps (the first from the port-1 guide, the last into the port-2 guide), and reference planes at the two ends. A
centred step couples only modes of equal parities of m and n, so the class of TE1,0 (m odd, n even) is solved alone
(exact). Truncation: every mode of the class with cutoff below cutoff_factor * f in every guide (the rule of `Step`).
The staircase converges as O(1/n) (each riser is a first-order approximation of the sloped wall); the result is
extrapolated with Richardson in 1/n from the two largest n, and the order is estimated from three.

    PYTHONPATH=src:benchmarks/validation_v2 python benchmarks/validation_v2/ref_taper_staircase.py
"""
import argparse
import json

import numpy as np

from wrmodes.gsm import GSM
from wrmodes.modematch import Step
from wrmodes.modes import Mode, ModeBasis

import vcommon as vc

C0 = 299792458.0
A1, B1, A2, B2, LT, F0 = 3.10e-3, 1.55e-3, 22.86e-3, 10.16e-3, 0.140, 75e9


def bottle(t, lam=3.5):
    return (1 - np.exp(-lam * t)) / (1 - np.exp(-lam))


PROFILES = {"linear": (lambda t: t, lambda t: t), "bottle_a": (bottle, lambda t: t)}


def class_basis(a, b, fmax, parity=(1, 0)):
    """Modes (m = parity[0] mod 2, n = parity[1] mod 2) of the a x b guide with cutoff below fmax."""
    mmax, nmax = int(2 * a * fmax / C0) + 1, int(2 * b * fmax / C0) + 1
    modes = []
    for m in range(mmax + 1):
        for n in range(nmax + 1):
            if m % 2 != parity[0] or n % 2 != parity[1]:
                continue
            fc = C0 / 2 * np.hypot(m / a, n / b)
            if fc >= fmax:
                continue
            if m + n > 0:
                modes.append(Mode("TE", m, n))
            if m >= 1 and n >= 1:
                modes.append(Mode("TM", m, n))
    return ModeBasis(a, b, modes, frame="H")


def straight_phase(basis, f, length):
    return np.exp(-1j * basis.beta(f) * length)


def staircase(profile, n, f, cutoff_factor):
    pa, pb = PROFILES[profile]
    fmax = cutoff_factor * f
    zm = (np.arange(n) + 0.5) / n
    dims = [(A1, B1)] + [(A1 + (A2 - A1) * pa(t), B1 + (B2 - B1) * pb(t)) for t in zm] + [(A2, B2)]
    bases = [class_basis(a, b, fmax) for a, b in dims]
    total = None
    for j in range(n + 1):
        (a1, b1), (a2, b2) = dims[j], dims[j + 1]
        g = Step(a1, b1, a2, b2, plane="H").solve(f, bases[j], bases[j + 1])
        if j < n:   # the straight section of guide j+1 follows the step
            D = straight_phase(bases[j + 1], f, LT / n)
            g = GSM(g.S11, g.S12 * D[None, :], D[:, None] * g.S21, D[:, None] * g.S22 * D[None, :],
                    g.basis1, g.basis2, f)
        total = g if total is None else total.star(g)
    return total


def propagating(g, f):
    i1 = np.where(g.basis1.propagating_mask(f))[0]
    i2 = np.where(g.basis2.propagating_mask(f))[0]
    return dict(S11=g.S11[np.ix_(i1, i1)], S21=g.S21[np.ix_(i2, i1)], S12=g.S12[np.ix_(i1, i2)],
                S22=g.S22[np.ix_(i2, i2)], names1=[g.basis1.names[i] for i in i1],
                names2=[g.basis2.names[i] for i in i2])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", nargs="*", default=list(PROFILES))
    p.add_argument("--n", nargs="*", type=int, default=[100, 200, 400, 800])
    p.add_argument("--cf", nargs="*", type=float, default=[2.0, 3.0])
    a = p.parse_args()
    for prof in a.profiles:
        arrays, rows = {}, []
        for cf in a.cf:
            for n in a.n:
                t = vc.Timer()
                g = staircase(prof, n, F0, cf)
                P = propagating(g, F0)
                key = f"{prof}_cf{cf:g}_n{n}"
                for k in ("S11", "S21", "S12", "S22"):
                    arrays[f"{key}_{k}"] = P[k]
                arrays[f"{key}_names1"] = np.array(P["names1"])
                arrays[f"{key}_names2"] = np.array(P["names2"])
                j = P["names2"].index("TE1,0")
                pw = {nm: float(abs(P["S21"][i, 0]) ** 2) for i, nm in enumerate(P["names2"])}
                row = dict(key=key, profile=prof, n=n, cutoff_factor=cf, seconds=round(t(), 1),
                           n_modes_large=len(g.basis2.modes), te10_db=float(vc.db(P["S21"][j, 0])),
                           te30_db=float(10 * np.log10(pw.get("TE3,0", 1e-300))), s11=float(abs(P["S11"][0, 0])))
                rows.append(row)
                print(json.dumps(row), flush=True)
        vc.save_npz(f"taper_staircase_{prof}.npz", __file__, arrays, profile=prof, rows=rows, f=F0,
                    geometry=dict(a1=A1, b1=B1, a2=A2, b2=B2, length=LT, bottle_lambda=3.5),
                    method="staircase of wrmodes.modematch.Step (WP4, solve on exact bases, parity class m odd n even)")
