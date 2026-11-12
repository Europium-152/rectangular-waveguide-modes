"""WP5 acceptance: staircase of WP4 `Step`s against the taper (card WP5, 'Staircase of WP4 Steps').

A taper of length L is replaced by n uniform sections of length L/n whose sides are those of the taper at the
section midpoints, joined by WP4 mode-matching `Step`s (default truncation, parity-reduced), with a step from the
end cross-sections of the taper to the first and last sections. The sections carry the modes of the coupled class
with cutoff below cf * f, cf = max(6, 4 n / (k L)), i.e. every mode with |beta| (L/n) <~ 4 (evanescent
interaction between neighbouring steps), checked by doubling it at the largest n. As n grows the
staircase converges to the smooth taper; its result is extrapolated in 1/n (Richardson with the observed order) and
compared with the WP5 taper (mode_factor 8, B5', atol 1e-9) in every propagating entry of the class.

Cases:
* 'H-plane': linear taper of a, 22.86 -> 45 mm, b = 10.16 mm, L = 0.1 m, 15 GHz (TE3,0 crosses cutoff), family n = 0;
* 'b-taper': linear taper of b, 22.86 x 10.16 -> 22.86 x 5.08 mm, L = 0.1 m, 35 GHz (TE1,2/TM1,2 cross cutoff),
  modes with m = 1 (the class of TE1,0: the step keeps a, so m is conserved).

Run: PYTHONPATH=src python benchmarks/taper/staircase.py   (about 3 min)
     PYTHONPATH=src python benchmarks/taper/staircase.py --cases b-taper --ns 80 160 320 640 1280 \
         --out staircase_b_long.npz   (about 1 h)
Output: benchmarks/taper/data/staircase.npz (and the files named by --out)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _meta import meta, save  # noqa: E402

from wrmodes.components import Taper, Guide  # noqa: E402
from wrmodes.gsm import straight  # noqa: E402
from wrmodes.modematch import Step  # noqa: E402
from wrmodes.modes import ModeBasis  # noqa: E402

CASES = {
    "H-plane": dict(f=15e9, L=0.1, start=(22.86e-3, 10.16e-3), end=(45e-3, 10.16e-3), sel=lambda m: m.n == 0),
    "b-taper": dict(f=35e9, L=0.1, start=(22.86e-3, 10.16e-3), end=(22.86e-3, 5.08e-3), sel=lambda m: m.m == 1),
}
NS = (20, 40, 80, 160, 320)
CF_MIN = 6.0
DECAY = 4.0      # sections keep the modes with |beta| h <= DECAY (h = L / n): neighbouring steps interact through them


def cf_sections(case, n, decay=DECAY):
    k = 2 * np.pi * case["f"] / 299792458.0
    return max(CF_MIN, decay * n / (k * case["L"]))


def basis(f, a, b, sel, cf):
    return ModeBasis(a, b, [m for m in ModeBasis.below(cf * f, a, b).modes if sel(m)], frame="H")


def staircase(case, n, cf=None):
    cf = cf_sections(case, n) if cf is None else cf
    f, L, (a0, b0), (a1, b1), sel = case["f"], case["L"], case["start"], case["end"], case["sel"]
    zm = (np.arange(n) + 0.5) / n
    dims = [(a0, b0)] + [(a0 + (a1 - a0) * t, b0 + (b1 - b0) * t) for t in zm] + [(a1, b1)]
    bases = [basis(f, a, b, sel, cf) for a, b in dims]
    g = None
    for i in range(len(dims) - 1):
        (aa, bb), (ac, bc) = dims[i], dims[i + 1]
        st = Step(aa, bb, ac, bc).gsm(f, bases[i], bases[i + 1])
        g = st if g is None else g @ st
        if i < n:
            g = g @ straight(bases[i + 1], f, L / n)
    return g


def taper(case):
    f, L, (a0, b0), (a1, b1), sel = case["f"], case["L"], case["start"], case["end"], case["sel"]
    p1 = basis(f, a0, b0, sel, 1.0)
    p2 = basis(f, a1, b1, sel, 1.0)
    tp = Taper(L, a1, b1, "linear", guide=Guide(a0, b0), mode_factor=8.0, atol=1e-9)
    return tp.gsm(f, p1, p2), p1, p2


def block(g, p1, p2):
    i1 = [g.basis1.index(m) for m in p1.modes]
    i2 = [g.basis2.index(m) for m in p2.modes]
    S = g.S
    idx = i1 + [len(g.basis1) + i for i in i2]
    return S[np.ix_(idx, idx)]


def main(cases=None, ns=None, decay=None, out="staircase.npz"):
    global NS, DECAY
    if ns is not None:
        NS = tuple(ns)
    if decay is not None:
        DECAY = float(decay)
    store = {}
    for name, case in CASES.items():
        if cases is not None and name not in cases:
            continue
        t0 = time.time()
        gt, p1, p2 = taper(case)
        ref = block(gt, p1, p2)
        tt = time.time() - t0
        Ss, times = [], []
        for n in NS:
            t0 = time.time()
            Ss.append(block(staircase(case, n), p1, p2))
            times.append(time.time() - t0)
        Ss = np.array(Ss)
        d = np.array([np.max(np.abs(S - ref)) for S in Ss])
        # observed order from the last three, then Richardson of the last two
        diffs = [np.max(np.abs(Ss[k + 1] - Ss[k])) for k in range(len(NS) - 1)]
        order = np.log2(diffs[-2] / diffs[-1])
        p = max(order, 0.5)
        rich = Ss[-1] + (Ss[-1] - Ss[-2]) / (2 ** p - 1)
        drich = np.max(np.abs(rich - ref))
        # sensitivity to the section truncation at the largest n
        S12 = block(staircase(case, NS[-1], cf_sections(case, NS[-1], 2 * DECAY)), p1, p2)
        dcf = np.max(np.abs(S12 - Ss[-1]))
        print(f"{name}: {len(p1)} + {len(p2)} propagating port modes; taper {tt:.1f} s")
        for n, dd, t in zip(NS, d, times):
            print(f"   n = {n:4d}: max |S_stair - S_taper| = {dd:.2e}   ({t:.1f} s)")
        print(f"   successive differences {', '.join(f'{x:.1e}' for x in diffs)}; observed order {order:.2f}")
        print(f"   Richardson (order {p:.2f}, n = {NS[-2]}, {NS[-1]}): max |S_extrap - S_taper| = {drich:.2e}")
        print(f"   section truncation x2 at n = {NS[-1]}: change {dcf:.1e}", flush=True)
        store_cf = [cf_sections(case, n) for n in NS]
        print(f"   section cutoff factors {', '.join(f'{c:.1f}' for c in store_cf)}")
        store[name] = dict(S=Ss, ref=ref, rich=rich, d=d, drich=drich, order=order, dcf=dcf, times=np.array(times))
    m = meta("benchmarks/taper/staircase.py", "staircase of WP4 Steps vs the WP5 taper",
             cases={k: dict(f=v["f"], L=v["L"], start=v["start"], end=v["end"]) for k, v in CASES.items()},
             n_steps=list(NS), cutoff_factor_sections=f"max({CF_MIN}, {DECAY} n / (k L))", taper="mode_factor 8, B5', atol 1e-9")
    arrays = {}
    for k, v in store.items():
        key = k.replace("-", "_")
        for kk, vv in v.items():
            arrays[f"{key}_{kk}"] = np.asarray(vv)
    save(out, m, n_steps=np.array(NS), **arrays)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--ns", nargs="*", type=int, default=None)
    ap.add_argument("--decay", type=float, default=None)
    ap.add_argument("--out", default="staircase.npz")
    a = ap.parse_args()
    main(a.cases, a.ns, a.decay, a.out)
