"""Validation 6: run times of the transfer-matrix solver and of the previous collocation solver (serial, one
process, so that the timings are not disturbed by other work).

  a) the COMSOL case of the paper (22 frequencies, 15 cm 60 deg hyperbolic secant E-plane bend);
  b) time per frequency point versus number of coupled modes (60 cm 60 deg sech bend, two orientations);
  c) dependence on the bend length (60 deg sech bend, TE10 H-plane, 60 and 89.5 GHz).
Each timing is the best of `REPEAT` runs. Results: results/timings.json
"""
import json, os, time, io, contextlib, warnings, platform
import numpy as np
import scipy
import wrmodes as wr

HERE = os.path.dirname(os.path.abspath(__file__))
REPEAT = 3
INCH = 0.0254


def sech(theta, L, T=4.0):
    th = np.radians(theta); g = 2 * T / L; R0 = 4 / th / g * np.arctan(np.tanh(T / 2))
    return lambda z: 1 / (R0 * np.cosh(g * (np.asarray(z) - L / 2)))


def best(fn):
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


def run_new(f, a, b, mi, L, k):
    return wr.solve_mode_coupling(f, a, b, mi, L, k, profile=False)


def run_bvp(f, a, b, mi, L, k):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return wr.solve_mode_coupling(f, a, b, mi, L, k, method='bvp', maximum_modes=50, max_nodes=1_000_000,
                                      resolution='auto', tol=0.5)


out = dict(machine=platform.platform(), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)

# a) COMSOL case
a, b, mi = 0.4 * INCH, 0.9 * INCH, ('TE', 0, 1)
th = np.radians(60); r0 = 0.055; S = 4; g = 4 / th / r0 * np.arctan(np.tanh(S / 2)); L = 2 * S / g
k = lambda z: 1 / (r0 * np.cosh(g * (np.asarray(z) - L / 2)))
F = np.arange(18e9, 60.1e9, 2e9)
out['comsol_case'] = dict(points=len(F), t_new=best(lambda: [run_new(f, a, b, mi, L, k) for f in F]),
                          t_bvp=best(lambda: [run_bvp(f, a, b, mi, L, k) for f in F]))
print('COMSOL case', out['comsol_case'], flush=True)

# b) time per point vs number of modes
rows = []
for orient, (a, b, mi) in {'TE10 H-plane': (0.9 * INCH, 0.4 * INCH, ('TE', 1, 0)),
                           'tall-guide TE01 E-plane': (0.9 * INCH, 0.4 * INCH, ('TE', 0, 1))}.items():
    k = sech(60, 0.6)
    for f in np.arange(20e9, 90.1e9, 10e9) - 0.5e9:
        m, _ = run_new(f, a, b, mi, 0.6, k)
        rows.append(dict(orient=orient, f=float(f), N=len(m), t_new=best(lambda: run_new(f, a, b, mi, 0.6, k)),
                         t_bvp=best(lambda: run_bvp(f, a, b, mi, 0.6, k))))
        print(rows[-1], flush=True)
out['per_point'] = rows

# c) bend length
rows = []
a, b, mi = 0.9 * INCH, 0.4 * INCH, ('TE', 1, 0)
for f in [60e9, 89.5e9]:
    for L in [0.15, 0.6, 2.0, 5.0]:
        k = sech(60, L)
        _, r = run_new(f, a, b, mi, L, k)
        rows.append(dict(f=f, L=L, slices=r.n_slices, t_new=best(lambda: run_new(f, a, b, mi, L, k)),
                         t_bvp=best(lambda: run_bvp(f, a, b, mi, L, k))))
        print(rows[-1], flush=True)
out['length'] = rows
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'results', 'timings.json'), 'w'), indent=1)
