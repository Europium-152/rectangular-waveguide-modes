"""Validation 5: the complete data set of the SPARC figure of the paper (60 cm 60 deg and 40 cm 90 deg WR-90
bends, constant and hyperbolic secant curvature, four orientations, 18-90 GHz in 0.5 GHz steps: 16 curves x 145 frequencies = 2320 points),
computed with the transfer-matrix solver (default settings), the reference solution and the previous
collocation solver (paper settings). Results: results/paper_dataset.pkl
"""
import os, sys, time, pickle, io, contextlib, warnings
import numpy as np
from multiprocessing import Pool
import wrmodes as wr
from vcommon import reference, loss_db

HERE = os.path.dirname(os.path.abspath(__file__))
INCH = 0.0254
ORIENT = {'TE10 H-plane': (0.9 * INCH, 0.4 * INCH, ('TE', 1, 0)), 'TE10 E-plane': (0.4 * INCH, 0.9 * INCH, ('TE', 0, 1)),
          'TE01 H-plane': (0.4 * INCH, 0.9 * INCH, ('TE', 1, 0)), 'TE01 E-plane': (0.9 * INCH, 0.4 * INCH, ('TE', 0, 1))}
BENDS = {'60cm_60deg': (60, 0.60), '40cm_90deg': (90, 0.40)}
FREQS = np.arange(18e9, 90.01e9, 0.5e9)


def kappa_of(profile, theta, L, T=4.0):
    th = np.radians(theta)
    if profile == 'constant':
        return (lambda z: np.full(np.shape(z), th / L)), True
    g = 2 * T / L; R0 = 4 / th / g * np.arctan(np.tanh(T / 2))
    return (lambda z: 1 / (R0 * np.cosh(g * (np.asarray(z) - L / 2)))), False


def one(job):
    bend, prof, orient, f = job
    theta, L = BENDS[bend]
    a, b, mi = ORIENT[orient]
    k, const = kappa_of(prof, theta, L)
    out = dict(bend=bend, profile=prof, orient=orient, f=f)
    t0 = time.perf_counter(); modes, beta, T, R, _ = reference(f, a, b, mi[2], L, k, (), const, rtol=1e-10)
    out['t_ref'] = time.perf_counter() - t0
    j = [m[0] for m in modes].index(f"TE{mi[1]},{mi[2]}")
    t0 = time.perf_counter(); _, r = wr.solve_transfer_matrix(f, a, b, mi, L, k, profile=False)
    out['t_new'] = time.perf_counter() - t0
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        mb, rb = wr.solve_mode_coupling(f, a, b, mi, L, k, method='bvp', maximum_modes=50, max_nodes=1_000_000,
                                        resolution='auto', tol=0.5)
        out['t_bvp'] = time.perf_counter() - t0
    out.update(N=len(modes), loss_ref=float(loss_db(T[j, j])), loss_new=float(loss_db(r.T[j, j])),
               loss_bvp=float(loss_db(rb.y[[m[0] for m in mb].index(modes[j][0]), -1])),
               err_P=float(np.max(np.abs(np.abs(r.T[:, j]) ** 2 - np.abs(T[:, j]) ** 2))), est=float(r.error_estimate),
               slices=r.n_slices, status=r.status, P_ref=np.abs(T[:, j]) ** 2, modes=[m[0] for m in modes])
    return out


if __name__ == '__main__':
    jobs = [(bnd, p, o, float(f)) for bnd in BENDS for p in ['constant', 'sech'] for o in ORIENT for f in FREQS]
    jobs.sort(key=lambda j: -j[3])
    res, t0 = [], time.time()
    with Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            res.append(r)
            if i % 200 == 0:
                print(i, f"{time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    pickle.dump(res, open(os.path.join(HERE, 'results', 'paper_dataset.pkl'), 'wb'))
    print('done', time.time() - t0)
