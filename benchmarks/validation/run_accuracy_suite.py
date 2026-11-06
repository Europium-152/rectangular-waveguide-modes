"""Validation 2: accuracy of the transfer-matrix solver over a broad test matrix.

8 guide/orientations x 4 bend geometries x 7 curvature profiles x 8 frequencies = 1792 cases. For each case:
  * reference solution (vcommon.reference: DOP853 at rtol 1e-11, exact exponential for constant curvature);
  * transfer-matrix solver with default settings (error_control='input', atol=1e-6): loss, power in every
    output mode, error estimate, number of slices;
  * transfer-matrix solver with error_control='full': all entries of T and R, unitarity of [T; R];
  * the previous collocation solver (method='bvp', paper settings) for the two mid-size geometries.
The 'step (no breakpoints)' profile deliberately omits the breakpoint to document that limitation.
Results: results/accuracy_suite.pkl. Runs on all cores (~1 h on 2 cores).
"""
import os, sys, time, pickle, io, contextlib, warnings
import numpy as np
from multiprocessing import Pool
import wrmodes as wr
from vcommon import GUIDES, GEOMETRIES, PROFILES, profile, reference, loss_db

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE_SET = PROFILES + ['step (no breakpoints)']
BVP_GEOMETRIES = ('90deg 7D', '90deg 26D')
NF = 8


def one(job):
    gname, geo, prof, f = job
    a, b, mode_in, _ = GUIDES[gname]
    th, LD = GEOMETRIES[geo]
    L = LD * max(a, b)
    pname = 'step' if prof.startswith('step') else prof
    kappa, bps, const = profile(pname, th, L)
    use_bps = None if prof == 'step (no breakpoints)' else (bps or None)
    out = dict(guide=gname, geometry=geo, profile=prof, f=f, L=L, theta=th)
    t0 = time.perf_counter()
    modes, beta, T_ref, R_ref, _ = reference(f, a, b, mode_in[2], L, kappa, bps, const)
    out['t_ref'] = time.perf_counter() - t0
    names = [m[0] for m in modes]
    j = names.index(f"TE{mode_in[1]},{mode_in[2]}")
    zs = np.linspace(0, L, 2001)
    out.update(N=len(modes), beta_in=float(beta[j].real), beta_min=float(beta.real.min()),
               kappa_max=float(np.max(np.abs(kappa(zs)))), loss_ref=float(loss_db(T_ref[j, j])),
               P_ref=np.abs(T_ref[:, j]) ** 2)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        _, r = wr.solve_transfer_matrix(f, a, b, mode_in, L, kappa, breakpoints=use_bps, profile=False)
        out['t_new'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, rf = wr.solve_transfer_matrix(f, a, b, mode_in, L, kappa, breakpoints=use_bps, profile=False,
                                         error_control='full')
        out['t_full'] = time.perf_counter() - t0
    out['warned'] = len(wlist) > 0
    P_new = np.abs(r.T[:, j]) ** 2
    S = np.vstack([rf.T, rf.R])
    out.update(loss_new=float(loss_db(r.T[j, j])), P_new=P_new, err_P=float(np.max(np.abs(P_new - out['P_ref']))),
               est=float(r.error_estimate), slices=r.n_slices, status=r.status,
               err_T_full=float(np.max(np.abs(rf.T - T_ref))), err_R_full=float(np.max(np.abs(rf.R - R_ref))),
               est_full=float(rf.error_estimate), slices_full=rf.n_slices, status_full=rf.status,
               max_R=float(np.max(np.abs(R_ref))),
               unitarity_new=float(np.max(np.abs(S.conj().T @ S - np.eye(len(modes))))))
    if geo in BVP_GEOMETRIES:
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mb, rb = wr.solve_mode_coupling(f, a, b, mode_in, L, kappa, method='bvp', maximum_modes=50,
                                                max_nodes=1_000_000, resolution='auto', tol=0.5)
            out['loss_bvp'] = float(loss_db(rb.y[[m[0] for m in mb].index(names[j]), -1]))
            out['status_bvp'] = int(rb.status)
        except Exception as e:  # noqa
            out['loss_bvp'], out['status_bvp'] = np.nan, -1
        out['t_bvp'] = time.perf_counter() - t0
    return out


if __name__ == '__main__':
    jobs = []
    for gname, (a, b, mode_in, (f1, f2)) in GUIDES.items():
        for geo in GEOMETRIES:
            for prof in PROFILE_SET:
                for f in np.linspace(f1, f2, NF):
                    jobs.append((gname, geo, prof, float(f)))
    # heaviest first for better load balance
    jobs.sort(key=lambda j: -(GEOMETRIES[j[1]][1] * j[3] / GUIDES[j[0]][3][1]))
    print(len(jobs), 'cases', flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    res, t0 = [], time.time()
    with Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
            res.append(r)
            if i % 100 == 0:
                print(f"{i:5d} {time.time()-t0:6.0f}s {r['guide']} {r['geometry']} {r['profile']} {r['f']/1e9:.1f} GHz "
                      f"N={r['N']} errP={r['err_P']:.1e} est={r['est']:.1e}", flush=True)
                pickle.dump(res, open(os.path.join(HERE, 'results', 'accuracy_suite.partial.pkl'), 'wb'))
    pickle.dump(res, open(os.path.join(HERE, 'results', 'accuracy_suite.pkl'), 'wb'))
    print('done', time.time() - t0)
