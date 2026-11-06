"""Summary statistics of results/accuracy_suite.pkl (written to results/accuracy_summary.json)."""
import os, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = pickle.load(open(os.path.join(HERE, 'results', 'accuracy_suite.pkl'), 'rb'))
main = [r for r in res if r['profile'] != 'step (no breakpoints)']
nobp = [r for r in res if r['profile'] == 'step (no breakpoints)']


def stats(rows):
    errP = np.array([r['err_P'] for r in rows]); est = np.array([r['est'] for r in rows])
    dL = np.array([abs(r['loss_new'] - r['loss_ref']) for r in rows])
    ok = np.array([r['status'] == 0 for r in rows])
    ratio = errP / np.maximum(est, 1e-16)
    sig = errP > 1e-10          # ratio only meaningful above round-off
    return dict(cases=len(rows), converged=int(ok.sum()), max_err_power=float(errP.max()),
                frac_err_power_above_atol=float(np.mean(errP > 1e-6)), max_err_power_converged=float(errP[ok].max()),
                max_loss_err_dB=float(dL.max()), median_loss_err_dB=float(np.median(dL)),
                p99_loss_err_dB=float(np.percentile(dL, 99)),
                actual_over_estimate_max=float(ratio[sig & ok].max()) if np.any(sig & ok) else None,
                actual_over_estimate_p95=float(np.percentile(ratio[sig & ok], 95)) if np.any(sig & ok) else None,
                max_unitarity_defect=float(max(r['unitarity_new'] for r in rows)),
                max_err_T_full=float(max(r['err_T_full'] for r in rows)),
                max_err_R_full=float(max(r['err_R_full'] for r in rows)),
                full_converged=int(sum(r['status_full'] == 0 for r in rows)),
                max_N=int(max(r['N'] for r in rows)), max_loss_dB=float(max(r['loss_ref'] for r in rows)),
                slices_median=float(np.median([r['slices'] for r in rows])), slices_max=int(max(r['slices'] for r in rows)),
                t_new_total=float(sum(r['t_new'] for r in rows)), t_ref_total=float(sum(r['t_ref'] for r in rows)),
                max_kappa_over_beta_in=float(max(r['kappa_max'] / r['beta_in'] for r in rows)))


summary = dict(all=stats(main), step_without_breakpoints=stats(nobp))
for key in ['profile', 'guide', 'geometry']:
    summary['by_' + key] = {v: stats([r for r in main if r[key] == v]) for v in sorted(set(r[key] for r in main))}
bvp = [r for r in main if 'loss_bvp' in r and np.isfinite(r['loss_bvp'])]
d = np.array([abs(r['loss_bvp'] - r['loss_ref']) for r in bvp])
small = np.array([r['loss_ref'] < 1 for r in bvp])
summary['bvp'] = dict(cases=len(bvp), failed=int(sum(r.get('status_bvp', 0) != 0 for r in bvp)),
                      max_loss_err_dB=float(d.max()), median_loss_err_dB=float(np.median(d)),
                      max_loss_err_dB_where_loss_below_1dB=float(d[small].max()),
                      t_bvp_total=float(sum(r['t_bvp'] for r in bvp)),
                      t_new_same_cases=float(sum(r['t_new'] for r in bvp)))
worst = sorted(main, key=lambda r: -r['err_P'])[:8]
summary['worst_cases'] = [dict(case=f"{r['guide']} | {r['geometry']} | {r['profile']} | {r['f']/1e9:.1f} GHz", N=r['N'],
                               err_P=r['err_P'], est=r['est'], slices=r['slices'], status=r['status'],
                               loss_ref=r['loss_ref']) for r in worst]
json.dump(summary, open(os.path.join(HERE, 'results', 'accuracy_summary.json'), 'w'), indent=1)
print(json.dumps({k: summary[k] for k in ['all', 'step_without_breakpoints', 'bvp', 'worst_cases']}, indent=1))
print("by profile:", {k: (v['max_err_power'], v['max_loss_err_dB'], v['slices_median']) for k, v in summary['by_profile'].items()})
print("by guide:", {k: (v['max_err_power'], v['max_N']) for k, v in summary['by_guide'].items()})
print("by geometry:", {k: (v['max_err_power'], v['max_kappa_over_beta_in'], v['max_loss_dB']) for k, v in summary['by_geometry'].items()})
