"""Summary of results/paper_dataset.pkl -> results/paper_dataset_summary.json"""
import os, json, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
res = pickle.load(open(os.path.join(HERE, 'results', 'paper_dataset.pkl'), 'rb'))
dn = np.array([abs(r['loss_new'] - r['loss_ref']) for r in res]); db = np.array([abs(r['loss_bvp'] - r['loss_ref']) for r in res])
eP = np.array([r['err_P'] for r in res]); small = np.array([r['loss_ref'] < 1 for r in res])
sechTE10 = np.array([r['profile'] == 'sech' and r['orient'].startswith('TE10') for r in res])
out = dict(points=len(res), curves=len(set((r['bend'], r['profile'], r['orient']) for r in res)),
           max_N=int(max(r['N'] for r in res)), converged=int(sum(r['status'] == 0 for r in res)),
           new_max_err_power=float(eP.max()), new_max_loss_err_dB=float(dn.max()), new_median_loss_err_dB=float(np.median(dn)),
           bvp_max_loss_err_dB=float(db.max()), bvp_max_loss_err_dB_loss_below_1dB=float(db[small].max()),
           bvp_max_loss_err_dB_sech_TE10=float(db[sechTE10].max()),
           t_new_total_s=float(sum(r['t_new'] for r in res)), t_bvp_total_s=float(sum(r['t_bvp'] for r in res)),
           t_ref_total_s=float(sum(r['t_ref'] for r in res)))
out['speedup_vs_bvp'] = out['t_bvp_total_s'] / out['t_new_total_s']
byN = {}
for r in res:
    byN.setdefault(r['N'], []).append((r['t_new'], r['t_bvp']))
out['mean_time_per_point_by_N'] = {int(N): dict(new=float(np.mean([a for a, b in v])), bvp=float(np.mean([b for a, b in v])))
                                   for N, v in sorted(byN.items())}
json.dump(out, open(os.path.join(HERE, 'results', 'paper_dataset_summary.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'mean_time_per_point_by_N'}, indent=1))
print({N: (round(v['new'], 3), round(v['bvp'], 2), round(v['bvp'] / v['new'], 1)) for N, v in out['mean_time_per_point_by_N'].items()})
