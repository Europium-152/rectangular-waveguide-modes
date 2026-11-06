"""Validation 4: convergence order of the Magnus slicing (fixed numbers of slices, no adaptivity).

For representative cases, the power error of every output mode (and the loss error) is computed against the
reference for 8 ... 2048 uniform slices; the observed order is fitted in the asymptotic range.
Results: results/convergence_order.json (+ data for the figure).
"""
import json, os
import numpy as np
from wrmodes.transfer_matrix import coupled_mode_matrices, select_coupled_modes, _slice_exponentials, _product, \
    _scattering_from_transfer
from vcommon import GUIDES, profile, reference, loss_db

HERE = os.path.dirname(os.path.abspath(__file__))
cases = [('WR-90 H', 0.6, 60, 'sech', 89.5e9), ('WR-90 E', 0.6, 60, 'sech', 89.5e9),
         ('WR-90 E', 0.15, 60, 'sech', 60e9), ('WR-90 H', 0.4, 90, 'cosine', 80e9), ('44x30 H', 1.0, 90, 'triangular', 18e9)]
# tall-guide orientation of the paper (TE01 input with the broad side in the bend plane)
TALL = (0.9 * 0.0254, 0.4 * 0.0254, ('TE', 0, 1))
out = []
for gname, L, th, prof, f in cases:
    a, b, mode_in, _ = GUIDES[gname]
    if gname == 'WR-90 E' and L == 0.6:
        a, b, mode_in = TALL
        gname = 'WR-90 tall-guide E'
    k, bps, const = profile(prof, th, L)
    modes, beta, T_ref, R_ref, _ = reference(f, a, b, mode_in[2], L, k, bps, const)
    N = len(modes)
    j = [m[0] for m in modes].index(f"TE{mode_in[1]},{mode_in[2]}")
    D, K, _ = coupled_mode_matrices(f, a, b, modes)
    ns, eP, eL, eT = [], [], [], []
    for n in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        edges = np.linspace(0, L, n + 1)
        T, R = _scattering_from_transfer(_product(_slice_exponentials(D, K, k, edges)), N)
        ns.append(n)
        eP.append(float(np.max(np.abs(np.abs(T[:, j]) ** 2 - np.abs(T_ref[:, j]) ** 2))))
        eL.append(float(abs(loss_db(T[j, j]) - loss_db(T_ref[j, j]))))
        eT.append(float(np.max(np.abs(T - T_ref))))
    ns_a, eP_a = np.array(ns), np.array(eP)
    sel = (eP_a < 1e-4) & (eP_a > 1e-8)           # asymptotic range, above the accuracy floor of the reference (~1e-9)
    order = float(-np.polyfit(np.log(ns_a[sel]), np.log(eP_a[sel]), 1)[0]) if sel.sum() >= 2 else float('nan')
    out.append(dict(case=f"{gname}, {L} m, {th} deg, {prof}, {f/1e9:g} GHz", N=N, n=ns, err_power=eP, err_loss_dB=eL,
                    err_T_full=eT, observed_order=order, max_beta_L=float(np.max(beta.real) * L)))
    print(out[-1]['case'], f"N={N}", f"order={order:.2f}", ' '.join(f"{e:.1e}" for e in eP))
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'results', 'convergence_order.json'), 'w'), indent=1)
