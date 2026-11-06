"""Validation 3: physical properties and edge cases of the transfer-matrix solver.

  a) power conservation: [T; R] has orthonormal columns (lossless system), and the net power flow
     sum |A+|^2 - |A-|^2 is constant along the bend (amplitude profile);
  b) reciprocity: traversing the bend in the opposite direction (curvature -kappa(L - z), since the transverse
     axis u flips with z) gives |T_rev[p, q]| = |T[q, p]|;
  c) mirror symmetry: bending the other way (kappa -> -kappa) leaves |T| unchanged;
  d) straight guide (kappa = 0): T = diag(exp(-i beta L)), R = 0;
  e) edge cases: a mode 1 ppm above its cutoff, a higher-order input mode, a very sharp bend (R0 ~ a), and the
     single-mode regime; each compared with the reference solution.
  f) amplitude profile along the bend vs the reference solution integrated to intermediate positions.
Results: results/properties.json
"""
import json, os, warnings
import numpy as np
from scipy.integrate import solve_ivp
import wrmodes as wr
from wrmodes.transfer_matrix import coupled_mode_matrices, select_coupled_modes
from vcommon import GUIDES, profile, reference, loss_db
from wrmodes.main import list_propagating_modes

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
cases = [('WR-90 H', 0.6, 60, 'sech', 89.5e9), ('WR-90 E', 0.15, 60, 'sech', 60e9), ('WR-90 E', 0.4, 90, 'cosine', 75e9),
         ('44x30 H', 0.3, 90, 'triangular', 18e9), ('sq20 E', 0.5, 90, 'S-bend', 38e9), ('WR-28 E', 0.2, 180, 'sech', 105e9)]

# a) b) c) d)
rows = []
for gname, L, th, prof, f in cases:
    a, b, mode_in, _ = GUIDES[gname]
    k, bps, const = profile(prof, th, L)
    kw = dict(breakpoints=bps or None, profile=False, error_control='full', atol=1e-9)
    modes, r = wr.solve_transfer_matrix(f, a, b, mode_in, L, k, **kw)
    S = np.vstack([r.T, r.R])
    unit = np.max(np.abs(S.conj().T @ S - np.eye(len(modes))))
    krev = lambda z, k=k, L=L: -k(L - np.asarray(z))
    _, rr = wr.solve_transfer_matrix(f, a, b, mode_in, L, krev, **dict(kw, breakpoints=[L - p for p in bps] or None))
    recip = np.max(np.abs(np.abs(rr.T) - np.abs(r.T).T))
    _, rm = wr.solve_transfer_matrix(f, a, b, mode_in, L, lambda z, k=k: -k(z), **kw)
    mirror = np.max(np.abs(np.abs(rm.T) - np.abs(r.T)))
    rows.append(dict(case=f"{gname} {L} m {th} deg {prof} {f/1e9:g} GHz", N=len(modes), unitarity=float(unit),
                     reciprocity=float(recip), mirror=float(mirror)))
_, rs = wr.solve_transfer_matrix(70e9, *GUIDES['WR-90 E'][:3], 1.0, lambda z: 0 * np.asarray(z))
straight = float(max(np.max(np.abs(rs.T - np.diag(np.exp(-1j * rs.beta * 1.0)))), np.max(np.abs(rs.R))))
out['symmetries'] = rows
out['straight_guide_max_deviation'] = straight

# e) edge cases
edge = []
a, b, mode_in, _ = GUIDES['WR-90 E']
fam = [m for m in list_propagating_modes(95e9, a, b) if m[2] == 1]
fc = fam[6][3]                                   # cutoff of the 7th mode of the family
for label, f, mi, L, th, prof in [
        (f"{fam[6][0]} 1 ppm above cutoff", fc * (1 + 1e-6), mode_in, 0.6, 60, 'sech'),
        (f"{fam[6][0]} 1 ppb above cutoff", fc * (1 + 1e-9), mode_in, 0.6, 60, 'sech'),
        ("higher-order input TM1,1", 60e9, ('TM', 1, 1), 0.6, 60, 'sech'),
        ("higher-order input TE2,1", 80e9, ('TE', 2, 1), 0.6, 60, 'sech'),
        ("very sharp bend R0 = 1.5 cm (90 deg, L = 2.4 cm)", 40e9, mode_in, 0.024, 90, 'constant'),
        ("single-mode family (8 GHz)", 8e9, mode_in, 0.2, 90, 'sech')]:
    k, bps, const = profile(prof, th, L)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        modes, r = wr.solve_transfer_matrix(f, a, b, mi, L, k, profile=False)
    mr, beta, T, R, _ = reference(f, a, b, mi[2], L, k, bps, const)
    j = r.input_index
    S = np.vstack([r.T[:, j:j + 1], r.R[:, j:j + 1]])
    edge.append(dict(case=label, f_GHz=f / 1e9, N=len(modes), beta_min=float(np.min(beta.real)),
                     loss_new=float(loss_db(r.T[j, j])), loss_ref=float(loss_db(T[j, j])),
                     err_P=float(np.max(np.abs(np.abs(r.T[:, j]) ** 2 - np.abs(T[:, j]) ** 2))),
                     power=float(np.sum(np.abs(S) ** 2)), finite=bool(np.all(np.isfinite(r.T))), warned=len(wl) > 0,
                     slices=r.n_slices))
out['edge_cases'] = edge

# f) amplitude profile vs reference integration of the state vector
prof_rows = []
for gname, L, th, prof, f in cases[:3]:
    a, b, mode_in, _ = GUIDES[gname]
    k, bps, const = profile(prof, th, L)
    z = np.linspace(0, L, 301)
    modes, r = wr.solve_transfer_matrix(f, a, b, mode_in, L, k, z_eval=z, atol=1e-8)
    D, K, beta = coupled_mode_matrices(f, a, b, modes)
    _, _, T, R, _ = reference(f, a, b, mode_in[2], L, k, bps, const)
    N, j = len(modes), r.input_index
    y0 = np.zeros(2 * N, complex); y0[j] = 1; y0[N:] = R[:, j]
    sol = solve_ivp(lambda zz, y: (D + float(k(zz)) * K) @ y, (0, L), y0, method='DOP853', rtol=1e-12, atol=1e-14,
                    t_eval=z)
    dev = np.max(np.abs(np.abs(r.y) ** 2 - np.abs(sol.y) ** 2))
    flow = np.sum(np.abs(r.y[:N]) ** 2, 0) - np.sum(np.abs(r.y[N:]) ** 2, 0)
    # the net power flow is constant along z and equal to 1 - (reflected power)
    prof_rows.append(dict(case=f"{gname} {L} m {prof} {f/1e9:g} GHz", N=N, max_power_profile_deviation=float(dev),
                          max_net_power_flow_change_along_z=float(np.max(np.abs(flow - flow[0]))),
                          net_flow_minus_one=float(flow[0] - 1), reflected_power=float(np.sum(np.abs(r.y[N:, 0]) ** 2))))
out['profiles'] = prof_rows

os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'results', 'properties.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
