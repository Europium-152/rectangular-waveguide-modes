"""Validation 1: the closed-form coupling integrals (eqs. 5.6-5.11 of the paper, wrmodes.mode_coupling.integral_1..6)
against direct numerical quadrature of their definitions with the mode potentials of eq. (5.5).

Integration domain u in [-a/2, a/2] (bend plane), v in [-b/2, b/2], xi = u / R with R = 1 (the closed forms
contain the factor 1/R). Gauss-Legendre quadrature with 120 x 120 nodes is exact to rounding for these
trigonometric integrands. Also checks that the integrals vanish when m_p and m_q have the same parity or when
n_p != n_q (the selection rules stated in Section 5).
"""
import itertools, json, os, sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from wrmodes.mode_coupling import integral_1, integral_2, integral_3, integral_4, integral_5, integral_6

HERE = os.path.dirname(os.path.abspath(__file__))
x, w = leggauss(120)


def eps(l):
    return 1 if l == 0 else 2


def potentials(m, n, a, b, kind, U, V):
    """Pi (TM) or Gamma (TE) of eq. (5.5) and their u, v derivatives on the grid."""
    norm = np.pi * np.sqrt(m ** 2 * b / a + n ** 2 * a / b)
    su, cu = np.sin(m * np.pi * (U + a / 2) / a), np.cos(m * np.pi * (U + a / 2) / a)
    sv, cv = np.sin(n * np.pi * (V + b / 2) / b), np.cos(n * np.pi * (V + b / 2) / b)
    ku, kv = m * np.pi / a, n * np.pi / b
    if kind == 'TM':
        c = 2 / norm
        return c * su * sv, c * ku * cu * sv, c * kv * su * cv
    c = np.sqrt(eps(m) * eps(n)) / norm
    return c * cu * cv, -c * ku * su * cv, -c * kv * cu * sv


def numerical(mp, nq_p, mq, nq_q, a, b):
    U = (a / 2 * x)[:, None] * np.ones((1, len(x)))
    V = np.ones((len(x), 1)) * (b / 2 * x)[None, :]
    W = (a / 2 * w)[:, None] * (b / 2 * w)[None, :]
    xi = U  # R = 1
    out = {}
    if mp >= 1 and nq_p >= 1 and mq >= 1 and nq_q >= 1:
        P, Pu, Pv = potentials(mp, nq_p, a, b, 'TM', U, V)
        Q, Qu, Qv = potentials(mq, nq_q, a, b, 'TM', U, V)
        out['I1'] = np.sum(W * xi * (Pu * Qu + Pv * Qv))
        out['I2'] = np.sum(W * xi * P * Q)
    if mp >= 1 and nq_p >= 1 and (mq + nq_q) > 0:          # p TM, q TE
        P, Pu, Pv = potentials(mp, nq_p, a, b, 'TM', U, V)
        G, Gu, Gv = potentials(mq, nq_q, a, b, 'TE', U, V)
        out['I3'] = np.sum(W * xi * (Pu * Gv - Pv * Gu))
    if mq >= 1 and nq_q >= 1 and (mp + nq_p) > 0:          # q TM, p TE
        P, Pu, Pv = potentials(mq, nq_q, a, b, 'TM', U, V)
        G, Gu, Gv = potentials(mp, nq_p, a, b, 'TE', U, V)
        out['I4'] = np.sum(W * xi * (Pu * Gv - Pv * Gu))
    if (mp + nq_p) > 0 and (mq + nq_q) > 0:
        G1, G1u, G1v = potentials(mp, nq_p, a, b, 'TE', U, V)
        G2, G2u, G2v = potentials(mq, nq_q, a, b, 'TE', U, V)
        out['I5'] = np.sum(W * xi * (G1u * G2u + G1v * G2v))
        out['I6'] = np.sum(W * xi * G1 * G2)
    return out


closed = {'I1': integral_1, 'I2': integral_2, 'I3': integral_3, 'I4': integral_4, 'I5': integral_5, 'I6': integral_6}
guides = {'WR-90 (a broad)': (0.9 * 0.0254, 0.4 * 0.0254), 'WR-90 (a narrow)': (0.4 * 0.0254, 0.9 * 0.0254),
          '44x30 mm': (44e-3, 30e-3), 'square 20 mm': (20e-3, 20e-3)}
rows, worst_rel, zero_viol = [], 0.0, 0.0
for gname, (a, b) in guides.items():
    for n in range(0, 4):
        for mp, mq in itertools.product(range(0, 9), repeat=2):
            if mp == mq:
                continue
            num = numerical(mp, n, mq, n, a, b)
            for key, val in num.items():
                coupled = (mp + mq) % 2 == 1
                if coupled:
                    cf = closed[key](mp, mq, n, a, b)
                    scale = max(abs(val), abs(cf), 1e-300)
                    rel = abs(cf - val) / scale
                    worst_rel = max(worst_rel, rel)
                    rows.append(dict(guide=gname, n=n, mp=mp, mq=mq, integral=key, numerical=val, closed_form=cf, rel_err=rel))
                else:
                    # same parity: must vanish (the code returns 0 without using the closed forms).
                    # Compare with the size of a typical coupled integral of the same kind (scale a for I1, I3-I5,
                    # a^3 for I2, I6, since xi = u ~ a and the potentials are normalised).
                    scale = a ** 3 if key in ('I2', 'I6') else a
                    zero_viol = max(zero_viol, abs(val) / scale)
# n_p != n_q must vanish as well
nviol = 0.0
for (a, b) in guides.values():
    for mp, mq in itertools.product(range(0, 6), repeat=2):
        for np_, nq in [(0, 1), (1, 2), (1, 3), (0, 2)]:
            num = numerical(mp, np_, mq, nq, a, b)
            if num:
                nviol = max(nviol, max(abs(v) for v in num.values()))

by_int = {}
for r in rows:
    by_int.setdefault(r['integral'], []).append(r['rel_err'])
summary = {k: dict(count=len(v), max_rel_err=float(max(v))) for k, v in sorted(by_int.items())}
summary['all'] = dict(count=len(rows), max_rel_err=float(worst_rel))
summary['same_parity_max_value_relative_to_scale'] = float(zero_viol)
summary['different_n_max_abs_value'] = float(nviol)
summary['typical_coupled_value_relative_to_scale'] = float(np.median([abs(r['numerical']) / ((guides[r['guide']][0]) ** 3 if r['integral'] in ('I2', 'I6') else guides[r['guide']][0]) for r in rows]))
print(json.dumps(summary, indent=2))
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
json.dump(dict(summary=summary, rows=rows), open(os.path.join(HERE, 'results', 'coupling_integrals.json'), 'w'), indent=1)
bad = [r for r in rows if r['rel_err'] > 1e-8]
print(f"{len(bad)} of {len(rows)} closed-form values differ from quadrature by more than 1e-8 (relative)")
for r in bad[:20]:
    print(r)
