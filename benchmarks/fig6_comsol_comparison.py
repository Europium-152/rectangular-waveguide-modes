"""Figure 6 of the paper: 15 cm, 60 deg WR-90 E-plane bends (TE10, electric field parallel to the narrow wall and
to the bend plane), hyperbolic secant and constant curvature, compared with COMSOL Multiphysics 6.2.

COMSOL setup (model file: comsol/hyperbolic_secant_E_bend_60_deg_lowfreq_parametric.java):
  RF module, Electromagnetic Waves Frequency Domain, 3D, vacuum, PEC walls; rectangular TE10 ports with slit
  (domain-backed) condition and 2 cm PML behind each port; free tetrahedral mesh, hmax = c/60 GHz/5;
  GMRES + geometric multigrid, rtol 1e-2; 18-60 GHz in 2 GHz steps; exported quantity emw.S21dB.
Prints the difference statistics and the runtime of the coupled-mode calculation.
"""
import time, os
import numpy as np
from common import *

HERE = os.path.dirname(os.path.abspath(__file__))
a, b, mode_in = ORIENTATIONS['TE10 E-plane']
L = 0.15
files = {'hyperbolic secant': 'comsol/15cm_60deg_hyperbolic_curve_side_a_short_e_field_horizontal',
         'constant': 'comsol/15cm_60deg_constant_curve_side_a_short_e_field_horizontal'}


def read_comsol(fname):
    lines = open(os.path.join(HERE, fname)).read().splitlines()
    hdr = [l for l in lines if l.startswith('% X')][0]
    f = np.array([float(s.split('freq=')[1].split()[0]) for s in hdr.split('emw.S21dB')[1:]]) * 1e9
    vals = np.array([l for l in lines if not l.startswith('%')][0].split()[3:], dtype=float)
    return f, -vals  # loss in dB


results = {}
for name, fname in files.items():
    f, comsol = read_comsol(fname)
    kappa = sech_curvature(60, L) if name == 'hyperbolic secant' else constant_curvature(60, L)
    t0 = time.time()
    cmt = np.array([mode_conversion_loss_db(fi, a, b, mode_in, L, kappa) for fi in f])
    dt = time.time() - t0
    d = cmt - comsol
    results[name] = (f, comsol, cmt)
    print(f"{name:18s}: {len(f)} frequencies in {dt:.1f} s; max|CMT-COMSOL| = {np.max(np.abs(d)):.4f} dB "
          f"(rms {np.sqrt(np.mean(d**2)):.4f}, mean {np.mean(d):+.4f}); max loss = {comsol.max():.2f} dB")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    for (name, (f, comsol, cmt)), c in zip(results.items(), ['C1', 'C0']):
        ax[0].plot(f / 1e9, cmt, color=c, label=f'{name} (coupled-mode theory)')
        ax[0].plot(f / 1e9, comsol, 'x', color=c, label=f'{name} (COMSOL)')
        ax[1].plot(f / 1e9, comsol - cmt, '+', color=c)
    ax[0].set_ylabel('Mode conversion loss [dB]'); ax[0].legend(fontsize=7, frameon=False)
    ax[1].set_ylabel('COMSOL - CMT [dB]'); ax[1].set_xlabel('Frequency [GHz]')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig6_comsol_comparison.png'), dpi=200)
    print('saved fig6_comsol_comparison.png')
except ImportError:
    pass
