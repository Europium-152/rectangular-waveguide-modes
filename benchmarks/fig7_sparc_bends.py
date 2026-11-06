"""Figure 7 / Table 1 of the paper: mode-conversion losses of a 60 cm 60 deg bend and a 40 cm 90 deg bend in WR-90,
constant and hyperbolic secant curvature, for the four orientations of figure 1, 18-90 GHz in 0.5 GHz steps.
Takes about 30 min on a laptop (the tall-guide E-plane cases with up to 27 modes dominate the cost).
Results are saved to fig7_sparc_bends.npz and a summary table is printed.
"""
import time, os, sys
import numpy as np
from common import *

HERE = os.path.dirname(os.path.abspath(__file__))
freqs = np.arange(18e9, 90.01e9, 0.5e9)
BENDS = {'60cm_60deg': (60, 0.60), '40cm_90deg': (90, 0.40)}
quick = '--quick' in sys.argv  # 2 GHz steps, for a fast check
if quick:
    freqs = np.arange(18e9, 90.01e9, 2e9)

out = {}
t0 = time.time()
for bend, (theta, L) in BENDS.items():
    for profile in ['constant', 'sech']:
        kappa = constant_curvature(theta, L) if profile == 'constant' else sech_curvature(theta, L)
        for orient, (a, b, mode_in) in ORIENTATIONS.items():
            loss = np.array([mode_conversion_loss_db(f, a, b, mode_in, L, kappa) for f in freqs])
            out[f'{bend}/{profile}/{orient}'] = loss
            above = freqs[loss > 1.0]
            print(f"{bend} {profile:8s} {orient:13s}: peak {loss.max():6.2f} dB at {freqs[np.argmax(loss)]/1e9:.1f} GHz; "
                  f">1 dB over {len(above)*(freqs[1]-freqs[0])/1e9:.1f} GHz  ({time.time()-t0:.0f} s)", flush=True)
np.savez(os.path.join(HERE, 'fig7_sparc_bends.npz'), freqs=freqs, **out)
