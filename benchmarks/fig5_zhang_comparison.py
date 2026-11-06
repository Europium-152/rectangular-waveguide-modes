"""Figure 5 of the paper: normalized TE10 and TE20 output power of a 90 deg H-plane constant-curvature bend
(44 x 30 mm waveguide, 9.5 GHz) versus curvature radius, as in Zhang et al., 2012."""
import os
import numpy as np
import wrmodes as wr
from common import PAPER_SETTINGS

HERE = os.path.dirname(os.path.abspath(__file__))
a, b, f, theta = 44e-3, 30e-3, 9.5e9, np.radians(90)
radii = np.arange(30e-3, 260e-3, 5e-3)
te10, te20 = [], []
for R0 in radii:
    kappa = lambda z, R0=R0: np.ones_like(np.asarray(z, dtype=float)) / R0
    modes, res = wr.solve_mode_coupling(f, a, b, ('TE', 1, 0), length=R0 * theta, curvature=kappa, **PAPER_SETTINGS)
    names = [m[0] for m in modes]
    te10.append(np.abs(res.y[names.index('TE1,0'), -1]) ** 2)
    te20.append(np.abs(res.y[names.index('TE2,0'), -1]) ** 2)
np.savetxt(os.path.join(HERE, 'fig5_zhang_comparison.txt'), np.c_[radii * 1e3, te10, te20], header='R [mm]  P_TE10  P_TE20')
print('saved fig5_zhang_comparison.txt')
