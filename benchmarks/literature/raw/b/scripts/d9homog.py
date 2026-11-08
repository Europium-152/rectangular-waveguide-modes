"""Alternative calibration of Doane Fig. 9: one projective map (homography) fitted by least squares to all tick
marks and the four frame corners. Used to estimate the calibration-model uncertainty of the Coons patch."""
import numpy as np
from d9map import BT, TT, LT, RT, BL, BR, TL, TR, yb, yt, xl, xr
data, pix = [], []
for f, x in zip([15.5, 16, 16.5, 17, 17.5], BT): data.append((f, 0)); pix.append((x, float(yb(x))))
for f, x in zip([15.5, 16, 16.5, 17, 17.5], TT): data.append((f, 2)); pix.append((x, float(yt(x))))
for p, y in zip([0.5, 1, 1.5], LT): data.append((15, p)); pix.append((float(xl(y)), y))
for p, y in zip([0.5, 1, 1.5], RT): data.append((18, p)); pix.append((float(xr(y)), y))
for d, q in [((15, 0), BL), ((18, 0), BR), ((15, 2), TL), ((18, 2), TR)]: data.append(d); pix.append(q)
data = np.array(data, float); pix = np.array(pix, float)

def fit_h(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u]); A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, Vt = np.linalg.svd(np.array(A)); return Vt[-1].reshape(3, 3)

# normalize data coordinates for conditioning
dn = (data - [16.5, 1.0]) / [1.5, 1.0]
Hm = fit_h(pix, dn)   # pixel -> normalized data

def inv_h(x, y):
    q = Hm @ np.vstack([np.atleast_1d(x), np.atleast_1d(y), np.ones(np.size(x))])
    q = q[:2] / q[2]
    return q[0] * 1.5 + 16.5, q[1] * 1.0 + 1.0

if __name__ == '__main__':
    f, p = inv_h(pix[:, 0], pix[:, 1])
    res = np.c_[f - data[:, 0], p - data[:, 1]]
    for d, r in zip(data, res): print(d, np.round(r, 4))
    print('max |res| f %.4f GHz, P %.4f %%' % tuple(np.abs(res).max(0)))
