"""Calibration of Doane & Anderson 1989 Fig. 9 (photographed, curved page): Coons patch through the four frame
lines, each parameterized by its tick marks (PCHIP between ticks). Pixel coordinates: half-resolution crop
d9_h.png = page 7 rendered at 300 dpi, cropped at (3200, 8350, 9200, 12400), then resized by 1/2."""
import numpy as np
from scipy.interpolate import PchipInterpolator, griddata
from d9frame import track, ticks

tb, cb = track('h', 420, 2795, 1668)
tt, ct = track('h', 425, 2790, 127)
tl, cl = track('v', 128, 1670, 429)
tr, cr = track('v', 150, 1720, 2768)

def smooth_line(ts, cs, w=15):
    ok = np.isfinite(cs)
    t, c = ts[ok], cs[ok]
    k = np.ones(w) / w
    cs2 = np.convolve(np.pad(c, w // 2, mode='edge'), k, mode='valid')
    return lambda q: np.interp(q, t, cs2), t[0], t[-1]

yb, *_ = smooth_line(tb, cb)   # y of bottom frame vs x
yt, *_ = smooth_line(tt, ct)
xl, *_ = smooth_line(tl, cl)   # x of left frame vs y
xr, *_ = smooth_line(tr, cr)

def corner(fh, fv, x0, y0):
    """intersection of horizontal-ish line y = fh(x) and vertical-ish line x = fv(y), by fixed point iteration."""
    x, y = x0, y0
    for _ in range(50):
        y = float(fh(x)); x = float(fv(y))
    return x, y

BL = corner(yb, xl, 415, 1669)
TL = corner(yt, xl, 427, 128)
BR = corner(yb, xr, 2797, 1726)
TR = corner(yt, xr, 2770, 148)
# tick positions (detected by ticks(); values listed in the CSV header)
BT = [810.508, 1194.5, 1589.467, 1986.8, 2398.0]          # bottom, f = 15.5..17.5
TT = [819.0, 1205.0, 1597.474, 1991.5, 2388.0]            # top, f = 15.5..17.5
LT = [1277.0, 888.0, 488.5]                               # left, P = 0.5, 1.0, 1.5
RT = [1319.377, 917.0, 509.867]                           # right, P = 0.5, 1.0, 1.5
FK = np.array([15, 15.5, 16, 16.5, 17, 17.5, 18.0]); PK = np.array([0, 0.5, 1, 1.5, 2.0])
bx = PchipInterpolator(FK, [BL[0]] + BT + [BR[0]])
tx = PchipInterpolator(FK, [TL[0]] + TT + [TR[0]])
ly = PchipInterpolator(PK, [BL[1]] + LT + [TL[1]])
ry = PchipInterpolator(PK, [BR[1]] + RT + [TR[1]])
B = lambda f: np.array([bx(f), yb(bx(f))]); T = lambda f: np.array([tx(f), yt(tx(f))])
L = lambda p: np.array([xl(ly(p)), ly(p)]); R = lambda p: np.array([xr(ry(p)), ry(p)])
P00, P10, P01, P11 = map(np.array, (BL, BR, TL, TR))

def fwd(f, p):
    f = np.asarray(f, float); p = np.asarray(p, float)
    u = (f - 15) / 3; v = p / 2
    out = ((1 - v) * B(f) + v * T(f) + (1 - u) * L(p) + u * R(p)
           - ((1 - u) * (1 - v))[None] * P00[:, None] - (u * (1 - v))[None] * P10[:, None]
           - ((1 - u) * v)[None] * P01[:, None] - (u * v)[None] * P11[:, None])
    return out

FG, PG = np.meshgrid(np.linspace(14.9, 18.1, 321), np.linspace(-0.1, 2.1, 221))
XY = fwd(FG.ravel(), PG.ravel())

def inv(x, y):
    """pixel -> (f GHz, P %) by linear interpolation of the dense forward map."""
    q = np.c_[np.atleast_1d(x), np.atleast_1d(y)]
    f = griddata(XY.T, FG.ravel(), q, method='linear'); p = griddata(XY.T, PG.ravel(), q, method='linear')
    return f, p

def local_scale(f, p, h=1e-3):
    """pixels per GHz along f and pixels per % along P at (f, p)."""
    df = np.linalg.norm(fwd(np.array([f + h]), np.array([p]))[:, 0] - fwd(np.array([f - h]), np.array([p]))[:, 0]) / (2 * h)
    dp = np.linalg.norm(fwd(np.array([f]), np.array([p + h]))[:, 0] - fwd(np.array([f]), np.array([p - h]))[:, 0]) / (2 * h)
    return df, dp

if __name__ == '__main__':
    print('corners BL TL BR TR', BL, TL, BR, TR)
    # self check: ticks map back
    for f, x in zip([15.5, 16, 16.5, 17, 17.5], BT):
        print('bottom tick', f, inv(x, yb(x)))
    for p, y in zip([0.5, 1, 1.5], RT):
        print('right tick', p, inv(xr(y), y))
    print('scale at centre px/GHz, px/%', local_scale(16.5, 1.0))
