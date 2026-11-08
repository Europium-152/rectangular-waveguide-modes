import numpy as np
from PIL import Image, ImageDraw
from d9map import *
from d9cal import D, a
from trace import assign_euclid

def seg_mask(shape, p0, p1, w):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    p0 = np.array(p0, float); p1 = np.array(p1, float); d = p1 - p0
    t = np.clip(((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / (d @ d), 0, 1)
    return np.hypot(xx - p0[0] - t * d[0], yy - p0[1] - t * d[1]) <= w

H, W = D.shape
# interior of the frame (6 px margin)
yy, xx = np.mgrid[0:H, 0:W]
inside = (yy > yt(xx) + 6) & (yy < yb(xx) - 6) & (xx > xl(yy) + 6) & (xx < xr(yy) - 6)
m = D & inside
# labels and pointer arrows (half-res pixel boxes / segments, read from the overlay at 1/2 scale)
for x0, x1, y0, y1 in [(650, 990, 560, 650), (1600, 1840, 656, 730), (2290, 2710, 1360, 1446)]:
    m[y0:y1, x0:x1] = False
for p0, p1, w in [((800, 664), (596, 880), 16), ((1642, 752), (1578, 984), 16), ((2215, 1400), (2300, 1400), 10),
                  ((2220, 1400), (2158, 1600), 16)]:
    m &= ~seg_mask(D.shape, p0, p1, w)
# tick marks inside the frame
for x in BT: m &= ~seg_mask(D.shape, (x, yb(x)), (x, yb(x) - 85), 6)
for x in TT: m &= ~seg_mask(D.shape, (x, yt(x)), (x, yt(x) + 85), 6)
for y in LT: m &= ~seg_mask(D.shape, (xl(y), y), (xl(y) + 85, y), 6)
for y in RT: m &= ~seg_mask(D.shape, (xr(y), y), (xr(y) - 85, y), 6)
G = {
 'constant': [(15.0, 1.36), (15.2, 1.1), (15.4, 0.8), (15.6, 0.52), (15.8, 0.3), (16.0, 0.16), (16.11, 0.094), (16.26, 0.068), (16.41, 0.047), (16.55, 0.06), (16.69, 0.112), (16.89, 0.27), (17.02, 0.37), (17.27, 0.63), (17.52, 0.99), (17.78, 1.38), (18.0, 1.73)],
 'cosine': [(15, 0.35), (15.5, 0.53), (16, 0.7), (16.5, 0.86), (17, 0.98), (17.5, 1.04), (18, 1.02)],
 'triangular': [(15, 0.05), (15.5, 0.06), (16, 0.075), (16.5, 0.08), (17, 0.08), (17.5, 0.08), (18, 0.08)],
}
gpx = {}
for k, v in G.items():
    f = np.array([p[0] for p in v]); p = np.array([p[1] for p in v])
    ff = np.interp(np.linspace(0, len(f) - 1, 40 * len(f)), np.arange(len(f)), f)
    pp = np.interp(np.linspace(0, len(f) - 1, 40 * len(f)), np.arange(len(f)), p)
    xy = fwd(ff, pp); gpx[k] = (xy[0], xy[1])

# the triangular curve is straight in the photo: guide directly in pixels
gpx['triangular'] = (np.array([420., 900, 1350, 2790]), np.array([1632., 1618, 1606, 1606]))
