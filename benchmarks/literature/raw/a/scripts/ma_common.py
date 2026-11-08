import sys; sys.path.insert(0, '/home/claude/wt/L1a-scratch')
from dig import *
from csvout import write_csv
from scipy import ndimage
import math, pickle

def find_axes(img, xl, yb, xr, yt, w=30):
    """img: page image; approximate left axis x=xl, bottom axis y=yb, right frame x=xr, top frame y=yt."""
    I = ink(img)
    pv, tv, nv = edges_v(I, xl - w // 2, xl + w // 2, yt + 20, yb - 20)
    ph, th, nh = edges_h(I, yb - w // 2, yb + w // 2, xl + 20, xr - 20)
    av = math.degrees(math.atan(pv[0])); ah = math.degrees(math.atan(-ph[0]))
    return pv, ph, av, ah, tv, th

def ring_template(r, t=2):
    s = int(r + t + 2); yy, xx = np.mgrid[-s:s + 1, -s:s + 1]; d = np.hypot(xx, yy)
    T = np.where(np.abs(d - r) <= t / 2 + 0.5, 1.0, np.where(d < r - t, -0.5, 0.0))
    return T

def find_rings(I, r, thr, region, t=2):
    import cv2
    x0, y0, x1, y1 = region
    sub = I[y0:y1, x0:x1].astype(np.float32)
    T = ring_template(r, t).astype(np.float32)
    R = cv2.filter2D(sub, -1, T, borderType=cv2.BORDER_CONSTANT)
    ring_n = (T > 0).sum()
    pk = (R == ndimage.maximum_filter(R, size=int(2 * r))) & (R > thr * ring_n)
    ys, xs = np.nonzero(pk)
    return [(x + x0, y + y0, R[y, x] / ring_n) for x, y in zip(xs, ys)]

def refine_ring(I, cx, cy, r, t=3):
    """least-squares circle fit to ink pixels within the annulus r+-t around (cx,cy)."""
    s = int(r + t + 1); x0, y0 = int(round(cx)), int(round(cy))
    sub = I[y0 - s:y0 + s + 1, x0 - s:x0 + s + 1]
    yy, xx = np.nonzero(sub); xx = xx + x0 - s; yy = yy + y0 - s
    d = np.hypot(xx - cx, yy - cy); k = np.abs(d - r) <= t
    xx = xx[k].astype(float); yy = yy[k].astype(float)
    A = np.c_[2 * xx, 2 * yy, np.ones_like(xx)]; b = xx ** 2 + yy ** 2
    (a_, b_, c_), *_ = np.linalg.lstsq(A, b, rcond=None)
    return a_, b_, np.sqrt(c_ + a_ ** 2 + b_ ** 2), len(xx)
