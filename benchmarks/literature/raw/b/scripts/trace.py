import numpy as np
from scipy import ndimage
from dig import *

def erase_crosses(m, crosses, half=11):
    m = m.copy()
    for x, y in crosses:
        x, y = int(round(x)), int(round(y))
        m[y - half:y + half + 1, x - half:x + half + 1] = False
    return m

def assign(m, guides_px, tol):
    """guides_px: dict name->(xs, ys) polyline in px. Each dark pixel goes to the nearest guide (vertical distance)
    if within tol px. Returns dict name->bool mask."""
    ys, xs = np.nonzero(m)
    names = list(guides_px)
    dist = np.full((len(names), xs.size), np.inf)
    for k, nm in enumerate(names):
        gx, gy = guides_px[nm]
        inside = (xs >= gx.min()) & (xs <= gx.max())
        dist[k, inside] = np.abs(ys[inside] - np.interp(xs[inside], gx, gy))
    best = np.argmin(dist, 0)
    out = {}
    for k, nm in enumerate(names):
        sel = (best == k) & (dist[k] <= tol[nm] if isinstance(tol, dict) else dist[k] <= tol)
        mk = np.zeros_like(m); mk[ys[sel], xs[sel]] = True
        out[nm] = mk
    return out

def centerline(mk):
    """per column mean y of the pixels of one curve. Returns xs, ys, counts"""
    cols = np.flatnonzero(mk.any(0))
    yc = np.array([np.nonzero(mk[:, c])[0].mean() for c in cols])
    cnt = np.array([mk[:, c].sum() for c in cols])
    return cols.astype(float), yc, cnt

def segments(cols, gap=2):
    br = np.flatnonzero(np.diff(cols) > gap)
    starts = np.r_[0, br + 1]; ends = np.r_[br, cols.size - 1]
    return list(zip(starts, ends))

def sample(cols, yc, step, mode='grid', minseg=3, xgrid=None):
    """mode 'grid': points at xgrid (px) with local average over +-2 px where ink exists.
    mode 'dash': one point per dash (segment of consecutive columns): mean x, mean y."""
    pts = []
    if mode == 'dash':
        for s, e in segments(cols):
            if e - s + 1 < minseg:
                continue
            pts.append((cols[s:e + 1].mean(), yc[s:e + 1].mean(), e - s + 1))
    else:
        for xg in xgrid:
            sel = np.abs(cols - xg) <= 2
            if sel.sum() >= 3:
                # local linear fit to get value at xg
                p = np.polyfit(cols[sel], yc[sel], 1)
                pts.append((xg, np.polyval(p, xg), sel.sum()))
    return pts

def assign_euclid(m, guides_px, tol, step=0.5):
    """Each dark pixel goes to the guide polyline at the smallest Euclidean distance, if within tol px."""
    from scipy.spatial import cKDTree
    ys, xs = np.nonzero(m)
    P = np.c_[xs, ys].astype(float)
    names = list(guides_px)
    dist = np.full((len(names), xs.size), np.inf)
    for k, nm in enumerate(names):
        gx, gy = guides_px[nm]
        seg = []
        for i in range(len(gx) - 1):
            n = int(np.ceil(np.hypot(gx[i + 1] - gx[i], gy[i + 1] - gy[i]) / step)) + 1
            t = np.linspace(0, 1, n)
            seg.append(np.c_[gx[i] + t * (gx[i + 1] - gx[i]), gy[i] + t * (gy[i + 1] - gy[i])])
        tree = cKDTree(np.vstack(seg))
        dist[k] = tree.query(P)[0]
    best = np.argmin(dist, 0)
    out = {}
    for k, nm in enumerate(names):
        sel = (best == k) & (dist[k] <= tol)
        mk = np.zeros_like(m); mk[ys[sel], xs[sel]] = True
        out[nm] = mk
    return out
