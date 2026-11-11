"""Ridge follower: trace a printed line by stepping along its tangent and re-centring on the ink
profile taken along the normal. At sharp turns the step direction is searched over +-75 deg."""
import numpy as np
from scipy.ndimage import map_coordinates

def profile(dark, q, n, h, tvec, ntan=2):
    ts = np.arange(-h, h + 0.25, 0.5)
    acc = np.zeros_like(ts)
    for s in np.linspace(-ntan, ntan, 2 * ntan + 1):
        pts = q[None, :] + ts[:, None] * n[None, :] + s * tvec[None, :]
        acc += map_coordinates(dark, [pts[:, 1], pts[:, 0]], order=1, mode='constant')
    return ts, acc / (2 * ntan + 1)

def centre(ts, pr, thr, maxw, h):
    ink = pr > thr
    if not ink.any():
        return None
    idx = np.where(ink)[0]
    br = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[br + 1]]; ends = np.r_[idx[br], idx[-1]]
    mid = len(ts) // 2
    k = int(np.argmin([0 if s <= mid <= e else min(abs(s - mid), abs(e - mid)) for s, e in zip(starts, ends)]))
    s, e = starts[k], ends[k]
    w = (e - s) * 0.5
    if s == 0 or e == len(ts) - 1 or w > maxw:
        return None
    wts = pr[s:e + 1] - thr / 2
    tc = (wts * ts[s:e + 1]).sum() / wts.sum()
    if abs(tc) > 0.6 * h:
        return None
    return tc, w

def rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

def follow(dark, p0, v0, xend, step=3.0, h=10.0, thr=90, maxw=7.0, maxpts=5000, stop=None, inertia=0.3,
           search=True, coast=True):
    """Returns rows (x, y, width, ok, vx, vy). ok = 0: point not re-centred (coasting through a gap,
    crossing or gridline)."""
    p = np.array(p0, float); v = np.array(v0, float); v /= np.linalg.norm(v)
    xdir = np.sign(xend - p[0]) or 1
    out = []
    for _ in range(maxpts):
        found = None
        angles = [0.0] + ([a for k in range(1, 6) for a in (np.radians(15 * k), -np.radians(15 * k))] if search else [])
        for a in angles:
            d = rot(v, a)
            q = p + step * d
            n = np.array([-d[1], d[0]])
            ts, pr = profile(dark, q, n, h, d)
            c = centre(ts, pr, thr, maxw, h)
            if c is not None:
                found = (q + c[0] * n, c[1]); break
        if found is not None:
            q, w = found
            dn = (q - p) / np.linalg.norm(q - p)
            v = inertia * v + (1 - inertia) * dn; v /= np.linalg.norm(v)
            ok = True
        else:
            if not coast:
                break
            q = p + step * v; w = 0.0; ok = False
        p = q
        out.append((p[0], p[1], w, ok, v[0], v[1]))
        if (xend - p[0]) * xdir <= 0:
            break
        if stop is not None and stop(p):
            break
    return np.array(out)
