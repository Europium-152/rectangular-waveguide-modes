"""Small pixel-digitizing toolkit (L1a). Scratch code, not part of the repository."""
import numpy as np
from PIL import Image, ImageDraw

def load_gray(path):
    return Image.open(path).convert('L')

def ink(img, thr=128):
    return np.array(img) < thr

def runs(idx):
    """Group sorted integer indices into runs -> list of (start, end)."""
    idx = list(idx)
    if not idx:
        return []
    out = []; s = p = idx[0]
    for i in idx[1:]:
        if i - p > 1:
            out.append((s, p)); s = i
        p = i
    out.append((s, p))
    return out

def axis_vertical_slope(I, x0, x1, y0, y1, step=10):
    """Thick vertical axis between columns x0..x1: centre x per row, slope dx/dy."""
    ys, xs = [], []
    for y in range(y0, y1, step):
        r = runs(np.nonzero(I[y, x0:x1])[0] + x0)
        if not r: continue
        best = max(r, key=lambda t: t[1] - t[0])
        if best[1] - best[0] < 3: continue
        ys.append(y); xs.append((best[0] + best[1]) / 2)
    p = np.polyfit(ys, xs, 1)
    res = np.array(xs) - np.polyval(p, ys)
    keep = np.abs(res) < 2.5
    p = np.polyfit(np.array(ys)[keep], np.array(xs)[keep], 1)
    return p  # x = p0*y + p1

def axis_horizontal_slope(I, y0, y1, x0, x1, step=10):
    xs, ys = [], []
    for x in range(x0, x1, step):
        r = runs(np.nonzero(I[y0:y1, x])[0] + y0)
        if not r: continue
        best = max(r, key=lambda t: t[1] - t[0])
        if best[1] - best[0] < 3: continue
        xs.append(x); ys.append((best[0] + best[1]) / 2)
    p = np.polyfit(xs, ys, 1)
    res = np.array(ys) - np.polyval(p, xs)
    keep = np.abs(res) < 2.5
    p = np.polyfit(np.array(xs)[keep], np.array(ys)[keep], 1)
    return p  # y = p0*x + p1

def deskew(img, angle_deg, center):
    """Rotate grayscale image by angle_deg (counter-clockwise positive, PIL convention) about center."""
    return img.rotate(angle_deg, resample=Image.BICUBIC, center=center, fillcolor=255)

def tick_centres(I, band_rows=None, band_cols=None, along='y', min_count=1):
    """Centres of tick marks. along='y': ticks on a vertical axis, scan rows within band_cols columns."""
    if along == 'y':
        c0, c1 = band_cols
        prof = I[:, c0:c1].sum(1)
        r0, r1 = band_rows
        idx = [i for i in range(r0, r1) if prof[i] >= min_count]
    else:
        r0, r1 = band_rows
        prof = I[r0:r1, :].sum(0)
        c0, c1 = band_cols
        idx = [i for i in range(c0, c1) if prof[i] >= min_count]
    out = []
    for s, e in runs(idx):
        w = prof[s:e + 1].astype(float); i = np.arange(s, e + 1)
        out.append(((w * i).sum() / w.sum(), e - s + 1))
    return out

def linfit(px, val):
    px = np.asarray(px, float); val = np.asarray(val, float)
    m, c = np.polyfit(px, val, 1)
    return m, c, val - (m * px + c)

def trace(I, path_px, halfwidth, mask=None, binlen=3.0):
    """Collect ink pixels within halfwidth (px, perpendicular) of a rough polyline path_px [(x,y),...],
    project onto the path, bin by arc length and return bin centroids (x, y, n, perp_std)."""
    P = np.asarray(path_px, float)
    J = I.copy()
    if mask is not None:
        J &= ~mask
    ys, xs = np.nonzero(J)
    pts = np.stack([xs, ys], 1).astype(float)
    # bounding box filter
    lo = P.min(0) - halfwidth - 2; hi = P.max(0) + halfwidth + 2
    sel = np.all((pts >= lo) & (pts <= hi), 1)
    pts = pts[sel]
    seglen = np.hypot(*(P[1:] - P[:-1]).T)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    best_d = np.full(len(pts), np.inf); best_s = np.zeros(len(pts))
    for k in range(len(P) - 1):
        a = P[k]; b = P[k + 1]; d = b - a; L2 = (d * d).sum()
        t = np.clip(((pts - a) @ d) / L2, 0, 1)
        proj = a + t[:, None] * d
        dist = np.hypot(*(pts - proj).T)
        better = dist < best_d
        best_d[better] = dist[better]; best_s[better] = cum[k] + t[better] * seglen[k]
    keep = best_d <= halfwidth
    pts = pts[keep]; s = best_s[keep]
    out = []
    nb = int(np.ceil(cum[-1] / binlen))
    bi = np.minimum((s / binlen).astype(int), nb - 1)
    for b in range(nb):
        q = pts[bi == b]
        if len(q) < 2: continue
        c = q.mean(0)
        out.append((c[0], c[1], len(q)))
    return np.array(out)

def overlay(img, pts_list, colors, path, crop=None, scale=1, r=2):
    im = img.convert('RGB')
    d = ImageDraw.Draw(im)
    for pts, col in zip(pts_list, colors):
        for x, y in pts:
            d.ellipse([x - r, y - r, x + r, y + r], outline=col)
    if crop: im = im.crop(crop)
    if scale != 1: im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)), Image.LANCZOS)
    im.save(path)

def find_crosses(I, arm=9, min_h=14, min_v=14, region=None):
    """Plus-shaped markers: pixels where both a horizontal and vertical ink run through them are long."""
    from scipy import ndimage
    H = ndimage.uniform_filter1d(I.astype(float), 2 * arm + 1, axis=1) * (2 * arm + 1)
    V = ndimage.uniform_filter1d(I.astype(float), 2 * arm + 1, axis=0) * (2 * arm + 1)
    # allow 1-px thickness wobble: max over +-1 in the other direction
    H = ndimage.maximum_filter1d(H, 3, axis=0); V = ndimage.maximum_filter1d(V, 3, axis=1)
    C = (H >= min_h) & (V >= min_v) & I
    if region is not None:
        x0, y0, x1, y1 = region
        M = np.zeros_like(C); M[y0:y1, x0:x1] = True; C &= M
    lab, n = ndimage.label(ndimage.binary_dilation(C, iterations=2))
    cents = ndimage.center_of_mass(C, lab, range(1, n + 1))
    return [(c[1], c[0]) for c in cents]

def refine_cross(I, cx, cy, arm=12):
    """Centre of a plus marker: x from the vertical arm, y from the horizontal arm."""
    x0, y0 = int(round(cx)), int(round(cy))
    # vertical arm: columns of pixels in rows away from centre (|dy|>=4)
    sub = I[y0 - arm:y0 + arm + 1, x0 - arm:x0 + arm + 1]
    yy, xx = np.mgrid[-arm:arm + 1, -arm:arm + 1]
    va = sub & (np.abs(yy) >= 4) & (np.abs(xx) <= 4)
    ha = sub & (np.abs(xx) >= 4) & (np.abs(yy) <= 4)
    x = x0 + xx[va].mean() if va.any() else np.nan
    y = y0 + yy[ha].mean() if ha.any() else np.nan
    return x, y, va.sum(), ha.sum()

def edges_v(I, x0, x1, y0, y1):
    """Thick vertical line in columns x0..x1, rows y0..y1: fits of left/right edges, median thickness."""
    Y = []; L = []; R = []
    for y in range(y0, y1):
        r = runs(np.nonzero(I[y, x0:x1])[0] + x0)
        if not r: continue
        best = max(r, key=lambda t: t[1] - t[0]); Y.append(y); L.append(best[0]); R.append(best[1])
    Y = np.array(Y); L = np.array(L); R = np.array(R); th = R - L
    k = np.abs(th - np.median(th)) <= 1
    pl = np.polyfit(Y[k], L[k], 1); pr = np.polyfit(Y[k], R[k], 1)
    return (pl + pr) / 2, np.median(th) + 1, k.sum()

def edges_h(I, y0, y1, x0, x1):
    X = []; T = []; B = []
    for x in range(x0, x1):
        r = runs(np.nonzero(I[y0:y1, x])[0] + y0)
        if not r: continue
        best = max(r, key=lambda t: t[1] - t[0]); X.append(x); T.append(best[0]); B.append(best[1])
    X = np.array(X); T = np.array(T); B = np.array(B); th = B - T
    k = np.abs(th - np.median(th)) <= 1
    pt = np.polyfit(X[k], T[k], 1); pb = np.polyfit(X[k], B[k], 1)
    return (pt + pb) / 2, np.median(th) + 1, k.sum()

def trace2(I, path_px, halfwidth, mask=None, binlen=3.0):
    """Like trace(), but also returns the half thickness of the line (perpendicular) in each bin."""
    P = np.asarray(path_px, float)
    J = I.copy()
    if mask is not None:
        J &= ~mask
    ys, xs = np.nonzero(J)
    pts = np.stack([xs, ys], 1).astype(float)
    lo = P.min(0) - halfwidth - 2; hi = P.max(0) + halfwidth + 2
    pts = pts[np.all((pts >= lo) & (pts <= hi), 1)]
    seglen = np.hypot(*(P[1:] - P[:-1]).T)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    best_d = np.full(len(pts), np.inf); best_s = np.zeros(len(pts)); best_k = np.zeros(len(pts), int)
    for k in range(len(P) - 1):
        a = P[k]; b = P[k + 1]; d = b - a; L2 = (d * d).sum()
        if L2 == 0: continue
        t = np.clip(((pts - a) @ d) / L2, 0, 1)
        proj = a + t[:, None] * d
        dist = np.hypot(*(pts - proj).T)
        better = dist < best_d
        best_d[better] = dist[better]; best_s[better] = cum[k] + t[better] * seglen[k]; best_k[better] = k
    keep = best_d <= halfwidth
    pts = pts[keep]; s = best_s[keep]; kk = best_k[keep]
    out = []
    nb = int(np.ceil(cum[-1] / binlen))
    bi = np.minimum((s / binlen).astype(int), nb - 1)
    for b in range(nb):
        sel = bi == b
        q = pts[sel]
        if len(q) < 2: continue
        c = q.mean(0)
        k = np.bincount(kk[sel]).argmax()
        d = P[k + 1] - P[k]; d = d / np.hypot(*d); nrm = np.array([-d[1], d[0]])
        off = (q - c) @ nrm
        hwid = 0.5 * (off.max() - off.min() + 1)
        out.append((c[0], c[1], len(q), hwid))
    return np.array(out)

def clean_outliers(pts, win=30.0, tol=2.0, deg=2):
    """Drop bin centroids farther than tol px from a local polynomial (in arc length) through their neighbours."""
    P = pts[:, :2]
    s = np.concatenate([[0], np.cumsum(np.hypot(*(P[1:] - P[:-1]).T))])
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        nb = (np.abs(s - s[i]) <= win) & (np.arange(len(P)) != i)
        if nb.sum() < deg + 2: continue
        fx = np.polyfit(s[nb], P[nb, 0], deg); fy = np.polyfit(s[nb], P[nb, 1], deg)
        if np.hypot(np.polyval(fx, s[i]) - P[i, 0], np.polyval(fy, s[i]) - P[i, 1]) > tol:
            keep[i] = False
    return pts[keep]

def dots(I, path_px, halfwidth, mask=None, amin=5, amax=60):
    """Centroids and half sizes of isolated dots (dotted curves) within halfwidth of path."""
    from scipy import ndimage
    J = I.copy()
    if mask is not None: J &= ~mask
    lab, n = ndimage.label(J, structure=np.ones((3, 3)))
    P = np.asarray(path_px, float)
    lo = P.min(0) - halfwidth; hi = P.max(0) + halfwidth
    out = []
    for i, s in enumerate(ndimage.find_objects(lab)):
        cy = (s[0].start + s[0].stop - 1) / 2; cx = (s[1].start + s[1].stop - 1) / 2
        if not (lo[0] <= cx <= hi[0] and lo[1] <= cy <= hi[1]): continue
        yy, xx = np.nonzero(lab[s] == i + 1)
        a = len(xx)
        if a < amin or a > amax: continue
        c = np.array([xx.mean() + s[1].start, yy.mean() + s[0].start])
        # distance to path
        dmin = np.inf
        for k in range(len(P) - 1):
            A = P[k]; B = P[k + 1]; d = B - A; t = np.clip(((c - A) @ d) / (d @ d), 0, 1)
            dmin = min(dmin, np.hypot(*(c - A - t * d)))
        if dmin <= halfwidth:
            out.append((c[0], c[1], a, 0.5 * max(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1)))
    return np.array(sorted(out))

def to_grid(xd, yd, hwpx, grid, mx, my, maxgap, cal_px=1.0):
    """Interpolate traced samples (data units, sorted by x) onto grid. Returns rows (x, y, u_y).
    u_y combines: half line width (perpendicular, converted to vertical at fixed x), calibration cal_px in
    both axes, and, for grid points more than 3 px from a sample, the difference between linear and
    quadratic interpolation."""
    o = np.argsort(xd); xd = xd[o]; yd = yd[o]; hwpx = hwpx[o]
    rows = []
    for g in grid:
        j = np.searchsorted(xd, g)
        if j == 0 or j == len(xd): continue
        x0, x1 = xd[j - 1], xd[j]
        if x1 - x0 > maxgap: continue
        t = (g - x0) / (x1 - x0) if x1 > x0 else 0.5
        y = yd[j - 1] + t * (yd[j] - yd[j - 1])
        # local slope in pixels (dy_px/dx_px) from neighbours +-3
        a = max(0, j - 4); b = min(len(xd), j + 3)
        if xd[b - 1] > xd[a]:
            sl = np.polyfit(xd[a:b], yd[a:b], 1)[0]
        else:
            sl = 0.0
        spx = sl * mx / my
        hw = 0.5 * (hwpx[j - 1] + hwpx[j])
        upx = np.sqrt((hw * np.sqrt(1 + spx ** 2)) ** 2 + cal_px ** 2 + (spx * cal_px) ** 2)
        u = abs(my) * upx
        gap_px = min(g - x0, x1 - g) / mx
        if gap_px > 3 and b - a >= 4:
            q = np.polyfit(xd[a:b], yd[a:b], 2)
            u = np.hypot(u, abs(np.polyval(q, g) - y))
        rows.append((g, y, u))
    return rows

def follow(I, start, direction, step=3.0, hw=6, maxgap=30, box=None, width_max=8, nfit=8, mask=None, maxsteps=5000, max_off=2.0, quad=False, gap_slope=0.15):
    """Track a curve from start (x, y) in the given direction. At each step, look at ink pixels in a thin
    slab perpendicular to the current direction, step px ahead; take the contiguous run of perpendicular
    offsets closest to the prediction. Runs wider than width_max (crossings, markers) and missing ink
    (dash gaps) are bridged by extrapolation. Returns array of (x, y, found_flag)."""
    J = I if mask is None else (I & ~mask)
    H, W = J.shape
    p = np.array(start, float); d = np.array(direction, float); d /= np.hypot(*d)
    pts = [(p[0], p[1], 1)]
    gap = 0.0
    for _ in range(maxsteps):
        q = p + step * d
        n = np.array([-d[1], d[0]])
        if box is not None:
            x0, y0, x1, y1 = box
            if not (x0 <= q[0] <= x1 and y0 <= q[1] <= y1): break
        # candidate pixels
        r = int(hw + step + 2)
        xa, xb = int(q[0]) - r, int(q[0]) + r + 1; ya, yb = int(q[1]) - r, int(q[1]) + r + 1
        if xa < 0 or ya < 0 or xb > W or yb > H: break
        sub = J[ya:yb, xa:xb]
        yy, xx = np.nonzero(sub)
        found = False
        if len(xx):
            v = np.stack([xx + xa, yy + ya], 1) - q
            along = v @ d; perp = v @ n
            sel = (np.abs(along) <= 1.0) & (np.abs(perp) <= hw)
            if sel.any():
                pv = np.sort(perp[sel])
                # contiguous runs in perp (gap > 1.5 px splits)
                splits = np.nonzero(np.diff(pv) > 1.5)[0]
                groups = np.split(pv, splits + 1)
                g = min(groups, key=lambda g_: abs(g_.mean()))
                wdt = g.max() - g.min() + 1
                off = 0.5 * (g.max() + g.min())
                if wdt <= width_max and abs(off) <= max_off + gap_slope * gap:
                    q = q + off * n
                    found = True
        if found:
            gap = 0.0
        else:
            gap += step
            if gap > maxgap: break
        pts.append((q[0], q[1], int(found)))
        p = q
        # update direction from last found points
        F = np.array([t for t in pts[-nfit * 4:] if t[2]])[-nfit:]
        if len(F) >= 3:
            dd = F[-1, :2] - F[0, :2]
            if np.hypot(*dd) > 0:
                dnew = dd / np.hypot(*dd)
                if quad and len(F) >= 5 and np.hypot(*dd) > 10:
                    ss = np.concatenate([[0], np.cumsum(np.hypot(*(F[1:, :2] - F[:-1, :2]).T))])
                    px_ = np.polyfit(ss, F[:, 0], 2); py_ = np.polyfit(ss, F[:, 1], 2)
                    # direction at the current position (extrapolated arc length)
                    se = ss[-1] + np.hypot(*(q - F[-1, :2]))
                    dq = np.array([np.polyval(np.polyder(px_), se), np.polyval(np.polyder(py_), se)])
                    if np.hypot(*dq) > 0:
                        dq /= np.hypot(*dq)
                        if dq @ dnew > 0.7: dnew = dq
                if dnew @ d > 0.5:
                    d = dnew
    pts = np.array(pts)
    # drop trailing extrapolated points
    last = np.nonzero(pts[:, 2])[0].max()
    return pts[:last + 1]
