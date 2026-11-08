import sys; sys.path.insert(0, '/home/claude/wt/L1a-scratch')
from dig import *
from csvout import write_csv
from scipy import ndimage
import pickle
img = load_gray('/home/claude/wt/L1a-scratch/rt4-000.png'); I = ink(img)
lab, n = ndimage.label(I, structure=np.ones((3, 3)))
SL = ndimage.find_objects(lab)
area = ndimage.sum(I, lab, range(1, n + 1))
def region_info(x0, y0, x1, y1):
    """grid-dot rows and columns inside a panel: small components (area <= 8)"""
    rows = []; cols = []
    for i, s in enumerate(SL):
        if s[0].start < y0 or s[0].stop > y1 or s[1].start < x0 or s[1].stop > x1: continue
        if area[i] <= 8 and (s[0].stop - s[0].start) <= 4 and (s[1].stop - s[1].start) <= 4:
            yy, xx = np.nonzero(lab[s] == i + 1)
            rows.append(yy.mean() + s[0].start); cols.append(xx.mean() + s[1].start)
    return np.array(rows), np.array(cols)
def clusters(v, tol=3, minn=8):
    v = np.sort(v); out = []; cur = [v[0]]
    for a in v[1:]:
        if a - cur[-1] <= tol: cur.append(a)
        else:
            if len(cur) >= minn: out.append((np.mean(cur), len(cur)))
            cur = [a]
    if len(cur) >= minn: out.append((np.mean(cur), len(cur)))
    return out
def long_mask(x0, y0, x1, y1, minlen=40):
    M = np.zeros_like(I)
    for i, s in enumerate(SL):
        if s[0].start < y0 or s[0].stop > y1 or s[1].start < x0 or s[1].stop > x1: continue
        if (s[1].stop - s[1].start) >= minlen or (s[0].stop - s[0].start) >= minlen:
            M[s] |= lab[s] == i + 1
    return M
def grid_calib(x0, y0, x1, y1, xvals, yvals):
    """Fit lines to the grid-dot rows/cols, intersect them, and fit an affine map pixel -> data.
    xvals/yvals: data values of the detected grid columns/rows in order (None to skip one)."""
    pts = []
    for i, s in enumerate(SL):
        if s[0].start < y0 or s[0].stop > y1 or s[1].start < x0 or s[1].stop > x1: continue
        if area[i] <= 8 and (s[0].stop - s[0].start) <= 4 and (s[1].stop - s[1].start) <= 4:
            yy, xx = np.nonzero(lab[s] == i + 1); pts.append((xx.mean() + s[1].start, yy.mean() + s[0].start))
    pts = np.array(pts)
    rc = clusters(pts[:, 1]); cc = clusters(pts[:, 0])
    hl = []; vl = []
    for (r, k), v in zip(rc, yvals):
        if v is None: continue
        q = pts[np.abs(pts[:, 1] - r) <= 3]; p = np.polyfit(q[:, 0], q[:, 1], 1); hl.append((p, v, np.abs(q[:, 1] - np.polyval(p, q[:, 0])).max()))
    for (c, k), v in zip(cc, xvals):
        if v is None: continue
        q = pts[np.abs(pts[:, 0] - c) <= 3]; p = np.polyfit(q[:, 1], q[:, 0], 1); vl.append((p, v, np.abs(q[:, 0] - np.polyval(p, q[:, 1])).max()))
    P = []
    for ph, Y, _ in hl:
        for pv, X, _ in vl:
            # y = ph0*x+ph1 ; x = pv0*y+pv1
            y = (ph[0] * pv[1] + ph[1]) / (1 - ph[0] * pv[0]); x = pv[0] * y + pv[1]
            P.append((x, y, X, Y))
    P = np.array(P)
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    cX, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None); cY, *_ = np.linalg.lstsq(A, P[:, 3], rcond=None)
    res = np.c_[A @ cX - P[:, 2], A @ cY - P[:, 3]]
    slopes = [np.degrees(np.arctan(h[0][0])) for h in hl] + [-np.degrees(np.arctan(v[0][0])) for v in vl]
    return cX, cY, P, res, slopes
def panel(x0, y0, x1, y1, cX, cY, xr, yr, pad=4):
    """data coords of every pixel in the box; frame mask; curve components inside the frame."""
    yy, xx = np.mgrid[y0:y1, x0:x1]
    X = cX[0] * xx + cX[1] * yy + cX[2]; Y = cY[0] * xx + cY[1] * yy + cY[2]
    mx = np.hypot(cX[0], cX[1]); my = np.hypot(cY[0], cY[1])
    inside = (X > xr[0] + pad * mx) & (X < xr[1] - pad * mx) & (Y < yr[1] - pad * my) & (Y > yr[0] + pad * my)
    J = np.zeros_like(I); J[y0:y1, x0:x1] = I[y0:y1, x0:x1] & inside
    l2, n2 = ndimage.label(J, structure=np.ones((3, 3)))
    comps = []
    for i, s in enumerate(ndimage.find_objects(l2)):
        w = s[1].stop - s[1].start; h = s[0].stop - s[0].start
        if w >= 40 or h >= 40:
            m = l2[s] == i + 1
            yy_, xx_ = np.nonzero(m)
            comps.append(dict(sl=s, mask=m, n=int(m.sum()), x0=s[1].start, x1=s[1].stop, y0=s[0].start, y1=s[0].stop,
                              Xr=(cX[0] * xx_.min() + cX[2] + cX[1] * (yy_.mean() + s[0].start) + cX[0] * s[1].start, None)))
    return J, comps, mx, my
def comp_trace(c, hw=5):
    """column centroids of a component -> path; then trace2 restricted to the component."""
    s = c['sl']; m = c['mask']
    C = np.zeros_like(I); C[s] = m
    path = []
    for x in range(s[1].start, s[1].stop, 2):
        ys = np.nonzero(C[:, x])[0]
        if len(ys): path.append((x, ys.mean()))
    path = np.array(path)
    t = trace2(C, path, hw, binlen=3.0)
    return clean_outliers(t, win=30, tol=1.5)
def todata(cX, cY, px, py): return cX[0] * px + cX[1] * py + cX[2], cY[0] * px + cY[1] * py + cY[2]
