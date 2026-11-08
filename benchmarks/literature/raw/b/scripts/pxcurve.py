"""Pixel-space centre lines and conversion through a 2-D calibration (L1b)."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

def remove_small(m, nmin=8):
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sz = ndimage.sum(m, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool); keep[1:] = sz >= nmin
    return keep[lab]

def centre_px(mk, step=4, steep=1.5):
    out = []
    cols = np.flatnonzero(mk.any(0))
    if cols.size == 0:
        return np.zeros((0, 3))
    yc = np.array([np.nonzero(mk[:, c])[0].mean() for c in cols])
    for c in range(cols.min(), cols.max() + 1, step):
        sel = np.abs(cols - c) <= 2
        if sel.sum() < 3:
            continue
        wide = np.abs(cols - c) <= 6
        s = np.polyfit(cols[wide], yc[wide], 1)[0] if wide.sum() >= 4 else 0
        if abs(s) > steep:
            continue
        p = np.polyfit(cols[sel], yc[sel], 1)
        out.append((c, np.polyval(p, c), s))
    cand = []
    for r in np.flatnonzero(mk.any(1)):
        xs_r = np.flatnonzero(mk[r]); br = np.flatnonzero(np.diff(xs_r) > 3)
        for a0, b0 in zip(np.r_[0, br + 1], np.r_[br, xs_r.size - 1]):
            cand.append((r, xs_r[a0:b0 + 1].mean()))
    cand = np.array(cand)
    for r, xv0 in cand:
        if int(r) % step:
            continue
        sel = (np.abs(cand[:, 0] - r) <= 3) & (np.abs(cand[:, 1] - xv0) <= 8)
        if sel.sum() < 5 or np.ptp(cand[sel, 0]) < 4:
            continue
        q = np.polyfit(cand[sel, 0], cand[sel, 1], 1)
        if abs(q[0]) * steep >= 1:
            continue
        out.append((np.polyval(q, r), r, 1 / q[0] if abs(q[0]) > 1e-3 else 1e3))
    return np.array(out) if out else np.zeros((0, 3))

def to_data(pts_px, inv, M, px_u=2.0, cal_px=1.0, ucap=10.0, others=None, cross_r=5):
    """pts_px: array (x, y, slope). Returns list of (X, Y, uY, uX). Uncertainty: px_u px of reading (both directions,
    converted through the local slope) + cal_px px of calibration, using the scales of the affine map M."""
    sx = np.hypot(M[0, 0], M[1, 0])   # px per x-unit
    sy = np.hypot(M[0, 1], M[1, 1])   # px per y-unit
    out = []
    for x, y, s in pts_px:
        if others is not None and any(t.query_ball_point((x, y), cross_r) for t in others):
            continue
        X, Y = inv(x, y)
        slope_data = -s * sx / sy  # data slope (pixel y grows downward)
        uY = min(np.hypot(px_u / sy, px_u / sx * abs(slope_data)) + cal_px / sy, ucap)
        uX = (px_u + cal_px) / sx
        out.append((float(X[0]), float(Y[0]), float(uY), float(uX)))
    return sorted(out)
