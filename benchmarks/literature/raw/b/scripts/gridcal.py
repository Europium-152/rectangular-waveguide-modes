"""Affine calibration from dotted gridlines (L1b): fit each gridline to the dark pixels in a band around its
approximate position, intersect, and fit an affine map data -> pixel by least squares."""
import numpy as np

def fit_line(d, approx, orient, span, band=5, maxrun=3):
    """orient 'v': x = p0*y + p1 using dark pixels in |x - approx| <= band for y in span; 'h': y = p0*x + p1.
    Only short runs (dots) are used, so that crossing curves are ignored."""
    pts = []
    lo, hi = span
    for t in range(lo, hi):
        a0 = int(approx - band); a1 = int(approx + band + 1)
        prof = d[t, a0:a1] if orient == 'v' else d[a0:a1, t]
        idx = np.flatnonzero(prof)
        if idx.size == 0 or idx.size > maxrun:
            continue
        if np.ptp(idx) > maxrun:
            continue
        pts.append((t, idx.mean() + a0))
    pts = np.array(pts)
    p = np.polyfit(pts[:, 0], pts[:, 1], 1)
    for _ in range(2):
        res = pts[:, 1] - np.polyval(p, pts[:, 0]); good = np.abs(res) < 1.5
        p = np.polyfit(pts[good, 0], pts[good, 1], 1)
    res = pts[good, 1] - np.polyval(p, pts[good, 0])
    return p, good.sum(), res.std()

def calibrate(d, vlines, hlines, vspan, hspan, band=5):
    """vlines: list of (x_value, approx_px); hlines: list of (y_value, approx_px).
    Returns forward affine (data->px), inverse (px->data), residual stats and line fits."""
    V = [(val, fit_line(d, ap, 'v', vspan, band)) for val, ap in vlines]
    Hh = [(val, fit_line(d, ap, 'h', hspan, band)) for val, ap in hlines]
    P, D = [], []
    for xv, (pv, nv, sv) in V:
        for yv, (ph, nh, sh) in Hh:
            # x = pv0*y + pv1 ; y = ph0*x + ph1
            y = (ph[0] * pv[1] + ph[1]) / (1 - ph[0] * pv[0]); x = pv[0] * y + pv[1]
            P.append((x, y)); D.append((xv, yv))
    P = np.array(P); D = np.array(D)
    A = np.c_[D, np.ones(len(D))]
    cx_, *_ = np.linalg.lstsq(A, P[:, 0], rcond=None); cy_, *_ = np.linalg.lstsq(A, P[:, 1], rcond=None)
    M = np.array([[cx_[0], cx_[1]], [cy_[0], cy_[1]]]); t = np.array([cx_[2], cy_[2]])
    res = P - (D @ M.T + t)
    Minv = np.linalg.inv(M)
    fwd = lambda X, Y: (np.c_[np.atleast_1d(X), np.atleast_1d(Y)] @ M.T + t).T
    inv = lambda x, y: ((np.c_[np.atleast_1d(x), np.atleast_1d(y)] - t) @ Minv.T).T
    return dict(fwd=fwd, inv=inv, M=M, t=t, res=np.abs(res).max(0), V=V, H=Hh, P=P, D=D)
