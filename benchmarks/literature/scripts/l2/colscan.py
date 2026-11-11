"""Column-run scanner (pass b): in every column, the dark runs (gray < thr) are listed and the run nearest the
prediction is taken; its centre (ink-weighted) is the curve ordinate at that column."""
import numpy as np


def runs(col, thr):
    d = col < thr
    if not d.any():
        return []
    idx = np.where(d)[0]
    br = np.where(np.diff(idx) > 1)[0]
    return list(zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]))


def scan(im, x0, x1, y0, ylo, yhi, thr=140, maxjump=8, maxlen=60, step=1, minlen=2, k=3, slope0=0.0):
    """Track from column x0 (seed ordinate y0) to x1. Returns rows (x, y, runlength); y = nan where no run
    within maxjump of the prediction."""
    out = []
    hist = []
    xs = range(x0, x1 + (1 if x1 >= x0 else -1), step if x1 >= x0 else -step)
    for x in xs:
        col = im[ylo:yhi, x]
        rs = [(a, b) for a, b in runs(col, thr) if minlen <= b - a + 1 <= maxlen]
        if len(hist) >= k:
            hx, hy = np.array(hist[-k:]).T
            c = np.polyfit(hx, hy, 1)
            pred = np.polyval(c, x)
        elif hist:
            pred = hist[-1][1] + slope0 * (x - hist[-1][0])
        else:
            pred = y0
        best = None
        for a, b in rs:
            # the run may extend beyond the prediction (steep flank): distance to the run, not to its centre
            dist = 0 if a + ylo <= pred <= b + ylo else min(abs(a + ylo - pred), abs(b + ylo - pred))
            if best is None or dist < best[0]:
                w = 255 - col[a:b + 1]
                c = (w * np.arange(a, b + 1)).sum() / w.sum() + ylo
                best = (dist, c, b - a + 1)
        if best is not None and best[0] <= maxjump:
            out.append((x, best[1], best[2]))
            hist.append((x, best[1]))
        else:
            out.append((x, np.nan, 0))
    return np.array(out)


def scan_guided(im, guide, maxjump=10, thr=140, maxlen=80, minlen=2, step=1):
    """Column scan along a coarse hand-placed guide polyline (x increasing): in each column the run nearest to the
    guide ordinate is taken if it is within maxjump of it."""
    g = np.array(guide, float)
    out = []
    for x in range(int(np.ceil(g[0, 0])), int(np.floor(g[-1, 0])) + 1, step):
        gy = np.interp(x, g[:, 0], g[:, 1])
        ylo = max(int(gy - maxjump - maxlen), 0)
        col = im[ylo:int(gy + maxjump + maxlen), x]
        best = None
        for a, b in runs(col, thr):
            if not (minlen <= b - a + 1 <= maxlen):
                continue
            d = 0 if a + ylo <= gy <= b + ylo else min(abs(a + ylo - gy), abs(b + ylo - gy))
            if best is None or d < best[0]:
                w = 255 - col[a:b + 1]
                best = (d, (w * np.arange(a, b + 1)).sum() / w.sum() + ylo, b - a + 1)
        if best is not None and best[0] <= maxjump:
            out.append((x, best[1], best[2]))
    return np.array(out).reshape(-1, 3)
