import numpy as np
from dig import vruns

def find_frame(d, x0, x1, y0, y1, frac=0.6):
    sub = d[y0:y1, x0:x1]
    rows = np.flatnonzero(sub.sum(1) > frac * (x1 - x0)) + y0
    cols = np.flatnonzero(sub.sum(0) > frac * (y1 - y0)) + x0
    return rows, cols

def tick_runs(d, axis, lo, hi, a, b, minfill):
    """axis 'x': ticks along a horizontal frame: scan columns lo..hi in rows a..b (a band next to the frame) and return
    centres of columns with >= minfill dark pixels. axis 'y': scan rows lo..hi in columns a..b."""
    if axis == 'x':
        v = d[a:b, lo:hi].sum(0) >= minfill
    else:
        v = d[lo:hi, a:b].sum(1) >= minfill
    return [((s + e) / 2 + lo, e - s + 1) for s, e in vruns(v)]
