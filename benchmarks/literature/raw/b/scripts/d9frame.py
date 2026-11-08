import numpy as np
from d9cal import D, vruns

def track(axis, lo, hi, start, maxw=10, win=8):
    """Follow a frame line. axis 'h': for each column t in lo..hi, centre row of the run near the previous value.
    axis 'v': for each row t, centre column. Returns t array, centre array (NaN where not found)."""
    ts = np.arange(lo, hi); cs = np.full(ts.size, np.nan); prev = start
    for i, t in enumerate(ts):
        a0 = int(prev - win); a1 = int(prev + win + 1)
        prof = D[a0:a1, t] if axis == 'h' else D[t, a0:a1]
        runs = [r for r in vruns(prof) if r[1] - r[0] + 1 <= maxw]
        if not runs:
            continue
        r = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 + a0 - prev))
        c = (r[0] + r[1]) / 2 + a0
        if abs(c - prev) <= 3:
            cs[i] = c; prev = c
    return ts, cs

def ticks(axis, ts, cs, side, L0=8, L1=40, minlen=15):
    """Tick marks perpendicular to a frame line: for each t, the length of the dark run starting next to the frame on
    side (+1/-1). Returns list of tick centres (t)."""
    lens = np.zeros(ts.size)
    ok = np.isfinite(cs)
    cs = np.interp(ts, ts[ok], cs[ok])
    for i, (t, c) in enumerate(zip(ts, cs)):
        if np.isnan(c):
            continue
        c = int(round(c))
        if axis == 'h':
            prof = D[c + side * L0: c + side * L1: side, t] if side > 0 else D[c - L0: c - L1: -1, t]
        else:
            prof = D[t, c + L0: c + L1] if side > 0 else D[t, c - L0: c - L1: -1]
        n = 0
        for v in prof:
            if v: n += 1
            else: break
        lens[i] = n
    hit = lens >= minlen
    out = []
    for s, e in vruns(hit):
        w = lens[s:e + 1]
        out.append(((ts[s:e + 1] * w).sum() / w.sum(), e - s + 1, w.max()))
    return out
