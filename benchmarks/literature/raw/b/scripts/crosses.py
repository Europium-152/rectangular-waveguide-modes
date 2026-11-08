import numpy as np, sys
from dig import *

def find_crosses(d, x0, x1, y0, y1, minlen=14, maxlen=32, exclude_cols=()):
    hits = []
    for x in range(x0, x1):
        if any(abs(x - c) <= 2 for c in exclude_cols):
            continue
        for s, e in vruns(d[y0:y1, x]):
            L = e - s + 1
            if minlen <= L <= maxlen:
                hits.append((x, s + y0, e + y0))
    # cluster by proximity
    hits.sort()
    clusters = []
    for h in hits:
        for c in clusters:
            if abs(h[0] - c[-1][0]) <= 3 and abs((h[1] + h[2]) / 2 - (c[-1][1] + c[-1][2]) / 2) <= 8:
                c.append(h); break
        else:
            clusters.append([h])
    out = []
    for c in clusters:
        xs = np.array([h[0] for h in c]); mids = np.array([(h[1] + h[2]) / 2 for h in c])
        lens = np.array([h[2] - h[1] + 1 for h in c])
        out.append((xs.mean(), mids.mean(), lens.max(), len(c)))
    return out
