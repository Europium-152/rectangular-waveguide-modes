import numpy as np
from PIL import Image
from dig import vruns
a = np.array(Image.open('../d9_h.png').convert('L')).astype(float)
D = a < 100

def frame_line(axis, lo, hi, band, guess):
    """axis 'v': vertical line, scan rows lo..hi, look in columns band; returns fit x = p0*y + p1.
    axis 'h': horizontal line, scan cols, rows band; returns y = p0*x + p1."""
    pts = []
    for t in range(lo, hi):
        prof = D[t, band[0]:band[1]] if axis == 'v' else D[band[0]:band[1], t]
        runs = vruns(prof)
        if not runs:
            continue
        r = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 + band[0] - guess))
        if 3 <= r[1] - r[0] + 1 <= 14:
            pts.append((t, (r[0] + r[1]) / 2 + band[0]))
    pts = np.array(pts)
    p = np.polyfit(pts[:, 0], pts[:, 1], 1)
    res = pts[:, 1] - np.polyval(p, pts[:, 0])
    good = np.abs(res) < 2.5
    p = np.polyfit(pts[good, 0], pts[good, 1], 1)
    res = pts[good, 1] - np.polyval(p, pts[good, 0])
    return p, res.std(), good.sum()

if __name__ == '__main__':
    for nm, args in {'left': ('v', 150, 1650, (380, 470), 420), 'right': ('v', 170, 1700, (2730, 2830), 2780),
                     'top': ('h', 450, 2750, (90, 190), 135), 'bottom': ('h', 450, 2750, (1630, 1760), 1690)}.items():
        print(nm, frame_line(*args))
