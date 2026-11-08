"""Spot check of the merged benchmark files against the page images (card L1).

For 3 points per figure the ink of the curve is read directly on the page: the page is cut along the line of
constant x through the chosen grid point, the ink runs on that cut are listed, and the run of the curve is
identified on a zoomed crop (written to OUT, one PNG per figure, with the final values marked). The value read is
the centre of that run. Calibrations: the tick or grid calibration of one digitizer (stated per figure below; the
two digitizers' calibrations agree within their residuals). The Doane photograph uses a bilinear Coons patch
through the frame/tick points of L1a's header.

Needs the reference PDFs (not in the repository): set REF to the folder that holds them.
    REF=/path/to/references OUT=/tmp/spot python3 benchmarks/literature/scripts/spot_check.py
"""
import csv
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
LIT = os.path.dirname(HERE)
REF = os.environ.get('REF', '/mnt/user-data/uploads/wrmodes_arbitrary_bends/references')
OUT = os.environ.get('OUT', '/tmp/l1_spot')

PDF = dict(
    ales='Alessandri et al. - 1994 - Rigorous mode matching analysis of mitered E-plane.pdf',
    ma='Analysis and design of mitered H‐plane bends in rectangular waveguides by combining the '
       'mode‐matchin.pdf',
    rei='Reiter and Arndt - 1995 - Rigorous analysis of arbitrarily shaped H- and E-plane discontinuities in '
        'rectangular waveguides by.pdf',
    doane='Doane and Anderson - 1989 - Oversized rectangular waveguides with mode-free be.pdf',
)


def page(key, p, native=False):
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f'{key}{p}')
    if native:
        png = stem + '-000.png'
        if not os.path.exists(png):
            subprocess.run(['pdfimages', '-png', '-f', str(p), '-l', str(p), os.path.join(REF, PDF[key]), stem],
                           check=True)
    else:
        png = f'{stem}-{p}.png'
        if not os.path.exists(png):
            subprocess.run(['pdftoppm', '-r', '300', '-f', str(p), '-l', str(p), '-png',
                            os.path.join(REF, PDF[key]), stem], check=True)
            # pdftoppm pads the page number to the page count width
            for f in os.listdir(OUT):
                if f.startswith(os.path.basename(stem) + '-') and f.endswith('.png') and 'spot' not in f:
                    os.rename(os.path.join(OUT, f), png)
    return png


def lin(px, val):
    """1-D least-squares map value -> px."""
    k = np.polyfit(val, px, 1)
    return lambda v: np.polyval(k, v)


class Sep:
    """Separable linear calibration (x from x ticks, y from y ticks)."""

    def __init__(self, xpx, xval, ypx, yval):
        self.fx, self.fy = lin(xpx, xval), lin(ypx, yval)

    def topx(self, x, y):
        return np.array([self.fx(x), self.fy(y)])


class Aff:
    """Affine calibration px = M (x, y) + t (L1b, Reiter Fig. 3)."""

    def __init__(self, M, t):
        self.M, self.t = np.array(M), np.array(t)

    def topx(self, x, y):
        return self.M @ np.array([x, y]) + self.t


class Coons:
    """Bilinear Coons patch through the frame/tick points of the Doane photograph (L1a header)."""

    def __init__(self):
        X = [15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0]
        self.bot = dict(zip(X, [(966.2, 2807.8), (1157.0, 2805.7), (1341.0, 2807.7), (1531.0, 2812.7),
                                (1721.5, 2819.7), (1919.0, 2827.7), (2112.0, 2835.0)]))
        self.top = dict(zip(X, [(973.0, 2064.2), (1161.0, 2066.8), (1346.5, 2069.4), (1535.0, 2071.7),
                                (1724.0, 2073.6), (1914.0, 2074.7), (2101.7, 2074.8)]))
        Y = [0, 0.5, 1, 1.5, 2]
        self.left = dict(zip(Y, [(966.2, 2807.8), (968.6, 2617.0), (970.5, 2430.5), (972.0, 2238.5),
                                 (973.0, 2064.2)]))
        self.right = dict(zip(Y, [(2112.0, 2835.0), (2105.6, 2637.0), (2099.0, 2444.0), (2096.5, 2248.5),
                                  (2101.7, 2074.8)]))

    @staticmethod
    def _curve(d, v):
        k = sorted(d)
        pts = np.array([d[q] for q in k])
        return np.array([np.interp(v, k, pts[:, 0]), np.interp(v, k, pts[:, 1])])

    def topx(self, x, y):
        u, v = (x - 15.0) / 3.0, y / 2.0
        B, T = self._curve(self.bot, x), self._curve(self.top, x)
        L, R = self._curve(self.left, y), self._curve(self.right, y)
        c00, c10 = np.array(self.bot[15.0]), np.array(self.bot[18.0])
        c01, c11 = np.array(self.top[15.0]), np.array(self.top[18.0])
        return ((1 - v) * B + v * T + (1 - u) * L + u * R
                - ((1 - u) * (1 - v) * c00 + u * (1 - v) * c10 + (1 - u) * v * c01 + u * v * c11))


def runs_along(img, cal, x, y0, y1, n=4000, thr=128):
    """Ink runs on the cut at constant x from y0 to y1: list of (centre, width) in y units."""
    ys = np.linspace(y0, y1, n)
    ink = []
    for y in ys:
        px, py = cal.topx(x, y)
        ink.append(img[int(round(py)), int(round(px))] < thr)
    ink = np.array(ink)
    out, i = [], 0
    while i < n:
        if ink[i]:
            j = i
            while j + 1 < n and ink[j + 1]:
                j += 1
            out.append((0.5 * (ys[i] + ys[j]), abs(ys[j] - ys[i])))
            i = j + 1
        else:
            i += 1
    return out


def final_value(fn, curve, x):
    with open(os.path.join(LIT, fn)) as fh:
        rows = list(csv.DictReader(l for l in fh if not l.startswith('#')))
    best = min((r for r in rows if r['curve'] == curve), key=lambda r: abs(float(r['x']) - x))
    return float(best['x']), float(best['y']), float(best['uncertainty']), best['source']


# figure: (file, page key, page, native, calibration, y window half width, [(curve, x, pick)])
# pick: index into the ink runs found within the window (sorted by distance to the final value) after
# visual identification on the crop; 0 = nearest run.
FIGS = [
    ('alessandri1994_fig4.csv', 'ales', 2, False,
     Sep([1540.0, 1678.5, 1816.5, 1954.5, 2090.5, 2228.0], [1, 1.2, 1.4, 1.6, 1.8, 2.0],
         [328.0, 448.5, 569.5, 691.0, 812.0], [0, -10, -20, -30, -40]), 1.5,
     [('d/b=1.41', 1.5), ('d/b=0.707', 1.3), ('d/b=0.707', 1.43)]),
    ('alessandri1994_fig5.csv', 'ales', 2, False,
     Sep([1507.5, 1646.5, 1783.5, 1922.0, 2059.0, 2196.0], [1, 1.2, 1.4, 1.6, 1.8, 2],
         [1117.5, 1215.5, 1312.5, 1410.5, 1507.5, 1604.0], [0, -10, -20, -30, -40, -50]), 1.5,
     [('d/b=0.9', 1.5), ('d/b=0.85', 1.2), ('d/b=0.87', 1.7)]),
    ('doane1989_fig9.csv', 'doane', 7, True, Coons(), 0.04,
     [('constant', 15.25), ('cosine', 16.5), ('triangular', 17.5)]),
    ('ma1997_fig5.csv', 'ma', 4, False,
     Sep([468.5, 610.25, 748.75, 889.0, 1030.25], [10, 12, 14, 16, 18],
         [2176.75, 2283.25, 2387.75, 2495.0], [0.8, 0.6, 0.4, 0.2]), 0.03,
     [('3/6/12 terms', 11.0), ('3/6/12 terms', 15.0), ('3/6/12 terms', 16.5)]),
    ('ma1997_fig6.csv', 'ma', 4, False,
     Sep([1592.0, 1730.0, 1866.0, 2004.5, 2144.0], [10, 12, 14, 16, 18],
         [228.0, 286.0, 347.0, 405.25, 465.75, 523.25, 583.25], [-2, -4, -6, -8, -10, -12, -14]), 0.6,
     [('40/60 steps', 11.0), ('10 steps', 11.5), ('40/60 steps', 14.0)]),
    ('ma1997_fig8.csv', 'ma', 5, False,
     Sep([393.25, 538.75, 683.5, 830.25, 976.25], [10, 12, 14, 16, 18],
         [247.25, 319.5, 392.0, 461.0, 533.0, 604.25], [-5, -10, -15, -20, -25, -30]), 1.2,
     [('(a)', 12.0), ('(b)', 13.0), ('(c)', 13.95, 0.6)]),
    ('ma1997_fig9.csv', 'ma', 5, False,
     Sep([395.5, 544.5, 689.25, 838.75, 986.25], [10, 12, 14, 16, 18],
         [2179.0, 2251.5, 2323.75, 2393.25, 2464.75, 2536.25], [-5, -10, -15, -20, -25, -30]), 1.2,
     [('(a)=(b)=(c)', 11.0), ('(a)=(b)=(c)', 14.0), ('(a)=(b)=(c)', 17.0)]),
    ('ma1997_fig11.csv', 'ma', 6, False,
     Sep([1522.0, 1675.0, 1828.25, 1981.75, 2135.75, 2289.0], [1, 1.2, 1.4, 1.6, 1.8, 2.0],
         [184.5, 293.25, 402.75, 511.25, 619.0, 729.0], [0, -10, -20, -30, -40, -50]), 1.2,
     [('c/a=0', 1.3), ('c/a=0.60', 1.5), ('c/a=0.70', 1.25)]),
]


def main():
    print('| Figure | Curve | x | final y +- u (source) | read on page | |final - read| | within u |')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    res = []
    for fn, key, p, native, cal, win, pts in FIGS + REI:
        if not pts:
            continue
        png = page(key, p, native)
        img = np.asarray(Image.open(png).convert('L'))
        marks = []
        for pt in pts:
            curve, x = pt[:2]
            w = pt[2] if len(pt) > 2 else win
            # dashed and dotted lines: move to the nearest grid point whose cut meets ink of the curve
            for k in (0, 1, -1, 2, -2, 3, -3, 4, -4):
                xf, yf, u, src = final_value(fn, curve, x + k * STEP[fn])
                runs = runs_along(img, cal, xf, yf + w, yf - w)
                if runs:
                    break
            runs.sort(key=lambda r: abs(r[0] - yf))
            read = runs[0][0] if runs else float('nan')
            ok = abs(read - yf) <= u
            res.append((fn, curve, xf, yf, u, src, read, ok))
            print(f'| {fn} | {curve} | {xf:g} | {yf:.4g} +- {u:.2g} ({src}) | {read:.4g} '
                  f'(run width {runs[0][1]:.2g}) | {abs(read - yf):.2g} | {"yes" if ok else "NO"} |')
            marks.append((xf, yf))
        crop_marks(img, cal, marks, os.path.join(OUT, fn.replace('.csv', '_spot.png')))
    return res


STEP = {'alessandri1994_fig4.csv': 0.005, 'alessandri1994_fig5.csv': 0.005, 'doane1989_fig9.csv': 0.025,
        'ma1997_fig5.csv': 0.05, 'ma1997_fig6.csv': 0.05, 'ma1997_fig8.csv': 0.05, 'ma1997_fig9.csv': 0.05,
        'ma1997_fig11.csv': 0.005, 'reiter1995_fig3a.csv': 0.05, 'reiter1995_fig3b.csv': 0.05,
        'reiter1995_fig3c.csv': 0.05, 'reiter1995_fig3d.csv': 0.05, 'reiter1995_fig4.csv': 0.025}


def crop_marks(img, cal, marks, out):
    im = Image.fromarray(img).convert('RGB')
    for x, y in marks:
        px, py = cal.topx(x, y)
        box = (int(px) - 40, int(py) - 40, int(px) + 40, int(py) + 40)
        c = im.crop(box).resize((320, 320), Image.NEAREST)
        d = ImageDraw.Draw(c)
        d.line((140, 160, 180, 160), fill=(255, 0, 255))
        d.line((160, 140, 160, 180), fill=(255, 0, 255))
        c.save(out.replace('_spot.png', f'_spot_{x:g}.png'))


REI = [
    ('reiter1995_fig3a.csv', 'rei', 4, False,
     Aff([[78.063, -0.1092], [-0.4344, -10.8679]], [-308.21, 537.25]), 1.2,
     [('gamma=90', 13.5), ('gamma=120', 15.0), ('gamma=150', 17.5)]),
    ('reiter1995_fig3b.csv', 'rei', 4, False,
     Aff([[77.3752, -0.0729], [-0.2995, -12.3486]], [664.03, 234.38]), 1.2,
     [('gamma=105', 13.0), ('gamma=120', 16.5), ('gamma=135', 17.0)]),
    ('reiter1995_fig3c.csv', 'rei', 4, False,
     Aff([[77.557, -0.068], [-0.3057, -10.836]], [-294.33, 1312.36]), 1.2,
     [('gamma=90', 13.5), ('gamma=120', 15.0), ('gamma=150', 17.5)]),
    ('reiter1995_fig3d.csv', 'rei', 4, False,
     Aff([[77.5828, -0.091], [-0.2266, -12.3112]], [662.35, 885.37]), 1.2,
     [('gamma=105', 14.0), ('gamma=120', 15.0), ('gamma=135', 13.0)]),
    # Fig. 4 on the native 339 dpi scan with L1a's tick calibration
    ('reiter1995_fig4.csv', 'rei', 4, True,
     Sep([573.0, 631.2, 728.0, 825.2, 923.0, 1020.0, 1117.0, 1214.0, 1311.0, 1389.0],
         [8.2, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.4],
         [2222.0, 2304.5, 2386.5, 2549.9, 2713.0], [-10, -20, -30, -50, -70]), 1.2,
     [('l=0', 9.0), ('l=b', 10.0), ('l=0', 12.0)]),
]

if __name__ == '__main__':
    main()
