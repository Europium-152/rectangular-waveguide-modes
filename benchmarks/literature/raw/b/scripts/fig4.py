"""Alessandri, Mongiardo & Sorrentino 1994, Fig. 4 (page 409 = PDF page 2), digitized by L1b."""
import numpy as np
from PIL import Image, ImageDraw
from dig import *
from trace import erase_crosses, assign
from crosses import find_crosses
from curvepts import curve_points, dot_points
im, d = load('../ales-2.png')
XC = ([1540.0, 1678.5, 1816.5, 1954.5, 2090.5, 2228.0], [1, 1.2, 1.4, 1.6, 1.8, 2.0])
YC = ([328.0, 448.5, 569.5, 691.0, 812.0], [0, -10, -20, -30, -40])
fx, rx, cx = fitcal(*XC); fy, ry, cy = fitcal(*YC)
px = lambda f: (np.asarray(f) - cx[1]) / cx[0]; py = lambda s: (np.asarray(s) - cy[1]) / cy[0]
m = np.zeros_like(d)
X0, X1, Y0, Y1 = 1554, 2224, 330, 801
m[Y0:Y1, X0:X1] = d[Y0:Y1, X0:X1]
GRID_ROWS = (447, 448, 449, 450, 451, 567, 568, 569, 570, 571, 689, 690, 691, 692, 693)
GRID_COLS = (1677, 1678, 1679, 1680, 1815, 1816, 1817, 1818, 1953, 1954, 1955, 1956, 2089, 2090, 2091, 2092)
for r in GRID_ROWS: m[r, :] = False
for c in GRID_COLS: m[:, c] = False
boxes = [(1650, 1790, 372, 408), (2050, 2210, 392, 432), (1715, 1865, 698, 738), (2015, 2185, 495, 650)]
for x0, x1, y0, y1 in boxes: m[y0:y1, x0:x1] = False
raw = find_crosses(d, 1545, 2226, 315, 808, exclude_cols=(1678, 1679, 1816, 1817, 1954, 1955, 2090, 2091, 2092, 2227))
inbox = lambda x, y: any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in boxes)
cr = [(c[0], c[1]) for c in raw if not inbox(c[0], c[1]) and c[0] > 1700]
G = {
 '1.41': [(1.015, -25), (1.02, -23.5), (1.05, -19), (1.1, -15), (1.15, -13), (1.2, -11.7), (1.3, -9.2), (1.4, -7.1), (1.5, -5.3), (1.6, -3.8), (1.7, -2.4), (1.8, -1.3), (1.9, -0.5), (2.0, -0.2)],
 '0.707': [(1.015, -31), (1.02, -29.5), (1.05, -26), (1.1, -23), (1.15, -20.5), (1.2, -19), (1.3, -16.5), (1.4, -15), (1.5, -13.6), (1.6, -12.3), (1.7, -11.3), (1.8, -10.6), (1.9, -9.6), (2, -8.7)],
 '0.87': [(1.01, -39.5), (1.05, -34), (1.1, -30.5), (1.2, -28), (1.3, -26.8), (1.4, -26.3), (1.5, -26.2), (1.6, -26.8), (1.65, -28.3), (1.7, -31.3), (1.75, -34.5), (1.78, -37.5), (1.80, -40.5), (1.85, -40.5), (1.86, -38.8), (1.9, -33.5), (1.95, -28), (1.98, -25)],
}
gpx = {k: (px(np.array([p[0] for p in v])), py(np.array([p[1] for p in v]))) for k, v in G.items()}
m2 = erase_crosses(m, cr, half=10)
parts = assign(m2, gpx, tol=22)
dbx, dby = abs(cx[0]), abs(cy[0])
xg = px(np.round(np.arange(1.0, 2.0001, 0.005), 4))
pts = {
 'd/b=1.41': curve_points(parts['1.41'], xg, dbx, dby, fx, fy, cal=ry),
 'd/b=0.707': curve_points(parts["0.707"], xg, dbx, dby, fx, fy, cal=ry, min_cols=3),
 'd/b=0.87': dot_points(parts['0.87'], fx, fy, dby, cal=ry),
}
for k in pts:
    for p in list(pts[k]):
        pass
crs = {k: [] for k in G}
for x, y in sorted(cr):
    dist = {k: abs(np.interp(x, *gpx[k]) - y) for k in gpx}
    lab = min(dist, key=dist.get)
    crs[lab].append((float(fx(x)), float(fy(y)), float(np.hypot(2 * dby, 2 * dby) + ry)))
for k in G:
    pts['ref7_d/b=' + k] = crs[k]
CAL = 'x: px %s -> %s ; y: px %s -> %s (full-page pixel coordinates of PDF page 2 rendered at 300 dpi; max fit residual x %.4f, y %.3f dB)' % (XC[0], XC[1], YC[0], YC[1], rx, ry)

def overlay(fname, pts_by, crop, fx_inv, fy_inv):
    rgb = Image.fromarray(np.stack([im] * 3, -1)).convert('RGB')
    dr = ImageDraw.Draw(rgb)
    colors = [(255, 0, 0), (0, 160, 0), (0, 0, 255), (255, 0, 255), (0, 180, 180), (255, 140, 0)]
    for (k, pp), col in zip(pts_by.items(), colors):
        for p in pp:
            x, y = fx_inv(p[0]), fy_inv(p[1])
            r = 6 if k.startswith('ref') else 2
            dr.ellipse((x - r, y - r, x + r, y + r), outline=col)
    rgb.crop(crop).save(fname)

if __name__ == '__main__':
    overlay('../ov4_all.png', pts, (1530, 310, 2240, 820), px, py)
    for k, v in pts.items():
        print(k, len(v), 'u range %.2f-%.2f' % (min(p[2] for p in v), max(p[2] for p in v)))
