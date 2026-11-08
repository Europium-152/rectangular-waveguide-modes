"""Alessandri, Mongiardo & Sorrentino 1994, Fig. 5 (page 409 = PDF page 2), digitized by L1b."""
import numpy as np
from PIL import Image, ImageDraw
from dig import *
from trace import assign_euclid
from curvepts import curve_points, dot_points, drop_near_others
im, d = load('../ales-2.png')
XC = ([1507.5, 1646.5, 1783.5, 1922.0, 2059.0, 2196.0], [1, 1.2, 1.4, 1.6, 1.8, 2])
YC = ([1117.5, 1215.5, 1312.5, 1410.5, 1507.5, 1604.0], [0, -10, -20, -30, -40, -50])
fx, rx, cx = fitcal(*XC); fy, ry, cy = fitcal(*YC)
px = lambda f: (np.asarray(f) - cx[1]) / cx[0]; py = lambda s: (np.asarray(s) - cy[1]) / cy[0]
m = np.zeros_like(d)
X0, X1, Y0, Y1 = 1515, 2196, 1120, 1598
m[Y0:Y1, X0:X1] = d[Y0:Y1, X0:X1]
boxes = [(1615, 1795, 1130, 1290), (1878, 2032, 1278, 1314), (1643, 1792, 1443, 1479), (1655, 1790, 1528, 1556), (2109, 2260, 1488, 1532)]
for x0, x1, y0, y1 in boxes: m[y0:y1, x0:x1] = False
G = {
 '0.9': [(1.0193, -33.8), (1.046, -29.7), (1.119, -25.7), (1.254, -22.6), (1.424, -20.8), (1.54, -20.4), (1.656, -20.4), (1.801, -21.8), (1.899, -24.5), (1.961, -27.2), (1.98, -28.7)],
 '0.87': [(1.019, -37.2), (1.041, -34.1), (1.133, -29), (1.279, -26.7), (1.438, -26.1), (1.54, -26.7), (1.656, -28.6), (1.753, -33.4), (1.802, -40.3), (1.826, -49.5), (1.83, -51)],
 '0.87r': [(1.846, -51), (1.85, -49.5), (1.874, -39.2), (1.923, -31.4), (1.961, -27.2), (1.98, -25.5)],
 '0.85': [(1.034, -40.6), (1.061, -36.2), (1.133, -33.8), (1.23, -32.4), (1.351, -32.2), (1.448, -33.4), (1.497, -35.5), (1.521, -36.5), (1.559, -38.2), (1.598, -43), (1.613, -48.5), (1.617, -51)],
 '0.85r': [(1.652, -51), (1.656, -47.8), (1.686, -42), (1.729, -34.8), (1.802, -27.2), (1.879, -23.8), (1.971, -19.7)],
 '0.83': [(1.018, -51), (1.022, -48.5), (1.041, -45.8), (1.104, -43.5), (1.158, -43.7), (1.196, -44.2), (1.225, -45.1), (1.254, -47.5), (1.283, -49.2), (1.29, -51)],
 '0.83r': [(1.385, -51), (1.39, -48.5), (1.414, -44.4), (1.453, -39.6), (1.54, -33.8), (1.584, -30.7), (1.656, -27.2), (1.802, -21.1), (1.899, -17.7), (1.98, -15.8)],
}
gpx = {k: (px(np.array([p[0] for p in v])), py(np.array([p[1] for p in v]))) for k, v in G.items()}
parts = assign_euclid(m, gpx, tol=14)
dbx, dby = abs(cx[0]), abs(cy[0])
xg = px(np.round(np.arange(1.0, 2.0001, 0.005), 4))
pts = {}
for k in ('0.9', '0.87', '0.87r'):
    pts[k] = curve_points(parts[k], xg, dbx, dby, fx, fy, cal=ry)
for k in ('0.85', '0.85r', '0.83', '0.83r'):
    pts[k] = curve_points(parts[k], xg, dbx, dby, fx, fy, cal=ry, min_cols=3)
pts, NREM = drop_near_others(pts, parts, px, py)
CAL = 'x: px %s -> %s ; y: px %s -> %s (full-page pixel coordinates of PDF page 2 rendered at 300 dpi; max fit residual x %.4f, y %.3f dB)' % (XC[0], XC[1], YC[0], YC[1], rx, ry)

def overlay(fname, pts_by, crop, scale=1):
    rgb = Image.fromarray(np.stack([im] * 3, -1)).convert('RGB')
    dr = ImageDraw.Draw(rgb)
    colors = [(255, 0, 0), (0, 160, 0), (0, 160, 0), (0, 0, 255), (0, 0, 255), (255, 0, 255), (255, 0, 255)]
    for (k, pp), col in zip(pts_by.items(), colors):
        for p in pp:
            x, y = px(p[0]), py(p[1])
            dr.ellipse((x - 2, y - 2, x + 2, y + 2), outline=col)
    c = rgb.crop(crop)
    c.resize((c.size[0] * scale, c.size[1] * scale)).save(fname)

if __name__ == '__main__':
    overlay('../ov5_all.png', pts, (1500, 1100, 2210, 1615))
    print('removed near crossings', NREM)
    for k, v in pts.items():
        print(k, len(v), 'u range %.2f-%.2f' % (min(p[2] for p in v), max(p[2] for p in v)))
