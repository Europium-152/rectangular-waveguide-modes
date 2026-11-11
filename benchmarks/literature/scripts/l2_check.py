"""Overlay check of the merged L2 files (card L2): every merged point is drawn on the rasterized figure and its
pixel residual to the printed ink is reported.

    python3 benchmarks/literature/scripts/l2_check.py --alimenti <Alimenti 1994 PDF> --kotik <Kotik 1965 PDF> \
        --out /tmp/l2_check

The PDFs are not in the repository; pages are rendered at 400 dpi into --out (outside the repository), together
with one overlay PNG per figure (points: red = both passes, blue = one pass only, magenta = markers). Nothing is
written into the repository.

Residuals, per curve:
* d_ink: distance (px, 400 dpi) from the point to the nearest ink pixel; 0 means the point lies on the printed
  stroke. Points of lines that run into a frame or a crossing are included.
* dy: for points on gentle slopes (|dy/dx| < 1 px/px at the point), the vertical offset (px) between the point and
  the centre of the ink run of that column nearest to it (the quantity the y uncertainty describes), together with
  the uncertainty converted to px.
Calibrations used for the mapping data -> px: the pass-a tick calibrations (scripts/l2/alimcal.py, measured anew
here) and, for Kotik & Sivov, the gridline positions of pass a (constants below).
"""
import argparse
import csv
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
LIT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, 'l2'))
from alimcal import TickCal  # noqa: E402

ALIM_CAL = {   # same arguments as in scripts/l2/alimenti_a.py (400 dpi render)
    2: dict(page=2, X=1235, Y=3489, xt=((6, 30), (1225, 2341)), yt=((-30, -6), (2662, 3485)),
            xvals=[1, 1.2, 1.4, 1.6, 1.8, 2.0], yvals=[0, -2, -4, -6, -8], extra_y=[-10], xr_axis=(1250, 2330),
            yr_axis=(2700, 3470), box=(1200, 2650, 2340, 3560)),
    3: dict(page=3, X=1224, Y=1386, xt=((-30, -6), (1300, 2200)), yt=((6, 30), (590, 1380)),
            xvals=[1.2, 1.4, 1.6, 1.8], yvals=[0, -10, -20, -30, -40, -50, -60], extra_x=[1.0], extra_y=[-70],
            xr_axis=(1240, 2200), yr_axis=(620, 1370), box=(1190, 600, 2300, 1420)),
    4: dict(page=3, X=1239, Y=3455, xt=((6, 30), (1230, 2360)), yt=((-30, -6), (2478, 3450)),
            xvals=[1, 1.2, 1.4, 1.6, 1.8, 2.0], yvals=[0, -10, -20, -30, -40], extra_y=[-50], xr_axis=(1260, 2340),
            yr_axis=(2720, 3440), box=(1200, 2690, 2290, 3490)),
    5: dict(page=4, X=993, Y=2035, xt=((6, 30), (985, 2320)), yt=((-30, -6), (996, 2030)),
            xvals=[.30, .32, .34, .36, .38, .40], yvals=[2.0, 1.8, 1.6, 1.4, 1.2], extra_y=[1.0],
            xr_axis=(1010, 2290), yr_axis=(1060, 2020), box=(960, 1020, 2340, 2070)),
    6: dict(page=4, X=1154, Y=3562, xt=((6, 30), (1145, 2286)), yt=((-30, -6), (2784, 3555)),
            xvals=[1, 1.2, 1.4, 1.6, 1.8, 2.0], yvals=[0, -10, -20, -30], extra_y=[-40], xr_axis=(1170, 2270),
            yr_axis=(2840, 3550), box=(1120, 2800, 2230, 3600)),
}
ALIM_FILES = {2: ['alimenti1994_fig2_mmfd.csv', 'alimenti1994_fig2_mm.csv'],
              3: ['alimenti1994_fig3_mm.csv', 'alimenti1994_fig3_mmfd.csv'],
              4: ['alimenti1994_fig4_t030.csv', 'alimenti1994_fig4_t034.csv', 'alimenti1994_fig4_t038.csv'],
              5: ['alimenti1994_fig5_matching_frequency.csv', 'alimenti1994_fig5_bandwidth.csv'],
              6: ['alimenti1994_fig6_mmfd.csv', 'alimenti1994_fig6_bcmm.csv']}
# Kotik & Sivov Fig. 3 on the 400 dpi render of PDF page 5 (pass a): vertical gridlines px = c0*py + c1, frame
# |B1|^2 = 1.0 and gridline 0.75 as straight lines py = c0*px + c1, refined locally (ink centroid near the column)
KOTIK_V = [(10, 0.0072, 676.19), (15, 0.0019, 935.61), (20, -0.0001, 1196.94), (25, -0.0074, 1462.01),
           (30, -0.0064, 1724.95)]
KOTIK_H = [(1.0, 0.0103, 142.11), (0.75, 0.0108, 396.30)]


def render(pdf, page, out):
    stem = os.path.join(out, f'{os.path.basename(pdf)[:12].replace(" ", "_")}-p{page}-400')
    if not os.path.exists(stem + '.png'):
        subprocess.run(['pdftoppm', '-r', '400', '-f', str(page), '-l', str(page), '-png', '-singlefile', pdf, stem],
                       check=True)
    return np.array(Image.open(stem + '.png').convert('L')).astype(float), stem + '.png'


def read(name):
    rows = []
    with open(os.path.join(LIT, name)) as fh:
        for r in csv.reader(ln for ln in fh if not ln.startswith('#')):
            if r and r[0] != 'curve':
                rows.append((r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), r[5]))
    return rows


def alim_inverse(cal):
    a, b = cal.cx
    c, d = cal.cy

    def f(x, y):
        pxa, pya = (x - b) / a, (y - d) / c
        # px = pxa + kx (py - Yax); py = pya + ky (px - Xax)
        M = np.array([[1.0, -cal.kx], [-cal.ky, 1.0]])
        rhs = np.array([pxa - cal.kx * cal.Yax, pya - cal.ky * cal.Xax])
        return np.linalg.solve(M, rhs)
    return f, abs(c), abs(a)


def kotik_inverse(im):
    def hline(px, c0, c1):
        """local ink centroid of a horizontal gridline near column px (follows the wobble of the print)"""
        y0 = int(round(c0 * px + c1))
        best = None
        for dx in (0, -25, 25, -50, 50):
            x = int(round(px)) + dx - 10
            sub = 255 - im[y0 - 8:y0 + 9, x:x + 20]
            sub[sub < 100] = 0
            w = sub.sum(1)
            if w.sum() > 1200:
                best = y0 - 8 + (w * np.arange(17)).sum() / w.sum()
                break
        return best if best is not None else c0 * px + c1

    def f(kd, v):
        py = 250.0
        for _ in range(3):
            X = np.array([c0 * py + c1 for _, c0, c1 in KOTIK_V])
            V = np.array([k for k, _, _ in KOTIK_V], float)
            if kd > 30:
                px = X[-1] + (kd - 30) * (X[-1] - X[-2]) / 5
            else:
                px = np.interp(kd, V, X)
            y1 = hline(px, KOTIK_H[0][1], KOTIK_H[0][2])
            y75 = hline(px, KOTIK_H[1][1], KOTIK_H[1][2])
            py = y1 + (1.0 - v) / 0.25 * (y75 - y1)
        return np.array([px, py])
    return f, 255.0 / 0.25, 260.0 / 5


def residuals(im, pts):
    """pts: list of (curve, px, py, u_px, is_marker, slope_px)"""
    ink = im < 128
    dist = ndi.distance_transform_edt(~ink)
    out = {}
    for name, px, py, upx, mk, slope in pts:
        xi, yi = int(round(px)), int(round(py))
        if not (0 <= yi < im.shape[0] and 0 <= xi < im.shape[1]):
            continue
        d = dist[yi, xi]
        dy = np.nan
        if not mk and abs(slope) < 1.0 and d == 0:
            col = ink[max(yi - 6, 0):yi + 7, xi]
            idx = np.where(col)[0]
            if len(idx):
                br = np.where(np.diff(idx) > 1)[0]
                starts, ends = np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]
                cen = [(s + e) / 2 + max(yi - 6, 0) for s, e in zip(starts, ends)]
                c = min(cen, key=lambda v: abs(v - py))
                dy = py - c
        out.setdefault(name, []).append((d, dy, upx))
    return out


def draw(png, box, pts, out, scale=1.0):
    img = Image.open(png).convert('RGB').crop(box)
    img = img.resize((int(img.width * scale), int(img.height * scale)))
    dr = ImageDraw.Draw(img)
    for name, px, py, upx, mk, slope, src in pts:
        X, Y = (px - box[0]) * scale, (py - box[1]) * scale
        col = (255, 0, 255) if mk else ((220, 0, 0) if src == 'ab' else (0, 120, 255))
        r = 4 if mk else 2
        dr.ellipse((X - r, Y - r, X + r, Y + r), outline=col)
    img.save(out)


def report(res, tag):
    print(f'\n{tag}')
    print(f'  {"curve":28s} {"n":>4s} {"on ink":>7s} {"d_ink rms/max px":>17s} {"n_dy":>5s} {"dy rms/max px":>14s} '
          f'{"|dy|<=u":>8s}')
    for name, v in res.items():
        a = np.array(v, float)
        d, dy, u = a[:, 0], a[:, 1], a[:, 2]
        m = ~np.isnan(dy)
        within = (np.abs(dy[m]) <= np.maximum(u[m], 1.0)).mean() if m.any() else np.nan
        print(f'  {name[:28]:28s} {len(a):4d} {np.mean(d == 0):7.1%} {np.sqrt(np.mean(d ** 2)):8.2f}/{d.max():6.2f} '
              f'{m.sum():5d} {np.sqrt(np.nanmean(dy[m] ** 2)) if m.any() else np.nan:7.2f}/'
              f'{np.nanmax(np.abs(dy[m])) if m.any() else np.nan:5.2f} {within:8.1%}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alimenti', required=True, help='path of the Alimenti et al. (1994) PDF')
    ap.add_argument('--kotik', required=True, help='path of the Kotik & Sivov (1965) PDF scan')
    ap.add_argument('--out', default='/tmp/l2_check', help='scratch folder for renders and overlays')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for k, spec in ALIM_CAL.items():
        im, png = render(args.alimenti, spec['page'], args.out)
        kw = {kk: v for kk, v in spec.items() if kk not in ('page', 'box')}
        cal = TickCal(im, label=f'fig{k}', **kw)
        inv, ypp, xpp = alim_inverse(cal)
        pts = []
        for fn in ALIM_FILES[k]:
            rows = read(fn)
            for i, (c, x, y, u, xu, s) in enumerate(rows):
                mk = xu > 0
                px, py = inv(x, y if not (k == 5 and c.startswith('bandwidth')) else 2 * y + 1)
                uu = (2 * u if (k == 5 and c.startswith('bandwidth')) else u) / ypp
                slope = 0.0
                if not mk and 0 < i < len(rows) - 1 and rows[i - 1][0] == c == rows[i + 1][0]:
                    p0 = inv(rows[i - 1][1], rows[i - 1][2] if not (k == 5 and c.startswith('bandwidth'))
                             else 2 * rows[i - 1][2] + 1)
                    p1 = inv(rows[i + 1][1], rows[i + 1][2] if not (k == 5 and c.startswith('bandwidth'))
                             else 2 * rows[i + 1][2] + 1)
                    slope = (p1[1] - p0[1]) / max(abs(p1[0] - p0[0]), 1e-6)
                pts.append((f'{fn[13:-4]}: {c}', px, py, uu, mk, slope, s))
        res = residuals(im, [p[:6] for p in pts])
        report(res, f'Alimenti et al. 1994, Fig. {k} (tick fit rms: x {cal.sx:.3g}, y {cal.sy:.3g})')
        draw(png, spec['box'], pts, os.path.join(args.out, f'overlay_alimenti_fig{k}.png'), 1.0)
    im, png = render(args.kotik, 5, args.out)
    inv, ypp, xpp = kotik_inverse(im)
    pts = []
    for fn in ('kotik1965_fig3_rigorous.csv', 'kotik1965_fig3_asymptotic.csv'):
        rows = read(fn)
        for i, (c, x, y, u, xu, s) in enumerate(rows):
            px, py = inv(x, y)
            slope = 0.0
            if 0 < i < len(rows) - 1:
                p0, p1 = inv(rows[i - 1][1], rows[i - 1][2]), inv(rows[i + 1][1], rows[i + 1][2])
                slope = (p1[1] - p0[1]) / max(abs(p1[0] - p0[0]), 1e-6)
            u_read = np.sqrt(max(u ** 2 - 0.005 ** 2, 0)) if c == 'rigorous' else u   # without the drafting term
            pts.append((f'{fn[:-4]}', px, py, u_read * ypp, False, slope, s))
    res = residuals(im, [p[:6] for p in pts])
    report(res, 'Kotik & Sivov 1965, Fig. 3 (dy compared with the reading uncertainty, drafting term removed)')
    draw(png, (660, 130, 1870, 440), pts, os.path.join(args.out, 'overlay_kotik_fig3.png'), 1.0)
    # x-axis labels of Kotik Fig. 3: centres of the printed labels vs the gridline scale
    band = im[672:715, 640:1900] < 128
    lab, n = ndi.label(ndi.binary_dilation(band, iterations=4))
    cents = sorted((xs.mean() + 640, len(xs)) for xs in
                   (np.where((lab == i) & band)[1] for i in range(1, n + 1)) if len(xs) > 30)
    names = [10, 4 * np.pi, 15, 5 * np.pi, 6 * np.pi, 20, 7 * np.pi, 25, 9 * np.pi, 30, 10 * np.pi, None]
    print('\nKotik Fig. 3 x labels (label centre -> kd on the gridline scale at the bottom frame):')
    X = np.array([c0 * 660 + c1 for _, c0, c1 in KOTIK_V])
    got = []
    for (cx, _), v in zip(cents, names):
        if v is None:
            continue
        kd = np.interp(cx, X, [10, 15, 20, 25, 30]) if cx <= X[-1] else 30 + (cx - X[-1]) * 5 / (X[-1] - X[-2])
        got.append((v, kd, cx))
        print(f'  label {v:7.3f}: centre px {cx:7.1f} -> kd {kd:7.3f} (diff {kd - v:+.3f})')
    g = np.array(got)
    c = np.polyfit(g[:, 2], g[:, 0], 1)
    r = g[:, 0] - np.polyval(c, g[:, 2])
    print(f'  linear fit kd(px) through all 11 label centres: {c[0]:.5f} kd/px, rms residual {np.sqrt(np.mean(r ** 2)):.3f} '
          f'kd, max {np.abs(r).max():.3f} kd; gridline scale at the bottom: {5 / np.mean(np.diff(X)):.5f} kd/px')
    print(f'\noverlays written to {args.out}')


if __name__ == '__main__':
    main()
