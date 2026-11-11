"""L2 pass a: Kotik & Sivov (1965) Fig. 3 (PDF page 5), ridge-following on a 400 dpi pdftoppm render.

    python3 kotik_a.py <reference folder> <scratch folder> <output folder>
Writes <output>/kotik1965_fig3_rigorous.csv and kotik1965_fig3_asymptotic.csv (raw set a)."""
import os
import sys
import numpy as np
from l2common import KOTIK_PDF, render, line_samples_v, line_samples_h, robust_linfit, bin_curve, write_raw
from ridge import follow

REF, SCR, OUT = sys.argv[1:4]
im, png = render(os.path.join(REF, KOTIK_PDF), 5, 400, SCR, 'kotik')
dark = 255 - im

# ---------------- calibration (a): gridlines ----------------
VX = [(10, 679.0), (15, 936.5), (20, 1197.0), (25, 1458.5), (30, 1722.0)]
vfits = []
for val, xc in VX:
    s = line_samples_v(im, xc, range(165, 640, 15), half=10)
    c, rms, mx, m = robust_linfit(s[:, 0], s[:, 1])
    vfits.append((val, c, rms, mx, int(m.sum())))
HY = [(1.0, 154.0), (0.75, 410.0)]
xs_h = [x for x in range(690, 1835, 15) if all(abs(x + 10 - v[1]) > 16 for v in VX)]
hfun = []
for val, yc in HY:
    s = line_samples_h(im, yc, xs_h, half=12, minmass=1200)
    c, rms, mx, m = robust_linfit(s[:, 0], s[:, 1])
    s = s[m]
    # piecewise-linear through 5-point running medians (follows the scan's wobble of the frame)
    sm = np.array([np.median(s[max(i - 2, 0):i + 3, 1]) for i in range(len(s))])
    resid = s[:, 1] - sm
    hfun.append((val, s[:, 0], sm, float(np.std(resid)), c))


def vline_x(py):
    return np.array([np.polyval(c, py) for _, c, *_ in vfits])


def to_data(px, py):
    X = vline_x(py)
    vals = np.array([v for v, *_ in vfits], float)
    kd = np.interp(px, X, vals)
    if px > X[-1]:
        kd = vals[-1] + (px - X[-1]) * (vals[-1] - vals[-2]) / (X[-1] - X[-2])
    if px < X[0]:
        kd = vals[0] + (px - X[0]) * (vals[1] - vals[0]) / (X[1] - X[0])
    y1 = np.interp(px, hfun[0][1], hfun[0][2])
    y75 = np.interp(px, hfun[1][1], hfun[1][2])
    return kd, 1.0 - 0.25 * (py - y1) / (y75 - y1)


KD_PER_PX = 5.0 / np.mean(np.diff([np.polyval(c, 280) for _, c, *_ in vfits]))
Y_PER_PX = 0.25 / np.mean(np.interp(np.arange(700, 1820), hfun[1][1], hfun[1][2]) -
                          np.interp(np.arange(700, 1820), hfun[0][1], hfun[0][2]))
SIG_CX = KD_PER_PX * max(r for _, _, r, _, _ in vfits)          # x calibration (kd), 1 sigma
SIG_CY = Y_PER_PX * max(h[3] for h in hfun)                      # y calibration, 1 sigma

# ---------------- trace ----------------
sol_l = follow(dark, (686, 304), (1, 0), 900, step=2.5, h=12, thr=100, maxw=7, stop=lambda p: p[1] > 398)
top = lambda p: p[1] < 172                                # never onto the frame
sol_r = np.vstack([follow(dark, (836, 395), (0.35, -1), 1112, step=2.5, h=12, thr=100, maxw=7, stop=top),
                   follow(dark, (1125, 254.5), (0.5, 1), 1452, step=2.5, h=12, thr=100, maxw=7, stop=top),
                   follow(dark, (1470, 240.5), (1, 0.8), 1715, step=2.5, h=12, thr=100, maxw=7, stop=top),
                   follow(dark, (1738, 189.5), (1, 0), 1796, step=2.5, h=10, thr=90, maxw=7, stop=top)])
dash = follow(dark, (684, 386), (1, -0.33), 1846, step=2.5, h=8, thr=100, maxw=5, inertia=0.85, search=False)
sol = np.vstack([sol_l, sol_r])


def near(a, b, dist):
    """mask of rows of a within dist px of any point of b"""
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]).min(1)
    return d < dist


def clean(tr, other, dist):
    m = tr[:, 3] == 1
    m &= ~near(tr, other, dist)
    X = np.array([vline_x(y) for y in tr[:, 1]])
    m &= np.abs(tr[:, 0][:, None] - X).min(1) > 7      # vertical gridlines
    return tr[m]


sol_c = clean(sol, dash[dash[:, 3] == 1], 7.0)
dash_c = clean(dash, sol[sol[:, 3] == 1], 8.0)
dash_c = dash_c[dash_c[:, 0] < 1843]                     # right frame


def convert(tr, half_w):
    rows = []
    for x, y, w, ok, vx, vy in tr:
        kd, val = to_data(x, y)
        cos = max(abs(vx), 0.12)
        slope = (vy / vx if abs(vx) > 1e-6 else 1e3) * (-Y_PER_PX / KD_PER_PX)   # dy/dkd
        sig_perp = max(1.0, 0.25 * w) * Y_PER_PX     # centre of the stroke: quarter of its width, at least 1 px
        u = np.sqrt((sig_perp / cos) ** 2 + SIG_CY ** 2 + (slope * SIG_CX) ** 2)
        rows.append((kd, val, u))
    return np.array(rows)


S = convert(sol_c, 6)
D = convert(dash_c, 4)
# dip minimum at 4 pi: the stroke touches the 0.75 gridline; centre = lower stroke edge - half the stroke width
mins = []
for x in range(820, 832):
    col = im[395:430, x]
    d = np.where(col < 128)[0]
    edge = d.max() + 395
    mins.append(to_data(x, edge - 3.5))
mins = np.array(mins)
S = np.vstack([S, np.c_[mins, np.full(len(mins), np.hypot(2 * Y_PER_PX, SIG_CY))]])
Sb = bin_curve(S[:, 0], S[:, 1], S[:, 2], 0.05)
Db = bin_curve(D[:, 0], D[:, 1], D[:, 2], 0.05)

hdr_common = [
    'source: I. P. Kotik and A. N. Sivov, "Diffraction of electromagnetic waves on a mirror in a break of a plane '
    'waveguide", Radio Engineering and Electronic Physics 10 (1965) 140-142 (English translation; scan, PDF pages 3-5)',
    'figure: Fig. 3, p. 142 (PDF p. 5): |B1|^2 vs kd; solid = rigorous solution (truncated system, N <= 18), dashed = '
    'asymptotic formula of Katsenelenbaum (1963)',
    'x: kd = 2 pi d / lambda (d = guide width), linear; y: |B1|^2 (TE10 power transmission), linear',
    'calibration (L2a): 400 dpi pdftoppm render of PDF page 5; x from the vertical gridlines kd = 10 (frame), 15, 20, '
    '25, 30, each fitted as a straight line px(py) (rms ' + ', '.join(f'{r:.2f}' for _, _, r, _, _ in vfits) +
    ' px), kd piecewise linear between them at the row of the point, extrapolated with the 25-30 spacing above 30; y '
    'from the frame line |B1|^2 = 1.0 and the gridline 0.75, each followed as a 5-point running median of ink '
    'centroids every 15 px (rms about the median ' + ', '.join(f'{h[3]:.2f}' for h in hfun) + ' px), y linear '
    'between them at the column of the point',
    'calibration (L2a): gridline px at py = 280: ' + ', '.join(f'{v}->{np.polyval(c, 280):.1f}' for v, c, *_ in vfits) +
    f'; scale {KD_PER_PX:.5f} kd/px, {Y_PER_PX:.6f} /px; 1-sigma calibration errors {SIG_CX:.4f} kd, {SIG_CY:.5f}',
    'method (L2a): ridge follower (step 2.5 px along the tangent, ink centroid on the normal profile, direction '
    'search at sharp turns); points within 7-8 px of the other curve or of a vertical gridline and points not '
    're-centred dropped; binned in kd bins of 0.05; uncertainty = sqrt((max(1 px, w/4)/|cos|)^2 + cal_y^2 + '
    '(slope cal_x)^2), w the stroke width, cos the cosine of the local tangent angle',
    'digitizer: L2 (pass a), 2026-11-11; pass b (raw/b) used a different raster, calibration and extraction',
]
sol_rows = [('rigorous', x, y, u, 0.0) for x, y, u in Sb]
dash_rows = [('asymptotic', x, y, u, 0.0) for x, y, u in Db]
write_raw(os.path.join(OUT, 'kotik1965_fig3_rigorous.csv'),
          hdr_common + ['curve: rigorous = solid line; the dip minimum at kd = 4 pi touches the 0.75 gridline and was '
                        'read from the lower edge of the stroke minus half its width'], sol_rows)
write_raw(os.path.join(OUT, 'kotik1965_fig3_asymptotic.csv'), hdr_common + ['curve: asymptotic = dashed line'],
          dash_rows)
np.save(os.path.join(SCR, 'kotik_a_px.npy'), np.vstack([np.c_[sol_c[:, :2], np.zeros(len(sol_c))],
                                                       np.c_[dash_c[:, :2], np.ones(len(dash_c))]]))
print('vfits', [(v, np.round(c, 4), round(r, 2), round(mx, 2), n) for v, c, r, mx, n in vfits])
print('hfun rms', [round(h[3], 2) for h in hfun], 'scales', KD_PER_PX, Y_PER_PX, 'cal sig', SIG_CX, SIG_CY)
print('solid', len(Sb), Sb[:, 0].min(), Sb[:, 0].max(), 'u', np.percentile(Sb[:, 2], [0, 50, 100]))
print('dash', len(Db), Db[:, 0].min(), Db[:, 0].max(), 'u', np.percentile(Db[:, 2], [0, 50, 100]))
print('min', mins[:, 1].min(), mins[np.argmin(mins[:, 1]), 0])
