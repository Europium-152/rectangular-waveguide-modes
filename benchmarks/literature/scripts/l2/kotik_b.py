"""L2 pass b: Kotik & Sivov (1965) Fig. 3, column-run scan of the embedded 300 ppi page scan (pdfimages).

    python3 kotik_b.py <reference folder> <scratch folder> <output folder>
Independent of pass a: other raster (the native scan, not a render), other calibration (straight-line fits of all
gridlines, kd from a quadratic through the five vertical gridlines, y linear between the straight 1.0 and 0.75
lines), other extraction (vertical ink runs per column instead of a ridge follower)."""
import os
import sys
import numpy as np
from l2common import KOTIK_PDF, native, line_samples_v, line_samples_h, robust_linfit, bin_curve, write_raw
from colscan import scan, runs

REF, SCR, OUT = sys.argv[1:4]
im, png = native(os.path.join(REF, KOTIK_PDF), 5, SCR, 'kotik')
assert im.shape == (2753, 1914), im.shape

VX = [(10, 509.0), (15, 702.0), (20, 897.5), (25, 1094.0), (30, 1291.0)]
vfits = []
for val, xc in VX:
    s = line_samples_v(im, xc, range(124, 480, 8), half=6, win=12, thr=110, minmass=600)
    c, rms, mx, m = robust_linfit(s[:, 0], s[:, 1])
    vfits.append((val, c, rms, mx))
hfits = []
xs_h = [x for x in range(515, 1380, 10) if all(abs(x + 6 - v[1]) > 10 for v in VX)]
for val, yc in [(1.0, 116.0), (0.75, 307.0)]:
    s = line_samples_h(im, yc, xs_h, half=9, win=12, thr=110, minmass=500)
    c, rms, mx, m = robust_linfit(s[:, 0], s[:, 1])
    hfits.append((val, c, rms, mx))


def to_data(px, py):
    X = np.array([np.polyval(c, py) for _, c, *_ in vfits])
    V = np.array([v for v, *_ in vfits], float)
    q = np.polyfit(X, V, 2)
    kd = np.polyval(q, px)
    y1 = np.polyval(hfits[0][1], px)
    y75 = np.polyval(hfits[1][1], px)
    return kd, 1.0 - 0.25 * (py - y1) / (y75 - y1)


Xq = np.array([np.polyval(c, 200) for _, c, *_ in vfits])
qres = np.array([v for v, *_ in vfits]) - np.polyval(np.polyfit(Xq, [v for v, *_ in vfits], 2), Xq)
KD_PER_PX = 20.0 / (Xq[-1] - Xq[0])
Y_PER_PX = 0.25 / (np.polyval(hfits[1][1], 900) - np.polyval(hfits[0][1], 900))
SIG_CX = np.hypot(KD_PER_PX * max(f[2] for f in vfits), np.abs(qres).max())
SIG_CY = Y_PER_PX * max(f[2] for f in hfits)

# ---- solid: segments (seed ordinates read from the run lists of single columns) ----
segs = [scan(im, 512, 593, 230, 120, 330, maxjump=5, maxlen=40),
        scan(im, 601, 610, 267, 120, 330, maxjump=8, maxlen=40),
        scan(im, 626, 1340, 297, 120, 330, maxjump=5, maxlen=40)]
sol = np.vstack(segs)
dash = scan(im, 512, 1382, 290, 120, 330, maxjump=2.5, maxlen=6, k=40, slope0=-0.35)
sol = sol[~np.isnan(sol[:, 1])]
dash = dash[~np.isnan(dash[:, 1])]

gl = lambda x, y: min(abs(x - np.polyval(c, y)) for _, c, *_ in vfits)
# a column is kept only if the two curves are apart there (> 6 px) and it is > 4 px from a vertical gridline
def keep(tr, other):
    out = []
    oy = dict((int(x), y) for x, y, _ in other)
    for x, y, L in tr:
        near_other = any(abs(oy.get(int(x) + d, 1e9) - y) < 6 + 0.5 * L for d in range(-3, 4))
        if near_other or gl(x, y) <= 4:
            continue
        out.append((x, y, L))
    return np.array(out)

sol_k = keep(sol, dash)
# run-length sanity check: a run much longer than its neighbours is a merge with other ink
Lm = np.array([np.median(sol_k[max(i - 3, 0):i + 4, 2]) for i in range(len(sol_k))])
sol_k = sol_k[sol_k[:, 2] <= Lm + 4]
dash_k = keep(dash, sol)
dash_k = dash_k[dash_k[:, 0] < 1381]

# dip bottom (columns 612-624): the stroke merges with the 0.75 gridline; ordinate = lower edge - 2.75 px
bottom = []
for x in range(612, 625):
    rr = runs(im[290:330, x], 128)
    edge = max(b for a, b in rr) + 290
    bottom.append((x, edge - 2.75, 5.5))
bottom = np.array(bottom)


def convert(tr):
    rows = []
    for i, (x, y, L) in enumerate(tr):
        kd, v = to_data(x, y)
        j0, j1 = max(i - 3, 0), min(i + 3, len(tr) - 1)
        sl = (tr[j1, 1] - tr[j0, 1]) / max(tr[j1, 0] - tr[j0, 0], 1) if j1 > j0 else 0.0
        slope = sl * (-Y_PER_PX / KD_PER_PX)
        u = np.sqrt((max(1.0, L / 4) * Y_PER_PX) ** 2 + SIG_CY ** 2 + (slope * SIG_CX) ** 2)
        rows.append((kd, v, u))
    return np.array(rows)


S = convert(np.vstack([sol_k, bottom]))
D = convert(dash_k)
Sb = bin_curve(S[:, 0], S[:, 1], S[:, 2], 0.05)
Db = bin_curve(D[:, 0], D[:, 1], D[:, 2], 0.05)
hdr = [
    'source: I. P. Kotik and A. N. Sivov, "Diffraction of electromagnetic waves on a mirror in a break of a plane '
    'waveguide", Radio Engineering and Electronic Physics 10 (1965) 140-142 (English translation; scan, PDF pages 3-5)',
    'figure: Fig. 3, p. 142 (PDF p. 5): |B1|^2 vs kd; solid = rigorous solution, dashed = asymptotic formula',
    'x: kd (d = guide width), linear; y: |B1|^2, linear',
    'calibration (L2b): embedded 300 ppi scan of PDF page 5 (pdfimages); vertical gridlines kd = 10, 15, 20, 25, 30 '
    'fitted as straight lines (rms ' + ', '.join(f'{f[2]:.2f}' for f in vfits) + ' px); kd = quadratic through '
    'the five line positions at the row of the point (residuals at py = 200: ' +
    ', '.join(f'{r:+.4f}' for r in qres) + ' kd); y linear between straight-line fits of the 1.0 frame and the 0.75 '
    'gridline (rms ' + ', '.join(f'{f[2]:.2f}' for f in hfits) + ' px; the frame wobbles, which this straight fit '
    'does not follow and which is counted in the calibration error)',
    'calibration (L2b): gridline px at py = 200: ' + ', '.join(f'{v}->{x:.1f}' for (v, *_), x in zip(vfits, Xq)) +
    f'; scale {KD_PER_PX:.5f} kd/px, {Y_PER_PX:.6f} /px; 1-sigma calibration errors {SIG_CX:.4f} kd, {SIG_CY:.5f}',
    'method (L2b): every column, the vertical ink run (gray < 140) nearest the linear prediction from the previous '
    'columns, centre = ink-weighted mean; solid in three segments (seeds from single-column run lists), dashed with '
    'a 40-column prediction and runs <= 6 px; columns where the curves are within 6 px of each other or within 4 px '
    'of a vertical gridline dropped; binned in kd bins of 0.05; uncertainty = sqrt((max(1, L/4) px)^2 + cal_y^2 + '
    '(slope cal_x)^2) with L the run length in the column',
    'digitizer: L2 (pass b), 2026-11-11',
]
write_raw(os.path.join(OUT, 'kotik1965_fig3_rigorous.csv'),
          hdr + ['curve: rigorous = solid line; dip bottom at kd = 4 pi from the lower stroke edge minus half the '
                 'stroke width (stroke merges with the 0.75 gridline)'],
          [('rigorous', x, y, u, 0.0) for x, y, u in Sb])
write_raw(os.path.join(OUT, 'kotik1965_fig3_asymptotic.csv'), hdr + ['curve: asymptotic = dashed line'],
          [('asymptotic', x, y, u, 0.0) for x, y, u in Db])
np.save(os.path.join(SCR, 'kotik_b_px.npy'), np.vstack([np.c_[sol_k[:, :2], np.zeros(len(sol_k))],
                                                       np.c_[bottom[:, :2], np.zeros(len(bottom))],
                                                       np.c_[dash_k[:, :2], np.ones(len(dash_k))]]))
print('vfits', [(v, np.round(c, 4), round(r, 2)) for v, c, r, mx in vfits], 'qres', np.round(qres, 4))
print('hfits', [(v, np.round(c, 5), round(r, 2), round(mx, 2)) for v, c, r, mx in hfits])
print('scale', KD_PER_PX, Y_PER_PX, 'cal', SIG_CX, SIG_CY)
print('solid', len(Sb), Sb[:, 0].min(), Sb[:, 0].max(), np.percentile(Sb[:, 2], [0, 50, 100]))
print('dash', len(Db), Db[:, 0].min(), Db[:, 0].max(), np.percentile(Db[:, 2], [0, 50, 100]))
