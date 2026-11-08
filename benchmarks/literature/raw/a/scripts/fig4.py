import sys; sys.path.insert(0, '/home/claude/wt/L1a-scratch')
from dig import *
from scipy import ndimage
import json

img = load_gray('al_p2_deskew.png'); J = ink(img)
# ---- calibration from tick centres (deskewed page image, pixels)
XT = [1711.06, 1868.09, 2024.03, 2180.03, 2334.03, 2489.46]; XV = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
YT = [354.09, 489.96, 626.78, 763.94, 901.0]; YV = [0, -10, -20, -30, -40]
mx, cx, rx = linfit(XT, XV); my, cy, ry = linfit(YT, YV)
print('x resid', rx, 'y resid', ry, 'per px', mx, my)
tox = lambda px: mx * px + cx; toy = lambda py: my * py + cy
topx = lambda X: (X - cx) / mx; topy = lambda Y: (Y - cy) / my

# ---- gridline removal: components fully inside a +-3 px band around a grid line
grid_rows = [topy(v) for v in (0, -10, -20, -30)]
grid_cols = [topx(v) for v in (1.2, 1.4, 1.6, 1.8, 2.0)]
lab, n = ndimage.label(J, structure=np.ones((3, 3)))
sl = ndimage.find_objects(lab)
K = J.copy()
removed = 0
for i, s in enumerate(sl):
    y0, y1, x0, x1 = s[0].start, s[0].stop - 1, s[1].start, s[1].stop - 1
    if any(y0 >= r - 3 and y1 <= r + 3 for r in grid_rows) or any(x0 >= c - 3 and x1 <= c + 3 for c in grid_cols):
        if (y1 - y0) < 12 and (x1 - x0) < 12:
            K[lab == i + 1] = False; removed += 1
print('grid dots removed', removed)

# ---- other masks: inset drawing, labels, axes region
other = np.zeros_like(K)
def box(X0, X1, Y0, Y1):
    other[int(topy(Y0)):int(topy(Y1)) + 1, int(topx(X0)):int(topx(X1)) + 1] = True
box(1.68, 1.94, -13.5, -26.2)   # inset
box(1.16, 1.36, -3.4, -6.6)   # label d/b=1.41
box(1.73, 1.97, -5.0, -8.1)    # label 0.707 (curve below)
box(1.24, 1.5, -30.3, -34.0) # label 0.87
other[:, :int(topx(1.0)) + 16] = True
other[int(topy(-40)) - 5:, :] = True
# ---- crosses
plot = (int(topx(1.0)) + 16, int(topy(0)) - 15, int(topx(2.0)) + 15, int(topy(-40)) - 6)
cr = find_crosses(K, arm=9, min_h=13, min_v=13, region=plot)
crosses = []
for (x, y) in cr:
    xr, yr, nv, nh = refine_cross(K, x, y)
    crosses.append((xr, yr, nv, nh))
crosses = [c for c in crosses if not other[int(c[1]), int(c[0])] and c[2] > 0 and c[3] > 0]
crosses.sort()
cross_mask = np.zeros_like(K)
for x, y, *_ in crosses:
    cross_mask[int(y) - 13:int(y) + 14, int(x) - 13:int(x) + 14] = True


def run_trace(guide, hw, mask, binlen=3.0):
    P = [(topx(X), topy(Y)) for X, Y in guide]
    return trace(K, P, hw, mask=mask, binlen=binlen)

solid_g = [(1.02,-23),(1.03,-21),(1.04,-19.5),(1.06,-17.5),(1.1,-15.5),(1.2,-11.8),(1.3,-8.7),(1.4,-6.3),(1.5,-4.3),(1.6,-2.9),(1.7,-1.6),(1.8,-0.8),(1.9,-0.25),(1.98,0)]
dash_g = [(1.02,-29.5),(1.04,-25),(1.1,-21.5),(1.2,-18.5),(1.3,-16),(1.4,-14.3),(1.5,-12.8),(1.6,-11.8),(1.7,-10.8),(1.8,-9.9),(1.9,-8.8),(1.98,-8.2)]
dot_g = [(1.01,-38),(1.05,-34),(1.1,-30.5),(1.2,-27.8),(1.3,-26.3),(1.45,-25.8),(1.55,-26.5),(1.65,-28.5),(1.7,-31),(1.75,-35),(1.79,-40)]
dot_g2 = [(1.86,-41),(1.9,-33),(1.93,-29),(1.97,-26)]
M = cross_mask | other
res = {}
for name, g, hw in [('solid', solid_g, 14), ('dash', dash_g, 14), ('dot', dot_g, 16), ('dot2', dot_g2, 16)]:
    t = run_trace(g, hw, M)
    # second pass: smooth guide from first pass
    xs = tox(t[:, 0]); ys = toy(t[:, 1])
    res[name] = t
    print(name, len(t))
np.save('f4_res.npy', {k: v for k, v in res.items()}, allow_pickle=True)
json.dump({'crosses': [(tox(x), toy(y), x, y, int(a), int(b)) for x, y, a, b in crosses],
           'cal': dict(mx=mx, cx=cx, my=my, cy=cy)}, open('f4_cal.json', 'w'), indent=1)
for x, y, a, b in crosses: print('cross %.4f %.3f  (%.1f,%.1f) %d %d' % (tox(x), toy(y), x, y, a, b))
overlay(img, [res['solid'][:, :2], res['dash'][:, :2], res['dot'][:, :2], res['dot2'][:, :2], [(c[0], c[1]) for c in crosses]],
        ['red', 'blue', 'green', 'green', 'magenta'], 'f4_overlay.png', crop=(1650, 320, 2520, 930), r=2)

# ================= pass 2 and output =================
from csvout import write_csv
final = {}
for name, hw2 in [('solid', 6), ('dash', 6)]:
    p1 = clean_outliers(res[name], win=30, tol=2.0)
    t2 = trace2(K, p1[:, :2], hw2, mask=M, binlen=3.0)
    t2 = clean_outliers(t2, win=30, tol=1.5)
    final[name] = t2
    print(name, 'pass2', len(t2), 'median half width px', np.median(t2[:, 3]))
for name in ('dot', 'dot2'):
    d = dots(K, res[name][:, :2], 8, mask=M)
    final[name] = d
    print(name, 'dots', len(d), 'areas', d[:, 2].min(), d[:, 2].max())
np.save('f4_final.npy', final, allow_pickle=True)
overlay(img, [final['solid'][:, :2], final['dash'][:, :2], final['dot'][:, :2], final['dot2'][:, :2], [(c[0], c[1]) for c in crosses]],
        ['red', 'blue', 'green', 'green', 'magenta'], 'f4_overlay2.png', crop=(1650, 320, 2520, 930), r=1)

rows = []
cal_px = 1.0
ux_cal = abs(mx) * cal_px; uy_cal = abs(my) * cal_px
for name, label in [('solid', 'd/b=1.41'), ('dash', 'd/b=0.707')]:
    t = final[name]
    xd = tox(t[:, 0]); yd = toy(t[:, 1])
    lo = xd.min(); hi = xd.max()
    grid = np.unique(np.round(np.concatenate([np.arange(1.0, 1.1, 0.005), np.arange(1.1, 2.0001, 0.01)]), 4))
    grid = grid[(grid >= lo) & (grid <= hi)]
    for g, y, u in to_grid(xd, yd, t[:, 3], grid, mx, my, maxgap=0.035, cal_px=cal_px):
        if name == 'solid' and y > -0.6:
            u = max(u, 0.3)   # curve merges with the dotted 0 dB frame line
        rows.append((label, g, y, u, 0.0))
for name in ('dot', 'dot2'):
    for x, y, a, h in final[name]:
        rows.append(('d/b=0.87', tox(x), toy(y), abs(my) * np.hypot(h, cal_px), abs(mx) * np.hypot(h, cal_px)))
# crosses: assign to series by position relative to the curves
for x, y, nv, nh in crosses:
    X, Y = tox(x), toy(y)
    # solid curve value at X
    ts = final['solid']; ys = np.interp(X, tox(ts[:, 0]), toy(ts[:, 1]))
    td = final['dash']; yd_ = np.interp(X, tox(td[:, 0]), toy(td[:, 1]))
    if abs(Y - ys) < 2: lab_ = 'ref7 d/b=1.41'
    elif abs(Y - yd_) < 2: lab_ = 'ref7 d/b=0.707'
    else: lab_ = 'ref7 d/b=0.87'
    rows.append((lab_, X, Y, abs(my) * np.hypot(1.0, cal_px), abs(mx) * np.hypot(1.0, cal_px)))
import pickle; pickle.dump(rows, open('f4_rows.pkl', 'wb'))
print(len(rows))

order = {'d/b=1.41': 0, 'd/b=0.707': 1, 'd/b=0.87': 2, 'ref7 d/b=1.41': 3, 'ref7 d/b=0.707': 4, 'ref7 d/b=0.87': 5}
rows.sort(key=lambda r: (order[r[0]], r[1]))
H = [
 "source: F. Alessandri, M. Mongiardo, R. Sorrentino, \"Rigorous mode matching analysis of mitered E-plane bends in rectangular waveguide\", IEEE Microwave and Guided Wave Letters 4(12), 408-410 (1994), doi:10.1109/75.336229",
 "figure: Fig. 4, p. 409 (PDF p. 2): \"Return loss of a 90 deg mitered E-plane bend in rectangular waveguide. The frequency is normalized with respect to the cut-off frequency. Crosses represent results from [7].\"",
 "ref [7] of the letter: J. M. Reiter and F. Arndt, \"A full-wave boundary contour mode-matching method (BCMM) for the rigorous CAD of single and cascaded optimized H-plane and E-plane bends\", 1994 IEEE MTT-S Int. Microwave Symp. Dig., San Diego, pp. 1021-1024",
 "curves: d/b=1.41 (solid line, no mitering), d/b=0.707 (dashed line, diagonal mitering), d/b=0.87 (dotted line, optimum); ref7 d/b=... = crosses of ref [7], assigned to the nearest curve",
 "geometry (inset of the figure and text p. 408): 90 deg E-plane bend; b = waveguide narrow wall (the side in the bend plane); the miter wall is perpendicular to the bend diagonal and d is its distance from the inner corner along the diagonal. d/b = 1.41 (sqrt 2): no mitering; d/b = 0.707 (1/sqrt 2): miter wall joining the ends of the two outer walls (only the triangular region D is present)",
 "guide dimensions: not stated in the letter (results are given against f/fc only)",
 "x: normalized frequency f/fc, fc = cut-off frequency of the fundamental mode (axis 1..2)",
 "y: |S11| in dB (axis label \"|S11| (dB)\", axis 0..-40 dB)",
 "uncertainty: reading uncertainty of y (dB) at the given x: half the printed line width (perpendicular, converted to vertical at fixed x with the local slope), 1 px calibration error in both axes, and interpolation error across dash gaps; where the d/b=1.41 curve merges with the 0 dB frame line (y > -0.6 dB) at least 0.3 dB",
 "x_uncertainty: for the dotted curve (dot centroids) and the crosses, the reading uncertainty of x; 0 for rows on the regular x grid of the solid and dashed curves (x is exact, its effect is in uncertainty)",
 "sampling: solid and dashed curves interpolated on x = 1.000:0.005:1.100 and 1.10:0.01:2.00 where ink exists (no point is interpolated across gaps > 0.035 in x); dotted curve: one row per printed dot (centroid); crosses: one row per cross (arm intersection)",
 "method: native 1-bit scan (339 dpi) extracted with pdfimages, deskewed by %.4f deg about the axis origin, ink traced along the curves with numpy/scipy (two passes, outlier rejection), dotted gridlines removed, crosses and labels masked; scripts in benchmarks/literature/raw/a/scripts/ (L1a; they read page images extracted from the reference PDFs, which are not in the repository)" % 0.6296,
 "calibration points (deskewed page pixels -> value): x ticks px " + ", ".join("%.2f->%.1f" % (a, b) for a, b in zip(XT, XV)) + "; y ticks px " + ", ".join("%.2f->%g" % (a, b) for a, b in zip(YT, YV)),
 "calibration fit: x = %.7g*px + %.6g (max residual %.4f), y = %.6g*py + %.6g (max residual %.3f dB); gridlines agree with the tick calibration within 1.5 px" % (mx, cx, abs(rx).max(), my, cy, abs(ry).max()),
 "axis resolution: minor ticks 0.05 in x every %.1f px (0.5*0.05/%.1f = %.5f), 2 dB in y every %.1f px (0.5*2/%.1f = %.3f dB)" % (0.05/mx, 0.05/mx, 0.025*mx/0.05, 2/abs(my), 2/abs(my), abs(my)/2),
 "digitizer: L1a (independent of L1b), 2026-11-08",
]
write_csv('/home/claude/wt/L1a-scratch/out/alessandri1994_fig4.csv', H, rows)
