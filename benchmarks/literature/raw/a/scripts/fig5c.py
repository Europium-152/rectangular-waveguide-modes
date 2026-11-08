import sys; sys.path.insert(0, '/home/claude/wt/L1a-scratch')
exec(open('fig5.py').read().split("p1 = {}")[0])
from scipy import ndimage
F = pickle.load(open('f5_follow.pkl', 'rb'))
keys = list(F)
H_, W_ = K.shape
def near_mask(paths, r):
    m = np.zeros_like(K)
    for P in paths:
        for x, y in P[:, :2]:
            m[int(y) - r:int(y) + r + 1, int(x) - r:int(x) + r + 1] = True
    return m
fin = {}
for k in keys:
    own = F[k][F[k][:, 2] > 0][:, :2]
    others = [F[j] for j in keys if j != k and j[:4] != k[:4]]  # other curves (branches of the same curve are not crossings)
    m = other | near_mask(others, 6)
    hw = 7 if k.startswith('0.85') else 5
    t = trace2(K, own, hw, mask=m, binlen=3.0)
    t = clean_outliers(t, win=30, tol=1.5)
    fin[k] = t
    print(k, len(t), 'median half width %.2f' % np.median(t[:, 3]))
cols = {'0.9': 'red', '0.87a': 'orange', '0.87b': 'orange', '0.85a': 'blue', '0.85b': 'blue', '0.83a': 'green', '0.83b': 'green'}
overlay(img, [fin[k][:, :2] for k in cols], [cols[k] for k in cols], 'f5_fin.png', crop=(1600, 1180, 2560, 1880), r=1)
overlay(img, [fin[k][:, :2] for k in cols], [cols[k] for k in cols], 'z9.png', crop=(2150, 1400, 2480, 1700), scale=2.5, r=1)
pickle.dump(fin, open('f5_fin.pkl', 'wb'))

rows = []; cal_px = 1.0
label = {'0.9': 'd/b=0.9', '0.87a': 'd/b=0.87', '0.87b': 'd/b=0.87', '0.85a': 'd/b=0.85', '0.85b': 'd/b=0.85', '0.83a': 'd/b=0.83', '0.83b': 'd/b=0.83'}
for k in keys:
    t = fin[k]; xd = tox(t[:, 0]); yd = toy(t[:, 1])
    grid = np.unique(np.round(np.arange(1.0, 2.00001, 0.005), 4))
    grid = grid[(grid >= xd.min()) & (grid <= xd.max())]
    for g, y, u in to_grid(xd, yd, t[:, 3], grid, mx, my, maxgap=0.035, cal_px=cal_px):
        if y < -49.6: continue   # within the bottom axis line
        rows.append((label[k], g, y, u, 0.0))
order = {'d/b=0.9': 0, 'd/b=0.87': 1, 'd/b=0.85': 2, 'd/b=0.83': 3}
rows.sort(key=lambda r: (order[r[0]], r[1]))
pickle.dump(rows, open('f5_rows.pkl', 'wb'))
H = [
 "source: F. Alessandri, M. Mongiardo, R. Sorrentino, \"Rigorous mode matching analysis of mitered E-plane bends in rectangular waveguide\", IEEE Microwave and Guided Wave Letters 4(12), 408-410 (1994), doi:10.1109/75.336229",
 "figure: Fig. 5, p. 409 (PDF p. 2): \"Return loss of a mitered E-plane bend in rectangular waveguide. The frequency is normalized with respect to the cut-off frequency. The dept[h] of the mitering is described by the parameter d/b.\"",
 "curves: d/b=0.9 (upper solid line), d/b=0.87 (lower solid line; zero near x=1.83-1.85, label at the right branch), d/b=0.85 (thick dashed), d/b=0.83 (thin dashed). Line-style attribution from the labels in the figure; the 0.87 curve agrees with the dotted d/b=0.87 curve of Fig. 4",
 "geometry (inset of the figure and text p. 408): 90 deg E-plane bend; b = waveguide narrow wall (the side in the bend plane); the miter wall is perpendicular to the bend diagonal and d is its distance from the inner corner along the diagonal (d/b = 1.41 unmitered, 0.707 diagonal miter)",
 "guide dimensions: not stated in the letter (results are given against f/fc only)",
 "x: normalized frequency f/fc, fc = cut-off frequency of the fundamental mode (axis 1..2)",
 "y: |S11| in dB (axis label \"|S11| (dB)\", axis 0..-50 dB); the curves leave the plot below -50 dB near their reflection zeros, so the zeros themselves are not in the data",
 "uncertainty: reading uncertainty of y (dB) at the given x: half the printed line width (perpendicular, converted to vertical at fixed x with the local slope), 1 px calibration error in both axes, and interpolation error across dash gaps and masked crossings",
 "x_uncertainty: 0 (x is on a regular grid; the effect of x reading errors is included in uncertainty)",
 "sampling: x = 1.000:0.005:2.000 where ink exists; no point interpolated across gaps > 0.035 in x; regions within 6 px of a crossing with another curve and points below -49.6 dB (bottom axis) are left out",
 "method: native 1-bit scan (339 dpi) extracted with pdfimages, deskewed by 0.5397 deg about the axis origin, curves tracked by a curvature-aware follower and then measured by ink centroids in 3 px bins along the track (numpy/scipy); labels and inset masked; scripts in benchmarks/literature/raw/a/scripts/ (L1a; they read page images extracted from the reference PDFs, which are not in the repository)",
 "calibration points (deskewed page pixels -> value): x ticks px " + ", ".join("%.2f->%.1f" % (a, b) for a, b in zip(XT, XV)) + "; y ticks px " + ", ".join("%.2f->%g" % (a, b) for a, b in zip(YT, YV)),
 "calibration fit: x = %.7g*px + %.6g (max residual %.4f), y = %.6g*py + %.6g (max residual %.3f dB)" % (mx, cx, abs(rx).max(), my, cy, abs(ry).max()),
 "axis resolution: minor ticks 0.05 in x every %.1f px (0.5*0.05/%.1f = %.5f), 2 dB in y every %.1f px (0.5*2/%.1f = %.3f dB)" % (0.05 / mx, 0.05 / mx, mx / 2, 2 / abs(my), 2 / abs(my), abs(my) / 2),
 "digitizer: L1a (independent of L1b), 2026-11-08",
]
write_csv('/home/claude/wt/L1a-scratch/out/alessandri1994_fig5.csv', H, rows)
print(len(rows))
