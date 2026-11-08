exec(open('/home/claude/wt/L1a-scratch/ma11.py').read())
F = pickle.load(open('ma11_F.pkl', 'rb'))
paths = {}
for k, fl in F.items():
    pts = np.vstack([f[f[:, 2] > 0][:, :2] for f in fl])
    if k == '0.70': pts = pts[~((tox(pts[:, 0]) > 1.775) & (tox(pts[:, 0]) < 1.823))]  # the (1.6,-24)/(1.75,-20.5) runs end at 1.773
    paths[k] = pts[np.argsort(pts[:, 0])]
fin = {}
for k in paths:
    om = np.zeros_like(I)
    for j in paths:
        if j == k: continue
        for x, y in paths[j]: om[int(y) - 4:int(y) + 5, int(x) - 4:int(x) + 5] = True
    t = trace2(K, paths[k], 4, mask=M | om, binlen=3.0)
    t = clean_outliers(t, win=24, tol=1.5)
    if k == '0.60':   # minimum region, where the c/a=0.631 hump touches the line: trace without the crossing mask
        p = paths[k]; sel = (tox(p[:, 0]) > 1.775) & (tox(p[:, 0]) < 1.85)
        t2 = trace2(K, p[sel], 3, mask=M, binlen=3.0)
        t2 = t2[(tox(t2[:, 0]) > 1.78) & (tox(t2[:, 0]) < 1.845)]
        t2[:, 3] = np.maximum(t2[:, 3], 3.0)   # overlap: widen the reading uncertainty
        t = np.vstack([t[(tox(t[:, 0]) <= 1.78) | (tox(t[:, 0]) >= 1.845)], t2]); t = t[np.argsort(t[:, 0])]
    fin[k] = t; print(k, len(t), 'median half width %.2f' % np.median(t[:, 3]))
cols = {'0.60': 'red', '0.631': 'blue', '0.70': 'green', '0': 'orange', '1.0': 'magenta', '1.30': 'cyan'}
overlay(img, [fin[k][:, :2] for k in fin], [cols[k] for k in fin], 'ma11_fin.png', crop=(1515, 180, 2295, 735), scale=1.5, r=1)
overlay(img, [fin[k][:, :2] for k in fin], [cols[k] for k in fin], 'ma11_fz.png', crop=(2080, 180, 2295, 735), scale=2.5, r=1)
pickle.dump(fin, open('ma11_fin.pkl', 'wb'))
rows = []; cal_px = 1.0
for k in ['0', '0.60', '0.631', '0.70', '1.0', '1.30']:
    t = fin[k]; xd = tox(t[:, 0]); yd = toy(t[:, 1])
    grid = np.round(np.arange(1.0, 2.00001, 0.005), 4); grid = grid[(grid >= xd.min()) & (grid <= xd.max())]
    for g, y, u in to_grid(xd, yd, t[:, 3], grid, mx, my, maxgap=0.03, cal_px=cal_px):
        if y < -49.5: continue
        if y > -0.8: u = max(u, 0.3)
        if k == '0.60' and 1.80 <= g <= 1.825: u = max(u, 0.7)   # pulled toward the touching c/a=0.631 line
        rows.append(('c/a=' + k, g, y, u, 0.0))
from ma_out import ma_header
H = ma_header('11', 6, 'Variation of the return loss of a mitered bend with different size of the mitered part of the bend.',
    "c/a=0 (long-dashed, unmitered corner), c/a=0.60 (solid), c/a=0.631 (long-short dashed; optimum, two reflection zeros near f/fc = 1.68 and 1.93), c/a=0.70 (short-dashed; zero near 1.38), c/a=1.0 (sparse dots), c/a=1.30 (dash pattern near the top). Line styles attributed by following each labelled curve; the horizontal dotted line at -30 dB is a reference line (removed). Curves are not digitized where they merge with the 0 dB frame (c/a=1.30 below f/fc = 1.40, c/a=0 near 1.81-1.91), within 4 px of a crossing with another curve, or below -49.5 dB. Exception: c/a=0.60 between 1.78 and 1.845, where the c/a=0.631 curve touches it near its minimum, is traced without the crossing mask and given at least 3 px half width (at least 0.7 dB for 1.800-1.825, where the centroid is pulled toward the c/a=0.631 line)",
    "normalized frequency f/fc, fc = TE10 cut-off = c/(2a) = 9.49 GHz for a = 15.799 mm (axis 1..2, ticks every 0.2)",
    "|S11| in dB (axis 0..-50 dB, ticks every 10 dB)",
    (mx, cx, my, cy, rx, ry, XT, XV, YT, YV),
    extra=["geometry: 90 deg mitered H-plane bend (inset of Figs. 6, 8, 9): the outer corner is cut by a straight miter whose ends lie at distance c from the removed outer corner along each outer wall; c/a = 0 is the unmitered corner. The text (p. 6) states that for 0.6 <= c/a <= 0.8 the return loss can be made small and that c/a = 0.631 gives the widest |S11| < -30 dB band, about 39% of the single-mode band. Staircase: 60 steps, division method (a) (text p. 5)",
           "uncertainty: reading uncertainty of y (dB) at the given x: half the printed line width (perpendicular, converted to vertical with the local slope), 1 px calibration error in both axes, interpolation across dash gaps; at least 0.3 dB where y > -0.8 dB (curve merges with the frame line)",
           "x_uncertainty: 0 (regular grid; x reading errors are included in uncertainty)",
           "sampling: x = 1.000:0.005:2.000 where ink exists, no interpolation across gaps > 0.03",
           "axis resolution: ticks 0.2 every %.1f px (0.5*0.2/%.1f = %.5f) and 10 dB every %.1f px (0.5*10/%.1f = %.3f dB); no minor ticks" % (0.2 / mx, 0.2 / mx, mx / 2, 10 / abs(my), 10 / abs(my), abs(my) / 2)],
    method_extra='; the -30 dB dotted reference line removed before tracing')
write_csv('/home/claude/wt/L1a-scratch/out/ma1997_fig11.csv', H, rows)
print('rows', len(rows))
import collections; print(collections.Counter(r[0] for r in rows))
