exec(open('/home/claude/wt/L1a-scratch/ma_common.py').read())
img = load_gray('ma/p-003.png'); I = ink(img)
XT = [433.5, 468.8, 503.6, 539.3, 574.6, 610.2, 644.9, 681.1, 716.0, 748.7, 784.1, 819.0, 853.9, 889.0, 924.0, 959.0, 995.0, 1030.0, 1064.4, 1100.0]
XV = list(np.arange(9.5, 19.01, 0.5))
YT = [2601.5, 2576.0, 2548.0, 2521.6, 2493.5, 2468.0, 2439.5, 2414.0, 2386.0, 2361.0, 2335.0, 2307.0, 2282.2, 2254.5, 2229.0, 2201.1, 2175.5, 2148.5, 2123.5]
YV = list(np.round(np.arange(0, 0.901, 0.05), 3))
mx, cx, rx = linfit(XT, XV); my, cy, ry = linfit(YT, YV)
print('cal resid max', abs(rx).max(), abs(ry).max(), 'per px', mx, my)
tox = lambda p: mx * np.asarray(p) + cx; toy = lambda p: my * np.asarray(p) + cy
topx = lambda X: (X - cx) / mx; topy = lambda Y: (Y - cy) / my
c2 = lambda X, Y: (float(topx(X)), float(topy(Y)))
M = np.zeros_like(I)
M[:, :440] = True; M[:, 1094:] = True; M[:2131, :] = True; M[2596:, :] = True
# legend and inset (crop offset 150,2100)
def cbox(u0, v0, u1, v1): M[v0 + 2100:v1 + 2100, u0 + 150:u1 + 150] = True
cbox(400, 95, 640, 235)    # legend
cbox(375, 248, 655, 470)   # inset drawing
# circles
J = I.copy(); J[2598:2607, :] = False
rings = find_rings(J, 8, 0.45, (440, 2131, 1094, 2615))
rings = [r for r in rings if not M[int(r[1]), int(r[0])] or r[1] > 2590]
R = []
for x, y, s in rings:
    a, b, rr, n = refine_ring(J, x, y, 8, t=3)
    if 6.5 < rr < 9.5 and n > 120 and not any(np.hypot(a - q[0], b - q[1]) < 5 for q in R):
        R.append((a, b, rr, n, s))
R.sort()
for r in R: print('ring %.3f %.4f  r=%.1f n=%d s=%.2f' % (tox(r[0]), toy(r[1]), r[2], r[3], r[4]))
for guess in [(1034, 2595), (1033, 2597)]:
    a, b, rr, n = refine_ring(J, guess[0], guess[1], 8, t=2.5)
    a2, b2, rr2, n2 = refine_ring(J, a, b, 8, t=2.5)
    print('seeded ring', tox(a2), toy(b2), rr2, n2)
R.append((a2, b2, rr2, n2, 0.0)); R.sort()
yy, xx = np.mgrid[0:I.shape[0], 0:I.shape[1]]
CM = np.zeros_like(I)
for r in R:
    x0, y0 = int(r[0]), int(r[1])
    CM[y0 - 13:y0 + 14, x0 - 13:x0 + 14] |= (np.hypot(xx[y0 - 13:y0 + 14, x0 - 13:x0 + 14] - r[0], yy[y0 - 13:y0 + 14, x0 - 13:x0 + 14] - r[1]) <= 11)
MM = M | CM
box = (441, 2131, 1093, 2597)
st = c2(13.0, 0.862)
r = 15; sub = I[int(st[1]) - r:int(st[1]) + r + 1, int(st[0]) - r:int(st[0]) + r + 1] & ~MM[int(st[1]) - r:int(st[1]) + r + 1, int(st[0]) - r:int(st[0]) + r + 1]
ys, xs = np.nonzero(sub); j = np.argmin((xs - r) ** 2 + (ys - r) ** 2); st = (st[0] - r + xs[j], st[1] - r + ys[j])
kw = dict(step=3, hw=8, maxgap=40, box=box, width_max=9, mask=MM, quad=True, nfit=10, gap_slope=0.2, max_off=2.0)
f1 = follow(I, st, (1, 0), **kw); f2 = follow(I, st, (-1, 0), **kw)
fs = [np.vstack([f2[::-1], f1[1:]])]
for (X, Y, dx, dy) in [(17.35, 0.08, 0.2, -0.25), (18.3, 0.08, 0.2, 0.25)]:
    s0 = c2(X, Y)
    sub = I[int(s0[1]) - r:int(s0[1]) + r + 1, int(s0[0]) - r:int(s0[0]) + r + 1] & ~MM[int(s0[1]) - r:int(s0[1]) + r + 1, int(s0[0]) - r:int(s0[0]) + r + 1]
    ys, xs = np.nonzero(sub); j = np.argmin((xs - r) ** 2 + (ys - r) ** 2); s0 = (s0[0] - r + xs[j], s0[1] - r + ys[j])
    d = np.array([dx / mx, dy / my]); d /= np.hypot(*d)
    g1 = follow(I, s0, d, **kw); g2 = follow(I, s0, -d, **kw)
    fs.append(np.vstack([g2[::-1], g1[1:]]))
for f in fs: print('follow x %.3f..%.3f' % (tox(f[:, 0].min()), tox(f[:, 0].max())))
paths = [f[f[:, 2] > 0][:, :2] for f in fs]
ts = [clean_outliers(trace2(I, p, 6, mask=MM, binlen=3.0), win=30, tol=1.5) for p in paths]
t = np.vstack(ts); t = t[np.argsort(t[:, 0])]
print('trace', len(t), 'median half width', np.median(t[:, 3]))
overlay(img, [t[:, :2], [(r[0], r[1]) for r in R]], ['red', 'blue'], 'ma5_ov.png', crop=(400, 2100, 1120, 2620), scale=1.4, r=2)
overlay(img, [t[:, :2], [(r[0], r[1]) for r in R]], ['red', 'blue'], 'ma5_z.png', crop=(900, 2350, 1100, 2615), scale=2.5, r=1)
pickle.dump(dict(t=t, R=R, cal=(mx, cx, my, cy, rx, ry, XT, XV, YT, YV)), open('ma5.pkl', 'wb'))
from ma_out import ma_header
cal_px = 1.5
rows = []
xd = tox(t[:, 0]); yd = toy(t[:, 1])
grid = np.round(np.arange(9.5, 19.0001, 0.05), 3)
grid = grid[(grid >= xd.min()) & (grid <= xd.max())]
for g, y, u in to_grid(xd, yd, t[:, 3], grid, mx, my, maxgap=0.2, cal_px=cal_px):
    rows.append(('12 terms', g, y, u, 0.0))
for (a, b, rr, n, s) in R:
    e = 2.0 if s == 0.0 else 1.0
    rows.append(('BEM [9]', float(tox(a)), float(toy(b)), abs(my) * np.hypot(e, cal_px), abs(mx) * np.hypot(e, cal_px)))
H = ma_header('5', 4, 'Variation of the power transmission coefficient of a waveguide right angle corner bend with different numbers of modal terms in the electromagnetic field expressions.',
    "12 terms (solid line; the 6-term dashed line coincides with it within the line width everywhere, the 3-term dotted line except near 16.5-17.5 GHz and above 18.1 GHz, where its dots lie up to about 0.02 left of the solid line; neither is digitized separately); BEM [9] = circles (ref. [9] of the paper, boundary-element results; the reference list ends at [8], text p. 4 cites them as [6], Koshiba and Suzuki 1986)",
    "frequency in GHz (axis 9.5..19.0, ticks every 0.5 GHz)",
    "|S21|^2, power transmission coefficient of the TE10 mode (linear, axis 0..0.9, ticks every 0.05)",
    (mx, cx, my, cy, rx, ry, XT, XV, YT, YV),
    extra=["geometry: right-angle (unmitered) H-plane corner bend (Fig. 1(a) and inset), both arms of width a",
           "uncertainty: reading uncertainty of y at the given x: half the printed line width (perpendicular, converted to vertical with the local slope) and a 1.5 px calibration error in both axes (the printed ticks are irregular by up to 1.7 px), plus the interpolation error across masked circles; circles: 1 px centre error (2 px for the circle at 18.05 GHz, which overlaps the axis and the curve) and 1.5 px calibration",
           "x_uncertainty: for circles the reading uncertainty of x; 0 for the regular grid of the line",
           "sampling: line on x = 9.50:0.05:19.00 GHz where ink exists (no interpolation across gaps > 0.2 GHz); the null near 17.6-18.1 GHz, where the line runs into the axis and the circles, is left out",
           "axis resolution: ticks 0.5 GHz every %.1f px (0.5*0.5/%.1f = %.4f GHz) and 0.05 every %.1f px (0.5*0.05/%.1f = %.5f)" % (0.5 / mx, 0.5 / mx, 0.25 * mx / 0.5 * 2 / 2, 0.05 / abs(my), 0.05 / abs(my), abs(my) / 2)])
write_csv('/home/claude/wt/L1a-scratch/out/ma1997_fig5.csv', H, rows)
print('rows', len(rows))
