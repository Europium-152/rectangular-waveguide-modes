exec(open('/home/claude/wt/L1a-scratch/ma_common.py').read())
from ma_out import ma_header
img = load_gray('ma/p-003.png'); I = ink(img)
XT = [1556.0, 1590.8, 1625.1, 1659.5, 1694.1, 1728.7, 1763.1, 1798.0, 1832.6, 1864.7, 1899.5, 1934.0, 1968.5, 2003.5, 2038.0, 2073.0, 2108.5, 2143.4, 2177.5, 2212.5]
XV = list(np.arange(9.5, 19.01, 0.5))
YT = [168.8, 229.3, 287.3, 348.5, 406.5, 466.6, 524.4, 584.6, 642.8]; YV = [0, -2, -4, -6, -8, -10, -12, -14, -16]
mx, cx, rx = linfit(XT, XV); my, cy, ry = linfit(YT, YV)
print('cal resid max', abs(rx).max(), abs(ry).max(), 'per px', mx, my)
tox = lambda p: mx * np.asarray(p) + cx; toy = lambda p: my * np.asarray(p) + cy
topx = lambda X: (X - cx) / mx; topy = lambda Y: (Y - cy) / my
M = np.zeros_like(I)
M[:, :1561] = True; M[:, 2209:] = True; M[:172, :] = True; M[638:, :] = True
def cbox(u0, v0, u1, v1): M[v0 + 100:v1 + 100, u0 + 1300:u1 + 1300] = True   # crop offset (1300,100)
cbox(370, 285, 560, 515)   # legend
cbox(575, 228, 900, 505)   # inset + c/a label
for r_ in (5, 6, 7):
    rings = find_rings(I, r_, 0.45, (1561, 172, 2209, 638))
    rings = [q for q in rings if not M[int(q[1]), int(q[0])]]
    print(r_, len(rings), [(round(float(tox(q[0])), 2), round(float(toy(q[1])), 2), round(q[2], 2)) for q in sorted(rings)])
rings = find_rings(I, 7, 0.7, (1561, 172, 2209, 638))
rings = [q for q in rings if not M[int(q[1]), int(q[0])]]
R = []
for x, y, s in sorted(rings):
    a, b, rr, n = refine_ring(I, x, y, 7, t=2.5)
    a, b, rr, n = refine_ring(I, a, b, 7, t=2.5)
    if not any(np.hypot(a - q[0], b - q[1]) < 6 for q in R): R.append((a, b, rr, n, 1.0))
J = I.copy(); J[164:172, :] = False
for gx in (2081, 2150, 2212):
    sub = J[160:190, gx - 12:gx + 13]; ys, xs = np.nonzero(sub)
    a, b, rr, n = refine_ring(J, gx, 174, 7, t=2.5); a, b, rr, n = refine_ring(J, a, b, 7, t=2.5)
    R.append((a, b, rr, n, 2.0))
for q in R: print('ring %.3f %.3f r=%.1f n=%d e=%g' % (tox(q[0]), toy(q[1]), q[2], q[3], q[4]))
R = [q for q in R if abs(tox(q[0]) - 13.477) > 0.01]
yy, xx = np.mgrid[0:I.shape[0], 0:I.shape[1]]
CM = np.zeros_like(I)
for q in R:
    x0, y0 = int(q[0]), int(q[1])
    sl = (slice(y0 - 12, y0 + 13), slice(x0 - 12, x0 + 13))
    CM[sl] |= np.hypot(xx[sl] - q[0], yy[sl] - q[1]) <= 10
box = (1561, 170, 2209, 638)
def run(seed, d, wm, maxgap, gs, mo, mask):
    s0 = seed; d = np.array(d, float); d /= np.hypot(*d)
    kw = dict(step=3, hw=7, maxgap=maxgap, box=box, width_max=wm, mask=mask, quad=True, nfit=10, gap_slope=gs, max_off=mo)
    f1 = follow(I, s0, d, **kw); f2 = follow(I, s0, -d, **kw)
    return np.vstack([f2[::-1], f1[1:]])
F = {}
F['60 steps'] = run((1650, 281), (30, -39), 8, 30, 0.15, 1.5, M | CM)
def snap(p, r=8):
    sub = I[p[1] - r:p[1] + r + 1, p[0] - r:p[0] + r + 1]; ys, xs = np.nonzero(sub); j = np.argmin((xs - r) ** 2 + (ys - r) ** 2)
    return (p[0] - r + xs[j], p[1] - r + ys[j])
F20 = [run(snap(p), d, 5, 26, 0.15, 1.5, M | CM) for p, d in [((1620, 334), (30, -39)), ((1586, 415), (10, -40)), ((1772, 221), (40, -12))]]
F['20 steps'] = np.vstack(F20)
F['10 steps'] = run((1650, 331), (30, -40), 6, 30, 0.2, 1.5, M | CM)
for k, f in F.items(): print(k, 'x %.3f..%.3f' % (tox(f[:, 0].min()), tox(f[:, 0].max())), len(f), int(f[:, 2].sum()))
overlay(img, [F[k][:, :2] for k in F] + [[(q[0], q[1]) for q in R]], ['red', 'green', 'blue', 'magenta'], 'ma6_ov.png', crop=(1555, 165, 1950, 600), scale=2.4, r=1)
pickle.dump(dict(F=F, R=R), open('ma6_f.pkl', 'wb'))
fin = {}
for k, hw, xmax in [('60 steps', 6, 99), ('20 steps', 4, 12.6), ('10 steps', 5, 99)]:
    f = F[k]; f = f[f[:, 2] > 0]; f = f[np.argsort(f[:, 0])]
    f = f[tox(f[:, 0]) <= xmax]
    others = np.vstack([F[j][F[j][:, 2] > 0][:, :2] for j in F if j != k])
    om = np.zeros_like(I)
    for x, y in others: om[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4] = True
    if k == '60 steps': om[:] = False   # the solid is the upper envelope; the others are below it
    t = trace2(I, f[:, :2], hw, mask=M | CM | om, binlen=3.0)
    t = clean_outliers(t, win=30, tol=1.5)
    t = t[tox(t[:, 0]) <= xmax]
    fin[k] = t; print(k, len(t), np.median(t[:, 3]))
overlay(img, [fin[k][:, :2] for k in fin] + [[(q[0], q[1]) for q in R]], ['red', 'green', 'blue', 'magenta'], 'ma6_ov2.png', crop=(1555, 165, 1800, 450), scale=3.5, r=1)
rows = []; cal_px = 1.5
for k in fin:
    t = fin[k]; xd = tox(t[:, 0]); yd = toy(t[:, 1])
    grid = np.round(np.arange(9.5, 19.0001, 0.05), 3); grid = grid[(grid >= xd.min()) & (grid <= xd.max())]
    for g, y, u in to_grid(xd, yd, t[:, 3], grid, mx, my, maxgap=0.2, cal_px=cal_px):
        rows.append((k, g, y, u, 0.0))
for (a, b, rr, n, e) in R:
    rows.append(('BEM [9]', float(tox(a)), float(toy(b)), abs(my) * np.hypot(e, cal_px), abs(mx) * np.hypot(e, cal_px)))
order = {'60 steps': 0, '20 steps': 1, '10 steps': 2, 'BEM [9]': 3}
rows.sort(key=lambda r: (order[r[0]], r[1]))
H = ma_header('6', 4, 'Variation of the power transmission coefficient of a mitered bend with different numbers of steps used in the staircase approximation.',
    "60 steps (solid line; the 40-step long-dashed line overlaps it within the line width and is not digitized separately), 20 steps (dotted line, digitized 9.85-12.6 GHz only; above 12.6 GHz it merges with the solid line and the circles), 10 steps (long-short dashed line, lowest curve, digitized to 12.7 GHz; above it is within the line width of the others), BEM [9] = circles (boundary-element results; the text p. 4 cites them as [6], Koshiba and Suzuki 1986). The BEM frequencies coincide with those of Fig. 5",
    "frequency in GHz (axis 9.5..19.0, ticks every 0.5 GHz)",
    "|S21| in dB, TE10 transmission (axis 0..-16 dB, ticks every 2 dB)",
    (mx, cx, my, cy, rx, ry, XT, XV, YT, YV),
    extra=["geometry: 90 deg mitered H-plane bend, guide width a; the outer corner is cut by a straight miter whose ends are at distance c from the (removed) outer corner along each outer wall (inset); c/a = 1.0 here. The slanted miter is modelled by a staircase with the stated number of steps (Fig. 3), division method (a) of Fig. 7",
           "uncertainty: reading uncertainty of y (dB) at the given x: half the printed line width (perpendicular, converted to vertical with the local slope), a 1.5 px calibration error in both axes, interpolation across masked circles; circles: 1 px centre error (2 px for the three circles on the 0 dB frame line at 17.1, 18.1, 19.0 GHz) and 1.5 px calibration",
           "x_uncertainty: for circles the reading uncertainty of x; 0 for the regular grid of the lines",
           "sampling: lines on x = 9.50:0.05:19.00 GHz where ink exists (no interpolation across gaps > 0.2 GHz)",
           "axis resolution: ticks 0.5 GHz every %.1f px (%.4f GHz) and 2 dB every %.1f px (%.3f dB) (half tick spacing divided by the pixels between ticks)" % (0.5 / mx, mx / 2, 2 / abs(my), abs(my) / 2)])
write_csv('/home/claude/wt/L1a-scratch/out/ma1997_fig6.csv', H, rows)
print('rows', len(rows))
