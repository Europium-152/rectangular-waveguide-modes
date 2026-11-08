exec(open('/home/claude/wt/L1a-scratch/ma_common.py').read())
from ma_out import ma_header
img = load_gray('ma/p-004.png'); I = ink(img)
FIG = sys.argv[1]
if FIG == '8':
    L, R, T, B = 354, 1048.5, 174, 676
    Xb = [355.7, 392.5, 539.5, 614.0, 650.0, 684.5, 721.5, 758.1, 794.5, 831.5, 867.5, 903.9, 941.0, 976.9, 1015.6, 1049.5]
    Xt = [353.9, 383.4, 409.0, 413.0, 427.5, 464.5, 501.5, 538.0, 575.5, 612.0, 648.5, 682.9, 720.1, 756.5, 792.5, 829.2, 865.5, 975.2, 1048.0]
    Yl = [179.1, 251.4, 322.7, 395.5, 464.5, 536.4, 607.8, 678.5]; Yr = [171.5, 243.5, 315.6, 388.5, 457.5, 529.2, 600.5, 672.0]
    crop = (150, 100)
else:
    L, R, T, B = 358.5, 1059, 2105.5, 2610
    Xb = [360.2, 396.1, 433.5, 471.0, 508.5, 545.5, 582.6, 619.4, 656.4, 690.0, 727.9, 764.6, 802.1, 840.0, 876.9, 913.5, 950.5, 987.5, 1023.5, 1041.6, 1060.5]
    Xt = [357.5, 370.0, 380.6, 393.3, 431.4, 469.0, 506.5, 543.0, 580.5, 617.4, 654.0, 688.1, 725.5, 762.6, 800.4, 837.5, 874.5, 911.0, 948.0, 984.6, 1021.0, 1058.4]
    Yl = [2109.1, 2180.0, 2252.6, 2324.5, 2394.0, 2465.2, 2537.0, 2611.5]; Yr = [2103.5, 2176.8, 2248.6, 2321.1, 2390.5, 2462.2, 2534.1, 2608.5]
    crop = (150, 2050)
YV = [0, -5, -10, -15, -20, -25, -30, -35]
def assign(xs, x0, x1):
    sp = (x1 - x0) / 19.0; out = []
    for x in xs:
        k = round((x - x0) / sp)
        if abs(x - (x0 + k * sp)) < 4: out.append((x, 9.5 + 0.5 * k))
    return out
xb = assign(Xb, Xb[0], Xb[-1]); xt = assign(Xt, Xt[0], Xt[-1])
# affine calibration: X from tick points (x, y_axisline), Y from (x_axisline, y)
ptsX = [(x, B, v) for x, v in xb] + [(x, T, v) for x, v in xt]
ptsY = [(L, y, v) for y, v in zip(Yl, YV)] + [(R, y, v) for y, v in zip(Yr, YV)]
A = np.array([[p[0], p[1], 1] for p in ptsX]); cX, *_ = np.linalg.lstsq(A, np.array([p[2] for p in ptsX]), rcond=None)
A2 = np.array([[p[0], p[1], 1] for p in ptsY]); cY, *_ = np.linalg.lstsq(A2, np.array([p[2] for p in ptsY]), rcond=None)
rX = A @ cX - np.array([p[2] for p in ptsX]); rY = A2 @ cY - np.array([p[2] for p in ptsY])
print('X ticks used', len(ptsX), 'max resid %.4f GHz' % abs(rX).max(), ' Y max resid %.3f dB' % abs(rY).max())
def todata(px, py): return cX[0] * px + cX[1] * py + cX[2], cY[0] * px + cY[1] * py + cY[2]
Minv = np.linalg.inv(np.array([[cX[0], cX[1]], [cY[0], cY[1]]]))
def topix(X, Y): return tuple(Minv @ (np.array([X - cX[2], Y - cY[2]])))
mx = np.hypot(cX[0], cX[1]); my = -np.hypot(cY[0], cY[1])   # GHz/px and dB/px (magnitudes)
M = np.zeros_like(I)
M[:, :int(L) + 6] = True; M[:, int(R) - 5:] = True; M[:int(T) + 3, :] = True; M[int(B) - 5:, :] = True
def cbox(u0, v0, u1, v1): M[v0 + crop[1]:v1 + crop[1], u0 + crop[0]:u1 + crop[0]] = True
cbox(205, 262, 545, 540)   # inset drawing and c/a label
cbox(580, 375, 830, 550)   # legend
print('runs at 12 GHz:', runs(np.nonzero((I & ~M)[:, int(topix(12, -5)[0])])[0]))
print('runs at 15 GHz:', runs(np.nonzero((I & ~M)[:, int(topix(15, -5)[0])])[0]))
print('pix of (12,-5)', topix(12, -5), 'of (15,-10)', topix(15, -10))
for X in (10.75, 11.25, 12.25, 13.25, 14.25, 15.25, 16.25):
    px = int(round(topix(X, -5)[0])); rr = runs(np.nonzero((I & ~M)[:, px])[0])
    print(X, px, [(a, b, round(todata(px, (a + b) / 2)[1], 2)) for a, b in rr])
box = (L + 6, T + 3, R - 5, B - 5)
def snap(p, r=6):
    p = (int(round(p[0])), int(round(p[1])))
    sub = I[p[1] - r:p[1] + r + 1, p[0] - r:p[0] + r + 1] & ~M[p[1] - r:p[1] + r + 1, p[0] - r:p[0] + r + 1]
    ys, xs = np.nonzero(sub); j = np.argmin((xs - r) ** 2 + (ys - r) ** 2)
    return (p[0] - r + xs[j], p[1] - r + ys[j])
def run(XY, dXY, wm, maxgap, mo=1.5, gs=0.15, hw=6):
    s0 = snap(topix(*XY)); p2 = topix(XY[0] + dXY[0] * 0.01, XY[1] + dXY[1] * 0.01)
    d = np.array(p2) - np.array(topix(*XY)); d /= np.hypot(*d)
    kw = dict(step=3, hw=hw, maxgap=maxgap, box=box, width_max=wm, mask=M, quad=True, nfit=10, gap_slope=gs, max_off=mo)
    f1 = follow(I, s0, d, **kw); f2 = follow(I, s0, -d, **kw)
    f = np.vstack([f2[::-1], f1[1:]]); return f[f[:, 2] > 0]
if FIG == '8':
    S = {'(a)': ([(13.25, -5.58, 6, 1.5), (12.75, -4.92, 6, 1.5), (17.25, -15.43, 16, 3.0), (17.75, -18.15, 16, 3.0), (10.25, -0.97, 8, 1.5)], 99),
         '(b)': ([(12.25, -3.0, 6, 1.5), (14.25, -6.37, 6, 1.5), (11.25, -1.79, 6, 1.5), (13.75, -5.43, 6, 1.5)], 16.8),
         '(c)': ([(12.75, -5.9, 6, 2.5), (13.25, -6.45, 6, 2.5), (14.25, -8.19, 6, 2.5), (15.75, -10.58, 6, 2.5)], 16.8)}
else:
    S = {'(a)/(b)/(c)': ([(14.25, -7.88, 16, 3.0), (12.25, -4.48, 16, 3.0), (16.25, -12.18, 16, 3.0), (17.75, -17.94, 16, 3.0), (18.25, -21.97, 16, 3.0), (10.25, -1.28, 16, 3.0), (11.25, -2.88, 16, 3.0)], 99)}
def xd(f): return np.array([todata(*p)[0] for p in f[:, :2]])
paths = {}
for k, (seeds, xmax) in S.items():
    fs = []
    for X, Y, wm, mo in seeds:
        f = run((X, Y), (1, -2.5), wm, 20, mo=mo, gs=0.3 if k == '(c)' else 0.15, hw=8 if wm > 10 else 6)
        fs.append(f)
    f = np.vstack(fs); f = f[np.argsort(f[:, 0])]; f = f[xd(f) <= xmax]
    q = []; cur = [f[0, :2]]
    for p in f[1:, :2]:
        if p[0] - cur[-1][0] < 1.5 and abs(p[1] - cur[-1][1]) < 4: cur.append(p)
        else: q.append(np.mean(cur, 0)); cur = [p]
    q.append(np.mean(cur, 0)); paths[k] = np.array(q)
def med_filter(p, w=15, tol=3.0):
    keep = np.ones(len(p), bool)
    for i in range(len(p)):
        nb = np.abs(p[:, 0] - p[i, 0]) <= w
        keep[i] = abs(p[i, 1] - np.median(p[nb, 1])) <= tol
    return p[keep]
for k in paths: paths[k] = med_filter(med_filter(paths[k]))
if FIG == '8':
    # (c) is a sparse dotted line that the follower confuses with (a): take it as isolated dots below (a)
    pa = paths['(a)']; pb = paths['(b)']
    am = np.zeros_like(I)
    for x, y in np.vstack([pa, pb]): am[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4] = True
    lab, n = ndimage.label(I & ~M & ~am, structure=np.ones((3, 3)))
    cand = []
    for i, sl_ in enumerate(ndimage.find_objects(lab)):
        yy_, xx_ = np.nonzero(lab[sl_] == i + 1)
        if not (3 <= len(xx_) <= 30): continue
        cxp = xx_.mean() + sl_[1].start; cyp = yy_.mean() + sl_[0].start
        ya = np.interp(cxp, pa[:, 0], pa[:, 1])
        if 4 <= cyp - ya <= 25 and todata(cxp, cyp)[0] <= 16.8:
            cand.append((cxp, cyp, len(xx_), 0.5 * max(xx_.max() - xx_.min() + 1, yy_.max() - yy_.min() + 1)))
    cand = np.array(sorted(cand)); cand = clean_outliers(cand, win=40, tol=2.0)
    paths['(c)'] = cand[:, :2]; DOTS_C = cand
    print('(c) dots', len(cand))
fin = {}
for k, (seeds, xmax) in S.items():
    if k == '(c)':
        fin[k] = DOTS_C; continue
    om = np.zeros_like(I)
    for j in paths:
        if j == k: continue
        for x, y in paths[j]: om[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4] = True
    if k == '(a)':
        om[:, int(topix(16.8, -13)[0]):] = False   # merged band: no crossing mask
    t = trace2(I, paths[k], 7 if FIG == '9' else 4, mask=M | om, binlen=3.0)
    t = clean_outliers(t, win=30, tol=1.5)
    t = t[xd(t) <= xmax]
    fin[k] = t
    D = np.array([todata(*p) for p in t[:, :2]])
    print(k, len(t), 'x %.2f..%.2f' % (D[:, 0].min(), D[:, 0].max()), 'median half width %.2f' % np.median(t[:, 3]))
overlay(img, [fin[k][:, :2] for k in fin], ['red', 'blue', 'green'][:len(fin)], 'ma%s_ov.png' % FIG, crop=(L - 10, T - 10, R + 10, B + 10), scale=1.3, r=1)
rows = []; cal_px = 1.5
for k, t in fin.items():
    D = np.array([todata(*p) for p in t[:, :2]])
    grid = np.round(np.arange(9.5, 19.0001, 0.05), 3); grid = grid[(grid >= D[:, 0].min()) & (grid <= D[:, 0].max())]
    for g, y, u in to_grid(D[:, 0], D[:, 1], t[:, 3], grid, mx, my, maxgap=0.2, cal_px=cal_px):
        rows.append((k, g, y, u, 0.0))
cal = (mx, cX[2], my, cY[2], rX, rY, [p[0] for p in ptsX], [p[2] for p in ptsX], [p[1] for p in ptsY], [p[2] for p in ptsY])
if FIG == '8':
    cap = 'Difference between the return loss of a mitered bend calculated with three different division methods (the number of the division steps is 20).'
    curves = "division method (a) (solid, 9.6-18.25 GHz), (b) (long-dashed, highest curve; digitized 10.7-14.9 GHz, above that its dashes come within about 3 px of (a) and are not separated), (c) (dotted, lowest curve; dots isolated from (a) between 10.0 and about 16.3 GHz, gridded from the dot centroids); above about 16.8 GHz the three lines coincide within the line width (the dotted (c) runs up to about 0.2 GHz right of (a) between 18.2 and 18.6 GHz, not digitized)"
    nsteps = 20
else:
    cap = 'Difference between the return loss of a mitered bend calculated with three different division methods (the number of the division steps is 60).'
    curves = "(a)/(b)/(c): the three division methods coincide within the printed line width over the whole band (text p. 5: with 60 steps the difference becomes much smaller); the centre of the merged ink band is digitized, its half width is in the uncertainty"
    nsteps = 60
H = ma_header(FIG, 5, cap, curves, "frequency in GHz (axis 9.5..19.0, ticks every 0.5 GHz)", "|S11| in dB (axis 0..-35 dB, ticks every 5 dB)", cal,
    extra=["geometry: 90 deg mitered H-plane bend with c/a = 1.0 (inset), staircase approximation of the miter with %d steps; division methods (a), (b), (c) of Fig. 7: steps through, below and above the slanted plane" % nsteps,
           "uncertainty: reading uncertainty of y (dB) at the given x: half the printed (merged) line width (perpendicular, converted to vertical with the local slope), 1.5 px calibration error in both axes, interpolation across dash gaps",
           "x_uncertainty: 0 (regular grid; x reading errors are included in uncertainty)",
           "sampling: x = 9.50:0.05:19.00 GHz where ink exists, no interpolation across gaps > 0.2 GHz",
           "calibration: affine fit (rotation and shear of the scan, about 0.2-0.6 deg) of X to the x ticks on the top and bottom frame and of Y to the y ticks on the left and right frame; the calibration points list x ticks as px -> GHz (paired with the frame row) and y ticks as py -> dB (left then right frame). Affine coefficients: X = %.6g*px + %.6g*py + %.6g, Y = %.6g*px + %.6g*py + %.6g" % (cX[0], cX[1], cX[2], cY[0], cY[1], cY[2]),
           "axis resolution: ticks 0.5 GHz every %.1f px (%.4f GHz) and 5 dB every %.1f px (%.3f dB) (half tick spacing divided by the pixels between ticks)" % (0.5 / mx, mx / 2, 5 / abs(my), abs(my) / 2)],
    method_extra='; scan skew handled by an affine calibration')
H = [h.replace('(no measurable skew)', '(skewed, see calibration)') for h in H]
write_csv('/home/claude/wt/L1a-scratch/out/ma1997_fig%s.csv' % FIG, H, rows)
print('rows', len(rows))
