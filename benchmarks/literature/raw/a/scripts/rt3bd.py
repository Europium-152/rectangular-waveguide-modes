exec(open('/home/claude/wt/L1a-scratch/rt_out.py').read())
FIG = sys.argv[1]
if FIG == '3b':
    box = (1640, 560, 2420, 1100); cal = grid_calib(1640, 580, 2420, 1080, [13, 14, 15, 16, 17], [-30, -35, -40, -45, -50]); yr = (-55, -25)
    SEEDS = {'gamma=105 (a_opt=0.949a)': [((13.0, -31), (1, -5)), ((16.5, -31), (1, 4))],
             'gamma=120 (a_opt=0.955a)': [((13.0, -36.5), (1, -4)), ((16.5, -39), (1, 3))],
             'gamma=135 (a_opt=0.956a)': [((12.7, -49.5), (1, -6)), ((17.2, -51), (1, 0))]}
else:
    box = (1640, 1340, 2420, 1920); cal = grid_calib(1640, 1360, 2420, 1900, [13, 14, 15, 16, 17], [-40, -45, -50, -55, -60]); yr = (-65, -35)
    SEEDS = {'gamma=105 (b_opt=0.873b)': [((13.5, -37.5), (1, -0.6)), ((17.6, -57), (1, 20))],
             'gamma=120 (b_opt=0.898b)': [((14.5, -42.9), (1, 0.2))],
             'gamma=135 (b_opt=0.918b)': [((13.5, -51.75), (1, 0))]}
cX, cY, P, res, slopes = cal
print('resid', abs(res).max(0), 'slopes', min(slopes), max(slopes))
J, comps, mx, my = panel(*box, cX, cY, (12, 18), yr)
K = np.zeros_like(I)
for c in comps: K[c['sl']] |= c['mask']
Minv = np.linalg.inv(np.array([[cX[0], cX[1]], [cY[0], cY[1]]]))
topix = lambda X, Y: tuple(Minv @ np.array([X - cX[2], Y - cY[2]]))
if SEEDS is None:
    for X in (12.5, 13.5, 14.5, 15.5, 16.5, 17.5):
        px = int(round(topix(X, -50)[0])); rr = runs(np.nonzero(K[:, px])[0])
        print(X, [(round(todata(cX, cY, px, (a + b) / 2)[1], 2)) for a, b in rr])
    sys.exit()
def snap(p, r=15):
    p = (int(round(p[0])), int(round(p[1])))
    ys, xs = np.nonzero(K[p[1] - r:p[1] + r + 1, p[0] - r:p[0] + r + 1]); j = np.argmin((xs - r) ** 2 + (ys - r) ** 2)
    return (p[0] - r + xs[j], p[1] - r + ys[j])
fb = (box[0] + 5, box[1], box[2] - 5, box[3])
rows = []; tr = []
for lab_, seeds in SEEDS.items():
    TT = []
    for (X, Y), (dx, dy) in seeds:
        s0 = snap(topix(X, Y)); p2 = topix(X + dx * 0.01, Y + dy * 0.01); d = np.array(p2) - np.array(topix(X, Y)); d /= np.hypot(*d)
        kw = dict(step=3, hw=6, maxgap=20, box=fb, width_max=6, mask=~J, quad=True, nfit=10, gap_slope=0.15, max_off=3.0)
        f1 = follow(K, s0, d, **kw); f2 = follow(K, s0, -d, **kw)
        f = np.vstack([f2[::-1], f1[1:]]); f = f[f[:, 2] > 0]
        t = trace2(K, f[:, :2], 4, mask=~J, binlen=3.0); t = clean_outliers(t, win=30, tol=1.5)
        D = np.array([todata(cX, cY, *p[:2]) for p in t])
        keep = D[:, 1] > yr[0] + 0.5
        t = t[keep]; tr.append(t)
        print(lab_, (X, Y), 'x %.2f..%.2f' % (D[keep, 0].min(), D[keep, 0].max()))
        TT.append(t)
    T = np.vstack(TT); T = T[np.argsort(T[:, 0])]
    rows += grid_rows(lab_, T, cX, cY, mx, my, step=0.05, maxgap=0.08)
rows.sort(key=lambda r: (r[0], r[1]))
overlay(img, [t[:, :2] for t in tr], ['red', 'red', 'red', 'blue', 'blue', 'blue', 'green', 'green', 'green', 'green'], 'rt%s_ov.png' % FIG, crop=box, r=1)
pickle.dump(rows, open('rt%s_rows.pkl' % FIG, 'wb'))

plane = 'H' if FIG == '3b' else 'E'; side = 'a' if FIG == '3b' else 'b'
if FIG == '3b':
    curves = "gamma=105 deg with a_opt = 0.949 a, gamma=120 deg with a_opt = 0.955 a, gamma=135 deg with a_opt = 0.956 a (solid lines, values from the labels); the reflection zeros lie below the -55 dB frame and are not in the data"
else:
    curves = "gamma=105 deg with b_2opt = 0.873 b, gamma=120 deg with b_2opt = 0.898 b, gamma=135 deg with b_2opt = 0.918 b (solid lines, values from the labels, printed as b with subscript '2opt'); the reflection zero of gamma=105 (16.6-17.2 GHz) lies below the -65 dB frame"
H = header(FIG[0] + '(' + FIG[1] + ')', "Return loss curves of arbitrary angle H-plane or E-plane bends (waveguide Ku-band, WR62 housing: 15.799 x 7.899 mm). (%s) Mitered %s-plane bend." % (FIG[1], plane),
    curves,
    "mitered %s-plane bend in WR62 (a = 15.799 mm, b = 7.899 mm), side %s in the bend plane; gamma is the interior angle between the inner walls of the two arms (inset; gamma = 180 deg is a straight guide, deflection = 180 deg - gamma); %s_opt is the distance from the inner corner to the miter wall, measured perpendicular to the miter wall (inset). Text p. 799: 8 modes in the waveguide ports and 15 modes in the cavity subregions" % (plane, side, side),
    "frequency in GHz (axis 12.0..18.0, dotted grid every 1 GHz)", "|S11| in dB (axis label |S11|/dB, axis %d..%d dB, dotted grid every 5 dB)" % (yr[1], yr[0]), P, res, cX, cY, slopes,
    extra=["uncertainty: reading uncertainty of y (dB) at the given x: half the printed line width (perpendicular, converted to vertical with the local slope) and 1 px calibration error in both axes; large near the steep reflection zeros",
           "x_uncertainty: 0 (regular grid)", "sampling: x on a 0.05 GHz grid where ink exists (no interpolation across gaps > 0.08 GHz); points within 0.5 dB of the bottom frame left out; leader lines from the labels touch the curves at one point each (small local bias, within the uncertainty)",
           "axis resolution: no ticks; grid every 1 GHz (%.1f px) and 5 dB (%.1f px): 0.5 px = %.4f GHz and %.3f dB" % (1 / mx, 5 / my, mx / 2, my / 2)])
write_csv('/home/claude/wt/L1a-scratch/out/reiter1995_fig%s.csv' % FIG, H, rows)
print('rows', len(rows))
