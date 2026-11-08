exec(open('/home/claude/wt/L1a-scratch/rt_out.py').read())
XT = [573.0, 631.2, 728.0, 825.2, 923.0, 1020.0, 1117.0, 1214.0, 1311.0, 1389.0]; XV = [8.2, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.4]
YT = [2222.0, 2304.5, 2386.5, 2549.9, 2713.0]; YV = [-10, -20, -30, -50, -70]
mx, cx, rx = linfit(XT, XV); my, cy, ry = linfit(YT, YV)
print('resid', abs(rx).max(), abs(ry).max(), mx, my)
cX = np.array([mx, 0, cx]); cY = np.array([0, my, cy])
tox = lambda p: mx * np.asarray(p) + cx; toy = lambda p: my * np.asarray(p) + cy
topix = lambda X, Y: ((X - cx) / mx, (Y - cy) / my)
box = (560, 2200, 1400, 2730)
J, comps, _, _ = panel(*box, cX, cY, (8.2, 12.4), (-70, -10))
K = np.zeros_like(I)
for c in comps: K[c['sl']] |= c['mask']
# mask legend and inset drawing (they are long components too)
def pbox(X0, X1, Y0, Y1):
    a = topix(X0, Y0); b = topix(X1, Y1); K[int(a[1]):int(b[1]), int(a[0]):int(b[0])] = False
pbox(9.0, 11.25, -38.5, -69)       # inset drawing
pbox(11.4, 12.4, -11, -18)         # legend
fb = (box[0] + 6, box[1] + 6, box[2] - 6, box[3] - 6)
def snap(p, r=30):
    p = (int(round(p[0])), int(round(p[1])))
    ys, xs = np.nonzero(K[p[1] - r:p[1] + r + 1, p[0] - r:p[0] + r + 1]); j = np.argmin((xs - r) ** 2 + (ys - r) ** 2)
    return (p[0] - r + xs[j], p[1] - r + ys[j])
def run(X, Y, dx, dy, wm=10, mo=2.0):
    s0 = snap(topix(X, Y)); p2 = topix(X + dx * 0.01, Y + dy * 0.01); d = np.array(p2) - np.array(topix(X, Y)); d /= np.hypot(*d)
    kw = dict(step=3, hw=7, maxgap=20, box=fb, width_max=wm, mask=None, quad=True, nfit=10, gap_slope=0.15, max_off=mo)
    f1 = follow(K, s0, d, **kw); f2 = follow(K, s0, -d, **kw)
    f = np.vstack([f2[::-1], f1[1:]]); return f[f[:, 2] > 0]
S = {'l=0': [((8.8, -17.8, 1, -2.5), (8.2, 9.7)), ((10.0, -23.26, 1, -4), (9.7, 10.3)), ((10.7, -28.15, 1, -4), (10.4, 11.0)), ((11.3, -27.11, 1, 2), (11.2, 11.6)), ((11.8, -25.5, 1, 2), (11.6, 12.4))],
     'l=b': [((8.8, -21.5, 1, -10), (8.2, 9.3)), ((10.0, -28.09, 1, 5), (9.6, 10.3)), ((10.7, -25.71, 1, -2), (10.4, 11.0)), ((11.3, -31.64, 1, -25), (11.2, 11.5)), ((12.1, -28, 1, 5), (11.85, 12.4))]}
fin = {}
for k, ss in S.items():
    TT = []
    for (X, Y, dx, dy), (lo, hi) in ss:
        f = run(X, Y, dx, dy)
        t = trace2(K, f[:, :2], 6, binlen=3.0); t = clean_outliers(t, win=30, tol=2.0)
        xs = tox(t[:, 0]); t = t[(xs >= lo) & (xs <= hi)]
        TT.append(t); print(k, (X, Y), 'x %.2f..%.2f' % (tox(t[:, 0]).min(), tox(t[:, 0]).max()), 'half width %.2f' % np.median(t[:, 3]))
    T = np.vstack(TT); fin[k] = T[np.argsort(T[:, 0])]
overlay(img, [fin[k][:, :2] for k in fin], ['red', 'blue'], 'rt4_ov.png', crop=box, r=1)
pickle.dump(dict(fin=fin, cal=(mx, cx, my, cy, rx, ry, XT, XV, YT, YV)), open('rt4.pkl', 'wb'))
rows = []
for k, t in fin.items():
    rows += grid_rows(k + ' (theory and measured merged)', t, cX, cY, mx, -my, step=0.025, maxgap=0.06, cal_px=1.0)
rows.append(('l=b measured minimum', 11.726, -59.76, abs(my) * np.hypot(6.0, 1.0), mx * 2.0))
rows.sort(key=lambda r: (r[0], r[1]))
P = np.array([[x, y] for x in XT for y in YT[:1]])
H = [
 "source: " + REF,
 "figure: Fig. 4, p. 799 (PDF p. 4): \"Calculated and measured return loss of a cascaded structure of mitered 90 H-plane and E-plane bends (waveguide X-band, WR90-housing: 22.86 x 10.16 mm). Optimum miter dimensions in (mm): H-plane a_opt = 0.976a, E-plane b_opt = 0.874b.\"",
 "curves: l=0 and l=b, each drawn as theory (dashed) and measured (solid). Where the two coincide within the line width they are digitized together as one band (label '(theory and measured merged)'); the band centre is given and its half width is in the uncertainty. Not digitized: the two reflection minima of l=b near 9.40 GHz and 11.72 GHz, where theory and measurement separate (theory about -43 to -44 dB near 9.40 GHz, its tip falls in a dash gap below -43.3 dB; measured about -40 dB; near 11.72 GHz theory about -54 dB, measured about -60 dB) and the crossings of the two curves near 10.35 and 11.1 GHz. For l=b above 12.1 GHz only the measured (solid) line is traced; the theory line lies up to about 2 dB higher there. One extra row gives the measured l=b minimum near 11.73 GHz (lowest ink run of the solid line)",
 "geometry: a mitered 90 deg H-plane bend followed by a mitered 90 deg E-plane bend (inset), connected by a straight section of length l (l = 0 and l = b), in WR90 (a = 22.86 mm, b = 10.16 mm); optimum miters a_opt = 0.976 a (H-plane) and b_opt = 0.874 b (E-plane), miter dimension defined as in Fig. 3 (distance from the inner corner to the miter wall)",
 "x: frequency in GHz (axis 8.2..12.4; ticks at 8.2, 8.5, 9.0, ..., 12.0, 12.4)",
 "y: |S11| in dB (axis label |S11|/dB, axis -10..-70 dB, ticks every 10 dB)",
 "uncertainty: reading uncertainty of y (dB) at the given x: half the printed band width (perpendicular, converted to vertical with the local slope; for the merged rows this covers both theory and measurement) and 1 px calibration error in both axes",
 "x_uncertainty: 0 for the gridded rows; 2 px for the minimum row",
 "sampling: x on a 0.025 GHz grid where the band is traced (no interpolation across gaps > 0.06 GHz)",
 "method: native 1-bit scan (339 dpi) extracted with pdfimages (rotation below 0.1 deg, neglected); curves tracked as long connected components with a curvature-aware follower started from several seeds, each run kept only inside a frequency window between crossings, measured by ink centroids in 3 px bins; inset and legend masked; scripts in benchmarks/literature/raw/a/scripts/ (L1a; they read page images extracted from the reference PDFs, which are not in the repository)",
 "calibration points (page pixels -> value): x ticks px " + ", ".join("%.1f->%g" % (a, b) for a, b in zip(XT, XV)) + "; y ticks px " + ", ".join("%.1f->%g" % (a, b) for a, b in zip(YT, YV)),
 "calibration fit: x = %.7g*px + %.6g (max residual %.4f GHz), y = %.6g*py + %.6g (max residual %.3f dB)" % (mx, cx, abs(rx).max(), my, cy, abs(ry).max()),
 "axis resolution: ticks 0.5 GHz every %.1f px (%.4f GHz) and 10 dB every %.1f px (%.3f dB) (half tick spacing divided by the pixels between ticks)" % (0.5 / mx, mx / 2, 10 / abs(my), abs(my) / 2),
 "digitizer: L1a (independent of L1b), 2026-11-08",
]
write_csv('/home/claude/wt/L1a-scratch/out/reiter1995_fig4.csv', H, rows)
print('rows', len(rows))
