exec(open('/home/claude/wt/L1a-scratch/doane2.py').read().split("SEED = {")[0])
import pickle
from csvout import write_csv
exec(open('/home/claude/wt/L1a-scratch/doane_calB.py').read())
F = pickle.load(open('dn_follow.pkl', 'rb'))
paths = {}
for k, fl in F.items():
    pts = np.vstack([f[f[:, 2] > 0][:, :2] for f in fl])
    pts = pts[np.argsort(pts[:, 0])]
    # dedupe: average points within 1.5 px in x
    q = []; cur = [pts[0]]
    for p in pts[1:]:
        if p[0] - cur[-1][0] < 1.5: cur.append(p)
        else: q.append(np.mean(cur, 0)); cur = [p]
    q.append(np.mean(cur, 0)); paths[k] = np.array(q)
def near_mask(P, r):
    m = np.zeros_like(I)
    for x, y in P:
        m[int(y) - r:int(y) + r + 1, int(x) - r:int(x) + r + 1] = True
    return m
fin = {}
for k in paths:
    m = M | near_mask(np.vstack([paths[j] for j in paths if j != k]), 6)
    t = trace2(I, paths[k], 5, mask=m, binlen=3.0)
    t = clean_outliers(t, win=30, tol=1.5)
    fin[k] = t
    print(k, len(t), 'median half width %.2f' % np.median(t[:, 3]))
overlay(img, [fin[k][:, :2] for k in fin], ['red', 'blue', 'green'], 'dn_fin.png', r=1)
overlay(img, [fin[k][:, :2] for k in fin], ['red', 'blue', 'green'], 'dn_z2.png', crop=(80, 860, 1240, 1045), scale=1, r=1)
# convert
def jac(X, Y, h=1e-4):
    J = np.zeros((2, 2))
    J[:, 0] = (coons(X + h, Y) - coons(X - h, Y)) / (2 * h); J[:, 1] = (coons(X, Y + h) - coons(X, Y - h)) / (2 * h)
    return J  # d(px,py)/d(X,Y)
rows = []
maxdB = 0.0
lab = {'constant': 'constant', 'cosine': 'cosine', 'triangular': 'triangular'}
for k, t in fin.items():
    D = []; g = (16.5, 1.0)
    for x, y, n, hw in t:
        X, Y = inv(x, y, guess=g); g = (X, Y)
        Jm = jac(X, Y); Ji = np.linalg.inv(Jm)   # d(X,Y)/d(px,py)
        dH = hmap(x, y) - (X, Y)
        XB, YB = invB(x, y, guess=(X, Y))
        D.append((X, Y, hw, abs(Ji[0, 0]), abs(Ji[1, 1]), dH[0], dH[1], YB - Y))
    D = np.array(D); D = D[np.argsort(D[:, 0])]
    grid = np.round(np.arange(15.0, 18.0001, 0.025), 4)
    for gx in grid:
        j = np.searchsorted(D[:, 0], gx)
        if j == 0 or j == len(D): continue
        a, b = D[j - 1], D[j]
        if b[0] - a[0] > 0.03: continue
        tt_ = (gx - a[0]) / (b[0] - a[0]); Y = a[1] + tt_ * (b[1] - a[1])
        lo = max(0, j - 4); hi = min(len(D), j + 3)
        sl = np.polyfit(D[lo:hi, 0], D[lo:hi, 1], 1)[0]   # dY/dX
        dXdpx, dYdpy = a[3], a[4]
        s_px = sl * (1 / dYdpy) / (1 / dXdpx)  # slope in px units
        hw = 0.5 * (a[2] + b[2])
        u_line = hw * np.sqrt(1 + s_px ** 2) * dYdpy
        ux_cal = np.hypot(0.5 * a[5], 1.0 * dXdpx); uy_cal = np.hypot(0.5 * a[6], 1.0 * dYdpy)
        dB = 0.5 * (a[7] + b[7]); maxdB = max(maxdB, abs(dB))
        u = np.sqrt(u_line ** 2 + uy_cal ** 2 + (sl * ux_cal) ** 2 + dB ** 2)
        rows.append((lab[k], gx, Y, u, 0.0))
order = {'constant': 0, 'cosine': 1, 'triangular': 2}
rows.sort(key=lambda r: (order[r[0]], r[1]))
pickle.dump(rows, open('dn_rows.pkl', 'wb'))
pts_str = "; ".join(["bottom frame (Y=0) at X=" + ", ".join("%.1f:(%.1f,%.1f)" % (X, x + OFF[0], P(pB, x) + OFF[1]) for X, x in zip(XK, bx)),
                     "top frame (Y=2) at X=" + ", ".join("%.1f:(%.1f,%.1f)" % (X, x + OFF[0], P(pT, x) + OFF[1]) for X, x in zip(XK, tx)),
                     "left frame (X=15) at Y=" + ", ".join("%g:(%.1f,%.1f)" % (Y, P(pL, y) + OFF[0], y + OFF[1]) for Y, y in zip(YK, ly)),
                     "right frame (X=18) at Y=" + ", ".join("%g:(%.1f,%.1f)" % (Y, P(pR, y) + OFF[0], y + OFF[1]) for Y, y in zip(YK, ry))])
H = [
 "source: J. L. Doane and T. N. Anderson, \"Oversized rectangular waveguides with mode-free bends and twists for broadband applications\", Microwave Journal 32(3), 153-160 (March 1989)",
 "figure: Fig. 9, p. 159 (PDF p. 7): \"Mode conversion in 20\" 90 deg twists in WR187 tall guide (1.872\" x 0.872\" ID) for various rates of twist variations.\"",
 "curves: constant (dashed line), cosine (solid line), triangular (solid line, nearly flat near 0.05-0.08 %); labels and arrows of the figure",
 "geometry: 90 deg twists, length 20 in (0.508 m), WR187 'tall' guide, inner dimensions 1.872 in x 0.872 in (47.55 mm x 22.15 mm); twist rate d(theta)/dz constant, cosine or triangular (linearly tapered, eq. 17) along the twist",
 "x: frequency in GHz (axis 15..18)",
 "y: total spurious mode power in percent of the input power (axis label \"TOTAL SPURIOUS MODE POWER (%)\", axis 0..2)",
 "uncertainty: reading uncertainty of y (%) at the given x: half the printed line width (perpendicular, converted to vertical with the local slope), and the calibration uncertainty (1 px in each axis combined with half the difference between the Coons-patch and a least-squares homography calibration at that point), the x part converted with the local slope",
 "y-scale anomaly: the side ticks 0, 0.5, 1, 1.5 are evenly spaced (about 190 px per 0.5 %), but the top frame, labelled 2, lies only about 174 px above the 1.5 ticks on both sides; with a tick-only scale the top frame would be at 1.94-1.96 %. The values use the frame as 2 % (calibration A); the difference to the tick-only calibration (B, linear extrapolation above 1.5) is added to uncertainty (at most MAXDB %, only above 1.5 % is it larger than 0.005 %)",
 "x_uncertainty: 0 (x is on a regular grid; the effect of x reading errors is included in uncertainty)",
 "sampling: x = 15.000:0.025:18.000 GHz where ink exists; no interpolation across gaps > 0.03 GHz; regions within 6 px of a crossing with another curve, arrow tips and the frame are left out",
 "method: the PDF page is a phone photograph (JPEG, 2263x3157 px) of the printed page, with perspective and page-curl distortion. Ink threshold 120/255. Frame lines fitted as cubic polynomials; calibration by a Coons patch through the four frame curves, parameterized by the tick marks (cubic splines through the ticks); curves tracked by a curvature-aware follower and measured by ink centroids in 3 px bins; scripts in benchmarks/literature/raw/a/scripts/ (L1a; they read page images extracted from the reference PDFs, which are not in the repository)",
 "calibration points (page-image pixels (x,y) of the frame/tick intersections): " + pts_str,
 "calibration check: a single homography fitted to these 22 points leaves residuals up to %.4f GHz and %.4f %% (page curl); in the plot interior the Coons patch and the homography differ by at most %.4f GHz and %.4f %% (rms %.4f GHz, %.4f %%)" % (abs(res[:, 0]).max(), abs(res[:, 1]).max(), abs(dif[:, 0]).max(), abs(dif[:, 1]).max(), np.sqrt((dif[:, 0]**2).mean()), np.sqrt((dif[:, 1]**2).mean())),
 "axis resolution: ticks every 0.5 GHz (about 188 px) and every 0.5 %% (about 188 px), no minor ticks: 0.5*0.5/188 = 0.0013 GHz and 0.0013 %%",
 "digitizer: L1a (independent of L1b), 2026-11-08",
]
H = [h.replace('%%', '%') if ('axis resolution' in h) else h for h in H]
H = [h.replace('MAXDB', '%.3f' % maxdB) for h in H]
write_csv('/home/claude/wt/L1a-scratch/out/doane1989_fig9.csv', H, rows)
print(len(rows))
for r in rows:
    if abs(r[1] * 4 - round(r[1] * 4)) < 1e-6: print('%s %.2f %.4f %.4f' % (r[0], r[1], r[2], r[3]))
