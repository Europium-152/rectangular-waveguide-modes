import sys; sys.path.insert(0, '/home/claude/wt/L1a-scratch')
from dig import *
from scipy import ndimage
I = np.array(Image.open('dn_f9.png')) < 120
OFF = (880, 1800)  # crop offset in the PDF page image (pdfimages page 7, 2263x3157)

def frame_v(x0, x1, ys, pick='longest'):
    out = []
    for y in ys:
        r = runs(np.nonzero(I[y, x0:x1])[0] + x0)
        r = [t for t in r if t[1] - t[0] <= 5]
        if r:
            b = r[-1] if pick == 'last' else r[0]
            out.append((y, (b[0] + b[1]) / 2))
    return np.array(out)
def frame_h(y0, y1, xs, pick='last'):
    out = []
    for x in xs:
        r = runs(np.nonzero(I[y0:y1, x])[0] + y0)
        r = [t for t in r if t[1] - t[0] <= 5]
        if r:
            b = r[-1] if pick == 'last' else r[0]
            out.append((x, (b[0] + b[1]) / 2))
    return np.array(out)
def robust_poly(a, b, deg=3, tol=1.5):
    k = np.ones(len(a), bool)
    for _ in range(5):
        p = np.polyfit(a[k], b[k], deg); r = b - np.polyval(p, a)
        k = np.abs(r) < tol
    return p, np.abs(r[k]).max(), k.sum(), len(a)
# left frame: first thin run in cols 80..100; right frame: last thin run in 1205..1240
Lp = frame_v(80, 100, range(275, 990)); pL = robust_poly(Lp[:, 0], Lp[:, 1])
Rp = frame_v(1210, 1240, range(285, 1025), pick='last'); pR = robust_poly(Rp[:, 0], Rp[:, 1])
Tp = frame_h(255, 285, range(100, 1210), pick='first'); pT = robust_poly(Tp[:, 0], Tp[:, 1])
Bp = frame_h(990, 1045, range(100, 1220), pick='last'); pB = robust_poly(Bp[:, 0], Bp[:, 1])
for n, p in [('left x(y)', pL), ('right x(y)', pR), ('top y(x)', pT), ('bottom y(x)', pB)]:
    print(n, p[0], 'maxres %.2f kept %d/%d' % p[1:])
P = lambda p, t: np.polyval(p[0], t)
def ticks_h_frame(pline, xs, d0, d1, minrows):
    """vertical ticks attached to a horizontal frame line y = pline(x); band rows from y+d0 to y+d1."""
    c = []
    for x in xs:
        y = P(pline, x); a, b = sorted((int(round(y + d0)), int(round(y + d1))))
        c.append(I[a:b + 1, x].sum())
    c = np.array(c); idx = [xs[i] for i in range(len(xs)) if c[i] >= minrows]
    out = []
    for s, e in runs(idx):
        w = c[s - xs[0]:e - xs[0] + 1].astype(float); i = np.arange(s, e + 1)
        out.append((w * i).sum() / w.sum())
    return out
def ticks_v_frame(pline, ys, d0, d1, mincols):
    c = []
    for y in ys:
        x = P(pline, y); a, b = sorted((int(round(x + d0)), int(round(x + d1))))
        c.append(I[y, a:b + 1].sum())
    c = np.array(c); idx = [ys[i] for i in range(len(ys)) if c[i] >= mincols]
    out = []
    for s, e in runs(idx):
        w = c[s - ys[0]:e - ys[0] + 1].astype(float); i = np.arange(s, e + 1)
        out.append((w * i).sum() / w.sum())
    return out
tt = ticks_h_frame(pT, list(range(110, 1200)), 5, 20, 12)
bt = ticks_h_frame(pB, list(range(110, 1210)), -14, -5, 9)
lt = ticks_v_frame(pL, list(range(285, 990)), 5, 28, 18)
rt = ticks_v_frame(pR, list(range(290, 1020)), -28, -5, 18)
print('top', np.round(tt, 2)); print('bottom', np.round(bt, 2)); print('left', np.round(lt, 2)); print('right', np.round(rt, 2))
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares
def corner(pv, ph):
    # x = pv(y), y = ph(x)
    f = lambda y: y - P(ph, P(pv, y))
    y = brentq(f, 200, 1100); return np.array([P(pv, y), y])
C00 = corner(pL, pB); C10 = corner(pR, pB); C01 = corner(pL, pT); C11 = corner(pR, pT)
print('corners', C00, C10, C01, C11)
XK = np.array([15, 15.5, 16, 16.5, 17, 17.5, 18.0])
YK = np.array([0, 0.5, 1.0, 1.5, 2.0])
bx = np.array([C00[0]] + list(bt) + [C10[0]]); tx = np.array([C01[0]] + list(tt) + [C11[0]])
ly = np.array([C00[1], 817.0, 630.5, 438.5, C01[1]]); ry = np.array([C10[1], 837.0, 644.0, 448.5, C11[1]])
LY = dict(zip(YK, ly))
SB = CubicSpline(XK, bx); ST = CubicSpline(XK, tx); SL = CubicSpline(YK, ly); SR = CubicSpline(YK, ry)
def Bc(X): x = SB(X); return np.array([x, P(pB, x)])
def Tc(X): x = ST(X); return np.array([x, P(pT, x)])
def Lc(Y): y = SL(Y); return np.array([P(pL, y), y])
def Rc(Y): y = SR(Y); return np.array([P(pR, y), y])
def coons(X, Y):
    u = (X - 15) / 3; v = Y / 2
    return ((1 - v) * Bc(X) + v * Tc(X) + (1 - u) * Lc(Y) + u * Rc(Y)
            - ((1 - u) * (1 - v) * C00 + u * (1 - v) * C10 + (1 - u) * v * C01 + u * v * C11))
def inv(px, py, guess=(16.5, 1.0)):
    r = least_squares(lambda q: coons(q[0], q[1]) - np.array([px, py]), guess, xtol=1e-12, ftol=1e-12)
    return r.x
# homography from all calibration points, for comparison
src = []; dst = []
for X, x in zip(XK, bx): src.append((x, P(pB, x))); dst.append((X, 0))
for X, x in zip(XK, tx): src.append((x, P(pT, x))); dst.append((X, 2))
for Y, y in zip(YK[1:-1], ly[1:-1]): src.append((P(pL, y), y)); dst.append((15, Y))
for Y, y in zip(YK[1:-1], ry[1:-1]): src.append((P(pR, y), y)); dst.append((18, Y))
src = np.array(src); dst = np.array(dst)
def fit_h(src, dst):
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X]); A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])
    _, _, Vt = np.linalg.svd(np.array(A)); return Vt[-1].reshape(3, 3)
Hm = fit_h(src, dst)
def hmap(x, y):
    v = Hm @ np.array([x, y, 1.0]); return v[:2] / v[2]
res = np.array([hmap(*s) - d for s, d in zip(src, dst)])
print('homography residuals on calibration points: max |dX| %.4f GHz, max |dY| %.4f %%' % (abs(res[:, 0]).max(), abs(res[:, 1]).max()))
# interior comparison
dif = []
for X in np.linspace(15.2, 17.8, 14):
    for Y in np.linspace(0.1, 1.9, 10):
        p = coons(X, Y); dif.append(hmap(*p) - (X, Y))
dif = np.array(dif); print('coons vs homography interior: max dX %.4f, max dY %.4f, rms dX %.4f dY %.4f' % (abs(dif[:, 0]).max(), abs(dif[:, 1]).max(), np.sqrt((dif[:, 0]**2).mean()), np.sqrt((dif[:, 1]**2).mean())))
# check coons reproduces calibration points
cr = np.array([inv(*s, guess=d) - d for s, d in zip(src, dst)]); print('coons calib residual', abs(cr).max())
