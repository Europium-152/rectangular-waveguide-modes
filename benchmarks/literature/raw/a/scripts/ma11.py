exec(open('/home/claude/wt/L1a-scratch/ma_common.py').read())
from ma_out import ma_header
img = load_gray('ma/p-005.png'); I = ink(img)
XT = list((np.array([1521.5, 1674.9, 1827.9, 1981.8, 2135.7, 2287.6]) + np.array([1520.7, 1673.5, 1826.5, 1980.0, 2134.0, 2286.6])) / 2)
XV = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
YT = list((np.array([184.7, 293.1, 402.5, 511.4, 618.5, 729.0]) + np.array([184.9, 293.9, 402.9, 511.5, 619.5, 729.6])) / 2)
YV = [0, -10, -20, -30, -40, -50]
mx, cx, rx = linfit(XT, XV); my, cy, ry = linfit(YT, YV)
print('cal resid max', abs(rx).max(), abs(ry).max(), 'per px', mx, my)
tox = lambda p: mx * np.asarray(p) + cx; toy = lambda p: my * np.asarray(p) + cy
topx = lambda X: (X - cx) / mx; topy = lambda Y: (Y - cy) / my
c2 = lambda X, Y: (float(topx(X)), float(topy(Y)))
M = np.zeros_like(I)
M[:, :1527] = True; M[:, 2283:] = True; M[:188, :] = True; M[724:, :] = True
def cbox(u0, v0, u1, v1): M[v0 + 100:v1 + 100, u0 + 1300:u1 + 1300] = True   # crop offset (1300,100)
cbox(385, 182, 485, 210)   # c/a=0
cbox(575, 197, 705, 230)   # c/a=1.0
cbox(778, 147, 910, 180)   # c/a=1.30
cbox(553, 268, 685, 298)   # c/a=0.60
cbox(330, 490, 470, 522)   # c/a=0.70
cbox(575, 540, 712, 572)   # c/a=0.631
# the horizontal dotted -30 dB reference line: remove its dots (components within +-3 px of the line)
yref = topy(-30)
lab, n = ndimage.label(I, structure=np.ones((3, 3)))
REF = np.zeros_like(I)
for i, s in enumerate(ndimage.find_objects(lab)):
    if s[0].start >= yref - 4 and s[0].stop - 1 <= yref + 4 and (s[1].stop - s[1].start) < 8:
        REF[s] |= lab[s] == i + 1
print('ref line row', yref, 'dots removed', REF.sum())
K = I & ~REF
pickle.dump(dict(cal=(mx, cx, my, cy, rx, ry, XT, XV, YT, YV)), open('ma11_cal.pkl', 'wb'))
box = (1527, 186, 2283, 726)
def snap(p, r=18, mask=None):
    p = (int(round(p[0])), int(round(p[1])))
    sub = K[p[1] - r:p[1] + r + 1, p[0] - r:p[0] + r + 1].copy()
    if mask is not None: sub &= ~mask[p[1] - r:p[1] + r + 1, p[0] - r:p[0] + r + 1]
    ys, xs = np.nonzero(sub); j = np.argmin((xs - r) ** 2 + (ys - r) ** 2)
    return (p[0] - r + xs[j], p[1] - r + ys[j])
def run(seedXY, dXY, wm=6, maxgap=30, gs=0.2, mo=1.5, mask=None, hw=7, quad=True):
    s0 = snap(c2(*seedXY), mask=M if mask is None else mask)
    d = np.array([dXY[0] / mx, dXY[1] / my]); d /= np.hypot(*d)
    kw = dict(step=3, hw=hw, maxgap=maxgap, box=box, width_max=wm, mask=M if mask is None else mask, quad=quad, nfit=10, gap_slope=gs, max_off=mo)
    f1 = follow(K, s0, d, **kw); f2 = follow(K, s0, -d, **kw)
    return np.vstack([f2[::-1], f1[1:]])
