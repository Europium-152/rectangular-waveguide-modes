import sys; sys.path.insert(0, '/home/claude/wt/L1a-scratch')
exec(open('fig5.py').read().split("p1 = {}")[0])
box = (topx(1.0) + 10, topy(0), topx(2.0) + 30, topy(-50) - 4)
W = {'0.85a': 10, '0.85b': 10}
GS = {'0.85a': 0.4, '0.85b': 0.4}
SEED = {'0.9': (1.47, -20.3, 1, 0), '0.87a': (1.34, -25.7, 1, 0), '0.87b': (1.91, -34, 0.03, 4),
        '0.85a': (1.28, -31.4, 1, 0), '0.85b': (1.79, -32.1, 0.06, 5.4), '0.83a': (1.13, -43.2, 1, 0),
        '0.83b': (1.67, -25.9, 0.13, 7.7)}
def snap(s):
    r = 25
    a = K[int(s[1]) - r:int(s[1]) + r + 1, int(s[0]) - r:int(s[0]) + r + 1] & ~other[int(s[1]) - r:int(s[1]) + r + 1, int(s[0]) - r:int(s[0]) + r + 1]
    ys, xs = np.nonzero(a); j = np.argmin((xs - r) ** 2 + (ys - r) ** 2)
    return (s[0] - r + xs[j], s[1] - r + ys[j])
def follow_both(k, **kw):
    X, Y, dx, dy = SEED[k]
    st = snap(c2(X, Y))
    d = np.array([dx, -dy * 0 + dy], float)  # data-space direction; pixel y is flipped
    dpx = np.array([dx / mx * mx, dy * mx / my])  # convert data slope to px: dy_px/dx_px = (dy/my)/(dx/mx)
    dpx = np.array([dx / mx, dy / my]); dpx /= np.hypot(*dpx)
    f1 = follow(K, st, dpx, **kw); f2 = follow(K, st, -dpx, **kw)
    f = np.vstack([f2[::-1], f1[1:]])
    return f[np.argsort(f[:, 0])] if True else f
F = {}
for k in SEED:
    F[k] = follow_both(k, step=3, hw=8, maxgap=36, box=box, width_max=W.get(k, 6), mask=other, quad=True, nfit=12, gap_slope=GS.get(k, 0.2))
    f = F[k]
    print(k, 'x range %.4f..%.4f' % (tox(f[:, 0]).min(), tox(f[:, 0]).max()), 'y range %.2f..%.2f' % (toy(f[:, 1]).min(), toy(f[:, 1]).max()), len(f), int(f[:, 2].sum()))
cols = {'0.9': 'red', '0.87a': 'orange', '0.87b': 'orange', '0.85a': 'blue', '0.85b': 'blue', '0.83a': 'green', '0.83b': 'green'}
overlay(img, [F[k][:, :2] for k in cols], [cols[k] for k in cols], 'f5_follow.png', crop=(1600, 1180, 2560, 1880), r=1)
overlay(img, [F[k][:, :2] for k in cols], [cols[k] for k in cols], 'z6.png', crop=(1660, 1500, 2000, 1800), scale=2, r=1)
pickle.dump(F, open('f5_follow.pkl', 'wb'))
