exec(open('/home/claude/wt/L1a-scratch/doane.py').read())
import pickle
img = Image.open('dn_f9.png').convert('L')
H_, W_ = I.shape
M = np.zeros_like(I)
yy, xx = np.mgrid[0:H_, 0:W_]
def mline(a, b, w):
    a = np.array(a, float); b = np.array(b, float); d = b - a; L2 = d @ d
    t = np.clip(((xx - a[0]) * d[0] + (yy - a[1]) * d[1]) / L2, 0, 1)
    dist = np.hypot(xx - a[0] - t * d[0], yy - a[1] - t * d[1])
    M[dist <= w] = True
def mbox(x0, y0, x1, y1): M[y0:y1, x0:x1] = True
# frame lines and ticks
for x in range(W_):
    for pl in (pT, pB):
        y = P(pl, x); M[max(0, int(y) - 4):int(y) + 5, x] = True
for y in range(H_):
    for pl in (pL, pR):
        x = P(pl, y); M[y, max(0, int(x) - 4):int(x) + 5] = True
for x in list(tt) + list(bt):
    M[:, int(x) - 3:int(x) + 4] &= True
    yb = P(pB, x); yt_ = P(pT, x)
    M[int(yt_):int(yt_) + 24, int(x) - 3:int(x) + 4] = True
    M[int(yb) - 24:int(yb) + 1, int(x) - 3:int(x) + 4] = True
for y in (438.5, 630.5, 817.0):
    x = P(pL, y); M[int(y) - 3:int(y) + 4, int(x):int(x) + 36] = True
for y in (448.5, 644.0, 837.0):
    x = P(pR, y); M[int(y) - 3:int(y) + 4, int(x) - 36:int(x)] = True
# labels and arrows
mbox(200, 478, 365, 512); mline((272, 527), (192, 608), 6); M[(xx - 183) ** 2 + (yy - 617) ** 2 < 14 ** 2] = True
mbox(655, 522, 775, 555); mline((676, 568), (652, 660), 6); M[(xx - 650) ** 2 + (yy - 668) ** 2 < 12 ** 2] = True
mbox(990, 860, 1190, 898); mline((990, 873), (948, 873), 5); mline((948, 873), (930, 955), 6); M[(xx - 927) ** 2 + (yy - 962) ** 2 < 12 ** 2] = True
M[:, :int(C00[0]) + 5] = True; M[:, int(C10[0]) - 3:] = True
pickle.dump(M, open('dn_mask.pkl', 'wb'))
box = (95, 270, 1222, 1030)
SEED = {'cosine': [((450, 740), (1, -0.25))], 'constant': [((400, 900), (1, 0.9)), ((620, 993), (1, 0)), ((900, 800), (1, -1.0))],
        'triangular': [((300, 982), (1, -0.01))]}
F = {}
for k, seeds in SEED.items():
    allp = []
    for s, d in seeds:
        r = 20; sub = I[s[1] - r:s[1] + r + 1, s[0] - r:s[0] + r + 1] & ~M[s[1] - r:s[1] + r + 1, s[0] - r:s[0] + r + 1]
        ys, xs = np.nonzero(sub); j = np.argmin((xs - r) ** 2 + (ys - r) ** 2); st = (s[0] - r + xs[j], s[1] - r + ys[j])
        d = np.array(d, float); d /= np.hypot(*d)
        kw = dict(step=3, hw=8, maxgap=40, box=box, width_max=6, mask=M, quad=(k == 'constant'), nfit=12, gap_slope=0.15, max_off=1.2)
        f1 = follow(I, st, d, **kw); f2 = follow(I, st, -d, **kw)
        f = np.vstack([f2[::-1], f1[1:]]); allp.append(f)
        print(k, s, 'x %.0f..%.0f' % (f[:, 0].min(), f[:, 0].max()))
    F[k] = allp
overlay(img, [f[:, :2] for k in F for f in F[k]], [c for k, c in zip(F, ['red', 'blue', 'green']) for f in F[k]], 'dn_follow.png', r=1)
pickle.dump(F, open('dn_follow.pkl', 'wb'))
