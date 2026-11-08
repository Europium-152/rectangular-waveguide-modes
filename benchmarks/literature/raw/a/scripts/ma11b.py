exec(open('/home/claude/wt/L1a-scratch/ma11.py').read())
S = {
 '0.60': [((1.25, -15.8), (1, -28), dict(wm=6, maxgap=20)), ((1.9, -24), (1, 150), dict(wm=6, maxgap=20))],
 '0.631': [((1.249, -17.2), (1, -30), dict(wm=6, maxgap=30)), ((1.76, -38), (1, 70), dict(wm=6, maxgap=30)),
           ((1.91, -36), (1, -400), dict(wm=6, maxgap=30)), ((1.94, -38), (1, 600), dict(wm=6, maxgap=30)), ((1.956, -25.5), (0.002, 4), dict(wm=6, maxgap=10))],
 '0.70': [((1.136, -19.8), (1, -63), dict(wm=6, maxgap=25)), ((1.6, -24), (1, 26), dict(wm=6, maxgap=25, mo=1.0, gs=0.1)),
          ((1.30, -35), (1, -120), dict(wm=6, maxgap=30)), ((1.22, -26), (1, -90), dict(wm=6, maxgap=30)), ((1.75, -20.5), (1, 25), dict(wm=6, maxgap=30, mo=1.0, gs=0.1)), ((1.857, -17.52), (1, 20), dict(wm=6, maxgap=30, mo=1.0, gs=0.1))],
 '0': [((1.166, -6.9), (1, -18), dict(wm=6, maxgap=45)), ((1.95, -0.5), (1, -3), dict(wm=6, maxgap=45, mo=1.0))],
 '1.0': [((1.253, -4.2), (1, -15), dict(wm=6, maxgap=30, gs=0.3))],
 '1.30': [((1.86, -4.2), (1, -29), dict(wm=6, maxgap=40)), ((1.6, -0.9), (1, -3), dict(wm=6, maxgap=40, mo=1.0))],
}
F = {}
for k, ss in S.items():
    F[k] = []
    for xy, d, kw in ss:
        f = run(xy, d, **kw); F[k].append(f)
        print(k, xy, 'x %.3f..%.3f  y %.1f..%.1f' % (tox(f[:, 0].min()), tox(f[:, 0].max()), toy(f[:, 1]).max(), toy(f[:, 1]).min()), len(f), int(f[:, 2].sum()))
cols = {'0.60': 'red', '0.631': 'blue', '0.70': 'green', '0': 'orange', '1.0': 'magenta', '1.30': 'cyan'}
overlay(img, [f[:, :2] for k in F for f in F[k]], [cols[k] for k in F for f in F[k]], 'ma11_ov.png', crop=(1515, 180, 2295, 735), scale=1.5, r=1)
pickle.dump(F, open('ma11_F.pkl', 'wb'))
