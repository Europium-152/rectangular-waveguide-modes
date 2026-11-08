import sys; sys.path.insert(0, '/home/claude/wt/L1a-scratch')
from dig import *
from csvout import write_csv
from scipy import ndimage
import pickle

img = load_gray('al_p2_deskew5.png'); J = ink(img)
XT = [1683.94, 1841.07, 1996.22, 2151.96, 2306.97, 2462.14]; XV = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
YT = [1246.5, 1356.5, 1466.53, 1577.0, 1687.46, 1796.5]; YV = [0, -10, -20, -30, -40, -50]
mx, cx, rx = linfit(XT, XV); my, cy, ry = linfit(YT, YV)
print('resid', rx, ry, mx, my)
tox = lambda px: mx * np.asarray(px) + cx; toy = lambda py: my * np.asarray(py) + cy
topx = lambda X: (X - cx) / mx; topy = lambda Y: (Y - cy) / my
c2 = lambda X, Y: (float(topx(X)), float(topy(Y)))
K = J.copy()
other = np.zeros_like(K)
def box(X0, X1, Y0, Y1):
    other[int(topy(Y0)):int(topy(Y1)) + 1, int(topx(X0)):int(topx(X1)) + 1] = True
# crop->data helper used to define masks: crop offset (1550,1180) of the undeskewed page; shifts are < 5 px
cd = lambda u, v: (1 + (u + 1550 - 1684) / 155.6 * 0.2, -(v + 1180 - 1246.5) / 11.0)
for (u0, v0, u1, v1) in [(250, 80, 455, 262),   # inset
                         (548, 240, 705, 280),  # d/b = 0.9
                         (275, 430, 435, 470),  # d/b = 0.85
                         (295, 525, 455, 562),  # d/b = 0.83
                         (824, 480, 975, 520)]: # d/b = 0.87
    X0, Y0 = cd(u0, v0); X1, Y1 = cd(u1, v1); box(X0, X1, Y0, Y1)
other[:, :int(topx(1.0)) + 14] = True     # axis and its ticks
other[int(topy(-50)) - 5:, :] = True

G = {
 '0.9':   [(1.018,-33.9),(1.033,-31.6),(1.085,-27.5),(1.213,-23),(1.342,-21),(1.47,-20.3),(1.6,-20.1),(1.727,-20.5),(1.856,-22.1),(1.93,-24.4),(1.98,-27.1)],
 '0.87a': [(1.02,-37.4),(1.085,-30.3),(1.21,-26.6),(1.34,-25.7),(1.47,-26.2),(1.6,-28.9),(1.727,-34.8),(1.79,-40.3),(1.824,-48.5),(1.83,-50)],
 '0.87b': [(1.85,-49.5),(1.87,-41.2),(1.91,-34),(1.96,-28.5),(2.0,-25.4)],
 '0.85a': [(1.02,-41.2),(1.06,-36.6),(1.15,-32.6),(1.28,-31.4),(1.41,-32.1),(1.5,-35.3),(1.56,-40.3),(1.6,-44.9),(1.62,-49)],
 '0.85b': [(1.67,-49),(1.69,-43),(1.73,-37.5),(1.79,-32.1),(1.86,-28.4),(1.92,-23.9),(2.0,-18.6)],
 '0.83a': [(1.018,-48.2),(1.046,-45.5),(1.12,-42.7),(1.19,-43.2),(1.25,-45.9),(1.29,-48.7)],
 '0.83b': [(1.39,-48.6),(1.45,-41.8),(1.54,-33.6),(1.67,-25.9),(1.79,-21),(1.92,-16.9),(2.0,-14.6)],
}
p1 = {}
for k, g in G.items():
    t = trace2(K, [c2(*q) for q in g], 16, mask=other)
    p1[k] = clean_outliers(t, win=30, tol=2.5)
    print(k, len(t), len(p1[k]))
overlay(img, [p1[k][:, :2] for k in G], ['red', 'orange', 'orange', 'blue', 'blue', 'green', 'green'], 'f5_p1.png', crop=(1600, 1180, 2560, 1880), r=1)
pickle.dump(dict(p1=p1, cal=(mx, cx, my, cy, rx, ry, XT, XV, YT, YV)), open('f5_p1.pkl', 'wb'))
