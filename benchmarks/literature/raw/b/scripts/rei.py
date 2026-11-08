"""Reiter & Arndt 1995, Figs. 3(a)-(d) and 4 (journal p. 799 = PDF p. 4), digitized by L1b."""
import numpy as np, sys
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from dig import load
from gridcal import calibrate
from trace import assign_euclid
from pxcurve import remove_small, centre_px, to_data
im, d = load('../rei-4.png')

def Z(pts, off, s=2):
    return [(off[0] + x / s, off[1] + y / s) for x, y in pts]

def panel(cal_args, frame, guides, masks, tol=6, cross_r=5, refine=None):
    c = calibrate(d, *cal_args)
    x0, x1, y0, y1 = frame
    m = np.zeros_like(d); m[y0:y1, x0:x1] = d[y0:y1, x0:x1]
    for mk_ in masks:
        if mk_[0] == 'seg':
            _, p0, p1, w = mk_
            yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
            p0 = np.array(p0, float); p1 = np.array(p1, float); dv = p1 - p0
            tt = np.clip(((xx - p0[0]) * dv[0] + (yy - p0[1]) * dv[1]) / (dv @ dv), 0, 1)
            m &= ~(np.hypot(xx - p0[0] - tt * dv[0], yy - p0[1] - tt * dv[1]) <= w)
        else:
            a, b, e, f = mk_
            m[e:f, a:b] = False
    m = remove_small(m, 10)
    gpx = {k: (np.array([p[0] for p in v]), np.array([p[1] for p in v])) for k, v in guides.items()}
    for wide in (refine or ()):
        # refine the guides: centre line of the pixels within `wide` px of the current guide, ordered along the guide
        parts = assign_euclid(m, gpx, tol=wide)
        for k, mk in parts.items():
            cp = centre_px(mk, step=3)
            if len(cp) < 4:
                continue
            gx, gy = gpx[k]
            # order the centre points by their position along the old guide (nearest guide vertex index)
            dd = (cp[:, 0][:, None] - gx[None]) ** 2 + (cp[:, 1][:, None] - gy[None]) ** 2
            order = np.lexsort((cp[:, 1] if abs(gy[-1] - gy[0]) > abs(gx[-1] - gx[0]) else cp[:, 0], np.argmin(dd, 1)))
            gpx[k] = (cp[order, 0], cp[order, 1])
    parts = assign_euclid(m, gpx, tol=tol)
    trees = {k: cKDTree(np.c_[np.nonzero(v)[1], np.nonzero(v)[0]]) for k, v in parts.items() if v.any()}
    pts = {}
    for k, mk in parts.items():
        others = [t for kk, t in trees.items() if kk.split('#')[0] != k.split('#')[0]]
        pts[k] = to_data(centre_px(mk), c['inv'], c['M'], cal_px=max(1.0, float(c['res'].max())), others=others, cross_r=cross_r)
    return c, parts, pts

def overlay(c, pts, fname, crop, scale=2):
    rgb = Image.fromarray(np.stack([im] * 3, -1)).convert('RGB'); dr = ImageDraw.Draw(rgb)
    cols = [(255, 0, 0), (0, 0, 255), (0, 160, 0), (255, 0, 255), (0, 170, 170), (255, 140, 0)]
    for i, (k, pp) in enumerate(pts.items()):
        col = cols[i % len(cols)]
        for X, Y, *_ in pp:
            x, y = c['fwd'](X, Y)[:, 0]
            dr.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), outline=col)
    for X, Y in c['D']:
        x, y = c['fwd'](X, Y)[:, 0]; dr.line((x - 4, y, x + 4, y), fill=(255, 140, 0)); dr.line((x, y - 4, x, y + 4), fill=(255, 140, 0))
    cc = rgb.crop(crop); cc.resize((cc.size[0] * scale, cc.size[1] * scale), Image.LANCZOS).save(fname)

OA = (560, 500)
A = dict(cal_args=([(13, 710), (14, 787.5), (15, 865), (16, 944), (17, 1022.5)],
                   [(-5, 586), (-10, 640), (-15, 693.5), (-20, 747.5), (-25, 802.5), (-30, 856.5)], (540, 905), (640, 1100)),
         frame=(634, 1100, 534, 908),
         guides={'gamma=90': Z([(140, 248), (300, 255), (400, 240), (530, 215), (650, 180), (770, 148)], OA),
                 'gamma=90#r': Z([(975, 85), (1020, 70), (1070, 60)], OA),
                 'gamma=105': Z([(140, 315), (300, 330), (450, 330), (600, 310), (770, 272)], OA),
                 'gamma=105#r': Z([(975, 180), (1030, 150), (1075, 110)], OA),
                 'gamma=120': Z([(140, 393), (300, 410), (450, 417), (600, 410), (770, 385)], OA),
                 'gamma=120#r': Z([(975, 318), (1030, 290), (1075, 250)], OA),
                 'gamma=135': Z([(140, 498), (300, 520), (450, 528), (600, 525), (770, 515)], OA),
                 'gamma=135#r': Z([(975, 465), (1030, 445), (1075, 410)], OA),
                 'gamma=150': Z([(140, 582), (300, 612), (450, 632), (600, 648), (770, 652)], OA),
                 'gamma=150#r': Z([(975, 632), (1030, 622), (1075, 598)], OA)},
         masks=[(946, 1047, 534, 908)])
if __name__ == '__main__' and 'A' in sys.argv:
    c, parts, pts = panel(**A)
    print('res', c['res'])
    for k, v in pts.items(): print(k, len(v))
    overlay(c, pts, '../ovr3a.png', (560, 500, 1130, 960))

OB = (1540, 500)
B = dict(cal_args=([(13, 1674), (14, 1752.5), (15, 1829), (16, 1905), (17, 1982.5)],
                   [(-30, 600), (-35, 662.5), (-40, 725), (-45, 787.5), (-50, 849)], (545, 905), (1600, 2055)),
         frame=(1598, 2058, 542, 906),
         guides={'gamma=105': Z([(108, 110), (210, 185), (260, 222), (300, 262), (350, 318), (400, 375), (430, 430), (470, 540), (500, 640), (520, 740), (530, 810)], OB),
                 'gamma=105#r': Z([(555, 810), (565, 700), (585, 600), (610, 480), (650, 410), (700, 360), (730, 327), (800, 265), (885, 198), (960, 150), (1040, 95)], OB),
                 'gamma=120': Z([(108, 275), (200, 330), (300, 410), (400, 500), (470, 610), (510, 730), (520, 810)], OB),
                 'gamma=120#r': Z([(615, 810), (630, 720), (680, 600), (760, 490), (860, 410), (960, 355), (1040, 315)], OB),
                 'gamma=135': Z([(108, 590), (200, 650), (300, 740), (350, 810)], OB),
                 'gamma=135#r': Z([(720, 810), (800, 750), (920, 715), (980, 725), (1020, 770), (1040, 805)], OB)},
         masks=[(1682, 1800, 555, 616), (1755, 1873, 635, 693), (1910, 2030, 772, 825), (1645, 1680, 586, 609),
                (1698, 1752, 690, 715), (1918, 1942, 838, 882)])
if __name__ == '__main__' and 'B' in sys.argv:
    c, parts, pts = panel(**B, tol=5, refine=(12,))
    print('res', c['res'])
    for k, v in pts.items(): print(k, len(v))
    overlay(c, pts, '../ovr3b.png', (1540, 500, 2100, 960))

OC = (560, 1280)
C = dict(cal_args=([(13, 716), (14, 794), (15, 871), (16, 949), (17, 1026)],
                   [(-5, 1363.5), (-10, 1417.5), (-15, 1471.5), (-20, 1525), (-25, 1579), (-30, 1633)], (1312, 1684), (640, 1098)),
         frame=(639, 1100, 1311, 1685),
         guides={'gamma=90': Z([(155, 298), (300, 262), (450, 210), (600, 165), (775, 133)], OC), 'gamma=90#r': Z([(930, 115), (1080, 98)], OC),
                 'gamma=105': Z([(155, 385), (300, 350), (450, 300), (600, 262), (775, 225)], OC), 'gamma=105#r': Z([(930, 185), (1080, 155)], OC),
                 'gamma=120': Z([(155, 490), (300, 440), (450, 400), (600, 360), (775, 328)], OC), 'gamma=120#r': Z([(930, 295), (1080, 258)], OC),
                 'gamma=135': Z([(155, 603), (300, 560), (450, 520), (600, 485), (775, 452)], OC), 'gamma=135#r': Z([(930, 412), (1080, 372)], OC),
                 'gamma=150': Z([(155, 760), (300, 718), (450, 680), (600, 645), (775, 610)], OC), 'gamma=150#r': Z([(930, 575), (1080, 540)], OC)},
         masks=[(948, 1018, 1311, 1685)])
OD = (1540, 1280)
Dd = dict(cal_args=([(13, 1677.5), (14, 1755), (15, 1832.5), (16, 1910), (17, 1987.5)],
                    [(-40, 1376), (-45, 1437.5), (-50, 1498.5), (-55, 1560), (-60, 1621)], (1318, 1679), (1602, 2060)),
          frame=(1600, 2062, 1317, 1681),
          guides={'gamma=105': Z([(115, 115), (300, 125), (450, 150), (530, 178), (580, 199), (657, 258), (735, 340), (781, 438), (805, 560), (820, 683), (836, 806)], OD),
                  'gamma=120': Z([(115, 298), (250, 280), (400, 262), (550, 258), (700, 262), (850, 270), (1045, 272)], OD),
                  'gamma=135': Z([(115, 505), (250, 485), (400, 485), (502, 500), (580, 515), (657, 545), (735, 592), (766, 640), (797, 700), (820, 760), (830, 800)], OD),
                  'right branch': Z([(930, 800), (980, 650), (1045, 435)], OD)},
          masks=[(1872, 1990, 1340, 1400), ('seg', (1847, 1391), (1878, 1386), 4), (1690, 1808, 1440, 1497),
                 ('seg', (1662, 1421), (1690, 1475), 4), (1705, 1822, 1558, 1622), ('seg', (1641, 1527), (1702, 1600), 4)])
if __name__ == '__main__' and 'C' in sys.argv:
    c, parts, pts = panel(**C, tol=6, refine=(12,))
    print('res', c['res'])
    for k, v in pts.items(): print(k, len(v))
    overlay(c, pts, '../ovr3c.png', (560, 1280, 1130, 1740))
if __name__ == '__main__' and 'D' in sys.argv:
    c, parts, pts = panel(**Dd, tol=5, cross_r=6)
    print('res', c['res'])
    for k, v in pts.items(): print(k, len(v))
    overlay(c, pts, '../ovr3d.png', (1540, 1280, 2100, 1740))

O4 = (440, 1940)
def Z4(pts): return Z(pts, O4, 1.5)
F4 = dict(guides={'l=0': Z4([(100, 108), (250, 125), (400, 150), (500, 180), (600, 200), (700, 225), (800, 240), (870, 240), (950, 215), (1050, 200), (1150, 190), (1180, 195)]),
                  'l=b': Z4([(100, 122), (200, 160), (300, 240), (370, 330), (400, 365), (430, 370), (470, 320), (520, 265), (600, 220), (700, 213), (800, 225), (900, 260), (950, 320), (990, 430), (1005, 520), (1010, 580)]),
                  'l=b#r': Z4([(1015, 580), (1030, 480), (1060, 370), (1100, 300), (1150, 255), (1180, 240)])},
          masks=[(1040, 1228, 1970, 2040), ('seg', (745, 2022), (785, 2007), 4), ('seg', (650, 2096), (690, 2086), 4),
                 (600, 1000, 2140, 2380)])

def fig4():
    """Frame-calibrated (no ticks or gridlines in this figure)."""
    from scipy.spatial import cKDTree
    from dig import vruns
    XF = (506.5, 1229.0); YF = (1966.5, 2401.0)
    M = np.array([[(XF[1] - XF[0]) / 4.2, 0.0], [0.0, -(YF[1] - YF[0]) / 60.0]])
    t = np.array([XF[0] - 8.2 * M[0, 0], YF[0] - 10.0 * (YF[1] - YF[0]) / 60.0])
    Minv = np.linalg.inv(M)
    c = dict(M=M, t=t, res=np.array([1.0, 1.0]), D=np.array([[9, -30], [10, -50], [11, -30]]),
             fwd=lambda X, Y: (np.c_[np.atleast_1d(X), np.atleast_1d(Y)] @ M.T + t).T,
             inv=lambda x, y: ((np.c_[np.atleast_1d(x), np.atleast_1d(y)] - t) @ Minv.T).T)
    m = np.zeros_like(d); m[1971:2397, 511:1225] = d[1971:2397, 511:1225]
    for mk_ in F4['masks']:
        if mk_[0] == 'seg':
            _, p0, p1, w = mk_
            yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
            p0 = np.array(p0, float); p1 = np.array(p1, float); dv = p1 - p0
            tt = np.clip(((xx - p0[0]) * dv[0] + (yy - p0[1]) * dv[1]) / (dv @ dv), 0, 1)
            m &= ~(np.hypot(xx - p0[0] - tt * dv[0], yy - p0[1] - tt * dv[1]) <= w)
        else:
            a, b, e, f = mk_; m[e:f, a:b] = False
    m = remove_small(m, 10)
    gpx = {k: (np.array([p[0] for p in v]), np.array([p[1] for p in v])) for k, v in F4['guides'].items()}
    parts = assign_euclid(m, gpx, tol=7)
    trees = {k: cKDTree(np.c_[np.nonzero(v)[1], np.nonzero(v)[0]]) for k, v in parts.items() if v.any()}
    pts = {}
    for k, mk in parts.items():
        cp = centre_px(mk)
        keep = []
        for x, y, s in cp:
            col = mk[:, int(round(x))]
            runs = vruns(col)
            if abs(s) <= 1.5 and len(runs) > 1 and (runs[-1][0] - runs[0][1]) > 2:
                continue  # theory and measurement separate here: not a single line
            keep.append((x, y, s))
        others = [tr for kk, tr in trees.items() if kk.split('#')[0] != k.split('#')[0]]
        pts[k] = to_data(np.array(keep) if keep else np.zeros((0, 3)), c['inv'], c['M'], cal_px=1.5, others=others, cross_r=6)
    return c, parts, pts

if __name__ == '__main__' and 'F4' in sys.argv:
    c, parts, pts = fig4()
    for k, v in pts.items(): print(k, len(v))
    overlay(c, pts, '../ovr4.png', (440, 1940, 1250, 2460))
