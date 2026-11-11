"""L2 pass b: Alimenti et al. (1994) Figs. 2-6 on the embedded 300 ppi page scans (pdfimages).
Calibration from major and minor ticks (minor ticks identified against the major-tick fit), column-run scanning of
each line (segments seeded by hand), markers from a morphological opening (disc larger than the stroke).

    python3 alimenti_b.py <reference folder> <scratch folder> [figure numbers]
"""
import os
import sys
import pickle
import numpy as np
from scipy import ndimage as ndi
from l2common import ALIMENTI_PDF, native, ticks, bin_curve
from colscan import scan, scan_guided

REF, SCR, OUT = sys.argv[1:4]
ONLY = [int(v) for v in sys.argv[4:]] or [2, 3, 4, 5, 6]
PDF = os.path.join(REF, ALIMENTI_PDF)
_pages = {}


def page(p):
    if p not in _pages:
        _pages[p] = native(PDF, p, SCR, 'alimenti')[0]
    return _pages[p]


class Cal:
    """Separable linear calibration from major + minor ticks; the axis lines' own positions are the end ticks."""

    def __init__(self, im, X, Y, xmaj, ymaj, xvals, yvals, xminor, yminor, xr, yr, extra_x=(), extra_y=()):
        sub = im[Y - 15:Y + 15, xr[0] + 40:xr[1] - 40] < 128
        self.Y = Y - 15 + (sub.sum(1) * np.arange(30)).sum() / sub.sum()
        sub = im[yr[0] + 40:yr[1] - 40, X - 15:X + 15] < 128
        self.X = X - 15 + (sub.sum(0) * np.arange(30)).sum() / sub.sum()
        tx = ticks(im, 'x', int(round(self.Y)), xmaj, xr, 0.7)
        ty = ticks(im, 'y', int(round(self.X)), ymaj, yr, 0.7)
        assert len(tx) == len(xvals) and len(ty) == len(yvals), (tx, ty)
        px, vx = list(tx) + [self.X] * len(extra_x), list(xvals) + list(extra_x)
        py, vy = list(ty) + [self.Y] * len(extra_y), list(yvals) + list(extra_y)
        c0x, c0y = np.polyfit(px, vx, 1), np.polyfit(py, vy, 1)
        self.nmin = [0, 0]
        for mspec, lst_p, lst_v, c0, axis, pos, rng in (
                (xminor, px, vx, c0x, 'x', int(round(self.Y)), xr), (yminor, py, vy, c0y, 'y', int(round(self.X)), yr)):
            if mspec is None:
                continue
            band, step = mspec
            for p in ticks(im, axis, pos, band, rng, 0.7):
                v = np.polyval(c0, p)
                vr = np.round(v / step) * step
                pe = (vr - c0[1]) / c0[0]
                if abs(p - pe) < 3 and all(abs(p - q) > 4 for q in lst_p):
                    lst_p.append(p); lst_v.append(vr)
                    self.nmin[0 if axis == 'x' else 1] += 1
        self.px, self.vx, self.py, self.vy = map(np.array, (px, vx, py, vy))
        self.cx, self.cy = np.polyfit(self.px, self.vx, 1), np.polyfit(self.py, self.vy, 1)
        self.sx = float(np.sqrt(np.mean((self.vx - np.polyval(self.cx, self.px)) ** 2)))
        self.sy = float(np.sqrt(np.mean((self.vy - np.polyval(self.cy, self.py)) ** 2)))

    def __call__(self, px, py):
        return np.polyval(self.cx, px), np.polyval(self.cy, py)

    def describe(self):
        return (f'{len(self.px)} x ticks ({self.nmin[0]} minor) and {len(self.py)} y ticks ({self.nmin[1]} minor) on '
                f'the native scan; fit x = {self.cx[0]:.7g}*px + {self.cx[1]:.7g} (rms residual {self.sx:.3g}), '
                f'y = {self.cy[0]:.7g}*py + {self.cy[1]:.7g} (rms residual {self.sy:.3g}); no tilt correction')


S = 0.75     # native px per render px (300 / 400 dpi); hand seeds were read on the 400 dpi renders


def sc(p):
    return tuple(int(round(v * S)) for v in p)


def G(*pts, **kw):
    """guided column scan segment; guide points in 400 dpi render px"""
    return dict(guide=[(x * S, y * S) for x, y in pts], kw=kw)


def C(x0, x1, y0, **kw):
    """column scan segment, coordinates in 400 dpi render px (converted)"""
    return dict(x0=int(round(x0 * S)), x1=int(round(x1 * S)), y0=y0 * S, kw=kw)


FIGS = {
    2: dict(page=2, cal=dict(X=923, Y=2615, xmaj=(4, 22), ymaj=(-22, -4), xvals=[1, 1.2, 1.4, 1.6, 1.8, 2.0],
                             yvals=[0, -2, -4, -6, -8], extra_y=[-10], xminor=None, yminor=((4, 12), 0.5),
                             xr=(918, 1755), yr=(1996, 2612)),
            lines={'MMFD': [C(1462, 2248, 3328, maxlen=60, maxjump=10)]},
            markers={'MM': dict(r=7, approx=[(1500, 3355), (1595, 3380), (1660, 3368), (1740, 3320), (1815, 3240), (1900, 3115), (1980, 2970), (2060, 2815), (2140, 2735), (2220, 2778)])}, ylim=(-10, 0)),
    3: dict(page=3, cal=dict(X=916, Y=1040, xmaj=(-22, -4), ymaj=(4, 22), xvals=[1.2, 1.4, 1.6, 1.8],
                             yvals=[0, -10, -20, -30, -40, -50, -60], extra_x=[1.0], extra_y=[-70],
                             xminor=((-12, -4), 0.05), yminor=((4, 12), 2.0), xr=(960, 1650), yr=(442, 1036)),
            lines={'MM': [C(1452, 2190, 692, maxlen=60, maxjump=10)]},
            markers={'MMFD': dict(r=6, approx=[(1487, 690), (1567, 712), (1650, 718), (1727, 724), (1805, 742), (1883, 763), (1960, 783), (2045, 810), (2125, 855), (2190, 905)])}, ylim=(-70, 0)),
    4: dict(page=3, cal=dict(X=927, Y=2590, xmaj=(4, 22), ymaj=(-22, -4), xvals=[1, 1.2, 1.4, 1.6, 1.8, 2.0],
                             yvals=[0, -10, -20, -30, -40], extra_y=[-50], xminor=None, yminor=((4, 12), 2.0),
                             xr=(922, 1770), yr=(1858, 2587)),
            lines={'t/a=0.30': [G((1459, 3221), (1513, 3367), (1522, 3440)),
                                G((1540, 3440), (1580, 3283), (1647, 3192), (1747, 3112), (1897, 3046), (2013, 3017),
                                  (2138, 2975), (2240, 2925))],
                   't/a=0.34': [G((1459, 3042), (1572, 3121), (1647, 3200), (1680, 3246), (1700, 3289), (1712, 3325), (1722, 3367), (1734, 3440)),
                                G((1760, 3440), (1797, 3300), (1847, 3221), (1897, 3158), (1972, 3100), (2055, 3050),
                                  (2122, 3000), (2200, 2962), (2225, 2972), (2240, 2988))],
                   't/a=0.38': [G((1457, 2978), (1520, 3000), (1600, 3040), (1680, 3075), (1740, 3105), (1790, 3150),
                                  (1830, 3200), (1850, 3220), (1860, 3228), (1875, 3249), (1890, 3285), (1900, 3318), (1910, 3348), (1915, 3440), maxjump=8),
                                G((1945, 3440), (1960, 3380), (1975, 3310), (2000, 3240), (2040, 3180), (2100, 3125),
                                  (2150, 3112), (2180, 3135), (2186, 3140), maxjump=8),
                                G((2187, 3142), (2199, 3176), (2203, 3240), (2206, 3285), maxjump=8, maxlen=60),
                                G((2210, 3330), (2214, 3264), (2219, 3200), (2225, 3080), (2236, 2958), maxjump=10,
                                  maxlen=60)]},
            markers={}, dashed=['t/a=0.38'], ylim=(-49.6, 0)),
    5: dict(page=4, cal=dict(X=742, Y=1524, xmaj=(4, 22), ymaj=(-22, -4), xvals=[.30, .32, .34, .36, .38, .40],
                             yvals=[2.0, 1.8, 1.6, 1.4, 1.2], extra_y=[1.0], xminor=((-12, -4), 0.005),
                             yminor=((4, 22), 0.05), xr=(738, 1735), yr=(746, 1521)),
            lines={'matching frequency': [G((1005, 1748), (1300, 1624.5), (1700, 1480.5), (1850, 1428), (1950, 1396),
                                            (2040, 1358), (2080, 1346.5), (2121, 1333), maxjump=5, maxlen=7),
                                          G((2124, 1331), (2160, 1312.5), (2250, 1260), (2290, 1235), (2300, 1230),
                                            maxjump=5, maxlen=7)],
                   'bandwidth, one zero': [G((1005, 1554.5), (1100, 1543), (1250, 1527.5), (1530, 1537), (1700, 1546.5),
                                             (1850, 1554.5), (1950, 1553), (2040, 1545.5), (2080, 1524), (2100, 1513),
                                             (2118, 1503), maxjump=5, maxlen=7)],
                   'bandwidth, two zeros': [G((2135, 1330), (2200, 1392), (2288, 1474), maxjump=5, maxlen=7)]},
            markers={}, ylim=(1.0, 2.0)),
    6: dict(page=4, cal=dict(X=863, Y=2670, xmaj=(4, 22), ymaj=(-22, -4), xvals=[1, 1.2, 1.4, 1.6, 1.8, 2.0],
                             yvals=[0, -10, -20, -30], extra_y=[-40], xminor=None, yminor=((4, 12), 2.0),
                             xr=(858, 1714), yr=(2088, 2667)),
            lines={'common': [G((1385, 3140), (1420, 3158), (1480, 3185), (1530, 3218), (1600, 3262), (1640, 3283),
                                (1655, 3292), maxjump=6)],
                   'MMFD': [G((1662, 3297), (1690, 3313), (1720, 3337), (1750, 3364.5), (1800, 3427.5), (1830, 3470),
                              (1850, 3520), (1856, 3545), maxjump=7),
                            G((1945, 3545), (1960, 3511.5), (2000, 3434.5), (2040, 3400), (2060, 3388), (2080, 3394),
                              (2100, 3424.5), (2110, 3462), (2118, 3545), maxjump=7),
                            G((2130, 3550), (2135, 3518), (2150, 3253), (2160, 3060), (2166, 3020), maxjump=12,
                              maxlen=200)],
                   'BCMM': [G((1662, 3300), (1690, 3328), (1720, 3355.5), (1750, 3388), (1770, 3419.5), (1790, 3456),
                              (1810, 3495), (1822, 3545), maxjump=7),
                            G((1932, 3545), (1960, 3487.5), (2000, 3434.5), (2040, 3400), (2060, 3404), (2082, 3410),
                              maxjump=7)]},
            markers={'BCMM': dict(r=6, fill=True, approx=[(1421, 3160), (1478, 3183), (1530, 3218), (1637, 3280), (1688, 3328), (1745, 3388), (1800, 3480), (1967, 3466), (2045, 3404), (2078, 3410)])}, ylim=(-39.7, 0)),
}
EXCL = {4: 6.0, 5: 4.5, 6: 3.0}
STEP = {2: 0.0025, 3: 0.0025, 4: 0.0025, 5: 0.0002, 6: 0.0025}


def markers(im, spec, L=None):
    """Matched filter: the ink (holes filled for open symbols) is correlated with a disc of the symbol's radius; each
    marker is the correlation maximum within +-search px of a hand-placed approximate position (read by eye on the
    page), refined to the centroid of the correlation values above 97 % of that maximum."""
    r, srch = spec['r'], spec.get('search', 9)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = ((xx ** 2 + yy ** 2) <= r * r).astype(float)
    out = []
    for ax, ay in spec['approx']:
        cx, cy = int(round(ax * S)), int(round(ay * S))
        y0, x0 = cy - srch - r - 2, cx - srch - r - 2
        ink = (im[y0:cy + srch + r + 3, x0:cx + srch + r + 3] < 128)
        if spec.get('fill'):
            ink = ndi.binary_fill_holes(ink)
        cc = ndi.correlate(ink.astype(float), disc, mode='constant')
        win = cc[r + 2:-(r + 2), r + 2:-(r + 2)]
        m = win >= 0.97 * win.max()
        jy, jx = np.where(m)
        w = win[m]
        out.append(((w * jx).sum() / w.sum() + x0 + r + 2, (w * jy).sum() / w.sum() + y0 + r + 2))
    return np.array(out)


def near(a, b, dist):
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]).min(1)
    return d < dist


def process(k):
    fig = FIGS[k]
    im = page(fig['page'])
    cal = Cal(im, **fig['cal'])
    raw = {}
    for name, segs in fig['lines'].items():
        pts = []
        for s in segs:
            if 'guide' in s:
                pts.append(scan_guided(im, s['guide'], **s['kw']))
                continue
            r = scan(im, s['x0'], s['x1'], s['y0'], max(int(s['y0']) - 700, 0), int(s['y0']) + 700, thr=140, **s['kw'])
            r = r[~np.isnan(r[:, 1])]
            pts.append(r)
        raw[name] = np.vstack(pts)
    # dashed curves: drop points on components much larger than a dash (the open squares)
    for name in [n for n in raw if fig.get('dashed') and n in fig['dashed']]:
        lab, n = ndi.label(im < 128, structure=np.ones((3, 3)))
        objs = ndi.find_objects(lab)
        r = raw[name]
        ids = np.array([lab[int(round(y)), int(round(x))] for x, y, _ in r])
        size = np.array([0 if i == 0 else (objs[i - 1][0].stop - objs[i - 1][0].start) *
                         (objs[i - 1][1].stop - objs[i - 1][1].start) for i in ids])
        mind = np.array([0 if i == 0 else min(objs[i - 1][0].stop - objs[i - 1][0].start,
                                              objs[i - 1][1].stop - objs[i - 1][1].start) for i in ids])
        med = np.median(size[size > 0])
        raw[name] = r[(size > 0) & (size <= 2.0 * med) & (mind < 14)]     # squares: >= 14 px in both directions
    # drop columns whose run is much longer than the neighbours' (merged with another stroke or a symbol) only for
    # the curves without symbols on them
    clean = {}
    for name, r in raw.items():
        r = r[np.argsort(r[:, 0])]
        if k in EXCL:
            for n2, o in raw.items():
                if n2 != name and not (k == 6 and 'common' in (name, n2)):
                    r = r[~near(r, o, EXCL[k])]
        clean[name] = r
    mk = {n: markers(im, spec) for n, spec in fig['markers'].items()}
    curves = {}
    for name, r in clean.items():
        if name == 'common':
            continue
        curves[name] = [r]
    if k == 6:
        for n in ('MMFD', 'BCMM'):
            curves[n].append(clean['common'])
    out = {}
    lo, hi = fig['ylim']
    extra = {}
    for name, rs in curves.items():
        r = np.vstack(rs)
        r = r[np.argsort(r[:, 0])]
        rows = []
        for i, (x, y, L) in enumerate(r):
            X, Y = cal(x, y)
            j0, j1 = max(i - 3, 0), min(i + 3, len(r) - 1)
            slp = (r[j1, 1] - r[j0, 1]) / max(r[j1, 0] - r[j0, 0], 1)
            slp = np.clip(slp, -11.4, 11.4) * cal.cy[0] / cal.cx[0]
            u = np.sqrt((max(1.0, L / 4) * abs(cal.cy[0])) ** 2 + cal.sy ** 2 + (slp * cal.sx) ** 2)
            if k == 5 and (min(abs(Y - g) for g in (1.2, 1.4, 1.6, 1.8)) < 3 * abs(cal.cy[0]) or
                           min(abs(X - g) for g in (0.32, 0.34, 0.36, 0.38)) < 3 * cal.cx[0]):
                continue      # on a dotted gridline: the run may be the gridline's
            if lo <= Y <= hi:
                rows.append((X, Y, u))
        P = np.array(rows)
        if k == 5 and name.startswith('bandwidth'):
            P[:, 1] = (P[:, 1] - 1) / 2
            P[:, 2] /= 2
        B = bin_curve(P[:, 0], P[:, 1], P[:, 2], STEP[k])
        out[name] = [(name, x, y, u, 0.0) for x, y, u in B]
    if k == 5:
        # jump: ink columns of the vertical stroke between the two branches (rows 1020-1110 native)
        sub = im[1020:1110, 1575:1605] < 128
        w = sub.sum(0).astype(float)
        xj = 1575 + (w * np.arange(30)).sum() / w.sum()
        rr = np.where(im[975:1140, int(round(xj))] < 128)[0]
        X, Yt = cal(xj, 975 + rr.min() + 1.5)
        _, Yb = cal(xj, 975 + rr.max() - 1.5)
        extra['jump'] = (X, (Yb - 1) / 2, (Yt - 1) / 2, float(np.hypot(1.5 * cal.cx[0], cal.sx)))
        u_end = float(np.hypot(1.5 * abs(cal.cy[0]), cal.sy)) / 2
        out['bandwidth, one zero'] = [r for r in out['bandwidth, one zero'] if r[1] < X - 0.0004] + \
            [('bandwidth, one zero', X, (Yb - 1) / 2, u_end, 0.0)]
        out['bandwidth, two zeros'] = [('bandwidth, two zeros', X, (Yt - 1) / 2, u_end, 0.0)] + \
            [r for r in out['bandwidth, two zeros'] if r[1] > X + 0.0004]
    for name, M in mk.items():
        out[name + ' markers'] = [(name + ' markers', *cal(cx, cy), float(np.hypot(1.5 * abs(cal.cy[0]), cal.sy)),
                                   float(np.hypot(1.5 * abs(cal.cx[0]), cal.sx))) for cx, cy in M]
    return cal, out, extra, clean, mk


def write_all(k, caldesc, out, extra, outdir):
    import alim_meta as M
    from l2common import write_raw
    for fname, (names, note) in M.FILES[k].items():
        rows = [(r[0].replace(',', ''),) + tuple(r[1:]) for n in names for r in out.get(n, [])]
        hdr = ['source: ' + M.SRC, 'figure: ' + M.FIG[k]['figure'], 'axes: ' + M.FIG[k]['axes'],
               'geometry: ' + M.FIG[k]['geom'] + '. ' + M.GEOM_T, 'curves: ' + note,
               'calibration (L2b): ' + caldesc, 'method (L2b): embedded 300 ppi page scan (pdfimages); column-run scan (ink-weighted centre of the vertical ink run nearest a linear prediction or a coarse hand-placed guide polyline) in hand-seeded segments; dashed line: components larger than twice the median dash (the squares) dropped; markers: correlation of the ink (holes filled for open symbols) with a disc of the symbol radius, maximum within 9 px of a position placed by eye; points within a few px of another curve dropped; binned in x; uncertainty = sqrt((max(1, L/4) px)^2 + cal_y^2 + (slope cal_x)^2), L the run length; markers: 1.5 px in each axis combined with the tick-fit residual',
               'digitizer: L2 (pass b), 2026-11-11']
        if extra.get('jump'):
            j = extra['jump']
            hdr.append(f'jump (L2b): t/a = {j[0]:.5f} +- {j[3]:.5f}, bandwidth {j[1]:.4f} -> {j[2]:.4f}')
        write_raw(os.path.join(outdir, fname), hdr, rows, ydec=5, xdec=5)


if __name__ == '__main__':
    res = {}
    for k in ONLY:
        cal, out, extra, clean, mk = process(k)
        res[k] = (cal.describe(), out, extra)
        write_all(k, cal.describe(), out, extra, OUT)
        np.save(os.path.join(SCR, f'alim_b_fig{k}_px.npy'), dict(lines=clean, markers=mk), allow_pickle=True)
        for n, rows in out.items():
            r = np.array([v[1:] for v in rows], float).reshape(-1, 4)
            if len(r) == 0:
                print(k, n, 0); continue
            print(k, n, len(rows), 'x %.4f-%.4f' % (r[:, 0].min(), r[:, 0].max()),
                  'u %.3g/%.3g/%.3g' % tuple(np.percentile(r[:, 2], [0, 50, 100])))
        if extra:
            print('   extra', extra)
        print('   cal', cal.describe())
    with open(os.path.join(SCR, 'alim_b_results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
