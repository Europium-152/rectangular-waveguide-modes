"""L2 pass a: Alimenti, Mongiardo & Sorrentino (1994), Figs. 2-6, 400 dpi pdftoppm renders of PDF pages 2-4.
Ridge follower along each printed line (segments with hand-set start points), markers from the maxima of the ink
distance transform, tick calibration (alimcal.TickCal).

    python3 alimenti_a.py <reference folder> <scratch folder> <output folder> [figure numbers]
"""
import os
import sys
import numpy as np
from scipy import ndimage as ndi
from l2common import ALIMENTI_PDF, render, bin_curve
from alimcal import TickCal
from ridge import follow

REF, SCR, OUT = sys.argv[1:4]
ONLY = [int(v) for v in sys.argv[4:]] or [2, 3, 4, 5, 6]
PDF = os.path.join(REF, ALIMENTI_PDF)
pages = {}


def page(p):
    if p not in pages:
        pages[p] = render(PDF, p, 400, SCR, 'alimenti')[0]
    return pages[p]


def F(p0, v0, xend, ybot=None, ytop=None, **kw):
    """segment spec"""
    return dict(p0=p0, v0=v0, xend=xend, ybot=ybot, ytop=ytop, kw=kw)


LINE = dict(step=2.0, h=12, thr=100, maxw=9)
THIN = dict(step=2.0, h=9, thr=90, maxw=6)
DASH = dict(step=2.0, h=8, thr=90, maxw=7, inertia=0.8, search=False)
W4 = dict(step=2.0, h=14, thr=100, maxw=12)
W6 = dict(step=2.0, h=10, thr=100, maxw=9)
D4 = dict(step=2.0, h=12, thr=100, maxw=12, inertia=0.8, search=False)

FIGS = {
    2: dict(page=2, cal=dict(X=1235, Y=3489, xt=((6, 30), (1225, 2341)), yt=((-30, -6), (2662, 3485)),
                             xvals=[1, 1.2, 1.4, 1.6, 1.8, 2.0], yvals=[0, -2, -4, -6, -8], extra_y=[-10],
                             xr_axis=(1250, 2330), yr_axis=(2700, 3470)),
            lines={'MMFD': [F((1462, 3328), (1, 0.5), 2248, **dict(LINE, h=22, maxw=24))]},
            markers={'MM': dict(box=(1440, 2700, 2300, 3480), dtmin=8.5, size=21, prom=1.75, line='MMFD')},
            files={'alimenti1994_fig2_mmfd.csv': ['MMFD'], 'alimenti1994_fig2_mm.csv': ['MM']}),
    3: dict(page=3, cal=dict(X=1224, Y=1386, xt=((-30, -6), (1300, 2200)), yt=((6, 30), (590, 1380)),
                             xvals=[1.2, 1.4, 1.6, 1.8], yvals=[0, -10, -20, -30, -40, -50, -60], extra_x=[1.0],
                             extra_y=[-70], xr_axis=(1240, 2200), yr_axis=(620, 1370)),
            lines={'MM': [F((1452, 692), (1, 0.05), 2190, **dict(LINE, h=22, maxw=26))]},
            markers={'MMFD': dict(box=(1440, 640, 2200, 960), dtmin=6.0, size=21, prom=1.75, line='MM')},
            files={'alimenti1994_fig3_mm.csv': ['MM']}),
    4: dict(page=3, cal=dict(X=1239, Y=3455, xt=((6, 30), (1230, 2360)), yt=((-30, -6), (2478, 3450)),
                             xvals=[1, 1.2, 1.4, 1.6, 1.8, 2.0], yvals=[0, -10, -20, -30, -40], extra_y=[-50],
                             xr_axis=(1260, 2340), yr_axis=(2720, 3440)),
            lines={'t/a=0.30': [F((1460, 3227), (0.3, 1), 1540, ybot=3425, **W4),
                                F((1557, 3394), (0.2, -1), 2242, **W4)],
                   't/a=0.34': [F((1460, 3042.5), (1, 0.4), 1628, **dict(W4, maxw=22, h=16)),
                                F((1656, 3216), (0.5, 1), 1760, ybot=3425, **dict(W4, maxw=22, h=16)),
                                F((1772, 3380), (0.25, -1), 2241, ytop=2965, **dict(W4, maxw=22, h=16))]},
            dashed={'t/a=0.38': dict(guide=[(1457, 2978), (1520, 3000), (1600, 3040), (1680, 3075), (1740, 3105),
                                            (1790, 3150), (1830, 3200), (1860, 3250), (1890, 3330), (1905, 3400),
                                            (1915, 3440)], band=22),
                    't/a=0.38 right': dict(guide=[(1945, 3440), (1960, 3380), (1975, 3310), (2000, 3240),
                                                  (2040, 3180), (2100, 3125), (2150, 3112), (2180, 3135), (2190, 3150)],
                                           band=22),
                    't/a=0.38 right2': dict(guide=[(2188, 3146), (2199, 3176), (2202.5, 3240), (2203, 3285)], band=5),
                    't/a=0.38 end': dict(guide=[(2213, 3335), (2212, 3297), (2214, 3264), (2219, 3234), (2218.6, 3173),
                                                (2225, 3140), (2225.5, 3018), (2235, 2959)], band=6)},
            markers={},
            files={'alimenti1994_fig4_t030.csv': ['t/a=0.30'], 'alimenti1994_fig4_t034.csv': ['t/a=0.34'],
                   'alimenti1994_fig4_t038.csv': ['t/a=0.38']}),
    5: dict(page=4, cal=dict(X=993, Y=2035, xt=((6, 30), (985, 2320)), yt=((-30, -6), (996, 2030)),
                             xvals=[.30, .32, .34, .36, .38, .40], yvals=[2.0, 1.8, 1.6, 1.4, 1.2], extra_y=[1.0],
                             xr_axis=(1010, 2290), yr_axis=(1060, 2020)),
            lines={'matching frequency': [F((1005, 1748), (1, -0.43), 2124, **dict(THIN, search=False, inertia=0.8)),
                                          F((2160, 1312.5), (-1, 0.58), 2128, **dict(THIN, search=False, inertia=0.8)),
                                          F((2160, 1312.5), (1, -0.58), 2303, **dict(THIN, search=False, inertia=0.8))],
                   'bandwidth, one zero': [F((1000, 1554.5), (1, -0.12), 2119, **THIN)],
                   'bandwidth, two zeros': [F((2200, 1392), (-1, -0.94), 2124, **dict(THIN, search=False, inertia=0.8)),
                                            F((2200, 1392), (1, 0.94), 2303, **dict(THIN, search=False, inertia=0.8))]},
            markers={},
            files={'alimenti1994_fig5_matching_frequency.csv': ['matching frequency'],
                   'alimenti1994_fig5_bandwidth.csv': ['bandwidth, one zero', 'bandwidth, two zeros']}),
    6: dict(page=4, cal=dict(X=1154, Y=3562, xt=((6, 30), (1145, 2286)), yt=((-30, -6), (2784, 3555)),
                             xvals=[1, 1.2, 1.4, 1.6, 1.8, 2.0], yvals=[0, -10, -20, -30], extra_y=[-40],
                             xr_axis=(1170, 2270), yr_axis=(2840, 3550)),
            lines={'common': [F((1385, 3140), (1, 0.4), 1655, **W6)],
                   'MMFD': [F((1720, 3337), (-1, -0.75), 1662, **W6), F((1720, 3337), (0.55, 1), 1870, ybot=3540, **W6),
                            F((1960, 3511.5), (0.5, -1), 2045, ytop=3394, **W6),
                            F((1960, 3511.5), (-0.5, 1), 1900, ybot=3540, **W6),
                            F((2100, 3424.5), (-0.6, -1), 2054, ytop=3394, **W6),
                            F((2100, 3424.5), (0.3, 1), 2125, ybot=3540, **W6),
                            F((2134, 3500), (0.1, -1), 2170, ytop=3025, **W6)],
                   'BCMM': [F((1720, 3355.5), (-1, -0.9), 1662, **W6), F((1720, 3355.5), (0.5, 1), 1830, ybot=3540, **W6),
                            F((1960, 3487.5), (0.6, -1), 2040, ytop=3396, **W6),
                            F((1960, 3487.5), (-0.5, 1), 1900, ybot=3540, **W6),
                            F((2060, 3404), (1, 0.2), 2082, **W6)]},
            peaks={'MMFD': dict(x=(2058, 2076), yband=(3376, 3400), half=2.75)},
            markers={'BCMM': dict(box=(1380, 3100, 2100, 3555), dtmin=4.0, size=21, prom=1.5, line='BCMMall', fill=True)},
            files={'alimenti1994_fig6_mmfd.csv': ['MMFD'], 'alimenti1994_fig6_bcmm.csv': ['BCMM']}),
}


def dashpts(im, spec, ylim=None, thr=128, binw=3.0):
    """Points of a dashed line: every ink component whose centroid lies within band px of the guide polyline
    (hand-placed, coarse) is cut into 3 px slices along its principal axis; the slice centroids are the points.
    Components larger than 2.5x the median (merged with another line) are dropped."""
    g = np.array(spec['guide'], float)
    xs = np.arange(int(g[:, 0].min()) - 30, int(g[:, 0].max()) + 30)
    ys = np.arange(int(g[:, 1].min()) - 30, int(g[:, 1].max()) + 30)
    y0, x0 = max(ys[0], 0), max(xs[0], 0)
    ink = im[y0:ys[-1], x0:xs[-1]] < thr
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    def dist_to_guide(p):
        d = []
        for a, b in zip(g[:-1], g[1:]):
            t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
            d.append(np.hypot(*(p - (a + t * (b - a)))))
        return min(d)
    comps = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        yy, xx = np.where(lab[sl] == i)
        if len(yy) < 15:
            continue
        P = np.c_[xx + sl[1].start + x0, yy + sl[0].start + y0].astype(float)
        if dist_to_guide(P.mean(0)) <= spec['band']:
            comps.append(P)
    if not comps:
        return np.zeros((0, 6))
    sizes = np.array([len(c) for c in comps])
    med = np.median(sizes)
    out = []
    for P in comps:
        if len(P) > 2.5 * med:
            continue
        c = P.mean(0)
        u, sv, vt = np.linalg.svd(P - c, full_matrices=False)
        ax = vt[0]
        t = (P - c) @ ax
        for b in np.arange(t.min(), t.max() + 1e-9, binw):
            m = (t >= b) & (t < b + binw)
            if m.sum() >= 3:
                q = P[m].mean(0)
                w = len(P[m]) / binw
                out.append((q[0], q[1], w, 1.0, ax[0] * np.sign(ax[0] or 1), ax[1] * np.sign(ax[0] or 1)))
    out = np.array(out)
    return out[np.argsort(out[:, 0])]


def trace(im, segs):
    dark = 255 - im
    out = []
    for s in segs:
        stop = None
        if s['ybot'] is not None or s['ytop'] is not None:
            yb, yt = s['ybot'], s['ytop']
            stop = lambda p, yb=yb, yt=yt: (yb is not None and p[1] > yb) or (yt is not None and p[1] < yt)
        r = follow(dark, s['p0'], s['v0'], s['xend'], stop=stop, **s['kw'])
        out.append(r)
    return out


def markers(im, spec, L):
    """Marker centres: maxima of the ink distance transform (largest inscribed disc) within 8 px of the traced
    line L that stand out by >= prom px above the median line half-width 12-40 px away along the line."""
    x0, y0, x1, y1 = spec['box']
    ink = im[y0:y1, x0:x1] < 128
    if spec.get('fill'):
        ink = ndi.binary_fill_holes(ink)
    dt = ndi.distance_transform_edt(ink)
    prof = np.array([dt[int(y - y0) - 7:int(y - y0) + 8, int(x - x0) - 7:int(x - x0) + 8].max()
                     if x0 + 8 < x < x1 - 8 and y0 + 8 < y < y1 - 8 else np.nan for x, y in L[:, :2]])
    mx = ndi.maximum_filter(dt, size=spec['size'])
    lab, n = ndi.label((dt == mx) & (dt >= spec['dtmin']))
    cen = ndi.center_of_mass(np.ones_like(dt), lab, range(1, n + 1))
    out = []
    for c in cen:
        X, Y = c[1] + x0, c[0] + y0
        dd = np.hypot(L[:, 0] - X, L[:, 1] - Y)
        i = int(dd.argmin())
        if dd[i] > 8:
            continue
        s = np.hypot(L[:, 0] - L[i, 0], L[:, 1] - L[i, 1])
        loc = np.nanmedian(prof[(s > 12) & (s < 40)])
        v = dt[int(round(c[0])), int(round(c[1]))]
        if v - loc >= spec['prom']:
            out.append((X, Y, v, v - loc))
    out = np.array(out)
    # one marker per 20 px
    keep = []
    for p in out[np.argsort(-out[:, 3])] if len(out) else []:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 20 for q in keep):
            keep.append(p)
    return np.array(sorted(keep, key=lambda p: p[0]))


def run(k):
    fig = FIGS[k]
    im = page(fig['page'])
    cal = TickCal(im, label=f'fig{k}', **fig['cal'])
    traced = {name: np.vstack(trace(im, segs)) for name, segs in fig['lines'].items()}
    for name, spec in fig.get('dashed', {}).items():
        traced[name] = dashpts(im, spec)
    for name, spec in fig.get('peaks', {}).items():
        # a maximum that touches a horizontal (dotted) line: centre = lower edge of the ink - half the stroke width
        pk = []
        for x in range(spec['x'][0], spec['x'][1] + 1, 2):
            col = im[spec['yband'][0]:spec['yband'][1], x]
            d = np.where(col < 128)[0]
            if len(d):
                pk.append((x, spec['yband'][0] + d.max() - spec['half'], 2 * spec['half'], 1.0, 1.0, 0.0))
        traced[name] = np.vstack([traced[name], np.array(pk)])
    if 'common' in traced:
        traced['BCMMall'] = np.vstack([traced['common'], traced['BCMM']])
    mk = {name: markers(im, spec, traced[spec['line']]) for name, spec in fig['markers'].items()}
    traced.pop('BCMMall', None)
    return im, cal, traced, mk


STEP = {2: 0.0025, 3: 0.0025, 4: 0.0025, 5: 0.0002, 6: 0.0025}
EXCL = {4: 7.0, 5: 6.0, 6: 4.0}          # px: points this close to another curve of the figure are dropped
YLIM = {2: (-10, 0), 3: (-70, 0), 4: (-49.6, 0), 5: (1.0, 2.0), 6: (-39.7, 0)}


def near(a, b, dist):
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    out = np.zeros(len(a), bool)
    for i in range(0, len(a), 400):
        d = np.hypot(a[i:i + 400, None, 0] - b[None, :, 0], a[i:i + 400, None, 1] - b[None, :, 1]).min(1)
        out[i:i + 400] = d < dist
    return out


def convert(cal, tr):
    rows = []
    for x, y, w, ok, vx, vy in tr:
        X, Y = cal(x, y)
        cos = max(abs(vx), 0.12)
        r = np.clip(vy / vx if abs(vx) > 1e-9 else 1e9, -11.4, 11.4)    # pixel slope capped at 85 deg
        slope = r * cal.cy[0] / cal.cx[0]
        sig = max(1.0, 0.25 * w) * abs(cal.cy[0]) / cos
        rows.append((X, Y, np.sqrt(sig ** 2 + cal.sy ** 2 + (slope * cal.sx) ** 2)))
    return np.array(rows)


def process(k):
    im, cal, traced, mk = run(k)
    clean = {}
    for name, tr in traced.items():
        t = tr[tr[:, 3] == 1]
        if k in EXCL:
            others = [o[o[:, 3] == 1] for n2, o in traced.items() if n2 != name and n2.split(' ')[0] != name.split(' ')[0]
                      and not (k == 6 and 'common' in (name, n2))]
            for o in others:
                t = t[~near(t, o, EXCL[k] + 0.5 * np.median(o[:, 2]))]
        clean[name] = t
    # figure-specific assembly
    curves = {}
    for name, t in clean.items():
        base = name.replace(' right2', '').replace(' right', '').replace(' end', '')
        if k == 6 and name == 'common':
            continue
        curves.setdefault(base, []).append(t)
    if k == 6:
        for n in ('MMFD', 'BCMM'):
            curves[n].append(clean['common'])
    data = {}
    for name, ts in curves.items():
        P = convert(cal, np.vstack(ts))
        lo, hi = YLIM[k]
        P = P[(P[:, 1] >= lo) & (P[:, 1] <= hi)]
        if k == 5:      # points on a dotted gridline (the profile may include the gridline's ink)
            gy = np.min([np.abs(P[:, 1] - g) for g in (1.2, 1.4, 1.6, 1.8)], axis=0)
            gx = np.min([np.abs(P[:, 0] - g) for g in (0.32, 0.34, 0.36, 0.38)], axis=0)
            P = P[(gy >= 3 * cal.ypp) & (gx >= 3 * cal.xpp)]
        data[name] = P
    extra = {}
    if k == 5:
        # the jump at t/a ~ 0.386: x of the vertical stroke, its ends
        sub = im[1360:1480, 2105:2140] < 128
        w = sub.sum(0).astype(float)
        xj = 2105 + (w * np.arange(sub.shape[1])).sum() / w.sum()
        col = im[1300:1520, int(round(xj))] < 128
        rr = np.where(col)[0]
        ytop, ybot = 1300 + rr.min() + 2.0, 1300 + rr.max() - 2.0
        Xj, Yt = cal(xj, ytop)
        _, Yb = cal(xj, ybot)
        uj = np.hypot(2 * cal.xpp, cal.sx)
        extra = dict(jump=(Xj, Yb, Yt, uj))
        u_end = np.hypot(2 * cal.ypp, cal.sy)
        data['bandwidth, one zero'] = np.vstack([data['bandwidth, one zero'][data['bandwidth, one zero'][:, 0] < Xj - 0.0004],
                                                 [(Xj, Yb, u_end)]])
        data['bandwidth, two zeros'] = np.vstack([[(Xj, Yt, u_end)],
                                                  data['bandwidth, two zeros'][data['bandwidth, two zeros'][:, 0] > Xj + 0.0004]])
        for n in ('bandwidth, one zero', 'bandwidth, two zeros'):
            P = data[n].copy()
            P[:, 1] = (P[:, 1] - 1.0) / 2.0          # right axis: 0 .. 0.5 on the gridlines 1.0 .. 2.0
            P[:, 2] = P[:, 2] / 2.0
            data[n] = P
        extra['jump'] = (Xj, (Yb - 1) / 2, (Yt - 1) / 2, uj)
    out = {}
    for name, P in data.items():
        B = bin_curve(P[:, 0], P[:, 1], P[:, 2], STEP[k])
        out[name] = [(name, x, y, u, 0.0) for x, y, u in B]
    for name, M in mk.items():
        rows = []
        for X, Y, *_ in M:
            x, y = cal(X, Y)
            rows.append((name + ' markers', x, y, np.hypot(2 * cal.ypp, cal.sy), np.hypot(2 * cal.xpp, cal.sx)))
        out[name + ' markers'] = rows
    return cal, out, extra, clean, mk


def write_all(k, caldesc, out, extra, outdir):
    import alim_meta as M
    from l2common import write_raw
    for fname, (names, note) in M.FILES[k].items():
        rows = [(r[0].replace(',', ''),) + tuple(r[1:]) for n in names for r in out.get(n, [])]
        hdr = ['source: ' + M.SRC, 'figure: ' + M.FIG[k]['figure'], 'axes: ' + M.FIG[k]['axes'],
               'geometry: ' + M.FIG[k]['geom'] + '. ' + M.GEOM_T, 'curves: ' + note,
               'calibration (L2a): ' + caldesc, 'method (L2a): 400 dpi pdftoppm render; ridge follower (2 px steps along the tangent, ink centroid on the normal profile, direction search at sharp turns) in hand-seeded segments; dashed lines: each dash component near a coarse hand-placed guide cut into 3 px slices along its principal axis; markers: maxima of the ink distance transform within 8 px of the line that exceed the local stroke half-width by >= 1.75 px (1.5 px for the open diamonds of Fig. 6, holes filled); points within a few px of another curve of the figure dropped; binned in x; uncertainty = sqrt((max(1 px, w/4)/|cos|)^2 + cal_y^2 + (slope cal_x)^2) with w the stroke width, cos of the tangent angle (>= 0.12), cal the rms tick-fit residuals; markers: 2 px in each axis combined with the tick-fit residual',
               'digitizer: L2 (pass a), 2026-11-11']
        if extra.get('jump'):
            j = extra['jump']
            hdr.append(f'jump (L2a): t/a = {j[0]:.5f} +- {j[3]:.5f}, bandwidth {j[1]:.4f} -> {j[2]:.4f}')
        write_raw(os.path.join(outdir, fname), hdr, rows, ydec=5, xdec=5)


if __name__ == '__main__':
    res = {}
    for k in ONLY:
        cal, out, extra, clean, mk = process(k)
        res[k] = (cal, out, extra)
        write_all(k, cal.describe(), out, extra, OUT)
        np.save(os.path.join(SCR, f'alim_a_fig{k}_px.npy'), dict(lines=clean, markers=mk), allow_pickle=True)
        for n, rows in out.items():
            r = np.array([v[1:] for v in rows], float)
            print(k, n, len(rows), 'x %.4f-%.4f' % (r[:, 0].min(), r[:, 0].max()),
                  'u %.3g/%.3g/%.3g' % tuple(np.percentile(r[:, 2], [0, 50, 100])))
        if extra:
            print('   extra', extra)
        print('   cal', cal.describe())
    import pickle
    with open(os.path.join(SCR, 'alim_a_results.pkl'), 'wb') as fh:
        pickle.dump({k: (v[0].describe(), v[1], v[2]) for k, v in res.items()}, fh)
