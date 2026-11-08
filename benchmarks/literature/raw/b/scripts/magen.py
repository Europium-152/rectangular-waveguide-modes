"""Generic digitization of Ma et al. 1997 Figs. 5, 6, 8, 9 (L1b)."""
import numpy as np
from PIL import Image, ImageDraw
from dig import *
from trace import assign_euclid
from curvepts import curve_points, drop_near_others

def avgcal(t1, t2, vals):
    """average of two tick sets (opposite axes) -> linear map; also returns half of the max difference in px"""
    p = (np.array(t1) + np.array(t2)) / 2
    f, r, c = fitcal(p, vals)
    return f, r, c, np.max(np.abs(np.array(t1) - np.array(t2))) / 2

def circles(d, region, rmin=5, rmax=9, n=40, thr=0.5):
    """BEM circle markers by Hough transform; region = (x0, x1, y0, y1). Returns list of (x, y, r)."""
    from skimage.transform import hough_circle, hough_circle_peaks
    x0, x1, y0, y1 = region
    sub = d[y0:y1, x0:x1]
    from skimage.feature import canny
    radii = np.arange(rmin, rmax + 1)
    hs = hough_circle(sub.astype(float), radii)
    acc, cx_, cy_, rr = hough_circle_peaks(hs, radii, total_num_peaks=n, min_xdistance=10, min_ydistance=10, threshold=thr * hs.max())
    return [(x + x0, y + y0, r, a) for x, y, r, a in zip(cx_, cy_, rr, acc)]

def run(page, frame, xcal, ycal, guides, masks, tol, xstep, circle_region=None, circle_list=None, drop_r=5):
    im, d = load(page)
    fx, rx, cx, hx = avgcal(*xcal)
    fy, ry, cy, hy = avgcal(*ycal)
    px = lambda f: (np.asarray(f) - cx[1]) / cx[0]; py = lambda s: (np.asarray(s) - cy[1]) / cy[0]
    xl, xr, yt, yb = frame
    m = np.zeros_like(d); m[yt + 4:yb - 3, xl + 4:xr - 3] = d[yt + 4:yb - 3, xl + 4:xr - 3]
    for x0, x1, y0, y1 in masks:
        m[y0:y1, x0:x1] = False
    gpx = {k: (np.array([p[0] for p in v], float), np.array([p[1] for p in v], float)) for k, v in guides.items()}
    out = dict(im=im, d=d, fx=fx, fy=fy, px=px, py=py, rx=rx, ry=ry, cx=cx, cy=cy, hx=hx, hy=hy)
    circ = []
    if circle_list is not None:
        circ = circle_list
        for x, y, r in circ:
            yy, xx = np.ogrid[:m.shape[0], :m.shape[1]]
            ring = ((xx - x) ** 2 + (yy - y) ** 2 <= (r + 3) ** 2) & ((xx - x) ** 2 + (yy - y) ** 2 >= (r - 3) ** 2)
            m &= ~ring
    parts = assign_euclid(m, gpx, tol=tol)
    dbx, dby = abs(cx[0]), abs(cy[0])
    xg = px(np.round(np.arange(fx(xl), fx(xr), xstep), 4))
    pts = {k: curve_points(parts[k], xg, dbx, dby, fx, fy, cal=ry + hy * dby, min_cols=3) for k in guides}
    pts, nrem = drop_near_others(pts, parts, px, py, r=drop_r)
    out.update(parts=parts, pts=pts, nrem=nrem, circ=circ, dbx=dbx, dby=dby, m=m)
    return out

def overlay(o, fname, crop, scale=2):
    rgb = Image.fromarray(np.stack([o['im']] * 3, -1)).convert('RGB'); dr = ImageDraw.Draw(rgb)
    cols = [(255, 0, 0), (0, 0, 255), (0, 160, 0), (255, 0, 255), (0, 170, 170)]
    for (k, pp), col in zip(o['pts'].items(), cols):
        for p in pp:
            x, y = o['px'](p[0]), o['py'](p[1]); dr.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), outline=col)
    for x, y, r in o['circ']:
        dr.ellipse((x - r, y - r, x + r, y + r), outline=(255, 140, 0))
        dr.line((x - 2, y, x + 2, y), fill=(255, 140, 0)); dr.line((x, y - 2, x, y + 2), fill=(255, 140, 0))
    c = rgb.crop(crop); c.resize((c.size[0] * scale, c.size[1] * scale), Image.LANCZOS).save(fname)
