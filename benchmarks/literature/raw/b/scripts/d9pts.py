import numpy as np
from scipy.spatial import cKDTree
from d9homog import inv_h
exec(open('d9trace.py').read())
parts = assign_euclid(m, gpx, tol=30)

def centre_px(mk, step=8, steep=1.5):
    """pixel-space centre points: column centres every `step` columns where shallow, row centres every `step` rows
    where steep. Returns array (x, y, slope dy/dx)."""
    out = []
    cols = np.flatnonzero(mk.any(0))
    yc = np.array([np.nonzero(mk[:, c])[0].mean() for c in cols])
    for c in range(cols.min(), cols.max() + 1, step):
        sel = np.abs(cols - c) <= 3
        if sel.sum() < 5:
            continue
        wide = np.abs(cols - c) <= 10
        s = np.polyfit(cols[wide], yc[wide], 1)[0]
        if abs(s) > steep:
            continue
        p = np.polyfit(cols[sel], yc[sel], 1)
        out.append((c, np.polyval(p, c), s))
    cand = []
    for r in np.flatnonzero(mk.any(1)):
        xs_r = np.flatnonzero(mk[r]); br = np.flatnonzero(np.diff(xs_r) > 3)
        for a0, b0 in zip(np.r_[0, br + 1], np.r_[br, xs_r.size - 1]):
            cand.append((r, xs_r[a0:b0 + 1].mean()))
    cand = np.array(cand)
    for r, xv0 in cand:
        if int(r) % step:
            continue
        sel = (np.abs(cand[:, 0] - r) <= 4) & (np.abs(cand[:, 1] - xv0) <= 10)
        if sel.sum() < 7 or np.ptp(cand[sel, 0]) < 6:
            continue
        q = np.polyfit(cand[sel, 0], cand[sel, 1], 1)
        if abs(q[0]) * steep >= 1:
            continue
        out.append((np.polyval(q, r), r, 1 / q[0] if abs(q[0]) > 1e-3 else 1e3))
    return np.array(out)

trees = {k: cKDTree(np.c_[np.nonzero(v)[1], np.nonzero(v)[0]]) for k, v in parts.items()}
PX_U = 2.5     # half the printed line width in half-res pixels
pts = {}
for k, mk in parts.items():
    c = centre_px(mk)
    keep = [i for i, (x, y, s) in enumerate(c) if not any(t.query_ball_point((x, y), 16) for kk, t in trees.items() if kk != k)]
    c = c[keep]
    f, p = inv(c[:, 0], c[:, 1])
    rows = []
    for (x, y, s), fi, pi in zip(c, f, p):
        if not np.isfinite(fi):
            continue
        sf, sp = local_scale(fi, pi)
        slope_data = s * sf / sp  # d%/dGHz approx (ignores skew)
        fh, ph = inv_h(x, y)
        # calibration-model difference, expressed in y at fixed x: dP + slope * df
        dcal = abs((ph[0] - pi) - slope_data * (fh[0] - fi))
        uy = np.hypot(PX_U / sp, PX_U / sf * abs(slope_data)) + dcal
        ux = PX_U / sf + abs(fh[0] - fi)
        rows.append((fi, pi, uy, ux, ph[0] - pi, fh[0] - fi))
    pts[k] = sorted(rows)
if __name__ == '__main__':
    for k, v in pts.items():
        print(k, len(v), v[0], v[-1])
    from PIL import Image, ImageDraw
    im = Image.fromarray(a.astype(np.uint8)).convert('RGB'); dr = ImageDraw.Draw(im)
    for (k, v), col in zip(pts.items(), [(255, 0, 0), (0, 170, 0), (0, 0, 255)]):
        for fi, pi, *_ in v:
            x, y = fwd(np.array([fi]), np.array([pi]))[:, 0]
            dr.ellipse((x - 4, y - 4, x + 4, y + 4), outline=col, width=2)
    im.resize((1500, 1012)).save('../d9_pts.png')
    im.crop((1000, 1300, 2000, 1750)).save('../d9_pts_z.png')
